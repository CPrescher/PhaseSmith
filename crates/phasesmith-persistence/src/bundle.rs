//! Lossless mixed-analysis ownership in the shared native project format.
use super::{
    PersistenceError, ProjectReadLimits, ProjectRecord, ProjectSaveOptions, RietveldProjectState,
    StructuralTofMultiBankProjectState, TofLeBailProjectState, TofMultiBankGeometryProjectState,
    load_project_parts, pawley, rietveld_wire, save_project_parts, tof_multibank_wire,
    tof_structural_wire, tof_wire,
};
use phasesmith_workflows::{
    PawleyAnalysis, PawleyProjectState, RietveldAnalysis, StructuralTofMultiBankAnalysis,
    TofLeBailAnalysis, TofMultiBankGeometryAnalysis, TofPawleyAnalysis, TofPawleyProjectState,
};
use std::fmt::Display;
use std::path::{Path, PathBuf};

/// All native analysis families retained together when loading and saving.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectBundle {
    /// Shared project snapshot.
    pub project: ProjectRecord,
    /// Structural CW analyses.
    pub rietveld_analyses: Vec<RietveldAnalysis>,
    /// Fixed-instrument single-bank TOF extraction.
    pub tof_lebail_analyses: Vec<TofLeBailAnalysis>,
    /// Joint TOF geometry extraction.
    pub tof_multibank_geometry_analyses: Vec<TofMultiBankGeometryAnalysis>,
    /// Joint structural TOF analyses.
    pub structural_tof_multibank_analyses: Vec<StructuralTofMultiBankAnalysis>,
    /// Cell-only CW Pawley analyses.
    pub pawley_analyses: Vec<PawleyAnalysis>,
    /// Joint cell-only TOF Pawley analyses.
    pub tof_pawley_analyses: Vec<TofPawleyAnalysis>,
}
fn invalid(e: impl Display) -> PersistenceError {
    PersistenceError::InvalidRecord {
        message: e.to_string(),
    }
}
impl ProjectBundle {
    /// Create an analysis-free bundle around a project snapshot.
    #[must_use]
    pub fn new(project: ProjectRecord) -> Self {
        Self {
            project,
            rietveld_analyses: Vec::new(),
            tof_lebail_analyses: Vec::new(),
            tof_multibank_geometry_analyses: Vec::new(),
            structural_tof_multibank_analyses: Vec::new(),
            pawley_analyses: Vec::new(),
            tof_pawley_analyses: Vec::new(),
        }
    }
    /// Validate every family and its shared histogram/phase references.
    /// # Errors
    /// Returns a validation error without dropping unrelated analysis state.
    pub fn validate(&self) -> Result<(), PersistenceError> {
        self.project.validate().map_err(PersistenceError::Domain)?;
        RietveldProjectState {
            project: self.project.clone(),
            analyses: self.rietveld_analyses.clone(),
        }
        .validate()
        .map_err(invalid)?;
        TofLeBailProjectState {
            project: self.project.clone(),
            analyses: self.tof_lebail_analyses.clone(),
        }
        .validate()
        .map_err(invalid)?;
        TofMultiBankGeometryProjectState {
            project: self.project.clone(),
            analyses: self.tof_multibank_geometry_analyses.clone(),
        }
        .validate()
        .map_err(invalid)?;
        StructuralTofMultiBankProjectState {
            project: self.project.clone(),
            analyses: self.structural_tof_multibank_analyses.clone(),
        }
        .validate()
        .map_err(invalid)?;
        TofPawleyProjectState {
            project: self.project.clone(),
            analyses: self.tof_pawley_analyses.clone(),
        }
        .validate()
        .map_err(invalid)?;
        PawleyProjectState {
            project: self.project.clone(),
            analyses: self.pawley_analyses.clone(),
        }
        .validate()
        .map_err(invalid)
    }
}
/// Save every supported analysis family without selecting or discarding methods.
/// # Errors
/// Rejects inconsistent state, array-name collisions and invalid destinations.
pub fn save_project_bundle(
    path: impl AsRef<Path>,
    bundle: &ProjectBundle,
    options: ProjectSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    bundle.validate()?;
    let rietveld = rietveld_wire::encode_analyses(&RietveldProjectState {
        project: bundle.project.clone(),
        analyses: bundle.rietveld_analyses.clone(),
    });
    let (tof, mut arrays) = tof_wire::encode_analyses(&TofLeBailProjectState {
        project: bundle.project.clone(),
        analyses: bundle.tof_lebail_analyses.clone(),
    })?;
    let (geometry, extra) =
        tof_multibank_wire::encode_analyses(&TofMultiBankGeometryProjectState {
            project: bundle.project.clone(),
            analyses: bundle.tof_multibank_geometry_analyses.clone(),
        })?;
    for (key, value) in extra {
        if arrays.insert(key, value).is_some() {
            return Err(invalid("duplicate mixed-analysis array key"));
        }
    }
    let structural = tof_structural_wire::encode_analyses(&StructuralTofMultiBankProjectState {
        project: bundle.project.clone(),
        analyses: bundle.structural_tof_multibank_analyses.clone(),
    });
    let pawley = pawley::encode_analyses(&PawleyProjectState {
        project: bundle.project.clone(),
        analyses: bundle.pawley_analyses.clone(),
    })?;
    let tof_pawley = super::tof_pawley::encode_analyses(&TofPawleyProjectState {
        project: bundle.project.clone(),
        analyses: bundle.tof_pawley_analyses.clone(),
    })?;
    save_project_parts(
        path.as_ref(),
        &bundle.project,
        rietveld,
        tof,
        geometry,
        structural,
        pawley,
        tof_pawley,
        arrays,
        options,
    )
}
/// Load formats 1–7 and retain all analysis families in stable stored order.
/// # Errors
/// Rejects unsupported versions, corruption, resource limits and inconsistent state.
pub fn load_project_bundle(
    path: impl AsRef<Path>,
    limits: ProjectReadLimits,
) -> Result<ProjectBundle, PersistenceError> {
    let (project, rietveld, tof, geometry, structural, pawley, tof_pawley, mut arrays) =
        load_project_parts(path.as_ref(), limits)?;
    let rietveld_analyses =
        rietveld_wire::decode_state(project.clone(), rietveld, limits)?.analyses;
    let tof_lebail_analyses =
        tof_wire::decode_state(project.clone(), tof, &mut arrays, limits)?.analyses;
    let tof_multibank_geometry_analyses =
        tof_multibank_wire::decode_state(project.clone(), geometry, &mut arrays, limits)?.analyses;
    let structural_tof_multibank_analyses =
        tof_structural_wire::decode_state(project.clone(), structural, limits)?.analyses;
    let pawley_analyses = pawley::decode_state(project.clone(), pawley, limits)?.analyses;
    let tof_pawley_analyses =
        super::tof_pawley::decode_state(project.clone(), tof_pawley, limits)?.analyses;
    if !arrays.is_empty() {
        return Err(invalid(
            "manifest contains arrays that are not referenced by the project",
        ));
    }
    Ok(ProjectBundle {
        project,
        rietveld_analyses,
        tof_lebail_analyses,
        tof_multibank_geometry_analyses,
        structural_tof_multibank_analyses,
        pawley_analyses,
        tof_pawley_analyses,
    })
}
/// Save a shared project containing only Pawley analyses.
/// # Errors
/// Uses the same validation and destination rules as `save_project_bundle`.
pub fn save_pawley_bundle(
    path: impl AsRef<Path>,
    state: &PawleyProjectState,
    options: ProjectSaveOptions,
) -> Result<PathBuf, PersistenceError> {
    let mut bundle = ProjectBundle::new(state.project.clone());
    bundle.pawley_analyses.clone_from(&state.analyses);
    save_project_bundle(path, &bundle, options)
}
/// Load a Pawley view after validating every family; use the bundle API to retain all methods.
/// # Errors
/// Uses the same corruption and resource checks as `load_project_bundle`.
pub fn load_pawley_bundle(
    path: impl AsRef<Path>,
    limits: ProjectReadLimits,
) -> Result<PawleyProjectState, PersistenceError> {
    let bundle = load_project_bundle(path, limits)?;
    Ok(PawleyProjectState {
        project: bundle.project,
        analyses: bundle.pawley_analyses,
    })
}
