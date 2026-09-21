//! Project ownership for cell-only Pawley analyses.
use crate::pawley::err;
use crate::{ConstraintTransform, PawleyCheckpoint, PawleyError, PawleyInput, PawleyOptions};
use phasesmith_model::{ProjectRecord, RadiationDefinition, RecordId};
use std::collections::BTreeSet;

/// A runnable Pawley analysis attached to a shared CW histogram.
/// Cell-only phase metadata lives in its typed input, without fictitious atoms.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyAnalysis {
    /// Owning histogram in the shared project.
    pub histogram_id: RecordId,
    /// Initial scientific state and stable family identities.
    pub input: PawleyInput,
    /// Numerical controls.
    pub options: PawleyOptions,
    /// Optional accepted continuation state.
    pub checkpoint: Option<PawleyCheckpoint>,
}
impl PawleyAnalysis {
    /// Validate scientific state and checkpoint ownership.
    /// # Errors
    /// Rejects invalid inputs, controls, parameter states or stale checkpoints.
    pub fn validate(&self) -> Result<(), PawleyError> {
        self.input.validate()?;
        self.options.validate()?;
        if let Some(cp) = &self.checkpoint {
            if cp.input != self.input
                || cp.options != self.options
                || !cp.damping.is_finite()
                || cp.damping <= 0.0
                || cp.chi_square_history.is_empty()
                || cp
                    .chi_square_history
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
                || cp.chi_square_history.windows(2).any(|v| v[1] >= v[0])
            {
                return Err(err("stale or corrupt Pawley analysis checkpoint"));
            }
            ConstraintTransform::new(
                self.input.parameters.clone(),
                self.input.constraints.clone(),
            )
            .map_err(err)?
            .unpack(&cp.free, false)
            .map_err(err)?;
        }
        Ok(())
    }
}
/// Shared project with one Pawley analysis per selected CW histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyProjectState {
    /// Histogram observations, experiment metadata and optional structural phases.
    pub project: ProjectRecord,
    /// Stable caller-owned analysis order.
    pub analyses: Vec<PawleyAnalysis>,
}
impl PawleyProjectState {
    /// Validate all histogram and optional shared structural-phase references.
    /// # Errors
    /// Rejects duplicate/missing owners and inconsistent shared experiment state.
    pub fn validate(&self) -> Result<(), PawleyError> {
        self.project.validate().map_err(err)?;
        let mut seen = BTreeSet::new();
        for analysis in &self.analyses {
            analysis.validate()?;
            if !seen.insert(&analysis.histogram_id) {
                return Err(err("duplicate Pawley histogram analysis"));
            }
            let histogram = self
                .project
                .histograms
                .iter()
                .find(|h| h.histogram_id == analysis.histogram_id)
                .ok_or_else(|| err("unknown Pawley histogram owner"))?;
            if histogram.pattern != analysis.input.pattern
                || histogram.experiment.instrument != analysis.input.instrument
                || histogram.experiment.axial_geometry != analysis.input.axial
                || histogram.experiment.position_correction.zero_shift_deg != 0.0
                || histogram
                    .experiment
                    .position_correction
                    .bragg_brentano_mm
                    .is_some()
                || histogram
                    .experiment
                    .position_correction
                    .debye_scherrer_micrometre
                    .is_some()
                || match &histogram.experiment.radiation {
                    RadiationDefinition::Monochromatic { .. } => {
                        analysis.input.fixed_spectrum.is_some()
                    }
                    RadiationDefinition::FixedSpectrum { spectrum, .. } => {
                        analysis.input.fixed_spectrum.as_ref() != Some(spectrum)
                    }
                }
            {
                return Err(err("Pawley analysis differs from shared histogram state"));
            }
            for phase in &analysis.input.phases {
                if let Some(domain) = &phase.lattice {
                    if let Some(shared) = self
                        .project
                        .phases
                        .iter()
                        .find(|p| p.phase_id.as_str() == phase.id)
                    {
                        if domain.parameterization().reference_cell() != shared.definition.cell
                            || domain.parameterization().space_group()
                                != &shared.definition.space_group
                        {
                            return Err(err(
                                "Pawley cell/symmetry differs from shared phase metadata",
                            ));
                        }
                    }
                }
            }
            if !histogram.phase_ids.is_empty()
                && histogram
                    .phase_ids
                    .iter()
                    .map(RecordId::as_str)
                    .collect::<Vec<_>>()
                    != analysis
                        .input
                        .phases
                        .iter()
                        .map(|p| p.id.as_str())
                        .collect::<Vec<_>>()
            {
                return Err(err("Pawley shared phase order mismatch"));
            }
        }
        Ok(())
    }
}
