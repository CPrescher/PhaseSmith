//! Native CIF phase import against one exact desktop histogram snapshot.

use std::path::PathBuf;

use phasesmith_core::OwnedCwContributions;
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionRange,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_io::{
    CifDiagnostic, CifDiagnosticSeverity, CifReadLimits, CifStructure, read_cif_file,
};
use phasesmith_model::{RadiationDefinition, RadiationProbe, RecordId, StructuralPhaseRecord};
use phasesmith_workflows::{LatticeBounds, LatticeParameterization, RietveldPhase};
use serde::{Deserialize, Serialize};

use crate::{DesktopError, DesktopErrorCode, DesktopProjectStore, ProjectSnapshot};

/// Integrated-intensity correction selected for a desktop-imported phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DesktopIntensityCorrectionInput {
    /// Raw multiplicity-weighted structural intensity.
    #[default]
    Neutral,
    /// Unpolarized symmetric Bragg--Brentano Lorentz-polarization correction.
    BraggBrentanoUnpolarizedLp,
    /// Polarized symmetric Bragg--Brentano correction.
    BraggBrentanoPolarizedLp {
        /// Fraction in the constant polarization term.
        polarization: f64,
    },
    /// Constant-wavelength neutron powder Lorentz correction.
    ConstantWavelengthNeutronLorentz,
}

impl DesktopIntensityCorrectionInput {
    fn native(
        self,
        probe: RadiationProbe,
        wavelength_angstrom: f64,
    ) -> Result<IntegratedIntensityCorrectionModel, DesktopError> {
        match (probe, self) {
            (_, Self::Neutral) => Ok(IntegratedIntensityCorrectionModel::Neutral),
            (RadiationProbe::Xray, Self::BraggBrentanoUnpolarizedLp) => Ok(
                IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                    wavelength_angstrom,
                },
            ),
            (RadiationProbe::Xray, Self::BraggBrentanoPolarizedLp { polarization }) => Ok(
                IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                    wavelength_angstrom,
                    polarization,
                },
            ),
            (RadiationProbe::Neutron, Self::ConstantWavelengthNeutronLorentz) => Ok(
                IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
                    wavelength_angstrom,
                },
            ),
            (RadiationProbe::Xray, Self::ConstantWavelengthNeutronLorentz) => Err(invalid_import(
                "neutron Lorentz correction requires a neutron histogram",
            )),
            (RadiationProbe::Neutron, _) => Err(invalid_import(
                "Bragg--Brentano polarization correction requires an X-ray histogram",
            )),
        }
    }
}

/// Bounded CIF import request tied to an existing histogram.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct CifPhaseImportRequest {
    /// Local CIF file path selected by the user.
    pub path: PathBuf,
    /// Existing histogram whose range and wavelength generate reflections.
    pub histogram_id: String,
    /// Stable project phase ID.
    pub phase_id: String,
    /// Optional display-name override; the CIF name is used when absent.
    pub name: Option<String>,
    /// Optional selected CIF data-block name.
    pub block: Option<String>,
    /// Whether unsupported or ambiguous CIF content is fatal.
    pub strict: bool,
    /// Initial non-negative structural scale.
    pub scale: f64,
    /// Symmetry-site coordinate deduplication tolerance.
    pub coordinate_tolerance: f64,
    /// Whether Friedel mates are merged into one powder family.
    pub merge_friedel: bool,
    /// Maximum candidate indices considered during reflection generation.
    pub max_candidates: usize,
    /// Integrated-intensity correction for this phase.
    pub correction: DesktopIntensityCorrectionInput,
}

/// Stable serializable CIF diagnostic severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopCifDiagnosticSeverity {
    /// Recoverable import condition.
    Warning,
    /// Non-recoverable condition retained by a partial parser record.
    Error,
}

/// Serializable diagnostic returned to a desktop presentation layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DesktopCifDiagnostic {
    /// Warning or error severity.
    pub severity: DesktopCifDiagnosticSeverity,
    /// Stable machine-readable diagnostic code.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// Related CIF tag when known.
    pub tag: Option<String>,
    /// Zero-based loop row when known.
    pub row: Option<usize>,
}

impl From<CifDiagnostic> for DesktopCifDiagnostic {
    fn from(value: CifDiagnostic) -> Self {
        Self {
            severity: match value.severity {
                CifDiagnosticSeverity::Warning => DesktopCifDiagnosticSeverity::Warning,
                CifDiagnosticSeverity::Error => DesktopCifDiagnosticSeverity::Error,
            },
            code: value.code,
            message: value.message,
            tag: value.tag,
            row: value.row,
        }
    }
}

/// Successful native CIF phase import.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CifPhaseImportResponse {
    /// New project revision.
    pub revision: u64,
    /// Imported stable phase ID.
    pub phase_id: String,
    /// Histogram to which the phase was attached.
    pub histogram_id: String,
    /// Number of asymmetric sites.
    pub site_count: usize,
    /// Number of generated powder-reflection families.
    pub reflection_count: usize,
    /// Selected CIF block.
    pub selected_block: String,
    /// Available CIF blocks in source order.
    pub available_blocks: Vec<String>,
    /// Visible native-import diagnostics.
    pub diagnostics: Vec<DesktopCifDiagnostic>,
}

impl DesktopProjectStore {
    /// Import one CIF phase using default bounded parser limits.
    ///
    /// # Errors
    ///
    /// Returns a revision conflict, import failure, unsupported state, invalid
    /// project, or shared-state error.
    pub fn import_cif_phase(
        &self,
        expected_revision: u64,
        request: CifPhaseImportRequest,
    ) -> Result<CifPhaseImportResponse, DesktopError> {
        self.import_cif_phase_with_limits(expected_revision, request, CifReadLimits::default())
    }

    /// Import one CIF phase with caller-selected resource limits.
    ///
    /// Parsing and reflection generation happen outside the state lock. The
    /// resulting phase is attached only if the exact source snapshot remains current.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::import_cif_phase`].
    pub fn import_cif_phase_with_limits(
        &self,
        expected_revision: u64,
        request: CifPhaseImportRequest,
        limits: CifReadLimits,
    ) -> Result<CifPhaseImportResponse, DesktopError> {
        let starting = self.snapshot()?;
        if starting.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                starting.revision(),
            ));
        }
        let (histogram_index, histogram) = starting
            .state()
            .project
            .histograms
            .iter()
            .enumerate()
            .find(|(_, histogram)| histogram.histogram_id.as_str() == request.histogram_id)
            .ok_or_else(|| invalid_import("target histogram does not exist"))?;
        let probe = histogram.experiment.radiation.probe();
        let wavelength_angstrom = histogram
            .experiment
            .radiation
            .reference_wavelength_angstrom();
        let reflection_range = match &histogram.experiment.radiation {
            RadiationDefinition::Monochromatic {
                wavelength_angstrom,
                ..
            } => ReflectionRange::CwTwoTheta {
                min_deg: *histogram.pattern.x_deg.first().ok_or_else(|| {
                    invalid_import("target histogram must contain at least one coordinate")
                })?,
                max_deg: *histogram.pattern.x_deg.last().ok_or_else(|| {
                    invalid_import("target histogram must contain at least one coordinate")
                })?,
                wavelength_angstrom: *wavelength_angstrom,
            },
            RadiationDefinition::FixedSpectrum { spectrum, .. } => {
                let minimum_wavelength = spectrum
                    .wavelengths_angstrom()
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .ok_or_else(|| invalid_import("fixed spectrum has no wavelengths"))?;
                let maximum_wavelength = spectrum
                    .wavelengths_angstrom()
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .ok_or_else(|| invalid_import("fixed spectrum has no wavelengths"))?;
                let min_theta = 0.5
                    * histogram
                        .pattern
                        .x_deg
                        .first()
                        .copied()
                        .ok_or_else(|| {
                            invalid_import("target histogram must contain at least one coordinate")
                        })?
                        .to_radians();
                let max_theta = 0.5
                    * histogram
                        .pattern
                        .x_deg
                        .last()
                        .copied()
                        .ok_or_else(|| {
                            invalid_import("target histogram must contain at least one coordinate")
                        })?
                        .to_radians();
                ReflectionRange::ScatteringVector {
                    min_inverse_angstrom: 4.0 * std::f64::consts::PI * min_theta.sin()
                        / maximum_wavelength,
                    max_inverse_angstrom: 4.0 * std::f64::consts::PI * max_theta.sin()
                        / minimum_wavelength,
                }
            }
        };
        let imported = read_cif_file(
            &request.path,
            request.block.as_deref(),
            request.strict,
            limits,
        )
        .map_err(|error| invalid_import(error.to_string()))?;
        let correction = request.correction.native(probe, wavelength_angstrom)?;
        let definition = phase_definition(
            &imported.structure,
            probe,
            reflection_range,
            request.merge_friedel,
            request.max_candidates,
            request.scale,
            request.coordinate_tolerance,
            correction,
        )?;
        self.install_imported_cif(&starting, histogram_index, request, imported, &definition)
    }

    fn install_imported_cif(
        &self,
        starting: &ProjectSnapshot,
        histogram_index: usize,
        request: CifPhaseImportRequest,
        imported: phasesmith_io::CifReadResult,
        definition: &StructuralPhaseDefinition,
    ) -> Result<CifPhaseImportResponse, DesktopError> {
        let phase_id =
            RecordId::new(&request.phase_id).map_err(|error| invalid_import(error.to_string()))?;
        let phase_name = request
            .name
            .clone()
            .unwrap_or_else(|| imported.structure.name.clone());
        let site_count = definition.fractional_xyz.len();
        let reflection_count = definition.hkl.len();
        let mut next = starting.state().clone();
        let analysis_phase = RietveldPhase::new(
            phase_id.clone(),
            phase_name.clone(),
            definition.clone(),
            OwnedCwContributions::neutral(reflection_count),
        )
        .map_err(|error| invalid_import(error.to_string()))?;
        next.project.phases.push(StructuralPhaseRecord {
            phase_id: phase_id.clone(),
            name: phase_name,
            definition: definition.clone(),
            required_providers: Vec::new(),
        });
        let target = next
            .project
            .histograms
            .get_mut(histogram_index)
            .ok_or_else(|| DesktopError::host_failure("cloned histogram index disappeared"))?;
        target.phase_ids.push(phase_id);
        if let Some(analysis) = next
            .analyses
            .iter_mut()
            .find(|analysis| analysis.histogram_id.as_str() == request.histogram_id)
        {
            let lattice_bounds = if analysis.selection.structural.lattice {
                let parameterization =
                    LatticeParameterization::new(definition.space_group.clone(), definition.cell)
                        .map_err(|error| invalid_import(error.to_string()))?;
                Some(
                    LatticeBounds::around(&parameterization, 0.05, 5.0)
                        .map_err(|error| invalid_import(error.to_string()))?,
                )
            } else {
                None
            };
            analysis.input.phases.push(analysis_phase);
            analysis.lattice_bounds.push(lattice_bounds);
            analysis.checkpoint = None;
            analysis
                .validate()
                .map_err(|error| invalid_import(error.to_string()))?;
        }
        let metadata_prefix = format!("phase.{}", request.phase_id);
        next.project.metadata.insert(
            format!("{metadata_prefix}.cif_block"),
            imported.selected_block.clone(),
        );
        next.project.metadata.insert(
            format!("{metadata_prefix}.cif_backend"),
            format!(
                "{}@{}",
                imported.structure.source.backend, imported.structure.source.backend_version
            ),
        );
        next.project.metadata.insert(
            format!("{metadata_prefix}.source_path"),
            request.path.to_string_lossy().into_owned(),
        );
        let installed = self.replace_snapshot(starting, next)?;
        Ok(CifPhaseImportResponse {
            revision: installed.revision(),
            phase_id: request.phase_id,
            histogram_id: request.histogram_id,
            site_count,
            reflection_count,
            selected_block: imported.selected_block,
            available_blocks: imported.available_blocks,
            diagnostics: imported
                .diagnostics
                .into_iter()
                .map(DesktopCifDiagnostic::from)
                .collect(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn phase_definition(
    structure: &CifStructure,
    probe: RadiationProbe,
    reflection_range: ReflectionRange,
    merge_friedel: bool,
    max_candidates: usize,
    scale: f64,
    coordinate_tolerance: f64,
    correction_model: IntegratedIntensityCorrectionModel,
) -> Result<StructuralPhaseDefinition, DesktopError> {
    if structure.sites.is_empty() {
        return Err(invalid_import(
            "structural calculation requires at least one CIF atom site",
        ));
    }
    let generator = PreparedReflectionGenerator::new(
        structure.space_group.clone(),
        merge_friedel,
        max_candidates,
    )
    .map_err(|error| invalid_import(error.to_string()))?;
    let generated = generator
        .generate(structure.cell, reflection_range)
        .map_err(|error| invalid_import(error.to_string()))?;
    if generated.is_empty() {
        return Err(invalid_import(
            "no structural reflections lie in the target histogram range",
        ));
    }
    let definition = StructuralPhaseDefinition {
        cell: structure.cell,
        space_group: structure.space_group.clone(),
        hkl: generated.iter().map(|reflection| reflection.hkl).collect(),
        multiplicity: generated
            .iter()
            .map(|reflection| reflection.multiplicity)
            .collect(),
        fractional_xyz: structure
            .sites
            .iter()
            .map(|site| site.fractional_xyz)
            .collect(),
        occupancy: structure.sites.iter().map(|site| site.occupancy).collect(),
        u_iso_angstrom2: structure
            .sites
            .iter()
            .map(|site| site.u_iso_angstrom2.unwrap_or(0.0))
            .collect(),
        anisotropic_mask: structure
            .sites
            .iter()
            .map(|site| site.anisotropic_displacement.is_some())
            .collect(),
        u_aniso_cif_angstrom2: structure
            .sites
            .iter()
            .map(|site| {
                site.anisotropic_displacement
                    .as_ref()
                    .map_or([0.0; 6], |value| value.u_cif_angstrom2)
            })
            .collect(),
        scattering_species: structure
            .sites
            .iter()
            .map(|site| scattering_key(site, probe))
            .collect(),
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale,
        coordinate_tolerance,
        scattering_model: match probe {
            RadiationProbe::Xray => BuiltInScatteringModel::XrayNonResonant,
            RadiationProbe::Neutron => BuiltInScatteringModel::NeutronNuclear,
        },
        correction_model,
    };
    definition
        .validate()
        .map_err(|error| invalid_import(error.to_string()))?;
    Ok(definition)
}

fn scattering_key(site: &phasesmith_io::CifAtomSite, probe: RadiationProbe) -> String {
    match probe {
        RadiationProbe::Neutron => site.isotope.map_or_else(
            || site.element_symbol.clone(),
            |isotope| format!("{}-{isotope}", site.element_symbol),
        ),
        RadiationProbe::Xray => {
            let ordinary = site.charge.map_or_else(
                || site.element_symbol.clone(),
                |charge| {
                    format!(
                        "{}{}{}",
                        site.element_symbol,
                        charge.unsigned_abs(),
                        if charge > 0 { '+' } else { '-' }
                    )
                },
            );
            let without_isotope = site
                .type_symbol
                .trim_start_matches(|value: char| value.is_ascii_digit());
            if !matches!(site.type_symbol.as_str(), "D" | "T") && without_isotope != ordinary {
                site.type_symbol.clone()
            } else {
                ordinary
            }
        }
    }
}

fn invalid_import(message: impl Into<String>) -> DesktopError {
    DesktopError::simple(DesktopErrorCode::Import, message)
}
