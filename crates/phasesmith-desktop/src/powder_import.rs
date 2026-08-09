//! Native powder-file import into revisioned desktop project snapshots.

use std::path::PathBuf;

use phasesmith_core::{ConstantWavelengthInstrument, FcjGeometry};
use phasesmith_engine::MonochromaticPositionCorrection;
use phasesmith_io::{PowderFormat, PowderReadLimits, read_powder_file};
use phasesmith_model::{
    ExperimentRecord, HistogramRecord, RadiationDefinition, RadiationProbe, RecordId,
};
use serde::{Deserialize, Serialize};

use crate::{DesktopError, DesktopErrorCode, DesktopProjectStore};

/// Powder format selected by a presentation adapter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PowderFormatInput {
    /// Detect the format from contents and filename.
    #[default]
    Auto,
    /// Two or three whitespace/comma-separated columns.
    Columns,
    /// Unpacked constant-wavelength GSAS FXYE.
    GsasFxye,
    /// Packed constant-step GSAS STD.
    GsasStd,
}

impl PowderFormatInput {
    const fn native(self) -> PowderFormat {
        match self {
            Self::Auto => PowderFormat::Auto,
            Self::Columns => PowderFormat::Columns,
            Self::GsasFxye => PowderFormat::GsasFxye,
            Self::GsasStd => PowderFormat::GsasStd,
        }
    }

    const fn from_native(value: PowderFormat) -> Self {
        match value {
            PowderFormat::Auto => Self::Auto,
            PowderFormat::Columns => Self::Columns,
            PowderFormat::GsasFxye => Self::GsasFxye,
            PowderFormat::GsasStd => Self::GsasStd,
        }
    }
}

/// Probe used by one imported constant-wavelength histogram.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopRadiationProbe {
    /// X-ray radiation.
    Xray,
    /// Nuclear-neutron radiation.
    Neutron,
}

impl DesktopRadiationProbe {
    const fn native(self) -> RadiationProbe {
        match self {
            Self::Xray => RadiationProbe::Xray,
            Self::Neutron => RadiationProbe::Neutron,
        }
    }
}

/// Optional FCJ axial-divergence geometry in dimensionless radius ratios.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct DesktopFcjGeometryInput {
    /// Sample axial half-height divided by diffractometer radius.
    pub sample_over_radius: f64,
    /// Receiving-slit axial half-height divided by diffractometer radius.
    pub detector_over_radius: f64,
}

impl DesktopFcjGeometryInput {
    const fn native(self) -> FcjGeometry {
        FcjGeometry {
            sample_over_radius: self.sample_over_radius,
            detector_over_radius: self.detector_over_radius,
        }
    }
}

/// Explicit monochromatic peak-position corrections for an imported histogram.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct DesktopPositionCorrectionInput {
    /// Constant additive shift in degrees `2theta`.
    pub zero_shift_deg: f64,
    /// Optional Bragg--Brentano sample displacement in millimetres.
    pub sample_displacement_mm: Option<f64>,
    /// Bragg--Brentano goniometer radius in millimetres when displacement is set.
    pub bragg_brentano_radius_mm: Option<f64>,
    /// Optional Debye--Scherrer X displacement in micrometres.
    pub debye_scherrer_x_micrometre: Option<f64>,
    /// Optional Debye--Scherrer Y displacement in micrometres.
    pub debye_scherrer_y_micrometre: Option<f64>,
    /// Debye--Scherrer goniometer radius in millimetres when displacement is set.
    pub debye_scherrer_radius_mm: Option<f64>,
}

impl DesktopPositionCorrectionInput {
    fn native(self) -> Result<MonochromaticPositionCorrection, DesktopError> {
        let bragg_brentano_mm = optional_geometry_pair(
            self.sample_displacement_mm,
            self.bragg_brentano_radius_mm,
            "Bragg--Brentano displacement and radius must be supplied together",
        )?;
        let debye_scherrer_micrometre = match (
            self.debye_scherrer_x_micrometre,
            self.debye_scherrer_y_micrometre,
            self.debye_scherrer_radius_mm,
        ) {
            (None, None, None) => None,
            (Some(x), Some(y), Some(radius)) => Some((x, y, radius)),
            _ => {
                return Err(DesktopError::simple(
                    DesktopErrorCode::InvalidProject,
                    "Debye--Scherrer X, Y, and radius must be supplied together",
                ));
            }
        };
        if bragg_brentano_mm.is_some() && debye_scherrer_micrometre.is_some() {
            return Err(DesktopError::simple(
                DesktopErrorCode::InvalidProject,
                "Bragg--Brentano and Debye--Scherrer corrections are mutually exclusive",
            ));
        }
        Ok(MonochromaticPositionCorrection {
            zero_shift_deg: self.zero_shift_deg,
            bragg_brentano_mm,
            debye_scherrer_micrometre,
        })
    }
}

/// Constant-wavelength experiment supplied alongside imported powder data.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct DesktopExperimentInput {
    /// Radiation probe.
    pub probe: DesktopRadiationProbe,
    /// Monochromatic wavelength in ångströms.
    pub wavelength_angstrom: f64,
    /// Gaussian profile U in degrees squared.
    pub u_deg2: f64,
    /// Gaussian profile V in degrees squared.
    pub v_deg2: f64,
    /// Gaussian profile W in degrees squared.
    pub w_deg2: f64,
    /// Lorentzian profile X in degrees.
    pub x_deg: f64,
    /// Lorentzian profile Y in degrees.
    pub y_deg: f64,
    /// Optional axial-divergence geometry.
    pub axial_geometry: Option<DesktopFcjGeometryInput>,
    /// Peak-position corrections.
    pub position_correction: DesktopPositionCorrectionInput,
}

impl DesktopExperimentInput {
    fn native(self) -> Result<ExperimentRecord, DesktopError> {
        let instrument = ConstantWavelengthInstrument {
            wavelength_angstrom: self.wavelength_angstrom,
            u_deg2: self.u_deg2,
            v_deg2: self.v_deg2,
            w_deg2: self.w_deg2,
            x_deg: self.x_deg,
            y_deg: self.y_deg,
        };
        ExperimentRecord::new(
            instrument,
            RadiationDefinition::Monochromatic {
                probe: self.probe.native(),
                wavelength_angstrom: self.wavelength_angstrom,
            },
            self.axial_geometry.map(DesktopFcjGeometryInput::native),
            self.position_correction.native()?,
        )
        .map_err(|error| DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string()))
    }
}

/// Bounded file-import request for one project histogram.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct PowderHistogramImportRequest {
    /// Local powder file path selected by the user.
    pub path: PathBuf,
    /// Stable histogram ID.
    pub histogram_id: String,
    /// Human-readable histogram name.
    pub name: String,
    /// Explicit or detected powder format.
    pub format: PowderFormatInput,
    /// Positive GSAS bank number; ignored by columns input.
    pub bank: usize,
    /// Constant-wavelength experiment metadata.
    pub experiment: DesktopExperimentInput,
    /// Existing project phase IDs activated for this histogram.
    pub phase_ids: Vec<String>,
}

/// Successful powder import and snapshot replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PowderHistogramImportResponse {
    /// New project revision.
    pub revision: u64,
    /// Imported histogram ID.
    pub histogram_id: String,
    /// Number of imported samples.
    pub sample_count: usize,
    /// Concrete detected or selected format.
    pub format: PowderFormatInput,
    /// Selected GSAS bank, absent for plain columns.
    pub bank: Option<usize>,
}

impl DesktopProjectStore {
    /// Import a bounded native powder file and append one histogram atomically.
    ///
    /// Parsing and validation happen outside the store lock. Installation uses
    /// exact-snapshot comparison, so a concurrent edit wins without data loss.
    ///
    /// # Errors
    ///
    /// Returns a revision conflict, import failure, invalid record, or shared-state error.
    pub fn import_powder_histogram(
        &self,
        expected_revision: u64,
        request: PowderHistogramImportRequest,
    ) -> Result<PowderHistogramImportResponse, DesktopError> {
        self.import_powder_histogram_with_limits(
            expected_revision,
            request,
            PowderReadLimits::default(),
        )
    }

    /// Import with caller-selected resource limits.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::import_powder_histogram`].
    pub fn import_powder_histogram_with_limits(
        &self,
        expected_revision: u64,
        request: PowderHistogramImportRequest,
        limits: PowderReadLimits,
    ) -> Result<PowderHistogramImportResponse, DesktopError> {
        let starting = self.snapshot()?;
        if starting.revision() != expected_revision {
            return Err(DesktopError::conflict(
                expected_revision,
                starting.revision(),
            ));
        }
        let imported =
            read_powder_file(&request.path, request.format.native(), request.bank, limits)
                .map_err(|error| {
                    DesktopError::simple(DesktopErrorCode::Import, error.to_string())
                })?;
        let histogram_id = RecordId::new(&request.histogram_id).map_err(|error| {
            DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
        })?;
        let phase_ids = request
            .phase_ids
            .into_iter()
            .map(RecordId::new)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                DesktopError::simple(DesktopErrorCode::InvalidProject, error.to_string())
            })?;
        let sample_count = imported.pattern.sample_count();
        let format = PowderFormatInput::from_native(imported.format);
        let bank = imported.bank;
        let mut next = starting.state().clone();
        next.project.histograms.push(HistogramRecord {
            histogram_id,
            name: request.name,
            pattern: imported.pattern,
            experiment: request.experiment.native()?,
            phase_ids,
        });
        let installed = self.replace_snapshot(&starting, next)?;
        Ok(PowderHistogramImportResponse {
            revision: installed.revision(),
            histogram_id: request.histogram_id,
            sample_count,
            format,
            bank,
        })
    }
}

fn optional_geometry_pair(
    first: Option<f64>,
    second: Option<f64>,
    message: &'static str,
) -> Result<Option<(f64, f64)>, DesktopError> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(first), Some(second)) => Ok(Some((first, second))),
        _ => Err(DesktopError::simple(
            DesktopErrorCode::InvalidProject,
            message,
        )),
    }
}
