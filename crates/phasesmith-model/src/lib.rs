//! Application-neutral owned records shared by scripting and native hosts.
//!
//! These types describe validated live domain state. Persistence wire records,
//! migrations, `PyO3` objects, and Tauri command payloads deliberately live in
//! adapter crates instead of being derived directly from this model.
//! Applications normally use this crate through
//! [`phasesmith::model`](https://docs.rs/phasesmith/latest/phasesmith/).
//!
//! # Core records
//!
//! - [`PatternRecord`] owns a strictly increasing `2θ` grid and aligned
//!   observations, uncertainties, mask, and fixed background.
//! - [`TofPatternRecord`] owns the corresponding explicitly microsecond-domain
//!   time-of-flight record; the coordinate types cannot be interchanged.
//! - [`StructuralPhaseRecord`] owns one crystal-structure phase and its provider
//!   requirements.
//! - [`HistogramRecord`] combines observed data, an experiment, and referenced
//!   phase IDs.
//! - [`ProjectRecord`] is a revisioned multi-histogram project snapshot.
//! - [`RecordId`] is the validated stable identifier shared across records.
//!
//! # Example
//!
//! ```
//! use phasesmith_model::PatternRecord;
//!
//! let pattern = PatternRecord::new(
//!     vec![20.0, 20.1],
//!     Some(vec![100.0, 120.0]),
//!     Some(vec![2.0, 2.5]),
//!     None,
//!     None,
//! )?;
//! assert_eq!(pattern.sample_count(), 2);
//! assert_eq!(pattern.background_y, [0.0, 0.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Constructors validate records at trust boundaries. Public fields remain
//! available for efficient adapter construction, so call `validate` again
//! after direct mutation or before crossing into persistence/workflow code.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    ConstantWavelengthInstrument, FcjGeometry, WavelengthComponentsError, WavelengthComponentsView,
};
use phasesmith_engine::{
    MonochromaticPositionCorrection, StructuralPatternError, StructuralPhaseDefinition,
};

/// Stable project-owned identifier used for projects, histograms, and phases.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordId(String);

impl RecordId {
    /// Validate and own an adapter-supplied stable identifier.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidId`] when the value is empty, too long, or
    /// contains characters outside ASCII letters, digits, `.`, `_`, and `-`.
    pub fn new(value: impl Into<String>) -> Result<Self, DomainError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 128
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(DomainError::InvalidId { value });
        }
        Ok(Self(value))
    }

    /// Borrow the stable textual representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for RecordId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// X-ray or neutron radiation selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiationProbe {
    /// Electromagnetic X-ray radiation.
    Xray,
    /// Constant-wavelength nuclear-neutron radiation.
    Neutron,
}

/// Validated fixed radiation components in input order.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedWavelengthSpectrum {
    wavelengths_angstrom: Vec<f64>,
    relative_intensities: Vec<f64>,
}

impl FixedWavelengthSpectrum {
    /// Validate and own fixed wavelength components.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::Radiation`] for invalid wavelengths or weights.
    pub fn new(
        wavelengths_angstrom: Vec<f64>,
        relative_intensities: Vec<f64>,
    ) -> Result<Self, DomainError> {
        WavelengthComponentsView::new(&wavelengths_angstrom, &relative_intensities)
            .map_err(DomainError::Radiation)?;
        Ok(Self {
            wavelengths_angstrom,
            relative_intensities,
        })
    }

    /// Borrow component wavelengths in ångströms.
    #[must_use]
    pub fn wavelengths_angstrom(&self) -> &[f64] {
        &self.wavelengths_angstrom
    }

    /// Borrow component intensities relative to the first component.
    #[must_use]
    pub fn relative_intensities(&self) -> &[f64] {
        &self.relative_intensities
    }
}

/// Radiation attached to one constant-wavelength histogram.
#[derive(Clone, Debug, PartialEq)]
pub enum RadiationDefinition {
    /// One monochromatic wavelength.
    Monochromatic {
        /// Probe family.
        probe: RadiationProbe,
        /// Wavelength in ångströms.
        wavelength_angstrom: f64,
    },
    /// Fixed discrete wavelength spectrum.
    FixedSpectrum {
        /// Probe family.
        probe: RadiationProbe,
        /// Validated spectrum components.
        spectrum: FixedWavelengthSpectrum,
    },
}

impl RadiationDefinition {
    /// Return the probe family.
    #[must_use]
    pub const fn probe(&self) -> RadiationProbe {
        match self {
            Self::Monochromatic { probe, .. } | Self::FixedSpectrum { probe, .. } => *probe,
        }
    }

    /// Return the reference wavelength used by the instrument record.
    #[must_use]
    pub fn reference_wavelength_angstrom(&self) -> f64 {
        match self {
            Self::Monochromatic {
                wavelength_angstrom,
                ..
            } => *wavelength_angstrom,
            Self::FixedSpectrum { spectrum, .. } => spectrum.wavelengths_angstrom[0],
        }
    }
}

/// Owned observed powder pattern for one histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternRecord {
    /// Strictly increasing coordinate grid in degrees `2theta`.
    pub x_deg: Vec<f64>,
    /// Optional observed intensity values.
    pub observed_y: Option<Vec<f64>>,
    /// Optional positive one-sigma uncertainties.
    pub uncertainty: Option<Vec<f64>>,
    /// Optional inclusion mask.
    pub mask: Option<Vec<bool>>,
    /// Supplied fixed background, always sample-aligned.
    pub background_y: Vec<f64>,
}

impl PatternRecord {
    /// Validate and own one pattern grid and its optional observations.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for non-finite, unordered, or mismatched arrays.
    pub fn new(
        x_deg: Vec<f64>,
        observed_y: Option<Vec<f64>>,
        uncertainty: Option<Vec<f64>>,
        mask: Option<Vec<bool>>,
        background_y: Option<Vec<f64>>,
    ) -> Result<Self, DomainError> {
        let sample_count = x_deg.len();
        let background_y = background_y.unwrap_or_else(|| vec![0.0; sample_count]);
        let record = Self {
            x_deg,
            observed_y,
            uncertainty,
            mask,
            background_y,
        };
        record.validate()?;
        Ok(record)
    }

    /// Return the number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.x_deg.len()
    }

    /// Revalidate all arrays after direct adapter-side record construction.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for non-finite, unordered, or mismatched arrays.
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.x_deg.iter().any(|value| !value.is_finite()) {
            return Err(DomainError::NonFiniteArray { name: "x_deg" });
        }
        if self.x_deg.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(DomainError::UnorderedGrid);
        }
        let sample_count = self.x_deg.len();
        validate_optional_f64(
            "observed_y",
            self.observed_y.as_deref(),
            sample_count,
            false,
        )?;
        validate_optional_f64(
            "uncertainty",
            self.uncertainty.as_deref(),
            sample_count,
            true,
        )?;
        if self
            .mask
            .as_ref()
            .is_some_and(|values| values.len() != sample_count)
        {
            return Err(DomainError::ArrayLengthMismatch { name: "mask" });
        }
        validate_f64("background_y", &self.background_y, sample_count, false)
    }
}

/// Owned time-of-flight powder pattern with an explicitly microsecond coordinate axis.
#[derive(Clone, Debug, PartialEq)]
pub struct TofPatternRecord {
    /// Strictly increasing time-of-flight grid in microseconds.
    pub tof_us: Vec<f64>,
    /// Optional observed intensity values.
    pub observed_y: Option<Vec<f64>>,
    /// Optional positive one-sigma uncertainties.
    pub uncertainty: Option<Vec<f64>>,
    /// Optional inclusion mask.
    pub mask: Option<Vec<bool>>,
    /// Supplied fixed background, always sample-aligned.
    pub background_y: Vec<f64>,
}

impl TofPatternRecord {
    /// Validate and own one TOF pattern without angle-domain conversion.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for non-finite, unordered, or mismatched arrays.
    pub fn new(
        tof_us: Vec<f64>,
        observed_y: Option<Vec<f64>>,
        uncertainty: Option<Vec<f64>>,
        mask: Option<Vec<bool>>,
        background_y: Option<Vec<f64>>,
    ) -> Result<Self, DomainError> {
        let sample_count = tof_us.len();
        let record = Self {
            tof_us,
            observed_y,
            uncertainty,
            mask,
            background_y: background_y.unwrap_or_else(|| vec![0.0; sample_count]),
        };
        record.validate()?;
        Ok(record)
    }

    /// Return the number of TOF samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.tof_us.len()
    }

    /// Revalidate all arrays after direct adapter-side construction.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for non-finite, unordered, or mismatched arrays.
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.tof_us.iter().any(|value| !value.is_finite()) {
            return Err(DomainError::NonFiniteArray { name: "tof_us" });
        }
        if self.tof_us.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(DomainError::UnorderedGrid);
        }
        let sample_count = self.tof_us.len();
        validate_optional_f64(
            "observed_y",
            self.observed_y.as_deref(),
            sample_count,
            false,
        )?;
        validate_optional_f64(
            "uncertainty",
            self.uncertainty.as_deref(),
            sample_count,
            true,
        )?;
        if self
            .mask
            .as_ref()
            .is_some_and(|values| values.len() != sample_count)
        {
            return Err(DomainError::ArrayLengthMismatch { name: "mask" });
        }
        validate_f64("background_y", &self.background_y, sample_count, false)
    }
}

/// Constant-wavelength experiment shared by native workflows and adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentRecord {
    /// Reference U/V/W/X/Y instrument.
    pub instrument: ConstantWavelengthInstrument,
    /// Monochromatic or fixed-spectrum radiation.
    pub radiation: RadiationDefinition,
    /// Optional axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit zero/specimen-displacement correction.
    pub position_correction: MonochromaticPositionCorrection,
}

impl ExperimentRecord {
    /// Validate agreement between radiation, instrument, and geometry.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when reference wavelengths disagree or geometry
    /// contains a non-finite/nonphysical value.
    pub fn new(
        instrument: ConstantWavelengthInstrument,
        radiation: RadiationDefinition,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
    ) -> Result<Self, DomainError> {
        let record = Self {
            instrument,
            radiation,
            axial_geometry,
            position_correction,
        };
        record.validate()?;
        Ok(record)
    }

    /// Revalidate the experiment after direct adapter-side construction.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] when reference wavelengths disagree or geometry
    /// contains a non-finite/nonphysical value.
    pub fn validate(&self) -> Result<(), DomainError> {
        if self.instrument.wavelength_angstrom.to_bits()
            != self.radiation.reference_wavelength_angstrom().to_bits()
        {
            return Err(DomainError::ReferenceWavelengthMismatch);
        }
        match &self.radiation {
            RadiationDefinition::Monochromatic {
                wavelength_angstrom,
                ..
            } if !wavelength_angstrom.is_finite() || *wavelength_angstrom <= 0.0 => {
                return Err(DomainError::InvalidRadiationWavelength);
            }
            RadiationDefinition::FixedSpectrum { spectrum, .. } => {
                WavelengthComponentsView::new(
                    &spectrum.wavelengths_angstrom,
                    &spectrum.relative_intensities,
                )
                .map_err(DomainError::Radiation)?;
            }
            RadiationDefinition::Monochromatic { .. } => {}
        }
        validate_instrument(self.instrument)?;
        validate_axial_geometry(self.axial_geometry)?;
        validate_position_correction(self.position_correction)
    }
}

/// Exact external provider capability required by a phase.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProviderRequirement {
    /// Stable provider family identifier.
    pub provider_id: String,
    /// Exact provider API/data version.
    pub provider_version: String,
}

impl ProviderRequirement {
    /// Validate a provider requirement.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidProviderRequirement`] for empty fields.
    pub fn new(
        provider_id: impl Into<String>,
        provider_version: impl Into<String>,
    ) -> Result<Self, DomainError> {
        let requirement = Self {
            provider_id: provider_id.into(),
            provider_version: provider_version.into(),
        };
        if requirement.provider_id.trim().is_empty()
            || requirement.provider_version.trim().is_empty()
        {
            return Err(DomainError::InvalidProviderRequirement);
        }
        Ok(requirement)
    }
}

/// One validated structural phase and its optional extension requirements.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPhaseRecord {
    /// Stable phase identifier.
    pub phase_id: RecordId,
    /// Human-readable phase label.
    pub name: String,
    /// Native crystallographic/scattering definition.
    pub definition: StructuralPhaseDefinition,
    /// External providers required in addition to built-in native models.
    pub required_providers: Vec<ProviderRequirement>,
}

/// One independently observed dataset in a multi-histogram project.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramRecord {
    /// Stable histogram identifier.
    pub histogram_id: RecordId,
    /// Human-readable dataset label.
    pub name: String,
    /// Observed grid and arrays.
    pub pattern: PatternRecord,
    /// Experiment attached to this dataset.
    pub experiment: ExperimentRecord,
    /// Ordered phase references active for this histogram.
    pub phase_ids: Vec<RecordId>,
}

/// Revisioned application-neutral project snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectRecord {
    /// Stable project identifier.
    pub project_id: RecordId,
    /// Monotonically increasing adapter-owned revision.
    pub revision: u64,
    /// Human-readable project label.
    pub name: String,
    /// Independently observed datasets.
    pub histograms: Vec<HistogramRecord>,
    /// Project-owned phase definitions.
    pub phases: Vec<StructuralPhaseRecord>,
    /// Small textual metadata; bulk arrays remain typed fields.
    pub metadata: BTreeMap<String, String>,
}

impl ProjectRecord {
    /// Validate cross-record identities and references.
    ///
    /// Empty projects are allowed so an application can create a project before
    /// importing data. Once histograms exist, every phase reference must resolve.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError`] for invalid labels, duplicate IDs/providers, or
    /// dangling/duplicate phase references.
    pub fn validate(&self) -> Result<(), DomainError> {
        validate_label("project", &self.name)?;
        let mut phase_ids = BTreeSet::new();
        for phase in &self.phases {
            validate_label("phase", &phase.name)?;
            phase
                .definition
                .validate()
                .map_err(DomainError::StructuralPhase)?;
            if !phase_ids.insert(phase.phase_id.clone()) {
                return Err(DomainError::DuplicatePhaseId {
                    phase_id: phase.phase_id.clone(),
                });
            }
            let mut requirements = BTreeSet::new();
            for requirement in &phase.required_providers {
                if requirement.provider_id.trim().is_empty()
                    || requirement.provider_version.trim().is_empty()
                {
                    return Err(DomainError::InvalidProviderRequirement);
                }
                if !requirements.insert(requirement.clone()) {
                    return Err(DomainError::DuplicateProviderRequirement {
                        phase_id: phase.phase_id.clone(),
                        provider_id: requirement.provider_id.clone(),
                    });
                }
            }
        }
        let mut histogram_ids = BTreeSet::new();
        for histogram in &self.histograms {
            validate_label("histogram", &histogram.name)?;
            histogram.pattern.validate()?;
            histogram.experiment.validate()?;
            if !histogram_ids.insert(histogram.histogram_id.clone()) {
                return Err(DomainError::DuplicateHistogramId {
                    histogram_id: histogram.histogram_id.clone(),
                });
            }
            let mut referenced = BTreeSet::new();
            for phase_id in &histogram.phase_ids {
                if !phase_ids.contains(phase_id) {
                    return Err(DomainError::UnknownPhaseReference {
                        histogram_id: histogram.histogram_id.clone(),
                        phase_id: phase_id.clone(),
                    });
                }
                if !referenced.insert(phase_id.clone()) {
                    return Err(DomainError::DuplicatePhaseReference {
                        histogram_id: histogram.histogram_id.clone(),
                        phase_id: phase_id.clone(),
                    });
                }
            }
        }
        if self.metadata.keys().any(|key| key.trim().is_empty()) {
            return Err(DomainError::InvalidMetadataKey);
        }
        Ok(())
    }

    /// Return all missing extension-provider capabilities in stable order.
    #[must_use]
    pub fn capability_diagnostics(
        &self,
        capabilities: &HostCapabilities,
    ) -> Vec<CapabilityDiagnostic> {
        self.phases
            .iter()
            .flat_map(|phase| {
                phase
                    .required_providers
                    .iter()
                    .filter(|requirement| !capabilities.supports(requirement))
                    .map(|requirement| CapabilityDiagnostic {
                        phase_id: phase.phase_id.clone(),
                        requirement: requirement.clone(),
                        reason: CapabilityReason::ProviderUnavailable,
                    })
            })
            .collect()
    }
}

/// Provider versions available to one application host.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostCapabilities {
    providers: BTreeSet<ProviderRequirement>,
}

impl HostCapabilities {
    /// Construct a host capability set from exact provider requirements.
    #[must_use]
    pub fn new(providers: impl IntoIterator<Item = ProviderRequirement>) -> Self {
        Self {
            providers: providers.into_iter().collect(),
        }
    }

    /// Return whether the host has this exact provider/version pair.
    #[must_use]
    pub fn supports(&self, requirement: &ProviderRequirement) -> bool {
        self.providers.contains(requirement)
    }
}

/// Stable reason a project capability is unavailable to a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityReason {
    /// The required provider/version pair was not registered by the host.
    ProviderUnavailable,
}

/// One host-capability diagnostic tied to a stable phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityDiagnostic {
    /// Phase requiring the missing capability.
    pub phase_id: RecordId,
    /// Exact missing provider/version pair.
    pub requirement: ProviderRequirement,
    /// Stable diagnostic category.
    pub reason: CapabilityReason,
}

/// Invalid native domain or project data.
#[derive(Debug)]
pub enum DomainError {
    /// A stable identifier is invalid.
    InvalidId {
        /// Rejected value.
        value: String,
    },
    /// A human-readable project record label is empty.
    InvalidLabel {
        /// Record family with the invalid label.
        record: &'static str,
    },
    /// A numeric array has an unexpected length.
    ArrayLengthMismatch {
        /// Stable field name.
        name: &'static str,
    },
    /// A numeric array contains a non-finite value.
    NonFiniteArray {
        /// Stable field name.
        name: &'static str,
    },
    /// A required-positive array contains zero or a negative value.
    NonPositiveArray {
        /// Stable field name.
        name: &'static str,
    },
    /// Pattern coordinates are not strictly increasing.
    UnorderedGrid,
    /// Radiation components are invalid.
    Radiation(WavelengthComponentsError),
    /// A monochromatic radiation wavelength is non-finite or non-positive.
    InvalidRadiationWavelength,
    /// Instrument and radiation reference wavelengths differ.
    ReferenceWavelengthMismatch,
    /// Instrument parameters are non-finite or nonphysical.
    InvalidInstrument,
    /// Axial geometry is non-finite or negative.
    InvalidAxialGeometry,
    /// Position-correction geometry is non-finite or nonphysical.
    InvalidPositionCorrection,
    /// A structural phase definition is invalid.
    StructuralPhase(StructuralPatternError),
    /// Provider ID or version is empty.
    InvalidProviderRequirement,
    /// A phase ID is repeated.
    DuplicatePhaseId {
        /// Repeated phase ID.
        phase_id: RecordId,
    },
    /// A histogram ID is repeated.
    DuplicateHistogramId {
        /// Repeated histogram ID.
        histogram_id: RecordId,
    },
    /// One histogram references a missing phase.
    UnknownPhaseReference {
        /// Histogram containing the reference.
        histogram_id: RecordId,
        /// Missing phase ID.
        phase_id: RecordId,
    },
    /// One histogram references the same phase twice.
    DuplicatePhaseReference {
        /// Histogram containing the duplicate.
        histogram_id: RecordId,
        /// Duplicated phase ID.
        phase_id: RecordId,
    },
    /// One phase repeats the same exact provider requirement.
    DuplicateProviderRequirement {
        /// Phase containing the duplicate.
        phase_id: RecordId,
        /// Repeated provider ID.
        provider_id: String,
    },
    /// A metadata key is empty.
    InvalidMetadataKey,
}

impl Display for DomainError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId { value } => write!(formatter, "invalid stable record ID {value:?}"),
            Self::InvalidLabel { record } => write!(formatter, "{record} label must not be empty"),
            Self::ArrayLengthMismatch { name } => {
                write!(formatter, "{name} must match the pattern sample count")
            }
            Self::NonFiniteArray { name } => write!(formatter, "{name} must contain finite values"),
            Self::NonPositiveArray { name } => {
                write!(formatter, "{name} must contain positive values")
            }
            Self::UnorderedGrid => formatter.write_str("x_deg must be strictly increasing"),
            Self::Radiation(error) => Display::fmt(error, formatter),
            Self::InvalidRadiationWavelength => {
                formatter.write_str("radiation wavelength must be positive and finite")
            }
            Self::ReferenceWavelengthMismatch => formatter
                .write_str("instrument wavelength must match the radiation reference wavelength"),
            Self::InvalidInstrument => {
                formatter.write_str("constant-wavelength instrument parameters are invalid")
            }
            Self::InvalidAxialGeometry => {
                formatter.write_str("axial geometry must be finite and non-negative")
            }
            Self::InvalidPositionCorrection => {
                formatter.write_str("position-correction geometry is invalid")
            }
            Self::StructuralPhase(error) => Display::fmt(error, formatter),
            Self::InvalidProviderRequirement => {
                formatter.write_str("provider ID and version must not be empty")
            }
            Self::DuplicatePhaseId { phase_id } => {
                write!(formatter, "duplicate phase ID {phase_id}")
            }
            Self::DuplicateHistogramId { histogram_id } => {
                write!(formatter, "duplicate histogram ID {histogram_id}")
            }
            Self::UnknownPhaseReference {
                histogram_id,
                phase_id,
            } => write!(
                formatter,
                "histogram {histogram_id} references unknown phase {phase_id}"
            ),
            Self::DuplicatePhaseReference {
                histogram_id,
                phase_id,
            } => write!(
                formatter,
                "histogram {histogram_id} repeats phase {phase_id}"
            ),
            Self::DuplicateProviderRequirement {
                phase_id,
                provider_id,
            } => write!(formatter, "phase {phase_id} repeats provider {provider_id}"),
            Self::InvalidMetadataKey => formatter.write_str("metadata keys must not be empty"),
        }
    }
}

impl Error for DomainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Radiation(error) => Some(error),
            Self::StructuralPhase(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_f64(
    name: &'static str,
    values: &[f64],
    expected: usize,
    positive: bool,
) -> Result<(), DomainError> {
    if values.len() != expected {
        return Err(DomainError::ArrayLengthMismatch { name });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DomainError::NonFiniteArray { name });
    }
    if positive && values.iter().any(|value| *value <= 0.0) {
        return Err(DomainError::NonPositiveArray { name });
    }
    Ok(())
}

fn validate_optional_f64(
    name: &'static str,
    values: Option<&[f64]>,
    expected: usize,
    positive: bool,
) -> Result<(), DomainError> {
    values.map_or(Ok(()), |values| {
        validate_f64(name, values, expected, positive)
    })
}

fn validate_label(record: &'static str, value: &str) -> Result<(), DomainError> {
    if value.trim().is_empty() {
        return Err(DomainError::InvalidLabel { record });
    }
    Ok(())
}

fn validate_instrument(instrument: ConstantWavelengthInstrument) -> Result<(), DomainError> {
    let values = [
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
    ];
    if values.iter().any(|value| !value.is_finite()) || instrument.wavelength_angstrom <= 0.0 {
        return Err(DomainError::InvalidInstrument);
    }
    Ok(())
}

fn validate_axial_geometry(geometry: Option<FcjGeometry>) -> Result<(), DomainError> {
    if geometry.is_some_and(|value| {
        !value.sample_over_radius.is_finite()
            || !value.detector_over_radius.is_finite()
            || value.sample_over_radius < 0.0
            || value.detector_over_radius < 0.0
    }) {
        return Err(DomainError::InvalidAxialGeometry);
    }
    Ok(())
}

fn validate_position_correction(
    correction: MonochromaticPositionCorrection,
) -> Result<(), DomainError> {
    let invalid = !correction.zero_shift_deg.is_finite()
        || correction
            .bragg_brentano_mm
            .is_some_and(|(displacement, radius)| {
                !displacement.is_finite() || !radius.is_finite() || radius <= 0.0
            })
        || correction
            .debye_scherrer_micrometre
            .is_some_and(|(x, y, radius)| {
                !x.is_finite() || !y.is_finite() || !radius.is_finite() || radius <= 0.0
            });
    if invalid
        || (correction.bragg_brentano_mm.is_some()
            && correction.debye_scherrer_micrometre.is_some())
    {
        return Err(DomainError::InvalidPositionCorrection);
    }
    Ok(())
}
