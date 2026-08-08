//! Python-free owned structural-pattern boundary for native Rietveld workflows.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    ConstantWavelengthInstrument, FcjGeometry, OwnedCwContributions, SupportPolicy,
};
use phasesmith_engine::{
    MonochromaticPositionCorrection, PreparedStructuralModel, PreparedStructuralMultiphase,
    PreparedStructuralPhase, StructuralCalculationRequest, StructuralModelInput,
    StructuralMultiphaseError, StructuralPatternError, StructuralPatternResult,
    StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{DomainError, PatternRecord, RecordId};

use crate::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals};

/// One owned monochromatic structural phase and its sample-physics inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldPhase {
    phase_id: RecordId,
    name: String,
    site_ids: Vec<RecordId>,
    definition: StructuralPhaseDefinition,
    contributions: OwnedCwContributions,
}

impl RietveldPhase {
    /// Validate and own one built-in structural phase.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for an empty name, invalid definition, or a
    /// contribution batch with the wrong reflection count.
    pub fn new(
        phase_id: RecordId,
        name: impl Into<String>,
        definition: StructuralPhaseDefinition,
        contributions: OwnedCwContributions,
    ) -> Result<Self, RietveldError> {
        let site_ids = (0..definition.fractional_xyz.len())
            .map(|index| RecordId::new(format!("site-{index}")))
            .collect::<Result<Vec<_>, _>>()
            .map_err(RietveldError::Pattern)?;
        Self::new_with_site_ids(phase_id, name, site_ids, definition, contributions)
    }

    /// Validate and own one phase with explicit stable asymmetric-site IDs.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] when site IDs are missing or duplicated, or
    /// when another phase invariant is invalid.
    pub fn new_with_site_ids(
        phase_id: RecordId,
        name: impl Into<String>,
        site_ids: Vec<RecordId>,
        definition: StructuralPhaseDefinition,
        contributions: OwnedCwContributions,
    ) -> Result<Self, RietveldError> {
        let phase = Self {
            phase_id,
            name: name.into(),
            site_ids,
            definition,
            contributions,
        };
        phase.validate()?;
        Ok(phase)
    }

    pub(crate) fn validate(&self) -> Result<(), RietveldError> {
        if self.name.trim().is_empty() {
            return Err(RietveldError::InvalidPhaseName);
        }
        self.definition
            .validate()
            .map_err(RietveldError::StructuralPattern)?;
        if self.site_ids.len() != self.definition.fractional_xyz.len() {
            return Err(RietveldError::SiteIdCountMismatch);
        }
        if self
            .site_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.site_ids.len()
        {
            return Err(RietveldError::DuplicateSiteId);
        }
        if self.contributions.reflection_count() != self.definition.hkl.len() {
            return Err(RietveldError::ContributionCountMismatch);
        }
        Ok(())
    }

    /// Borrow the stable phase ID.
    #[must_use]
    pub const fn phase_id(&self) -> &RecordId {
        &self.phase_id
    }

    /// Borrow the human-readable phase name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow stable asymmetric-site IDs in structural-array order.
    #[must_use]
    pub fn site_ids(&self) -> &[RecordId] {
        &self.site_ids
    }

    /// Borrow the complete structural definition.
    #[must_use]
    pub const fn definition(&self) -> &StructuralPhaseDefinition {
        &self.definition
    }

    /// Borrow the owned sample-physics contribution batch.
    #[must_use]
    pub const fn contributions(&self) -> &OwnedCwContributions {
        &self.contributions
    }
}

/// Observations, experiment state, and ordered structural phases.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldInput {
    /// Observed pattern and fixed supplied background.
    pub pattern: PatternRecord,
    /// Monochromatic constant-wavelength profile.
    pub instrument: ConstantWavelengthInstrument,
    /// Optional Finger--Cox--Jephcoat axial-divergence geometry.
    pub axial_geometry: Option<FcjGeometry>,
    /// Explicit instrument/sample position correction.
    pub position_correction: MonochromaticPositionCorrection,
    /// Ordered non-empty built-in structural phases.
    pub phases: Vec<RietveldPhase>,
}

impl RietveldInput {
    /// Validate one native monochromatic Rietveld calculation request.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError`] for invalid observations, experiment state,
    /// phase state, or duplicate phase IDs.
    pub fn new(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        axial_geometry: Option<FcjGeometry>,
        position_correction: MonochromaticPositionCorrection,
        phases: Vec<RietveldPhase>,
    ) -> Result<Self, RietveldError> {
        let input = Self {
            pattern,
            instrument,
            axial_geometry,
            position_correction,
            phases,
        };
        input.validate()?;
        Ok(input)
    }

    pub(crate) fn validate(&self) -> Result<(), RietveldError> {
        self.pattern.validate().map_err(RietveldError::Pattern)?;
        if self.pattern.observed_y.is_none() {
            return Err(RietveldError::MissingObservations);
        }
        self.instrument
            .validate()
            .map_err(|_| RietveldError::InvalidInstrument)?;
        if self.axial_geometry.is_some_and(|geometry| {
            !geometry.sample_over_radius.is_finite()
                || !geometry.detector_over_radius.is_finite()
                || geometry.sample_over_radius < 0.0
                || geometry.detector_over_radius < 0.0
        }) {
            return Err(RietveldError::InvalidAxialGeometry);
        }
        let correction = self.position_correction;
        if !correction.zero_shift_deg.is_finite()
            || (correction.bragg_brentano_mm.is_some()
                && correction.debye_scherrer_micrometre.is_some())
            || correction
                .bragg_brentano_mm
                .is_some_and(|(displacement, radius)| {
                    !displacement.is_finite() || !radius.is_finite() || radius <= 0.0
                })
            || correction
                .debye_scherrer_micrometre
                .is_some_and(|(x, y, radius)| {
                    !x.is_finite() || !y.is_finite() || !radius.is_finite() || radius <= 0.0
                })
        {
            return Err(RietveldError::InvalidPositionCorrection);
        }
        if self.phases.is_empty() {
            return Err(RietveldError::EmptyPhases);
        }
        let mut identities = std::collections::BTreeSet::new();
        for phase in &self.phases {
            phase.validate()?;
            if !identities.insert(phase.phase_id.clone()) {
                return Err(RietveldError::DuplicatePhaseId);
            }
        }
        Ok(())
    }
}

/// Deterministic calculation controls shared by later native refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCalculationOptions {
    /// Exact finite profile support in multiples of FWHM.
    pub support_fwhm: f64,
    /// Apply supplied one-sigma uncertainties to residual metrics.
    pub use_uncertainty: bool,
    /// Persistent bounded execution policy.
    pub execution: ExecutionPolicy,
}

impl RietveldCalculationOptions {
    /// Validate explicit support and execution controls.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldError::InvalidOptions`] for invalid support.
    pub fn new(
        support_fwhm: f64,
        use_uncertainty: bool,
        execution: ExecutionPolicy,
    ) -> Result<Self, RietveldError> {
        let options = Self {
            support_fwhm,
            use_uncertainty,
            execution,
        };
        options.validate()?;
        Ok(options)
    }

    pub(crate) fn validate(&self) -> Result<(), RietveldError> {
        if !self.support_fwhm.is_finite() || self.support_fwhm <= 0.0 {
            return Err(RietveldError::InvalidOptions);
        }
        Ok(())
    }
}

/// One labeled phase contribution and its crystallographic intermediates.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldPhaseCalculation {
    /// Stable phase ID.
    pub phase_id: RecordId,
    /// Human-readable phase name.
    pub name: String,
    /// Complete structural/profile result from the native engine.
    pub result: StructuralPatternResult,
}

/// Display-ready structural calculation and residual metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCalculation {
    /// Sum of structural phase profiles before fixed background.
    pub profile_y: Vec<f64>,
    /// Fixed supplied background in sample order.
    pub background_y: Vec<f64>,
    /// Complete calculated pattern (`profile_y + background_y`).
    pub y: Vec<f64>,
    /// Phase calculations in input order.
    pub phases: Vec<RietveldPhaseCalculation>,
    /// Residual arrays and scalar fit metrics.
    pub metrics: ResidualEvaluation,
}

/// Calculate a complete built-in monochromatic structural pattern.
///
/// # Errors
///
/// Returns [`RietveldError`] for invalid phase preparation, calculation, or
/// residual state.
pub fn calculate_rietveld_pattern(
    input: &RietveldInput,
    options: &RietveldCalculationOptions,
) -> Result<RietveldCalculation, RietveldError> {
    input.validate()?;
    options.validate()?;
    let models = input
        .phases
        .iter()
        .map(|phase| {
            PreparedStructuralPhase::new(
                phase.definition.clone(),
                options.execution.context().clone(),
            )
            .map(PreparedStructuralModel::monochromatic)
            .map_err(RietveldError::StructuralPattern)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = PreparedStructuralMultiphase::new(models, options.execution.clone())
        .map_err(RietveldError::StructuralMultiphase)?;
    let request = StructuralCalculationRequest {
        x_deg: input.pattern.x_deg.clone(),
        instrument: input.instrument,
        axial_geometry: input.axial_geometry,
        position_correction: input.position_correction,
        phase_inputs: input
            .phases
            .iter()
            .map(|phase| StructuralModelInput {
                contributions: vec![phase.contributions.clone()],
            })
            .collect(),
        support: SupportPolicy::FwhmMultiple(options.support_fwhm),
    };
    let calculated = prepared
        .calculate_request(request)
        .map_err(RietveldError::StructuralMultiphase)?;
    let background_y = input.pattern.background_y.clone();
    let y = calculated
        .profile_y
        .iter()
        .zip(&background_y)
        .map(|(profile, background)| profile + background)
        .collect::<Vec<_>>();
    if y.iter().any(|value| !value.is_finite()) {
        return Err(RietveldError::NonFiniteCalculation);
    }
    let metrics = evaluate_residuals(
        &input.pattern,
        &y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: 0,
        },
    )
    .map_err(RietveldError::Residual)?;
    let phases = input
        .phases
        .iter()
        .zip(calculated.phases)
        .map(|(phase, result)| RietveldPhaseCalculation {
            phase_id: phase.phase_id.clone(),
            name: phase.name.clone(),
            result,
        })
        .collect();
    Ok(RietveldCalculation {
        profile_y: calculated.profile_y,
        background_y,
        y,
        phases,
        metrics,
    })
}

/// Invalid owned native Rietveld calculation state.
#[derive(Debug)]
pub enum RietveldError {
    /// Pattern domain state is invalid.
    Pattern(DomainError),
    /// Observations are required for residual-bearing Rietveld requests.
    MissingObservations,
    /// The constant-wavelength instrument is invalid.
    InvalidInstrument,
    /// Finger--Cox--Jephcoat geometry is invalid.
    InvalidAxialGeometry,
    /// Monochromatic position correction is invalid.
    InvalidPositionCorrection,
    /// At least one phase is required.
    EmptyPhases,
    /// Phase names must be non-empty.
    InvalidPhaseName,
    /// Phase IDs must be unique.
    DuplicatePhaseId,
    /// Sample-physics contributions must match the reflection count.
    ContributionCountMismatch,
    /// Stable site IDs must match the asymmetric-site count.
    SiteIdCountMismatch,
    /// Stable site IDs must be unique within a phase.
    DuplicateSiteId,
    /// One structural phase could not be prepared.
    StructuralPattern(StructuralPatternError),
    /// Native multiphase structural calculation failed.
    StructuralMultiphase(StructuralMultiphaseError),
    /// Residual evaluation failed.
    Residual(ResidualError),
    /// Calculation controls are invalid.
    InvalidOptions,
    /// Profile/background composition overflowed or became non-finite.
    NonFiniteCalculation,
}

impl Display for RietveldError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => formatter.write_str("observed_y is required for Rietveld"),
            Self::InvalidInstrument => formatter.write_str("Rietveld instrument is invalid"),
            Self::InvalidAxialGeometry => formatter.write_str("Rietveld axial geometry is invalid"),
            Self::InvalidPositionCorrection => {
                formatter.write_str("Rietveld position correction is invalid")
            }
            Self::EmptyPhases => formatter.write_str("at least one Rietveld phase is required"),
            Self::InvalidPhaseName => formatter.write_str("Rietveld phase names must be non-empty"),
            Self::DuplicatePhaseId => formatter.write_str("Rietveld phase IDs must be unique"),
            Self::ContributionCountMismatch => formatter
                .write_str("sample-physics contributions must match the phase reflection count"),
            Self::SiteIdCountMismatch => {
                formatter.write_str("Rietveld site IDs must match the asymmetric-site count")
            }
            Self::DuplicateSiteId => {
                formatter.write_str("Rietveld site IDs must be unique within a phase")
            }
            Self::StructuralPattern(error) => Display::fmt(error, formatter),
            Self::StructuralMultiphase(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
            Self::InvalidOptions => formatter.write_str("Rietveld calculation options are invalid"),
            Self::NonFiniteCalculation => {
                formatter.write_str("Rietveld calculated pattern is non-finite")
            }
        }
    }
}

impl Error for RietveldError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::StructuralPattern(error) => Some(error),
            Self::StructuralMultiphase(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::MissingObservations
            | Self::InvalidInstrument
            | Self::InvalidAxialGeometry
            | Self::InvalidPositionCorrection
            | Self::EmptyPhases
            | Self::InvalidPhaseName
            | Self::DuplicatePhaseId
            | Self::ContributionCountMismatch
            | Self::SiteIdCountMismatch
            | Self::DuplicateSiteId
            | Self::InvalidOptions
            | Self::NonFiniteCalculation => None,
        }
    }
}
