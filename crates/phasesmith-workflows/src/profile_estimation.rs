//! Staged estimation of an effective CW starting profile from one dominant phase.

use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{ConstantWavelengthInstrument, CwProfileParameters};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::PatternRecord;

use crate::{
    CovarianceMatrix, LeBailError, LeBailInput, LeBailOptions, LeBailPhase, LeBailResult,
    build_lebail_parameter_set, build_lebail_parameter_set_with_lattice, refine_lebail,
};

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;
const W_PARAMETERS: [&str; 1] = ["w_deg2"];
const UVW_PARAMETERS: [&str; 3] = ["u_deg2", "v_deg2", "w_deg2"];
const UVWXY_PARAMETERS: [&str; 5] = ["u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"];

/// Requested complexity for effective-profile estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileEstimationMode {
    /// Fit only the constant Gaussian variance `W`.
    WOnly,
    /// Fit `W`, then explicitly fit `U/V/W`.
    Uvw,
    /// Fit `W`, `U/V/W`, then explicitly fit `U/V/W/X/Y`.
    Uvwxy,
    /// Add `U/V` and then `X/Y` only when conservative diagnostics support them.
    Automatic,
}

impl ProfileEstimationMode {
    /// Return the stable scripting/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WOnly => "w_only",
            Self::Uvw => "uvw",
            Self::Uvwxy => "uvwxy",
            Self::Automatic => "automatic",
        }
    }
}

/// Stable name for one attempted estimation stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileEstimationStageKind {
    /// Optional bounded lattice alignment with the profile held fixed.
    LatticeAlignment,
    /// Constant Gaussian variance only.
    W,
    /// Caglioti `U/V/W` Gaussian variance.
    Uvw,
    /// Full TCH `U/V/W/X/Y` effective profile.
    Uvwxy,
}

impl ProfileEstimationStageKind {
    /// Return the stable scripting/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LatticeAlignment => "lattice_alignment",
            Self::W => "w",
            Self::Uvw => "uvw",
            Self::Uvwxy => "uvwxy",
        }
    }
}

/// Validated input for one predominantly single-phase profile estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileEstimationInput {
    /// Integrated one-dimensional pattern. Existing masks and background are respected.
    pub pattern: PatternRecord,
    /// Fixed-wavelength starting profile. The wavelength is never selected for refinement.
    pub instrument: ConstantWavelengthInstrument,
    /// Dominant phase with independent starting reflection intensities.
    pub phase: LeBailPhase,
}

impl ProfileEstimationInput {
    /// Validate and own the estimation input.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileEstimationError`] for invalid pattern, instrument, or phase state.
    pub fn new(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        phase: LeBailPhase,
    ) -> Result<Self, ProfileEstimationError> {
        LeBailInput::new(pattern.clone(), instrument, vec![phase.clone()])?;
        Ok(Self {
            pattern,
            instrument,
            phase,
        })
    }
}

/// Deterministic controls for staged effective-profile estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileEstimationOptions {
    /// Requested model-selection behavior.
    pub mode: ProfileEstimationMode,
    /// Align a bounded dynamic phase cell before fitting widths.
    pub align_lattice: bool,
    /// Minimum fractional Rwp decrease required by an automatic complexity increase.
    pub minimum_relative_rwp_improvement: f64,
    /// Minimum absolute Rwp decrease required by an automatic complexity increase.
    pub minimum_absolute_rwp_improvement: f64,
    /// Largest permitted absolute covariance correlation for an automatic candidate.
    pub maximum_absolute_correlation: f64,
    /// Underlying independent-intensity extraction and profile-solver controls.
    pub lebail: LeBailOptions,
}

impl ProfileEstimationOptions {
    /// Construct and validate explicit estimation controls.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileEstimationError::InvalidOptions`] for invalid thresholds.
    pub fn new(
        mode: ProfileEstimationMode,
        align_lattice: bool,
        minimum_relative_rwp_improvement: f64,
        minimum_absolute_rwp_improvement: f64,
        maximum_absolute_correlation: f64,
        lebail: LeBailOptions,
    ) -> Result<Self, ProfileEstimationError> {
        let options = Self {
            mode,
            align_lattice,
            minimum_relative_rwp_improvement,
            minimum_absolute_rwp_improvement,
            maximum_absolute_correlation,
            lebail,
        };
        options.validate()?;
        Ok(options)
    }

    /// Conservative scripting defaults with an explicit native execution policy.
    ///
    /// Automatic candidates must lower Rwp by at least 0.2% relative and
    /// `1e-5` absolute, and keep every pairwise absolute covariance
    /// correlation at or below 0.98.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileEstimationError`] if the underlying controls are invalid.
    pub fn scripting_defaults(execution: ExecutionPolicy) -> Result<Self, ProfileEstimationError> {
        Self::new(
            ProfileEstimationMode::Automatic,
            false,
            0.002,
            1.0e-5,
            0.98,
            LeBailOptions::scripting_defaults(execution)?,
        )
    }

    fn validate(&self) -> Result<(), ProfileEstimationError> {
        if !self.minimum_relative_rwp_improvement.is_finite()
            || self.minimum_relative_rwp_improvement < 0.0
        {
            return Err(ProfileEstimationError::InvalidOptions {
                message: "minimum_relative_rwp_improvement must be non-negative and finite"
                    .to_owned(),
            });
        }
        if !self.minimum_absolute_rwp_improvement.is_finite()
            || self.minimum_absolute_rwp_improvement < 0.0
        {
            return Err(ProfileEstimationError::InvalidOptions {
                message: "minimum_absolute_rwp_improvement must be non-negative and finite"
                    .to_owned(),
            });
        }
        if !self.maximum_absolute_correlation.is_finite()
            || !(0.0..1.0).contains(&self.maximum_absolute_correlation)
        {
            return Err(ProfileEstimationError::InvalidOptions {
                message: "maximum_absolute_correlation must lie in [0, 1)".to_owned(),
            });
        }
        Ok(())
    }
}

/// Compact audit record for one attempted stage.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileEstimationStage {
    /// Stable stage identity.
    pub kind: ProfileEstimationStageKind,
    /// Instrument parameters selected during this stage.
    pub instrument_parameters: Vec<String>,
    /// Whether this stage became the starting state for the next stage/final result.
    pub accepted: bool,
    /// Machine-displayable decision explanation.
    pub decision: String,
    /// Rwp obtained by this candidate.
    pub rwp: f64,
    /// Candidate profile, retained even when automatic selection rejects it.
    pub instrument: ConstantWavelengthInstrument,
    /// Largest absolute off-diagonal covariance correlation, when identifiable.
    pub maximum_absolute_correlation: Option<f64>,
}

/// Final effective starting profile and the complete staged selection audit.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileEstimationResult {
    /// Accepted profile model.
    pub instrument: ConstantWavelengthInstrument,
    /// Accepted instrument parameter names in stable order.
    pub active_parameters: Vec<String>,
    /// Final Le Bail result, including independent intensities and covariance.
    pub lebail: LeBailResult,
    /// Every attempted stage in execution order.
    pub stages: Vec<ProfileEstimationStage>,
    /// Scientific-interpretation warnings.
    pub warnings: Vec<String>,
}

/// Construct a valid all-Gaussian starting profile from an approximate FWHM.
///
/// The returned `W` is `(FWHM / 2.354820045...)^2`; `U`, `V`, `X`, and `Y`
/// are zero. This is only an optimizer seed, not a calibrated instrument model.
///
/// # Errors
///
/// Returns [`ProfileEstimationError`] for a non-positive/non-finite input.
pub fn starting_profile_from_fwhm(
    wavelength_angstrom: f64,
    fwhm_deg: f64,
) -> Result<ConstantWavelengthInstrument, ProfileEstimationError> {
    if !wavelength_angstrom.is_finite()
        || wavelength_angstrom <= 0.0
        || !fwhm_deg.is_finite()
        || fwhm_deg <= 0.0
    {
        return Err(ProfileEstimationError::InvalidStartingFwhm);
    }
    Ok(ConstantWavelengthInstrument {
        wavelength_angstrom,
        u_deg2: 0.0,
        v_deg2: 0.0,
        w_deg2: (fwhm_deg / GAUSSIAN_FWHM_PER_SIGMA).powi(2),
        x_deg: 0.0,
        y_deg: 0.0,
    })
}

/// Estimate an effective CW starting profile from one dominant phase.
///
/// Wavelength is immutable by construction: no wavelength parameter is added
/// to any stage, and the final bit pattern is checked against the input. The
/// result is deliberately called *effective* because crystallite size,
/// microstrain, pressure gradients, and other sample broadening can be folded
/// into the fitted coefficients.
///
/// # Errors
///
/// Returns [`ProfileEstimationError`] for invalid controls, an unavailable
/// lattice-alignment domain, or a failed Le Bail stage.
pub fn estimate_effective_profile(
    input: &ProfileEstimationInput,
    options: &ProfileEstimationOptions,
) -> Result<ProfileEstimationResult, ProfileEstimationError> {
    options.validate()?;
    let wavelength_bits = input.instrument.wavelength_angstrom.to_bits();
    let mut instrument = input.instrument;
    let mut phases = vec![input.phase.clone()];
    let mut stages = Vec::new();

    if options.align_lattice {
        let aligned = align_lattice_stage(input, instrument, phases, options, wavelength_bits)?;
        instrument = aligned.0;
        phases = aligned.1;
        stages.push(aligned.2);
    }

    let mut accepted = fit_profile_stage(
        &input.pattern,
        instrument,
        phases,
        &W_PARAMETERS,
        &options.lebail,
    )?;
    ensure_fixed_wavelength(wavelength_bits, accepted.instrument)?;
    validate_profile_domain(&input.pattern, &accepted)?;
    stages.push(stage_record(
        ProfileEstimationStageKind::W,
        &W_PARAMETERS,
        &accepted,
        true,
        "accepted required baseline model".to_owned(),
    ));
    let mut active = W_PARAMETERS.iter().map(ToString::to_string).collect();

    if options.mode != ProfileEstimationMode::WOnly {
        let uvw = fit_profile_stage(
            &input.pattern,
            accepted.instrument,
            accepted.phases.clone(),
            &UVW_PARAMETERS,
            &options.lebail,
        )?;
        ensure_fixed_wavelength(wavelength_bits, uvw.instrument)?;
        validate_profile_domain(&input.pattern, &uvw)?;
        let (accept, decision) = candidate_decision(&accepted, &uvw, options, 3);
        let forced = options.mode != ProfileEstimationMode::Automatic;
        let accepted_stage = forced || accept;
        stages.push(stage_record(
            ProfileEstimationStageKind::Uvw,
            &UVW_PARAMETERS,
            &uvw,
            accepted_stage,
            if forced {
                "accepted because UVW was explicitly requested".to_owned()
            } else {
                decision
            },
        ));
        if accepted_stage {
            accepted = uvw;
            active = UVW_PARAMETERS.iter().map(ToString::to_string).collect();
        } else {
            return finish_result(accepted, active, stages, wavelength_bits);
        }
    }

    if options.mode == ProfileEstimationMode::Uvwxy
        || options.mode == ProfileEstimationMode::Automatic
    {
        let uvwxy = fit_profile_stage(
            &input.pattern,
            accepted.instrument,
            accepted.phases.clone(),
            &UVWXY_PARAMETERS,
            &options.lebail,
        )?;
        ensure_fixed_wavelength(wavelength_bits, uvwxy.instrument)?;
        validate_profile_domain(&input.pattern, &uvwxy)?;
        let (accept, decision) = candidate_decision(&accepted, &uvwxy, options, 5);
        let forced = options.mode == ProfileEstimationMode::Uvwxy;
        let accepted_stage = forced || accept;
        stages.push(stage_record(
            ProfileEstimationStageKind::Uvwxy,
            &UVWXY_PARAMETERS,
            &uvwxy,
            accepted_stage,
            if forced {
                "accepted because UVWXY was explicitly requested".to_owned()
            } else {
                decision
            },
        ));
        if accepted_stage {
            accepted = uvwxy;
            active = UVWXY_PARAMETERS.iter().map(ToString::to_string).collect();
        }
    }

    finish_result(accepted, active, stages, wavelength_bits)
}

fn align_lattice_stage(
    input: &ProfileEstimationInput,
    instrument: ConstantWavelengthInstrument,
    phases: Vec<LeBailPhase>,
    options: &ProfileEstimationOptions,
    wavelength_bits: u64,
) -> Result<
    (
        ConstantWavelengthInstrument,
        Vec<LeBailPhase>,
        ProfileEstimationStage,
    ),
    ProfileEstimationError,
> {
    if phases[0].reflection_domain().is_none() {
        return Err(ProfileEstimationError::LatticeAlignmentRequiresDynamicPhase);
    }
    let parameters =
        build_lebail_parameter_set_with_lattice(instrument, &phases, &[], false, false, true)?;
    let result = run_stage(
        &input.pattern,
        instrument,
        phases,
        parameters,
        &options.lebail,
    )?;
    ensure_fixed_wavelength(wavelength_bits, result.instrument)?;
    let stage = ProfileEstimationStage {
        kind: ProfileEstimationStageKind::LatticeAlignment,
        instrument_parameters: Vec::new(),
        accepted: true,
        decision: "accepted bounded nuisance lattice alignment".to_owned(),
        rwp: result.metrics.rwp,
        instrument: result.instrument,
        maximum_absolute_correlation: covariance_maximum_correlation(result.covariance.as_ref()),
    };
    Ok((result.instrument, result.phases, stage))
}

fn fit_profile_stage(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    phases: Vec<LeBailPhase>,
    names: &[&str],
    options: &LeBailOptions,
) -> Result<LeBailResult, ProfileEstimationError> {
    let parameters = build_lebail_parameter_set(instrument, &phases, names, false, false)?;
    run_stage(pattern, instrument, phases, parameters, options)
}

fn run_stage(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    phases: Vec<LeBailPhase>,
    parameters: crate::ParameterSet,
    options: &LeBailOptions,
) -> Result<LeBailResult, ProfileEstimationError> {
    let request = LeBailInput::new_with_parameters(
        pattern.clone(),
        instrument,
        phases,
        parameters,
        Vec::new(),
    )?;
    refine_lebail(&request, options, None).map_err(ProfileEstimationError::LeBail)
}

fn candidate_decision(
    baseline: &LeBailResult,
    candidate: &LeBailResult,
    options: &ProfileEstimationOptions,
    expected_parameters: usize,
) -> (bool, String) {
    let denominator = baseline.metrics.rwp.abs().max(f64::MIN_POSITIVE);
    let absolute_improvement = baseline.metrics.rwp - candidate.metrics.rwp;
    let improvement = absolute_improvement / denominator;
    if absolute_improvement < options.minimum_absolute_rwp_improvement {
        return (
            false,
            format!(
                "rejected: absolute Rwp improvement {absolute_improvement:.6} is below {:.6}",
                options.minimum_absolute_rwp_improvement
            ),
        );
    }
    if improvement < options.minimum_relative_rwp_improvement {
        return (
            false,
            format!(
                "rejected: relative Rwp improvement {improvement:.6} is below {:.6}",
                options.minimum_relative_rwp_improvement
            ),
        );
    }
    let Some(covariance) = candidate.covariance.as_ref() else {
        return (
            false,
            "rejected: candidate covariance is not identifiable".to_owned(),
        );
    };
    if covariance.size != expected_parameters {
        return (
            false,
            "rejected: candidate covariance dimension is inconsistent".to_owned(),
        );
    }
    let Some(correlation) = covariance_maximum_correlation(Some(covariance)) else {
        return (
            false,
            "rejected: candidate covariance has a non-positive variance".to_owned(),
        );
    };
    if correlation > options.maximum_absolute_correlation {
        return (
            false,
            format!(
                "rejected: maximum absolute correlation {correlation:.6} exceeds {:.6}",
                options.maximum_absolute_correlation
            ),
        );
    }
    (
        true,
        format!(
            "accepted: relative Rwp improvement {improvement:.6}, maximum absolute correlation {correlation:.6}"
        ),
    )
}

fn covariance_maximum_correlation(covariance: Option<&CovarianceMatrix>) -> Option<f64> {
    let covariance = covariance?;
    if covariance.size == 0 {
        return Some(0.0);
    }
    let mut maximum = 0.0_f64;
    for row in 0..covariance.size {
        let row_variance = covariance.values[row * covariance.size + row];
        if !row_variance.is_finite() || row_variance <= 0.0 {
            return None;
        }
        for column in 0..row {
            let column_variance = covariance.values[column * covariance.size + column];
            if !column_variance.is_finite() || column_variance <= 0.0 {
                return None;
            }
            let correlation = covariance.values[row * covariance.size + column].abs()
                / (row_variance * column_variance).sqrt();
            if !correlation.is_finite() {
                return None;
            }
            maximum = maximum.max(correlation);
        }
    }
    Some(maximum)
}

fn stage_record(
    kind: ProfileEstimationStageKind,
    parameters: &[&str],
    result: &LeBailResult,
    accepted: bool,
    decision: String,
) -> ProfileEstimationStage {
    ProfileEstimationStage {
        kind,
        instrument_parameters: parameters.iter().map(ToString::to_string).collect(),
        accepted,
        decision,
        rwp: result.metrics.rwp,
        instrument: result.instrument,
        maximum_absolute_correlation: covariance_maximum_correlation(result.covariance.as_ref()),
    }
}

fn validate_profile_domain(
    pattern: &PatternRecord,
    result: &LeBailResult,
) -> Result<(), ProfileEstimationError> {
    let mut angles = result
        .phases
        .iter()
        .flat_map(|phase| phase.two_theta_deg().iter().copied())
        .collect::<Vec<_>>();
    if let (Some(first), Some(last)) = (pattern.x_deg.first(), pattern.x_deg.last()) {
        if *first > 0.0 && *first < 180.0 {
            angles.push(*first);
        }
        if *last > 0.0 && *last < 180.0 {
            angles.push(*last);
        }
    }
    for angle in angles {
        CwProfileParameters::from_instrument(angle, result.instrument).map_err(|error| {
            ProfileEstimationError::ProfileOutsideDomain {
                angle_deg: angle,
                message: error.to_string(),
            }
        })?;
    }
    Ok(())
}

fn ensure_fixed_wavelength(
    expected_bits: u64,
    instrument: ConstantWavelengthInstrument,
) -> Result<(), ProfileEstimationError> {
    if instrument.wavelength_angstrom.to_bits() != expected_bits {
        return Err(ProfileEstimationError::WavelengthChanged);
    }
    Ok(())
}

fn finish_result(
    accepted: LeBailResult,
    active_parameters: Vec<String>,
    stages: Vec<ProfileEstimationStage>,
    wavelength_bits: u64,
) -> Result<ProfileEstimationResult, ProfileEstimationError> {
    ensure_fixed_wavelength(wavelength_bits, accepted.instrument)?;
    Ok(ProfileEstimationResult {
        instrument: accepted.instrument,
        active_parameters,
        lebail: accepted,
        stages,
        warnings: vec![
            "effective profile may include crystallite-size, microstrain, pressure-gradient, and other sample broadening"
                .to_owned(),
            "mask unidentified impurity peaks or excluded regions before using this estimate"
                .to_owned(),
        ],
    })
}

/// Errors from effective-profile estimation.
#[derive(Debug)]
pub enum ProfileEstimationError {
    /// Invalid estimator controls.
    InvalidOptions {
        /// Human-readable validation detail.
        message: String,
    },
    /// Approximate FWHM or wavelength cannot seed a valid profile.
    InvalidStartingFwhm,
    /// Lattice alignment was requested for a fixed-reflection phase.
    LatticeAlignmentRequiresDynamicPhase,
    /// A stage unexpectedly changed the fixed wavelength.
    WavelengthChanged,
    /// A fitted profile is invalid at an observed or reflection angle.
    ProfileOutsideDomain {
        /// Angle at which validation failed.
        angle_deg: f64,
        /// Underlying profile-domain failure.
        message: String,
    },
    /// Underlying Le Bail validation or solve failure.
    LeBail(LeBailError),
}

impl Display for ProfileEstimationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions { message } => write!(formatter, "invalid options: {message}"),
            Self::InvalidStartingFwhm => write!(
                formatter,
                "starting wavelength and approximate FWHM must be positive and finite"
            ),
            Self::LatticeAlignmentRequiresDynamicPhase => write!(
                formatter,
                "lattice alignment requires a phase with a bounded lattice reflection domain"
            ),
            Self::WavelengthChanged => {
                write!(formatter, "fixed wavelength changed during estimation")
            }
            Self::ProfileOutsideDomain { angle_deg, message } => write!(
                formatter,
                "fitted profile is invalid at {angle_deg} degrees 2theta: {message}"
            ),
            Self::LeBail(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ProfileEstimationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LeBail(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LeBailError> for ProfileEstimationError {
    fn from(value: LeBailError) -> Self {
        Self::LeBail(value)
    }
}
