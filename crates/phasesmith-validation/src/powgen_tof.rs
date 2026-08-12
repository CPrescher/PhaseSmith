//! POWGEN `LaB6` time-of-flight profile and Le Bail acceptance workflow.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

use phasesmith_core::{BackgroundError, TofProfileParameters, smooth_bruckner};
use phasesmith_crystallography::{
    PreparedReflectionGenerator, ReflectionGenerationError, ReflectionRange, UnitCell,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    GsasTofInstrumentIoError, GsasTofInstrumentReadLimits, PowderIoError, PowderReadLimits,
    SpaceGroupLookupError, read_gsas_tof_instrument_file, read_tof_powder_file,
    space_group_by_number,
};
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};
use phasesmith_workflows::{
    ResidualOptions, TofChebyshevBackground, TofLeBailError, TofLeBailInput, TofLeBailOptions,
    TofLeBailPhase, calculate_tof_lebail_pattern, evaluate_tof_residuals, refine_tof_lebail,
};

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "powgen-lab6-tof-calibration";
const EXPECTED_BANK: usize = 2;
const EXPECTED_SAMPLES: usize = 6_824;
const LAB6_LATTICE_ANGSTROM: f64 = 4.156_826;
const MINIMUM_SEARCH_D_ANGSTROM: f64 = 0.25;
const MAXIMUM_SEARCH_D_ANGSTROM: f64 = 5.0;

/// Run the complete native fixed-instrument TOF Le Bail acceptance workflow.
///
/// The official bank-2 type-3 profile record is translated once at the validation
/// boundary into `PhaseSmith`'s published 15-coefficient convention. Reflection
/// families are generated for certified cubic `LaB6`; values and analytical
/// derivatives are accumulated on the logarithmic microsecond grid before
/// nonnegative integrated-intensity extraction.
///
/// # Errors
///
/// Returns [`PowgenTofValidationError`] for missing/corrupt data or malformed tutorial records.
#[allow(clippy::too_many_lines)]
pub fn run_powgen_tof_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, PowgenTofValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let pattern = read_tof_powder_file(
        dataset_directory.join("PG3_17541.gsa"),
        EXPECTED_BANK,
        PowderReadLimits::default(),
    )?;
    let calibration = read_gsas_tof_instrument_file(
        dataset_directory.join("PGHR_60-2015A.prm"),
        EXPECTED_BANK,
        GsasTofInstrumentReadLimits::default(),
    )?;
    let bank_two_theta_deg = calibration
        .bank_geometry
        .map(|geometry| geometry.two_theta_deg);
    let kernel_instrument = calibration.instrument;
    let at_one_angstrom = TofProfileParameters::from_instrument(1.0, kernel_instrument)
        .map_err(|error| PowgenTofValidationError::Kernel(error.to_string()))?;
    let expected_position = kernel_instrument.zero_us
        + kernel_instrument.difc_us_per_angstrom
        + kernel_instrument.difa_us_per_angstrom2
        + kernel_instrument.difb_us_angstrom;
    let derivative_step = 1.0e-6;
    let position_plus =
        TofProfileParameters::from_instrument(1.0 + derivative_step, kernel_instrument)
            .map_err(|error| PowgenTofValidationError::Kernel(error.to_string()))?
            .position_us;
    let position_minus =
        TofProfileParameters::from_instrument(1.0 - derivative_step, kernel_instrument)
            .map_err(|error| PowgenTofValidationError::Kernel(error.to_string()))?
            .position_us;
    let finite_position_derivative = (position_plus - position_minus) / (2.0 * derivative_step);
    let derivative_relative_error = ((finite_position_derivative - at_one_angstrom.d_position_d_d)
        / at_one_angstrom.d_position_d_d)
        .abs();
    let calibration_positions = [0.3, 1.0, 4.5]
        .map(|d_spacing| TofProfileParameters::from_instrument(d_spacing, kernel_instrument))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PowgenTofValidationError::Kernel(error.to_string()))?;
    let observed = pattern
        .pattern
        .observed_y
        .as_deref()
        .ok_or(PowgenTofValidationError::MissingObservations)?;
    let background_seed = smooth_bruckner(observed, 20, 50)?;
    let workflow_pattern = TofPatternRecord::new(
        pattern.pattern.tof_us.clone(),
        Some(observed.to_vec()),
        pattern.pattern.uncertainty.clone(),
        pattern.pattern.mask.clone(),
        Some(background_seed),
    )?;
    let reflections =
        PreparedReflectionGenerator::new(space_group_by_number(221)?.space_group, true, 1_000_000)?
            .generate(
                lab6_cell(),
                ReflectionRange::Tof {
                    min_us: workflow_pattern.tof_us[0],
                    max_us: *workflow_pattern
                        .tof_us
                        .last()
                        .ok_or(PowgenTofValidationError::MissingObservations)?,
                    search_min_d_angstrom: MINIMUM_SEARCH_D_ANGSTROM,
                    search_max_d_angstrom: MAXIMUM_SEARCH_D_ANGSTROM,
                    zero_us: kernel_instrument.zero_us,
                    difc_us_per_angstrom: kernel_instrument.difc_us_per_angstrom,
                    difa_us_per_angstrom2: kernel_instrument.difa_us_per_angstrom2,
                    difb_us_angstrom: kernel_instrument.difb_us_angstrom,
                },
            )?;
    let phase = TofLeBailPhase::new(
        RecordId::new("lab6")?,
        "NIST SRM 660b LaB6 calibration material",
        reflections
            .iter()
            .map(|reflection| reflection.reflection_id.clone())
            .collect(),
        reflections
            .iter()
            .map(|reflection| reflection.hkl)
            .collect(),
        reflections
            .iter()
            .map(|reflection| reflection.d_spacing_angstrom)
            .collect(),
        vec![0.0; reflections.len()],
        1.0,
    )?;
    let background = TofChebyshevBackground::new(
        RecordId::new("powgen-chebyshev-background")?,
        vec![0.0; 16],
        [
            pattern.pattern.tof_us[0],
            pattern.pattern.tof_us[pattern.pattern.tof_us.len() - 1],
        ],
    )?;
    let request = TofLeBailInput::new(workflow_pattern, kernel_instrument, vec![phase])?
        .with_refinable_background(background)?;
    let options = TofLeBailOptions::new(
        12,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2)?,
    )?
    .with_redistribution_uncertainty(false);
    let initial_calculation = calculate_tof_lebail_pattern(&request, &options)?;
    let initial_metrics = evaluate_tof_residuals(
        &request.pattern,
        &initial_calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: 0,
        },
    )
    .map_err(TofLeBailError::Residual)?;
    let result = refine_tof_lebail(&request, &options)?;
    let first_rwp = result
        .history
        .first()
        .ok_or(PowgenTofValidationError::EmptyHistory)?
        .metrics
        .rwp;
    let correlation = pearson_correlation(
        observed,
        &result.calculation.background_y,
        &result.calculation.profile_y,
        request.pattern.mask.as_deref(),
    )?;
    let finite_nonnegative = result
        .intensities
        .iter()
        .all(|item| item.integrated_intensity.is_finite() && item.integrated_intensity >= 0.0);
    let positions_in_range = reflections.iter().all(|reflection| {
        TofProfileParameters::from_instrument(reflection.d_spacing_angstrom, kernel_instrument)
            .is_ok_and(|parameters| {
                parameters.position_us >= request.pattern.tof_us[0]
                    && parameters.position_us
                        <= request.pattern.tof_us[request.pattern.sample_count() - 1]
            })
    });
    let checks = vec![
        ValidationCheck::new(
            "tof_bank_geometry",
            if bank_two_theta_deg.is_some_and(|value| value.to_bits() == 90.0_f64.to_bits()) {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            "The independently documented BNKPAR scattering-angle field is imported as typed bank geometry.",
            bank_two_theta_deg,
            Some("POWGEN bank 2 two_theta_deg = 90.000".to_owned()),
        )?,
        ValidationCheck::new(
            "tof_fxye_grid",
            if pattern.pattern.tof_us.len() == EXPECTED_SAMPLES
                && pattern
                    .pattern
                    .tof_us
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
                && pattern
                    .pattern
                    .uncertainty
                    .as_deref()
                    .is_some_and(|values| {
                        values.iter().all(|value| value.is_finite() && *value > 0.0)
                    })
            {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            "The SLOG FXYE bank is parsed in native microseconds without the constant-wavelength centidegree conversion.",
            Some(f64::from(
                u32::try_from(pattern.pattern.tof_us.len()).map_err(|_| {
                    PowgenTofValidationError::InvalidData("sample count exceeds u32".to_owned())
                })?,
            )),
            Some(
                "6824 sorted bin-center TOF density samples with finite positive uncertainties"
                    .to_owned(),
            ),
        )?,
        ValidationCheck::new(
            "tof_position_kernel",
            if at_one_angstrom.position_us.to_bits() == expected_position.to_bits() {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            "The production TOF position law consumes the pinned bank-2 zero/DIFC/DIFA/DIFB coefficients in documented units.",
            Some(at_one_angstrom.position_us),
            Some("position(d=1 A) = zero + DIFC + DIFA + DIFB".to_owned()),
        )?,
        ValidationCheck::new(
            "tof_position_derivative",
            if derivative_relative_error <= 1.0e-9 {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            "The analytical d-spacing derivative of the pinned TOF calibration matches a centered finite difference.",
            Some(derivative_relative_error),
            Some("relative centered-difference error at d=1 A <= 1e-9".to_owned()),
        )?,
        ValidationCheck::new(
            "tof_calibration_range",
            if calibration_positions
                .windows(2)
                .all(|pair| pair[0].position_us < pair[1].position_us)
                && calibration_positions.iter().all(|parameters| {
                    parameters.position_us >= pattern.pattern.tof_us[0]
                        && parameters.position_us
                            <= pattern.pattern.tof_us[pattern.pattern.tof_us.len() - 1]
                })
            {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            "Representative d-spacings map monotonically across the observed bank range.",
            Some(calibration_positions[1].position_us),
            Some("d=0.3, 1.0, and 4.5 A positions are ordered and observed".to_owned()),
        )?,
        check(
            "tof_profile_improvement",
            result.metrics.rwp < initial_metrics.rwp,
            "Native TOF redistribution with a refinable 16-term Chebyshev background lowers Rwp relative to the zero-reflection starting pattern.",
            Some(result.metrics.rwp / initial_metrics.rwp),
            "final Rwp / zero-reflection Rwp < 1",
        )?,
        check(
            "tof_profile_correlation",
            correlation >= 0.95,
            "The background-subtracted POWGEN observations and extracted native TOF profile remain aligned.",
            Some(correlation),
            "masked Pearson correlation >= 0.95",
        )?,
        check(
            "tof_integrated_intensities",
            finite_nonnegative,
            "Every extracted LaB6 integrated intensity is finite and nonnegative.",
            None,
            "all finite and >= 0",
        )?,
        check(
            "tof_reflection_coverage",
            positions_in_range && reflections.len() >= 300,
            "Every generated LaB6 family center lies inside the observed microsecond interval.",
            Some(count_as_f64(reflections.len())?),
            "at least 300 generated families, all centers inside the observed range",
        )?,
        check(
            "tof_analytical_derivatives",
            result
                .calculation
                .accumulation
                .derivatives
                .global
                .as_ref()
                .is_some_and(|jacobian| jacobian.parameter_count == 15),
            "The real-pattern calculation returned all shared instrument derivatives from the same fused pass.",
            Some(15.0),
            "15 instrument derivative rows",
        )?,
        check(
            "tof_chebyshev_background",
            result
                .calculation
                .background_basis
                .as_ref()
                .is_some_and(|basis| {
                    basis.rows == pattern.pattern.tof_us.len() && basis.columns == 16
                })
                && result.background.as_ref().is_some_and(|background| {
                    background
                        .coefficients()
                        .iter()
                        .all(|value| value.is_finite())
                }),
            "The final TOF calculation exposes all analytical Chebyshev coefficient derivatives and finite refined coefficients.",
            Some(16.0),
            "16 finite coefficients and 16 sample-major derivative columns",
        )?,
    ];
    RealDataValidationReport::new(
        DATASET_ID,
        pattern.pattern.tof_us.len(),
        Some(reflections.len()),
        started.elapsed().as_secs_f64(),
        checks,
        vec![
            format!(
                "Bank 2 range={:.6}..{:.6} us; DIFC={:.6} us/A, DIFA={:.6} us/A^2, DIFB={:.6} us A.",
                pattern.pattern.tof_us[0],
                pattern.pattern.tof_us[pattern.pattern.tof_us.len() - 1],
                kernel_instrument.difc_us_per_angstrom,
                kernel_instrument.difa_us_per_angstrom2,
                kernel_instrument.difb_us_angstrom,
            ),
            format!(
                "Initial Rwp={:.8}; first-cycle Rwp={first_rwp:.8}; final Rwp={:.8}; correlation={correlation:.8}; background=16-term Chebyshev.",
                initial_metrics.rwp,
                result.metrics.rwp,
            ),
            "The SLOG FXYE data remain in a TofPatternRecord; no angle-domain conversion is used."
                .to_owned(),
        ],
    )
    .map_err(Into::into)
}

/// Backward-compatible name retained for callers of the initial readiness runner.
///
/// # Errors
///
/// Returns [`PowgenTofValidationError`] under the same conditions as
/// [`run_powgen_tof_validation`].
pub fn run_powgen_tof_readiness(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, PowgenTofValidationError> {
    run_powgen_tof_validation(dataset_directory)
}

fn lab6_cell() -> UnitCell {
    UnitCell {
        a_angstrom: LAB6_LATTICE_ANGSTROM,
        b_angstrom: LAB6_LATTICE_ANGSTROM,
        c_angstrom: LAB6_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn pearson_correlation(
    observed: &[f64],
    background: &[f64],
    calculated_profile: &[f64],
    mask: Option<&[bool]>,
) -> Result<f64, PowgenTofValidationError> {
    if observed.len() != background.len() || observed.len() != calculated_profile.len() {
        return Err(PowgenTofValidationError::InvalidData(
            "TOF correlation arrays have different lengths".to_owned(),
        ));
    }
    let selected = (0..observed.len())
        .filter(|index| mask.is_none_or(|values| values[*index]))
        .map(|index| {
            (
                observed[index] - background[index],
                calculated_profile[index],
            )
        })
        .collect::<Vec<_>>();
    let count = count_as_f64(selected.len())?;
    let observed_mean = selected.iter().map(|item| item.0).sum::<f64>() / count;
    let calculated_mean = selected.iter().map(|item| item.1).sum::<f64>() / count;
    let (numerator, observed_square, calculated_square) = selected.iter().fold(
        (0.0, 0.0, 0.0),
        |(numerator, observed_square, calculated_square), (observed, calculated)| {
            let observed = observed - observed_mean;
            let calculated = calculated - calculated_mean;
            (
                numerator + observed * calculated,
                observed_square + observed * observed,
                calculated_square + calculated * calculated,
            )
        },
    );
    let value = numerator / (observed_square * calculated_square).sqrt();
    value.is_finite().then_some(value).ok_or_else(|| {
        PowgenTofValidationError::InvalidData("invalid TOF profile correlation".to_owned())
    })
}

fn check(
    check_id: &str,
    passed: bool,
    detail: &str,
    measured: Option<f64>,
    criterion: &str,
) -> Result<ValidationCheck, ValidationContractError> {
    ValidationCheck::new(
        check_id,
        if passed {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        },
        detail,
        measured,
        Some(criterion.to_owned()),
    )
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> Result<f64, PowgenTofValidationError> {
    if value == 0 {
        return Err(PowgenTofValidationError::InvalidData(
            "TOF validation selection is empty".to_owned(),
        ));
    }
    Ok(value as f64)
}

/// POWGEN TOF readiness setup, input, or report failure.
#[derive(Debug)]
pub enum PowgenTofValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Text input failed.
    Io(std::io::Error),
    /// Typed TOF/domain record validation failed.
    Domain(DomainError),
    /// Typed TOF powder input failed.
    Powder(PowderIoError),
    /// Legacy GSAS TOF calibration import failed.
    Instrument(GsasTofInstrumentIoError),
    /// Space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Reflection generation failed.
    Reflection(ReflectionGenerationError),
    /// Background estimation failed.
    Background(BackgroundError),
    /// TOF workflow evaluation failed.
    Workflow(TofLeBailError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// The observed pattern is absent or an iteration history is empty.
    MissingObservations,
    /// The workflow returned no accepted iterations.
    EmptyHistory,
    /// Pinned GSAS records did not match their bounded schema.
    InvalidData(String),
    /// The production TOF kernel rejected the pinned calibration.
    Kernel(String),
    /// Report construction failed.
    Report(ValidationContractError),
}

impl Display for PowgenTofValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Powder(error) => Display::fmt(error, formatter),
            Self::Instrument(error) => Display::fmt(error, formatter),
            Self::SpaceGroup(error) => Display::fmt(error, formatter),
            Self::Reflection(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::MissingObservations => formatter.write_str("POWGEN observations are missing"),
            Self::EmptyHistory => formatter.write_str("POWGEN TOF extraction history is empty"),
            Self::InvalidData(message) | Self::Kernel(message) => formatter.write_str(message),
            Self::Report(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for PowgenTofValidationError {}

impl From<DatasetVerificationError> for PowgenTofValidationError {
    fn from(value: DatasetVerificationError) -> Self {
        Self::Dataset(value)
    }
}

impl From<std::io::Error> for PowgenTofValidationError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<PowderIoError> for PowgenTofValidationError {
    fn from(value: PowderIoError) -> Self {
        Self::Powder(value)
    }
}

impl From<GsasTofInstrumentIoError> for PowgenTofValidationError {
    fn from(value: GsasTofInstrumentIoError) -> Self {
        Self::Instrument(value)
    }
}

impl From<DomainError> for PowgenTofValidationError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<SpaceGroupLookupError> for PowgenTofValidationError {
    fn from(value: SpaceGroupLookupError) -> Self {
        Self::SpaceGroup(value)
    }
}

impl From<ReflectionGenerationError> for PowgenTofValidationError {
    fn from(value: ReflectionGenerationError) -> Self {
        Self::Reflection(value)
    }
}

impl From<BackgroundError> for PowgenTofValidationError {
    fn from(value: BackgroundError) -> Self {
        Self::Background(value)
    }
}

impl From<TofLeBailError> for PowgenTofValidationError {
    fn from(value: TofLeBailError) -> Self {
        Self::Workflow(value)
    }
}

impl From<ExecutionPolicyError> for PowgenTofValidationError {
    fn from(value: ExecutionPolicyError) -> Self {
        Self::Execution(value)
    }
}

impl From<ValidationContractError> for PowgenTofValidationError {
    fn from(value: ValidationContractError) -> Self {
        Self::Report(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_parser_is_bank_specific_and_finite() {
        let source = concat!(
            "INS  1 ICONS1 2 3 4\n",
            "INS  2 ICONS22581.63 0 4.41 0\n",
            "INS  2PRCF1     3 21 0.002\n",
            "INS  2PRCF11 0.257460 0.091563 0.017334 0\n",
            "INS  2PRCF12 10 203.581 0 10.651\n",
        );
        let calibration = phasesmith_io::parse_gsas_tof_instrument_text(
            source,
            2,
            GsasTofInstrumentReadLimits::default(),
        )
        .unwrap()
        .instrument;
        assert_eq!(
            calibration.difc_us_per_angstrom.to_bits(),
            22_581.63_f64.to_bits()
        );
        assert_eq!(
            calibration.difa_us_per_angstrom2.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(calibration.zero_us.to_bits(), 4.41_f64.to_bits());
        assert_eq!(calibration.difb_us_angstrom.to_bits(), 0.0_f64.to_bits());
        assert_eq!(calibration.sigma0_us2.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            calibration.sigma1_us2_per_angstrom2.to_bits(),
            10.0_f64.to_bits()
        );
        assert_eq!(
            calibration.sigma2_us2_per_angstrom4.to_bits(),
            203.581_f64.to_bits()
        );
        assert!(
            phasesmith_io::parse_gsas_tof_instrument_text(
                "INS  1 ICONS1 2 3 4\n",
                2,
                GsasTofInstrumentReadLimits::default(),
            )
            .is_err()
        );
    }
}
