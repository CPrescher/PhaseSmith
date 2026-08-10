use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};
use phasesmith_core::{BackgroundError, ConstantWavelengthInstrument, smooth_bruckner};
use phasesmith_crystallography::{
    PreparedReflectionGenerator, ReflectionGenerationError, ReflectionRange, UnitCell,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    PowderFormat, PowderIoError, PowderReadLimits, SpaceGroupLookupError, read_powder_file,
    space_group_by_number,
};
use phasesmith_model::{DomainError, PatternRecord};
use phasesmith_workflows::{
    BackgroundModel, ChebyshevBackground, LeBailError, LeBailInput, LeBailOptions, LeBailPhase,
    LeBailResult, build_lebail_parameter_set, refine_lebail,
};

const DATASET_ID: &str = "aps-sucrose-11bmb";

/// Run the checksum-pinned APS sucrose Le Bail validation without Python.
///
/// # Errors
///
/// Returns [`SucroseValidationError`] for invalid/missing data or native numerical failures.
pub fn run_sucrose_lebail_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, SucroseValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let imported = read_powder_file(
        dataset_directory.join("11bmb_8716.fxye"),
        PowderFormat::GsasFxye,
        1,
        PowderReadLimits::default(),
    )?;
    let selected = selected_pattern(&imported.pattern, 1.0, 24.0)?;
    let observed = selected
        .observed_y
        .as_deref()
        .ok_or(SucroseValidationError::MissingObservations)?;
    let fixed_background = smooth_bruckner(observed, 100, 50)?;
    let background = BackgroundModel::Chebyshev(
        ChebyshevBackground::new(
            "sucrose-background",
            vec![0.0],
            [
                selected.x_deg[0],
                selected.x_deg[selected.sample_count() - 1],
            ],
        )
        .map_err(LeBailError::Background)?,
    );
    let pattern = PatternRecord::new(
        selected.x_deg.clone(),
        Some(observed.to_vec()),
        selected.uncertainty.clone(),
        selected.mask.clone(),
        Some(fixed_background),
    )?;
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: 0.413_259,
        u_deg2: 1.163e-4,
        v_deg2: -0.126e-4,
        w_deg2: 0.063e-4,
        x_deg: 0.173e-2,
        y_deg: 0.0,
    };
    let phase = sucrose_phase(&pattern, instrument.wavelength_angstrom)?;
    let parameters = build_lebail_parameter_set(
        instrument,
        std::slice::from_ref(&phase),
        &["u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"],
        false,
        false,
    )?;
    let request =
        LeBailInput::new_with_parameters(pattern, instrument, vec![phase], parameters, Vec::new())?
            .with_refinable_background(background)?;
    let execution = ExecutionPolicy::new(Some(1), 2)?;
    let options = LeBailOptions::new(
        20,
        2,
        1.0e-7,
        1.0e-9,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        30.0,
        execution,
    )?
    .with_profile_controls(1.0e-10, 0.2, 8)?;
    let result = refine_lebail(&request, &options, None)?;
    let first_rwp = result
        .history
        .first()
        .ok_or(SucroseValidationError::EmptyHistory)?
        .rwp;
    let final_rwp = result.metrics.rwp;
    let finite_nonnegative = result
        .intensities
        .iter()
        .all(|item| item.integrated_intensity.is_finite() && item.integrated_intensity >= 0.0);
    let corrected_observed = observed
        .iter()
        .zip(&result.calculation.background_y)
        .map(|(observed, background)| observed - background)
        .collect::<Vec<_>>();
    let profile_correlation =
        pearson_correlation(&corrected_observed, &result.calculation.profile_y)?;
    sucrose_report(
        selected.sample_count(),
        started.elapsed().as_secs_f64(),
        &result,
        first_rwp,
        final_rwp,
        profile_correlation,
        finite_nonnegative,
    )
}

#[allow(clippy::too_many_arguments)]
fn sucrose_report(
    sample_count: usize,
    elapsed_seconds: f64,
    result: &LeBailResult,
    first_rwp: f64,
    final_rwp: f64,
    profile_correlation: f64,
    finite_nonnegative: bool,
) -> Result<RealDataValidationReport, SucroseValidationError> {
    let checks = vec![
        check(
            "profile_improvement",
            final_rwp < first_rwp,
            "Profile refinement lowers weighted residual relative to the first extraction cycle.",
            Some(final_rwp / first_rwp),
            "final Rwp / first-cycle Rwp < 1",
        )?,
        check(
            "smoke_rwp",
            final_rwp <= 0.22,
            "Current-model real-data smoke gate; this is not a GSAS-II equivalence tolerance.",
            Some(final_rwp),
            "Rwp <= 0.22",
        )?,
        check(
            "profile_correlation",
            profile_correlation >= 0.97,
            "Background-subtracted observed and calculated profiles remain strongly aligned.",
            Some(profile_correlation),
            "Pearson correlation >= 0.97",
        )?,
        check(
            "integrated_intensities",
            finite_nonnegative,
            "Every extracted integrated intensity is finite and nonnegative.",
            None,
            "all finite and >= 0",
        )?,
    ];
    RealDataValidationReport::new(
        DATASET_ID,
        sample_count,
        Some(result.intensities.len()),
        elapsed_seconds,
        checks,
        vec![
            format!(
                "First-cycle Rwp={first_rwp:.8}; final Rwp={final_rwp:.8}; final Rp={:.8}.",
                result.metrics.rp
            ),
            format!(
                "Termination={}; iterations={}.",
                result.termination_reason.as_str(),
                result.history.len()
            ),
            concat!(
                "Both workflows use the identical fixed Smooth Bruckner array plus one ",
                "refinable constant Chebyshev residual starting from zero."
            )
            .to_owned(),
            concat!(
                "Both matched workflows start from U=1.163, V=-0.126, W=0.063, X=0.173, ",
                "Y=0 in GSAS units and use the symmetric SH/L=0 profile."
            )
            .to_owned(),
        ],
    )
    .map_err(Into::into)
}

fn sucrose_phase(
    pattern: &PatternRecord,
    wavelength_angstrom: f64,
) -> Result<LeBailPhase, SucroseValidationError> {
    let cell = UnitCell {
        a_angstrom: 7.715_231_369_389_035,
        b_angstrom: 8.663_866_877_499_101,
        c_angstrom: 10.809_618_877_725_404,
        alpha_deg: 90.0,
        beta_deg: 102.982_491_937_325_56,
        gamma_deg: 90.0,
    };
    let group = space_group_by_number(4)?.space_group;
    let reflections = PreparedReflectionGenerator::new(group, true, 50_000_000)?.generate(
        cell,
        ReflectionRange::CwTwoTheta {
            min_deg: pattern.x_deg[0],
            max_deg: *pattern
                .x_deg
                .last()
                .ok_or(SucroseValidationError::EmptySelectedPattern)?,
            wavelength_angstrom,
        },
    )?;
    let positions = reflections
        .iter()
        .map(|reflection| {
            2.0 * (0.5 * wavelength_angstrom / reflection.d_spacing_angstrom)
                .asin()
                .to_degrees()
        })
        .collect::<Vec<_>>();
    LeBailPhase::new(
        "sucrose",
        "Sucrose validation cell",
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
        positions,
        vec![0.0; reflections.len()],
        1.0,
        Vec::new(),
    )
    .map_err(Into::into)
}

fn selected_pattern(
    source: &PatternRecord,
    minimum: f64,
    maximum: f64,
) -> Result<PatternRecord, SucroseValidationError> {
    let indices = source
        .x_deg
        .iter()
        .enumerate()
        .filter_map(|(index, x)| (*x >= minimum && *x <= maximum).then_some(index))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(SucroseValidationError::EmptySelectedPattern);
    }
    let select = |values: &[f64]| {
        indices
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>()
    };
    PatternRecord::new(
        select(&source.x_deg),
        source.observed_y.as_deref().map(select),
        source.uncertainty.as_deref().map(select),
        source
            .mask
            .as_deref()
            .map(|values| indices.iter().map(|index| values[*index]).collect()),
        Some(select(&source.background_y)),
    )
    .map_err(Into::into)
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Result<f64, SucroseValidationError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(SucroseValidationError::InvalidCorrelation);
    }
    let count = f64::from(
        u32::try_from(left.len()).map_err(|_| SucroseValidationError::SampleCountOverflow)?,
    );
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut numerator = 0.0;
    let mut left_squared = 0.0;
    let mut right_squared = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = left - left_mean;
        let right = right - right_mean;
        numerator += left * right;
        left_squared += left * left;
        right_squared += right * right;
    }
    let result = numerator / (left_squared * right_squared).sqrt();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SucroseValidationError::InvalidCorrelation)
    }
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

/// Native sucrose validation setup, input, or numerical failure.
#[derive(Debug)]
pub enum SucroseValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Powder input parsing failed.
    Powder(PowderIoError),
    /// Pattern construction failed.
    Pattern(DomainError),
    /// Smooth Bruckner startup estimation failed.
    Background(BackgroundError),
    /// Space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Reflection generation failed.
    Reflection(ReflectionGenerationError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// Native Le Bail setup or refinement failed.
    LeBail(LeBailError),
    /// Report construction failed.
    Report(ValidationContractError),
    /// The imported powder pattern did not contain observations.
    MissingObservations,
    /// No samples remained after selecting the validation interval.
    EmptySelectedPattern,
    /// The native solver returned no accepted iteration history.
    EmptyHistory,
    /// Profile correlation was undefined or non-finite.
    InvalidCorrelation,
    /// The sample count is too large for exact correlation accumulation.
    SampleCountOverflow,
}

impl Display for SucroseValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Powder(error) => Display::fmt(error, formatter),
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::SpaceGroup(error) => Display::fmt(error, formatter),
            Self::Reflection(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::LeBail(error) => Display::fmt(error, formatter),
            Self::Report(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("sucrose pattern observations are missing")
            }
            Self::EmptySelectedPattern => {
                formatter.write_str("sucrose validation interval contains no samples")
            }
            Self::EmptyHistory => {
                formatter.write_str("sucrose refinement returned no iteration history")
            }
            Self::InvalidCorrelation => {
                formatter.write_str("sucrose profile correlation is undefined")
            }
            Self::SampleCountOverflow => {
                formatter.write_str("sucrose sample count exceeds supported correlation size")
            }
        }
    }
}

impl Error for SucroseValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dataset(error) => Some(error),
            Self::Powder(error) => Some(error),
            Self::Pattern(error) => Some(error),
            Self::Background(error) => Some(error),
            Self::SpaceGroup(error) => Some(error),
            Self::Reflection(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::LeBail(error) => Some(error),
            Self::Report(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for SucroseValidationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(DatasetVerificationError, Dataset);
error_conversion!(PowderIoError, Powder);
error_conversion!(DomainError, Pattern);
error_conversion!(BackgroundError, Background);
error_conversion!(SpaceGroupLookupError, SpaceGroup);
error_conversion!(ReflectionGenerationError, Reflection);
error_conversion!(ExecutionPolicyError, Execution);
error_conversion!(LeBailError, LeBail);
error_conversion!(ValidationContractError, Report);
