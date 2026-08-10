//! ANSTO Echidna constant-wavelength neutron `LaB6` validation.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

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
    DifferentiableBackground, LeBailError, LeBailInput, LeBailOptions, LeBailPhase,
    build_lebail_parameter_set, refine_lebail,
};

use crate::background::{fitted_chebyshev_from_anchors, regular_background_anchors};
use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "ansto-echidna-lab6-cw-neutron";
const WAVELENGTH_ANGSTROM: f64 = 2.047;
const LAB6_LATTICE_ANGSTROM: f64 = 4.156_826;

/// Run a native profile/Le Bail smoke gate on the checksum-pinned Echidna pattern.
///
/// The deposited wavelength is documented as approximate, so this gate evaluates the
/// production CW profile and non-negative extraction while keeping the certified `LaB6` cell and
/// deposited wavelength fixed. It does not reinterpret the run as a wavelength certification.
///
/// # Errors
///
/// Returns [`EchidnaValidationError`] for invalid data or native numerical failures.
#[allow(clippy::too_many_lines)]
pub fn run_echidna_lab6_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, EchidnaValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let imported = read_powder_file(
        dataset_directory.join("ECH0034258_LaB6.xyd"),
        PowderFormat::Columns,
        1,
        PowderReadLimits::default(),
    )?;
    let selected = selected_pattern(&imported.pattern, 20.0, 125.5)?;
    let observed = selected
        .observed_y
        .as_deref()
        .ok_or(EchidnaValidationError::MissingObservations)?;
    let baseline = smooth_bruckner(observed, 20, 50)?;
    let (anchor_x, anchor_y) = regular_background_anchors(&selected.x_deg, &baseline, 20)?;
    let background = fitted_chebyshev_from_anchors(
        "echidna-background",
        [
            selected.x_deg[0],
            selected.x_deg[selected.sample_count() - 1],
        ],
        &anchor_x,
        &anchor_y,
        10,
    )?;
    let pattern = PatternRecord::new(
        selected.x_deg.clone(),
        Some(observed.to_vec()),
        selected.uncertainty.clone(),
        selected.mask.clone(),
        Some(
            background
                .calculate(&selected.x_deg)
                .map_err(LeBailError::Background)?,
        ),
    )?;
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: WAVELENGTH_ANGSTROM,
        u_deg2: 0.014,
        v_deg2: -0.253,
        w_deg2: 0.516,
        x_deg: 0.05,
        y_deg: 0.0,
    };
    let phase = lab6_phase(&pattern, instrument.wavelength_angstrom)?;
    let parameters = build_lebail_parameter_set(
        instrument,
        std::slice::from_ref(&phase),
        &["u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"],
        false,
        false,
    )?;
    let request =
        LeBailInput::new_with_parameters(pattern, instrument, vec![phase], parameters, Vec::new())?;
    let options = LeBailOptions::new(
        24,
        2,
        1.0e-7,
        1.0e-9,
        1.0,
        1.0e-15,
        1.0e-12,
        true,
        1.0 - 1.0e-10,
        false,
        8.0,
        ExecutionPolicy::new(Some(1), 2)?,
    )?
    .with_profile_controls(1.0e-10, 0.2, 8)?;
    let result = refine_lebail(&request, &options, None)?;
    let first_rwp = result
        .history
        .first()
        .ok_or(EchidnaValidationError::EmptyHistory)?
        .rwp;
    let final_rwp = result.metrics.rwp;
    let corrected_observed = observed
        .iter()
        .zip(&result.calculation.background_y)
        .map(|(observed, background)| observed - background)
        .collect::<Vec<_>>();
    let correlation = pearson_correlation(&corrected_observed, &result.calculation.profile_y)?;
    let uncertainty = selected
        .uncertainty
        .as_deref()
        .ok_or(EchidnaValidationError::MissingUncertainty)?;
    let manual_rwp = weighted_rwp(observed, &result.calculation.y, uncertainty)?;
    let reflection_positions = result
        .phases
        .first()
        .ok_or(EchidnaValidationError::MissingPhase)?
        .two_theta_deg();
    let reflection_coverage = reflection_positions.len() == 13
        && reflection_positions.iter().all(|position| {
            *position >= selected.x_deg[0]
                && *position <= selected.x_deg[selected.sample_count() - 1]
        });
    let finite_nonnegative = result
        .intensities
        .iter()
        .all(|item| item.integrated_intensity.is_finite() && item.integrated_intensity >= 0.0);
    let checks = vec![
        check(
            "observed_grid",
            selected.sample_count() >= 2_000
                && selected.uncertainty.as_ref().is_some_and(|values| {
                    values.iter().all(|value| value.is_finite() && *value > 0.0)
                }),
            "The deposited three-column pattern supplies a sorted real grid with positive uncertainties.",
            Some(count_as_f64(selected.sample_count())?),
            "at least 2000 selected samples with finite positive uncertainties",
        )?,
        check(
            "profile_improvement",
            final_rwp < first_rwp,
            "Native analytical profile refinement lowers Rwp relative to the first extraction cycle.",
            Some(final_rwp / first_rwp),
            "final Rwp / first-cycle Rwp < 1",
        )?,
        check(
            "profile_correlation",
            correlation >= 0.90,
            "Background-subtracted observed and calculated neutron profiles remain aligned.",
            Some(correlation),
            "Pearson correlation >= 0.90",
        )?,
        check(
            "integrated_intensities",
            finite_nonnegative,
            "Every extracted LaB6 integrated intensity is finite and nonnegative.",
            None,
            "all finite and >= 0",
        )?,
        check(
            "uncertainty_weighting",
            (manual_rwp - final_rwp).abs() <= 2.0e-15,
            "An independent direct sum reproduces the reported uncertainty-weighted Rwp.",
            Some((manual_rwp - final_rwp).abs()),
            "absolute Rwp recomputation difference <= 2e-15",
        )?,
        check(
            "reflection_coverage",
            reflection_coverage,
            "Every generated LaB6 reflection center lies inside the selected observed interval.",
            Some(count_as_f64(reflection_positions.len())?),
            "13 reflection centers inside 20-125.5 degrees",
        )?,
    ];
    RealDataValidationReport::new(
        DATASET_ID,
        selected.sample_count(),
        Some(result.intensities.len()),
        started.elapsed().as_secs_f64(),
        checks,
        vec![
            format!(
                "First-cycle Rwp={first_rwp:.8}; final Rwp={final_rwp:.8}; correlation={correlation:.8}."
            ),
            format!(
                "Termination={}; iterations={}.",
                result.termination_reason.as_str(),
                result.history.len()
            ),
            concat!(
                "The fixed background is a ten-term Chebyshev polynomial initialized from ",
                "20 regular anchors on a Smooth Bruckner estimate; the Smooth Bruckner array ",
                "is not part of the refinement model."
            )
            .to_owned(),
            concat!(
                "The Zenodo metadata labels 2.047 A as approximate; wavelength calibration and ",
                "instrument-specific detector-zero refinement are intentionally not claimed."
            )
            .to_owned(),
        ],
    )
    .map_err(Into::into)
}

fn lab6_phase(
    pattern: &PatternRecord,
    wavelength_angstrom: f64,
) -> Result<LeBailPhase, EchidnaValidationError> {
    let cell = UnitCell {
        a_angstrom: LAB6_LATTICE_ANGSTROM,
        b_angstrom: LAB6_LATTICE_ANGSTROM,
        c_angstrom: LAB6_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    };
    let reflections =
        PreparedReflectionGenerator::new(space_group_by_number(221)?.space_group, true, 1_000_000)?
            .generate(
                cell,
                ReflectionRange::CwTwoTheta {
                    min_deg: pattern.x_deg[0],
                    max_deg: *pattern
                        .x_deg
                        .last()
                        .ok_or(EchidnaValidationError::EmptySelectedPattern)?,
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
        "lab6",
        "NIST SRM 660c LaB6 calibration material",
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
) -> Result<PatternRecord, EchidnaValidationError> {
    let indices = source
        .x_deg
        .iter()
        .enumerate()
        .filter_map(|(index, x)| (*x >= minimum && *x <= maximum).then_some(index))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(EchidnaValidationError::EmptySelectedPattern);
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

fn pearson_correlation(left: &[f64], right: &[f64]) -> Result<f64, EchidnaValidationError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(EchidnaValidationError::InvalidCorrelation);
    }
    let count = count_as_f64(left.len())?;
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
    let value = numerator / (left_squared * right_squared).sqrt();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EchidnaValidationError::InvalidCorrelation)
    }
}

fn weighted_rwp(
    observed: &[f64],
    calculated: &[f64],
    uncertainty: &[f64],
) -> Result<f64, EchidnaValidationError> {
    if observed.len() != calculated.len() || observed.len() != uncertainty.len() {
        return Err(EchidnaValidationError::InvalidWeightedResidual);
    }
    let (numerator, denominator) = observed.iter().zip(calculated).zip(uncertainty).fold(
        (0.0, 0.0),
        |(numerator, denominator), ((observed, calculated), sigma)| {
            let residual = (calculated - observed) / sigma;
            let weighted_observed = observed / sigma;
            (
                numerator + residual * residual,
                denominator + weighted_observed * weighted_observed,
            )
        },
    );
    let value = (numerator / denominator).sqrt();
    value
        .is_finite()
        .then_some(value)
        .ok_or(EchidnaValidationError::InvalidWeightedResidual)
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

fn count_as_f64(count: usize) -> Result<f64, EchidnaValidationError> {
    u32::try_from(count)
        .map(f64::from)
        .map_err(|_| EchidnaValidationError::Arithmetic("sample count exceeds u32"))
}

/// Echidna validation setup, input, or numerical failure.
#[derive(Debug)]
pub enum EchidnaValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Powder input parsing failed.
    Powder(PowderIoError),
    /// Pattern construction failed.
    Pattern(DomainError),
    /// Legacy fixed-background preprocessing failed.
    Background(BackgroundError),
    /// Space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Reflection generation failed.
    Reflection(ReflectionGenerationError),
    /// Execution-policy construction failed.
    Execution(ExecutionPolicyError),
    /// Native Le Bail setup or refinement failed.
    LeBail(LeBailError),
    /// Report construction failed.
    Report(ValidationContractError),
    /// Observed data was missing.
    MissingObservations,
    /// Deposited uncertainties were missing.
    MissingUncertainty,
    /// Refined phase state was unexpectedly absent.
    MissingPhase,
    /// No samples remained in the supported validation interval.
    EmptySelectedPattern,
    /// The native solver returned no accepted history.
    EmptyHistory,
    /// Profile correlation was undefined.
    InvalidCorrelation,
    /// Direct uncertainty-weighted residual calculation was undefined.
    InvalidWeightedResidual,
    /// A bounded count could not be represented for a statistic.
    Arithmetic(&'static str),
}

impl Display for EchidnaValidationError {
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
            Self::MissingObservations => formatter.write_str("Echidna pattern has no observations"),
            Self::MissingUncertainty => formatter.write_str("Echidna pattern has no uncertainties"),
            Self::MissingPhase => formatter.write_str("Echidna refinement returned no phase"),
            Self::EmptySelectedPattern => formatter.write_str("Echidna interval has no samples"),
            Self::EmptyHistory => formatter.write_str("Echidna refinement returned no history"),
            Self::InvalidCorrelation => formatter.write_str("Echidna correlation is undefined"),
            Self::InvalidWeightedResidual => {
                formatter.write_str("Echidna weighted residual is undefined")
            }
            Self::Arithmetic(message) => formatter.write_str(message),
        }
    }
}

impl Error for EchidnaValidationError {}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for EchidnaValidationError {
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
