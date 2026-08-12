//! Non-POWGEN TOF transferability gate using the classic LANL nickel example.

#![allow(clippy::cast_precision_loss)]

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

use phasesmith_core::{BackgroundError, smooth_bruckner};
use phasesmith_crystallography::{
    PreparedReflectionGenerator, ReflectionGenerationError, ReflectionRange, UnitCell,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    GsasTofInstrumentIoError, GsasTofInstrumentReadLimits, PowderIoError, PowderReadLimits,
    SpaceGroupLookupError, TofPowderFormat, read_gsas_tof_instrument_file, read_tof_powder_file,
    space_group_by_number,
};
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};
use phasesmith_workflows::{
    TofChebyshevBackground, TofLeBailError, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    refine_tof_lebail,
};

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "lanl-nickel-tof";
const BANK: usize = 2;
const FIT_MIN_US: f64 = 1_101.6;
const FIT_MAX_US: f64 = 8_189.6;
const NICKEL_LATTICE_ANGSTROM: f64 = 3.523_4;

/// Run the complete native fixed-instrument TOF Le Bail transferability gate.
///
/// This intentionally uses the LANL/GSAS packed constant-step format and legacy
/// profile function 1, rather than POWGEN's SLOG FXYE/profile-function-3 pair.
///
/// # Errors
///
/// Returns [`NickelTofValidationError`] for corrupt data, unsupported records,
/// invalid numerical state, or report construction failures.
#[allow(clippy::too_many_lines)]
pub fn run_nickel_tof_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, NickelTofValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let imported = read_tof_powder_file(
        dataset_directory.join("nickel.raw"),
        BANK,
        PowderReadLimits::default(),
    )?;
    let calibration = read_gsas_tof_instrument_file(
        dataset_directory.join("inst_tof.prm"),
        BANK,
        GsasTofInstrumentReadLimits::default(),
    )?;
    let instrument = calibration.instrument;
    let start = imported
        .pattern
        .tof_us
        .partition_point(|value| *value < FIT_MIN_US);
    let end = imported
        .pattern
        .tof_us
        .partition_point(|value| *value <= FIT_MAX_US);
    if start >= end {
        return Err(NickelTofValidationError::InvalidData(
            "LANL nickel fit interval is empty".to_owned(),
        ));
    }
    let tof_us = imported.pattern.tof_us[start..end].to_vec();
    let observed = imported
        .pattern
        .observed_y
        .as_deref()
        .ok_or(NickelTofValidationError::MissingObservations)?[start..end]
        .to_vec();
    let uncertainty = imported
        .pattern
        .uncertainty
        .as_deref()
        .map(|values| values[start..end].to_vec());
    let mask = imported
        .pattern
        .mask
        .as_deref()
        .map(|values| values[start..end].to_vec());
    let fixed_background = smooth_bruckner(&observed, 20, 50)?;
    let pattern = TofPatternRecord::new(
        tof_us,
        Some(observed.clone()),
        uncertainty,
        mask,
        Some(fixed_background),
    )?;
    let reflections =
        PreparedReflectionGenerator::new(space_group_by_number(225)?.space_group, true, 100_000)?
            .generate(
            nickel_cell(),
            ReflectionRange::Tof {
                min_us: pattern.tof_us[0],
                max_us: pattern.tof_us[pattern.sample_count() - 1],
                search_min_d_angstrom: 0.2,
                search_max_d_angstrom: 3.0,
                zero_us: instrument.zero_us,
                difc_us_per_angstrom: instrument.difc_us_per_angstrom,
                difa_us_per_angstrom2: instrument.difa_us_per_angstrom2,
                difb_us_angstrom: instrument.difb_us_angstrom,
            },
        )?;
    let phase = TofLeBailPhase::new(
        RecordId::new("nickel")?,
        "FCC nickel powder standard",
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
        RecordId::new("nickel-chebyshev-background")?,
        vec![0.0; 12],
        [
            pattern.tof_us[0],
            pattern.tof_us[pattern.sample_count() - 1],
        ],
    )?;
    let request = TofLeBailInput::new(pattern, instrument, vec![phase])?
        .with_refinable_background(background)?;
    let options = TofLeBailOptions::new(
        20,
        1.0,
        1.0e-12,
        1.0e-15,
        20.0,
        20.0,
        true,
        ExecutionPolicy::new(Some(1), 2)?,
    )?
    .with_redistribution_uncertainty(false);
    let result = refine_tof_lebail(&request, &options)?;
    let correlation = pearson_correlation(
        &observed,
        &result.calculation.background_y,
        &result.calculation.profile_y,
        request.pattern.mask.as_deref(),
    )?;
    let monotonic = result
        .history
        .windows(2)
        .all(|pair| pair[1].metrics.rwp <= pair[0].metrics.rwp);
    let checks = vec![
        check(
            "tof_non_powgen_format",
            imported.format == TofPowderFormat::GsasConstStd
                && imported.bank == Some(BANK)
                && imported.pattern.sample_count() == 5_120,
            "The native adapter resolves the packed constant-step microsecond convention independently of POWGEN SLOG FXYE.",
            Some(imported.pattern.sample_count() as f64),
            "bank 2, gsas_const_std, 5120 imported bins",
        )?,
        check(
            "tof_profile_function_one",
            calibration.profile_function == 1
                && instrument.difc_us_per_angstrom.to_bits() == 4_368.97_f64.to_bits()
                && instrument.alpha_coefficient.to_bits() == 0.142_760_f64.to_bits(),
            "Legacy GSAS profile function 1 is translated into the common typed 15-coefficient model.",
            Some(calibration.profile_function as f64),
            "profile function 1 with DIFC=4368.97 us/A and alpha=0.142760",
        )?,
        check(
            "tof_nickel_fit_window",
            request.pattern.sample_count() == 4_431
                && (request.pattern.tof_us[0] - FIT_MIN_US).abs() <= 1.0e-9
                && (request.pattern.tof_us[request.pattern.sample_count() - 1] - FIT_MAX_US).abs()
                    <= 1.0e-9,
            "The published tutorial fit interval uses bin centers in microseconds.",
            Some(request.pattern.sample_count() as f64),
            "4431 centers from 1101.6 through 8189.6 us",
        )?,
        check(
            "tof_nickel_reflections",
            reflections.len() == 102,
            "The Fm-3m nickel cell generates stable reflection families across the selected bank.",
            Some(reflections.len() as f64),
            "102 reflection families",
        )?,
        check(
            "tof_nickel_profile_fit",
            result.metrics.rwp <= 0.026,
            "The native fixed-instrument TOF extraction fits the non-POWGEN nickel observations.",
            Some(result.metrics.rwp),
            "uncertainty-weighted Rwp <= 0.026",
        )?,
        check(
            "tof_nickel_profile_correlation",
            correlation >= 0.998,
            "The background-subtracted observations and extracted profile remain aligned.",
            Some(correlation),
            "masked Pearson correlation >= 0.998",
        )?,
        check(
            "tof_nickel_convergence",
            monotonic && result.history.len() == 20,
            "Every accepted nonnegative redistribution cycle is non-increasing in Rwp.",
            Some(result.history.len() as f64),
            "20 cycles with non-increasing Rwp",
        )?,
        check(
            "tof_nickel_derivatives",
            result
                .calculation
                .accumulation
                .derivatives
                .global
                .as_ref()
                .is_some_and(|jacobian| jacobian.parameter_count == 15),
            "The real-pattern calculation retains all analytical instrument rows from the fused pass.",
            Some(15.0),
            "15 instrument derivative rows",
        )?,
    ];
    RealDataValidationReport::new(
        DATASET_ID,
        request.pattern.sample_count(),
        Some(reflections.len()),
        started.elapsed().as_secs_f64(),
        checks,
        vec![
            format!(
                "LANL bank 2 profile function 1; range={FIT_MIN_US:.1}..{FIT_MAX_US:.1} us; DIFC={:.5} us/A.",
                instrument.difc_us_per_angstrom,
            ),
            format!(
                "Final Rwp={:.8}; Rp={:.8}; correlation={correlation:.8}; background=Smooth Bruckner + 12-term Chebyshev.",
                result.metrics.rwp, result.metrics.rp,
            ),
            "This is a fixed-instrument Le Bail extraction, not structural TOF Rietveld parameter refinement."
                .to_owned(),
        ],
    )
    .map_err(Into::into)
}

fn nickel_cell() -> UnitCell {
    UnitCell {
        a_angstrom: NICKEL_LATTICE_ANGSTROM,
        b_angstrom: NICKEL_LATTICE_ANGSTROM,
        c_angstrom: NICKEL_LATTICE_ANGSTROM,
        alpha_deg: 90.0,
        beta_deg: 90.0,
        gamma_deg: 90.0,
    }
}

fn pearson_correlation(
    observed: &[f64],
    background: &[f64],
    profile: &[f64],
    mask: Option<&[bool]>,
) -> Result<f64, NickelTofValidationError> {
    if observed.len() != background.len() || observed.len() != profile.len() {
        return Err(NickelTofValidationError::InvalidData(
            "nickel correlation arrays have different lengths".to_owned(),
        ));
    }
    let selected = (0..observed.len())
        .filter(|index| mask.is_none_or(|values| values[*index]))
        .map(|index| (observed[index] - background[index], profile[index]))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(NickelTofValidationError::InvalidData(
            "nickel correlation selection is empty".to_owned(),
        ));
    }
    let count = selected.len() as f64;
    let left_mean = selected.iter().map(|item| item.0).sum::<f64>() / count;
    let right_mean = selected.iter().map(|item| item.1).sum::<f64>() / count;
    let (cross, left_square, right_square) = selected.iter().fold(
        (0.0, 0.0, 0.0),
        |(cross, left_square, right_square), (left, right)| {
            let left = left - left_mean;
            let right = right - right_mean;
            (
                cross + left * right,
                left_square + left * left,
                right_square + right * right,
            )
        },
    );
    let value = cross / (left_square * right_square).sqrt();
    value.is_finite().then_some(value).ok_or_else(|| {
        NickelTofValidationError::InvalidData("invalid nickel profile correlation".to_owned())
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

/// LANL nickel TOF setup, input, numerical, or report failure.
#[derive(Debug)]
pub enum NickelTofValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Typed TOF/domain validation failed.
    Domain(DomainError),
    /// Powder text import failed.
    Powder(PowderIoError),
    /// Instrument calibration import failed.
    Instrument(GsasTofInstrumentIoError),
    /// Space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Reflection generation failed.
    Reflection(ReflectionGenerationError),
    /// Fixed background estimation failed.
    Background(BackgroundError),
    /// Native TOF workflow failed.
    Workflow(TofLeBailError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// The imported bank contains no observations.
    MissingObservations,
    /// Pinned records or returned arrays violate the gate contract.
    InvalidData(String),
    /// Stable report construction failed.
    Report(ValidationContractError),
}

impl Display for NickelTofValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Domain(error) => Display::fmt(error, formatter),
            Self::Powder(error) => Display::fmt(error, formatter),
            Self::Instrument(error) => Display::fmt(error, formatter),
            Self::SpaceGroup(error) => Display::fmt(error, formatter),
            Self::Reflection(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::Workflow(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("LANL nickel observations are missing")
            }
            Self::InvalidData(message) => formatter.write_str(message),
            Self::Report(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for NickelTofValidationError {}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for NickelTofValidationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(DatasetVerificationError, Dataset);
from_error!(DomainError, Domain);
from_error!(PowderIoError, Powder);
from_error!(GsasTofInstrumentIoError, Instrument);
from_error!(SpaceGroupLookupError, SpaceGroup);
from_error!(ReflectionGenerationError, Reflection);
from_error!(BackgroundError, Background);
from_error!(TofLeBailError, Workflow);
from_error!(ExecutionPolicyError, Execution);
from_error!(ValidationContractError, Report);
