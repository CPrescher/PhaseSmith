//! POWGEN `LaB6` time-of-flight profile and Le Bail acceptance workflow.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::time::Instant;

use phasesmith_core::{
    BackgroundError, OwnedCwContributions, TofInstrumentParameter, TofProfileParameters,
    smooth_bruckner,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionGenerationError,
    ReflectionRange, UnitCell,
};
use phasesmith_engine::{BuiltInScatteringModel, StructuralPhaseDefinition};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    GsasTofInstrumentIoError, GsasTofInstrumentReadLimits, PowderIoError, PowderReadLimits,
    SpaceGroupLookupError, read_gsas_tof_instrument_file, read_tof_powder_file,
    space_group_by_number,
};
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};
use phasesmith_workflows::{
    LatticeBounds, LatticeError, LatticeParameterization, ParameterBounds, ParameterError,
    PreparedStructuralTofMultiBankObjective, RefinementLimits, ResidualOptions, RietveldError,
    RietveldPhase, RietveldStructuralSelection, RuntimeError, StructuralTofBank,
    StructuralTofMultiBankError, StructuralTofMultiBankInput,
    StructuralTofMultiBankRefinementError, StructuralTofMultiBankRefinementOptions,
    TofChebyshevBackground, TofInstrumentParameterBound, TofLeBailError, TofLeBailInput,
    TofLeBailOptions, TofLeBailPhase, TofMultiBankGeometryError, calculate_tof_lebail_pattern,
    evaluate_tof_residuals, refine_structural_tof_multibank, refine_tof_lebail,
};

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "powgen-lab6-tof-calibration";
const EXPECTED_BANK: usize = 2;
const EXPECTED_SAMPLES: usize = 6_824;
const LAB6_LATTICE_ANGSTROM: f64 = 4.156_826;
// Structural acceptance targets follow Huq et al., J. Appl. Cryst. 52 (2019)
// 1189-1201, DOI 10.1107/S160057671900833X. The isotropic displacement
// initializer is an explicit simplified model, not a transcription of GSAS-II.
const PUBLISHED_LAB6_LATTICE_ANGSTROM: f64 = 4.157_5;
const PUBLISHED_BORON_X: f64 = 0.199_6;
const INITIAL_BORON_X: f64 = 0.2;
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
    let pattern_source = fs::read_to_string(dataset_directory.join("PG3_17541.gsa"))?;
    let reduced_header = pattern_source.contains("Vanadium Run: 17534")
        && pattern_source.contains("with Y multiplied by the bin widths.")
        && pattern_source.contains("Normalised to pCharge");
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
        Some(background_seed.clone()),
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
    let structural = run_powgen_structural(
        &pattern.pattern,
        &calibration,
        background_seed,
        &reflections,
        reduced_header,
    )?;
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
        check(
            "tof_powgen_structural_contract",
            structural.incident_already_normalized
                && structural.uses_tof_lorentz
                && structural.history_counts.iter().all(|count| *count > 0),
            "The POWGEN header and published reduction define already-normalized observations; the one-bank structural request explicitly applies the named TOF-neutron Lorentz correction.",
            Some(count_as_f64(
                structural.history_counts.iter().sum::<usize>(),
            )?),
            "already normalized, explicit TOF Lorentz, and every staged solve accepts at least one step",
        )?,
        check(
            "tof_powgen_structural_fit",
            structural.rwp <= 0.16 && structural.profile_correlation >= 0.97,
            "The structural LaB6 intensities fit the checksum-pinned reduced POWGEN bank without Le Bail redistribution.",
            Some(structural.rwp),
            "uncertainty-weighted Rwp <= 0.16 and profile correlation >= 0.97",
        )?,
        check(
            "tof_powgen_structural_lattice",
            (structural.lattice_angstrom - PUBLISHED_LAB6_LATTICE_ANGSTROM).abs() <= 0.002,
            "The staged one-bank structural objective returns the published POWGEN SRM-660b cubic lattice.",
            Some(structural.lattice_angstrom),
            "|a - 4.1575 A| <= 0.002 A",
        )?,
        check(
            "tof_powgen_structural_boron_x",
            (structural.boron_x - PUBLISHED_BORON_X).abs() <= 0.001,
            "The symmetry-reduced boron coordinate remains consistent with the published POWGEN SRM-660b refinement.",
            Some(structural.boron_x),
            "|x(B) - 0.1996| <= 0.001",
        )?,
        check(
            "tof_powgen_structural_displacement",
            (0.0..=0.05).contains(&structural.lanthanum_u_iso)
                && (0.0..=0.05).contains(&structural.boron_u_iso),
            "Both refined isotropic displacement values remain inside an explicit physical validation interval.",
            Some(structural.lanthanum_u_iso.max(structural.boron_u_iso)),
            "0 <= Uiso(La), Uiso(B) <= 0.05 A^2",
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
            format!(
                "Structural POWGEN result: Rwp={:.8}; correlation={:.8}; a={:.8} A; x(B)={:.8}; Uiso(La)={:.8} A^2; Uiso(B)={:.8} A^2.",
                structural.rwp,
                structural.profile_correlation,
                structural.lattice_angstrom,
                structural.boron_x,
                structural.lanthanum_u_iso,
                structural.boron_u_iso,
            ),
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

#[derive(Clone, Debug)]
struct PowgenStructuralAcceptance {
    rwp: f64,
    profile_correlation: f64,
    lattice_angstrom: f64,
    boron_x: f64,
    lanthanum_u_iso: f64,
    boron_u_iso: f64,
    history_counts: [usize; 3],
    incident_already_normalized: bool,
    uses_tof_lorentz: bool,
}

#[allow(clippy::too_many_lines)]
fn run_powgen_structural(
    imported: &TofPatternRecord,
    calibration: &phasesmith_io::GsasTofInstrumentData,
    fixed_background: Vec<f64>,
    reflections: &[phasesmith_crystallography::GeneratedReflection],
    reduced_header: bool,
) -> Result<PowgenStructuralAcceptance, PowgenTofValidationError> {
    let geometry = calibration.bank_geometry.ok_or_else(|| {
        PowgenTofValidationError::InvalidData(
            "POWGEN structural validation requires bank geometry".to_owned(),
        )
    })?;
    if calibration.incident_spectrum.is_some() {
        return Err(PowgenTofValidationError::InvalidData(
            "POWGEN reduced bank unexpectedly carries an incident spectrum".to_owned(),
        ));
    }
    let pattern = TofPatternRecord::new(
        imported.tof_us.clone(),
        imported.observed_y.clone(),
        imported.uncertainty.clone(),
        imported.mask.clone(),
        Some(fixed_background),
    )?;
    let space_group = space_group_by_number(221)?.space_group;
    let definition = StructuralPhaseDefinition {
        cell: lab6_cell(),
        space_group: space_group.clone(),
        hkl: reflections
            .iter()
            .map(|reflection| reflection.hkl)
            .collect(),
        multiplicity: reflections
            .iter()
            .map(|reflection| reflection.multiplicity)
            .collect(),
        fractional_xyz: vec![[0.0, 0.0, 0.0], [INITIAL_BORON_X, 0.5, 0.5]],
        occupancy: vec![1.0, 1.0],
        u_iso_angstrom2: vec![0.004_8, 0.003],
        anisotropic_mask: vec![false, false],
        u_aniso_cif_angstrom2: vec![[0.0; 6]; 2],
        scattering_species: vec!["La".to_owned(), "B".to_owned()],
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-10,
        scattering_model: BuiltInScatteringModel::NeutronNuclear,
        correction_model: IntegratedIntensityCorrectionModel::Neutral,
    };
    let phase = RietveldPhase::new_with_site_ids(
        RecordId::new("lab6")?,
        "NIST SRM 660b LaB6 calibration material",
        vec![RecordId::new("lanthanum")?, RecordId::new("boron")?],
        definition,
        OwnedCwContributions::neutral(reflections.len()),
    )?;
    let mut input = StructuralTofMultiBankInput {
        phase,
        structural_selection: RietveldStructuralSelection::default(),
        lattice_bounds: None,
        banks: vec![StructuralTofBank {
            bank_id: RecordId::new("powgen-bank-2")?,
            pattern,
            instrument: calibration.instrument,
            geometry,
            correction_model: IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: geometry.two_theta_deg,
            },
            scale: 1.0,
            scale_bounds: ParameterBounds::default(),
            refine_scale: true,
            background: Some(TofChebyshevBackground::new(
                RecordId::new("powgen-structural-background")?,
                vec![0.0; 16],
                [
                    imported.tof_us[0],
                    imported.tof_us[imported.sample_count() - 1],
                ],
            )?),
            refine_background: true,
            instrument_bounds: vec![
                TofInstrumentParameterBound::new(
                    TofInstrumentParameter::Zero,
                    calibration.instrument.zero_us - 20.0,
                    calibration.instrument.zero_us + 20.0,
                )
                .map_err(TofMultiBankGeometryError::from)?,
            ],
        }],
        support_fwhm: 20.0,
        tail_log: 20.0,
        use_uncertainty: true,
        execution: ExecutionPolicy::new(Some(1), 2)?,
    };
    let unit = PreparedStructuralTofMultiBankObjective::new(input.clone())?.calculate()?;
    let bank = &input.banks[0];
    let observed = bank
        .pattern
        .observed_y
        .as_deref()
        .ok_or(PowgenTofValidationError::MissingObservations)?;
    let uncertainty = bank.pattern.uncertainty.as_deref();
    let mask = bank.pattern.mask.as_deref();
    let calculated = &unit.banks[0];
    let (numerator, denominator) = calculated.profile_y.iter().enumerate().fold(
        (0.0, 0.0),
        |(numerator, denominator), (sample, profile)| {
            if mask.is_none_or(|values| values[sample]) {
                let weight = uncertainty.map_or(1.0, |values| 1.0 / values[sample].powi(2));
                let target = observed[sample] - calculated.background_y[sample];
                (
                    numerator + weight * profile * target,
                    denominator + weight * profile * profile,
                )
            } else {
                (numerator, denominator)
            }
        },
    );
    let scale = (numerator / denominator).max(f64::EPSILON);
    if !scale.is_finite() {
        return Err(PowgenTofValidationError::InvalidData(
            "POWGEN structural scale estimate is invalid".to_owned(),
        ));
    }
    input.banks[0].scale = scale;
    input.banks[0].scale_bounds = ParameterBounds::new(0.01 * scale, 100.0 * scale)?;
    let parameterization = LatticeParameterization::new(space_group, lab6_cell())?;
    let lattice_bounds = LatticeBounds::around(&parameterization, 0.01, 1.0)?;
    let options = StructuralTofMultiBankRefinementOptions::new(
        RefinementLimits::new(20, 600, None, 10)?,
        1,
        1.0e-10,
        1.0e-8,
        1.0e-3,
        10.0,
        0.3,
        0.5,
        8,
    )?;
    input.structural_selection = RietveldStructuralSelection {
        lattice: true,
        ..RietveldStructuralSelection::default()
    };
    input.lattice_bounds = Some(lattice_bounds.clone());
    let geometry_result = refine_structural_tof_multibank(&input, options, None, None)?;
    input = geometry_result.input;
    input.structural_selection.coordinates = true;
    let coordinate_result = refine_structural_tof_multibank(&input, options, None, None)?;
    input = coordinate_result.input;
    input.structural_selection.u_iso = true;
    let displacement_result = refine_structural_tof_multibank(&input, options, None, None)?;
    let bank = &displacement_result.input.banks[0];
    let calculated = &displacement_result.calculation.banks[0];
    let profile_correlation = pearson_correlation(
        bank.pattern
            .observed_y
            .as_deref()
            .ok_or(PowgenTofValidationError::MissingObservations)?,
        &calculated.background_y,
        &calculated.profile_y,
        bank.pattern.mask.as_deref(),
    )?;
    let uses_tof_lorentz = matches!(
        bank.correction_model,
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
            if two_theta_deg.to_bits() == geometry.two_theta_deg.to_bits()
    );
    Ok(PowgenStructuralAcceptance {
        rwp: calculated.metrics.rwp,
        profile_correlation,
        lattice_angstrom: displacement_result.input.phase.definition().cell.a_angstrom,
        boron_x: displacement_result.input.phase.definition().fractional_xyz[1][0],
        lanthanum_u_iso: displacement_result.input.phase.definition().u_iso_angstrom2[0],
        boron_u_iso: displacement_result.input.phase.definition().u_iso_angstrom2[1],
        history_counts: [
            geometry_result.history.len(),
            coordinate_result.history.len(),
            displacement_result.history.len(),
        ],
        incident_already_normalized: reduced_header && calibration.incident_spectrum.is_none(),
        uses_tof_lorentz,
    })
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
    /// Structural TOF objective construction failed.
    Structural(StructuralTofMultiBankError),
    /// Structural TOF staged refinement failed.
    StructuralRefinement(StructuralTofMultiBankRefinementError),
    /// Rietveld phase or lattice construction failed.
    Rietveld(RietveldError),
    /// Physical parameter bounds failed.
    Parameter(ParameterError),
    /// TOF geometry-bound construction failed.
    Geometry(TofMultiBankGeometryError),
    /// Lattice parameterization failed.
    Lattice(LatticeError),
    /// Refinement-limit construction failed.
    Runtime(RuntimeError),
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
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::StructuralRefinement(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Geometry(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
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

impl From<StructuralTofMultiBankError> for PowgenTofValidationError {
    fn from(value: StructuralTofMultiBankError) -> Self {
        Self::Structural(value)
    }
}

impl From<StructuralTofMultiBankRefinementError> for PowgenTofValidationError {
    fn from(value: StructuralTofMultiBankRefinementError) -> Self {
        Self::StructuralRefinement(value)
    }
}

impl From<RietveldError> for PowgenTofValidationError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}

impl From<ParameterError> for PowgenTofValidationError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}

impl From<TofMultiBankGeometryError> for PowgenTofValidationError {
    fn from(value: TofMultiBankGeometryError) -> Self {
        Self::Geometry(value)
    }
}

impl From<LatticeError> for PowgenTofValidationError {
    fn from(value: LatticeError) -> Self {
        Self::Lattice(value)
    }
}

impl From<RuntimeError> for PowgenTofValidationError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
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
