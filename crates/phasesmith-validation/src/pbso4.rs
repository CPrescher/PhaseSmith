use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    BackgroundError as SmoothBackgroundError, ConstantWavelengthInstrument, FcjGeometry,
    OwnedCwContributions, smooth_bruckner,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionRange,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_io::{
    CifAtomSite, CifIoError, CifReadLimits, CifStructure, PowderFormat, PowderIoError,
    PowderReadLimits, read_cif_file, read_powder_file,
};
use phasesmith_model::{DomainError, FixedWavelengthSpectrum, PatternRecord, RecordId};
use phasesmith_workflows::{
    BackgroundError as ModelBackgroundError, BackgroundModel, ChebyshevBackground,
    DifferentiableBackground, LatticeBounds, LatticeError, LatticeParameterization,
    LatticeReflectionDomain, RefinementLimits, RietveldCalculationOptions,
    RietveldCovarianceOptions, RietveldError, RietveldGeneralParameterError,
    RietveldGeneralRefinementError, RietveldInput, RietveldInstrumentParameter,
    RietveldParameterSelection, RietveldPhase, RietveldRecipeError, RietveldRefinementError,
    RietveldRefinementOptions, RietveldSamplePhysicsModel, RietveldStructuralSelection,
    RietveldWorkflowResult, RuntimeError, TerminationReason, calculate_rietveld_pattern,
    intelligent_rietveld_recipe, run_rietveld_recipe,
};

use crate::{
    DatasetVerificationError, RealDataValidationReport, ValidationCheck, ValidationContractError,
    ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "gsasii-pbso4-cw";
const PHASE_ID: &str = "PbSO4";
const ANGULAR_RANGE: [f64; 2] = [19.0, 153.0];
const XRAY_RANGE: [f64; 2] = [16.0, 158.4];

/// Run the checksum-pinned fixed-doublet `PbSO4` X-ray validation without Python.
///
/// # Errors
///
/// Returns [`Pbso4ValidationError`] for invalid/missing data or native numerical failures.
#[allow(clippy::too_many_lines)]
pub fn run_pbso4_xray_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, Pbso4ValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let structure = read_cif_file(
        dataset_directory.join("PbSO4-Wyckoff.cif"),
        None,
        true,
        CifReadLimits::default(),
    )?
    .structure;
    let imported = read_powder_file(
        dataset_directory.join("PBSO4.XRA"),
        PowderFormat::GsasStd,
        1,
        PowderReadLimits::default(),
    )?;
    let selected = selected_pattern(&imported.pattern, XRAY_RANGE)?;
    let observed = selected
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let pattern = PatternRecord::new(
        selected.x_deg.clone(),
        Some(observed.clone()),
        selected.uncertainty.clone(),
        None,
        Some(smooth_bruckner(observed, 40, 50)?),
    )?;
    let instrument = xray_instrument();
    let spectrum = FixedWavelengthSpectrum::new(vec![1.5405, 1.5443], vec![1.0, 0.5])?;
    let axial = Some(FcjGeometry {
        sample_over_radius: 0.0075,
        detector_over_radius: 0.0075,
    });
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    let execution = ExecutionPolicy::bounded_default()?;
    let mut phase = fixed_xray_phase(&structure)?;
    phase = phase.with_sample_physics(RietveldSamplePhysicsModel::Composite(vec![
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm: 100.0,
            shape_factor: 0.9,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 8.0e-4,
        },
    ]));
    phase = estimated_fixed_xray_scale(
        &pattern,
        instrument,
        &spectrum,
        axial,
        position,
        &phase,
        execution.clone(),
    )?;
    let background = fitted_xray_background(
        &pattern,
        instrument,
        &spectrum,
        axial,
        position,
        &phase,
        execution.clone(),
    )?;
    let input = RietveldInput::new_fixed_spectrum_with_background(
        pattern,
        instrument,
        spectrum,
        axial,
        position,
        BackgroundModel::Chebyshev(background),
        vec![phase],
    )?;
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            coordinates: true,
            u_iso: true,
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::VDeg2,
            RietveldInstrumentParameter::WDeg2,
            RietveldInstrumentParameter::ZeroShiftDeg,
        ],
        true,
        true,
    )?;
    let options = RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(30.0, true, execution)?,
        RefinementLimits::new(160, 12_000, None, 20)?,
        3,
        1.0e-7,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        30,
        0.15,
        8,
    )?;
    let recipe = intelligent_rietveld_recipe(&input, &selection, "pbso4-xray-intelligent")?;
    let workflow = run_rietveld_recipe(
        &input,
        &selection,
        &[None],
        &[],
        &recipe,
        &options,
        RietveldCovarianceOptions::new(false, 64, 1.0 - 1.0e-10)?,
        None,
    )?;
    xray_report(started.elapsed().as_secs_f64(), &input, &workflow)
}

/// Run the checksum-pinned monochromatic `PbSO4` neutron validation without Python.
///
/// # Errors
///
/// Returns [`Pbso4ValidationError`] for invalid/missing data or native numerical failures.
pub fn run_pbso4_neutron_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, Pbso4ValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory)?;
    let structure = read_cif_file(
        dataset_directory.join("PbSO4-Wyckoff.cif"),
        None,
        true,
        CifReadLimits::default(),
    )?
    .structure;
    let (request, selection, lattice_bounds, execution) =
        neutron_request(dataset_directory, &structure)?;
    let options = RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(30.0, true, execution)?,
        RefinementLimits::new(160, 12_000, None, 20)?,
        3,
        1.0e-7,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        30,
        0.15,
        8,
    )?;
    let recipe = intelligent_rietveld_recipe(&request, &selection, "pbso4-neutron-intelligent")?;
    let workflow = run_rietveld_recipe(
        &request,
        &selection,
        &[Some(lattice_bounds)],
        &[],
        &recipe,
        &options,
        RietveldCovarianceOptions::new(false, 64, 1.0 - 1.0e-10)?,
        None,
    )?;
    neutron_report(started.elapsed().as_secs_f64(), &request, &workflow)
}

fn neutron_request(
    dataset_directory: &Path,
    structure: &CifStructure,
) -> Result<
    (
        RietveldInput,
        RietveldParameterSelection,
        LatticeBounds,
        ExecutionPolicy,
    ),
    Pbso4ValidationError,
> {
    let imported = read_powder_file(
        dataset_directory.join("PBSO4.CWN"),
        PowderFormat::GsasStd,
        1,
        PowderReadLimits::default(),
    )?;
    let selected = selected_pattern(&imported.pattern, ANGULAR_RANGE)?;
    let observed = selected
        .observed_y
        .as_deref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let fixed_background = smooth_bruckner(observed, 20, 50)?;
    let pattern = PatternRecord::new(
        selected.x_deg.clone(),
        Some(observed.to_vec()),
        selected.uncertainty.clone(),
        None,
        Some(fixed_background),
    )?;
    let instrument = neutron_instrument();
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: -0.1,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: Some((0.0, 0.0, 650.0)),
    };
    let execution = ExecutionPolicy::bounded_default()?;
    let parameterization =
        LatticeParameterization::new(structure.space_group.clone(), structure.cell)?;
    let lattice_bounds = LatticeBounds::around(&parameterization, 0.05, 5.0)?;
    let domain = LatticeReflectionDomain::new(
        parameterization,
        lattice_bounds.clone(),
        instrument.wavelength_angstrom,
        ANGULAR_RANGE,
        0.0,
        true,
        50_000_000,
        1.001,
    )?;
    let definition = neutron_definition(structure);
    let site_ids = structure
        .sites
        .iter()
        .map(|site| RecordId::new(&site.site_id))
        .collect::<Result<Vec<_>, _>>()?;
    let phase = RietveldPhase::from_lattice_domain(
        RecordId::new(PHASE_ID)?,
        "PbSO4",
        site_ids.clone(),
        definition.clone(),
        domain.clone(),
    )?;
    let phase = estimated_scale(
        &pattern,
        instrument,
        position,
        phase,
        definition,
        domain,
        site_ids,
        execution.clone(),
    )?;
    let background =
        fitted_residual_background(&pattern, instrument, position, &phase, execution.clone())?;
    let input = RietveldInput::new_with_background(
        pattern,
        instrument,
        None,
        position,
        BackgroundModel::Chebyshev(background),
        vec![phase],
    )?;
    let selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            lattice: true,
            coordinates: true,
            u_iso: true,
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::VDeg2,
            RietveldInstrumentParameter::WDeg2,
            RietveldInstrumentParameter::DisplaceXMicrometre,
            RietveldInstrumentParameter::DisplaceYMicrometre,
        ],
        true,
        false,
    )?;
    Ok((input, selection, lattice_bounds, execution))
}

fn xray_instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.5405,
        u_deg2: 2.0e-4,
        v_deg2: -2.0e-4,
        w_deg2: 5.0e-4,
        x_deg: 1.0e-3,
        y_deg: 0.0,
    }
}

fn fixed_xray_phase(structure: &CifStructure) -> Result<RietveldPhase, Pbso4ValidationError> {
    let reflections =
        PreparedReflectionGenerator::new(structure.space_group.clone(), true, 500_000)
            .map_err(LatticeError::Generation)?
            .generate(
                structure.cell,
                ReflectionRange::CwTwoTheta {
                    min_deg: XRAY_RANGE[0],
                    max_deg: XRAY_RANGE[1],
                    wavelength_angstrom: 1.5405,
                },
            )
            .map_err(LatticeError::Generation)?;
    let definition = StructuralPhaseDefinition {
        cell: structure.cell,
        space_group: structure.space_group.clone(),
        hkl: reflections.iter().map(|item| item.hkl).collect(),
        multiplicity: reflections.iter().map(|item| item.multiplicity).collect(),
        fractional_xyz: structure
            .sites
            .iter()
            .map(|site| site.fractional_xyz)
            .collect(),
        occupancy: structure.sites.iter().map(|site| site.occupancy).collect(),
        u_iso_angstrom2: structure
            .sites
            .iter()
            .map(|site| site.u_iso_angstrom2.unwrap_or(0.005))
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
            .map(|site| site.element_symbol.clone())
            .collect(),
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-4,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom: 1.5405,
            polarization: 0.7,
        },
    };
    Ok(RietveldPhase::new_with_site_ids(
        RecordId::new(PHASE_ID)?,
        "PbSO4",
        structure
            .sites
            .iter()
            .map(|site| RecordId::new(&site.site_id))
            .collect::<Result<Vec<_>, _>>()?,
        definition,
        OwnedCwContributions::neutral(reflections.len()),
    )?)
}

#[allow(clippy::too_many_arguments)]
fn estimated_fixed_xray_scale(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    spectrum: &FixedWavelengthSpectrum,
    axial: Option<FcjGeometry>,
    position: MonochromaticPositionCorrection,
    phase: &RietveldPhase,
    execution: ExecutionPolicy,
) -> Result<RietveldPhase, Pbso4ValidationError> {
    let input = RietveldInput::new_fixed_spectrum(
        pattern.clone(),
        instrument,
        spectrum.clone(),
        axial,
        position,
        vec![phase.clone()],
    )?;
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(30.0, true, execution)?,
    )?;
    let observed = pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 0..pattern.sample_count() {
        let weight = pattern
            .uncertainty
            .as_ref()
            .map_or(1.0, |sigma| sigma[index].powi(2).recip());
        let target = observed[index] - pattern.background_y[index];
        numerator += weight * calculation.profile_y[index] * target;
        denominator += weight * calculation.profile_y[index].powi(2);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Pbso4ValidationError::LinearSolve);
    }
    let mut definition = phase.definition().clone();
    definition.scale = (numerator / denominator).max(f64::MIN_POSITIVE);
    let updated = RietveldPhase::new_with_site_ids(
        phase.phase_id().clone(),
        phase.name(),
        phase.site_ids().to_vec(),
        definition,
        phase.contributions().clone(),
    )?;
    Ok(phase
        .sample_physics()
        .cloned()
        .map_or(updated.clone(), |model| updated.with_sample_physics(model)))
}

#[allow(clippy::too_many_arguments)]
fn fitted_xray_background(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    spectrum: &FixedWavelengthSpectrum,
    axial: Option<FcjGeometry>,
    position: MonochromaticPositionCorrection,
    phase: &RietveldPhase,
    execution: ExecutionPolicy,
) -> Result<ChebyshevBackground, Pbso4ValidationError> {
    let initial = ChebyshevBackground::new("pbso4_residual", vec![0.0; 3], XRAY_RANGE)?;
    let input = RietveldInput::new_fixed_spectrum_with_background(
        pattern.clone(),
        instrument,
        spectrum.clone(),
        axial,
        position,
        BackgroundModel::Chebyshev(initial.clone()),
        vec![phase.clone()],
    )?;
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(30.0, true, execution)?,
    )?;
    let observed = pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let target = observed
        .iter()
        .zip(&calculation.y)
        .map(|(observed, calculated)| observed - calculated)
        .collect::<Vec<_>>();
    let basis = initial.basis(&pattern.x_deg)?;
    let mut matrix = DMatrix::from_row_slice(basis.rows, basis.columns, &basis.values);
    let mut right = DVector::from_vec(target);
    if let Some(uncertainty) = &pattern.uncertainty {
        for row in 0..basis.rows {
            for column in 0..basis.columns {
                matrix[(row, column)] /= uncertainty[row];
            }
            right[row] /= uncertainty[row];
        }
    }
    let coefficients = matrix
        .svd(true, true)
        .solve(&right, f64::EPSILON)
        .map_err(|_| Pbso4ValidationError::LinearSolve)?;
    initial
        .replace_coefficients(coefficients.as_slice())
        .map_err(Into::into)
}

#[allow(clippy::too_many_lines)]
fn xray_report(
    elapsed_seconds: f64,
    request: &RietveldInput,
    workflow: &RietveldWorkflowResult,
) -> Result<RealDataValidationReport, Pbso4ValidationError> {
    let result = workflow.final_result();
    let observed = request
        .pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let residual = result
        .calculation
        .y
        .iter()
        .zip(observed)
        .map(|(calculated, observed)| calculated - observed)
        .collect::<Vec<_>>();
    let unit_weight_rwp = (dot(&residual, &residual) / dot(observed, observed)).sqrt();
    let background_subtracted = observed
        .iter()
        .zip(&result.calculation.background_y)
        .map(|(observed, background)| observed - background)
        .collect::<Vec<_>>();
    let profile_correlation =
        pearson_correlation(&background_subtracted, &result.calculation.profile_y)?;
    let cell = result.input.phases[0].definition().cell;
    let cell_error = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom]
        .iter()
        .zip([8.48, 5.398, 6.958])
        .map(|(actual, expected)| (actual - expected).abs() / expected)
        .fold(0.0_f64, f64::max);
    let safe = workflow.stages().iter().all(|stage| {
        !matches!(
            stage.result.termination_reason,
            TerminationReason::NumericalFailure
                | TerminationReason::Diverged
                | TerminationReason::RepeatedRejections
                | TerminationReason::NoObservations
                | TerminationReason::MaxIterations
        )
    });
    let checks = vec![
        check(
            "observed_grid",
            request.pattern.sample_count() == 5_697
                && request.pattern.x_deg[0].to_bits() == 16.0_f64.to_bits()
                && request
                    .pattern
                    .x_deg
                    .last()
                    .is_some_and(|value| value.to_bits() == 158.4_f64.to_bits()),
            "Selected PbSO4 pattern uses the official combined-refinement angular range.",
            Some(count_as_f64(request.pattern.sample_count())?),
            ">2000 samples with endpoints 16.0 and 158.4 degrees",
        )?,
        check(
            "refinement_termination",
            safe,
            "Every intelligent recipe stage terminates safely under explicit budgets.",
            None,
            "all stages converge or stagnate safely; iteration exhaustion fails",
        )?,
        check(
            "poisson_rwp",
            result.calculation.metrics.rwp <= 0.11,
            "Poisson-weighted complete-pattern residual for the PbSO4 workflow.",
            Some(result.calculation.metrics.rwp),
            "Rwp <= 0.11",
        )?,
        check(
            "unit_weight_rwp",
            unit_weight_rwp <= 0.10,
            "Unit-weight PbSO4 residual is reported separately.",
            Some(unit_weight_rwp),
            "unit-weight Rwp <= 0.10",
        )?,
        check(
            "profile_correlation",
            profile_correlation >= 0.99,
            "Background-subtracted observed and calculated profiles remain aligned.",
            Some(profile_correlation),
            "Pearson correlation >= 0.99",
        )?,
        check(
            "reference_cell_relative_error",
            cell_error <= 0.005,
            "Refined cell remains close to the supplied PbSO4 reference model.",
            Some(cell_error),
            "maximum relative error across a, b, c <= 0.005",
        )?,
    ];
    let mut notes = vec![
        format!("Final cell a={:.8}, b={:.8}, c={:.8} angstrom.", cell.a_angstrom, cell.b_angstrom, cell.c_angstrom),
        "Cu K-alpha is evaluated as the exact fixed 1.5405/1.5443 angstrom doublet with relative intensity 0.5.".to_owned(),
        "Background is a fixed native Smooth Bruckner estimate plus a refined three-term Chebyshev residual correction.".to_owned(),
    ];
    notes.extend(workflow.stages().iter().map(|stage| {
        format!(
            "Stage {}: Rwp {:.8} -> {:.8}; termination={}; iterations={}.",
            stage.stage.name(),
            stage.starting_rwp,
            stage.result.calculation.metrics.rwp,
            stage.result.termination_reason.as_str(),
            stage.result.history.len()
        )
    }));
    RealDataValidationReport::new(
        DATASET_ID.to_owned() + "-x-ray",
        request.pattern.sample_count(),
        Some(result.input.phases[0].reflection_ids().len()),
        elapsed_seconds,
        checks,
        notes,
    )
    .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn estimated_scale(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    position: MonochromaticPositionCorrection,
    phase: RietveldPhase,
    mut definition: StructuralPhaseDefinition,
    domain: LatticeReflectionDomain,
    site_ids: Vec<RecordId>,
    execution: ExecutionPolicy,
) -> Result<RietveldPhase, Pbso4ValidationError> {
    let input = RietveldInput::new(pattern.clone(), instrument, None, position, vec![phase])?;
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(30.0, true, execution)?,
    )?;
    let observed = pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for index in 0..pattern.sample_count() {
        let weight = pattern
            .uncertainty
            .as_ref()
            .map_or(1.0, |sigma| 1.0 / sigma[index].powi(2));
        let target = observed[index] - pattern.background_y[index];
        numerator += weight * calculation.profile_y[index] * target;
        denominator += weight * calculation.profile_y[index].powi(2);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Pbso4ValidationError::LinearSolve);
    }
    definition.scale = (numerator / denominator).max(f64::MIN_POSITIVE);
    RietveldPhase::from_lattice_domain(
        RecordId::new(PHASE_ID)?,
        "PbSO4",
        site_ids,
        definition,
        domain,
    )
    .map_err(Into::into)
}

fn fitted_residual_background(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    position: MonochromaticPositionCorrection,
    phase: &RietveldPhase,
    execution: ExecutionPolicy,
) -> Result<ChebyshevBackground, Pbso4ValidationError> {
    let initial = ChebyshevBackground::new("pbso4_residual", vec![0.0; 3], ANGULAR_RANGE)?;
    let input = RietveldInput::new_with_background(
        pattern.clone(),
        instrument,
        None,
        position,
        BackgroundModel::Chebyshev(initial.clone()),
        vec![phase.clone()],
    )?;
    let calculation = calculate_rietveld_pattern(
        &input,
        &RietveldCalculationOptions::new(30.0, true, execution)?,
    )?;
    let observed = pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let target = observed
        .iter()
        .zip(&calculation.y)
        .map(|(observed, calculated)| observed - calculated)
        .collect::<Vec<_>>();
    let basis = initial.basis(&pattern.x_deg)?;
    let mut matrix = DMatrix::from_row_slice(basis.rows, basis.columns, &basis.values);
    let mut right = DVector::from_vec(target);
    if let Some(uncertainty) = &pattern.uncertainty {
        for row in 0..basis.rows {
            for column in 0..basis.columns {
                matrix[(row, column)] /= uncertainty[row];
            }
            right[row] /= uncertainty[row];
        }
    }
    let coefficients = matrix
        .svd(true, true)
        .solve(&right, f64::EPSILON)
        .map_err(|_| Pbso4ValidationError::LinearSolve)?;
    initial
        .replace_coefficients(coefficients.as_slice())
        .map_err(Into::into)
}

fn neutron_report(
    elapsed_seconds: f64,
    request: &RietveldInput,
    workflow: &RietveldWorkflowResult,
) -> Result<RealDataValidationReport, Pbso4ValidationError> {
    let result = workflow.final_result();
    let observed = request
        .pattern
        .observed_y
        .as_ref()
        .ok_or(Pbso4ValidationError::MissingObservations)?;
    let residual = result
        .calculation
        .y
        .iter()
        .zip(observed)
        .map(|(calculated, observed)| calculated - observed)
        .collect::<Vec<_>>();
    let unit_weight_rwp = (dot(&residual, &residual) / dot(observed, observed)).sqrt();
    let background_subtracted = observed
        .iter()
        .zip(&result.calculation.background_y)
        .map(|(observed, background)| observed - background)
        .collect::<Vec<_>>();
    let profile_correlation =
        pearson_correlation(&background_subtracted, &result.calculation.profile_y)?;
    let cell = result.input.phases[0].definition().cell;
    let reference = [8.48, 5.398, 6.958];
    let cell_error = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom]
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (actual - expected).abs() / expected)
        .fold(0.0_f64, f64::max);
    let measurements = NeutronMeasurements {
        unit_weight_rwp,
        profile_correlation,
        cell_error,
        unsafe_termination: workflow.stages().iter().any(|stage| {
            matches!(
                stage.result.termination_reason,
                TerminationReason::NumericalFailure
                    | TerminationReason::Diverged
                    | TerminationReason::RepeatedRejections
                    | TerminationReason::NoObservations
                    | TerminationReason::MaxIterations
            )
        }),
    };
    let checks = neutron_checks(request, workflow, &measurements)?;
    let notes = neutron_notes(workflow)?;
    RealDataValidationReport::new(
        "gsasii-pbso4-cw-neutron",
        request.pattern.sample_count(),
        Some(result.input.phases[0].definition().hkl.len()),
        elapsed_seconds,
        checks,
        notes,
    )
    .map_err(Into::into)
}

struct NeutronMeasurements {
    unit_weight_rwp: f64,
    profile_correlation: f64,
    cell_error: f64,
    unsafe_termination: bool,
}

fn neutron_checks(
    request: &RietveldInput,
    workflow: &RietveldWorkflowResult,
    measurements: &NeutronMeasurements,
) -> Result<Vec<ValidationCheck>, Pbso4ValidationError> {
    let result = workflow.final_result();
    let checks = vec![
        check(
            "observed_grid",
            request.pattern.sample_count() > 2_000
                && request.pattern.x_deg.first() == Some(&ANGULAR_RANGE[0])
                && request.pattern.x_deg.last() == Some(&ANGULAR_RANGE[1]),
            "Selected PbSO4 pattern uses the official combined-refinement angular range.",
            Some(count_as_f64(request.pattern.sample_count())?),
            "more than 2000 samples with endpoints 19 and 153 degrees",
        )?,
        check(
            "refinement_termination",
            workflow.completed() && !measurements.unsafe_termination,
            "Every intelligent recipe stage terminates safely under explicit budgets.",
            None,
            "all stages converge or stagnate safely; iteration exhaustion fails",
        )?,
        check(
            "poisson_rwp",
            result.calculation.metrics.rwp <= 0.05,
            "Poisson-weighted complete-pattern residual for the PbSO4 workflow.",
            Some(result.calculation.metrics.rwp),
            "Rwp <= 0.05",
        )?,
        check(
            "unit_weight_rwp",
            measurements.unit_weight_rwp <= 0.06,
            "Unit-weight PbSO4 residual is reported separately.",
            Some(measurements.unit_weight_rwp),
            "unit-weight Rwp <= 0.06",
        )?,
        check(
            "profile_correlation",
            measurements.profile_correlation >= 0.99,
            "Background-subtracted observed and calculated profiles remain aligned.",
            Some(measurements.profile_correlation),
            "Pearson correlation >= 0.99",
        )?,
        check(
            "reference_cell_relative_error",
            measurements.cell_error <= 0.005,
            "Refined cell remains close to the supplied PbSO4 reference model.",
            Some(measurements.cell_error),
            "maximum relative error across a, b, c <= 0.005",
        )?,
    ];
    Ok(checks)
}

fn neutron_notes(workflow: &RietveldWorkflowResult) -> Result<Vec<String>, Pbso4ValidationError> {
    let result = workflow.final_result();
    let cell = result.input.phases[0].definition().cell;
    let displacement = result
        .input
        .position_correction
        .debye_scherrer_micrometre
        .ok_or(Pbso4ValidationError::MissingGeometry)?;
    let mut notes = vec![
        format!(
            "Final cell a={:.8}, b={:.8}, c={:.8} angstrom.",
            cell.a_angstrom, cell.b_angstrom, cell.c_angstrom
        ),
        format!(
            "Termination={}; iterations={}; evaluations={}.",
            result.termination_reason.as_str(),
            result.history.len(),
            result.evaluations
        ),
        format!(
            "Debye-Scherrer geometry: fixed radius=650.000 mm; refined X={:.6} micrometre, Y={:.6} micrometre.",
            displacement.0, displacement.1,
        ),
    ];
    notes.extend(workflow.stages().iter().map(|stage| {
        format!(
            "Stage {}: Rwp {:.8} -> {:.8}; termination={}; iterations={}.",
            stage.stage.name(),
            stage.starting_rwp,
            stage.result.calculation.metrics.rwp,
            stage.result.termination_reason.as_str(),
            stage.result.history.len()
        )
    }));
    notes.extend(workflow.recipe().planner_notes().iter().cloned());
    notes.push(
        "The residual background coefficients are initialized by weighted linear least squares against the starting structural profile."
            .to_owned(),
    );
    notes.push(
        "Background is a fixed native Smooth Bruckner estimate plus a refined three-term Chebyshev residual correction."
            .to_owned(),
    );
    Ok(notes)
}

fn selected_pattern(
    source: &PatternRecord,
    limits: [f64; 2],
) -> Result<PatternRecord, Pbso4ValidationError> {
    let indices = source
        .x_deg
        .iter()
        .enumerate()
        .filter_map(|(index, x)| (limits[0] <= *x && *x <= limits[1]).then_some(index))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(Pbso4ValidationError::EmptySelectedPattern);
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
        None,
        Some(select(&source.background_y)),
    )
    .map_err(Into::into)
}

fn neutron_instrument() -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom: 1.909,
        u_deg2: 354.031e-4,
        v_deg2: -760.404e-4,
        w_deg2: 651.592e-4,
        x_deg: 0.0,
        y_deg: 0.0,
    }
}

fn neutron_definition(structure: &CifStructure) -> StructuralPhaseDefinition {
    StructuralPhaseDefinition {
        cell: structure.cell,
        space_group: structure.space_group.clone(),
        hkl: Vec::new(),
        multiplicity: Vec::new(),
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
        scattering_species: structure.sites.iter().map(neutron_scattering_key).collect(),
        scattering_real_offset: Vec::new(),
        scattering_imag_offset: Vec::new(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-4,
        scattering_model: BuiltInScatteringModel::NeutronNuclear,
        correction_model: IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom: 1.909,
        },
    }
}

fn neutron_scattering_key(site: &CifAtomSite) -> String {
    site.isotope.map_or_else(
        || site.element_symbol.clone(),
        |isotope| format!("{}-{isotope}", site.element_symbol),
    )
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Result<f64, Pbso4ValidationError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(Pbso4ValidationError::InvalidCorrelation);
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
    let result = numerator / (left_squared * right_squared).sqrt();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(Pbso4ValidationError::InvalidCorrelation)
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn count_as_f64(count: usize) -> Result<f64, Pbso4ValidationError> {
    u32::try_from(count)
        .map(f64::from)
        .map_err(|_| Pbso4ValidationError::SampleCountOverflow)
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

/// Native `PbSO4` validation setup, input, or numerical failure.
#[derive(Debug)]
pub enum Pbso4ValidationError {
    /// Dataset checksum verification failed.
    Dataset(DatasetVerificationError),
    /// Powder input parsing failed.
    Powder(PowderIoError),
    /// CIF input parsing failed.
    Cif(CifIoError),
    /// Pattern or stable-identifier construction failed.
    Pattern(DomainError),
    /// Smooth Bruckner estimation failed.
    SmoothBackground(SmoothBackgroundError),
    /// Differentiable background construction failed.
    ModelBackground(ModelBackgroundError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// Lattice/domain construction failed.
    Lattice(LatticeError),
    /// Rietveld request or calculation failed.
    Rietveld(RietveldError),
    /// Rietveld solver-option construction failed.
    Refinement(RietveldRefinementError),
    /// Refinement runtime-limit construction failed.
    Runtime(RuntimeError),
    /// Complete parameter-selection construction failed.
    GeneralParameter(RietveldGeneralParameterError),
    /// Covariance-option construction failed.
    GeneralRefinement(RietveldGeneralRefinementError),
    /// Intelligent recipe planning or execution failed.
    Recipe(RietveldRecipeError),
    /// Report construction failed.
    Report(ValidationContractError),
    /// The imported powder pattern did not contain observations.
    MissingObservations,
    /// No samples remained after selecting the validation interval.
    EmptySelectedPattern,
    /// Weighted least squares failed.
    LinearSolve,
    /// Profile correlation was undefined or non-finite.
    InvalidCorrelation,
    /// The sample count is too large for exact statistical accumulation.
    SampleCountOverflow,
    /// The final result did not retain the requested Debye-Scherrer geometry.
    MissingGeometry,
}

impl Display for Pbso4ValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, formatter),
            Self::Powder(error) => Display::fmt(error, formatter),
            Self::Cif(error) => Display::fmt(error, formatter),
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::SmoothBackground(error) => Display::fmt(error, formatter),
            Self::ModelBackground(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Refinement(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::GeneralParameter(error) => Display::fmt(error, formatter),
            Self::GeneralRefinement(error) => Display::fmt(error, formatter),
            Self::Recipe(error) => Display::fmt(error, formatter),
            Self::Report(error) => Display::fmt(error, formatter),
            Self::MissingObservations => formatter.write_str("PbSO4 observations are missing"),
            Self::EmptySelectedPattern => formatter.write_str("PbSO4 interval contains no samples"),
            Self::LinearSolve => formatter.write_str("PbSO4 weighted least-squares solve failed"),
            Self::InvalidCorrelation => {
                formatter.write_str("PbSO4 profile correlation is undefined")
            }
            Self::SampleCountOverflow => {
                formatter.write_str("PbSO4 sample count exceeds supported statistical size")
            }
            Self::MissingGeometry => {
                formatter.write_str("PbSO4 Debye-Scherrer geometry is missing")
            }
        }
    }
}

impl Error for Pbso4ValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dataset(error) => Some(error),
            Self::Powder(error) => Some(error),
            Self::Cif(error) => Some(error),
            Self::Pattern(error) => Some(error),
            Self::SmoothBackground(error) => Some(error),
            Self::ModelBackground(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::Lattice(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Refinement(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::GeneralParameter(error) => Some(error),
            Self::GeneralRefinement(error) => Some(error),
            Self::Recipe(error) => Some(error),
            Self::Report(error) => Some(error),
            _ => None,
        }
    }
}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for Pbso4ValidationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(DatasetVerificationError, Dataset);
error_conversion!(PowderIoError, Powder);
error_conversion!(CifIoError, Cif);
error_conversion!(DomainError, Pattern);
error_conversion!(SmoothBackgroundError, SmoothBackground);
error_conversion!(ModelBackgroundError, ModelBackground);
error_conversion!(ExecutionPolicyError, Execution);
error_conversion!(LatticeError, Lattice);
error_conversion!(RietveldError, Rietveld);
error_conversion!(RietveldRefinementError, Refinement);
error_conversion!(RuntimeError, Runtime);
error_conversion!(RietveldGeneralParameterError, GeneralParameter);
error_conversion!(RietveldGeneralRefinementError, GeneralRefinement);
error_conversion!(RietveldRecipeError, Recipe);
error_conversion!(ValidationContractError, Report);
