//! Native three-phase Cu K-alpha QARR real-data validation.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    ConstantWavelengthInstrument, FcjGeometry, OwnedCwContributions, smooth_bruckner,
};
use phasesmith_crystallography::{
    IntegratedIntensityCorrectionModel, PreparedReflectionGenerator, ReflectionRange,
};
use phasesmith_engine::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPhaseDefinition,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_io::{
    CifReadLimits, CifStructure, PowderFormat, PowderReadLimits, read_cif_file, read_powder_file,
};
use phasesmith_model::{FixedWavelengthSpectrum, PatternRecord, RecordId};
use phasesmith_workflows::{
    QuantitativePhase, RefinementLimits, RietveldCalculationOptions, RietveldCovarianceOptions,
    RietveldInput, RietveldInstrumentParameter, RietveldParameterSelection, RietveldPhase,
    RietveldRefinementOptions, RietveldSamplePhysicsModel, RietveldStructuralSelection,
    TerminationReason, calculate_rietveld_pattern, quantitative_phase_analysis,
    refine_general_rietveld,
};

use crate::{
    RealDataValidationReport, ValidationCheck, ValidationStatus, verify_validation_dataset,
};

const DATASET_ID: &str = "iucr-qarr-1g";
const PHASES: [&str; 3] = ["Al2O3", "ZnO", "CaF2"];
const EXPECTED_FRACTIONS: [f64; 3] = [0.3137, 0.3421, 0.3442];
const QPA: [(f64, f64); 3] = [(6.0, 101.961_276), (2.0, 81.38), (4.0, 78.074_806)];
const EXPECTED_EXPANDED: [usize; 3] = [30, 4, 12];

/// Run the checksum-pinned three-phase QARR refinement entirely in Rust.
///
/// # Errors
///
/// Returns [`QarrValidationError`] for invalid/missing external data or native
/// calculation/refinement failures.
#[allow(clippy::too_many_lines)]
pub fn run_qarr_1g_validation(
    dataset_directory: &Path,
) -> Result<RealDataValidationReport, QarrValidationError> {
    let started = Instant::now();
    verify_validation_dataset(DATASET_ID, dataset_directory).map_err(qerr)?;
    let imported = read_powder_file(
        dataset_directory.join("cpd-1g.prn"),
        PowderFormat::Columns,
        1,
        PowderReadLimits::default(),
    )
    .map_err(qerr)?;
    let observed = imported
        .pattern
        .observed_y
        .as_ref()
        .ok_or_else(|| QarrValidationError("QARR observations are missing".to_owned()))?;
    let background = smooth_bruckner(observed, 50, 50).map_err(qerr)?;
    let uncertainty = observed
        .iter()
        .map(|value| value.max(1.0).sqrt())
        .collect::<Vec<_>>();
    let pattern = PatternRecord::new(
        imported.pattern.x_deg,
        Some(observed.clone()),
        Some(uncertainty),
        None,
        Some(background),
    )
    .map_err(qerr)?;
    let instrument_values = read_instrument(&dataset_directory.join("cuka.instprm"))?;
    let instrument = ConstantWavelengthInstrument {
        wavelength_angstrom: instrument_values.lam1,
        u_deg2: instrument_values.u * 1.0e-4,
        v_deg2: instrument_values.v * 1.0e-4,
        w_deg2: instrument_values.w * 1.0e-4,
        x_deg: instrument_values.x * 1.0e-2,
        y_deg: instrument_values.y * 1.0e-2,
    };
    let spectrum = FixedWavelengthSpectrum::new(
        vec![instrument_values.lam1, instrument_values.lam2],
        vec![1.0, instrument_values.ratio],
    )
    .map_err(qerr)?;
    let position = MonochromaticPositionCorrection {
        zero_shift_deg: 0.0,
        bragg_brentano_mm: None,
        debye_scherrer_micrometre: None,
    };
    let axial = Some(FcjGeometry {
        sample_over_radius: instrument_values.sh_over_l / 2.0,
        detector_over_radius: instrument_values.sh_over_l / 2.0,
    });
    let execution = ExecutionPolicy::bounded_default().map_err(qerr)?;
    let mut phases = Vec::new();
    let mut expanded_counts = Vec::new();
    for phase_id in PHASES {
        let structure = read_cif_file(
            dataset_directory.join(format!("{phase_id}.cif")),
            None,
            true,
            CifReadLimits::default(),
        )
        .map_err(qerr)?
        .structure;
        expanded_counts.push(expanded_site_count(&structure)?);
        phases.push(qarr_phase(
            phase_id,
            &structure,
            &pattern,
            instrument_values.polarization,
            instrument_values.lam1,
        )?);
    }
    let starting = RietveldInput::new_fixed_spectrum(
        pattern.clone(),
        instrument,
        spectrum.clone(),
        axial,
        position,
        phases,
    )
    .map_err(qerr)?;
    let scales = initial_scales(&starting, execution.clone())?;
    let mut phases = starting.phases.clone();
    for (phase, scale) in phases.iter_mut().zip(scales) {
        let mut definition = phase.definition().clone();
        definition.scale = scale;
        let sample_physics = phase.sample_physics().cloned();
        let mut replacement = RietveldPhase::new_with_site_ids(
            phase.phase_id().clone(),
            phase.name(),
            phase.site_ids().to_vec(),
            definition,
            phase.contributions().clone(),
        )
        .map_err(qerr)?;
        if let Some(model) = sample_physics {
            replacement = replacement.with_sample_physics(model);
        }
        *phase = replacement;
    }
    let input =
        RietveldInput::new_fixed_spectrum(pattern, instrument, spectrum, axial, position, phases)
            .map_err(qerr)?;
    let first_selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        vec![
            RietveldInstrumentParameter::UDeg2,
            RietveldInstrumentParameter::VDeg2,
            RietveldInstrumentParameter::WDeg2,
            RietveldInstrumentParameter::ZeroShiftDeg,
        ],
        false,
        false,
    )
    .map_err(qerr)?;
    let first = refine_stage(&input, &first_selection, 20, 800, 2, 0.2, execution.clone())?;
    let second_selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            u_iso: true,
            ..RietveldStructuralSelection::default()
        },
        first_selection.instrument.clone(),
        false,
        true,
    )
    .map_err(qerr)?;
    let second = refine_stage(
        &first.input,
        &second_selection,
        35,
        1_500,
        3,
        0.15,
        execution.clone(),
    )?;
    let polish_selection = RietveldParameterSelection::new(
        RietveldStructuralSelection {
            phase_scale: true,
            ..RietveldStructuralSelection::default()
        },
        Vec::new(),
        false,
        false,
    )
    .map_err(qerr)?;
    let result = refine_stage(&second.input, &polish_selection, 10, 200, 1, 1.0, execution)?;
    build_report(
        started.elapsed().as_secs_f64(),
        &input,
        &expanded_counts,
        &first,
        &second,
        &result,
    )
}

fn refine_stage(
    input: &RietveldInput,
    selection: &RietveldParameterSelection,
    max_iterations: usize,
    max_evaluations: usize,
    min_iterations: usize,
    max_step: f64,
    execution: ExecutionPolicy,
) -> Result<phasesmith_workflows::RietveldGeneralRefinementResult, QarrValidationError> {
    let options = RietveldRefinementOptions::new(
        RietveldCalculationOptions::new(30.0, true, execution).map_err(qerr)?,
        RefinementLimits::new(max_iterations, max_evaluations, None, 20).map_err(qerr)?,
        min_iterations,
        1.0e-7,
        1.0e-7,
        1.0e-6,
        10.0,
        0.3,
        1.0e-6,
        30,
        max_step,
        8,
    )
    .map_err(qerr)?;
    refine_general_rietveld(
        input,
        selection,
        &[None, None, None],
        &[],
        &options,
        RietveldCovarianceOptions::new(false, 64, 1.0 - 1.0e-10).map_err(qerr)?,
        None,
        None,
    )
    .map_err(qerr)
}

fn qarr_phase(
    phase_id: &str,
    structure: &CifStructure,
    pattern: &PatternRecord,
    polarization: f64,
    wavelength_angstrom: f64,
) -> Result<RietveldPhase, QarrValidationError> {
    let reflections =
        PreparedReflectionGenerator::new(structure.space_group.clone(), true, 50_000_000)
            .map_err(qerr)?
            .generate(
                structure.cell,
                ReflectionRange::CwTwoTheta {
                    min_deg: pattern.x_deg[0],
                    max_deg: *pattern
                        .x_deg
                        .last()
                        .ok_or_else(|| QarrValidationError("QARR pattern is empty".to_owned()))?,
                    wavelength_angstrom,
                },
            )
            .map_err(qerr)?;
    let offsets = structure
        .sites
        .iter()
        .map(|site| dispersion(&site.element_symbol))
        .collect::<Result<Vec<_>, _>>()?;
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
            .map(|site| {
                if site.anisotropic_displacement.is_some() {
                    0.0
                } else {
                    site.u_iso_angstrom2.unwrap_or(0.005)
                }
            })
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
        scattering_real_offset: offsets.iter().map(|value| value.0).collect(),
        scattering_imag_offset: offsets.iter().map(|value| value.1).collect(),
        scale: 1.0,
        coordinate_tolerance: 1.0e-4,
        scattering_model: BuiltInScatteringModel::XrayNonResonant,
        correction_model: IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom,
            polarization,
        },
    };
    let site_ids = structure
        .sites
        .iter()
        .map(|site| RecordId::new(&site.site_id).map_err(qerr))
        .collect::<Result<Vec<_>, _>>()?;
    let mut providers = vec![
        RietveldSamplePhysicsModel::IsotropicSize {
            crystallite_size_nm: 100.0,
            shape_factor: 0.9,
        },
        RietveldSamplePhysicsModel::IsotropicMicrostrain {
            rms_microstrain: 8.0e-4,
        },
    ];
    if matches!(phase_id, "Al2O3" | "ZnO") {
        providers.push(RietveldSamplePhysicsModel::MarchDollase {
            ratio: 1.0,
            preferred_axis_hkl: [0.0, 0.0, 1.0],
        });
    }
    RietveldPhase::new_with_site_ids(
        RecordId::new(phase_id).map_err(qerr)?,
        phase_id,
        site_ids,
        definition,
        OwnedCwContributions::neutral(reflections.len()),
    )
    .map(|phase| phase.with_sample_physics(RietveldSamplePhysicsModel::Composite(providers)))
    .map_err(qerr)
}

fn initial_scales(
    input: &RietveldInput,
    execution: ExecutionPolicy,
) -> Result<Vec<f64>, QarrValidationError> {
    let calculation = calculate_rietveld_pattern(
        input,
        &RietveldCalculationOptions::new(30.0, true, execution).map_err(qerr)?,
    )
    .map_err(qerr)?;
    let observed = input
        .pattern
        .observed_y
        .as_ref()
        .ok_or_else(|| QarrValidationError("QARR observations are missing".to_owned()))?;
    let uncertainty = input
        .pattern
        .uncertainty
        .as_ref()
        .ok_or_else(|| QarrValidationError("QARR uncertainty is missing".to_owned()))?;
    let rows = input.pattern.sample_count();
    let columns = calculation.phases.len();
    let mut design = Vec::with_capacity(rows * columns);
    for (row, sigma) in uncertainty.iter().enumerate().take(rows) {
        for phase in &calculation.phases {
            design.push(phase.result.accumulation.y[row] / sigma);
        }
    }
    let target = observed
        .iter()
        .zip(&input.pattern.background_y)
        .zip(uncertainty)
        .map(|((observed, background), sigma)| (observed - background) / sigma)
        .collect::<Vec<_>>();
    let solution = DMatrix::from_row_slice(rows, columns, &design)
        .svd(true, true)
        .solve(&DVector::from_vec(target), f64::EPSILON)
        .map_err(|_| QarrValidationError("QARR initial scale solve failed".to_owned()))?;
    if solution
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(QarrValidationError(
            "QARR initial scales are invalid".to_owned(),
        ));
    }
    Ok(solution.as_slice().to_vec())
}

#[allow(clippy::too_many_lines)]
fn build_report(
    elapsed_seconds: f64,
    starting: &RietveldInput,
    expanded_counts: &[usize],
    first: &phasesmith_workflows::RietveldGeneralRefinementResult,
    second: &phasesmith_workflows::RietveldGeneralRefinementResult,
    result: &phasesmith_workflows::RietveldGeneralRefinementResult,
) -> Result<RealDataValidationReport, QarrValidationError> {
    let observed = starting
        .pattern
        .observed_y
        .as_ref()
        .ok_or_else(|| QarrValidationError("QARR observations are missing".to_owned()))?;
    let residual = result
        .calculation
        .y
        .iter()
        .zip(observed)
        .map(|(calculated, observed)| calculated - observed)
        .collect::<Vec<_>>();
    let unit_rwp = (dot(&residual, &residual) / dot(observed, observed)).sqrt();
    let background_subtracted = observed
        .iter()
        .zip(&result.calculation.background_y)
        .map(|(observed, background)| observed - background)
        .collect::<Vec<_>>();
    let correlation = pearson(&background_subtracted, &result.calculation.profile_y)?;
    let quantitative = result
        .input
        .phases
        .iter()
        .enumerate()
        .map(|(index, phase)| {
            QuantitativePhase::new(
                phase.phase_id().as_str(),
                phase.definition().scale,
                QPA[index].0,
                QPA[index].1,
                phase
                    .definition()
                    .cell
                    .geometry()
                    .map_err(qerr)?
                    .volume_angstrom3,
            )
            .map_err(qerr)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fractions = quantitative_phase_analysis(&quantitative).map_err(qerr)?;
    let max_fraction_error = fractions
        .iter()
        .zip(EXPECTED_FRACTIONS)
        .map(|(item, expected)| (item.weight_fraction - expected).abs())
        .fold(0.0_f64, f64::max);
    let safe = [first, second, result].iter().all(|stage| {
        !matches!(
            stage.termination_reason,
            TerminationReason::NumericalFailure
                | TerminationReason::Diverged
                | TerminationReason::RepeatedRejections
                | TerminationReason::NoObservations
        )
    });
    let checks = vec![
        check(
            "observed_grid",
            starting.pattern.sample_count() == 7_251
                && starting.pattern.x_deg[0].to_bits() == 5.0_f64.to_bits()
                && starting.pattern.x_deg.last().unwrap_or(&0.0).to_bits() == 150.0_f64.to_bits(),
            "QARR 1g pattern spans the published 5-150 degree grid.",
            Some(count_as_f64(starting.pattern.sample_count())?),
            "7251 samples with endpoints 5 and 150 degrees",
        )?,
        check(
            "site_expansion",
            expanded_counts == EXPECTED_EXPANDED,
            "The documented 1e-4 CIF coordinate tolerance gives physical unit-cell contents.",
            Some(count_as_f64(expanded_counts.iter().sum::<usize>())?),
            "expanded site counts Al2O3=30, ZnO=4, CaF2=12",
        )?,
        check(
            "refinement_termination",
            safe,
            "Refinement returns a finite last accepted state under explicit iteration budgets.",
            None,
            "no numerical failure, divergence, repeated rejection, or empty data",
        )?,
        check(
            "poisson_rwp",
            result.calculation.metrics.rwp <= 0.20,
            "Poisson-weighted QARR profile gate with fixed CIF anisotropic displacement.",
            Some(result.calculation.metrics.rwp),
            "Rwp with sigma=sqrt(max(counts, 1)) <= 0.20",
        )?,
        check(
            "unit_weight_rwp",
            unit_rwp <= 0.15,
            "Unit-weight profile residual is reported separately from Poisson-weighted Rwp.",
            Some(unit_rwp),
            "unit-weight Rwp <= 0.15",
        )?,
        check(
            "profile_correlation",
            correlation >= 0.98,
            "Background-subtracted observed and calculated profiles remain strongly aligned.",
            Some(correlation),
            "Pearson correlation >= 0.98",
        )?,
        check(
            "qpa_weight_fraction",
            max_fraction_error <= 0.02,
            "Hill--Howard weight fractions agree with the independently weighed phase fractions.",
            Some(max_fraction_error),
            "maximum absolute phase error <= 0.02",
        )?,
    ];
    let fraction_note = fractions
        .iter()
        .map(|item| format!("{}={:.3}%", item.phase_id, 100.0 * item.weight_fraction))
        .collect::<Vec<_>>()
        .join(", ");
    RealDataValidationReport::new(
        DATASET_ID,
        starting.pattern.sample_count(),
        Some(
            result
                .input
                .phases
                .iter()
                .map(|phase| phase.reflection_ids().len())
                .sum(),
        ),
        elapsed_seconds,
        checks,
        vec![
            format!("Calculated crystalline weight fractions: {fraction_note}."),
            stage_note("Stage 1", first),
            stage_note("Stage 2", second),
            stage_note("Stage 3 scale polish", result),
            format!("Expanded sites at tolerance 1e-4: {expanded_counts:?}."),
            "Fixed Cu K-alpha1 dispersion offsets are used for both fixed spectrum components."
                .to_owned(),
        ],
    )
    .map_err(qerr)
}

fn stage_note(
    name: &str,
    result: &phasesmith_workflows::RietveldGeneralRefinementResult,
) -> String {
    format!(
        "{name} termination={}, iterations={}, evaluations={}, Rwp={:.8}.",
        result.termination_reason.as_str(),
        result.history.len(),
        result.evaluations,
        result.calculation.metrics.rwp
    )
}

fn expanded_site_count(structure: &CifStructure) -> Result<usize, QarrValidationError> {
    structure
        .space_group
        .expand_sites(
            &structure
                .sites
                .iter()
                .map(|site| site.fractional_xyz)
                .collect::<Vec<_>>(),
            1.0e-4,
        )
        .map(|value| value.fractional_xyz.len())
        .map_err(qerr)
}

fn dispersion(element: &str) -> Result<(f64, f64), QarrValidationError> {
    match element {
        "Al" => Ok((0.212_567, 0.245_496)),
        "O" => Ok((0.049_383_9, 0.032_232_4)),
        "Zn" => Ok((-1.545_988, 0.677_687)),
        "Ca" => Ok((0.365_107, 1.285_341)),
        "F" => Ok((0.073_083_9, 0.053_346_8)),
        _ => Err(QarrValidationError(format!(
            "unsupported QARR element {element}"
        ))),
    }
}

#[derive(Clone, Copy)]
struct InstrumentValues {
    lam1: f64,
    lam2: f64,
    ratio: f64,
    polarization: f64,
    u: f64,
    v: f64,
    w: f64,
    x: f64,
    y: f64,
    sh_over_l: f64,
}

fn read_instrument(path: &Path) -> Result<InstrumentValues, QarrValidationError> {
    let text = fs::read_to_string(path).map_err(qerr)?;
    let value = |name: &str| -> Result<f64, QarrValidationError> {
        text.lines()
            .find_map(|line| {
                line.split_once(':')
                    .filter(|(key, _)| *key == name)
                    .and_then(|(_, value)| value.parse().ok())
            })
            .ok_or_else(|| QarrValidationError(format!("QARR instrument is missing {name}")))
    };
    Ok(InstrumentValues {
        lam1: value("Lam1")?,
        lam2: value("Lam2")?,
        ratio: value("I(L2)/I(L1)")?,
        polarization: value("Polariz.")?,
        u: value("U")?,
        v: value("V")?,
        w: value("W")?,
        x: value("X")?,
        y: value("Y")?,
        sh_over_l: value("SH/L")?,
    })
}

fn pearson(left: &[f64], right: &[f64]) -> Result<f64, QarrValidationError> {
    let count = count_as_f64(left.len())?;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let numerator = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>();
    let left_squared = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_squared = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    let value = numerator / (left_squared * right_squared).sqrt();
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| QarrValidationError("QARR profile correlation is undefined".to_owned()))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn count_as_f64(count: usize) -> Result<f64, QarrValidationError> {
    u32::try_from(count)
        .map(f64::from)
        .map_err(|_| QarrValidationError("QARR count exceeds exact statistical range".to_owned()))
}

fn check(
    check_id: &str,
    passed: bool,
    detail: &str,
    measured: Option<f64>,
    criterion: &str,
) -> Result<ValidationCheck, QarrValidationError> {
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
    .map_err(qerr)
}

fn qerr(error: impl Display) -> QarrValidationError {
    QarrValidationError(error.to_string())
}

/// Native QARR validation setup or numerical failure.
#[derive(Debug)]
pub struct QarrValidationError(String);

impl Display for QarrValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for QarrValidationError {}
