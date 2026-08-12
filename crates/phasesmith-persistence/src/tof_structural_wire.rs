//! Explicit wire records for resumable structural multi-bank neutron TOF analyses.

use phasesmith_core::{
    OwnedCwContributions, TofBankGeometry, TofInstrument, TofInstrumentParameter,
};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, UnitCell};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, ParameterChange, ParameterKey, ParameterSet,
    ParameterSpec, RefinementLimits, RietveldPhase, RietveldStructuralSelection, StructuralTofBank,
    StructuralTofMultiBankAnalysis, StructuralTofMultiBankCheckpoint, StructuralTofMultiBankInput,
    StructuralTofMultiBankIterationRecord, StructuralTofMultiBankProjectState,
    StructuralTofMultiBankRefinementOptions, TofChebyshevBackground, TofInstrumentParameterBound,
};
use serde::{Deserialize, Serialize};

use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireStructuralTofAnalysis {
    analysis_id: String,
    phase_id: String,
    site_ids: Vec<String>,
    structural_selection: WireStructuralSelection,
    lattice_bounds: Option<WireLatticeBounds>,
    banks: Vec<WireStructuralBank>,
    input_options: WireInputOptions,
    options: WireRefinementOptions,
    checkpoint: Option<WireCheckpoint>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)] // Mirrors the public independent selection families.
struct WireStructuralSelection {
    lattice: bool,
    coordinates: bool,
    occupancy: bool,
    u_iso: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLatticeBounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStructuralBank {
    bank_id: String,
    two_theta_deg: f64,
    correction: WireTofCorrection,
    scale: f64,
    scale_bounds: [Option<f64>; 2],
    refine_scale: bool,
    background: Option<WireBackground>,
    refine_background: bool,
    instrument_bounds: Vec<WireInstrumentBound>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireTofCorrection {
    Neutral,
    TimeOfFlightNeutronLorentz,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBackground {
    background_id: String,
    coefficients: Vec<f64>,
    domain_us: [f64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstrumentBound {
    parameter: String,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInputOptions {
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    requested_threads: Option<usize>,
    minimum_parallel_tasks: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRefinementOptions {
    max_iterations: usize,
    max_evaluations: usize,
    max_runtime_seconds: Option<f64>,
    max_consecutive_rejections: usize,
    min_iterations: usize,
    objective_tolerance: f64,
    parameter_tolerance: f64,
    initial_damping: f64,
    damping_increase: f64,
    damping_decrease: f64,
    max_scaled_parameter_step: f64,
    max_backtracks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCheckpoint {
    phase: WireAcceptedPhase,
    banks: Vec<WireAcceptedBank>,
    parameters: Vec<WireParameter>,
    objective: f64,
    damping: f64,
    history: Vec<WireIteration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedPhase {
    cell: [f64; 6],
    fractional_xyz: Vec<[f64; 3]>,
    occupancy: Vec<f64>,
    u_iso_angstrom2: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAcceptedBank {
    bank_id: String,
    instrument: Vec<f64>,
    scale: f64,
    background_coefficients: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameter {
    module: String,
    owner_id: String,
    name: String,
    value: f64,
    unit: String,
    bounds: [Option<f64>; 2],
    scale: f64,
    refine: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireIteration {
    iteration: usize,
    objective: f64,
    objective_change: f64,
    scaled_step_norm: f64,
    damping: f64,
    backtracks: usize,
    parameter_changes: Vec<WireParameterChange>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameterChange {
    module: String,
    owner_id: String,
    name: String,
    before: f64,
    after: f64,
    scaled_change: f64,
}

pub(crate) fn encode_analyses(
    state: &StructuralTofMultiBankProjectState,
) -> Vec<WireStructuralTofAnalysis> {
    state.analyses.iter().map(encode_analysis).collect()
}

fn encode_analysis(analysis: &StructuralTofMultiBankAnalysis) -> WireStructuralTofAnalysis {
    let input = &analysis.input;
    WireStructuralTofAnalysis {
        analysis_id: analysis.analysis_id.as_str().to_owned(),
        phase_id: input.phase.phase_id().as_str().to_owned(),
        site_ids: input
            .phase
            .site_ids()
            .iter()
            .map(|site_id| site_id.as_str().to_owned())
            .collect(),
        structural_selection: WireStructuralSelection {
            lattice: input.structural_selection.lattice,
            coordinates: input.structural_selection.coordinates,
            occupancy: input.structural_selection.occupancy,
            u_iso: input.structural_selection.u_iso,
        },
        lattice_bounds: input
            .lattice_bounds
            .as_ref()
            .map(|bounds| WireLatticeBounds {
                lower: bounds.lower().to_vec(),
                upper: bounds.upper().to_vec(),
            }),
        banks: input.banks.iter().map(encode_bank).collect(),
        input_options: WireInputOptions {
            support_fwhm: input.support_fwhm,
            tail_log: input.tail_log,
            use_uncertainty: input.use_uncertainty,
            requested_threads: input.execution.requested_threads(),
            minimum_parallel_tasks: input.execution.minimum_parallel_tasks(),
        },
        options: encode_options(analysis.options),
        checkpoint: analysis.checkpoint.as_ref().map(encode_checkpoint),
    }
}

fn encode_bank(bank: &StructuralTofBank) -> WireStructuralBank {
    WireStructuralBank {
        bank_id: bank.bank_id.as_str().to_owned(),
        two_theta_deg: bank.geometry.two_theta_deg,
        correction: match bank.correction_model {
            IntegratedIntensityCorrectionModel::Neutral => WireTofCorrection::Neutral,
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { .. } => {
                WireTofCorrection::TimeOfFlightNeutronLorentz
            }
            _ => unreachable!("validated structural TOF correction"),
        },
        scale: bank.scale,
        scale_bounds: [
            bank.scale_bounds
                .lower()
                .is_finite()
                .then_some(bank.scale_bounds.lower()),
            bank.scale_bounds
                .upper()
                .is_finite()
                .then_some(bank.scale_bounds.upper()),
        ],
        refine_scale: bank.refine_scale,
        background: bank.background.as_ref().map(|background| WireBackground {
            background_id: background.background_id().as_str().to_owned(),
            coefficients: background.coefficients().to_vec(),
            domain_us: background.domain_us(),
        }),
        refine_background: bank.refine_background,
        instrument_bounds: bank
            .instrument_bounds
            .iter()
            .map(|bound| WireInstrumentBound {
                parameter: bound.parameter.name().to_owned(),
                lower: bound.lower,
                upper: bound.upper,
            })
            .collect(),
    }
}

fn encode_options(options: StructuralTofMultiBankRefinementOptions) -> WireRefinementOptions {
    WireRefinementOptions {
        max_iterations: options.limits.max_iterations(),
        max_evaluations: options.limits.max_evaluations(),
        max_runtime_seconds: options.limits.max_runtime_seconds(),
        max_consecutive_rejections: options.limits.max_consecutive_rejections(),
        min_iterations: options.min_iterations,
        objective_tolerance: options.objective_tolerance,
        parameter_tolerance: options.parameter_tolerance,
        initial_damping: options.initial_damping,
        damping_increase: options.damping_increase,
        damping_decrease: options.damping_decrease,
        max_scaled_parameter_step: options.max_scaled_parameter_step,
        max_backtracks: options.max_backtracks,
    }
}

fn encode_checkpoint(checkpoint: &StructuralTofMultiBankCheckpoint) -> WireCheckpoint {
    let definition = checkpoint.input.phase.definition();
    WireCheckpoint {
        phase: WireAcceptedPhase {
            cell: cell_values(definition.cell),
            fractional_xyz: definition.fractional_xyz.clone(),
            occupancy: definition.occupancy.clone(),
            u_iso_angstrom2: definition.u_iso_angstrom2.clone(),
        },
        banks: checkpoint
            .input
            .banks
            .iter()
            .map(|bank| WireAcceptedBank {
                bank_id: bank.bank_id.as_str().to_owned(),
                instrument: bank.instrument.values().to_vec(),
                scale: bank.scale,
                background_coefficients: bank
                    .background
                    .as_ref()
                    .map(|background| background.coefficients().to_vec()),
            })
            .collect(),
        parameters: checkpoint
            .parameters
            .specs()
            .iter()
            .map(|parameter| WireParameter {
                module: parameter.key().module().to_owned(),
                owner_id: parameter.key().owner_id().to_owned(),
                name: parameter.key().name().to_owned(),
                value: parameter.value(),
                unit: parameter.unit().to_owned(),
                bounds: [
                    parameter
                        .bounds()
                        .lower()
                        .is_finite()
                        .then_some(parameter.bounds().lower()),
                    parameter
                        .bounds()
                        .upper()
                        .is_finite()
                        .then_some(parameter.bounds().upper()),
                ],
                scale: parameter.scale(),
                refine: parameter.refine(),
            })
            .collect(),
        objective: checkpoint.objective,
        damping: checkpoint.damping,
        history: checkpoint.history.iter().map(encode_iteration).collect(),
    }
}

fn encode_iteration(row: &StructuralTofMultiBankIterationRecord) -> WireIteration {
    WireIteration {
        iteration: row.iteration,
        objective: row.objective,
        objective_change: row.objective_change,
        scaled_step_norm: row.scaled_step_norm,
        damping: row.damping,
        backtracks: row.backtracks,
        parameter_changes: row
            .parameter_changes
            .iter()
            .map(|change| WireParameterChange {
                module: change.key.module().to_owned(),
                owner_id: change.key.owner_id().to_owned(),
                name: change.key.name().to_owned(),
                before: change.before,
                after: change.after,
                scaled_change: change.scaled_change,
            })
            .collect(),
    }
}

pub(crate) fn decode_state(
    project: ProjectRecord,
    analyses: Vec<WireStructuralTofAnalysis>,
    limits: ProjectReadLimits,
) -> Result<StructuralTofMultiBankProjectState, PersistenceError> {
    if analyses.len() > limits.max_histograms {
        return Err(PersistenceError::LimitExceeded {
            message: "project exceeds max_histograms structural TOF analyses".to_owned(),
        });
    }
    let analyses = analyses
        .into_iter()
        .map(|analysis| decode_analysis(&project, analysis, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let state = StructuralTofMultiBankProjectState { project, analyses };
    state.validate().map_err(|error| {
        invalid_record(format!("invalid structural TOF project state: {error}"))
    })?;
    Ok(state)
}

fn decode_analysis(
    project: &ProjectRecord,
    wire: WireStructuralTofAnalysis,
    limits: ProjectReadLimits,
) -> Result<StructuralTofMultiBankAnalysis, PersistenceError> {
    let checkpoint_too_large = wire.checkpoint.as_ref().is_some_and(|checkpoint| {
        checkpoint.history.len() > limits.max_array_elements
            || checkpoint.parameters.len() > limits.max_array_elements
            || checkpoint.banks.len() > limits.max_histograms
    });
    if wire.banks.len() > limits.max_histograms
        || wire.site_ids.len() > limits.max_array_elements
        || checkpoint_too_large
    {
        return Err(PersistenceError::LimitExceeded {
            message: "structural TOF analysis exceeds project read limits".to_owned(),
        });
    }
    let analysis_id = RecordId::new(wire.analysis_id).map_err(domain_record)?;
    let phase_id = RecordId::new(wire.phase_id).map_err(domain_record)?;
    let phase_record = project
        .phases
        .iter()
        .find(|phase| phase.phase_id == phase_id)
        .ok_or_else(|| invalid_record(format!("unknown structural TOF phase {phase_id}")))?;
    let site_ids = wire
        .site_ids
        .into_iter()
        .map(|site_id| RecordId::new(site_id).map_err(domain_record))
        .collect::<Result<Vec<_>, _>>()?;
    let phase = RietveldPhase::new_with_site_ids(
        phase_record.phase_id.clone(),
        phase_record.name.clone(),
        site_ids,
        phase_record.definition.clone(),
        OwnedCwContributions::neutral(phase_record.definition.hkl.len()),
    )
    .map_err(workflow_record)?;
    let selection = RietveldStructuralSelection {
        lattice: wire.structural_selection.lattice,
        coordinates: wire.structural_selection.coordinates,
        occupancy: wire.structural_selection.occupancy,
        u_iso: wire.structural_selection.u_iso,
        phase_scale: false,
    };
    let lattice_bounds = wire
        .lattice_bounds
        .map(|bounds| {
            let parameterization = LatticeParameterization::new(
                phase.definition().space_group.clone(),
                phase.definition().cell,
            )
            .map_err(workflow_record)?;
            LatticeBounds::new(&parameterization, bounds.lower, bounds.upper)
                .map_err(workflow_record)
        })
        .transpose()?;
    let banks = wire
        .banks
        .into_iter()
        .map(|bank| decode_bank(project, bank))
        .collect::<Result<Vec<_>, _>>()?;
    let input = StructuralTofMultiBankInput {
        phase,
        structural_selection: selection,
        lattice_bounds,
        banks,
        support_fwhm: wire.input_options.support_fwhm,
        tail_log: wire.input_options.tail_log,
        use_uncertainty: wire.input_options.use_uncertainty,
        execution: ExecutionPolicy::new(
            wire.input_options.requested_threads,
            wire.input_options.minimum_parallel_tasks,
        )
        .map_err(workflow_record)?,
    };
    input.validate().map_err(workflow_record)?;
    let options = decode_options(wire.options)?;
    let checkpoint = wire
        .checkpoint
        .map(|checkpoint| decode_checkpoint(&input, checkpoint))
        .transpose()?;
    Ok(StructuralTofMultiBankAnalysis {
        analysis_id,
        input,
        options,
        checkpoint,
    })
}

fn decode_bank(
    project: &ProjectRecord,
    wire: WireStructuralBank,
) -> Result<StructuralTofBank, PersistenceError> {
    let bank_id = RecordId::new(wire.bank_id).map_err(domain_record)?;
    let histogram = project
        .tof_histograms
        .iter()
        .find(|histogram| histogram.histogram_id == bank_id)
        .ok_or_else(|| invalid_record(format!("unknown structural TOF bank {bank_id}")))?;
    let geometry = TofBankGeometry {
        two_theta_deg: wire.two_theta_deg,
    };
    let correction_model = match wire.correction {
        WireTofCorrection::Neutral => IntegratedIntensityCorrectionModel::Neutral,
        WireTofCorrection::TimeOfFlightNeutronLorentz => {
            IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz {
                two_theta_deg: wire.two_theta_deg,
            }
        }
    };
    Ok(StructuralTofBank {
        bank_id,
        pattern: histogram.pattern.clone(),
        instrument: histogram.experiment.instrument,
        geometry,
        correction_model,
        scale: wire.scale,
        scale_bounds: phasesmith_workflows::ParameterBounds::new(
            wire.scale_bounds[0].unwrap_or(f64::NEG_INFINITY),
            wire.scale_bounds[1].unwrap_or(f64::INFINITY),
        )
        .map_err(workflow_record)?,
        refine_scale: wire.refine_scale,
        background: wire.background.map(decode_background).transpose()?,
        refine_background: wire.refine_background,
        instrument_bounds: wire
            .instrument_bounds
            .iter()
            .map(decode_instrument_bound)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn decode_background(wire: WireBackground) -> Result<TofChebyshevBackground, PersistenceError> {
    TofChebyshevBackground::new(
        RecordId::new(wire.background_id).map_err(domain_record)?,
        wire.coefficients,
        wire.domain_us,
    )
    .map_err(workflow_record)
}

fn decode_instrument_bound(
    wire: &WireInstrumentBound,
) -> Result<TofInstrumentParameterBound, PersistenceError> {
    let parameter = TofInstrumentParameter::ALL
        .into_iter()
        .find(|parameter| parameter.name() == wire.parameter)
        .ok_or_else(|| {
            invalid_record(format!(
                "unknown structural TOF instrument parameter {:?}",
                wire.parameter
            ))
        })?;
    TofInstrumentParameterBound::new(parameter, wire.lower, wire.upper).map_err(workflow_record)
}

fn decode_options(
    wire: WireRefinementOptions,
) -> Result<StructuralTofMultiBankRefinementOptions, PersistenceError> {
    let limits = RefinementLimits::new(
        wire.max_iterations,
        wire.max_evaluations,
        wire.max_runtime_seconds,
        wire.max_consecutive_rejections,
    )
    .map_err(workflow_record)?;
    StructuralTofMultiBankRefinementOptions::new(
        limits,
        wire.min_iterations,
        wire.objective_tolerance,
        wire.parameter_tolerance,
        wire.initial_damping,
        wire.damping_increase,
        wire.damping_decrease,
        wire.max_scaled_parameter_step,
        wire.max_backtracks,
    )
    .map_err(workflow_record)
}

fn decode_checkpoint(
    request: &StructuralTofMultiBankInput,
    wire: WireCheckpoint,
) -> Result<StructuralTofMultiBankCheckpoint, PersistenceError> {
    let mut accepted = request.clone();
    let mut definition = accepted.phase.definition().clone();
    definition.cell = cell_from_values(wire.phase.cell);
    definition.fractional_xyz = wire.phase.fractional_xyz;
    definition.occupancy = wire.phase.occupancy;
    definition.u_iso_angstrom2 = wire.phase.u_iso_angstrom2;
    accepted.phase = RietveldPhase::new_with_site_ids(
        accepted.phase.phase_id().clone(),
        accepted.phase.name(),
        accepted.phase.site_ids().to_vec(),
        definition,
        OwnedCwContributions::neutral(accepted.phase.definition().hkl.len()),
    )
    .map_err(workflow_record)?;
    if wire.banks.len() != accepted.banks.len() {
        return Err(invalid_record(
            "structural TOF checkpoint bank count does not match request",
        ));
    }
    for (bank, saved) in accepted.banks.iter_mut().zip(wire.banks) {
        if bank.bank_id.as_str() != saved.bank_id {
            return Err(invalid_record(
                "structural TOF checkpoint bank order does not match request",
            ));
        }
        let values: [f64; phasesmith_core::TOF_GLOBAL_PARAMETER_COUNT] =
            saved.instrument.try_into().map_err(|_| {
                invalid_record("structural TOF checkpoint instrument length is invalid")
            })?;
        bank.instrument = TofInstrument::from_values(values).map_err(workflow_record)?;
        bank.scale = saved.scale;
        match (&bank.background, saved.background_coefficients) {
            (Some(background), Some(coefficients)) => {
                bank.background = Some(
                    TofChebyshevBackground::new(
                        background.background_id().clone(),
                        coefficients,
                        background.domain_us(),
                    )
                    .map_err(workflow_record)?,
                );
            }
            (None, None) => {}
            _ => {
                return Err(invalid_record(
                    "structural TOF checkpoint background contract does not match request",
                ));
            }
        }
    }
    accepted.validate().map_err(workflow_record)?;
    let parameters = ParameterSet::new(
        wire.parameters
            .into_iter()
            .map(|parameter| {
                ParameterSpec::new(
                    ParameterKey::new(parameter.module, parameter.owner_id, parameter.name)
                        .map_err(workflow_record)?,
                    parameter.value,
                    parameter.unit,
                    phasesmith_workflows::ParameterBounds::new(
                        parameter.bounds[0].unwrap_or(f64::NEG_INFINITY),
                        parameter.bounds[1].unwrap_or(f64::INFINITY),
                    )
                    .map_err(workflow_record)?,
                    parameter.scale,
                    parameter.refine,
                )
                .map_err(workflow_record)
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
    )
    .map_err(workflow_record)?;
    let history = wire
        .history
        .into_iter()
        .map(decode_iteration)
        .collect::<Result<Vec<_>, _>>()?;
    let checkpoint = StructuralTofMultiBankCheckpoint {
        request: request.clone(),
        input: accepted,
        parameters,
        objective: wire.objective,
        damping: wire.damping,
        history,
    };
    checkpoint.validate_for(request).map_err(workflow_record)?;
    Ok(checkpoint)
}

fn decode_iteration(
    wire: WireIteration,
) -> Result<StructuralTofMultiBankIterationRecord, PersistenceError> {
    Ok(StructuralTofMultiBankIterationRecord {
        iteration: wire.iteration,
        objective: wire.objective,
        objective_change: wire.objective_change,
        scaled_step_norm: wire.scaled_step_norm,
        damping: wire.damping,
        backtracks: wire.backtracks,
        parameter_changes: wire
            .parameter_changes
            .into_iter()
            .map(|change| {
                Ok(ParameterChange {
                    key: ParameterKey::new(change.module, change.owner_id, change.name)
                        .map_err(workflow_record)?,
                    before: change.before,
                    after: change.after,
                    scaled_change: change.scaled_change,
                })
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
    })
}

const fn cell_values(cell: UnitCell) -> [f64; 6] {
    [
        cell.a_angstrom,
        cell.b_angstrom,
        cell.c_angstrom,
        cell.alpha_deg,
        cell.beta_deg,
        cell.gamma_deg,
    ]
}

const fn cell_from_values(values: [f64; 6]) -> UnitCell {
    UnitCell {
        a_angstrom: values[0],
        b_angstrom: values[1],
        c_angstrom: values[2],
        alpha_deg: values[3],
        beta_deg: values[4],
        gamma_deg: values[5],
    }
}

fn invalid_record(message: impl Into<String>) -> PersistenceError {
    PersistenceError::InvalidRecord {
        message: message.into(),
    }
}

fn domain_record(error: impl std::fmt::Display) -> PersistenceError {
    invalid_record(error.to_string())
}

fn workflow_record(error: impl std::fmt::Display) -> PersistenceError {
    invalid_record(error.to_string())
}
