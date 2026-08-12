//! Explicit wire records for resumable joint multi-bank TOF geometry analyses.

use std::collections::BTreeMap;

use phasesmith_core::{TOF_GLOBAL_PARAMETER_COUNT, TofInstrument, TofInstrumentParameter};
use phasesmith_engine::crystallography::UnitCell;
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_workflows::{
    LatticeBounds, LatticeParameterization, ResidualEvaluation, TofBankInstrumentModel,
    TofChebyshevBackground, TofInstrumentParameterBound, TofInstrumentParameterChange,
    TofLatticeParameterChange, TofLeBailBank, TofLeBailInput, TofLeBailOptions, TofLeBailPhase,
    TofMultiBankGeometryAnalysis, TofMultiBankGeometryCheckpoint,
    TofMultiBankGeometryCheckpointBank, TofMultiBankGeometryInput,
    TofMultiBankGeometryIterationRecord, TofMultiBankGeometryOptions,
    TofMultiBankGeometryProjectState, TofMultiBankInput, TofMultiBankLatticeInput,
    TofMultiBankMetrics, TofSharedLatticePhase, TofSharedLatticeState,
};
use serde::{Deserialize, Serialize};

use crate::arrays::ArrayData;
use crate::wire::{ArrayReference, take_bool, take_f64, take_i32};
use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTofMultiBankGeometryAnalysis {
    analysis_id: String,
    banks: Vec<WireBank>,
    lattice_phases: Vec<WireLatticePhase>,
    instrument_models: Vec<WireInstrumentModel>,
    options: WireGeometryOptions,
    checkpoint: Option<WireGeometryCheckpoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBank {
    bank_id: String,
    phases: Vec<WirePhase>,
    background: Option<WireBackground>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePhase {
    phase_id: String,
    name: String,
    reflection_ids: Vec<String>,
    hkl: ArrayReference,
    d_spacing_angstrom: ArrayReference,
    integrated_intensity: ArrayReference,
    scale: f64,
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
struct WireLatticePhase {
    phase_id: String,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstrumentModel {
    bank_id: String,
    bounds: Vec<WireInstrumentBound>,
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
struct WireGeometryOptions {
    cycles: usize,
    redistribution_damping: f64,
    initial_intensity_floor: f64,
    minimum_calculated: f64,
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    redistribution_use_uncertainty: bool,
    requested_threads: Option<usize>,
    minimum_parallel_tasks: usize,
    geometry_damping: f64,
    max_scaled_geometry_step: f64,
    max_geometry_backtracks: usize,
    unresolved_correlation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryCheckpoint {
    completed_iterations: usize,
    banks: Vec<WireCheckpointBank>,
    lattice_phases: Vec<WireLatticeState>,
    history: Vec<WireGeometryIteration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCheckpointBank {
    bank_id: String,
    instrument: Vec<f64>,
    phases: Vec<WirePhase>,
    background: Option<WireBackground>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLatticeState {
    phase_id: String,
    cell: [f64; 6],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGeometryIteration {
    iteration: usize,
    bank_metrics: Vec<WireResidual>,
    metrics: WireAggregateMetrics,
    maximum_relative_intensity_change: f64,
    maximum_absolute_background_change: f64,
    scaled_geometry_step_norm: f64,
    lattice_parameter_changes: Vec<WireLatticeChange>,
    instrument_parameter_changes: Vec<WireInstrumentChange>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResidual {
    included: ArrayReference,
    residual: ArrayReference,
    weighted_residual: ArrayReference,
    rp: f64,
    rwp: f64,
    chi_square: f64,
    reduced_chi_square: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAggregateMetrics {
    included_samples: usize,
    rp: f64,
    rwp: f64,
    chi_square: f64,
    reduced_chi_square: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLatticeChange {
    phase_id: String,
    parameter_name: String,
    before: f64,
    after: f64,
    scaled_change: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstrumentChange {
    bank_id: String,
    parameter: String,
    before: f64,
    after: f64,
    scaled_change: f64,
}

pub(crate) fn encode_analyses(
    state: &TofMultiBankGeometryProjectState,
) -> Result<
    (
        Vec<WireTofMultiBankGeometryAnalysis>,
        BTreeMap<String, ArrayData>,
    ),
    PersistenceError,
> {
    let mut arrays = BTreeMap::new();
    let analyses = state
        .analyses
        .iter()
        .map(|analysis| encode_analysis(analysis, &mut arrays))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((analyses, arrays))
}

fn encode_analysis(
    analysis: &TofMultiBankGeometryAnalysis,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireTofMultiBankGeometryAnalysis, PersistenceError> {
    let prefix = format!("tof_multibank.{}", analysis.analysis_id.as_str());
    let banks = analysis
        .input
        .lattice
        .multibank
        .banks
        .iter()
        .enumerate()
        .map(|(index, bank)| encode_bank(bank, &format!("{prefix}.bank.{index}"), arrays))
        .collect::<Result<Vec<_>, _>>()?;
    let lattice_phases = analysis
        .input
        .lattice
        .lattice_phases
        .iter()
        .map(|phase| WireLatticePhase {
            phase_id: phase.phase_id().as_str().to_owned(),
            lower: phase.bounds().lower().to_vec(),
            upper: phase.bounds().upper().to_vec(),
        })
        .collect();
    let instrument_models = analysis
        .input
        .instrument_models
        .iter()
        .map(encode_instrument_model)
        .collect();
    let checkpoint = analysis
        .checkpoint
        .as_ref()
        .map(|checkpoint| encode_checkpoint(checkpoint, &prefix, arrays))
        .transpose()?;
    Ok(WireTofMultiBankGeometryAnalysis {
        analysis_id: analysis.analysis_id.as_str().to_owned(),
        banks,
        lattice_phases,
        instrument_models,
        options: encode_options(&analysis.options),
        checkpoint,
    })
}

fn encode_bank(
    bank: &TofLeBailBank,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireBank, PersistenceError> {
    Ok(WireBank {
        bank_id: bank.bank_id.as_str().to_owned(),
        phases: encode_phases(&bank.input.phases, prefix, arrays)?,
        background: bank.input.background.as_ref().map(encode_background),
    })
}

fn encode_phases(
    phases: &[TofLeBailPhase],
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<Vec<WirePhase>, PersistenceError> {
    phases
        .iter()
        .enumerate()
        .map(|(index, phase)| encode_phase(phase, &format!("{prefix}.phase.{index}"), arrays))
        .collect()
}

fn encode_phase(
    phase: &TofLeBailPhase,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WirePhase, PersistenceError> {
    let count = phase.reflection_ids().len();
    Ok(WirePhase {
        phase_id: phase.phase_id().as_str().to_owned(),
        name: phase.name().to_owned(),
        reflection_ids: phase.reflection_ids().to_vec(),
        hkl: add_array(
            arrays,
            format!("{prefix}.hkl"),
            ArrayData::i32(
                phase.hkl().iter().flatten().copied().collect(),
                vec![count, 3],
            )?,
        )?,
        d_spacing_angstrom: add_array(
            arrays,
            format!("{prefix}.d_spacing_angstrom"),
            ArrayData::f64(phase.d_spacing_angstrom().to_vec(), vec![count])?,
        )?,
        integrated_intensity: add_array(
            arrays,
            format!("{prefix}.integrated_intensity"),
            ArrayData::f64(phase.integrated_intensity().to_vec(), vec![count])?,
        )?,
        scale: phase.scale(),
    })
}

fn encode_background(background: &TofChebyshevBackground) -> WireBackground {
    WireBackground {
        background_id: background.background_id().as_str().to_owned(),
        coefficients: background.coefficients().to_vec(),
        domain_us: background.domain_us(),
    }
}

fn encode_instrument_model(model: &TofBankInstrumentModel) -> WireInstrumentModel {
    WireInstrumentModel {
        bank_id: model.bank_id().as_str().to_owned(),
        bounds: model
            .bounds()
            .iter()
            .map(|bound| WireInstrumentBound {
                parameter: bound.parameter.name().to_owned(),
                lower: bound.lower,
                upper: bound.upper,
            })
            .collect(),
    }
}

fn encode_options(options: &TofMultiBankGeometryOptions) -> WireGeometryOptions {
    WireGeometryOptions {
        cycles: options.lebail.cycles,
        redistribution_damping: options.lebail.redistribution_damping,
        initial_intensity_floor: options.lebail.initial_intensity_floor,
        minimum_calculated: options.lebail.minimum_calculated,
        support_fwhm: options.lebail.support_fwhm,
        tail_log: options.lebail.tail_log,
        use_uncertainty: options.lebail.use_uncertainty,
        redistribution_use_uncertainty: options.lebail.redistribution_use_uncertainty,
        requested_threads: options.lebail.execution.requested_threads(),
        minimum_parallel_tasks: options.lebail.execution.minimum_parallel_tasks(),
        geometry_damping: options.geometry_damping,
        max_scaled_geometry_step: options.max_scaled_geometry_step,
        max_geometry_backtracks: options.max_geometry_backtracks,
        unresolved_correlation: options.unresolved_correlation,
    }
}

fn encode_checkpoint(
    checkpoint: &TofMultiBankGeometryCheckpoint,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireGeometryCheckpoint, PersistenceError> {
    let banks = checkpoint
        .banks
        .iter()
        .enumerate()
        .map(|(index, bank)| {
            let bank_prefix = format!("{prefix}.checkpoint.bank.{index}");
            Ok(WireCheckpointBank {
                bank_id: bank.bank_id.as_str().to_owned(),
                instrument: bank.instrument.values().to_vec(),
                phases: encode_phases(&bank.phases, &bank_prefix, arrays)?,
                background: bank.background.as_ref().map(encode_background),
            })
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;
    let lattice_phases = checkpoint
        .lattice_phases
        .iter()
        .map(|state| WireLatticeState {
            phase_id: state.phase_id.as_str().to_owned(),
            cell: cell_values(state.cell),
        })
        .collect();
    let history = checkpoint
        .history
        .iter()
        .enumerate()
        .map(|(index, record)| {
            encode_iteration(record, &format!("{prefix}.history.{index}"), arrays)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WireGeometryCheckpoint {
        completed_iterations: checkpoint.completed_iterations,
        banks,
        lattice_phases,
        history,
    })
}

fn encode_iteration(
    record: &TofMultiBankGeometryIterationRecord,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireGeometryIteration, PersistenceError> {
    let bank_metrics = record
        .bank_metrics
        .iter()
        .enumerate()
        .map(|(index, metrics)| encode_residual(metrics, &format!("{prefix}.bank.{index}"), arrays))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WireGeometryIteration {
        iteration: record.iteration,
        bank_metrics,
        metrics: encode_aggregate(record.metrics),
        maximum_relative_intensity_change: record.maximum_relative_intensity_change,
        maximum_absolute_background_change: record.maximum_absolute_background_change,
        scaled_geometry_step_norm: record.scaled_geometry_step_norm,
        lattice_parameter_changes: record
            .lattice_parameter_changes
            .iter()
            .map(|change| WireLatticeChange {
                phase_id: change.phase_id.as_str().to_owned(),
                parameter_name: change.parameter_name.clone(),
                before: change.before,
                after: change.after,
                scaled_change: change.scaled_change,
            })
            .collect(),
        instrument_parameter_changes: record
            .instrument_parameter_changes
            .iter()
            .map(|change| WireInstrumentChange {
                bank_id: change.bank_id.as_str().to_owned(),
                parameter: change.parameter.name().to_owned(),
                before: change.before,
                after: change.after,
                scaled_change: change.scaled_change,
            })
            .collect(),
    })
}

fn encode_residual(
    metrics: &ResidualEvaluation,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireResidual, PersistenceError> {
    let count = metrics.included.len();
    if metrics.residual.len() != count
        || metrics.weighted_residual.len() != count
        || [
            metrics.rp,
            metrics.rwp,
            metrics.chi_square,
            metrics.reduced_chi_square,
        ]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("joint TOF history contains invalid metrics"));
    }
    Ok(WireResidual {
        included: add_array(
            arrays,
            format!("{prefix}.included"),
            ArrayData::bool(metrics.included.clone(), vec![count])?,
        )?,
        residual: add_array(
            arrays,
            format!("{prefix}.residual"),
            ArrayData::f64(metrics.residual.clone(), vec![count])?,
        )?,
        weighted_residual: add_array(
            arrays,
            format!("{prefix}.weighted_residual"),
            ArrayData::f64(metrics.weighted_residual.clone(), vec![count])?,
        )?,
        rp: metrics.rp,
        rwp: metrics.rwp,
        chi_square: metrics.chi_square,
        reduced_chi_square: metrics.reduced_chi_square,
    })
}

const fn encode_aggregate(metrics: TofMultiBankMetrics) -> WireAggregateMetrics {
    WireAggregateMetrics {
        included_samples: metrics.included_samples,
        rp: metrics.rp,
        rwp: metrics.rwp,
        chi_square: metrics.chi_square,
        reduced_chi_square: metrics.reduced_chi_square,
    }
}

pub(crate) fn decode_state(
    project: ProjectRecord,
    wires: Vec<WireTofMultiBankGeometryAnalysis>,
    arrays: &mut BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<TofMultiBankGeometryProjectState, PersistenceError> {
    if wires.len() > limits.max_histograms {
        return Err(PersistenceError::LimitExceeded {
            message: "project exceeds max_histograms joint TOF analyses".to_owned(),
        });
    }
    let analyses = wires
        .into_iter()
        .map(|wire| decode_analysis(wire, &project, arrays, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let state = TofMultiBankGeometryProjectState { project, analyses };
    state
        .validate()
        .map_err(|error| invalid(format!("invalid joint TOF project state: {error}")))?;
    Ok(state)
}

fn decode_analysis(
    wire: WireTofMultiBankGeometryAnalysis,
    project: &ProjectRecord,
    arrays: &mut BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<TofMultiBankGeometryAnalysis, PersistenceError> {
    if wire.banks.len() > limits.max_histograms || wire.lattice_phases.len() > limits.max_phases {
        return Err(PersistenceError::LimitExceeded {
            message: "joint TOF analysis exceeds project limits".to_owned(),
        });
    }
    let banks = wire
        .banks
        .into_iter()
        .map(|bank| decode_bank(bank, project, arrays, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let lattice_phases = wire
        .lattice_phases
        .into_iter()
        .map(|phase| decode_lattice_phase(phase, project))
        .collect::<Result<Vec<_>, _>>()?;
    let instrument_models = wire
        .instrument_models
        .into_iter()
        .map(decode_instrument_model)
        .collect::<Result<Vec<_>, _>>()?;
    let options = decode_options(wire.options)?;
    let input = TofMultiBankGeometryInput {
        lattice: TofMultiBankLatticeInput {
            multibank: TofMultiBankInput { banks },
            lattice_phases,
        },
        instrument_models,
    };
    let checkpoint = wire
        .checkpoint
        .map(|checkpoint| decode_checkpoint(checkpoint, &input, arrays))
        .transpose()?;
    Ok(TofMultiBankGeometryAnalysis {
        analysis_id: RecordId::new(wire.analysis_id).map_err(PersistenceError::Domain)?,
        input,
        options,
        checkpoint,
    })
}

fn decode_bank(
    wire: WireBank,
    project: &ProjectRecord,
    arrays: &mut BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<TofLeBailBank, PersistenceError> {
    if wire.phases.len() > limits.max_phases {
        return Err(PersistenceError::LimitExceeded {
            message: "joint TOF bank exceeds max_phases".to_owned(),
        });
    }
    let bank_id = RecordId::new(wire.bank_id).map_err(PersistenceError::Domain)?;
    let histogram = project
        .tof_histograms
        .iter()
        .find(|histogram| histogram.histogram_id == bank_id)
        .ok_or_else(|| invalid(format!("unknown joint TOF histogram {bank_id}")))?;
    let phases = wire
        .phases
        .into_iter()
        .map(|phase| decode_phase(phase, arrays))
        .collect::<Result<Vec<_>, _>>()?;
    let mut input = TofLeBailInput::new(
        histogram.pattern.clone(),
        histogram.experiment.instrument,
        phases,
    )
    .map_err(workflow)?;
    if let Some(background) = wire.background {
        input = input
            .with_refinable_background(decode_background(background)?)
            .map_err(workflow)?;
    }
    Ok(TofLeBailBank { bank_id, input })
}

fn decode_phase(
    wire: WirePhase,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<TofLeBailPhase, PersistenceError> {
    let count = wire.reflection_ids.len();
    let flat_hkl = take_i32(arrays, &wire.hkl, Some(&[count, 3]))?;
    let hkl = flat_hkl
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    TofLeBailPhase::new(
        RecordId::new(wire.phase_id).map_err(PersistenceError::Domain)?,
        wire.name,
        wire.reflection_ids,
        hkl,
        take_f64(arrays, &wire.d_spacing_angstrom, Some(&[count]))?,
        take_f64(arrays, &wire.integrated_intensity, Some(&[count]))?,
        wire.scale,
    )
    .map_err(workflow)
}

fn decode_background(wire: WireBackground) -> Result<TofChebyshevBackground, PersistenceError> {
    TofChebyshevBackground::new(
        RecordId::new(wire.background_id).map_err(PersistenceError::Domain)?,
        wire.coefficients,
        wire.domain_us,
    )
    .map_err(workflow)
}

fn decode_lattice_phase(
    wire: WireLatticePhase,
    project: &ProjectRecord,
) -> Result<TofSharedLatticePhase, PersistenceError> {
    let phase_id = RecordId::new(wire.phase_id).map_err(PersistenceError::Domain)?;
    let stored = project
        .phases
        .iter()
        .find(|phase| phase.phase_id == phase_id)
        .ok_or_else(|| invalid(format!("unknown joint TOF lattice phase {phase_id}")))?;
    let parameterization = LatticeParameterization::new(
        stored.definition.space_group.clone(),
        stored.definition.cell,
    )
    .map_err(workflow)?;
    let bounds = LatticeBounds::new(&parameterization, wire.lower, wire.upper).map_err(workflow)?;
    TofSharedLatticePhase::new(phase_id, parameterization, bounds, stored.definition.cell)
        .map_err(workflow)
}

fn decode_instrument_model(
    wire: WireInstrumentModel,
) -> Result<TofBankInstrumentModel, PersistenceError> {
    let bounds = wire
        .bounds
        .into_iter()
        .map(|bound| {
            TofInstrumentParameterBound::new(
                decode_parameter(&bound.parameter)?,
                bound.lower,
                bound.upper,
            )
            .map_err(workflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    TofBankInstrumentModel::new(
        RecordId::new(wire.bank_id).map_err(PersistenceError::Domain)?,
        bounds,
    )
    .map_err(workflow)
}

fn decode_options(
    wire: WireGeometryOptions,
) -> Result<TofMultiBankGeometryOptions, PersistenceError> {
    let execution = ExecutionPolicy::new(wire.requested_threads, wire.minimum_parallel_tasks)
        .map_err(workflow)?;
    let lebail = TofLeBailOptions::new(
        wire.cycles,
        wire.redistribution_damping,
        wire.initial_intensity_floor,
        wire.minimum_calculated,
        wire.support_fwhm,
        wire.tail_log,
        wire.use_uncertainty,
        execution,
    )
    .map_err(workflow)?
    .with_redistribution_uncertainty(wire.redistribution_use_uncertainty);
    TofMultiBankGeometryOptions::new(
        lebail,
        wire.geometry_damping,
        wire.max_scaled_geometry_step,
        wire.max_geometry_backtracks,
        wire.unresolved_correlation,
    )
    .map_err(workflow)
}

fn decode_checkpoint(
    wire: WireGeometryCheckpoint,
    input: &TofMultiBankGeometryInput,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<TofMultiBankGeometryCheckpoint, PersistenceError> {
    let banks = wire
        .banks
        .into_iter()
        .map(|bank| decode_checkpoint_bank(bank, arrays))
        .collect::<Result<Vec<_>, _>>()?;
    let lattice_phases = wire
        .lattice_phases
        .into_iter()
        .map(|state| {
            Ok(TofSharedLatticeState {
                phase_id: RecordId::new(state.phase_id).map_err(PersistenceError::Domain)?,
                cell: decode_cell(state.cell)?,
            })
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;
    let history = wire
        .history
        .into_iter()
        .map(|record| decode_iteration(record, input, arrays))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TofMultiBankGeometryCheckpoint {
        completed_iterations: wire.completed_iterations,
        banks,
        lattice_phases,
        history,
    })
}

fn decode_checkpoint_bank(
    wire: WireCheckpointBank,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<TofMultiBankGeometryCheckpointBank, PersistenceError> {
    let values: [f64; TOF_GLOBAL_PARAMETER_COUNT] = wire
        .instrument
        .try_into()
        .map_err(|_| invalid("joint TOF instrument must contain 15 coefficients"))?;
    Ok(TofMultiBankGeometryCheckpointBank {
        bank_id: RecordId::new(wire.bank_id).map_err(PersistenceError::Domain)?,
        instrument: TofInstrument::from_values(values).map_err(workflow)?,
        phases: wire
            .phases
            .into_iter()
            .map(|phase| decode_phase(phase, arrays))
            .collect::<Result<Vec<_>, _>>()?,
        background: wire.background.map(decode_background).transpose()?,
    })
}

fn decode_iteration(
    wire: WireGeometryIteration,
    input: &TofMultiBankGeometryInput,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<TofMultiBankGeometryIterationRecord, PersistenceError> {
    if wire.bank_metrics.len() != input.lattice.multibank.banks.len() {
        return Err(invalid("joint TOF history bank count differs from input"));
    }
    let bank_metrics = wire
        .bank_metrics
        .into_iter()
        .zip(&input.lattice.multibank.banks)
        .map(|(metrics, bank)| decode_residual(&metrics, arrays, bank.input.pattern.sample_count()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TofMultiBankGeometryIterationRecord {
        iteration: wire.iteration,
        bank_metrics,
        metrics: decode_aggregate(wire.metrics),
        maximum_relative_intensity_change: wire.maximum_relative_intensity_change,
        maximum_absolute_background_change: wire.maximum_absolute_background_change,
        scaled_geometry_step_norm: wire.scaled_geometry_step_norm,
        lattice_parameter_changes: wire
            .lattice_parameter_changes
            .into_iter()
            .map(|change| {
                Ok(TofLatticeParameterChange {
                    phase_id: RecordId::new(change.phase_id).map_err(PersistenceError::Domain)?,
                    parameter_name: change.parameter_name,
                    before: change.before,
                    after: change.after,
                    scaled_change: change.scaled_change,
                })
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
        instrument_parameter_changes: wire
            .instrument_parameter_changes
            .into_iter()
            .map(|change| {
                Ok(TofInstrumentParameterChange {
                    bank_id: RecordId::new(change.bank_id).map_err(PersistenceError::Domain)?,
                    parameter: decode_parameter(&change.parameter)?,
                    before: change.before,
                    after: change.after,
                    scaled_change: change.scaled_change,
                })
            })
            .collect::<Result<Vec<_>, PersistenceError>>()?,
    })
}

fn decode_residual(
    wire: &WireResidual,
    arrays: &mut BTreeMap<String, ArrayData>,
    sample_count: usize,
) -> Result<ResidualEvaluation, PersistenceError> {
    Ok(ResidualEvaluation {
        included: take_bool(arrays, &wire.included, &[sample_count])?,
        residual: take_f64(arrays, &wire.residual, Some(&[sample_count]))?,
        weighted_residual: take_f64(arrays, &wire.weighted_residual, Some(&[sample_count]))?,
        rp: wire.rp,
        rwp: wire.rwp,
        chi_square: wire.chi_square,
        reduced_chi_square: wire.reduced_chi_square,
    })
}

const fn decode_aggregate(wire: WireAggregateMetrics) -> TofMultiBankMetrics {
    TofMultiBankMetrics {
        included_samples: wire.included_samples,
        rp: wire.rp,
        rwp: wire.rwp,
        chi_square: wire.chi_square,
        reduced_chi_square: wire.reduced_chi_square,
    }
}

fn decode_parameter(name: &str) -> Result<TofInstrumentParameter, PersistenceError> {
    TofInstrumentParameter::ALL
        .into_iter()
        .find(|parameter| parameter.name() == name)
        .ok_or_else(|| invalid(format!("unknown TOF instrument parameter {name:?}")))
}

fn add_array(
    arrays: &mut BTreeMap<String, ArrayData>,
    name: String,
    value: ArrayData,
) -> Result<ArrayReference, PersistenceError> {
    if arrays.insert(name.clone(), value).is_some() {
        return Err(invalid(format!("duplicate wire array name {name:?}")));
    }
    Ok(ArrayReference { array: name })
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

fn decode_cell(values: [f64; 6]) -> Result<UnitCell, PersistenceError> {
    let cell = UnitCell {
        a_angstrom: values[0],
        b_angstrom: values[1],
        c_angstrom: values[2],
        alpha_deg: values[3],
        beta_deg: values[4],
        gamma_deg: values[5],
    };
    cell.geometry().map_err(workflow)?;
    Ok(cell)
}

fn workflow(error: impl std::fmt::Display) -> PersistenceError {
    invalid(format!("invalid joint TOF analysis record: {error}"))
}

fn invalid(message: impl Into<String>) -> PersistenceError {
    PersistenceError::InvalidRecord {
        message: message.into(),
    }
}
