//! Explicit wire records for resumable fixed-instrument TOF Le Bail analyses.

use std::collections::BTreeMap;

use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{ProjectRecord, RecordId};
use phasesmith_workflows::{
    ResidualEvaluation, TofChebyshevBackground, TofLeBailAnalysis, TofLeBailCheckpoint,
    TofLeBailInput, TofLeBailIterationRecord, TofLeBailOptions, TofLeBailPhase,
    TofLeBailProjectState,
};
use serde::{Deserialize, Serialize};

use crate::arrays::ArrayData;
use crate::wire::{ArrayReference, take_bool, take_f64, take_i32};
use crate::{PersistenceError, ProjectReadLimits};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireTofLeBailAnalysis {
    histogram_id: String,
    phases: Vec<WireTofPhase>,
    background: Option<WireTofBackground>,
    options: WireTofOptions,
    checkpoint: Option<WireTofCheckpoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTofPhase {
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
struct WireTofBackground {
    background_id: String,
    coefficients: Vec<f64>,
    domain_us: [f64; 2],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTofOptions {
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTofCheckpoint {
    completed_iterations: usize,
    phase_integrated_intensity: Vec<ArrayReference>,
    background_coefficients: Option<Vec<f64>>,
    history: Vec<WireTofIteration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTofIteration {
    iteration: usize,
    included: ArrayReference,
    residual: ArrayReference,
    weighted_residual: ArrayReference,
    rp: f64,
    rwp: f64,
    chi_square: f64,
    reduced_chi_square: f64,
    maximum_relative_intensity_change: f64,
    maximum_absolute_background_change: f64,
}

pub(crate) fn encode_analyses(
    state: &TofLeBailProjectState,
) -> Result<(Vec<WireTofLeBailAnalysis>, BTreeMap<String, ArrayData>), PersistenceError> {
    let mut arrays = BTreeMap::new();
    let analyses = state
        .analyses
        .iter()
        .map(|analysis| encode_analysis(analysis, &mut arrays))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((analyses, arrays))
}

fn encode_analysis(
    analysis: &TofLeBailAnalysis,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireTofLeBailAnalysis, PersistenceError> {
    let prefix = format!("tof_analysis.{}", analysis.histogram_id.as_str());
    let phases = analysis
        .input
        .phases
        .iter()
        .enumerate()
        .map(|(index, phase)| encode_phase(phase, &format!("{prefix}.phase.{index}"), arrays))
        .collect::<Result<Vec<_>, _>>()?;
    let checkpoint = analysis
        .checkpoint
        .as_ref()
        .map(|checkpoint| encode_checkpoint(checkpoint, &prefix, arrays))
        .transpose()?;
    Ok(WireTofLeBailAnalysis {
        histogram_id: analysis.histogram_id.as_str().to_owned(),
        phases,
        background: analysis.input.background.as_ref().map(encode_background),
        options: encode_options(&analysis.options),
        checkpoint,
    })
}

fn encode_phase(
    phase: &TofLeBailPhase,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireTofPhase, PersistenceError> {
    let count = phase.reflection_ids().len();
    Ok(WireTofPhase {
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

fn encode_background(background: &TofChebyshevBackground) -> WireTofBackground {
    WireTofBackground {
        background_id: background.background_id().as_str().to_owned(),
        coefficients: background.coefficients().to_vec(),
        domain_us: background.domain_us(),
    }
}

fn encode_options(options: &TofLeBailOptions) -> WireTofOptions {
    WireTofOptions {
        cycles: options.cycles,
        redistribution_damping: options.redistribution_damping,
        initial_intensity_floor: options.initial_intensity_floor,
        minimum_calculated: options.minimum_calculated,
        support_fwhm: options.support_fwhm,
        tail_log: options.tail_log,
        use_uncertainty: options.use_uncertainty,
        redistribution_use_uncertainty: options.redistribution_use_uncertainty,
        requested_threads: options.execution.requested_threads(),
        minimum_parallel_tasks: options.execution.minimum_parallel_tasks(),
    }
}

fn encode_checkpoint(
    checkpoint: &TofLeBailCheckpoint,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireTofCheckpoint, PersistenceError> {
    let phase_integrated_intensity = checkpoint
        .phases
        .iter()
        .enumerate()
        .map(|(index, phase)| {
            add_array(
                arrays,
                format!("{prefix}.checkpoint.phase.{index}.integrated_intensity"),
                ArrayData::f64(
                    phase.integrated_intensity().to_vec(),
                    vec![phase.integrated_intensity().len()],
                )?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let history = checkpoint
        .history
        .iter()
        .enumerate()
        .map(|(index, record)| {
            encode_iteration(record, &format!("{prefix}.history.{index}"), arrays)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WireTofCheckpoint {
        completed_iterations: checkpoint.completed_iterations,
        phase_integrated_intensity,
        background_coefficients: checkpoint
            .background
            .as_ref()
            .map(|background| background.coefficients().to_vec()),
        history,
    })
}

fn encode_iteration(
    record: &TofLeBailIterationRecord,
    prefix: &str,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<WireTofIteration, PersistenceError> {
    let count = record.metrics.included.len();
    if record.metrics.residual.len() != count
        || record.metrics.weighted_residual.len() != count
        || [
            record.metrics.rp,
            record.metrics.rwp,
            record.metrics.chi_square,
            record.metrics.reduced_chi_square,
            record.maximum_relative_intensity_change,
            record.maximum_absolute_background_change,
        ]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid_record(
            "TOF checkpoint history contains mismatched or non-finite metrics".to_owned(),
        ));
    }
    Ok(WireTofIteration {
        iteration: record.iteration,
        included: add_array(
            arrays,
            format!("{prefix}.included"),
            ArrayData::bool(record.metrics.included.clone(), vec![count])?,
        )?,
        residual: add_array(
            arrays,
            format!("{prefix}.residual"),
            ArrayData::f64(record.metrics.residual.clone(), vec![count])?,
        )?,
        weighted_residual: add_array(
            arrays,
            format!("{prefix}.weighted_residual"),
            ArrayData::f64(record.metrics.weighted_residual.clone(), vec![count])?,
        )?,
        rp: record.metrics.rp,
        rwp: record.metrics.rwp,
        chi_square: record.metrics.chi_square,
        reduced_chi_square: record.metrics.reduced_chi_square,
        maximum_relative_intensity_change: record.maximum_relative_intensity_change,
        maximum_absolute_background_change: record.maximum_absolute_background_change,
    })
}

fn add_array(
    arrays: &mut BTreeMap<String, ArrayData>,
    name: String,
    value: ArrayData,
) -> Result<ArrayReference, PersistenceError> {
    if arrays.insert(name.clone(), value).is_some() {
        return Err(invalid_record(format!(
            "duplicate wire array name {name:?}"
        )));
    }
    Ok(ArrayReference { array: name })
}

pub(crate) fn decode_state(
    project: ProjectRecord,
    wires: Vec<WireTofLeBailAnalysis>,
    arrays: &mut BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<TofLeBailProjectState, PersistenceError> {
    if wires.len() > limits.max_histograms {
        return Err(PersistenceError::LimitExceeded {
            message: "project exceeds max_histograms TOF analyses".to_owned(),
        });
    }
    let analyses = wires
        .into_iter()
        .map(|wire| decode_analysis(wire, &project, arrays, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let state = TofLeBailProjectState { project, analyses };
    state
        .validate()
        .map_err(|error| invalid_record(format!("invalid native TOF project state: {error}")))?;
    Ok(state)
}

fn decode_analysis(
    wire: WireTofLeBailAnalysis,
    project: &ProjectRecord,
    arrays: &mut BTreeMap<String, ArrayData>,
    limits: ProjectReadLimits,
) -> Result<TofLeBailAnalysis, PersistenceError> {
    if wire.phases.len() > limits.max_phases {
        return Err(PersistenceError::LimitExceeded {
            message: "TOF analysis exceeds max_phases".to_owned(),
        });
    }
    let histogram_id = RecordId::new(wire.histogram_id).map_err(PersistenceError::Domain)?;
    let histogram = project
        .tof_histograms
        .iter()
        .find(|histogram| histogram.histogram_id == histogram_id)
        .ok_or_else(|| invalid_record(format!("unknown TOF histogram {histogram_id}")))?;
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
    .map_err(workflow_error)?;
    if let Some(background) = wire.background {
        input = input
            .with_refinable_background(decode_background(background)?)
            .map_err(workflow_error)?;
    }
    let options = decode_options(wire.options)?;
    let checkpoint = wire
        .checkpoint
        .map(|checkpoint| decode_checkpoint(checkpoint, &input, arrays))
        .transpose()?;
    Ok(TofLeBailAnalysis {
        histogram_id,
        input,
        options,
        checkpoint,
    })
}

fn decode_phase(
    wire: WireTofPhase,
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
    .map_err(workflow_error)
}

fn decode_background(wire: WireTofBackground) -> Result<TofChebyshevBackground, PersistenceError> {
    TofChebyshevBackground::new(
        RecordId::new(wire.background_id).map_err(PersistenceError::Domain)?,
        wire.coefficients,
        wire.domain_us,
    )
    .map_err(workflow_error)
}

fn decode_options(wire: WireTofOptions) -> Result<TofLeBailOptions, PersistenceError> {
    let execution = ExecutionPolicy::new(wire.requested_threads, wire.minimum_parallel_tasks)
        .map_err(|error| invalid_record(format!("invalid TOF execution policy: {error}")))?;
    TofLeBailOptions::new(
        wire.cycles,
        wire.redistribution_damping,
        wire.initial_intensity_floor,
        wire.minimum_calculated,
        wire.support_fwhm,
        wire.tail_log,
        wire.use_uncertainty,
        execution,
    )
    .map(|options| options.with_redistribution_uncertainty(wire.redistribution_use_uncertainty))
    .map_err(workflow_error)
}

fn decode_checkpoint(
    wire: WireTofCheckpoint,
    input: &TofLeBailInput,
    arrays: &mut BTreeMap<String, ArrayData>,
) -> Result<TofLeBailCheckpoint, PersistenceError> {
    if wire.phase_integrated_intensity.len() != input.phases.len() {
        return Err(invalid_record(
            "TOF checkpoint phase intensity count differs from input".to_owned(),
        ));
    }
    let phases = input
        .phases
        .iter()
        .zip(wire.phase_integrated_intensity)
        .map(|(phase, reference)| {
            let count = phase.reflection_ids().len();
            TofLeBailPhase::new(
                phase.phase_id().clone(),
                phase.name(),
                phase.reflection_ids().to_vec(),
                phase.hkl().to_vec(),
                phase.d_spacing_angstrom().to_vec(),
                take_f64(arrays, &reference, Some(&[count]))?,
                phase.scale(),
            )
            .map_err(workflow_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let background = match (&input.background, wire.background_coefficients) {
        (None, None) => None,
        (Some(original), Some(coefficients)) => Some(
            TofChebyshevBackground::new(
                original.background_id().clone(),
                coefficients,
                original.domain_us(),
            )
            .map_err(workflow_error)?,
        ),
        _ => {
            return Err(invalid_record(
                "TOF checkpoint background presence differs from input".to_owned(),
            ));
        }
    };
    let sample_count = input.pattern.sample_count();
    let history = wire
        .history
        .into_iter()
        .map(|record| decode_iteration(&record, arrays, sample_count))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TofLeBailCheckpoint {
        completed_iterations: wire.completed_iterations,
        phases,
        background,
        history,
    })
}

fn decode_iteration(
    wire: &WireTofIteration,
    arrays: &mut BTreeMap<String, ArrayData>,
    sample_count: usize,
) -> Result<TofLeBailIterationRecord, PersistenceError> {
    Ok(TofLeBailIterationRecord {
        iteration: wire.iteration,
        metrics: ResidualEvaluation {
            included: take_bool(arrays, &wire.included, &[sample_count])?,
            residual: take_f64(arrays, &wire.residual, Some(&[sample_count]))?,
            weighted_residual: take_f64(arrays, &wire.weighted_residual, Some(&[sample_count]))?,
            rp: wire.rp,
            rwp: wire.rwp,
            chi_square: wire.chi_square,
            reduced_chi_square: wire.reduced_chi_square,
        },
        maximum_relative_intensity_change: wire.maximum_relative_intensity_change,
        maximum_absolute_background_change: wire.maximum_absolute_background_change,
    })
}

fn workflow_error(error: impl std::fmt::Display) -> PersistenceError {
    invalid_record(format!("invalid TOF analysis record: {error}"))
}

fn invalid_record(message: String) -> PersistenceError {
    PersistenceError::InvalidRecord { message }
}
