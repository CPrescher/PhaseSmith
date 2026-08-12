//! Bounded bank-local instrument refinement over atomic multi-bank TOF Le Bail cycles.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    TOF_GLOBAL_PARAMETER_COUNT, TofError, TofInstrument, TofInstrumentParameter,
    TofProfileParameters,
};
use phasesmith_model::RecordId;

use crate::tof_lebail::{initialize_intensities, normal_tof_stop};
use crate::tof_multibank::{
    AcceptedBankState, MultiBankCycleCandidate, MultiBankCycleOutcome, aggregate_metrics,
    calculate_states, evaluate_bank_metrics, fitted_parameter_count, prepare_multibank_cycle,
    reflection_intensities,
};
use crate::{
    DiagnosticValue, RefinementEventKind, RefinementLimits, RefinementRuntime, ResidualEvaluation,
    RuntimeError, TerminationReason, TofChebyshevBackground, TofLeBailError, TofLeBailOptions,
    TofLeBailPhase, TofMultiBankError, TofMultiBankInput, TofMultiBankMetrics,
    TofMultiBankResultBank,
};

/// Closed physical bounds for one selected bank-local instrument coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TofInstrumentParameterBound {
    /// Coefficient matching one fused dense global derivative row.
    pub parameter: TofInstrumentParameter,
    /// Inclusive physical lower bound.
    pub lower: f64,
    /// Inclusive physical upper bound.
    pub upper: f64,
}

impl TofInstrumentParameterBound {
    /// Construct one finite non-degenerate physical interval.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError::InvalidModel`] for invalid bounds.
    pub fn new(
        parameter: TofInstrumentParameter,
        lower: f64,
        upper: f64,
    ) -> Result<Self, TofMultiBankInstrumentError> {
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(TofMultiBankInstrumentError::InvalidModel(
                "TOF instrument bounds must be finite and increasing",
            ));
        }
        Ok(Self {
            parameter,
            lower,
            upper,
        })
    }
}

/// Selected bank-local coefficients in deterministic solver order.
#[derive(Clone, Debug, PartialEq)]
pub struct TofBankInstrumentModel {
    bank_id: RecordId,
    bounds: Vec<TofInstrumentParameterBound>,
}

impl TofBankInstrumentModel {
    /// Construct one non-empty bank-local coefficient selection.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError::InvalidModel`] for empty or
    /// duplicate selections.
    pub fn new(
        bank_id: RecordId,
        bounds: Vec<TofInstrumentParameterBound>,
    ) -> Result<Self, TofMultiBankInstrumentError> {
        let result = Self { bank_id, bounds };
        result.validate_selection()?;
        Ok(result)
    }

    /// Stable bank identity.
    #[must_use]
    pub const fn bank_id(&self) -> &RecordId {
        &self.bank_id
    }

    /// Selected coefficients and physical bounds in solver order.
    #[must_use]
    pub fn bounds(&self) -> &[TofInstrumentParameterBound] {
        &self.bounds
    }

    fn validate_selection(&self) -> Result<(), TofMultiBankInstrumentError> {
        if self.bounds.is_empty() {
            return Err(TofMultiBankInstrumentError::InvalidModel(
                "bank-local TOF instrument selection must not be empty",
            ));
        }
        let mut selected = BTreeSet::new();
        for bound in &self.bounds {
            if !bound.lower.is_finite() || !bound.upper.is_finite() || bound.lower >= bound.upper {
                return Err(TofMultiBankInstrumentError::InvalidModel(
                    "TOF instrument bounds must be finite and increasing",
                ));
            }
            if !selected.insert(bound.parameter) {
                return Err(TofMultiBankInstrumentError::InvalidModel(
                    "bank-local TOF instrument parameters must be unique",
                ));
            }
        }
        Ok(())
    }
}

/// Multi-bank observations plus selected local instrument coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentInput {
    /// Atomic fixed-cell bank-local TOF request.
    pub multibank: TofMultiBankInput,
    /// Selected banks in deterministic parameter packing order.
    pub instrument_models: Vec<TofBankInstrumentModel>,
}

impl TofMultiBankInstrumentInput {
    /// Validate bank identities, selections, bounds, and initial values.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError`] for invalid bank-local state.
    pub fn validate(&self) -> Result<(), TofMultiBankInstrumentError> {
        self.multibank.validate()?;
        if self.instrument_models.is_empty() {
            return Err(TofMultiBankInstrumentError::InvalidModel(
                "multi-bank TOF instrument refinement requires at least one selected bank",
            ));
        }
        let mut bank_ids = BTreeSet::new();
        for model in &self.instrument_models {
            model.validate_selection()?;
            if !bank_ids.insert(model.bank_id.clone()) {
                return Err(TofMultiBankInstrumentError::InvalidModel(
                    "bank-local TOF instrument model IDs must be unique",
                ));
            }
            let bank = self
                .multibank
                .banks
                .iter()
                .find(|bank| &bank.bank_id == model.bank_id())
                .ok_or(TofMultiBankInstrumentError::InvalidModel(
                    "selected TOF instrument bank is absent from the request",
                ))?;
            let values = bank.input.instrument.values();
            if model.bounds.iter().any(|bound| {
                let value = values[bound.parameter.index()];
                value < bound.lower || value > bound.upper
            }) {
                return Err(TofMultiBankInstrumentError::InvalidModel(
                    "initial TOF instrument coefficient lies outside its declared bounds",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn parameter_count(&self) -> Result<usize, TofMultiBankInstrumentError> {
        self.instrument_models
            .iter()
            .try_fold(0_usize, |count, model| {
                count
                    .checked_add(model.bounds.len())
                    .ok_or(TofMultiBankInstrumentError::AllocationOverflow)
            })
    }
}

/// Controls for alternating local Le Bail updates and one instrument step.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentOptions {
    /// Existing TOF redistribution, support, weighting, and execution controls.
    pub lebail: TofLeBailOptions,
    /// Non-negative diagonal regularization in scaled instrument coordinates.
    pub instrument_damping: f64,
    /// Maximum absolute coefficient step in scaled coordinates.
    pub max_scaled_instrument_step: f64,
    /// Number of objective backtracking halvings after the initial trial.
    pub max_instrument_backtracks: usize,
    /// Absolute weighted-column correlation reported as unresolved.
    pub unresolved_correlation: f64,
}

impl TofMultiBankInstrumentOptions {
    /// Construct validated bank-local instrument controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError::InvalidOptions`] for invalid controls.
    pub fn new(
        lebail: TofLeBailOptions,
        instrument_damping: f64,
        max_scaled_instrument_step: f64,
        max_instrument_backtracks: usize,
        unresolved_correlation: f64,
    ) -> Result<Self, TofMultiBankInstrumentError> {
        let result = Self {
            lebail,
            instrument_damping,
            max_scaled_instrument_step,
            max_instrument_backtracks,
            unresolved_correlation,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError::InvalidOptions`] for invalid fields.
    pub fn validate(&self) -> Result<(), TofMultiBankInstrumentError> {
        self.lebail.validate()?;
        if !self.instrument_damping.is_finite()
            || self.instrument_damping < 0.0
            || !self.max_scaled_instrument_step.is_finite()
            || self.max_scaled_instrument_step <= 0.0
            || !self.unresolved_correlation.is_finite()
            || !(0.0..=1.0).contains(&self.unresolved_correlation)
        {
            return Err(TofMultiBankInstrumentError::InvalidOptions);
        }
        Ok(())
    }
}

/// One accepted bank-local instrument.
#[derive(Clone, Debug, PartialEq)]
pub struct TofBankInstrumentState {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Accepted calibration/profile coefficients.
    pub instrument: TofInstrument,
}

/// One accepted physical bank-local coefficient change.
#[derive(Clone, Debug, PartialEq)]
pub struct TofInstrumentParameterChange {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Selected coefficient.
    pub parameter: TofInstrumentParameter,
    /// Physical value before the step.
    pub before: f64,
    /// Physical value after the step.
    pub after: f64,
    /// Change divided by the parameter scale.
    pub scaled_change: f64,
}

/// One unresolved weighted-Jacobian column pair.
#[derive(Clone, Debug, PartialEq)]
pub struct TofInstrumentCorrelation {
    /// Bank and coefficient for the first column.
    pub left_bank_id: RecordId,
    /// First selected coefficient.
    pub left_parameter: TofInstrumentParameter,
    /// Bank and coefficient for the second column.
    pub right_bank_id: RecordId,
    /// Second selected coefficient.
    pub right_parameter: TofInstrumentParameter,
    /// Normalized weighted-column dot product.
    pub correlation: f64,
}

/// Identifiability diagnostics for the final selected instrument system.
#[derive(Clone, Debug, PartialEq)]
pub struct TofInstrumentDiagnostics {
    /// Number of selected physical coefficients.
    pub parameter_count: usize,
    /// Numerical rank of the weighted Jacobian.
    pub jacobian_rank: usize,
    /// Largest finite absolute correlation between nonzero columns.
    pub maximum_absolute_correlation: Option<f64>,
    /// Pairs at or above the configured unresolved threshold.
    pub unresolved_correlations: Vec<TofInstrumentCorrelation>,
}

/// One atomically accepted extraction plus bank-local instrument cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentIterationRecord {
    /// One-based accepted cycle index.
    pub iteration: usize,
    /// Bank-local residual records in input order.
    pub bank_metrics: Vec<ResidualEvaluation>,
    /// Aggregate metrics with selected instrument coefficients counted once.
    pub metrics: TofMultiBankMetrics,
    /// Largest relative intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Largest absolute background-coefficient change.
    pub maximum_absolute_background_change: f64,
    /// Norm of the accepted instrument step in scaled coordinates.
    pub scaled_instrument_step_norm: f64,
    /// Physical changes in deterministic bank/selection order.
    pub instrument_parameter_changes: Vec<TofInstrumentParameterChange>,
}

/// One bank's accepted checkpoint state.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentCheckpointBank {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Accepted instrument.
    pub instrument: TofInstrument,
    /// Accepted extracted phase intensities.
    pub phases: Vec<TofLeBailPhase>,
    /// Accepted refinable background, when configured.
    pub background: Option<TofChebyshevBackground>,
}

/// Complete last-accepted state for exact instrument continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentCheckpoint {
    /// Number of atomically accepted cycles.
    pub completed_iterations: usize,
    /// Accepted bank-local instrument/intensity/background state.
    pub banks: Vec<TofMultiBankInstrumentCheckpointBank>,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankInstrumentIterationRecord>,
}

/// Complete result from analytical bank-local TOF instrument refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInstrumentResult {
    /// Final bank-local display states and fused derivative calculations.
    pub banks: Vec<TofMultiBankResultBank>,
    /// Accepted instruments in bank input order.
    pub instruments: Vec<TofBankInstrumentState>,
    /// Final aggregate residual metrics.
    pub metrics: TofMultiBankMetrics,
    /// Final selected-system identifiability diagnostics.
    pub diagnostics: TofInstrumentDiagnostics,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankInstrumentIterationRecord>,
    /// Stable bounded-runtime termination category.
    pub termination_reason: TerminationReason,
    /// Complete last accepted state.
    pub checkpoint: TofMultiBankInstrumentCheckpoint,
}

impl TofMultiBankInstrumentCheckpoint {
    /// Revalidate this continuation against the exact local/instrument contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankInstrumentError`] for stale identities, topology,
    /// bounds, backgrounds, instruments, or history.
    pub fn validate_for(
        &self,
        input: &TofMultiBankInstrumentInput,
        options: &TofMultiBankInstrumentOptions,
    ) -> Result<(), TofMultiBankInstrumentError> {
        input.validate()?;
        options.validate()?;
        if self.completed_iterations != self.history.len()
            || self.completed_iterations > options.lebail.cycles
            || self.banks.len() != input.multibank.banks.len()
        {
            return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
                "TOF instrument checkpoint counts differ from the request",
            ));
        }
        for (saved, original) in self.banks.iter().zip(&input.multibank.banks) {
            validate_checkpoint_bank(saved, original, input)?;
        }
        if self.history.iter().enumerate().any(|(index, record)| {
            record.iteration != index + 1
                || record.bank_metrics.len() != input.multibank.banks.len()
        }) {
            return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
                "TOF instrument checkpoint history is not contiguous and bank-aligned",
            ));
        }
        Ok(())
    }
}

/// Refine selected bank-local instrument coefficients against all TOF banks.
///
/// # Errors
///
/// Returns [`TofMultiBankInstrumentError`] for invalid contracts, calculation,
/// linear solve, or runtime state.
pub fn refine_tof_multibank_instrument(
    input: &TofMultiBankInstrumentInput,
    options: &TofMultiBankInstrumentOptions,
) -> Result<TofMultiBankInstrumentResult, TofMultiBankInstrumentError> {
    input.validate()?;
    options.validate()?;
    let evaluations_per_cycle = options
        .max_instrument_backtracks
        .checked_add(4)
        .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?;
    let max_evaluations = options
        .lebail
        .cycles
        .checked_mul(evaluations_per_cycle)
        .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?;
    let limits = RefinementLimits::new(options.lebail.cycles, max_evaluations, None, 1)?;
    let mut runtime = RefinementRuntime::new(limits, None)?;
    refine_tof_multibank_instrument_with_runtime(input, options, None, &mut runtime)
}

/// Refine selected bank-local instruments with cancellation and continuation.
///
/// Candidate instruments, intensities, and backgrounds publish only after the
/// complete joint cycle succeeds. A stop during backtracking discards the
/// entire in-progress cycle.
///
/// # Errors
///
/// Returns [`TofMultiBankInstrumentError`] for invalid contracts, calculation,
/// linear solve, or non-normal runtime failures.
#[allow(clippy::too_many_lines)]
pub fn refine_tof_multibank_instrument_with_runtime(
    input: &TofMultiBankInstrumentInput,
    options: &TofMultiBankInstrumentOptions,
    checkpoint: Option<&TofMultiBankInstrumentCheckpoint>,
    runtime: &mut RefinementRuntime<TofMultiBankInstrumentCheckpoint>,
) -> Result<TofMultiBankInstrumentResult, TofMultiBankInstrumentError> {
    input.validate()?;
    options.validate()?;
    let parameter_count = input.parameter_count()?;
    let restored = restore_state(input, options, checkpoint)?;
    let mut states = restored.states;
    let mut instruments = restored.instruments;
    let mut history = restored.history;
    if let Some(checkpoint) = checkpoint {
        runtime.resume_accepted(checkpoint.completed_iterations)?;
    }
    runtime.emit(
        RefinementEventKind::Start,
        "tof_multibank_instrument",
        "bank-local multi-bank TOF instrument refinement started",
        vec![
            (
                "bank_count".to_owned(),
                DiagnosticValue::Unsigned(input.multibank.banks.len() as u64),
            ),
            (
                "instrument_parameter_count".to_owned(),
                DiagnosticValue::Unsigned(parameter_count as u64),
            ),
        ],
    )?;
    let mut accepted_calculations = None;
    let mut termination = TerminationReason::MaxIterations;
    for iteration in restored.first_iteration..=options.lebail.cycles {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_tof_stop(error)?;
            break;
        }
        let live = live_input(input, &instruments)?;
        let candidate = match prepare_multibank_cycle(
            &live,
            &states,
            &options.lebail,
            parameter_count,
            runtime,
        )? {
            MultiBankCycleOutcome::Candidate(candidate) => candidate,
            MultiBankCycleOutcome::Stopped(reason) => {
                termination = reason;
                break;
            }
        };
        let updated = match instrument_update(input, options, &instruments, candidate, runtime)? {
            InstrumentUpdateOutcome::Candidate(candidate) => candidate,
            InstrumentUpdateOutcome::Stopped(reason) => {
                termination = reason;
                break;
            }
        };
        states = updated.cycle.states;
        instruments = updated.instruments;
        accepted_calculations = Some(updated.cycle.calculations.clone());
        history.push(TofMultiBankInstrumentIterationRecord {
            iteration,
            bank_metrics: updated.cycle.bank_metrics.clone(),
            metrics: updated.cycle.metrics,
            maximum_relative_intensity_change: updated.cycle.maximum_relative_intensity_change,
            maximum_absolute_background_change: updated.cycle.maximum_absolute_background_change,
            scaled_instrument_step_norm: updated.step_norm,
            instrument_parameter_changes: updated.changes,
        });
        let accepted = checkpoint_from_state(input, &states, &instruments, &history);
        runtime.accept_step(Some(&accepted))?;
        runtime.emit(
            RefinementEventKind::Iteration,
            "tof_multibank_instrument_iteration",
            "bank-local multi-bank TOF instrument cycle accepted",
            vec![
                (
                    "rwp".to_owned(),
                    DiagnosticValue::Float(updated.cycle.metrics.rwp),
                ),
                (
                    "scaled_instrument_step_norm".to_owned(),
                    DiagnosticValue::Float(updated.step_norm),
                ),
            ],
        )?;
    }
    let live = live_input(input, &instruments)?;
    let calculations = match accepted_calculations {
        Some(calculations) => calculations,
        None => calculate_states(&live, &states, &options.lebail)?,
    };
    let bank_metrics = evaluate_bank_metrics(&live, &states, &calculations, &options.lebail)?;
    let fitted = fitted_parameter_count(&states)?
        .checked_add(parameter_count)
        .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?;
    let metrics = aggregate_metrics(&live, &calculations, &options.lebail, fitted)?;
    let packed = pack_instruments(input, &instruments)?;
    let (jacobian, _) =
        instrument_system(input, &live, &calculations, &packed.scales, &options.lebail)?;
    let diagnostics = instrument_diagnostics(input, &jacobian, options.unresolved_correlation)?;
    let checkpoint = checkpoint_from_state(input, &states, &instruments, &history);
    checkpoint.validate_for(input, options)?;
    let banks = live
        .banks
        .iter()
        .zip(states)
        .zip(calculations)
        .zip(bank_metrics)
        .map(
            |(((bank, state), calculation), metrics)| TofMultiBankResultBank {
                bank_id: bank.bank_id.clone(),
                calculation,
                metrics,
                intensities: reflection_intensities(&state.phases),
                phases: state.phases,
                background: state.background,
            },
        )
        .collect();
    let instruments = instrument_states(input, &instruments);
    runtime.emit(
        RefinementEventKind::Termination,
        "tof_multibank_instrument",
        "bank-local multi-bank TOF instrument refinement terminated",
        vec![
            (
                "termination_reason".to_owned(),
                DiagnosticValue::String(termination.as_str().to_owned()),
            ),
            (
                "jacobian_rank".to_owned(),
                DiagnosticValue::Unsigned(diagnostics.jacobian_rank as u64),
            ),
        ],
    )?;
    Ok(TofMultiBankInstrumentResult {
        banks,
        instruments,
        metrics,
        diagnostics,
        history,
        termination_reason: termination,
        checkpoint,
    })
}

struct RestoredState {
    states: Vec<AcceptedBankState>,
    instruments: Vec<TofInstrument>,
    history: Vec<TofMultiBankInstrumentIterationRecord>,
    first_iteration: usize,
}

fn restore_state(
    input: &TofMultiBankInstrumentInput,
    options: &TofMultiBankInstrumentOptions,
    checkpoint: Option<&TofMultiBankInstrumentCheckpoint>,
) -> Result<RestoredState, TofMultiBankInstrumentError> {
    let Some(checkpoint) = checkpoint else {
        let states = input
            .multibank
            .banks
            .iter()
            .map(|bank| {
                Ok(AcceptedBankState {
                    phases: initialize_intensities(&bank.input, &options.lebail)?,
                    background: bank.input.background.clone(),
                })
            })
            .collect::<Result<Vec<_>, TofMultiBankInstrumentError>>()?;
        return Ok(RestoredState {
            states,
            instruments: input
                .multibank
                .banks
                .iter()
                .map(|bank| bank.input.instrument)
                .collect(),
            history: Vec::new(),
            first_iteration: 1,
        });
    };
    checkpoint.validate_for(input, options)?;
    Ok(RestoredState {
        states: checkpoint
            .banks
            .iter()
            .map(|bank| AcceptedBankState {
                phases: bank.phases.clone(),
                background: bank.background.clone(),
            })
            .collect(),
        instruments: checkpoint
            .banks
            .iter()
            .map(|bank| bank.instrument)
            .collect(),
        history: checkpoint.history.clone(),
        first_iteration: checkpoint
            .completed_iterations
            .checked_add(1)
            .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?,
    })
}

struct InstrumentUpdate {
    cycle: MultiBankCycleCandidate,
    instruments: Vec<TofInstrument>,
    step_norm: f64,
    changes: Vec<TofInstrumentParameterChange>,
}

enum InstrumentUpdateOutcome {
    Candidate(InstrumentUpdate),
    Stopped(TerminationReason),
}

#[allow(clippy::too_many_lines)]
fn instrument_update<C>(
    input: &TofMultiBankInstrumentInput,
    options: &TofMultiBankInstrumentOptions,
    instruments: &[TofInstrument],
    mut cycle: MultiBankCycleCandidate,
    runtime: &mut RefinementRuntime<C>,
) -> Result<InstrumentUpdateOutcome, TofMultiBankInstrumentError> {
    let packed = pack_instruments(input, instruments)?;
    let current = live_input(input, instruments)?;
    let (jacobian, residual) = instrument_system(
        input,
        &current,
        &cycle.calculations,
        &packed.scales,
        &options.lebail,
    )?;
    let normal = jacobian.transpose() * &jacobian;
    let rhs = jacobian.transpose() * &residual;
    let mut regularized = normal;
    for parameter in 0..regularized.nrows() {
        regularized[(parameter, parameter)] += options.instrument_damping;
    }
    let mut step = if let Some(solution) = regularized.lu().solve(&rhs) {
        solution
    } else {
        jacobian
            .svd(true, true)
            .solve(&residual, f64::EPSILON)
            .map_err(|_| TofMultiBankInstrumentError::LinearSolve)?
    };
    if step.iter().any(|value| !value.is_finite()) {
        return Err(TofMultiBankInstrumentError::LinearSolve);
    }
    for parameter in 0..step.len() {
        let current = packed.values[parameter] / packed.scales[parameter];
        let lower = (packed.lower[parameter] / packed.scales[parameter] - current)
            .max(-options.max_scaled_instrument_step);
        let upper = (packed.upper[parameter] / packed.scales[parameter] - current)
            .min(options.max_scaled_instrument_step);
        step[parameter] = step[parameter].clamp(lower, upper);
    }
    if step.norm() == 0.0 {
        return Ok(InstrumentUpdateOutcome::Candidate(InstrumentUpdate {
            cycle,
            instruments: instruments.to_vec(),
            step_norm: 0.0,
            changes: Vec::new(),
        }));
    }
    let parameter_count = input.parameter_count()?;
    let local_parameter_count = fitted_parameter_count(&cycle.states)?;
    let total_parameter_count = local_parameter_count
        .checked_add(parameter_count)
        .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?;
    let mut factor = 1.0;
    for _ in 0..=options.max_instrument_backtracks {
        let trial_values = packed
            .values
            .iter()
            .zip(&packed.scales)
            .zip(step.iter())
            .map(|((value, scale), step)| value + factor * scale * step)
            .collect::<Vec<_>>();
        let trial_instruments = match unpack_instruments(input, instruments, &trial_values) {
            Ok(instruments) => instruments,
            Err(error) if recoverable_trial_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error),
        };
        let trial_input = match live_input(input, &trial_instruments) {
            Ok(input) => input,
            Err(error) if recoverable_trial_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error),
        };
        if let Err(error) = runtime.begin_evaluation() {
            return Ok(InstrumentUpdateOutcome::Stopped(normal_tof_stop(error)?));
        }
        let trial_calculations =
            match calculate_states(&trial_input, &cycle.states, &options.lebail) {
                Ok(calculations) => calculations,
                Err(error) if recoverable_multibank_trial_error(&error) => {
                    factor *= 0.5;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
        let trial_metrics = aggregate_metrics(
            &trial_input,
            &trial_calculations,
            &options.lebail,
            total_parameter_count,
        )?;
        if trial_metrics.chi_square < cycle.metrics.chi_square {
            let trial_bank_metrics = evaluate_bank_metrics(
                &trial_input,
                &cycle.states,
                &trial_calculations,
                &options.lebail,
            )?;
            let changes = instrument_changes(input, &packed, &trial_values)?;
            cycle.calculations = trial_calculations;
            cycle.bank_metrics = trial_bank_metrics;
            cycle.metrics = trial_metrics;
            return Ok(InstrumentUpdateOutcome::Candidate(InstrumentUpdate {
                cycle,
                instruments: trial_instruments,
                step_norm: factor * step.norm(),
                changes,
            }));
        }
        factor *= 0.5;
    }
    Ok(InstrumentUpdateOutcome::Candidate(InstrumentUpdate {
        cycle,
        instruments: instruments.to_vec(),
        step_norm: 0.0,
        changes: Vec::new(),
    }))
}

fn recoverable_trial_error(error: &TofMultiBankInstrumentError) -> bool {
    matches!(
        error,
        TofMultiBankInstrumentError::Profile(_)
            | TofMultiBankInstrumentError::MultiBank(TofMultiBankError::LeBail(
                TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_)
            ))
    )
}

fn recoverable_multibank_trial_error(error: &TofMultiBankError) -> bool {
    matches!(
        error,
        TofMultiBankError::LeBail(TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_))
    )
}

pub(crate) struct PackedInstruments {
    pub(crate) values: Vec<f64>,
    pub(crate) scales: Vec<f64>,
    pub(crate) lower: Vec<f64>,
    pub(crate) upper: Vec<f64>,
}

pub(crate) fn pack_instruments(
    input: &TofMultiBankInstrumentInput,
    instruments: &[TofInstrument],
) -> Result<PackedInstruments, TofMultiBankInstrumentError> {
    if instruments.len() != input.multibank.banks.len() {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let mut values = Vec::new();
    let mut scales = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for model in &input.instrument_models {
        let bank_index = bank_index(input, model.bank_id())?;
        let instrument_values = instruments[bank_index].values();
        for bound in &model.bounds {
            let value = instrument_values[bound.parameter.index()];
            let half_span = 0.5 * (bound.upper - bound.lower);
            values.push(value);
            scales.push(value.abs().max(half_span).max(f64::EPSILON.sqrt()));
            lower.push(bound.lower);
            upper.push(bound.upper);
        }
    }
    Ok(PackedInstruments {
        values,
        scales,
        lower,
        upper,
    })
}

pub(crate) fn unpack_instruments(
    input: &TofMultiBankInstrumentInput,
    current: &[TofInstrument],
    values: &[f64],
) -> Result<Vec<TofInstrument>, TofMultiBankInstrumentError> {
    if current.len() != input.multibank.banks.len() || values.len() != input.parameter_count()? {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let mut result = current.to_vec();
    let mut offset = 0;
    for model in &input.instrument_models {
        let index = bank_index(input, model.bank_id())?;
        let mut instrument_values = result[index].values();
        for bound in &model.bounds {
            instrument_values[bound.parameter.index()] = values[offset];
            offset += 1;
        }
        result[index] = TofInstrument::from_values(instrument_values)?;
    }
    Ok(result)
}

pub(crate) fn live_input(
    input: &TofMultiBankInstrumentInput,
    instruments: &[TofInstrument],
) -> Result<TofMultiBankInput, TofMultiBankInstrumentError> {
    if instruments.len() != input.multibank.banks.len() {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let mut result = input.multibank.clone();
    for (bank, instrument) in result.banks.iter_mut().zip(instruments) {
        bank.input.instrument = *instrument;
    }
    result.validate()?;
    Ok(result)
}

fn bank_index(
    input: &TofMultiBankInstrumentInput,
    bank_id: &RecordId,
) -> Result<usize, TofMultiBankInstrumentError> {
    input
        .multibank
        .banks
        .iter()
        .position(|bank| &bank.bank_id == bank_id)
        .ok_or(TofMultiBankInstrumentError::InternalInvariant)
}

pub(crate) fn instrument_system(
    input: &TofMultiBankInstrumentInput,
    live: &TofMultiBankInput,
    calculations: &[crate::TofLeBailCalculation],
    scales: &[f64],
    options: &TofLeBailOptions,
) -> Result<(DMatrix<f64>, DVector<f64>), TofMultiBankInstrumentError> {
    let columns = input.parameter_count()?;
    if columns != scales.len()
        || live.banks.len() != input.multibank.banks.len()
        || calculations.len() != live.banks.len()
    {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let rows = live.banks.iter().try_fold(0_usize, |count, bank| {
        let included = bank.input.pattern.mask.as_ref().map_or_else(
            || bank.input.pattern.sample_count(),
            |mask| mask.iter().filter(|value| **value).count(),
        );
        count
            .checked_add(included)
            .ok_or(TofMultiBankInstrumentError::AllocationOverflow)
    })?;
    if rows == 0 {
        return Err(TofMultiBankInstrumentError::LinearSolve);
    }
    let mut selected_jacobian = DMatrix::zeros(rows, columns);
    let mut selected_residual = DVector::zeros(rows);
    let mut model_offsets = Vec::with_capacity(input.instrument_models.len());
    let mut packed_offset = 0_usize;
    for model in &input.instrument_models {
        model_offsets.push(packed_offset);
        packed_offset = packed_offset
            .checked_add(model.bounds.len())
            .ok_or(TofMultiBankInstrumentError::AllocationOverflow)?;
    }
    if packed_offset != columns {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let mut row_offset = 0;
    for (bank, calculation) in live.banks.iter().zip(calculations) {
        let samples = bank.input.pattern.sample_count();
        let global = calculation
            .accumulation
            .derivatives
            .global
            .as_ref()
            .ok_or(TofMultiBankInstrumentError::InternalInvariant)?;
        if global.parameter_count != TOF_GLOBAL_PARAMETER_COUNT
            || global.sample_count != samples
            || global.values.len() != TOF_GLOBAL_PARAMETER_COUNT * samples
        {
            return Err(TofMultiBankInstrumentError::InternalInvariant);
        }
        let selected = input
            .instrument_models
            .iter()
            .position(|model| model.bank_id() == &bank.bank_id);
        let observed = bank
            .input
            .pattern
            .observed_y
            .as_ref()
            .ok_or(TofLeBailError::MissingObservations)?;
        for sample in 0..samples {
            if bank
                .input
                .pattern
                .mask
                .as_ref()
                .is_some_and(|mask| !mask[sample])
            {
                continue;
            }
            let weight = if options.use_uncertainty {
                bank.input
                    .pattern
                    .uncertainty
                    .as_ref()
                    .map_or(1.0, |sigma| sigma[sample].recip())
            } else {
                1.0
            };
            selected_residual[row_offset] = (observed[sample] - calculation.y[sample]) * weight;
            if let Some(model_index) = selected {
                let model = &input.instrument_models[model_index];
                let offset = model_offsets[model_index];
                for (local, bound) in model.bounds.iter().enumerate() {
                    selected_jacobian[(row_offset, offset + local)] = global.values
                        [bound.parameter.index() * samples + sample]
                        * scales[offset + local]
                        * weight;
                }
            }
            row_offset += 1;
        }
    }
    if row_offset != rows {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    Ok((selected_jacobian, selected_residual))
}

fn instrument_changes(
    input: &TofMultiBankInstrumentInput,
    packed: &PackedInstruments,
    after: &[f64],
) -> Result<Vec<TofInstrumentParameterChange>, TofMultiBankInstrumentError> {
    if packed.values.len() != after.len() || packed.values.len() != packed.scales.len() {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let mut changes = Vec::new();
    let mut offset = 0;
    for model in &input.instrument_models {
        for bound in &model.bounds {
            if packed.values[offset].to_bits() != after[offset].to_bits() {
                changes.push(TofInstrumentParameterChange {
                    bank_id: model.bank_id.clone(),
                    parameter: bound.parameter,
                    before: packed.values[offset],
                    after: after[offset],
                    scaled_change: (after[offset] - packed.values[offset]) / packed.scales[offset],
                });
            }
            offset += 1;
        }
    }
    Ok(changes)
}

fn instrument_diagnostics(
    input: &TofMultiBankInstrumentInput,
    jacobian: &DMatrix<f64>,
    threshold: f64,
) -> Result<TofInstrumentDiagnostics, TofMultiBankInstrumentError> {
    let parameter_count = input.parameter_count()?;
    if jacobian.ncols() != parameter_count {
        return Err(TofMultiBankInstrumentError::InternalInvariant);
    }
    let singular = jacobian.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0_f64, f64::max);
    let dimension = u32::try_from(jacobian.nrows().max(jacobian.ncols())).unwrap_or(u32::MAX);
    let tolerance = f64::from(dimension) * f64::EPSILON * maximum;
    let jacobian_rank = singular.iter().filter(|value| **value > tolerance).count();
    let keys = instrument_keys(input);
    let norms = (0..parameter_count)
        .map(|column| jacobian.column(column).norm())
        .collect::<Vec<_>>();
    let mut maximum_absolute_correlation = None::<f64>;
    let mut unresolved_correlations = Vec::new();
    for left in 0..parameter_count {
        if norms[left] == 0.0 {
            continue;
        }
        for right in left + 1..parameter_count {
            if norms[right] == 0.0 {
                continue;
            }
            let correlation = (jacobian.column(left).dot(&jacobian.column(right))
                / (norms[left] * norms[right]))
                .clamp(-1.0, 1.0);
            let absolute = correlation.abs();
            maximum_absolute_correlation =
                Some(maximum_absolute_correlation.map_or(absolute, |value| value.max(absolute)));
            if absolute >= threshold {
                unresolved_correlations.push(TofInstrumentCorrelation {
                    left_bank_id: keys[left].0.clone(),
                    left_parameter: keys[left].1,
                    right_bank_id: keys[right].0.clone(),
                    right_parameter: keys[right].1,
                    correlation,
                });
            }
        }
    }
    Ok(TofInstrumentDiagnostics {
        parameter_count,
        jacobian_rank,
        maximum_absolute_correlation,
        unresolved_correlations,
    })
}

fn instrument_keys(input: &TofMultiBankInstrumentInput) -> Vec<(RecordId, TofInstrumentParameter)> {
    input
        .instrument_models
        .iter()
        .flat_map(|model| {
            model
                .bounds
                .iter()
                .map(|bound| (model.bank_id.clone(), bound.parameter))
        })
        .collect()
}

fn checkpoint_from_state(
    input: &TofMultiBankInstrumentInput,
    states: &[AcceptedBankState],
    instruments: &[TofInstrument],
    history: &[TofMultiBankInstrumentIterationRecord],
) -> TofMultiBankInstrumentCheckpoint {
    TofMultiBankInstrumentCheckpoint {
        completed_iterations: history.len(),
        banks: input
            .multibank
            .banks
            .iter()
            .zip(states)
            .zip(instruments)
            .map(
                |((bank, state), instrument)| TofMultiBankInstrumentCheckpointBank {
                    bank_id: bank.bank_id.clone(),
                    instrument: *instrument,
                    phases: state.phases.clone(),
                    background: state.background.clone(),
                },
            )
            .collect(),
        history: history.to_vec(),
    }
}

fn instrument_states(
    input: &TofMultiBankInstrumentInput,
    instruments: &[TofInstrument],
) -> Vec<TofBankInstrumentState> {
    input
        .multibank
        .banks
        .iter()
        .zip(instruments)
        .map(|(bank, instrument)| TofBankInstrumentState {
            bank_id: bank.bank_id.clone(),
            instrument: *instrument,
        })
        .collect()
}

fn validate_checkpoint_bank(
    saved: &TofMultiBankInstrumentCheckpointBank,
    original: &crate::TofLeBailBank,
    input: &TofMultiBankInstrumentInput,
) -> Result<(), TofMultiBankInstrumentError> {
    saved.instrument.validate()?;
    if saved.bank_id != original.bank_id || saved.phases.len() != original.input.phases.len() {
        return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
            "TOF instrument checkpoint bank identity or phase count changed",
        ));
    }
    for (saved_phase, original_phase) in saved.phases.iter().zip(&original.input.phases) {
        saved_phase.validate()?;
        if saved_phase.phase_id() != original_phase.phase_id()
            || saved_phase.name() != original_phase.name()
            || saved_phase.reflection_ids() != original_phase.reflection_ids()
            || saved_phase.hkl() != original_phase.hkl()
            || saved_phase.d_spacing_angstrom() != original_phase.d_spacing_angstrom()
            || saved_phase.scale().to_bits() != original_phase.scale().to_bits()
        {
            return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
                "TOF instrument checkpoint bank topology changed",
            ));
        }
        for d_spacing in saved_phase.d_spacing_angstrom() {
            TofProfileParameters::from_instrument(*d_spacing, saved.instrument)?;
        }
    }
    let selected = input
        .instrument_models
        .iter()
        .find(|model| model.bank_id() == &saved.bank_id);
    let saved_values = saved.instrument.values();
    let original_values = original.input.instrument.values();
    for parameter in TofInstrumentParameter::ALL {
        let bound = selected.and_then(|model| {
            model
                .bounds
                .iter()
                .find(|bound| bound.parameter == parameter)
        });
        match bound {
            Some(bound)
                if saved_values[parameter.index()] >= bound.lower
                    && saved_values[parameter.index()] <= bound.upper => {}
            Some(_) => {
                return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
                    "TOF instrument checkpoint coefficient lies outside its bounds",
                ));
            }
            None if saved_values[parameter.index()].to_bits()
                == original_values[parameter.index()].to_bits() => {}
            None => {
                return Err(TofMultiBankInstrumentError::InvalidCheckpoint(
                    "unselected TOF instrument coefficient changed in a checkpoint",
                ));
            }
        }
    }
    match (&saved.background, &original.input.background) {
        (None, None) => Ok(()),
        (Some(saved), Some(original))
            if saved.background_id() == original.background_id()
                && saved
                    .domain_us()
                    .iter()
                    .zip(original.domain_us())
                    .all(|(saved, original)| saved.to_bits() == original.to_bits())
                && saved.coefficients().len() == original.coefficients().len() =>
        {
            Ok(())
        }
        _ => Err(TofMultiBankInstrumentError::InvalidCheckpoint(
            "TOF instrument checkpoint background contract changed",
        )),
    }
}

/// Invalid bank-local TOF instrument request or numerical state.
#[derive(Debug)]
pub enum TofMultiBankInstrumentError {
    /// Bank-local or fixed-cell multi-bank contract failed.
    MultiBank(TofMultiBankError),
    /// TOF instrument/profile validation failed.
    Profile(TofError),
    /// Bank/parameter selection is inconsistent.
    InvalidModel(&'static str),
    /// Solver controls are invalid.
    InvalidOptions,
    /// Weighted instrument normal equations could not be solved finitely.
    LinearSolve,
    /// A continuation state disagrees with the immutable request contract.
    InvalidCheckpoint(&'static str),
    /// Checked allocation arithmetic overflowed.
    AllocationOverflow,
    /// Bounded runtime or event/checkpoint delivery failed.
    Runtime(RuntimeError),
    /// Internal array alignment failed after validated construction.
    InternalInvariant,
}

impl Display for TofMultiBankInstrumentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultiBank(error) => Display::fmt(error, formatter),
            Self::Profile(error) => Display::fmt(error, formatter),
            Self::InvalidModel(message) | Self::InvalidCheckpoint(message) => {
                formatter.write_str(message)
            }
            Self::InvalidOptions => {
                formatter.write_str("invalid multi-bank TOF instrument options")
            }
            Self::LinearSolve => {
                formatter.write_str("multi-bank TOF instrument linear solve failed")
            }
            Self::AllocationOverflow => {
                formatter.write_str("multi-bank TOF instrument allocation overflow")
            }
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::InternalInvariant => {
                formatter.write_str("multi-bank TOF instrument internal array invariant failed")
            }
        }
    }
}

impl Error for TofMultiBankInstrumentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MultiBank(error) => Some(error),
            Self::Profile(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TofMultiBankError> for TofMultiBankInstrumentError {
    fn from(value: TofMultiBankError) -> Self {
        Self::MultiBank(value)
    }
}

impl From<TofLeBailError> for TofMultiBankInstrumentError {
    fn from(value: TofLeBailError) -> Self {
        Self::MultiBank(TofMultiBankError::LeBail(value))
    }
}

impl From<TofError> for TofMultiBankInstrumentError {
    fn from(value: TofError) -> Self {
        Self::Profile(value)
    }
}

impl From<RuntimeError> for TofMultiBankInstrumentError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
