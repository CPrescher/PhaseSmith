//! Bounded accepted-state solver for the guarded structural TOF objective.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};

use crate::{
    CancellationToken, DiagnosticValue, ParameterChange, ParameterError, ParameterSet,
    PreparedStructuralTofMultiBankObjective, RefinementEventKind, RefinementLimits,
    RefinementRuntime, RuntimeError, StructuralTofMultiBankCalculation,
    StructuralTofMultiBankError, StructuralTofMultiBankInput, StructuralTofMultiBankLayout,
    TerminationReason,
};

/// Numerical and bounded-runtime controls for structural multi-bank TOF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructuralTofMultiBankRefinementOptions {
    /// Hard iteration/evaluation/time/rejection limits.
    pub limits: RefinementLimits,
    /// Minimum accepted iterations before objective convergence.
    pub min_iterations: usize,
    /// Relative accepted objective-change tolerance.
    pub objective_tolerance: f64,
    /// Scaled step-norm tolerance.
    pub parameter_tolerance: f64,
    /// Initial positive Levenberg damping in scaled coordinates.
    pub initial_damping: f64,
    /// Multiplier applied after an unsuccessful iteration.
    pub damping_increase: f64,
    /// Multiplier applied after an accepted iteration.
    pub damping_decrease: f64,
    /// Maximum Euclidean scaled step norm.
    pub max_scaled_parameter_step: f64,
    /// Number of half-step retries after the full trial.
    pub max_backtracks: usize,
}

impl StructuralTofMultiBankRefinementOptions {
    /// Construct validated structural TOF numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankRefinementError::InvalidOptions`] for
    /// invalid tolerances, damping, iteration, or step controls.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        limits: RefinementLimits,
        min_iterations: usize,
        objective_tolerance: f64,
        parameter_tolerance: f64,
        initial_damping: f64,
        damping_increase: f64,
        damping_decrease: f64,
        max_scaled_parameter_step: f64,
        max_backtracks: usize,
    ) -> Result<Self, StructuralTofMultiBankRefinementError> {
        let result = Self {
            limits,
            min_iterations,
            objective_tolerance,
            parameter_tolerance,
            initial_damping,
            damping_increase,
            damping_decrease,
            max_scaled_parameter_step,
            max_backtracks,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate adapter-decoded structural TOF controls.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankRefinementError::InvalidOptions`] for
    /// invalid fields.
    pub fn validate(self) -> Result<(), StructuralTofMultiBankRefinementError> {
        let positive = [
            self.objective_tolerance,
            self.parameter_tolerance,
            self.initial_damping,
            self.damping_increase,
            self.damping_decrease,
            self.max_scaled_parameter_step,
        ];
        if self.min_iterations == 0
            || self.min_iterations > self.limits.max_iterations()
            || positive
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self.damping_increase <= 1.0
            || self.damping_decrease >= 1.0
        {
            return Err(StructuralTofMultiBankRefinementError::InvalidOptions);
        }
        Ok(())
    }
}

/// One atomically accepted structural TOF step.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankIterationRecord {
    /// One-based accepted iteration.
    pub iteration: usize,
    /// Accepted half summed weighted residual square.
    pub objective: f64,
    /// Previous objective minus accepted objective.
    pub objective_change: f64,
    /// Accepted scaled Euclidean step norm.
    pub scaled_step_norm: f64,
    /// Damping used for the accepted trial.
    pub damping: f64,
    /// Half-step backtracks used.
    pub backtracks: usize,
    /// Accepted physical changes in stable joint order.
    pub parameter_changes: Vec<ParameterChange>,
}

/// Complete last-accepted state for deterministic continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankCheckpoint {
    /// Exact original request contract.
    pub request: StructuralTofMultiBankInput,
    /// Last accepted physical request state.
    pub input: StructuralTofMultiBankInput,
    /// Last accepted joint physical parameters.
    pub parameters: ParameterSet,
    /// Last accepted objective.
    pub objective: f64,
    /// Damping for the next attempted iteration.
    pub damping: f64,
    /// Complete accepted history.
    pub history: Vec<StructuralTofMultiBankIterationRecord>,
}

impl StructuralTofMultiBankCheckpoint {
    /// Return the number of accepted iterations.
    #[must_use]
    pub fn completed_iterations(&self) -> usize {
        self.history.len()
    }

    /// Revalidate this checkpoint against an exact request.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankRefinementError`] when any contract or
    /// accepted numerical invariant is stale.
    pub fn validate_for(
        &self,
        request: &StructuralTofMultiBankInput,
    ) -> Result<(), StructuralTofMultiBankRefinementError> {
        if &self.request != request
            || !self.objective.is_finite()
            || self.objective < 0.0
            || !self.damping.is_finite()
            || self.damping <= 0.0
            || self.history.iter().enumerate().any(|(index, row)| {
                row.iteration != index + 1
                    || !row.objective.is_finite()
                    || row.objective < 0.0
                    || !row.objective_change.is_finite()
                    || row.objective_change < 0.0
                    || !row.scaled_step_norm.is_finite()
                    || row.scaled_step_norm < 0.0
                    || !row.damping.is_finite()
                    || row.damping <= 0.0
                    || row.parameter_changes.iter().any(|change| {
                        !change.before.is_finite()
                            || !change.after.is_finite()
                            || !change.scaled_change.is_finite()
                    })
            })
            || self
                .history
                .last()
                .is_some_and(|row| row.objective.to_bits() != self.objective.to_bits())
        {
            return Err(StructuralTofMultiBankRefinementError::InvalidCheckpoint);
        }
        let request_layout = StructuralTofMultiBankLayout::new(request)?;
        let accepted_layout = StructuralTofMultiBankLayout::new(&self.input)?;
        if !same_contract(request_layout.parameters(), &self.parameters)
            || !same_values(accepted_layout.parameters(), &self.parameters)
        {
            return Err(StructuralTofMultiBankRefinementError::InvalidCheckpoint);
        }
        Ok(())
    }
}

/// Final accepted structural TOF refinement state.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankRefinementResult {
    /// Final accepted shared/local request state.
    pub input: StructuralTofMultiBankInput,
    /// Final display-ready bank calculations.
    pub calculation: StructuralTofMultiBankCalculation,
    /// Final physical parameters.
    pub parameters: ParameterSet,
    /// Complete accepted history.
    pub history: Vec<StructuralTofMultiBankIterationRecord>,
    /// Stable bounded termination category.
    pub termination_reason: TerminationReason,
    /// Restartable final checkpoint.
    pub checkpoint: StructuralTofMultiBankCheckpoint,
    /// Model evaluations consumed by this call.
    pub evaluations: usize,
}

/// Refine a guarded structural TOF objective through a process-local runtime.
///
/// # Errors
///
/// Returns [`StructuralTofMultiBankRefinementError`] for invalid controls,
/// checkpoint state, objective products, or linear solves.
pub fn refine_structural_tof_multibank(
    input: &StructuralTofMultiBankInput,
    options: StructuralTofMultiBankRefinementOptions,
    checkpoint: Option<&StructuralTofMultiBankCheckpoint>,
    cancellation: Option<CancellationToken>,
) -> Result<StructuralTofMultiBankRefinementResult, StructuralTofMultiBankRefinementError> {
    let mut runtime = RefinementRuntime::new(options.limits, cancellation)?;
    refine_structural_tof_multibank_with_runtime(input, options, checkpoint, &mut runtime)
}

/// Refine with caller-owned cancellation, event, and checkpoint sinks.
///
/// One trial installs all shared and bank-local values and is published only
/// after the summed objective decreases.
///
/// # Errors
///
/// Returns [`StructuralTofMultiBankRefinementError`] for invalid state.
#[allow(clippy::too_many_lines)]
pub fn refine_structural_tof_multibank_with_runtime(
    input: &StructuralTofMultiBankInput,
    options: StructuralTofMultiBankRefinementOptions,
    checkpoint: Option<&StructuralTofMultiBankCheckpoint>,
    runtime: &mut RefinementRuntime<StructuralTofMultiBankCheckpoint>,
) -> Result<StructuralTofMultiBankRefinementResult, StructuralTofMultiBankRefinementError> {
    options
        .validate()
        .map_err(|_| StructuralTofMultiBankRefinementError::InvalidOptions)?;
    let initial_layout = StructuralTofMultiBankLayout::new(input)?;
    if initial_layout.parameters().specs().is_empty() {
        return Err(StructuralTofMultiBankRefinementError::NoParameters);
    }
    let (mut live, mut parameters, mut history, mut damping) = if let Some(checkpoint) = checkpoint
    {
        checkpoint.validate_for(input)?;
        runtime.resume_accepted(checkpoint.completed_iterations())?;
        (
            checkpoint.input.clone(),
            checkpoint.parameters.clone(),
            checkpoint.history.clone(),
            checkpoint.damping,
        )
    } else {
        (
            input.clone(),
            initial_layout.parameters().clone(),
            Vec::new(),
            options.initial_damping,
        )
    };
    runtime.emit(
        RefinementEventKind::Start,
        "structural_tof_multibank",
        "joint structural TOF refinement started",
        vec![(
            "parameter_count".to_owned(),
            DiagnosticValue::Unsigned(parameters.specs().len() as u64),
        )],
    )?;
    let mut termination = TerminationReason::MaxIterations;
    'iterations: for iteration in history.len() + 1..=options.limits.max_iterations() {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_stop(error)?;
            break;
        }
        if let Err(error) = runtime.begin_evaluation() {
            termination = normal_stop(error)?;
            break;
        }
        let objective = PreparedStructuralTofMultiBankObjective::new(live.clone())?;
        let evaluated = objective.gradient()?;
        let current_objective = evaluated.calculation.objective;
        let scales = parameters
            .specs()
            .iter()
            .map(crate::ParameterSpec::scale)
            .collect::<Vec<_>>();
        let rhs = evaluated
            .gradient
            .iter()
            .zip(&scales)
            .map(|(gradient, scale)| -gradient * scale)
            .collect::<Vec<_>>();
        let mut normal = match scaled_normal_matrix(&objective, &scales, runtime) {
            Ok(normal) => normal,
            Err(StructuralTofMultiBankRefinementError::Runtime(RuntimeError::Stopped(stop))) => {
                termination = stop.reason;
                break;
            }
            Err(error) => return Err(error),
        };
        for index in 0..normal.nrows() {
            normal[(index, index)] += damping;
        }
        let rhs = DVector::from_vec(rhs);
        let mut step = normal
            .clone()
            .lu()
            .solve(&rhs)
            .or_else(|| normal.svd(true, true).solve(&rhs, 1.0e-12).ok())
            .ok_or(StructuralTofMultiBankRefinementError::LinearSolve)?;
        if step.iter().any(|value| !value.is_finite()) {
            return Err(StructuralTofMultiBankRefinementError::LinearSolve);
        }
        let mut step_norm = step.norm();
        if step_norm > options.max_scaled_parameter_step {
            step *= options.max_scaled_parameter_step / step_norm;
            step_norm = options.max_scaled_parameter_step;
        }
        if step_norm < options.parameter_tolerance {
            termination = TerminationReason::Converged;
            break;
        }
        let current = parameters
            .specs()
            .iter()
            .map(crate::ParameterSpec::value)
            .collect::<Vec<_>>();
        let mut accepted = None;
        for backtrack in 0..=options.max_backtracks {
            let factor = 0.5_f64.powi(i32::try_from(backtrack).unwrap_or(i32::MAX));
            let trial_values = parameters
                .specs()
                .iter()
                .zip(step.iter())
                .map(|(spec, delta)| {
                    spec.bounds()
                        .clip(spec.value() + factor * spec.scale() * delta)
                })
                .collect::<Vec<_>>();
            let Ok(trial) = objective.layout().apply_values(&live, &trial_values) else {
                runtime.reject_step().map_err(normal_or_error)?;
                continue;
            };
            if let Err(error) = runtime.begin_evaluation() {
                termination = normal_stop(error)?;
                break 'iterations;
            }
            let calculation =
                PreparedStructuralTofMultiBankObjective::new(trial.clone())?.calculate()?;
            if calculation.objective < current_objective {
                accepted = Some((backtrack, factor, trial_values, trial, calculation));
                break;
            }
            if let Err(error) = runtime.reject_step() {
                termination = normal_stop(error)?;
                break 'iterations;
            }
        }
        let Some((backtracks, factor, trial_values, trial, calculation)) = accepted else {
            damping *= options.damping_increase;
            continue;
        };
        let objective_change = current_objective - calculation.objective;
        let parameter_changes = parameters
            .specs()
            .iter()
            .zip(&current)
            .zip(&trial_values)
            .filter(|((_, before), after)| before.to_bits() != after.to_bits())
            .map(|((spec, before), after)| ParameterChange {
                key: spec.key().clone(),
                before: *before,
                after: *after,
                scaled_change: (after - before) / spec.scale(),
            })
            .collect();
        history.push(StructuralTofMultiBankIterationRecord {
            iteration: history.len() + 1,
            objective: calculation.objective,
            objective_change,
            scaled_step_norm: factor * step_norm,
            damping,
            backtracks,
            parameter_changes,
        });
        live = trial;
        parameters = parameter_set_with_values(&parameters, &trial_values)?;
        damping = (damping * options.damping_decrease).max(1.0e-18);
        let state = checkpoint_state(
            input,
            &live,
            &parameters,
            calculation.objective,
            damping,
            &history,
        );
        runtime.accept_step(Some(&state))?;
        runtime.emit(
            RefinementEventKind::StepAccepted,
            "structural_tof_multibank_step",
            "joint structural TOF step accepted",
            vec![(
                "objective".to_owned(),
                DiagnosticValue::Float(calculation.objective),
            )],
        )?;
        if history.len() >= options.min_iterations
            && objective_change <= options.objective_tolerance * calculation.objective.max(1.0)
        {
            termination = TerminationReason::Converged;
            break;
        }
    }
    runtime.begin_evaluation().or_else(|error| match error {
        RuntimeError::Stopped(_) => Ok(()),
        other => Err(other),
    })?;
    let calculation = PreparedStructuralTofMultiBankObjective::new(live.clone())?.calculate()?;
    let checkpoint = checkpoint_state(
        input,
        &live,
        &parameters,
        calculation.objective,
        damping,
        &history,
    );
    checkpoint.validate_for(input)?;
    runtime.emit(
        RefinementEventKind::Termination,
        "structural_tof_multibank",
        "joint structural TOF refinement terminated",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(StructuralTofMultiBankRefinementResult {
        input: live,
        calculation,
        parameters,
        history,
        termination_reason: termination,
        checkpoint,
        evaluations: runtime.evaluations(),
    })
}

fn scaled_normal_matrix(
    objective: &PreparedStructuralTofMultiBankObjective,
    scales: &[f64],
    runtime: &mut RefinementRuntime<StructuralTofMultiBankCheckpoint>,
) -> Result<DMatrix<f64>, StructuralTofMultiBankRefinementError> {
    let count = scales.len();
    let mut matrix = DMatrix::zeros(count, count);
    for column in 0..count {
        runtime.begin_evaluation().map_err(normal_or_error)?;
        let mut direction = vec![0.0; count];
        direction[column] = scales[column];
        let product = objective.normal_product(&direction, 0.0)?;
        for row in 0..count {
            matrix[(row, column)] = scales[row] * product[row];
        }
    }
    Ok(0.5 * (&matrix + matrix.transpose()))
}

fn parameter_set_with_values(
    template: &ParameterSet,
    values: &[f64],
) -> Result<ParameterSet, StructuralTofMultiBankRefinementError> {
    let replacements = template
        .specs()
        .iter()
        .zip(values)
        .map(|(spec, value)| (spec.key().clone(), *value))
        .collect();
    Ok(template.replace_values(&replacements)?)
}

fn checkpoint_state(
    request: &StructuralTofMultiBankInput,
    input: &StructuralTofMultiBankInput,
    parameters: &ParameterSet,
    objective: f64,
    damping: f64,
    history: &[StructuralTofMultiBankIterationRecord],
) -> StructuralTofMultiBankCheckpoint {
    StructuralTofMultiBankCheckpoint {
        request: request.clone(),
        input: input.clone(),
        parameters: parameters.clone(),
        objective,
        damping,
        history: history.to_vec(),
    }
}

fn same_contract(left: &ParameterSet, right: &ParameterSet) -> bool {
    left.specs().len() == right.specs().len()
        && left.specs().iter().zip(right.specs()).all(|(left, right)| {
            left.key() == right.key()
                && left.unit() == right.unit()
                && left.bounds() == right.bounds()
                && left.scale().to_bits() == right.scale().to_bits()
        })
}

fn same_values(left: &ParameterSet, right: &ParameterSet) -> bool {
    left.specs().len() == right.specs().len()
        && left.specs().iter().zip(right.specs()).all(|(left, right)| {
            left.key() == right.key() && left.value().to_bits() == right.value().to_bits()
        })
}

fn normal_stop(
    error: RuntimeError,
) -> Result<TerminationReason, StructuralTofMultiBankRefinementError> {
    match error {
        RuntimeError::Stopped(stop) => Ok(stop.reason),
        other => Err(StructuralTofMultiBankRefinementError::Runtime(other)),
    }
}

fn normal_or_error(error: RuntimeError) -> StructuralTofMultiBankRefinementError {
    StructuralTofMultiBankRefinementError::Runtime(error)
}

/// Invalid bounded structural TOF refinement state.
#[derive(Debug)]
pub enum StructuralTofMultiBankRefinementError {
    /// Solver controls are invalid.
    InvalidOptions,
    /// No physical parameter was selected.
    NoParameters,
    /// Restart state is stale or numerically invalid.
    InvalidCheckpoint,
    /// The scaled normal system could not be solved.
    LinearSolve,
    /// Joint objective state is invalid.
    Objective(StructuralTofMultiBankError),
    /// Stable parameter replacement failed.
    Parameter(ParameterError),
    /// Runtime boundary failed.
    Runtime(RuntimeError),
}

impl Display for StructuralTofMultiBankRefinementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => {
                formatter.write_str("structural TOF solver options are invalid")
            }
            Self::NoParameters => formatter
                .write_str("structural TOF solver requires at least one selected parameter"),
            Self::InvalidCheckpoint => formatter.write_str("structural TOF checkpoint is invalid"),
            Self::LinearSolve => {
                formatter.write_str("structural TOF normal system could not be solved")
            }
            Self::Objective(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for StructuralTofMultiBankRefinementError {}

impl From<StructuralTofMultiBankError> for StructuralTofMultiBankRefinementError {
    fn from(value: StructuralTofMultiBankError) -> Self {
        Self::Objective(value)
    }
}
impl From<ParameterError> for StructuralTofMultiBankRefinementError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<RuntimeError> for StructuralTofMultiBankRefinementError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
