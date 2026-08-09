//! Bounded matrix-free native structural Rietveld refinement.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    CancellationToken, DiagnosticValue, LatticeBounds, ParameterChange, ParameterSet,
    ParameterSpec, PreparedRietveldObjective, RefinementEventKind, RefinementLimits,
    RefinementRuntime, ResidualOptions, RietveldCalculation, RietveldCalculationOptions,
    RietveldError, RietveldInput, RietveldObjectiveError, RietveldParameterError, RietveldPhase,
    RietveldStructuralLayout, RietveldStructuralSelection, RietveldTopologyChange, RuntimeError,
    TerminationReason, calculate_rietveld_pattern, evaluate_residuals,
};

/// Numerical and bounded-runtime controls for native structural refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldRefinementOptions {
    /// Forward-calculation and execution controls.
    pub calculation: RietveldCalculationOptions,
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
    /// Relative conjugate-gradient residual tolerance.
    pub cg_tolerance: f64,
    /// Maximum conjugate-gradient iterations per attempted step.
    pub max_cg_iterations: usize,
    /// Maximum Euclidean scaled step norm.
    pub max_scaled_parameter_step: f64,
    /// Number of half-step retries after the full trial.
    pub max_backtracks: usize,
}

impl RietveldRefinementOptions {
    /// Construct and validate native solver controls.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldRefinementError::InvalidOptions`] for inconsistent
    /// tolerances, damping, iteration, or step controls.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        calculation: RietveldCalculationOptions,
        limits: RefinementLimits,
        min_iterations: usize,
        objective_tolerance: f64,
        parameter_tolerance: f64,
        initial_damping: f64,
        damping_increase: f64,
        damping_decrease: f64,
        cg_tolerance: f64,
        max_cg_iterations: usize,
        max_scaled_parameter_step: f64,
        max_backtracks: usize,
    ) -> Result<Self, RietveldRefinementError> {
        let result = Self {
            calculation,
            limits,
            min_iterations,
            objective_tolerance,
            parameter_tolerance,
            initial_damping,
            damping_increase,
            damping_decrease,
            cg_tolerance,
            max_cg_iterations,
            max_scaled_parameter_step,
            max_backtracks,
        };
        result.validate()?;
        Ok(result)
    }

    pub(crate) fn validate(&self) -> Result<(), RietveldRefinementError> {
        let positive = [
            self.objective_tolerance,
            self.parameter_tolerance,
            self.initial_damping,
            self.damping_increase,
            self.damping_decrease,
            self.cg_tolerance,
            self.max_scaled_parameter_step,
        ];
        if self.min_iterations == 0
            || self.min_iterations > self.limits.max_iterations()
            || self.max_cg_iterations == 0
            || positive
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self.damping_increase <= 1.0
            || self.damping_decrease >= 1.0
        {
            return Err(RietveldRefinementError::InvalidOptions);
        }
        Ok(())
    }
}

/// One accepted native Rietveld Gauss--Newton iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldIterationRecord {
    /// One-based attempted iteration.
    pub iteration: usize,
    /// Accepted weighted residual sum of squares divided by two.
    pub objective: f64,
    /// Previous objective minus accepted objective.
    pub objective_change: f64,
    /// Accepted scaled Euclidean step norm.
    pub scaled_step_norm: f64,
    /// Damping used to construct the accepted step.
    pub damping: f64,
    /// Conjugate-gradient iterations used.
    pub cg_iterations: usize,
    /// Half-step backtracks used.
    pub backtracks: usize,
    /// Accepted physical parameter changes.
    pub parameter_changes: Vec<ParameterChange>,
    /// Reflection families added or removed by accepted lattice motion.
    pub topology_changes: Vec<RietveldTopologyChange>,
    /// Accepted Rwp.
    pub rwp: f64,
}

/// Complete accepted state for deterministic native continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCheckpoint {
    /// Accepted iteration count.
    pub completed_iterations: usize,
    /// Last accepted phase state.
    pub phases: Vec<RietveldPhase>,
    /// Last accepted physical parameters.
    pub parameters: ParameterSet,
    /// Last accepted objective.
    pub objective: f64,
    /// Damping for the next attempted iteration.
    pub damping: f64,
    /// Complete accepted history.
    pub history: Vec<RietveldIterationRecord>,
}

impl RietveldCheckpoint {
    fn validate(&self, input: &RietveldInput) -> Result<(), RietveldRefinementError> {
        let phase_ids = input
            .phases
            .iter()
            .map(RietveldPhase::phase_id)
            .collect::<std::collections::BTreeSet<_>>();
        if self.completed_iterations != self.history.len()
            || self.phases.len() != input.phases.len()
            || self
                .phases
                .iter()
                .zip(&input.phases)
                .any(|(stored, requested)| !stored.restart_compatible(requested))
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
                    || !row.rwp.is_finite()
                    || row.parameter_changes.iter().any(|change| {
                        !change.before.is_finite()
                            || !change.after.is_finite()
                            || !change.scaled_change.is_finite()
                    })
                    || row.topology_changes.iter().any(|change| {
                        let added = change
                            .added_reflection_ids
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>();
                        let removed = change
                            .removed_reflection_ids
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>();
                        !phase_ids.contains(&change.phase_id)
                            || added.len() != change.added_reflection_ids.len()
                            || removed.len() != change.removed_reflection_ids.len()
                            || !added.is_disjoint(&removed)
                    })
            })
        {
            return Err(RietveldRefinementError::InvalidCheckpoint);
        }
        Ok(())
    }
}

/// Final accepted native structural refinement result.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldRefinementResult {
    /// Final display-ready calculation.
    pub calculation: RietveldCalculation,
    /// Final accepted phases.
    pub phases: Vec<RietveldPhase>,
    /// Final accepted physical parameters.
    pub parameters: ParameterSet,
    /// Complete accepted history.
    pub history: Vec<RietveldIterationRecord>,
    /// Stable bounded termination category.
    pub termination_reason: TerminationReason,
    /// Final restart checkpoint.
    pub checkpoint: RietveldCheckpoint,
    /// Model-product/evaluation count for this call.
    pub evaluations: usize,
}

/// Refine with a process-local runtime and optional cancellation token.
///
/// # Errors
///
/// Returns [`RietveldRefinementError`] for invalid input, checkpoint, model, or
/// numerical solver state.
pub fn refine_rietveld(
    input: &RietveldInput,
    selection: RietveldStructuralSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    options: &RietveldRefinementOptions,
    checkpoint: Option<&RietveldCheckpoint>,
    cancellation: Option<CancellationToken>,
) -> Result<RietveldRefinementResult, RietveldRefinementError> {
    let mut runtime = RefinementRuntime::new(options.limits, cancellation)?;
    refine_rietveld_with_runtime(
        input,
        selection,
        lattice_bounds,
        options,
        checkpoint,
        &mut runtime,
    )
}

/// Refine through a caller-owned runtime for events and durable checkpoints.
///
/// # Errors
///
/// Returns [`RietveldRefinementError`] for invalid state, non-normal runtime
/// failures, or numerical products.
#[allow(clippy::too_many_lines)]
pub fn refine_rietveld_with_runtime(
    input: &RietveldInput,
    selection: RietveldStructuralSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    options: &RietveldRefinementOptions,
    checkpoint: Option<&RietveldCheckpoint>,
    runtime: &mut RefinementRuntime<RietveldCheckpoint>,
) -> Result<RietveldRefinementResult, RietveldRefinementError> {
    input.validate()?;
    options.validate()?;
    let (mut phases, mut history, mut damping) = if let Some(checkpoint) = checkpoint {
        checkpoint.validate(input)?;
        let expected =
            RietveldStructuralLayout::new(&checkpoint.phases, selection, lattice_bounds)?;
        if expected.parameters() != &checkpoint.parameters {
            return Err(RietveldRefinementError::InvalidCheckpoint);
        }
        runtime.resume_accepted(checkpoint.completed_iterations)?;
        (
            checkpoint.phases.clone(),
            checkpoint.history.clone(),
            checkpoint.damping,
        )
    } else {
        (input.phases.clone(), Vec::new(), options.initial_damping)
    };
    runtime.emit(
        RefinementEventKind::Start,
        "rietveld",
        "native structural refinement started",
        Vec::new(),
    )?;
    let has_observations = input
        .pattern
        .mask
        .as_ref()
        .is_none_or(|mask| mask.iter().any(|included| *included));
    let mut termination = if has_observations {
        TerminationReason::MaxIterations
    } else {
        TerminationReason::NoObservations
    };
    let first_iteration = history.len() + 1;
    let last_iteration = if has_observations {
        options.limits.max_iterations()
    } else {
        history.len()
    };
    'iterations: for iteration in first_iteration..=last_iteration {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_stop(&error)?;
            break;
        }
        let layout = RietveldStructuralLayout::new(&phases, selection, lattice_bounds)?;
        let specs = layout.parameters().specs();
        if specs.is_empty() {
            termination = TerminationReason::Converged;
            break;
        }
        let live_input = replace_phases(input, phases.clone())?;
        let objective = PreparedRietveldObjective::new(
            live_input.clone(),
            options.calculation.clone(),
            layout.clone(),
        )?;
        if let Err(error) = reserve_products(runtime, 2) {
            termination = normal_stop(&error)?;
            break;
        }
        let (_, physical_gradient) = objective.gradient()?;
        let scales = specs.iter().map(ParameterSpec::scale).collect::<Vec<_>>();
        let scaled_gradient = physical_gradient
            .iter()
            .zip(&scales)
            .map(|(gradient, scale)| gradient * scale)
            .collect::<Vec<_>>();
        let right_hand_side = scaled_gradient
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let solve = conjugate_gradient(
            &right_hand_side,
            options.cg_tolerance,
            options.max_cg_iterations,
            |direction| {
                reserve_products(runtime, 2)?;
                let physical = direction
                    .iter()
                    .zip(&scales)
                    .map(|(value, scale)| value * scale)
                    .collect::<Vec<_>>();
                let product = objective.normal_product(&physical, 0.0)?;
                Ok(product
                    .iter()
                    .zip(&scales)
                    .zip(direction)
                    .map(|((value, scale), direction)| value * scale + damping * direction)
                    .collect())
            },
        );
        let (mut step, cg_iterations) = match solve {
            Ok(result) => result,
            Err(RietveldRefinementError::Runtime(RuntimeError::Stopped(stop))) => {
                termination = stop.reason;
                break 'iterations;
            }
            Err(error) => return Err(error),
        };
        let mut step_norm = norm(&step);
        if step_norm > options.max_scaled_parameter_step {
            let factor = options.max_scaled_parameter_step / step_norm;
            for value in &mut step {
                *value *= factor;
            }
            step_norm = options.max_scaled_parameter_step;
        }
        if step_norm < options.parameter_tolerance {
            termination = TerminationReason::Converged;
            break;
        }
        if let Err(error) = runtime.begin_evaluation() {
            termination = normal_stop(&error)?;
            break;
        }
        let current_calculation = calculate_rietveld_pattern(&live_input, &options.calculation)?;
        let current_objective = 0.5 * current_calculation.metrics.chi_square;
        let current_values = specs.iter().map(ParameterSpec::value).collect::<Vec<_>>();
        let mut accepted = None;
        for backtrack in 0..=options.max_backtracks {
            let factor = 0.5_f64.powi(i32::try_from(backtrack).unwrap_or(i32::MAX));
            let trial_values = specs
                .iter()
                .zip(&current_values)
                .zip(step.iter().zip(&scales))
                .map(|((spec, current), (step, scale))| {
                    spec.bounds().clip(current + factor * step * scale)
                })
                .collect::<Vec<_>>();
            let trial_phases = layout.apply_values(&phases, &trial_values)?;
            if let Err(error) = runtime.begin_evaluation() {
                termination = normal_stop(&error)?;
                break 'iterations;
            }
            let trial_input = replace_phases(input, trial_phases.clone())?;
            let trial_calculation = calculate_rietveld_pattern(&trial_input, &options.calculation)?;
            let trial_objective = 0.5 * trial_calculation.metrics.chi_square;
            runtime.emit(
                RefinementEventKind::Trial,
                "rietveld_step",
                "native structural trial evaluated",
                vec![(
                    "objective".to_owned(),
                    DiagnosticValue::Float(trial_objective),
                )],
            )?;
            if trial_objective < current_objective {
                accepted = Some((
                    backtrack,
                    factor,
                    trial_values,
                    trial_phases,
                    trial_calculation,
                    trial_objective,
                ));
                break;
            }
            if let Err(error) = runtime.reject_step() {
                termination = normal_stop(&error)?;
                break;
            }
        }
        let Some((backtracks, factor, trial_values, trial_phases, trial_calculation, objective)) =
            accepted
        else {
            if termination == TerminationReason::MaxIterations {
                termination = TerminationReason::Stagnated;
            }
            damping *= options.damping_increase;
            break;
        };
        let objective_change = current_objective - objective;
        let parameter_changes = specs
            .iter()
            .zip(&current_values)
            .zip(&trial_values)
            .filter(|((_, before), after)| before.to_bits() != after.to_bits())
            .map(|((spec, before), after)| ParameterChange {
                key: spec.key().clone(),
                before: *before,
                after: *after,
                scaled_change: (after - before) / spec.scale(),
            })
            .collect::<Vec<_>>();
        let topology_changes = phases
            .iter()
            .zip(&trial_phases)
            .filter_map(|(before, after)| topology_change(before, after))
            .collect::<Vec<_>>();
        history.push(RietveldIterationRecord {
            iteration,
            objective,
            objective_change,
            scaled_step_norm: factor * step_norm,
            damping,
            cg_iterations,
            backtracks,
            parameter_changes,
            topology_changes: topology_changes.clone(),
            rwp: trial_calculation.metrics.rwp,
        });
        phases = trial_phases;
        damping = (damping * options.damping_decrease).max(1.0e-18);
        let accepted_layout = RietveldStructuralLayout::new(&phases, selection, lattice_bounds)?;
        let state = RietveldCheckpoint {
            completed_iterations: history.len(),
            phases: phases.clone(),
            parameters: accepted_layout.parameters().clone(),
            objective,
            damping,
            history: history.clone(),
        };
        runtime.accept_step(Some(&state))?;
        runtime.emit(
            RefinementEventKind::StepAccepted,
            "rietveld_step",
            "native structural step accepted",
            vec![
                ("objective".to_owned(), DiagnosticValue::Float(objective)),
                (
                    "topology_changes".to_owned(),
                    DiagnosticValue::Integer(
                        i64::try_from(topology_changes.len()).unwrap_or(i64::MAX),
                    ),
                ),
            ],
        )?;
        if iteration >= options.min_iterations
            && objective_change <= options.objective_tolerance * objective.max(1.0)
        {
            termination = TerminationReason::Converged;
            break;
        }
    }
    let final_layout = RietveldStructuralLayout::new(&phases, selection, lattice_bounds)?;
    let final_input = replace_phases(input, phases.clone())?;
    runtime.begin_evaluation().or_else(|error| match error {
        RuntimeError::Stopped(_) => Ok(()),
        other => Err(other),
    })?;
    let mut calculation = calculate_rietveld_pattern(&final_input, &options.calculation)?;
    calculation.metrics = evaluate_residuals(
        &input.pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.calculation.use_uncertainty,
            parameter_count: final_layout.parameters().specs().len(),
        },
    )?;
    let checkpoint = RietveldCheckpoint {
        completed_iterations: history.len(),
        phases: phases.clone(),
        parameters: final_layout.parameters().clone(),
        objective: 0.5 * calculation.metrics.chi_square,
        damping,
        history: history.clone(),
    };
    checkpoint.validate(input)?;
    runtime.emit(
        RefinementEventKind::Termination,
        "rietveld",
        "native structural refinement terminated",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(RietveldRefinementResult {
        calculation,
        phases,
        parameters: final_layout.parameters().clone(),
        history,
        termination_reason: termination,
        checkpoint,
        evaluations: runtime.evaluations(),
    })
}

pub(crate) fn topology_change(
    before: &RietveldPhase,
    after: &RietveldPhase,
) -> Option<RietveldTopologyChange> {
    let previous = before
        .reflection_ids()
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    let current = after
        .reflection_ids()
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    let added_reflection_ids = after
        .reflection_ids()
        .iter()
        .filter(|id| !previous.contains(id))
        .cloned()
        .collect::<Vec<_>>();
    let removed_reflection_ids = before
        .reflection_ids()
        .iter()
        .filter(|id| !current.contains(id))
        .cloned()
        .collect::<Vec<_>>();
    if added_reflection_ids.is_empty() && removed_reflection_ids.is_empty() {
        return None;
    }
    Some(RietveldTopologyChange {
        phase_id: after.phase_id().clone(),
        added_reflection_ids,
        removed_reflection_ids,
        preserved_reflection_count: current.intersection(&previous).count(),
    })
}

fn replace_phases(
    input: &RietveldInput,
    phases: Vec<RietveldPhase>,
) -> Result<RietveldInput, RietveldError> {
    let mut replaced = RietveldInput::new(
        input.pattern.clone(),
        input.instrument,
        input.axial_geometry,
        input.position_correction,
        phases,
    )?;
    replaced.background.clone_from(&input.background);
    replaced.validate()?;
    Ok(replaced)
}

pub(crate) fn reserve_products<T>(
    runtime: &mut RefinementRuntime<T>,
    count: usize,
) -> Result<(), RuntimeError> {
    for _ in 0..count {
        runtime.begin_evaluation()?;
    }
    Ok(())
}

pub(crate) fn conjugate_gradient(
    right_hand_side: &[f64],
    tolerance: f64,
    max_iterations: usize,
    mut operator: impl FnMut(&[f64]) -> Result<Vec<f64>, RietveldRefinementError>,
) -> Result<(Vec<f64>, usize), RietveldRefinementError> {
    let mut solution = vec![0.0; right_hand_side.len()];
    let mut residual = right_hand_side.to_vec();
    let mut direction = residual.clone();
    let mut squared = dot(&residual, &residual);
    let target = tolerance * norm(right_hand_side).max(1.0);
    if squared.sqrt() <= target {
        return Ok((solution, 0));
    }
    for iteration in 1..=max_iterations {
        let product = operator(&direction)?;
        let denominator = dot(&direction, &product);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(RietveldRefinementError::NonPositiveNormalOperator);
        }
        let alpha = squared / denominator;
        for index in 0..solution.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        let next_squared = dot(&residual, &residual);
        if !next_squared.is_finite() {
            return Err(RietveldRefinementError::NonFiniteSolve);
        }
        if next_squared.sqrt() <= target {
            return Ok((solution, iteration));
        }
        let beta = next_squared / squared;
        for index in 0..direction.len() {
            direction[index] = residual[index] + beta * direction[index];
        }
        squared = next_squared;
    }
    Ok((solution, max_iterations))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

pub(crate) fn normal_stop(
    error: &RuntimeError,
) -> Result<TerminationReason, RietveldRefinementError> {
    if let RuntimeError::Stopped(stop) = error {
        Ok(stop.reason)
    } else {
        Err(RietveldRefinementError::RuntimeMessage(error.to_string()))
    }
}

/// Invalid native structural refinement state.
#[derive(Debug)]
pub enum RietveldRefinementError {
    /// Solver controls are invalid.
    InvalidOptions,
    /// Restart state is inconsistent with the request.
    InvalidCheckpoint,
    /// Matrix-free normal operator lost positive definiteness.
    NonPositiveNormalOperator,
    /// Conjugate gradients produced non-finite state.
    NonFiniteSolve,
    /// A non-normal runtime failure occurred.
    RuntimeMessage(String),
    /// Owned calculation state is invalid.
    Rietveld(RietveldError),
    /// Structural parameter transform failed.
    Parameter(RietveldParameterError),
    /// Complete parameter layout or installation failed.
    GeneralParameter(crate::RietveldGeneralParameterError),
    /// Prepared objective failed.
    Objective(RietveldObjectiveError),
    /// Complete prepared objective failed.
    GeneralObjective(crate::RietveldGeneralObjectiveError),
    /// Constraint graph or transform failed.
    Constraint(crate::ConstraintError),
    /// Runtime boundary failed.
    Runtime(RuntimeError),
    /// Residual evaluation failed.
    Residual(crate::ResidualError),
}

impl Display for RietveldRefinementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => formatter.write_str("native Rietveld options are invalid"),
            Self::InvalidCheckpoint => formatter.write_str("native Rietveld checkpoint is invalid"),
            Self::NonPositiveNormalOperator => {
                formatter.write_str("native Rietveld normal operator is not positive definite")
            }
            Self::NonFiniteSolve => formatter.write_str("native Rietveld solve became non-finite"),
            Self::RuntimeMessage(message) => formatter.write_str(message),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::GeneralParameter(error) => Display::fmt(error, formatter),
            Self::Objective(error) => Display::fmt(error, formatter),
            Self::GeneralObjective(error) => Display::fmt(error, formatter),
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RietveldRefinementError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rietveld(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::GeneralParameter(error) => Some(error),
            Self::Objective(error) => Some(error),
            Self::GeneralObjective(error) => Some(error),
            Self::Constraint(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::InvalidOptions
            | Self::InvalidCheckpoint
            | Self::NonPositiveNormalOperator
            | Self::NonFiniteSolve
            | Self::RuntimeMessage(_) => None,
        }
    }
}
impl From<RietveldError> for RietveldRefinementError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}
impl From<RietveldParameterError> for RietveldRefinementError {
    fn from(value: RietveldParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<RietveldObjectiveError> for RietveldRefinementError {
    fn from(value: RietveldObjectiveError) -> Self {
        Self::Objective(value)
    }
}
impl From<crate::RietveldGeneralParameterError> for RietveldRefinementError {
    fn from(value: crate::RietveldGeneralParameterError) -> Self {
        Self::GeneralParameter(value)
    }
}
impl From<crate::RietveldGeneralObjectiveError> for RietveldRefinementError {
    fn from(value: crate::RietveldGeneralObjectiveError) -> Self {
        Self::GeneralObjective(value)
    }
}
impl From<crate::ConstraintError> for RietveldRefinementError {
    fn from(value: crate::ConstraintError) -> Self {
        Self::Constraint(value)
    }
}
impl From<RuntimeError> for RietveldRefinementError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
impl From<crate::ResidualError> for RietveldRefinementError {
    fn from(value: crate::ResidualError) -> Self {
        Self::Residual(value)
    }
}
