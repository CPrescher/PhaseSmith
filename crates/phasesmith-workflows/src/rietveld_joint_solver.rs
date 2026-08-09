//! Bounded constraint-aware solver for a joint multi-histogram objective.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::rietveld_solver::{
    ConjugateGradientError, conjugate_gradient_core, norm, topology_change,
};
use crate::{
    CancellationToken, Constraint, ConstraintDerivativeMatrix, ConstraintError,
    ConstraintTransform, DiagnosticValue, JointRietveldError, JointRietveldHistogram,
    JointRietveldLayout, ParameterChange, ParameterError, ParameterKey, ParameterSet,
    PreparedJointRietveldObjective, RefinementEventKind, RefinementLimits, RefinementRuntime,
    RietveldCalculation, RietveldTopologyChange, RuntimeError, TerminationReason,
    calculate_rietveld_pattern,
};

/// Numerical and bounded-runtime controls for a joint native refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointRietveldRefinementOptions {
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

impl JointRietveldRefinementOptions {
    /// Construct validated joint numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldRefinementError::InvalidOptions`] for invalid
    /// tolerances, damping, iteration, or step controls.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Result<Self, JointRietveldRefinementError> {
        let result = Self {
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

    /// Revalidate adapter-decoded joint numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldRefinementError::InvalidOptions`] for invalid
    /// fields.
    pub fn validate(self) -> Result<(), JointRietveldRefinementError> {
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
            return Err(JointRietveldRefinementError::InvalidOptions);
        }
        Ok(())
    }
}

/// One topology change owned by a specific histogram.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointRietveldTopologyChange {
    /// Stable histogram identity.
    pub histogram_id: phasesmith_model::RecordId,
    /// Per-phase reflection-family change.
    pub change: RietveldTopologyChange,
}

/// Aggregate powder residual metrics over all joint histograms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointRietveldMetrics {
    /// Total number of included observations across all histograms.
    pub included_samples: usize,
    /// Sum of absolute residuals divided by sum of absolute observations.
    pub rp: f64,
    /// Square root of joint chi-square divided by weighted observed square sum.
    pub rwp: f64,
    /// Sum of all selected squared weighted residuals.
    pub chi_square: f64,
    /// Joint chi-square divided by `included_samples - free_parameters`.
    pub reduced_chi_square: f64,
}

/// One accepted joint Gauss--Newton iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldIterationRecord {
    /// One-based attempted iteration.
    pub iteration: usize,
    /// Accepted half summed weighted residual square.
    pub objective: f64,
    /// Previous joint objective minus accepted objective.
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
    /// Reflection topology changes in stable histogram order.
    pub topology_changes: Vec<JointRietveldTopologyChange>,
    /// Accepted aggregate residual metrics.
    pub metrics: JointRietveldMetrics,
}

/// Complete accepted state for deterministic joint continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldCheckpoint {
    /// Exact original request contract.
    pub request: Vec<JointRietveldHistogram>,
    /// Last accepted joint histogram state.
    pub histograms: Vec<JointRietveldHistogram>,
    /// Ordered constraint graph.
    pub constraints: Vec<Constraint>,
    /// Last accepted joint physical parameters.
    pub parameters: ParameterSet,
    /// Last accepted joint objective.
    pub objective: f64,
    /// Damping for the next attempted iteration.
    pub damping: f64,
    /// Complete accepted history.
    pub history: Vec<JointRietveldIterationRecord>,
}

impl JointRietveldCheckpoint {
    /// Return the number of accepted iterations.
    #[must_use]
    pub fn completed_iterations(&self) -> usize {
        self.history.len()
    }

    /// Revalidate this checkpoint against an exact request and constraints.
    ///
    /// # Errors
    ///
    /// Returns [`JointRietveldRefinementError::InvalidCheckpoint`] when any
    /// accepted-state or continuation invariant is stale.
    pub fn validate_for(
        &self,
        request: &[JointRietveldHistogram],
        constraints: &[Constraint],
    ) -> Result<(), JointRietveldRefinementError> {
        if self.request != request {
            return Err(JointRietveldRefinementError::InvalidCheckpoint {
                reason: "request contract changed",
            });
        }
        if self.constraints != constraints {
            return Err(JointRietveldRefinementError::InvalidCheckpoint {
                reason: "constraint graph changed",
            });
        }
        if !self.objective.is_finite()
            || self.objective < 0.0
            || !self.damping.is_finite()
            || self.damping <= 0.0
            || !valid_history(&self.history)
            || self
                .history
                .last()
                .is_some_and(|row| row.objective.to_bits() != self.objective.to_bits())
        {
            return Err(JointRietveldRefinementError::InvalidCheckpoint {
                reason: "accepted numerical state is invalid",
            });
        }
        let request_layout = JointRietveldLayout::new(request)?;
        if !parameter_contract_matches(request_layout.parameters(), &self.parameters) {
            return Err(JointRietveldRefinementError::InvalidCheckpoint {
                reason: "parameter contract changed",
            });
        }
        let layout = JointRietveldLayout::new(&self.histograms)?;
        if !parameter_values_match(layout.parameters(), &self.parameters) {
            return Err(JointRietveldRefinementError::InvalidCheckpoint {
                reason: "accepted parameters do not match histogram state",
            });
        }
        let transform = ConstraintTransform::new(self.parameters.clone(), constraints.to_vec())?;
        validate_constraint_state(layout.parameters(), &transform)?;
        Ok(())
    }
}

/// Final accepted joint native refinement result.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRietveldRefinementResult {
    /// Final accepted histogram state.
    pub histograms: Vec<JointRietveldHistogram>,
    /// Final display-ready calculations in histogram order.
    pub calculations: Vec<RietveldCalculation>,
    /// Final aggregate residual metrics over every histogram.
    pub metrics: JointRietveldMetrics,
    /// Final joint physical parameters.
    pub parameters: ParameterSet,
    /// Stable scaled-free parameter order.
    pub free_keys: Vec<ParameterKey>,
    /// Complete accepted history.
    pub history: Vec<JointRietveldIterationRecord>,
    /// Stable bounded termination category.
    pub termination_reason: TerminationReason,
    /// Final restart checkpoint.
    pub checkpoint: JointRietveldCheckpoint,
    /// Joint model-product/evaluation count for this call.
    pub evaluations: usize,
}

/// Refine a joint objective through a process-local runtime.
///
/// # Errors
///
/// Returns [`JointRietveldRefinementError`] for invalid joint layouts,
/// constraints, checkpoints, runtime state, or numerical products.
pub fn refine_joint_rietveld(
    histograms: &[JointRietveldHistogram],
    constraints: &[Constraint],
    options: JointRietveldRefinementOptions,
    checkpoint: Option<&JointRietveldCheckpoint>,
    cancellation: Option<CancellationToken>,
) -> Result<JointRietveldRefinementResult, JointRietveldRefinementError> {
    let mut runtime = RefinementRuntime::new(options.limits, cancellation)?;
    refine_joint_rietveld_with_runtime(histograms, constraints, options, checkpoint, &mut runtime)
}

/// Refine a joint objective through a caller-owned runtime.
///
/// One trial evaluates every histogram and is accepted only when their summed
/// objective decreases.
///
/// # Errors
///
/// Returns [`JointRietveldRefinementError`] for invalid state or products.
#[allow(clippy::too_many_lines)]
pub fn refine_joint_rietveld_with_runtime(
    histograms: &[JointRietveldHistogram],
    constraints: &[Constraint],
    options: JointRietveldRefinementOptions,
    checkpoint: Option<&JointRietveldCheckpoint>,
    runtime: &mut RefinementRuntime<JointRietveldCheckpoint>,
) -> Result<JointRietveldRefinementResult, JointRietveldRefinementError> {
    options.validate()?;
    let initial_layout = JointRietveldLayout::new(histograms)?;
    let initial_transform =
        ConstraintTransform::new(initial_layout.parameters().clone(), constraints.to_vec())?;
    validate_constraint_state(initial_layout.parameters(), &initial_transform)?;
    let (mut live, mut history, mut damping, parameter_template) =
        if let Some(checkpoint) = checkpoint {
            checkpoint.validate_for(histograms, constraints)?;
            runtime.resume_accepted(checkpoint.completed_iterations())?;
            (
                checkpoint.histograms.clone(),
                checkpoint.history.clone(),
                checkpoint.damping,
                checkpoint.parameters.clone(),
            )
        } else {
            (
                histograms.to_vec(),
                Vec::new(),
                options.initial_damping,
                initial_layout.parameters().clone(),
            )
        };
    runtime.emit(
        RefinementEventKind::Start,
        "joint_rietveld",
        "native joint Rietveld refinement started",
        vec![(
            "histograms".to_owned(),
            DiagnosticValue::Integer(i64::try_from(histograms.len()).unwrap_or(i64::MAX)),
        )],
    )?;
    let has_observations = histograms.iter().any(|histogram| {
        histogram
            .input
            .pattern
            .mask
            .as_ref()
            .is_none_or(|mask| mask.iter().any(|included| *included))
    });
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
            termination = normal_stop(error)?;
            break;
        }
        let layout = JointRietveldLayout::new(&live)?;
        let solver_parameters = parameter_template.replace_values(&layout.parameters().values())?;
        let transform = ConstraintTransform::new(solver_parameters.clone(), constraints.to_vec())?;
        if transform.free_keys().is_empty() {
            termination = TerminationReason::Converged;
            break;
        }
        let derivative = transform.derivative_matrix()?;
        let objective = PreparedJointRietveldObjective::new(live.clone(), layout.clone())?;
        if let Err(error) = reserve_products(runtime, objective.preparation_evaluation_count()) {
            match error {
                JointRietveldRefinementError::Runtime(RuntimeError::Stopped(stop)) => {
                    termination = stop.reason;
                    break;
                }
                other => return Err(other),
            }
        }
        let evaluated = objective.gradient()?;
        let current_objective = evaluated.objective;
        let scaled_gradient = transpose_product(&derivative, &evaluated.gradient);
        let right_hand_side = scaled_gradient
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let solve = conjugate_gradient(
            &right_hand_side,
            options.cg_tolerance,
            options.max_cg_iterations,
            |direction| {
                reserve_products(runtime, objective.normal_product_evaluation_count())?;
                let physical = forward_product(&derivative, direction);
                let physical_product = objective.normal_product(&physical, 0.0)?;
                let mut result = transpose_product(&derivative, &physical_product);
                for (value, direction) in result.iter_mut().zip(direction) {
                    *value += damping * direction;
                }
                Ok(result)
            },
        );
        let (mut step, cg_iterations) = match solve {
            Ok(result) => result,
            Err(JointRietveldRefinementError::Runtime(RuntimeError::Stopped(stop))) => {
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
        let current_values = solver_parameters
            .specs()
            .iter()
            .map(crate::ParameterSpec::value)
            .collect::<Vec<_>>();
        let packed = transform.pack()?;
        let mut accepted = None;
        for backtrack in 0..=options.max_backtracks {
            let factor = 0.5_f64.powi(i32::try_from(backtrack).unwrap_or(i32::MAX));
            let trial_free = packed
                .iter()
                .zip(&step)
                .map(|(value, step)| value + factor * step)
                .collect::<Vec<_>>();
            let trial_map = match transform.unpack(&trial_free, true) {
                Ok(values) => values,
                Err(ConstraintError::ExpandedValueOutsideBounds { .. }) => {
                    emit_rejected_trial(runtime, "constraint result outside physical bounds")?;
                    if let Err(error) = runtime.reject_step() {
                        termination = normal_stop(error)?;
                        break 'iterations;
                    }
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let trial_values = solver_parameters
                .specs()
                .iter()
                .map(|spec| {
                    trial_map
                        .get(spec.key())
                        .copied()
                        .ok_or(JointRietveldRefinementError::InternalInvariant)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let Ok(trial) = layout.apply_values(&live, &trial_values) else {
                emit_rejected_trial(runtime, "trial outside the numerical model domain")?;
                if let Err(error) = runtime.reject_step() {
                    termination = normal_stop(error)?;
                    break 'iterations;
                }
                continue;
            };
            if let Err(error) = runtime.begin_evaluation() {
                termination = normal_stop(error)?;
                break 'iterations;
            }
            let (trial_calculations, trial_metrics) =
                calculate_joint(&trial, transform.free_keys().len())?;
            let trial_objective = 0.5 * trial_metrics.chi_square;
            runtime.emit(
                RefinementEventKind::Trial,
                "joint_rietveld_step",
                "native joint Rietveld trial evaluated",
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
                    trial,
                    trial_calculations,
                    trial_metrics,
                    trial_objective,
                ));
                break;
            }
            if let Err(error) = runtime.reject_step() {
                termination = normal_stop(error)?;
                break 'iterations;
            }
        }
        let Some((
            backtracks,
            factor,
            trial_values,
            trial,
            _,
            accepted_metrics,
            accepted_objective,
        )) = accepted
        else {
            damping *= options.damping_increase;
            if termination == TerminationReason::MaxIterations {
                continue;
            }
            break;
        };
        let objective_change = current_objective - accepted_objective;
        let parameter_changes = solver_parameters
            .specs()
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
            .collect();
        let topology_changes = collect_topology_changes(&live, &trial);
        history.push(JointRietveldIterationRecord {
            iteration: history.len() + 1,
            objective: accepted_objective,
            objective_change,
            scaled_step_norm: factor * step_norm,
            damping,
            cg_iterations,
            backtracks,
            parameter_changes,
            topology_changes: topology_changes.clone(),
            metrics: accepted_metrics,
        });
        live = trial;
        damping = (damping * options.damping_decrease).max(1.0e-18);
        let accepted_layout = JointRietveldLayout::new(&live)?;
        let accepted_parameters =
            parameter_template.replace_values(&accepted_layout.parameters().values())?;
        let state = JointRietveldCheckpoint {
            request: histograms.to_vec(),
            histograms: live.clone(),
            constraints: constraints.to_vec(),
            parameters: accepted_parameters,
            objective: accepted_objective,
            damping,
            history: history.clone(),
        };
        runtime.accept_step(Some(&state))?;
        runtime.emit(
            RefinementEventKind::StepAccepted,
            "joint_rietveld_step",
            "native joint Rietveld step accepted",
            vec![
                (
                    "objective".to_owned(),
                    DiagnosticValue::Float(accepted_objective),
                ),
                (
                    "topology_changes".to_owned(),
                    DiagnosticValue::Integer(
                        i64::try_from(topology_changes.len()).unwrap_or(i64::MAX),
                    ),
                ),
            ],
        )?;
        if history.len() >= options.min_iterations
            && objective_change <= options.objective_tolerance * accepted_objective.max(1.0)
        {
            termination = TerminationReason::Converged;
            break;
        }
    }
    let final_layout = JointRietveldLayout::new(&live)?;
    let final_parameters =
        parameter_template.replace_values(&final_layout.parameters().values())?;
    let final_transform = ConstraintTransform::new(final_parameters.clone(), constraints.to_vec())?;
    runtime.begin_evaluation().or_else(|error| match error {
        RuntimeError::Stopped(_) => Ok(()),
        other => Err(other),
    })?;
    let (calculations, metrics) = calculate_joint(&live, final_transform.free_keys().len())?;
    let objective = 0.5 * metrics.chi_square;
    let checkpoint = JointRietveldCheckpoint {
        request: histograms.to_vec(),
        histograms: live.clone(),
        constraints: constraints.to_vec(),
        parameters: final_parameters.clone(),
        objective,
        damping,
        history: history.clone(),
    };
    checkpoint.validate_for(histograms, constraints)?;
    runtime.emit(
        RefinementEventKind::Termination,
        "joint_rietveld",
        "native joint Rietveld refinement terminated",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(JointRietveldRefinementResult {
        histograms: live,
        calculations,
        metrics,
        parameters: final_parameters,
        free_keys: final_transform.free_keys().to_vec(),
        history,
        termination_reason: termination,
        checkpoint,
        evaluations: runtime.evaluations(),
    })
}

fn calculate_joint(
    histograms: &[JointRietveldHistogram],
    parameter_count: usize,
) -> Result<(Vec<RietveldCalculation>, JointRietveldMetrics), JointRietveldRefinementError> {
    let calculations = histograms
        .iter()
        .map(|histogram| calculate_rietveld_pattern(&histogram.input, &histogram.calculation))
        .collect::<Result<Vec<_>, _>>()?;
    let metrics = joint_metrics(histograms, &calculations, parameter_count)?;
    Ok((calculations, metrics))
}

fn joint_metrics(
    histograms: &[JointRietveldHistogram],
    calculations: &[RietveldCalculation],
    parameter_count: usize,
) -> Result<JointRietveldMetrics, JointRietveldRefinementError> {
    let mut included_samples = 0_usize;
    let mut absolute_residual_sum = 0.0;
    let mut absolute_observed_sum = 0.0;
    let mut weighted_observed_square_sum = 0.0;
    let mut chi_square = 0.0;
    for (histogram, calculation) in histograms.iter().zip(calculations) {
        let observed = histogram
            .input
            .pattern
            .observed_y
            .as_ref()
            .ok_or(crate::RietveldError::MissingObservations)?;
        let uncertainty = histogram
            .calculation
            .use_uncertainty
            .then_some(histogram.input.pattern.uncertainty.as_deref())
            .flatten();
        for index in 0..histogram.input.pattern.sample_count() {
            if histogram
                .input
                .pattern
                .mask
                .as_ref()
                .is_some_and(|mask| !mask[index])
            {
                continue;
            }
            included_samples += 1;
            let residual = calculation.y[index] - observed[index];
            absolute_residual_sum += residual.abs();
            absolute_observed_sum += observed[index].abs();
            let (weighted_residual, weighted_observed) = uncertainty.map_or_else(
                || (residual, observed[index]),
                |sigma| (residual / sigma[index], observed[index] / sigma[index]),
            );
            chi_square += weighted_residual * weighted_residual;
            weighted_observed_square_sum += weighted_observed * weighted_observed;
        }
    }
    let degrees_of_freedom = included_samples.checked_sub(parameter_count);
    Ok(JointRietveldMetrics {
        included_samples,
        rp: if absolute_observed_sum == 0.0 {
            f64::INFINITY
        } else {
            absolute_residual_sum / absolute_observed_sum
        },
        rwp: if weighted_observed_square_sum == 0.0 {
            f64::INFINITY
        } else {
            (chi_square / weighted_observed_square_sum).sqrt()
        },
        chi_square,
        reduced_chi_square: match degrees_of_freedom {
            Some(degrees) if degrees > 0 => chi_square / count_as_f64(degrees),
            _ => f64::INFINITY,
        },
    })
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    value as f64
}

fn collect_topology_changes(
    before: &[JointRietveldHistogram],
    after: &[JointRietveldHistogram],
) -> Vec<JointRietveldTopologyChange> {
    before
        .iter()
        .zip(after)
        .flat_map(|(before, after)| {
            before
                .input
                .phases
                .iter()
                .zip(&after.input.phases)
                .filter_map(|(left, right)| topology_change(left, right))
                .map(|change| JointRietveldTopologyChange {
                    histogram_id: after.histogram_id.clone(),
                    change,
                })
        })
        .collect()
}

fn validate_constraint_state(
    parameters: &ParameterSet,
    transform: &ConstraintTransform,
) -> Result<(), JointRietveldRefinementError> {
    let constrained = transform.unpack(&transform.pack()?, false)?;
    for spec in parameters.specs() {
        let value = constrained
            .get(spec.key())
            .copied()
            .ok_or(JointRietveldRefinementError::InternalInvariant)?;
        if (value - spec.value()).abs() > 2.0e-12 {
            return Err(JointRietveldRefinementError::UnsatisfiedConstraint {
                key: spec.key().clone(),
            });
        }
    }
    Ok(())
}

fn parameter_values_match(domain: &ParameterSet, stored: &ParameterSet) -> bool {
    domain.specs().len() == stored.specs().len()
        && domain
            .specs()
            .iter()
            .zip(stored.specs())
            .all(|(domain, stored)| {
                domain.key() == stored.key() && domain.value().to_bits() == stored.value().to_bits()
            })
}

fn parameter_contract_matches(domain: &ParameterSet, stored: &ParameterSet) -> bool {
    domain.specs().len() == stored.specs().len()
        && domain
            .specs()
            .iter()
            .zip(stored.specs())
            .all(|(domain, stored)| {
                domain.key() == stored.key()
                    && domain.unit() == stored.unit()
                    && domain.bounds() == stored.bounds()
                    && domain.refine() == stored.refine()
                    && domain.scale().to_bits() == stored.scale().to_bits()
            })
}

fn valid_history(history: &[JointRietveldIterationRecord]) -> bool {
    history.iter().enumerate().all(|(index, row)| {
        row.iteration == index + 1
            && row.objective.is_finite()
            && row.objective >= 0.0
            && row.objective_change.is_finite()
            && row.objective_change >= 0.0
            && row.scaled_step_norm.is_finite()
            && row.scaled_step_norm >= 0.0
            && row.damping.is_finite()
            && row.damping > 0.0
            && row.metrics.chi_square.is_finite()
            && row.metrics.chi_square >= 0.0
            && !row.metrics.rp.is_nan()
            && row.metrics.rp >= 0.0
            && !row.metrics.rwp.is_nan()
            && row.metrics.rwp >= 0.0
            && !row.metrics.reduced_chi_square.is_nan()
            && row.metrics.reduced_chi_square >= 0.0
            && row.parameter_changes.iter().all(|change| {
                change.before.is_finite()
                    && change.after.is_finite()
                    && change.scaled_change.is_finite()
            })
    })
}

fn forward_product(derivative: &ConstraintDerivativeMatrix, free: &[f64]) -> Vec<f64> {
    debug_assert_eq!(free.len(), derivative.columns);
    derivative
        .values
        .chunks_exact(derivative.columns)
        .map(|row| row.iter().zip(free).map(|(left, right)| left * right).sum())
        .collect()
}

fn transpose_product(derivative: &ConstraintDerivativeMatrix, physical: &[f64]) -> Vec<f64> {
    debug_assert_eq!(physical.len(), derivative.rows);
    let mut result = vec![0.0; derivative.columns];
    for (coefficient, row) in physical
        .iter()
        .zip(derivative.values.chunks_exact(derivative.columns))
    {
        for (target, value) in result.iter_mut().zip(row) {
            *target += coefficient * value;
        }
    }
    result
}

fn reserve_products(
    runtime: &mut RefinementRuntime<JointRietveldCheckpoint>,
    count: usize,
) -> Result<(), JointRietveldRefinementError> {
    for _ in 0..count {
        runtime.begin_evaluation()?;
    }
    Ok(())
}

fn conjugate_gradient(
    right_hand_side: &[f64],
    tolerance: f64,
    max_iterations: usize,
    operator: impl FnMut(&[f64]) -> Result<Vec<f64>, JointRietveldRefinementError>,
) -> Result<(Vec<f64>, usize), JointRietveldRefinementError> {
    conjugate_gradient_core(right_hand_side, tolerance, max_iterations, operator).map_err(|error| {
        match error {
            ConjugateGradientError::Operator(error) => error,
            ConjugateGradientError::NonPositiveOperator
            | ConjugateGradientError::NonFiniteState => {
                JointRietveldRefinementError::NumericalBreakdown
            }
        }
    })
}

fn emit_rejected_trial(
    runtime: &mut RefinementRuntime<JointRietveldCheckpoint>,
    reason: &str,
) -> Result<(), JointRietveldRefinementError> {
    runtime.emit(
        RefinementEventKind::StepRejected,
        "joint_rietveld_step",
        "native joint Rietveld trial rejected",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(reason.to_owned()),
        )],
    )?;
    Ok(())
}

fn normal_stop(error: RuntimeError) -> Result<TerminationReason, JointRietveldRefinementError> {
    match error {
        RuntimeError::Stopped(stop) => Ok(stop.reason),
        other => Err(other.into()),
    }
}

/// Invalid joint constrained-refinement state.
#[derive(Debug)]
pub enum JointRietveldRefinementError {
    /// Numerical controls are invalid.
    InvalidOptions,
    /// Restart state is inconsistent with the request or constraints.
    InvalidCheckpoint {
        /// Stable diagnostic explaining the rejected checkpoint.
        reason: &'static str,
    },
    /// Supplied physical state does not satisfy its constraint graph.
    UnsatisfiedConstraint {
        /// First inconsistent physical identity.
        key: ParameterKey,
    },
    /// A validated parameter mapping invariant failed.
    InternalInvariant,
    /// Conjugate gradient encountered a non-positive or non-finite curvature.
    NumericalBreakdown,
    /// Joint layout or objective state failed.
    Joint(JointRietveldError),
    /// Stable scalar parameter state failed.
    Parameter(ParameterError),
    /// Constraint graph or transform failed.
    Constraint(ConstraintError),
    /// One histogram calculation failed.
    Rietveld(crate::RietveldError),
    /// Runtime boundary failed.
    Runtime(RuntimeError),
}

impl Display for JointRietveldRefinementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => {
                formatter.write_str("native joint Rietveld solver options are invalid")
            }
            Self::InvalidCheckpoint { reason } => {
                write!(
                    formatter,
                    "native joint Rietveld checkpoint is invalid: {reason}"
                )
            }
            Self::UnsatisfiedConstraint { key } => {
                write!(
                    formatter,
                    "joint Rietveld parameter {key} does not satisfy its constraint"
                )
            }
            Self::InternalInvariant => {
                formatter.write_str("native joint Rietveld parameter invariant failed")
            }
            Self::NumericalBreakdown => {
                formatter.write_str("native joint Rietveld linear solve broke down")
            }
            Self::Joint(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for JointRietveldRefinementError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Joint(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Constraint(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::InvalidOptions
            | Self::InvalidCheckpoint { .. }
            | Self::UnsatisfiedConstraint { .. }
            | Self::InternalInvariant
            | Self::NumericalBreakdown => None,
        }
    }
}

impl From<JointRietveldError> for JointRietveldRefinementError {
    fn from(value: JointRietveldError) -> Self {
        Self::Joint(value)
    }
}

impl From<ParameterError> for JointRietveldRefinementError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}

impl From<ConstraintError> for JointRietveldRefinementError {
    fn from(value: ConstraintError) -> Self {
        Self::Constraint(value)
    }
}

impl From<crate::RietveldError> for JointRietveldRefinementError {
    fn from(value: crate::RietveldError) -> Self {
        Self::Rietveld(value)
    }
}

impl From<RuntimeError> for JointRietveldRefinementError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
