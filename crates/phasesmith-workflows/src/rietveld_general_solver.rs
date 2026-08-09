//! Constraint-aware bounded solver over the complete native Rietveld layout.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::DMatrix;

use crate::rietveld_solver::{
    conjugate_gradient, norm, normal_stop, reserve_products, topology_change,
};
use crate::{
    BackgroundModel, CancellationToken, Constraint, ConstraintDerivativeMatrix, ConstraintError,
    ConstraintTransform, DiagnosticValue, LatticeBounds, ParameterChange, ParameterKey,
    ParameterSet, PreparedGeneralRietveldObjective, RefinementEventKind, RefinementRuntime,
    ResidualOptions, RietveldCalculation, RietveldGeneralParameterError, RietveldInput,
    RietveldInstrumentParameter, RietveldIterationRecord, RietveldParameterLayout,
    RietveldParameterSelection, RietveldRefinementError, RietveldRefinementOptions, RuntimeError,
    TerminationReason, calculate_rietveld_pattern, evaluate_residuals,
};

/// Controls for the optional small parameter-space identifiability analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RietveldCovarianceOptions {
    /// Whether final Jacobian diagnostics are evaluated.
    pub enabled: bool,
    /// Maximum scaled-free dimension allowed for explicit diagnostics.
    pub max_parameters: usize,
    /// Absolute column correlation reported as unresolved.
    pub unresolved_correlation: f64,
}

impl RietveldCovarianceOptions {
    /// Construct validated covariance controls.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralRefinementError::InvalidCovarianceOptions`]
    /// for a zero dimension or threshold outside `[0, 1]`.
    pub fn new(
        enabled: bool,
        max_parameters: usize,
        unresolved_correlation: f64,
    ) -> Result<Self, RietveldGeneralRefinementError> {
        if max_parameters == 0
            || !unresolved_correlation.is_finite()
            || !(0.0..=1.0).contains(&unresolved_correlation)
        {
            return Err(RietveldGeneralRefinementError::InvalidCovarianceOptions);
        }
        Ok(Self {
            enabled,
            max_parameters,
            unresolved_correlation,
        })
    }
}

impl Default for RietveldCovarianceOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parameters: 64,
            unresolved_correlation: 1.0 - 1.0e-10,
        }
    }
}

/// One nearly collinear pair of scaled-free Jacobian columns.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldParameterCorrelation {
    /// First stable free-parameter identity.
    pub left: ParameterKey,
    /// Second stable free-parameter identity.
    pub right: ParameterKey,
    /// Signed normalized weighted-column dot product.
    pub correlation: f64,
}

/// Square row-major covariance over the complete physical parameter order.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldCovarianceMatrix {
    /// Matrix dimension, equal to the complete physical parameter count.
    pub size: usize,
    /// Row-major values with length `size * size`.
    pub values: Vec<f64>,
}

/// Complete accepted state for deterministic constrained continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldGeneralCheckpoint {
    /// Accepted iteration count.
    pub completed_iterations: usize,
    /// Last accepted complete request state.
    pub input: RietveldInput,
    /// Selection contract used to build the complete layout.
    pub selection: RietveldParameterSelection,
    /// Per-phase lattice bounds in phase order.
    pub lattice_bounds: Vec<Option<LatticeBounds>>,
    /// Ordered constraint graph.
    pub constraints: Vec<Constraint>,
    /// Last accepted complete physical parameters.
    pub parameters: ParameterSet,
    /// Last accepted objective.
    pub objective: f64,
    /// Damping for the next attempted iteration.
    pub damping: f64,
    /// Complete accepted history.
    pub history: Vec<RietveldIterationRecord>,
}

impl RietveldGeneralCheckpoint {
    /// Revalidate this accepted state against its requested solver contract.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralRefinementError`] when the checkpoint is
    /// internally invalid or no longer matches the requested input contract.
    pub fn validate_for(
        &self,
        requested: &RietveldInput,
        selection: &RietveldParameterSelection,
        lattice_bounds: &[Option<LatticeBounds>],
        constraints: &[Constraint],
    ) -> Result<(), RietveldGeneralRefinementError> {
        self.input.validate()?;
        let invalid = if &self.selection != selection {
            Some("parameter selection changed")
        } else if self.lattice_bounds != lattice_bounds {
            Some("lattice bounds changed")
        } else if self.constraints != constraints {
            Some("constraints changed")
        } else if !checkpoint_request_compatible(&self.input, requested, selection) {
            Some("request contract changed")
        } else if self.completed_iterations != self.history.len() {
            Some("accepted iteration count does not match history")
        } else if !self.objective.is_finite() || self.objective < 0.0 {
            Some("objective is invalid")
        } else if !self.damping.is_finite() || self.damping <= 0.0 {
            Some("damping is invalid")
        } else if !valid_history(&self.history, requested) {
            Some("history is invalid")
        } else {
            None
        };
        if let Some(reason) = invalid {
            return Err(RietveldGeneralRefinementError::InvalidCheckpoint { reason });
        }
        let accepted_layout = RietveldParameterLayout::new(&self.input, selection, lattice_bounds)?;
        let requested_layout = RietveldParameterLayout::new(requested, selection, lattice_bounds)?;
        if !parameter_contract_matches(requested_layout.parameters(), &self.parameters)
            || !parameter_values_match(accepted_layout.parameters(), &self.parameters)
        {
            return Err(RietveldGeneralRefinementError::InvalidCheckpoint {
                reason: "parameter contract changed",
            });
        }
        let transform = ConstraintTransform::new(self.parameters.clone(), constraints.to_vec())?;
        validate_constraint_state(&accepted_layout, &transform)?;
        Ok(())
    }
}

/// Final complete native constrained-refinement result.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldGeneralRefinementResult {
    /// Final display-ready calculation.
    pub calculation: RietveldCalculation,
    /// Final accepted complete request state.
    pub input: RietveldInput,
    /// Final complete physical parameters.
    pub parameters: ParameterSet,
    /// Stable scaled-free parameter order.
    pub free_keys: Vec<ParameterKey>,
    /// Complete accepted history.
    pub history: Vec<RietveldIterationRecord>,
    /// Stable bounded termination category.
    pub termination_reason: TerminationReason,
    /// Final restart checkpoint.
    pub checkpoint: RietveldGeneralCheckpoint,
    /// Model-product/evaluation count for this call.
    pub evaluations: usize,
    /// Rank of the final weighted scaled-free normal matrix when evaluated.
    pub jacobian_rank: Option<usize>,
    /// Full physical-parameter covariance when the free normal matrix is full rank.
    pub covariance: Option<RietveldCovarianceMatrix>,
    /// Nearly collinear scaled-free column pairs.
    pub unresolved_correlations: Vec<RietveldParameterCorrelation>,
}

/// Refine the complete native layout with a process-local runtime.
///
/// # Errors
///
/// Returns [`RietveldGeneralRefinementError`] for invalid layouts, constraints,
/// checkpoints, runtime state, or numerical products.
#[allow(clippy::too_many_arguments)]
pub fn refine_general_rietveld(
    input: &RietveldInput,
    selection: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    options: &RietveldRefinementOptions,
    covariance: RietveldCovarianceOptions,
    checkpoint: Option<&RietveldGeneralCheckpoint>,
    cancellation: Option<CancellationToken>,
) -> Result<RietveldGeneralRefinementResult, RietveldGeneralRefinementError> {
    let mut runtime = RefinementRuntime::new(options.limits, cancellation)?;
    refine_general_rietveld_with_runtime(
        input,
        selection,
        lattice_bounds,
        constraints,
        options,
        covariance,
        checkpoint,
        &mut runtime,
    )
}

/// Refine the complete native layout through a caller-owned runtime.
///
/// # Errors
///
/// Returns [`RietveldGeneralRefinementError`] for invalid state or numerical
/// products. Normal bounded stops return the last accepted state.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn refine_general_rietveld_with_runtime(
    input: &RietveldInput,
    selection: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    options: &RietveldRefinementOptions,
    covariance: RietveldCovarianceOptions,
    checkpoint: Option<&RietveldGeneralCheckpoint>,
    runtime: &mut RefinementRuntime<RietveldGeneralCheckpoint>,
) -> Result<RietveldGeneralRefinementResult, RietveldGeneralRefinementError> {
    input.validate()?;
    options.validate()?;
    covariance.validate()?;
    let initial_layout = RietveldParameterLayout::new(input, selection, lattice_bounds)?;
    let initial_transform =
        ConstraintTransform::new(initial_layout.parameters().clone(), constraints.to_vec())?;
    validate_constraint_state(&initial_layout, &initial_transform)?;
    let (mut live_input, mut history, mut damping, parameter_template) =
        if let Some(checkpoint) = checkpoint {
            checkpoint.validate_for(input, selection, lattice_bounds, constraints)?;
            runtime.resume_accepted(checkpoint.completed_iterations)?;
            (
                checkpoint.input.clone(),
                checkpoint.history.clone(),
                checkpoint.damping,
                checkpoint.parameters.clone(),
            )
        } else {
            (
                input.clone(),
                Vec::new(),
                options.initial_damping,
                initial_layout.parameters().clone(),
            )
        };
    runtime.emit(
        RefinementEventKind::Start,
        "rietveld",
        "native complete Rietveld refinement started",
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
        let layout = RietveldParameterLayout::new(&live_input, selection, lattice_bounds)?;
        let solver_parameters = parameter_template
            .replace_values(&layout.parameters().values())
            .map_err(RietveldGeneralParameterError::Parameter)?;
        let transform = ConstraintTransform::new(solver_parameters.clone(), constraints.to_vec())?;
        if transform.free_keys().is_empty() {
            termination = TerminationReason::Converged;
            break;
        }
        let derivative = transform.derivative_matrix()?;
        let objective = PreparedGeneralRietveldObjective::new(
            live_input.clone(),
            options.calculation.clone(),
            layout.clone(),
        )?;
        if let Err(error) = reserve_products(runtime, objective.preparation_evaluation_count()) {
            termination = normal_stop(&error)?;
            break;
        }
        let (_, physical_gradient) = objective.gradient()?;
        let scaled_gradient = transpose_product(&derivative, &physical_gradient);
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
            Err(RietveldRefinementError::Runtime(RuntimeError::Stopped(stop))) => {
                termination = stop.reason;
                break 'iterations;
            }
            Err(error) => return Err(error.into()),
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
        let current_calculation = objective.calculation().clone();
        let current_objective = 0.5 * current_calculation.metrics.chi_square;
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
                        termination = normal_stop(&error)?;
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
                        .ok_or(RietveldGeneralRefinementError::InternalInvariant)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let Ok(trial_input) = layout.apply_values(&live_input, &trial_values) else {
                emit_rejected_trial(runtime, "trial outside the numerical model domain")?;
                if let Err(error) = runtime.reject_step() {
                    termination = normal_stop(&error)?;
                    break 'iterations;
                }
                continue;
            };
            if let Err(error) = runtime.begin_evaluation() {
                termination = normal_stop(&error)?;
                break 'iterations;
            }
            let Ok(trial_calculation) =
                calculate_rietveld_pattern(&trial_input, &options.calculation)
            else {
                emit_rejected_trial(runtime, "trial outside the calculation domain")?;
                if let Err(error) = runtime.reject_step() {
                    termination = normal_stop(&error)?;
                    break 'iterations;
                }
                continue;
            };
            let trial_objective = 0.5 * trial_calculation.metrics.chi_square;
            runtime.emit(
                RefinementEventKind::Trial,
                "rietveld_step",
                "native complete Rietveld trial evaluated",
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
                    trial_input,
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
        let Some((backtracks, factor, trial_values, trial_input, trial_calculation, objective)) =
            accepted
        else {
            if termination == TerminationReason::MaxIterations {
                termination = TerminationReason::Stagnated;
            }
            damping *= options.damping_increase;
            break;
        };
        let objective_change = current_objective - objective;
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
            .collect::<Vec<_>>();
        let topology_changes = live_input
            .phases
            .iter()
            .zip(&trial_input.phases)
            .filter_map(|(before, after)| topology_change(before, after))
            .collect::<Vec<_>>();
        let accepted_metrics = evaluate_residuals(
            &input.pattern,
            &trial_calculation.y,
            ResidualOptions {
                use_uncertainty: options.calculation.use_uncertainty,
                parameter_count: transform.free_keys().len(),
            },
        )?;
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
            rwp: accepted_metrics.rwp,
            rp: accepted_metrics.rp,
            chi_square: accepted_metrics.chi_square,
            reduced_chi_square: accepted_metrics.reduced_chi_square,
        });
        live_input = trial_input;
        damping = (damping * options.damping_decrease).max(1.0e-18);
        let accepted_layout = RietveldParameterLayout::new(&live_input, selection, lattice_bounds)?;
        let accepted_parameters = parameter_template
            .replace_values(&accepted_layout.parameters().values())
            .map_err(RietveldGeneralParameterError::Parameter)?;
        let state = RietveldGeneralCheckpoint {
            completed_iterations: history.len(),
            input: live_input.clone(),
            selection: selection.clone(),
            lattice_bounds: lattice_bounds.to_vec(),
            constraints: constraints.to_vec(),
            parameters: accepted_parameters,
            objective,
            damping,
            history: history.clone(),
        };
        runtime.accept_step(Some(&state))?;
        runtime.emit(
            RefinementEventKind::StepAccepted,
            "rietveld_step",
            "native complete Rietveld step accepted",
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
    let final_layout = RietveldParameterLayout::new(&live_input, selection, lattice_bounds)?;
    let final_parameters = parameter_template
        .replace_values(&final_layout.parameters().values())
        .map_err(RietveldGeneralParameterError::Parameter)?;
    let final_transform = ConstraintTransform::new(final_parameters.clone(), constraints.to_vec())?;
    runtime.begin_evaluation().or_else(|error| match error {
        RuntimeError::Stopped(_) => Ok(()),
        other => Err(other),
    })?;
    let mut calculation = calculate_rietveld_pattern(&live_input, &options.calculation)?;
    calculation.metrics = evaluate_residuals(
        &input.pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.calculation.use_uncertainty,
            parameter_count: final_transform.free_keys().len(),
        },
    )?;
    let checkpoint = RietveldGeneralCheckpoint {
        completed_iterations: history.len(),
        input: live_input.clone(),
        selection: selection.clone(),
        lattice_bounds: lattice_bounds.to_vec(),
        constraints: constraints.to_vec(),
        parameters: final_parameters.clone(),
        objective: 0.5 * calculation.metrics.chi_square,
        damping,
        history: history.clone(),
    };
    checkpoint.validate_for(input, selection, lattice_bounds, constraints)?;
    let diagnostics = match covariance_diagnostics(
        &live_input,
        &final_layout,
        &final_transform,
        options,
        covariance,
        &calculation,
        runtime,
    ) {
        Ok(value) => value,
        Err(RietveldGeneralRefinementError::Runtime(RuntimeError::Stopped(_))) => {
            CovarianceDiagnostics::default()
        }
        Err(error) => return Err(error),
    };
    runtime.emit(
        RefinementEventKind::Termination,
        "rietveld",
        "native complete Rietveld refinement terminated",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(RietveldGeneralRefinementResult {
        calculation,
        input: live_input,
        parameters: final_parameters,
        free_keys: final_transform.free_keys().to_vec(),
        history,
        termination_reason: termination,
        checkpoint,
        evaluations: runtime.evaluations(),
        jacobian_rank: diagnostics.rank,
        covariance: diagnostics.covariance,
        unresolved_correlations: diagnostics.correlations,
    })
}

impl RietveldCovarianceOptions {
    fn validate(self) -> Result<(), RietveldGeneralRefinementError> {
        Self::new(
            self.enabled,
            self.max_parameters,
            self.unresolved_correlation,
        )
        .map(|_| ())
    }
}

fn emit_rejected_trial(
    runtime: &mut RefinementRuntime<RietveldGeneralCheckpoint>,
    reason: &str,
) -> Result<(), RietveldGeneralRefinementError> {
    runtime.emit(
        RefinementEventKind::StepRejected,
        "rietveld_step",
        "native complete Rietveld trial rejected",
        vec![(
            "reason".to_owned(),
            DiagnosticValue::String(reason.to_owned()),
        )],
    )?;
    Ok(())
}

fn validate_constraint_state(
    layout: &RietveldParameterLayout,
    transform: &ConstraintTransform,
) -> Result<(), RietveldGeneralRefinementError> {
    let constrained = transform.unpack(&transform.pack()?, false)?;
    for spec in layout.parameters().specs() {
        let value = constrained
            .get(spec.key())
            .copied()
            .ok_or(RietveldGeneralRefinementError::InternalInvariant)?;
        if (value - spec.value()).abs() > 2.0e-12 {
            return Err(RietveldGeneralRefinementError::UnsatisfiedConstraint {
                key: spec.key().clone(),
            });
        }
    }
    Ok(())
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

#[derive(Default)]
struct CovarianceDiagnostics {
    rank: Option<usize>,
    covariance: Option<RietveldCovarianceMatrix>,
    correlations: Vec<RietveldParameterCorrelation>,
}

#[allow(clippy::too_many_arguments)]
fn covariance_diagnostics(
    input: &RietveldInput,
    layout: &RietveldParameterLayout,
    transform: &ConstraintTransform,
    options: &RietveldRefinementOptions,
    covariance_options: RietveldCovarianceOptions,
    calculation: &RietveldCalculation,
    runtime: &mut RefinementRuntime<RietveldGeneralCheckpoint>,
) -> Result<CovarianceDiagnostics, RietveldGeneralRefinementError> {
    let free_count = transform.free_keys().len();
    if !covariance_options.enabled
        || free_count == 0
        || free_count > covariance_options.max_parameters
    {
        return Ok(CovarianceDiagnostics::default());
    }
    let objective = PreparedGeneralRietveldObjective::new(
        input.clone(),
        options.calculation.clone(),
        layout.clone(),
    )?;
    reserve_products(runtime, objective.preparation_evaluation_count())?;
    let derivative = transform.derivative_matrix()?;
    let sample_count = input.pattern.sample_count();
    let element_count = sample_count
        .checked_mul(free_count)
        .ok_or(RietveldGeneralRefinementError::AllocationOverflow)?;
    let mut columns = vec![0.0; element_count];
    for column in 0..free_count {
        reserve_products(runtime, objective.jvp_evaluation_count())?;
        let mut basis = vec![0.0; free_count];
        basis[column] = 1.0;
        let physical = forward_product(&derivative, &basis);
        let (_, values) = objective.jvp(&physical)?;
        for (sample, value) in values.into_iter().enumerate() {
            let included = input.pattern.mask.as_ref().is_none_or(|mask| mask[sample]);
            let weighted = if !included {
                0.0
            } else if options.calculation.use_uncertainty {
                input
                    .pattern
                    .uncertainty
                    .as_ref()
                    .map_or(value, |sigma| value / sigma[sample])
            } else {
                value
            };
            columns[sample * free_count + column] = weighted;
        }
    }
    let jacobian = DMatrix::from_row_slice(sample_count, free_count, &columns);
    let normal = jacobian.transpose() * &jacobian;
    let rank = matrix_rank(&normal);
    let correlations = unresolved_correlations(
        &columns,
        sample_count,
        transform.free_keys(),
        covariance_options.unresolved_correlation,
    );
    if rank != free_count {
        return Ok(CovarianceDiagnostics {
            rank: Some(rank),
            covariance: None,
            correlations,
        });
    }
    let Some(mut free_covariance) = normal.try_inverse() else {
        return Ok(CovarianceDiagnostics {
            rank: Some(rank),
            covariance: None,
            correlations,
        });
    };
    if calculation.metrics.reduced_chi_square.is_finite() {
        free_covariance *= calculation.metrics.reduced_chi_square;
    }
    let covariance_count = derivative
        .rows
        .checked_mul(derivative.rows)
        .ok_or(RietveldGeneralRefinementError::AllocationOverflow)?;
    let chain = DMatrix::from_row_slice(derivative.rows, derivative.columns, &derivative.values);
    let physical = &chain * free_covariance * chain.transpose();
    let mut values = Vec::with_capacity(covariance_count);
    for row in 0..physical.nrows() {
        for column in 0..physical.ncols() {
            values.push(physical[(row, column)]);
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Ok(CovarianceDiagnostics {
            rank: Some(rank),
            covariance: None,
            correlations,
        });
    }
    Ok(CovarianceDiagnostics {
        rank: Some(rank),
        covariance: Some(RietveldCovarianceMatrix {
            size: derivative.rows,
            values,
        }),
        correlations,
    })
}

fn unresolved_correlations(
    columns: &[f64],
    sample_count: usize,
    free_keys: &[ParameterKey],
    threshold: f64,
) -> Vec<RietveldParameterCorrelation> {
    let free_count = free_keys.len();
    let norms = (0..free_count)
        .map(|column| {
            (0..sample_count)
                .map(|sample| columns[sample * free_count + column].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    for left in 0..free_count {
        if norms[left] == 0.0 {
            continue;
        }
        for right in left + 1..free_count {
            if norms[right] == 0.0 {
                continue;
            }
            let correlation = (0..sample_count)
                .map(|sample| {
                    columns[sample * free_count + left] * columns[sample * free_count + right]
                })
                .sum::<f64>()
                / (norms[left] * norms[right]);
            let correlation = correlation.clamp(-1.0, 1.0);
            if correlation.abs() >= threshold {
                result.push(RietveldParameterCorrelation {
                    left: free_keys[left].clone(),
                    right: free_keys[right].clone(),
                    correlation,
                });
            }
        }
    }
    result
}

fn matrix_rank(matrix: &DMatrix<f64>) -> usize {
    let singular = matrix.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0_f64, f64::max);
    let dimension = u32::try_from(matrix.nrows().max(matrix.ncols())).unwrap_or(u32::MAX);
    let tolerance = f64::from(dimension) * f64::EPSILON * maximum;
    singular.iter().filter(|value| **value > tolerance).count()
}

fn checkpoint_request_compatible(
    accepted: &RietveldInput,
    requested: &RietveldInput,
    selection: &RietveldParameterSelection,
) -> bool {
    accepted.pattern == requested.pattern
        && accepted.axial_geometry == requested.axial_geometry
        && accepted.phases.len() == requested.phases.len()
        && accepted
            .phases
            .iter()
            .zip(&requested.phases)
            .all(|(accepted, requested)| {
                accepted.restart_compatible_with_wavelength(
                    requested,
                    selection
                        .instrument
                        .contains(&RietveldInstrumentParameter::WavelengthAngstrom),
                ) && phase_values_compatible(accepted, requested, selection)
            })
        && background_compatible(
            accepted.background.as_ref(),
            requested.background.as_ref(),
            selection.background,
        )
        && instrument_compatible(accepted, requested, selection)
}

fn phase_values_compatible(
    accepted: &crate::RietveldPhase,
    requested: &crate::RietveldPhase,
    selection: &RietveldParameterSelection,
) -> bool {
    let left = accepted.definition();
    let right = requested.definition();
    (selection.structural.phase_scale || left.scale.to_bits() == right.scale.to_bits())
        && (selection.structural.lattice || left.cell == right.cell)
        && (selection.structural.coordinates || left.fractional_xyz == right.fractional_xyz)
        && (selection.structural.occupancy || left.occupancy == right.occupancy)
        && (selection.structural.u_iso || left.u_iso_angstrom2 == right.u_iso_angstrom2)
        && (selection.sample_physics || accepted.sample_physics() == requested.sample_physics())
        && (accepted.reflection_domain().is_some()
            || accepted.sample_physics().is_some()
            || accepted.contributions() == requested.contributions())
}

fn background_compatible(
    accepted: Option<&BackgroundModel>,
    requested: Option<&BackgroundModel>,
    selected: bool,
) -> bool {
    match (accepted, requested) {
        (None, None) => true,
        (Some(accepted), Some(requested)) => {
            if selected {
                accepted.restart_compatible(requested)
            } else {
                accepted == requested
            }
        }
        _ => false,
    }
}

fn instrument_compatible(
    accepted: &RietveldInput,
    requested: &RietveldInput,
    selection: &RietveldParameterSelection,
) -> bool {
    let selected = |parameter| selection.instrument.contains(&parameter);
    let left = accepted.instrument;
    let right = requested.instrument;
    (selected(RietveldInstrumentParameter::UDeg2)
        || left.u_deg2.to_bits() == right.u_deg2.to_bits())
        && (selected(RietveldInstrumentParameter::VDeg2)
            || left.v_deg2.to_bits() == right.v_deg2.to_bits())
        && (selected(RietveldInstrumentParameter::WDeg2)
            || left.w_deg2.to_bits() == right.w_deg2.to_bits())
        && (selected(RietveldInstrumentParameter::XDeg)
            || left.x_deg.to_bits() == right.x_deg.to_bits())
        && (selected(RietveldInstrumentParameter::YDeg)
            || left.y_deg.to_bits() == right.y_deg.to_bits())
        && (selected(RietveldInstrumentParameter::WavelengthAngstrom)
            || left.wavelength_angstrom.to_bits() == right.wavelength_angstrom.to_bits())
        && position_correction_compatible(accepted, requested, &selected)
}

fn position_correction_compatible(
    accepted: &RietveldInput,
    requested: &RietveldInput,
    selected: &impl Fn(RietveldInstrumentParameter) -> bool,
) -> bool {
    let left = accepted.position_correction;
    let right = requested.position_correction;
    if !selected(RietveldInstrumentParameter::ZeroShiftDeg)
        && left.zero_shift_deg.to_bits() != right.zero_shift_deg.to_bits()
    {
        return false;
    }
    let bragg = match (left.bragg_brentano_mm, right.bragg_brentano_mm) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.1.to_bits() == right.1.to_bits()
                && (selected(RietveldInstrumentParameter::SampleDisplacementMm)
                    || left.0.to_bits() == right.0.to_bits())
        }
        _ => false,
    };
    let debye = match (
        left.debye_scherrer_micrometre,
        right.debye_scherrer_micrometre,
    ) {
        (None, None) => true,
        (Some(left), Some(right)) => {
            left.2.to_bits() == right.2.to_bits()
                && (selected(RietveldInstrumentParameter::DisplaceXMicrometre)
                    || left.0.to_bits() == right.0.to_bits())
                && (selected(RietveldInstrumentParameter::DisplaceYMicrometre)
                    || left.1.to_bits() == right.1.to_bits())
        }
        _ => false,
    };
    bragg && debye
}

fn valid_history(history: &[RietveldIterationRecord], input: &RietveldInput) -> bool {
    let phase_ids = input
        .phases
        .iter()
        .map(crate::RietveldPhase::phase_id)
        .collect::<std::collections::BTreeSet<_>>();
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
            && row.rwp.is_finite()
            && row.rp.is_finite()
            && row.chi_square.is_finite()
            && row.chi_square >= 0.0
            && !row.reduced_chi_square.is_nan()
            && row.reduced_chi_square >= 0.0
            && row.parameter_changes.iter().all(|change| {
                change.before.is_finite()
                    && change.after.is_finite()
                    && change.scaled_change.is_finite()
            })
            && row.topology_changes.iter().all(|change| {
                let added = change
                    .added_reflection_ids
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>();
                let removed = change
                    .removed_reflection_ids
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>();
                phase_ids.contains(&change.phase_id)
                    && added.len() == change.added_reflection_ids.len()
                    && removed.len() == change.removed_reflection_ids.len()
                    && added.is_disjoint(&removed)
            })
    })
}

/// Invalid complete constrained Rietveld refinement state.
#[derive(Debug)]
pub enum RietveldGeneralRefinementError {
    /// Covariance controls are invalid.
    InvalidCovarianceOptions,
    /// Restart state is inconsistent with the request or solver contract.
    InvalidCheckpoint {
        /// Stable diagnostic explaining the rejected checkpoint contract.
        reason: &'static str,
    },
    /// Checked diagnostic allocation overflowed.
    AllocationOverflow,
    /// Internal validated layout invariant failed.
    InternalInvariant,
    /// The supplied physical state does not satisfy its constraint graph.
    UnsatisfiedConstraint {
        /// First inconsistent physical identity.
        key: ParameterKey,
    },
    /// Existing bounded solver or product failed.
    Refinement(RietveldRefinementError),
    /// Complete parameter layout or installation failed.
    Parameter(RietveldGeneralParameterError),
    /// Constraint graph or transform failed.
    Constraint(ConstraintError),
    /// Owned calculation state failed.
    Rietveld(crate::RietveldError),
    /// Complete objective failed.
    Objective(crate::RietveldGeneralObjectiveError),
    /// Runtime boundary failed.
    Runtime(RuntimeError),
    /// Residual evaluation failed.
    Residual(crate::ResidualError),
}

impl Display for RietveldGeneralRefinementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCovarianceOptions => {
                formatter.write_str("native Rietveld covariance options are invalid")
            }
            Self::InvalidCheckpoint { reason } => write!(
                formatter,
                "native complete Rietveld checkpoint is invalid: {reason}"
            ),
            Self::AllocationOverflow => {
                formatter.write_str("native Rietveld diagnostic allocation overflowed")
            }
            Self::InternalInvariant => {
                formatter.write_str("native complete Rietveld invariant failed")
            }
            Self::UnsatisfiedConstraint { key } => write!(
                formatter,
                "initial physical value does not satisfy the constraint for {}",
                key.label()
            ),
            Self::Refinement(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Objective(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RietveldGeneralRefinementError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Refinement(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Constraint(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Objective(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::InvalidCovarianceOptions
            | Self::InvalidCheckpoint { .. }
            | Self::AllocationOverflow
            | Self::InternalInvariant
            | Self::UnsatisfiedConstraint { .. } => None,
        }
    }
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for RietveldGeneralRefinementError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(RietveldRefinementError, Refinement);
from_error!(RietveldGeneralParameterError, Parameter);
from_error!(ConstraintError, Constraint);
from_error!(crate::RietveldError, Rietveld);
from_error!(crate::RietveldGeneralObjectiveError, Objective);
from_error!(RuntimeError, Runtime);
from_error!(crate::ResidualError, Residual);
