//! Deterministic bounded dense and matrix-free Pawley Gauss–Newton solvers.
// Matrix equations use conventional short names.
#![allow(clippy::many_single_char_names)]
use crate::pawley::{err, weighted};
use crate::pawley_support::{support_guard, support_membership};
use crate::{
    ConstraintTransform, PawleyError, PawleyEvaluation, PawleyInput, RefinementEventKind,
    RefinementLimits, RefinementRuntime, RuntimeError, TerminationReason,
    evaluate_pawley_with_storage,
};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Linear algebra used for bounded Gauss–Newton steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PawleySolver {
    /// Column-scaled dense QR reference solver.
    #[default]
    Dense,
    /// Support-block products and projected conjugate-gradient face solves.
    MatrixFree,
}
/// Scientific and allocation controls. Runtime budgets are separate.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyOptions {
    /// Dense reference or matrix-free products.
    pub solver: PawleySolver,
    /// Relative residual tolerance for iterative linear face solves.
    pub linear_tolerance: f64,
    /// Iterative linear solve ceiling per active face.
    pub max_linear_iterations: usize,
    /// Exact finite-support multiplier.
    pub support_fwhm: f64,
    /// Use supplied standard deviations.
    pub use_uncertainty: bool,
    /// Upper bound on estimated dense floating-point workspace elements.
    pub max_elements: usize,
    /// Relative singular-value threshold on column-normalized Jacobians.
    pub rank_tolerance: f64,
    /// Normalized feasible-step convergence threshold.
    pub tolerance: f64,
    /// Initial positive damping; decreases after accepted steps.
    pub damping: f64,
    /// Active-set subproblem iteration ceiling.
    pub max_active_iterations: usize,
}
impl Default for PawleyOptions {
    fn default() -> Self {
        Self {
            solver: PawleySolver::Dense,
            linear_tolerance: 1e-11,
            max_linear_iterations: 4000,
            support_fwhm: 20.0,
            use_uncertainty: true,
            max_elements: 50_000_000,
            rank_tolerance: 1e-10,
            tolerance: 1e-9,
            damping: 1e-6,
            max_active_iterations: 2000,
        }
    }
}
impl PawleyOptions {
    /// Validate controls before allocating or evaluating.
    ///
    /// # Errors
    /// Rejects nonfinite, nonpositive or out-of-range controls.
    pub fn validate(&self) -> Result<(), PawleyError> {
        if ![
            self.support_fwhm,
            self.rank_tolerance,
            self.tolerance,
            self.damping,
            self.linear_tolerance,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            || self.rank_tolerance >= 1.0
            || self.tolerance >= 1.0
            || self.max_elements == 0
            || self.max_active_iterations == 0
            || self.max_linear_iterations == 0
            || self.linear_tolerance >= 1.0
        {
            return Err(err("invalid Pawley controls"));
        }
        Ok(())
    }
}
/// Last accepted state bound to the full scientific request and algorithm controls.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyCheckpoint {
    /// Exact original request, including data, constraints and topology.
    pub input: PawleyInput,
    /// Immutable numerical controls.
    pub options: PawleyOptions,
    /// Scaled free coordinates at acceptance.
    pub free: Vec<f64>,
    /// Initial and accepted chi-square history.
    pub chi_square_history: Vec<f64>,
    /// Next damping parameter.
    pub damping: f64,
    /// Whether the fixed-geometry initialization has completed.
    pub linear_initialized: bool,
    /// Continue one-sided local optimization after an upward support jump.
    pub support_local: bool,
}
/// Work counters and wall times; excluded from scientific checkpoint identity.
#[derive(Clone, Debug, Default)]
pub struct PawleyDiagnostics {
    /// Profile/Jacobian evaluations in this invocation.
    pub evaluations: usize,
    /// Active-face iterations in this invocation.
    pub linear_iterations: usize,
    /// Conjugate-gradient iterations in matrix-free face solves.
    pub krylov_iterations: usize,
    /// Time in native objective evaluation.
    pub evaluation_seconds: f64,
    /// Time in augmented QR reduction.
    pub qr_seconds: f64,
    /// Time solving active faces.
    pub face_seconds: f64,
    /// Time computing active constraint multipliers.
    pub multiplier_seconds: f64,
    /// Time in final rank and covariance diagnostics.
    pub diagnostic_seconds: f64,
    /// Rejected line-search trials, including infeasible models.
    pub backtrack_rejections: usize,
    /// Backtracks rejected by physical model or bound validation.
    pub infeasible_trials: usize,
    /// Most recent physical trial rejection, if any.
    pub last_rejected_error: Option<String>,
    /// Last proposed step norm in normalized coordinates.
    pub last_step_norm: f64,
    /// Last computed projected-gradient norm, if checked.
    pub projected_gradient_norm: Option<f64>,
    /// Criterion that established convergence, if any.
    pub convergence_criterion: Option<&'static str>,
}
/// Accepted solution and uncertainty diagnostics.
#[derive(Clone, Debug)]
pub struct PawleyResult {
    /// Work counters, timings and convergence explanation.
    pub diagnostics: PawleyDiagnostics,
    /// Values, derivatives and fit metrics at the accepted state.
    pub evaluation: PawleyEvaluation,
    /// Restart state; rejected trials never replace it.
    pub checkpoint: PawleyCheckpoint,
    /// Stable stop category; budget exhaustion is not convergence.
    pub termination_reason: TerminationReason,
    /// Rank of the normalized, undamped weighted free Jacobian.
    pub rank: Option<usize>,
    /// Number of supported free columns.
    pub observed_free_parameters: usize,
    /// Physical bound-active parameter indices with nonzero free derivative.
    pub active_bounds: Vec<usize>,
    /// Whether a refinable composed Gaussian or Lorentzian width is at its boundary.
    pub active_width_bounds: bool,
    /// Scaled-free covariance, available only for interior full-rank fits.
    pub covariance: Option<DMatrix<f64>>,
    /// Reason covariance was omitted, if applicable.
    pub covariance_limitation: Option<String>,
}
/// Refine with default finite runtime limits.
///
/// # Errors
/// Returns validation, numerical, or runtime infrastructure failures.
pub fn refine_pawley(
    input: &PawleyInput,
    options: &PawleyOptions,
) -> Result<PawleyResult, PawleyError> {
    let mut runtime = RefinementRuntime::new(RefinementLimits::default(), None).map_err(err)?;
    refine_pawley_with_runtime(input, options, None, &mut runtime)
}
/// Refine with cooperative cancellation and accepted-state checkpoint delivery.
///
/// # Errors
/// Rejects stale/corrupt restart records and invalid initial models.
#[allow(clippy::too_many_lines)]
pub fn refine_pawley_with_runtime(
    input: &PawleyInput,
    options: &PawleyOptions,
    restart: Option<&PawleyCheckpoint>,
    runtime: &mut RefinementRuntime<PawleyCheckpoint>,
) -> Result<PawleyResult, PawleyError> {
    input.validate()?;
    options.validate()?;
    let transform = ConstraintTransform::new(input.parameters.clone(), input.constraints.clone())
        .map_err(err)?;
    let mut diagnostics = PawleyDiagnostics::default();
    let chain = transform.derivative_matrix().map_err(err)?;
    let nonlinear_columns: Vec<usize> = (0..transform.free_keys().len())
        .filter(|&j| {
            input.parameters.specs().iter().enumerate().any(|(i, s)| {
                matches!(s.key().module(), "pawley_profile" | "pawley_lattice")
                    && chain.values[i * chain.columns + j] != 0.0
            })
        })
        .collect();
    let initial = transform.pack().map_err(err)?;
    let mut checkpoint = if let Some(cp) = restart {
        if cp.input != *input
            || cp.options != *options
            || !cp.damping.is_finite()
            || cp.damping <= 0.0
            || cp.chi_square_history.is_empty()
            || cp
                .chi_square_history
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || cp.chi_square_history.windows(2).any(|v| v[1] >= v[0])
        {
            return Err(err("stale or corrupt Pawley checkpoint"));
        }
        cp.clone()
    } else {
        PawleyCheckpoint {
            input: input.clone(),
            options: options.clone(),
            free: initial,
            chi_square_history: Vec::new(),
            damping: options.damping,
            linear_initialized: nonlinear_columns.is_empty(),
            support_local: false,
        }
    };
    // Initial/restart evaluation is necessary to return a usable accepted state even on pre-cancellation.
    let mut current = timed_evaluate(input, options, &checkpoint.free, &mut diagnostics)?;
    if !current.residuals.included.iter().any(|v| *v) {
        return Err(err("Pawley has no included observations"));
    }
    if checkpoint.free.len() == current.inactive_columns.len() && !checkpoint.free.is_empty() {
        return Err(err("Pawley has no observable free parameters"));
    }
    if let Some(last) = checkpoint.chi_square_history.last() {
        if last.to_bits() != current.residuals.chi_square.to_bits() {
            return Err(err("Pawley checkpoint objective mismatch"));
        }
    } else {
        checkpoint
            .chi_square_history
            .push(current.residuals.chi_square);
    }
    runtime
        .resume_accepted(checkpoint.chi_square_history.len() - 1)
        .map_err(err)?;
    runtime
        .emit(
            RefinementEventKind::Start,
            "pawley",
            "Pawley least squares started",
            Vec::new(),
        )
        .map_err(err)?;
    let stop = |e: RuntimeError| -> Result<TerminationReason, PawleyError> {
        match e {
            RuntimeError::Stopped(s) => Ok(s.reason),
            other => Err(err(other)),
        }
    };
    let mut reason = TerminationReason::Converged;
    let mut damping = checkpoint.damping;
    let mut attempt = checkpoint.chi_square_history.len();
    if let Err(e) = runtime.begin_evaluation() {
        reason = stop(e)?;
    } else {
        loop {
            if checkpoint.free.is_empty() {
                diagnostics.convergence_criterion = Some("no_free_parameters");
                break;
            }
            if let Err(e) = runtime.begin_iteration(attempt) {
                reason = stop(e)?;
                break;
            }
            attempt += 1;
            let initializing = !checkpoint.linear_initialized;
            let frozen_nonlinear = if initializing {
                nonlinear_columns.clone()
            } else {
                Vec::new()
            };
            let design =
                LinearDesign::new(input, &current, options.use_uncertainty, frozen_nonlinear)?;
            let norms = design.norms.clone();
            let r = DVector::from_iterator(
                current.residuals.weighted_residual.len(),
                current
                    .residuals
                    .weighted_residual
                    .iter()
                    .zip(&current.residuals.included)
                    .map(|(r, yes)| if *yes { *r } else { 0.0 }),
            );
            let guard = if !initializing && (checkpoint.support_local || damping > options.damping)
            {
                support_guard(
                    input,
                    &transform,
                    &checkpoint.free,
                    &current,
                    options.support_fwhm,
                    options.use_uncertainty,
                )?
            } else {
                None
            };
            let guard = guard.filter(|g| {
                g.near(
                    &norms,
                    options.tolerance * (1.0 + current.residuals.chi_square.sqrt()),
                )
            });
            if guard.is_some() && !checkpoint.support_local {
                damping = options.damping;
            }
            let (a, b) = inequalities(input, &transform, &checkpoint.free, &norms, &current)?;
            let (a, b) = if let Some(g) = &guard {
                g.constrain(&a, &b, &norms)
            } else {
                (a, b)
            };
            // Freeze unsupported and initialization-only coordinates inside the
            // constrained solve, rather than repairing a coupled feasible step afterwards.
            let mut frozen = current.inactive_columns.clone();
            if initializing {
                frozen.extend(&nonlinear_columns);
            }
            frozen.sort_unstable();
            frozen.dedup();
            let (a, b) = freeze_columns(a, b, &frozen);

            let columns = norms.len();
            let proposal = if let Some(j) = &design.dense {
                let rows = j.nrows();
                let qr_start = Instant::now();
                let mut augmented = DMatrix::zeros(rows + columns, columns);
                augmented.rows_mut(0, rows).copy_from(j);
                for c in 0..columns {
                    augmented[(rows + c, c)] = damping.sqrt();
                }
                let mut target = DVector::zeros(rows + columns);
                target.rows_mut(0, rows).copy_from(&(-&r));
                let qr = augmented.qr();
                qr.q_tr_mul(&mut target);
                let reduced = qr.r();
                let target = target.rows(0, columns).into_owned();
                diagnostics.qr_seconds += qr_start.elapsed().as_secs_f64();
                active_step(
                    &reduced,
                    &target,
                    &a,
                    &b,
                    options.max_active_iterations,
                    runtime,
                    &mut diagnostics,
                )
            } else {
                let model = ProductStep {
                    design: &design,
                    residual: &r,
                    damping,
                    options,
                    runtime,
                    iterations: std::cell::Cell::new(0),
                    preconditioner: preconditioner(&design, damping),
                };
                let proposal = active_step_model(
                    &model,
                    &a,
                    &b,
                    options.max_active_iterations,
                    runtime,
                    &mut diagnostics,
                );
                diagnostics.krylov_iterations += model.iterations.get();
                proposal
            };
            let mut delta = match proposal {
                Ok(d) => d,
                Err(StepError::Runtime(e)) => {
                    reason = stop(e)?;
                    break;
                }
                Err(StepError::Numerical) => {
                    reason = TerminationReason::NumericalFailure;
                    break;
                }
            };
            // No data can update these coordinates; preserve their accepted state exactly.
            for &column in &current.inactive_columns {
                delta[column] = 0.0;
            }
            if initializing {
                for &column in &nonlinear_columns {
                    delta[column] = 0.0;
                }
            }
            diagnostics.last_step_norm = delta.norm();
            // A nearly singular joint design can retain a finite Gauss–Newton step
            // after reaching first-order stationarity. Check the undamped projected
            // gradient, rather than declaring convergence from damping-induced shrinkage.
            if !initializing
                && delta.norm()
                    <= options.tolerance.sqrt() * (1.0 + current.residuals.chi_square.sqrt())
            {
                let gradient = design.transpose(&r)?;
                match active_step_model(
                    &IdentityStep { target: -gradient },
                    &a,
                    &b,
                    options.max_active_iterations,
                    runtime,
                    &mut diagnostics,
                ) {
                    Ok(projected)
                        if projected.norm()
                            <= options.tolerance * (1.0 + current.residuals.chi_square.sqrt()) =>
                    {
                        diagnostics.projected_gradient_norm = Some(projected.norm());
                        diagnostics.convergence_criterion = Some(if guard.is_some() {
                            "support_projected_gradient"
                        } else {
                            "projected_gradient"
                        });
                        reason = TerminationReason::Converged;
                        break;
                    }
                    Err(StepError::Runtime(e)) => {
                        reason = stop(e)?;
                        break;
                    }
                    Ok(projected) => {
                        diagnostics.projected_gradient_norm = Some(projected.norm());
                    }
                    Err(StepError::Numerical) => {}
                }
            }
            if delta.norm() <= options.tolerance * (1.0 + current.residuals.chi_square.sqrt()) {
                if initializing {
                    checkpoint.linear_initialized = true;
                    continue;
                }
                reason = if damping <= options.damping {
                    diagnostics.convergence_criterion = Some(if guard.is_some() {
                        "support_feasible_step"
                    } else {
                        "normalized_step"
                    });
                    TerminationReason::Converged
                } else {
                    TerminationReason::Stagnated
                };
                break;
            }
            let mut accepted = None;
            let mut stopped = None;
            // An interior roundoff margin prevents a feasible active-face solve from
            // crossing a strict width boundary by a last-bit rounding error.
            let mut fraction = 1.0 - 1e-8;
            for _ in 0..30 {
                if let Err(e) = runtime.begin_evaluation() {
                    stopped = Some(stop(e)?);
                    break;
                }
                let mut trial: Vec<f64> = checkpoint
                    .free
                    .iter()
                    .zip(delta.iter().zip(&norms))
                    .map(|(v, (d, n))| v + fraction * d / n)
                    .collect();
                let retracted = guard
                    .as_ref()
                    .map_or(Ok(()), |g| g.retract(input, &transform, &mut trial));
                match retracted
                    .and_then(|()| timed_evaluate(input, options, &trial, &mut diagnostics))
                {
                    Ok(evaluation)
                        if evaluation.residuals.chi_square < current.residuals.chi_square =>
                    {
                        accepted = Some((trial, evaluation));
                        break;
                    }
                    Err(e) => {
                        diagnostics.infeasible_trials += 1;
                        diagnostics.last_rejected_error = Some(e.to_string());
                    }
                    Ok(_) => {}
                }
                diagnostics.backtrack_rejections += 1;
                fraction *= 0.5;
            }
            // Locate a detected support jump with width-only membership checks.
            // Keep the original improving trial unless the final full evaluation
            // also improves the objective. No extra objective is synthesized.
            if accepted.is_some() && !initializing && fraction < 0.1 {
                let trial_at = |f: f64| -> Result<Vec<f64>, PawleyError> {
                    let mut v: Vec<f64> = checkpoint
                        .free
                        .iter()
                        .zip(delta.iter().zip(&norms))
                        .map(|(v, (d, n))| v + f * d / n)
                        .collect();
                    if let Some(g) = &guard {
                        g.retract(input, &transform, &mut v)?;
                    }
                    Ok(v)
                };
                let membership_at = |f| {
                    trial_at(f).and_then(|v| {
                        support_membership(
                            input,
                            &transform,
                            &v,
                            &current.positions,
                            options.support_fwhm,
                        )
                    })
                };
                let mut lower = fraction;
                let mut upper = 2.0 * fraction;
                if let (Ok(Some(base)), Ok(Some(end))) =
                    (membership_at(lower), membership_at(upper))
                {
                    if base != end {
                        for _ in 0..48 {
                            let middle = lower + (upper - lower) * 0.5;
                            if middle.to_bits() == lower.to_bits()
                                || middle.to_bits() == upper.to_bits()
                            {
                                break;
                            }
                            match membership_at(middle) {
                                Ok(Some(m)) if m == base => lower = middle,
                                _ => upper = middle,
                            }
                        }
                        if let Err(e) = runtime.begin_evaluation() {
                            stopped = Some(stop(e)?);
                        } else if let Ok(trial) = trial_at(lower) {
                            if let Ok(evaluation) =
                                timed_evaluate(input, options, &trial, &mut diagnostics)
                            {
                                if evaluation.residuals.chi_square < current.residuals.chi_square {
                                    fraction = lower;
                                    accepted = Some((trial, evaluation));
                                }
                            }
                        }
                    }
                }
            }
            if let Some(stop) = stopped {
                reason = stop;
                break;
            }
            if let Some((trial, evaluation)) = accepted {
                let scaled_step = DVector::from_iterator(
                    columns,
                    trial
                        .iter()
                        .zip(&checkpoint.free)
                        .zip(&norms)
                        .map(|((new, old), norm)| (new - old) * norm),
                );
                let full_step = scaled_step.norm() >= 0.1 * delta.norm();
                let change = design.apply(&scaled_step)?;
                let predicted_reduction = -2.0 * r.dot(&change) - change.norm_squared();
                let actual_reduction =
                    current.residuals.chi_square - evaluation.residuals.chi_square;
                let cost_tolerance = options.tolerance * current.residuals.chi_square;
                // Relative objective convergence requires agreement with the undamped
                // local model, so backtracking or large damping cannot fake success.
                let cost_converged = !initializing
                    && damping <= options.damping
                    && full_step
                    && predicted_reduction > 0.0
                    && predicted_reduction <= cost_tolerance
                    && actual_reduction <= cost_tolerance
                    && actual_reduction >= 0.1 * predicted_reduction;
                current = evaluation;
                checkpoint.linear_initialized = true;
                checkpoint.support_local = guard.is_some();
                checkpoint.free = trial;
                checkpoint
                    .chi_square_history
                    .push(current.residuals.chi_square);
                // Heavy backtracking means the full local model was unreliable,
                // including when moving hard-support edges crosses sample points.
                // Keep the next proposal local instead of repeatedly undoing the
                // line search by reducing damping after a tiny accepted fraction.
                damping = if fraction < 0.1 {
                    (damping * 10.0).min(1e12)
                } else {
                    (damping * 0.25).max(1e-12)
                };
                checkpoint.damping = damping;
                runtime.accept_step(Some(&checkpoint)).map_err(err)?;
                runtime
                    .emit(
                        RefinementEventKind::StepAccepted,
                        "pawley",
                        "accepted joint least-squares step",
                        Vec::new(),
                    )
                    .map_err(err)?;
                if cost_converged {
                    diagnostics.convergence_criterion = Some(if guard.is_some() {
                        "support_relative_objective"
                    } else {
                        "relative_objective"
                    });
                    reason = TerminationReason::Converged;
                    break;
                }
            } else {
                if let Err(e) = runtime.reject_step() {
                    reason = stop(e)?;
                    break;
                }
                damping *= 10.0;
                if !damping.is_finite() {
                    reason = TerminationReason::NumericalFailure;
                    break;
                }
            }
        }
    }
    runtime
        .emit(
            RefinementEventKind::Termination,
            "pawley",
            reason.as_str(),
            Vec::new(),
        )
        .map_err(err)?;
    let start = Instant::now();
    let mut result = result(input, options, checkpoint, current, reason)?;
    diagnostics.diagnostic_seconds = start.elapsed().as_secs_f64();
    result.diagnostics = diagnostics;
    Ok(result)
}

fn timed_evaluate(
    input: &PawleyInput,
    options: &PawleyOptions,
    free: &[f64],
    diagnostics: &mut PawleyDiagnostics,
) -> Result<PawleyEvaluation, PawleyError> {
    let start = Instant::now();
    let result = evaluate_pawley_with_storage(
        input,
        free,
        options.support_fwhm,
        options.use_uncertainty,
        options.max_elements,
        options.solver == PawleySolver::Dense,
    );
    diagnostics.evaluations += 1;
    diagnostics.evaluation_seconds += start.elapsed().as_secs_f64();
    result
}
#[allow(clippy::too_many_lines)] // Physical bounds and composed-width chains share one ordering.
fn inequalities(
    input: &PawleyInput,
    t: &ConstraintTransform,
    z: &[f64],
    norms: &[f64],
    evaluation: &PawleyEvaluation,
) -> Result<(DMatrix<f64>, DVector<f64>), PawleyError> {
    let values = t.unpack(z, false).map_err(err)?;
    let chain = t.derivative_matrix().map_err(err)?;
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    for (i, spec) in input.parameters.specs().iter().enumerate() {
        let row: Vec<f64> = (0..z.len())
            .map(|j| chain.values[i * z.len() + j] / norms[j])
            .collect();
        let scale = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if scale == 0.0 {
            continue;
        }
        for (bound, sign) in [(spec.bounds().lower(), 1.0), (spec.bounds().upper(), -1.0)] {
            if bound.is_finite() {
                rows.extend(row.iter().map(|v| sign * v / scale));
                rhs.push(sign * (bound - values[spec.key()]) / scale);
            }
        }
    }
    // Positive composed widths are physical inequalities, not independent signs on U/V/W/X/Y.
    let profile_indices: Vec<usize> = crate::PAWLEY_PROFILE_NAMES
        .iter()
        .map(|name| {
            input
                .parameters
                .index_of(&crate::pawley_key("profile", "instrument", name)?)
                .ok_or_else(|| err("missing profile parameter"))
        })
        .collect::<Result<_, PawleyError>>()?;
    let profile: Vec<f64> = profile_indices
        .iter()
        .map(|i| values[input.parameters.specs()[*i].key()])
        .collect();
    let mut reflection = 0;
    for phase in &input.phases {
        let geometry = if let Some(domain) = &phase.lattice {
            let par = domain.parameterization();
            let v: Vec<f64> = par
                .parameter_names()
                .iter()
                .map(|name| Ok(values[&crate::pawley_key("lattice", &phase.id, name)?]))
                .collect::<Result<_, PawleyError>>()?;
            Some(
                crate::cw_lattice_geometry(
                    par,
                    par.to_cell(&v).map_err(err)?,
                    &phase.hkl,
                    input.instrument.wavelength_angstrom,
                )
                .map_err(err)?,
            )
        } else {
            None
        };
        for r in 0..phase.reflection_ids.len() {
            let theta = evaluation.positions[reflection].to_radians() * 0.5;
            let tangent = theta.tan();
            let secant = theta.cos().recip();
            let half_degree = std::f64::consts::PI / 360.0;
            let variance = profile[0] * tangent * tangent + profile[1] * tangent + profile[2];
            let lorentz = profile[3] * secant + profile[4] * tangent;
            let bases = [
                [tangent * tangent, tangent, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, secant, tangent],
            ];
            let position_chains = [
                (2.0 * profile[0] * tangent + profile[1]) * secant * secant * half_degree,
                (profile[3] * secant * tangent + profile[4] * secant * secant) * half_degree,
            ];
            for term in 0..2 {
                let mut physical = vec![0.0; input.parameters.specs().len()];
                for (j, &index) in profile_indices.iter().enumerate() {
                    physical[index] = bases[term][j];
                }
                if let Some(g) = &geometry {
                    for (j, name) in g.parameter_names.iter().enumerate() {
                        let index = input
                            .parameters
                            .index_of(&crate::pawley_key("lattice", &phase.id, name)?)
                            .ok_or_else(|| err("missing cell parameter"))?;
                        physical[index] = position_chains[term]
                            * g.d_two_theta_d_parameters[r * g.parameter_names.len() + j];
                    }
                }
                let terms: Vec<(usize, f64)> = physical
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, v)| *v != 0.0)
                    .collect();
                let row: Vec<f64> = (0..z.len())
                    .map(|j| {
                        terms
                            .iter()
                            .map(|(i, v)| v * chain.values[i * z.len() + j])
                            .sum::<f64>()
                            / norms[j]
                    })
                    .collect();
                let scale = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if scale > 0.0 {
                    rows.extend(row.iter().map(|v| v / scale));
                    rhs.push(-[variance, lorentz][term] / scale);
                }
            }
            reflection += 1;
        }
    }
    Ok((
        DMatrix::from_row_slice(rhs.len(), z.len(), &rows),
        DVector::from_vec(rhs),
    ))
}
fn freeze_columns(
    a: DMatrix<f64>,
    b: DVector<f64>,
    frozen: &[usize],
) -> (DMatrix<f64>, DVector<f64>) {
    if frozen.is_empty() {
        return (a, b);
    }
    let mut extended = DMatrix::zeros(a.nrows() + 2 * frozen.len(), a.ncols());
    extended.rows_mut(0, a.nrows()).copy_from(&a);
    let mut rhs = DVector::zeros(extended.nrows());
    rhs.rows_mut(0, b.len()).copy_from(&b);
    for (i, &column) in frozen.iter().enumerate() {
        extended[(a.nrows() + 2 * i, column)] = 1.0;
        extended[(a.nrows() + 2 * i + 1, column)] = -1.0;
    }
    (extended, rhs)
}
struct LinearDesign<'a> {
    evaluation: &'a PawleyEvaluation,
    dense: Option<DMatrix<f64>>,
    weights: Vec<f64>,
    norms: Vec<f64>,
    frozen: Vec<usize>,
}
impl<'a> LinearDesign<'a> {
    fn new(
        input: &PawleyInput,
        evaluation: &'a PawleyEvaluation,
        uncertainty: bool,
        frozen: Vec<usize>,
    ) -> Result<Self, PawleyError> {
        let weights: Vec<f64> = evaluation
            .residuals
            .included
            .iter()
            .enumerate()
            .map(|(i, yes)| {
                if !yes {
                    0.0
                } else if uncertainty {
                    input
                        .pattern
                        .uncertainty
                        .as_ref()
                        .map_or(1.0, |s| 1.0 / s[i])
                } else {
                    1.0
                }
            })
            .collect();
        let mut dense = evaluation
            .jacobian
            .as_ref()
            .map(|_| weighted(input, evaluation, uncertainty).0);
        let mut norms = if let Some(j) = &mut dense {
            for &c in &frozen {
                j.column_mut(c).fill(0.0);
            }
            (0..j.ncols())
                .map(|c| j.column(c).norm())
                .collect::<Vec<_>>()
        } else {
            evaluation.jacobian_operator.column_norms(&weights)?
        };
        for &c in &frozen {
            norms[c] = 0.0;
        }
        for v in &mut norms {
            if *v == 0.0 {
                *v = 1.0;
            }
        }
        if let Some(j) = &mut dense {
            for (c, n) in norms.iter().enumerate() {
                j.column_mut(c).scale_mut(1.0 / n);
            }
        }
        Ok(Self {
            evaluation,
            dense,
            weights,
            norms,
            frozen,
        })
    }
    fn apply(&self, v: &DVector<f64>) -> Result<DVector<f64>, PawleyError> {
        if let Some(j) = &self.dense {
            return Ok(j * v);
        }
        let mut scaled: Vec<f64> = v.iter().zip(&self.norms).map(|(v, n)| v / n).collect();
        for &c in &self.frozen {
            scaled[c] = 0.0;
        }
        let values = self.evaluation.jacobian_operator.jvp(&scaled)?;
        Ok(DVector::from_iterator(
            values.len(),
            values.iter().zip(&self.weights).map(|(v, w)| v * w),
        ))
    }
    fn transpose(&self, v: &DVector<f64>) -> Result<DVector<f64>, PawleyError> {
        if let Some(j) = &self.dense {
            return Ok(j.transpose() * v);
        }
        let scaled: Vec<f64> = v.iter().zip(&self.weights).map(|(v, w)| v * w).collect();
        let mut values = self.evaluation.jacobian_operator.vjp(&scaled)?;
        for (v, n) in values.iter_mut().zip(&self.norms) {
            *v /= n;
        }
        for &c in &self.frozen {
            values[c] = 0.0;
        }
        Ok(DVector::from_vec(values))
    }
}
type BlockFactor = (usize, nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>);
fn preconditioner(design: &LinearDesign<'_>, damping: f64) -> Vec<BlockFactor> {
    design
        .evaluation
        .jacobian_operator
        .gram_blocks(&design.weights, &design.norms, &design.frozen, 32)
        .into_iter()
        .filter_map(|(start, mut block)| {
            // Stabilization changes only the preconditioner, never the objective
            // or the operator used to check the actual linear residual.
            for i in 0..block.nrows() {
                block[(i, i)] += damping.max(1e-2);
            }
            block.cholesky().map(|factor| (start, factor))
        })
        .collect()
}
struct ProductStep<'a, 'b> {
    preconditioner: Vec<BlockFactor>,
    iterations: std::cell::Cell<usize>,
    design: &'a LinearDesign<'b>,
    residual: &'a DVector<f64>,
    damping: f64,
    options: &'a PawleyOptions,
    runtime: &'a RefinementRuntime<PawleyCheckpoint>,
}
impl ProductStep<'_, '_> {
    fn precondition(&self, residual: &DVector<f64>, tangent: &Tangent) -> DVector<f64> {
        let mut result = residual.clone();
        for (start, factor) in &self.preconditioner {
            let size = factor.l_dirty().nrows();
            let block = factor.solve(&residual.rows(*start, size).into_owned());
            result.rows_mut(*start, size).copy_from(&block);
        }
        tangent.project(&result)
    }
    fn hessian(&self, v: &DVector<f64>) -> Result<DVector<f64>, StepError> {
        let product = self.design.apply(v).map_err(|_| StepError::Numerical)?;
        Ok(self
            .design
            .transpose(&product)
            .map_err(|_| StepError::Numerical)?
            + v * self.damping)
    }
}
impl StepModel for ProductStep<'_, '_> {
    fn dimensions(&self) -> usize {
        self.design.norms.len()
    }
    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>, StepError> {
        Ok(self.hessian(point)?
            + self
                .design
                .transpose(self.residual)
                .map_err(|_| StepError::Numerical)?)
    }
    fn candidate(
        &self,
        constraint: &DMatrix<f64>,
        bound: &DVector<f64>,
    ) -> Result<DVector<f64>, StepError> {
        let (particular, tangent) = tangent_space(constraint, bound)?;
        let mut solution = particular;
        let mut residual = -tangent.project(&self.gradient(&solution)?);
        let threshold = self.options.linear_tolerance * (1.0 + residual.norm());
        let mut direction = self.precondition(&residual, &tangent);
        let mut squared = residual.dot(&direction);
        for _ in 0..self.options.max_linear_iterations {
            self.runtime.check_boundary().map_err(StepError::Runtime)?;
            if residual.norm() <= threshold {
                return Ok(solution);
            }
            self.iterations.set(self.iterations.get() + 1);
            let product = tangent.project(&self.hessian(&direction)?);
            let denominator = direction.dot(&product);
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(StepError::Numerical);
            }
            let alpha = squared / denominator;
            solution += &direction * alpha;
            residual -= product * alpha;
            let preconditioned = self.precondition(&residual, &tangent);
            let next = residual.dot(&preconditioned);
            if residual.norm() <= threshold {
                // Recursive CG residuals drift on ill-conditioned overlaps.
                // Convergence is based on the recomputed undelayed product.
                residual = -tangent.project(&self.gradient(&solution)?);
                if residual.norm() <= threshold {
                    return Ok(solution);
                }
                direction = self.precondition(&residual, &tangent);
                squared = residual.dot(&direction);
            } else {
                direction = preconditioned + direction * (next / squared);
                squared = next;
            }
        }
        Err(StepError::Numerical)
    }
}
enum Tangent {
    All,
    Coordinates(Vec<usize>),
    RowSpace(DMatrix<f64>),
}
impl Tangent {
    fn project(&self, v: &DVector<f64>) -> DVector<f64> {
        match self {
            Self::All => v.clone(),
            Self::Coordinates(free) => {
                let mut result = DVector::zeros(v.len());
                for &i in free {
                    result[i] = v[i];
                }
                result
            }
            Self::RowSpace(rows) => {
                let mut result = v.clone();
                // Reorthogonalize vector products directly; never materialize
                // a nearly singular full projector or a dense null basis.
                for _ in 0..2 {
                    result -= rows.transpose() * (rows * &result);
                }
                result
            }
        }
    }
}
fn tangent_space(c: &DMatrix<f64>, b: &DVector<f64>) -> Result<(DVector<f64>, Tangent), StepError> {
    let n = c.ncols();
    if c.nrows() == 0 {
        return Ok((DVector::zeros(n), Tangent::All));
    }
    let mut fixed = vec![None; n];
    let mut boxes = true;
    for row in 0..c.nrows() {
        let columns = (0..n).filter(|&j| c[(row, j)] != 0.0).collect::<Vec<_>>();
        if let [j] = columns.as_slice() {
            fixed[*j] = Some(b[row] / c[(row, *j)]);
        } else {
            boxes = false;
            break;
        }
    }
    if boxes {
        return Ok((
            DVector::from_iterator(n, fixed.iter().map(|v| v.unwrap_or(0.0))),
            Tangent::Coordinates((0..n).filter(|&j| fixed[j].is_none()).collect()),
        ));
    }
    let svd = c.clone().svd(true, true);
    let threshold = svd.singular_values.amax() * 1e-13;
    let rank = svd
        .singular_values
        .iter()
        .filter(|v| **v > threshold)
        .count();
    let vt = svd.v_t.as_ref().ok_or(StepError::Numerical)?.clone();
    let particular = svd.solve(b, threshold).map_err(|_| StepError::Numerical)?;
    if rank == n {
        return Ok((particular, Tangent::Coordinates(Vec::new())));
    }
    Ok((particular, Tangent::RowSpace(vt.rows(0, rank).into_owned())))
}

enum StepError {
    Runtime(RuntimeError),
    Numerical,
}
// For independent box faces, solve only the remaining columns. Avoiding a
// square projector full of exact null columns materially reduces large NNLS work.
fn face_candidate(
    design: &DMatrix<f64>,
    target: &DVector<f64>,
    constraint: &DMatrix<f64>,
    bound: &DVector<f64>,
) -> Result<DVector<f64>, StepError> {
    let n = design.ncols();
    if constraint.nrows() == 0 {
        // Callers supply an upper-triangular augmented QR factor or identity.
        return design
            .solve_upper_triangular(target)
            .ok_or(StepError::Numerical);
    }
    let mut fixed = vec![None; n];
    let mut boxes_only = true;
    for row in 0..constraint.nrows() {
        let nonzero: Vec<usize> = (0..n).filter(|&j| constraint[(row, j)] != 0.0).collect();
        if let [column] = nonzero.as_slice() {
            fixed[*column] = Some(bound[row] / constraint[(row, *column)]);
        } else {
            boxes_only = false;
            break;
        }
    }
    if boxes_only {
        let mut candidate = DVector::from_iterator(n, fixed.iter().map(|v| v.unwrap_or(0.0)));
        let free: Vec<usize> = (0..n).filter(|&j| fixed[j].is_none()).collect();
        if free.is_empty() {
            return Ok(candidate);
        }
        let mut reduced = DMatrix::zeros(design.nrows(), free.len());
        for (j, &column) in free.iter().enumerate() {
            reduced.column_mut(j).copy_from(&design.column(column));
        }
        // This design includes positive damping (or is identity for the
        // projected-gradient check), so each column subset has full rank.
        let qr = reduced.qr();
        let mut remaining = target - design * &candidate;
        qr.q_tr_mul(&mut remaining);
        let tangent = qr
            .r()
            .solve_upper_triangular(&remaining.rows(0, free.len()).into_owned())
            .ok_or(StepError::Numerical)?;
        for (j, &column) in free.iter().enumerate() {
            candidate[column] = tangent[j];
        }
        return Ok(candidate);
    }
    let svd = constraint.clone().svd(true, true);
    let threshold = svd.singular_values.amax() * 1e-13;
    let rank = svd
        .singular_values
        .iter()
        .filter(|v| **v > threshold)
        .count();
    let vt = svd.v_t.as_ref().ok_or(StepError::Numerical)?.clone();
    let inverse = svd
        .pseudo_inverse(threshold)
        .map_err(|_| StepError::Numerical)?;
    let particular = &inverse * bound;
    // Use an explicit orthonormal null basis. A dense I-C^+C projector
    // contains roundoff-sized spurious columns that a second SVD can mistake
    // for physical directions when several width faces are dependent.
    let mut basis = Vec::<DVector<f64>>::new();
    for column in 0..n {
        let mut v = DVector::zeros(n);
        v[column] = 1.0;
        for _ in 0..2 {
            for row in 0..rank {
                let normal = vt.row(row).transpose();
                v -= &normal * normal.dot(&v);
            }
            for previous in &basis {
                v -= previous * previous.dot(&v);
            }
        }
        let norm = v.norm();
        if norm > 1e-10 {
            basis.push(v / norm);
        }
        if basis.len() == n - rank {
            break;
        }
    }
    if basis.len() != n - rank {
        return Err(StepError::Numerical);
    }
    if basis.is_empty() {
        return Ok(particular);
    }
    let null = DMatrix::from_columns(&basis);
    let qr = (design * &null).qr();
    let mut remaining = target - design * &particular;
    qr.q_tr_mul(&mut remaining);
    let tangent = qr
        .r()
        .solve_upper_triangular(&remaining.rows(0, basis.len()).into_owned())
        .ok_or(StepError::Numerical)?;
    Ok(particular + null * tangent)
}
// Feasible active-set least squares in a QR-reduced design. Constraints use A d >= b.
// Each face is solved by an SVD in its exact constraint null space.
#[allow(clippy::too_many_lines)] // Feasible block entry and general active-face iteration.
fn active_step(
    design: &DMatrix<f64>,
    target: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    limit: usize,
    runtime: &RefinementRuntime<PawleyCheckpoint>,
    diagnostics: &mut PawleyDiagnostics,
) -> Result<DVector<f64>, StepError> {
    active_step_model(
        &DenseStep { design, target },
        a,
        b,
        limit,
        runtime,
        diagnostics,
    )
}
trait StepModel {
    fn dimensions(&self) -> usize;
    fn candidate(
        &self,
        constraint: &DMatrix<f64>,
        bound: &DVector<f64>,
    ) -> Result<DVector<f64>, StepError>;
    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>, StepError>;
}
// Euclidean projection needs no dense identity factorization.
struct IdentityStep {
    target: DVector<f64>,
}
impl StepModel for IdentityStep {
    fn dimensions(&self) -> usize {
        self.target.len()
    }
    fn candidate(
        &self,
        constraint: &DMatrix<f64>,
        bound: &DVector<f64>,
    ) -> Result<DVector<f64>, StepError> {
        let (particular, tangent) = tangent_space(constraint, bound)?;
        Ok(&particular + tangent.project(&(&self.target - &particular)))
    }
    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>, StepError> {
        Ok(point - &self.target)
    }
}
struct DenseStep<'a> {
    design: &'a DMatrix<f64>,
    target: &'a DVector<f64>,
}
impl StepModel for DenseStep<'_> {
    fn dimensions(&self) -> usize {
        self.target.len()
    }
    fn candidate(
        &self,
        constraint: &DMatrix<f64>,
        bound: &DVector<f64>,
    ) -> Result<DVector<f64>, StepError> {
        face_candidate(self.design, self.target, constraint, bound)
    }
    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>, StepError> {
        Ok(self.design.transpose() * (self.design * point - self.target))
    }
}
#[allow(clippy::too_many_lines)]
fn active_step_model(
    model: &impl StepModel,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    limit: usize,
    runtime: &RefinementRuntime<PawleyCheckpoint>,
    diagnostics: &mut PawleyDiagnostics,
) -> Result<DVector<f64>, StepError> {
    let n = model.dimensions();
    let mut d = DVector::zeros(n);
    let mut active = Vec::<usize>::new();
    for _ in 0..limit {
        runtime.check_boundary().map_err(StepError::Runtime)?;
        let mut constraint = DMatrix::zeros(active.len(), n);
        let mut bound = DVector::zeros(active.len());
        for (q, &row) in active.iter().enumerate() {
            constraint.row_mut(q).copy_from(&a.row(row));
            bound[q] = b[row];
        }
        diagnostics.linear_iterations += 1;
        let face_start = Instant::now();
        let candidate = model.candidate(&constraint, &bound)?;
        diagnostics.face_seconds += face_start.elapsed().as_secs_f64();
        if candidate.iter().any(|v| !v.is_finite()) {
            return Err(StepError::Numerical);
        }
        // Enter independent box faces together when their projection is feasible
        // for every coupled constraint. This avoids one full SVD per negative
        // area in large fixed-geometry initializations; general faces retain the
        // feasible single-blocker algorithm below.
        if active.is_empty() {
            let mut projected = candidate.clone();
            let mut box_faces = Vec::new();
            for row in 0..a.nrows() {
                if a.row(row).transpose().dot(&candidate) >= b[row] {
                    continue;
                }
                let nonzero: Vec<usize> = (0..n).filter(|&j| a[(row, j)] != 0.0).collect();
                if let [column] = nonzero.as_slice() {
                    projected[*column] = b[row] / a[(row, *column)];
                    box_faces.push(row);
                }
            }
            // A homogeneous coupled face (for example a support-radius
            // boundary) can be entered together with independent area bounds.
            // The global feasibility check below remains authoritative.
            let coupled: Vec<usize> = (0..a.nrows())
                .filter(|&row| {
                    b[row] == 0.0
                        && a.row(row).transpose().dot(&projected) < 0.0
                        && !box_faces.contains(&row)
                })
                .collect();
            if box_faces.len() > 1 && !coupled.is_empty() {
                let faces: Vec<usize> = box_faces.iter().chain(&coupled).copied().collect();
                let mut c = DMatrix::zeros(faces.len(), n);
                let mut rhs = DVector::zeros(faces.len());
                for (i, &row) in faces.iter().enumerate() {
                    c.row_mut(i).copy_from(&a.row(row));
                    rhs[i] = b[row];
                }
                if let Ok(p) = (IdentityStep {
                    target: candidate.clone(),
                })
                .candidate(&c, &rhs)
                {
                    projected = p;
                    box_faces = faces;
                }
            }
            if box_faces.len() > 1
                && (0..a.nrows()).all(|row| {
                    a.row(row).transpose().dot(&projected) >= b[row] - 1e-13 * (1.0 + b[row].abs())
                })
            {
                d = projected;
                active = box_faces;
                continue;
            }
        }
        let multiplier_start = Instant::now();
        let multipliers = if active.is_empty() {
            DVector::zeros(0)
        } else {
            let gradient = model.gradient(&candidate)?;
            constraint
                .transpose()
                .svd(true, true)
                .solve(&gradient, 1e-13)
                .map_err(|_| StepError::Numerical)?
        };
        diagnostics.multiplier_seconds += multiplier_start.elapsed().as_secs_f64();
        let direction = &candidate - &d;
        let mut alpha = 1.0;
        let mut blocker = None;
        for row in 0..a.nrows() {
            if active.contains(&row) {
                continue;
            }
            let candidate_slack = a.row(row).transpose().dot(&candidate) - b[row];
            let feasibility_tolerance = 1e-13 * (1.0 + b[row].abs() + candidate.norm());
            // A numerically satisfied face must not be re-added as a dependent blocker.
            if candidate_slack >= -feasibility_tolerance {
                continue;
            }
            let motion = a.row(row).transpose().dot(&direction);
            if motion < -1e-14 {
                let fraction = ((a.row(row).transpose().dot(&d) - b[row]) / (-motion)).max(0.0);
                if fraction < alpha {
                    alpha = fraction;
                    blocker = Some(row);
                }
            }
        }
        d += direction * alpha;
        if let Some(row) = blocker {
            active.push(row);
            continue;
        }
        let remove = active
            .iter()
            .enumerate()
            .filter(|(q, _)| multipliers[*q] < -1e-10)
            .min_by(|(q, _), (r, _)| multipliers[*q].total_cmp(&multipliers[*r]))
            .map(|(q, _)| q);
        if let Some(q) = remove {
            active.remove(q);
        } else {
            if (0..a.nrows()).any(|row| {
                a.row(row).transpose().dot(&d) < b[row] - 1e-13 * (1.0 + b[row].abs() + d.norm())
            }) {
                return Err(StepError::Numerical);
            }
            // Roundoff-sized violations on inactive independent box rows must
            // not make a physically infeasible trial. Correct them within the
            // QP, then recheck every coupled inequality. This is not clipping
            // fitted areas after optimization.
            for row in 0..a.nrows() {
                let slack = a.row(row).transpose().dot(&d) - b[row];
                if slack < 0.0 {
                    let nonzero: Vec<usize> = (0..n).filter(|&j| a[(row, j)] != 0.0).collect();
                    if let [column] = nonzero.as_slice() {
                        d[*column] = b[row] / a[(row, *column)];
                    }
                }
            }
            if (0..a.nrows()).any(|row| {
                a.row(row).transpose().dot(&d) < b[row] - 1e-13 * (1.0 + b[row].abs() + d.norm())
            }) {
                return Err(StepError::Numerical);
            }
            return Ok(d);
        }
    }
    Err(StepError::Numerical)
}
#[allow(clippy::too_many_lines)] // Diagnostics are assembled from one undamped factorization.
fn result(
    input: &PawleyInput,
    options: &PawleyOptions,
    checkpoint: PawleyCheckpoint,
    evaluation: PawleyEvaluation,
    reason: TerminationReason,
) -> Result<PawleyResult, PawleyError> {
    let design = LinearDesign::new(input, &evaluation, options.use_uncertainty, Vec::new())?;
    let n = design.norms.len();
    let norms = design.norms.clone();
    let dense = design.dense.clone();
    let t = ConstraintTransform::new(input.parameters.clone(), input.constraints.clone())
        .map_err(err)?;
    let values = t.unpack(&checkpoint.free, false).map_err(err)?;
    let chain = t.derivative_matrix().map_err(err)?;
    let active_bounds: Vec<usize> = input
        .parameters
        .specs()
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            let v = values[s.key()];
            let tol = 1e-9 * (1.0 + v.abs());
            chain.row(*i).is_some_and(|r| r.iter().any(|v| *v != 0.0))
                && ((v - s.bounds().lower()).abs() <= tol || (v - s.bounds().upper()).abs() <= tol)
        })
        .map(|(i, _)| i)
        .collect();
    let physical_bound_rows: usize = input
        .parameters
        .specs()
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            chain
                .row(*i)
                .is_some_and(|row| row.iter().any(|v| *v != 0.0))
        })
        .map(|(_, spec)| {
            usize::from(spec.bounds().lower().is_finite())
                + usize::from(spec.bounds().upper().is_finite())
        })
        .sum();
    let (_, bound_distances) = inequalities(input, &t, &checkpoint.free, &norms, &evaluation)?;
    let active_width_bounds = bound_distances
        .iter()
        .skip(physical_bound_rows)
        .any(|v| v.abs() <= 1e-9);
    if n == 0 {
        return Ok(PawleyResult {
            diagnostics: PawleyDiagnostics::default(),
            evaluation,
            checkpoint,
            termination_reason: reason,
            rank: Some(0),
            observed_free_parameters: 0,
            active_bounds,
            active_width_bounds,
            covariance: None,
            covariance_limitation: Some("no free parameters".into()),
        });
    }
    let Some(j) = dense else {
        let observed_free_parameters = n - evaluation.inactive_columns.len();
        return Ok(PawleyResult {
            diagnostics: PawleyDiagnostics::default(),
            evaluation,
            checkpoint,
            termination_reason: reason,
            rank: None,
            observed_free_parameters,
            active_bounds,
            active_width_bounds,
            covariance: None,
            covariance_limitation: Some(
                "matrix-free solve: exact rank and full covariance not computed".into(),
            ),
        });
    };
    let svd = j.svd(false, true);
    let threshold = options.rank_tolerance * svd.singular_values.amax();
    let rank = svd
        .singular_values
        .iter()
        .filter(|v| **v > threshold)
        .count();
    let observed_free_parameters = n - evaluation.inactive_columns.len();
    let support_boundary = support_guard(
        input,
        &t,
        &checkpoint.free,
        &evaluation,
        options.support_fwhm,
        options.use_uncertainty,
    )?
    .is_some();
    let limitation = if support_boundary {
        Some("hard-support boundary: Gaussian covariance omitted")
    } else if rank < n {
        Some("rank deficient or unobserved free columns")
    } else if !active_bounds.is_empty() || active_width_bounds {
        Some("bound-active solution: Gaussian covariance omitted")
    } else if reason != TerminationReason::Converged {
        Some("refinement did not converge")
    } else if n == 0 {
        Some("no free parameters")
    } else if !evaluation.residuals.reduced_chi_square.is_finite() {
        Some("nonpositive residual degrees of freedom")
    } else {
        None
    };
    let covariance = if limitation.is_none() {
        let vt = svd.v_t.ok_or_else(|| err("missing singular vectors"))?;
        let mut c = DMatrix::zeros(n, n);
        let variance = if options.use_uncertainty && input.pattern.uncertainty.is_some() {
            1.0
        } else {
            evaluation.residuals.reduced_chi_square
        };
        for i in 0..n {
            for j in 0..n {
                c[(i, j)] = (0..n)
                    .map(|k| vt[(k, i)] * vt[(k, j)] / svd.singular_values[k].powi(2))
                    .sum::<f64>()
                    * variance
                    / (norms[i] * norms[j]);
            }
        }
        Some(c)
    } else {
        None
    };
    Ok(PawleyResult {
        diagnostics: PawleyDiagnostics::default(),
        evaluation,
        checkpoint,
        termination_reason: reason,
        rank: Some(rank),
        observed_free_parameters,
        active_bounds,
        active_width_bounds,
        covariance,
        covariance_limitation: limitation.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_projection_matches_dense_for_dependent_coupled_faces() {
        let target = DVector::from_vec(vec![3.0, -2.0, 1.0, 4.0]);
        let c = DMatrix::from_row_slice(
            3,
            4,
            &[1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        );
        let b = DVector::from_vec(vec![1.0, 2.0, 0.5]);
        let expected = face_candidate(&DMatrix::identity(4, 4), &target, &c, &b)
            .ok()
            .unwrap();
        let actual = IdentityStep { target }.candidate(&c, &b).ok().unwrap();
        assert!((&actual - expected).norm() < 1e-12);
        assert!((c * actual - b).norm() < 1e-12);
    }

    #[test]
    fn coupled_faces_do_not_create_spurious_null_directions() {
        let design = DMatrix::identity(3, 3);
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let constraint = DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 1.0, 1.0, 1.0, 1.000_001, 1.0, 1.0, 1.0, 1.000_001],
        );
        let solution = face_candidate(&design, &target, &constraint, &DVector::zeros(3))
            .ok()
            .unwrap();
        assert!(solution.norm() < 1e-12);
        let constraint = constraint.rows(0, 2).into_owned();
        let solution = face_candidate(&design, &target, &constraint, &DVector::zeros(2))
            .ok()
            .unwrap();
        assert!((solution - DVector::from_vec(vec![-1.0, 0.0, 1.0])).norm() < 1e-8);
    }

    #[test]
    fn roundoff_sized_inactive_box_violation_returns_a_physically_feasible_step() {
        // Exact constrained minimizer of (x + 1e-14)^2 + (y - 1)^2, x >= 0.
        let runtime = RefinementRuntime::new(RefinementLimits::default(), None).unwrap();
        let result = active_step(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1e-14, 1.0]),
            &DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            &DVector::zeros(1),
            20,
            &runtime,
            &mut PawleyDiagnostics::default(),
        )
        .ok()
        .unwrap();
        assert!(result[0] >= 0.0 && result[0] <= f64::EPSILON);
        assert!((result[1] - 1.0).abs() < 1e-15);
    }
}
