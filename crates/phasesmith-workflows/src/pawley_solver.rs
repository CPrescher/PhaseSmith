//! Deterministic, bounded, dense Pawley Gauss–Newton solver.
// Matrix equations use conventional short names.
#![allow(clippy::many_single_char_names)]
use crate::pawley::{err, weighted};
use crate::{
    ConstraintTransform, PawleyError, PawleyEvaluation, PawleyInput, RefinementEventKind,
    RefinementLimits, RefinementRuntime, RuntimeError, TerminationReason, evaluate_pawley,
};
use nalgebra::{DMatrix, DVector};

/// Scientific and allocation controls. Runtime budgets are separate.
#[derive(Clone, Debug, PartialEq)]
pub struct PawleyOptions {
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
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            || self.rank_tolerance >= 1.0
            || self.tolerance >= 1.0
            || self.max_elements == 0
            || self.max_active_iterations == 0
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
}
/// Accepted solution and uncertainty diagnostics.
#[derive(Clone, Debug)]
pub struct PawleyResult {
    /// Values, derivatives and fit metrics at the accepted state.
    pub evaluation: PawleyEvaluation,
    /// Restart state; rejected trials never replace it.
    pub checkpoint: PawleyCheckpoint,
    /// Stable stop category; budget exhaustion is not convergence.
    pub termination_reason: TerminationReason,
    /// Rank of the normalized, undamped weighted free Jacobian.
    pub rank: usize,
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
    let evaluate = |z: &[f64]| {
        evaluate_pawley(
            input,
            z,
            options.support_fwhm,
            options.use_uncertainty,
            options.max_elements,
        )
    };
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
        }
    };
    // Initial/restart evaluation is necessary to return a usable accepted state even on pre-cancellation.
    let mut current = evaluate(&checkpoint.free)?;
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
                break;
            }
            if let Err(e) = runtime.begin_iteration(attempt) {
                reason = stop(e)?;
                break;
            }
            attempt += 1;
            let (mut j, r) = weighted(input, &current, options.use_uncertainty);
            let initializing = !checkpoint.linear_initialized;
            if initializing {
                for &column in &nonlinear_columns {
                    j.column_mut(column).fill(0.0);
                }
            }

            let norms: Vec<f64> = (0..j.ncols())
                .map(|c| {
                    let norm = j.column(c).norm();
                    if norm == 0.0 { 1.0 } else { norm }
                })
                .collect();
            for (c, norm) in norms.iter().enumerate() {
                j.column_mut(c).scale_mut(1.0 / norm);
            }
            let (a, b) = inequalities(input, &transform, &checkpoint.free, &norms, &current)?;
            // Freeze unsupported and initialization-only coordinates inside the
            // constrained solve, rather than repairing a coupled feasible step afterwards.
            let mut frozen = current.inactive_columns.clone();
            if initializing {
                frozen.extend(&nonlinear_columns);
            }
            frozen.sort_unstable();
            frozen.dedup();
            let (a, b) = freeze_columns(a, b, &frozen);

            // Orthogonal reduction avoids squaring the profile design's condition number.
            let rows = j.nrows();
            let columns = j.ncols();
            let mut augmented = DMatrix::zeros(rows + columns, columns);
            augmented.rows_mut(0, rows).copy_from(&j);
            for c in 0..columns {
                augmented[(rows + c, c)] = damping.sqrt();
            }
            let mut target = DVector::zeros(rows + columns);
            target.rows_mut(0, rows).copy_from(&(-&r));
            let qr = augmented.qr();
            qr.q_tr_mul(&mut target);
            let reduced = qr.r();
            let target = target.rows(0, columns).into_owned();
            let mut delta = match active_step(
                &reduced,
                &target,
                &a,
                &b,
                options.max_active_iterations,
                runtime,
            ) {
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
            // A nearly singular joint design can retain a finite Gauss–Newton step
            // after reaching first-order stationarity. Check the undamped projected
            // gradient, rather than declaring convergence from damping-induced shrinkage.
            if !initializing
                && delta.norm()
                    <= options.tolerance.sqrt() * (1.0 + current.residuals.chi_square.sqrt())
            {
                let gradient = j.transpose() * &r;
                match active_step(
                    &DMatrix::identity(columns, columns),
                    &(-gradient),
                    &a,
                    &b,
                    options.max_active_iterations,
                    runtime,
                ) {
                    Ok(projected)
                        if projected.norm()
                            <= options.tolerance * (1.0 + current.residuals.chi_square.sqrt()) =>
                    {
                        reason = TerminationReason::Converged;
                        break;
                    }
                    Err(StepError::Runtime(e)) => {
                        reason = stop(e)?;
                        break;
                    }
                    _ => {}
                }
            }
            if delta.norm() <= options.tolerance * (1.0 + current.residuals.chi_square.sqrt()) {
                if initializing {
                    checkpoint.linear_initialized = true;
                    continue;
                }
                reason = if damping <= options.damping {
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
                let trial: Vec<f64> = checkpoint
                    .free
                    .iter()
                    .zip(delta.iter().zip(&norms))
                    .map(|(v, (d, n))| v + fraction * d / n)
                    .collect();
                if let Ok(evaluation) = evaluate(&trial) {
                    if evaluation.residuals.chi_square < current.residuals.chi_square {
                        accepted = Some((trial, evaluation));
                        break;
                    }
                }
                fraction *= 0.5;
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
                let change = &j * scaled_step;
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
                checkpoint.free = trial;
                checkpoint
                    .chi_square_history
                    .push(current.residuals.chi_square);
                damping = (damping * 0.25).max(1e-12);
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
    result(input, options, checkpoint, current, reason)
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
                let row: Vec<f64> = (0..z.len())
                    .map(|j| {
                        physical
                            .iter()
                            .enumerate()
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
    let inverse = svd
        .pseudo_inverse(threshold)
        .map_err(|_| StepError::Numerical)?;
    let particular = &inverse * bound;
    let projector = DMatrix::identity(n, n) - &inverse * constraint;
    let svd = (design * &projector).svd(true, true);
    let threshold = svd.singular_values.amax() * 1e-13;
    let tangent = svd
        .solve(&(target - design * &particular), threshold)
        .map_err(|_| StepError::Numerical)?;
    Ok(particular + projector * tangent)
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
) -> Result<DVector<f64>, StepError> {
    let n = target.len();
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
        let candidate = face_candidate(design, target, &constraint, &bound)?;
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
        let multipliers = if active.is_empty() {
            DVector::zeros(0)
        } else {
            let gradient = design.transpose() * (design * &candidate - target);
            constraint
                .transpose()
                .svd(true, true)
                .solve(&gradient, 1e-13)
                .map_err(|_| StepError::Numerical)?
        };
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
    let (mut j, _) = weighted(input, &evaluation, options.use_uncertainty);
    let n = j.ncols();
    let norms: Vec<f64> = (0..n)
        .map(|c| {
            let norm = j.column(c).norm();
            if norm == 0.0 { 1.0 } else { norm }
        })
        .collect();
    for (c, norm) in norms.iter().enumerate() {
        j.column_mut(c).scale_mut(1.0 / norm);
    }
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
            evaluation,
            checkpoint,
            termination_reason: reason,
            rank: 0,
            observed_free_parameters: 0,
            active_bounds,
            active_width_bounds,
            covariance: None,
            covariance_limitation: Some("no free parameters".into()),
        });
    }
    let svd = j.svd(false, true);
    let threshold = options.rank_tolerance * svd.singular_values.amax();
    let rank = svd
        .singular_values
        .iter()
        .filter(|v| **v > threshold)
        .count();
    let observed_free_parameters = n - evaluation.inactive_columns.len();
    let limitation = if rank < n {
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
        evaluation,
        checkpoint,
        termination_reason: reason,
        rank,
        observed_free_parameters,
        active_bounds,
        active_width_bounds,
        covariance,
        covariance_limitation: limitation.map(str::to_owned),
    })
}
