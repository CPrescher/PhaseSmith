//! Deterministic bounded damped normal steps for native profile/scaling fits.
use crate::{SpotError, geometry_linear::solve, invalid};

/// Fused objective, first derivative and positive Gauss--Newton approximation.
pub struct Model {
    /// Scalar objective (normal convention: half squared residual for scaling).
    pub cost: f64,
    /// Gradient in original parameter units.
    pub gradient: Vec<f64>,
    /// Symmetric row-major normal matrix, same parameter order.
    pub normal: Vec<f64>,
}
/// Explicit effort and stopping conventions, independent of model selection.
#[derive(Clone, Copy)]
pub struct Options {
    /// Maximum accepted steps.
    pub iterations: usize,
    /// Maximum objective evaluations, including initial/rejected trials.
    pub evaluations: usize,
    /// Relative accepted-cost stopping tolerance.
    pub ftol: f64,
    /// Original-unit projected-gradient infinity-norm tolerance.
    pub gtol: f64,
    /// Scaled step tolerance; a tiny damped step alone is not convergence.
    pub xtol: f64,
    /// Absolute floor in the relative-cost test (one for profile SSE, zero for LS).
    pub cost_floor: f64,
}
/// Result of the actual bounded iterations; no automatic retry or tolerance relaxation.
pub struct Fit {
    /// Final feasible parameter vector.
    pub parameters: Vec<f64>,
    /// Fused model at the retained parameters.
    pub model: Model,
    /// A projected-gradient or accepted-cost criterion was reached.
    pub converged: bool,
    /// Actual objective evaluations.
    pub evaluations: usize,
    /// Accepted monotonically decreasing costs, including the start.
    pub history: Vec<f64>,
}
fn check(model: &Model, n: usize) -> Result<(), SpotError> {
    if !model.cost.is_finite()
        || model.gradient.len() != n
        || model.normal.len() != n * n
        || model
            .gradient
            .iter()
            .chain(&model.normal)
            .any(|v| !v.is_finite())
    {
        return Err(invalid("invalid fused bounded objective"));
    }
    Ok(())
}
/// Deterministic row-order least-squares accumulation, skipping exact zero entries.
/// # Errors
/// Rejects malformed/nonfinite arrays or the declared parameter/entry budget.
pub fn least_squares(residual: &[f64], jacobian: &[f64], n: usize) -> Result<Model, SpotError> {
    if n == 0
        || n > 256
        || jacobian.len() != residual.len().saturating_mul(n)
        || jacobian.len() > 20_000_000
        || residual.iter().chain(jacobian).any(|v| !v.is_finite())
    {
        return Err(invalid("invalid native least-squares arrays"));
    }
    let mut m = Model {
        cost: 0.,
        gradient: vec![0.; n],
        normal: vec![0.; n * n],
    };
    let mut active = Vec::with_capacity(n);
    for (row, &r) in residual.iter().enumerate() {
        m.cost += 0.5 * r * r;
        active.clear();
        for (j, &v) in jacobian[row * n..(row + 1) * n].iter().enumerate() {
            if v != 0. {
                m.gradient[j] += v * r;
                active.push((j, v));
            }
        }
        for (a, &(i, x)) in active.iter().enumerate() {
            for &(j, y) in &active[..=a] {
                m.normal[i * n + j] += x * y;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            m.normal[j * n + i] = m.normal[i * n + j];
        }
    }
    check(&m, n)?;
    Ok(m)
}
/// Minimize a native value/gradient/normal callback under independent closed bounds.
/// Column-norm scales are updated monotonically; active outward-gradient variables
/// are fixed for each step. Only feasible strictly improving states are retained.
/// # Errors
/// Rejects invalid dimensions, bounds/controls or an invalid initial model.
#[allow(clippy::too_many_lines)]
pub fn minimize(
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: Options,
    mut evaluate: impl FnMut(&[f64]) -> Result<Model, SpotError>,
) -> Result<Fit, SpotError> {
    let n = start.len();
    if n == 0
        || n > 256
        || lower.len() != n
        || upper.len() != n
        || start
            .iter()
            .chain(lower)
            .chain(upper)
            .any(|v| !v.is_finite())
        || (0..n).any(|i| lower[i] > upper[i] || start[i] < lower[i] || start[i] > upper[i])
        || options.evaluations == 0
        || options.evaluations > 10_000
        || options.iterations == 0
        || options.iterations > 1000
        || [options.ftol, options.gtol, options.xtol]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        || !options.cost_floor.is_finite()
        || options.cost_floor < 0.
    {
        return Err(invalid("invalid native optimizer bounds/controls"));
    }
    let mut p = start.to_vec();
    let mut model = evaluate(&p)?;
    check(&model, n)?;
    let mut evaluations = 1;
    let mut accepted = 0;
    let mut history = vec![model.cost];
    let mut damping = 1e-3;
    let mut norms = vec![0_f64; n];
    let mut converged = false;
    loop {
        let free: Vec<_> = (0..n)
            .filter(|&i| {
                lower[i] < upper[i]
                    && !((p[i] <= lower[i] && model.gradient[i] > 0.)
                        || (p[i] >= upper[i] && model.gradient[i] < 0.))
            })
            .collect();
        let gradient = free
            .iter()
            .map(|&i| model.gradient[i].abs())
            .fold(0., f64::max);
        if gradient <= options.gtol {
            converged = true;
            break;
        }
        if evaluations >= options.evaluations || accepted >= options.iterations {
            break;
        }
        for (i, norm) in norms.iter_mut().enumerate() {
            *norm = norm.max(model.normal[i * n + i].max(0.).sqrt());
        }
        let scales: Vec<_> = norms
            .iter()
            .map(|v| if *v > 0. { 1. / v } else { 1. })
            .collect();
        let nf = free.len();
        let mut matrix = vec![0.; nf * nf];
        let mut rhs = vec![0.; nf];
        for (a, &i) in free.iter().enumerate() {
            rhs[a] = -model.gradient[i] * scales[i];
            for (b, &j) in free.iter().enumerate() {
                matrix[a * nf + b] = model.normal[i * n + j] * scales[i] * scales[j];
            }
            matrix[a * nf + a] += damping * matrix[a * nf + a].abs().max(1.);
        }
        let Ok(delta) = solve(&matrix, &rhs) else {
            damping *= 10.;
            if damping > 1e24 {
                break;
            }
            continue;
        };
        let mut trial = p.clone();
        for (a, &i) in free.iter().enumerate() {
            trial[i] = (p[i] + scales[i] * delta[a]).clamp(lower[i], upper[i]);
        }
        let step: Vec<_> = trial.iter().zip(&p).map(|(a, b)| a - b).collect();
        let norm = step
            .iter()
            .zip(&scales)
            .map(|(v, s)| (v / s).powi(2))
            .sum::<f64>()
            .sqrt();
        let pnorm = p
            .iter()
            .zip(&scales)
            .map(|(v, s)| (v / s).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm <= options.xtol * (options.xtol + pnorm) {
            break;
        }
        evaluations += 1;
        let Ok(next) = evaluate(&trial) else {
            damping *= 10.;
            if damping > 1e24 {
                break;
            }
            continue;
        };
        if check(&next, n).is_err() {
            damping *= 10.;
            if damping > 1e24 {
                break;
            }
            continue;
        }
        let improvement = model.cost - next.cost;
        let mut prediction = -model
            .gradient
            .iter()
            .zip(&step)
            .map(|(g, d)| g * d)
            .sum::<f64>();
        for i in 0..n {
            for j in 0..n {
                prediction -= 0.5 * step[i] * model.normal[i * n + j] * step[j];
            }
        }
        let ratio = if prediction > 0. {
            improvement / prediction
        } else {
            0.
        };
        if improvement > 0. {
            let cost_converged = improvement
                <= options.ftol
                    * options
                        .cost_floor
                        .max(model.cost.abs())
                        .max(next.cost.abs())
                && ratio > 0.25;
            p = trial;
            model = next;
            accepted += 1;
            history.push(model.cost);
            damping = if ratio > 0.75 {
                (damping / 3.).max(1e-15)
            } else if ratio < 0.25 {
                damping * 4.
            } else {
                damping
            };
            if cost_converged {
                converged = true;
                break;
            }
        } else {
            damping *= 10.;
        }
        if damping > 1e24 {
            break;
        }
    }
    Ok(Fit {
        parameters: p,
        model,
        converged,
        evaluations,
        history,
    })
}

/// Bounded limited-memory quasi-Newton steps for finite-support shape objectives.
/// Uses ten positive-curvature pairs and a feasible Armijo backtracking line
/// search (at most thirty trials), using the fused analytical shape gradient.
/// # Errors
/// Invalid controls/bounds, nonfinite initial model, or invalid objective arrays.
#[allow(clippy::too_many_lines)]
pub fn quasi_newton(
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: Options,
    mut evaluate: impl FnMut(&[f64]) -> Result<Model, SpotError>,
) -> Result<Fit, SpotError> {
    if options.evaluations == 0 || options.evaluations > 10_000 {
        return Err(invalid("invalid quasi-Newton evaluation budget"));
    }
    // Reuse the complete input and initial-objective validation, without taking a step.
    let mut out = minimize(
        start,
        lower,
        upper,
        Options {
            evaluations: 1,
            ..options
        },
        &mut evaluate,
    )?;
    let n = start.len();
    let mut pairs: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    for _ in 0..options.iterations {
        let p = &out.parameters;
        let g = &out.model.gradient;
        let projected: Vec<_> = (0..n)
            .map(|i| {
                if lower[i] >= upper[i]
                    || (p[i] <= lower[i] && g[i] > 0.)
                    || (p[i] >= upper[i] && g[i] < 0.)
                {
                    0.
                } else {
                    g[i]
                }
            })
            .collect();
        if projected.iter().map(|v| v.abs()).fold(0., f64::max) <= options.gtol {
            out.converged = true;
            break;
        }
        let mut direction: Vec<_> = projected.iter().map(|v| -v).collect();
        let mut alpha = vec![0.; pairs.len()];
        for (i, (s, y, rho)) in pairs.iter().enumerate().rev() {
            alpha[i] = rho * dot(s, &direction);
            for j in 0..n {
                direction[j] -= alpha[i] * y[j];
            }
        }
        if let Some((s, y, _)) = pairs.last() {
            let gamma = dot(s, y) / dot(y, y);
            for v in &mut direction {
                *v *= gamma;
            }
        }
        for (i, (s, y, rho)) in pairs.iter().enumerate() {
            let beta = rho * dot(y, &direction);
            for j in 0..n {
                direction[j] += s[j] * (alpha[i] - beta);
            }
        }
        for j in 0..n {
            direction[j] = (p[j] + direction[j]).clamp(lower[j], upper[j]) - p[j];
        }
        if dot(&direction, g) >= 0. {
            pairs.clear();
            for j in 0..n {
                direction[j] = (p[j] - projected[j]).clamp(lower[j], upper[j]) - p[j];
            }
        }
        let slope = dot(&direction, g);
        let mut step = 1.;
        let mut accepted = None;
        for _ in 0..30 {
            if out.evaluations >= options.evaluations {
                break;
            }
            let trial: Vec<_> = (0..n)
                .map(|j| (p[j] + step * direction[j]).clamp(lower[j], upper[j]))
                .collect();
            out.evaluations += 1;
            if let Ok(model) = evaluate(&trial) {
                if check(&model, n).is_ok()
                    && model.cost < out.model.cost
                    && model.cost <= out.model.cost + 1e-4 * step * slope
                {
                    accepted = Some((trial, model));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((trial, model)) = accepted else {
            break;
        };
        let s: Vec<_> = trial.iter().zip(p).map(|(a, b)| a - b).collect();
        let y: Vec<_> = model.gradient.iter().zip(g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            if pairs.len() == 10 {
                pairs.remove(0);
            }
            pairs.push((s, y, 1. / sy));
        }
        let change = out.model.cost - model.cost;
        let cost_scale = options
            .cost_floor
            .max(out.model.cost.abs())
            .max(model.cost.abs());
        out.parameters = trial;
        out.model = model;
        out.history.push(out.model.cost);
        if change <= options.ftol * cost_scale {
            out.converged = true;
            break;
        }
    }
    Ok(out)
}
