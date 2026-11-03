//! Deterministic bounded damped normal steps for native profile/scaling fits.
use crate::spots::{SpotError, geometry_linear::solve, invalid};

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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn product(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    matrix
        .chunks_exact(vector.len())
        .map(|row| dot(row, vector))
        .collect()
}
// Explicit small dense representation of ten limited-memory BFGS pairs. The
// initial scalar theta and rank-two secant updates define the quadratic model.
fn bfgs_matrix(n: usize, pairs: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    let theta = pairs.last().map_or(1., |(s, y)| dot(y, y) / dot(s, y));
    let mut b = vec![0.; n * n];
    for i in 0..n {
        b[i * n + i] = theta;
    }
    for (s, y) in pairs {
        let bs = product(&b, s);
        let sbs = dot(s, &bs);
        let sy = dot(s, y);
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] += y[i] * y[j] / sy - bs[i] * bs[j] / sbs;
            }
        }
    }
    b
}
// Minimize the secant quadratic along the piecewise projected-gradient path,
// then on the free-variable subspace. This is the generalized Cauchy point,
// rather than clipping an unconstrained inverse-Hessian direction.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn bounded_direction(
    p: &[f64],
    g: &[f64],
    b: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<Vec<f64>, SpotError> {
    let n = p.len();
    let mut cp = p.to_vec();
    let mut d: Vec<_> = g.iter().map(|g| -g).collect();
    let mut breaks = Vec::new();
    for i in 0..n {
        if lower[i] >= upper[i] {
            d[i] = 0.;
        } else if g[i] > 0. {
            breaks.push(((p[i] - lower[i]) / g[i], i, lower[i]));
        } else if g[i] < 0. {
            breaks.push(((p[i] - upper[i]) / g[i], i, upper[i]));
        }
    }
    breaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut time = 0.;
    let mut k = 0;
    loop {
        let delta: Vec<_> = cp.iter().zip(p).map(|(a, b)| a - b).collect();
        let bd = product(b, &d);
        let slope = dot(g, &d) + dot(&delta, &bd);
        let curve = dot(&d, &bd);
        if slope >= 0. || curve <= 0. {
            break;
        }
        let step = -slope / curve;
        let next = breaks.get(k).map_or(f64::INFINITY, |v| v.0);
        if step <= (next - time).max(0.) {
            for i in 0..n {
                cp[i] = (cp[i] + step * d[i]).clamp(lower[i], upper[i]);
            }
            break;
        }
        let interval = (next - time).max(0.);
        for i in 0..n {
            cp[i] = (cp[i] + interval * d[i]).clamp(lower[i], upper[i]);
        }
        time = next;
        while k < breaks.len() && breaks[k].0 <= time {
            let (_, i, bound) = breaks[k];
            cp[i] = bound;
            d[i] = 0.;
            k += 1;
        }
        if k == breaks.len() {
            break;
        }
    }
    let cauchy = cp.clone();
    let delta: Vec<_> = cp.iter().zip(p).map(|(a, b)| a - b).collect();
    let bd = product(b, &delta);
    let free: Vec<_> = (0..n)
        .filter(|&i| cp[i] > lower[i] && cp[i] < upper[i])
        .collect();
    if !free.is_empty() {
        let matrix: Vec<_> = free
            .iter()
            .flat_map(|&i| free.iter().map(move |&j| b[i * n + j]))
            .collect();
        let rhs: Vec<_> = free.iter().map(|&i| -g[i] - bd[i]).collect();
        let q = solve(&matrix, &rhs)?;
        let mut alpha = 1_f64;
        for (&i, &q) in free.iter().zip(&q) {
            if q > 0. {
                alpha = alpha.min((upper[i] - cp[i]) / q);
            } else if q < 0. {
                alpha = alpha.min((lower[i] - cp[i]) / q);
            }
        }
        for (&i, &q) in free.iter().zip(&q) {
            cp[i] = (cp[i] + alpha * q).clamp(lower[i], upper[i]);
        }
    }
    let direction: Vec<_> = cp.iter().zip(p).map(|(a, b)| a - b).collect();
    if dot(&direction, g) < 0. {
        Ok(direction)
    } else {
        Ok(cauchy.iter().zip(p).map(|(a, b)| a - b).collect())
    }
}

// Safeguarded More–Thuente search of the feasible ray. The external BSD helper
// provides only cubic/quadratic interval interpolation (dcstep), no model,
// optimizer state, Python, BLAS, native-code dependency or fitting policy.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn wolfe_search(
    p: &[f64],
    initial: &Model,
    d: &[f64],
    lower: &[f64],
    upper: &[f64],
    first: bool,
    evaluations: &mut usize,
    budget: usize,
    evaluate: &mut impl FnMut(&[f64]) -> Result<Model, SpotError>,
) -> Option<(Vec<f64>, Model)> {
    let slope = dot(&initial.gradient, d);
    if slope >= 0. {
        return None;
    }
    let mut maximum = if first { 1_f64 } else { 1e10_f64 };
    for i in 0..p.len() {
        if d[i] > 0. {
            maximum = maximum.min((upper[i] - p[i]) / d[i]);
        } else if d[i] < 0. {
            maximum = maximum.min((lower[i] - p[i]) / d[i]);
        }
    }
    if maximum <= 0. {
        return None;
    }
    let mut stp = 1_f64.min(maximum);
    let gtest = 1e-3 * slope;
    let (mut stx, mut fx, mut gx) = (0., initial.cost, slope);
    let (mut sty, mut fy, mut gy) = (0., initial.cost, slope);
    let mut brackt = false;
    let mut stage1 = true;
    let (mut stmin, mut stmax) = (0., stp + 4. * stp);
    let mut width = maximum;
    let mut width1 = 2. * width;
    for _ in 0..30 {
        if *evaluations >= budget {
            return None;
        }
        let trial: Vec<_> = (0..p.len())
            .map(|i| (p[i] + stp * d[i]).clamp(lower[i], upper[i]))
            .collect();
        *evaluations += 1;
        let model = evaluate(&trial).ok()?;
        if check(&model, p.len()).is_err() {
            return None;
        }
        let f = model.cost;
        let g = dot(&model.gradient, d);
        let ftest = initial.cost + stp * gtest;
        if stage1 && f <= ftest && g >= 0. {
            stage1 = false;
        }
        let warning = (brackt && (stp <= stmin || stp >= stmax || stmax - stmin <= 0.1 * stmax))
            || (stp >= maximum && f <= ftest && g <= gtest);
        if (f <= ftest && g.abs() <= 0.9 * (-slope)) || warning {
            return (f <= initial.cost).then_some((trial, model));
        }
        if stage1 && f <= fx && f > ftest {
            let (fm, mut fxm, mut fym) = (f - stp * gtest, fx - stx * gtest, fy - sty * gtest);
            let (gm, mut gxm, mut gym) = (g - gtest, gx - gtest, gy - gtest);
            crate::spots::profile_line_search::dcstep(
                &mut stx,
                &mut fxm,
                &mut gxm,
                &mut sty,
                &mut fym,
                &mut gym,
                &mut stp,
                fm,
                gm,
                &mut brackt,
                stmin,
                stmax,
            );
            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        } else {
            crate::spots::profile_line_search::dcstep(
                &mut stx,
                &mut fx,
                &mut gx,
                &mut sty,
                &mut fy,
                &mut gy,
                &mut stp,
                f,
                g,
                &mut brackt,
                stmin,
                stmax,
            );
        }
        if brackt {
            if (sty - stx).abs() >= 0.66 * width1 {
                stp = stx + 0.5 * (sty - stx);
            }
            width1 = width;
            width = (sty - stx).abs();
            stmin = stx.min(sty);
            stmax = stx.max(sty);
        } else {
            stmin = stp + 1.1 * (stp - stx);
            stmax = stp + 4. * (stp - stx);
        }
        stp = stp.clamp(0., maximum);
        if brackt && (stp <= stmin || stp >= stmax || stmax - stmin <= 0.1 * stmax) {
            stp = stx;
        }
        if !stp.is_finite() || stp <= 0. {
            return None;
        }
    }
    None
}

/// Bounded limited-memory BFGS for the small finite-support profile problem.
/// Ten secant pairs define the generalized Cauchy/free-subspace model; the
/// feasible ray uses a safeguarded More–Thuente search, capped at 30 evaluations.
/// # Errors
/// Invalid controls/bounds, nonfinite initial model, or invalid objective arrays.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
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
    let mut pairs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    while out.history.len() - 1 < options.iterations && out.evaluations < options.evaluations {
        let p = &out.parameters;
        let g = &out.model.gradient;
        let projected = (0..n)
            .map(|i| {
                if g[i] < 0. {
                    g[i].max(p[i] - upper[i]).abs()
                } else {
                    g[i].min(p[i] - lower[i]).abs()
                }
            })
            .fold(0., f64::max);
        if projected <= options.gtol {
            out.converged = true;
            break;
        }
        let b = bfgs_matrix(n, &pairs);
        let d = bounded_direction(p, g, &b, lower, upper)?;
        let Some((trial, model)) = wolfe_search(
            p,
            &out.model,
            &d,
            lower,
            upper,
            out.history.len() == 1,
            &mut out.evaluations,
            options.evaluations,
            &mut evaluate,
        ) else {
            if pairs.is_empty() {
                break;
            }
            // A failed secant-model search restarts the quadratic at the same
            // retained state, as in bounded limited-memory BFGS; never retain a
            // rejected trial or invent convergence.
            pairs.clear();
            continue;
        };
        let s: Vec<_> = trial.iter().zip(p).map(|(a, b)| a - b).collect();
        let y: Vec<_> = model.gradient.iter().zip(g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > f64::EPSILON * (-dot(g, &s)) {
            if pairs.len() == 10 {
                pairs.remove(0);
            }
            pairs.push((s, y));
        }
        let change = out.model.cost - model.cost;
        let scale = options
            .cost_floor
            .max(out.model.cost.abs())
            .max(model.cost.abs());
        out.parameters = trial;
        out.model = model;
        out.history.push(out.model.cost);
        if change <= options.ftol * scale {
            out.converged = true;
            break;
        }
    }
    Ok(out)
}
