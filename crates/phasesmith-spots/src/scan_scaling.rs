//! Structure-independent relative scan scaling from repeated symmetry families.
//!
//! Published scaling decomposition: Beilsten-Edmands et al. (2020),
//! doi:10.1107/S2059798320003198. This deliberately small model uses
//! `log(I_i) = intercept_family + beta * normalized_omega_i`.
//! Family intercepts are eliminated by weighted centering; no atomic model or
//! external intensity table is accepted. Huber IRLS operates only on strong
//! training observations. The signed target intensities are never fitted in log space.

use crate::{SpotError, invalid};

/// Explicit strong-reference and robust-fit controls.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Reference intensity must exceed this many counting standard deviations.
    pub minimum_snr: f64,
    /// Lower relative variance contribution, squared in the log variance.
    pub relative_error_floor: f64,
    /// Huber threshold in conditional log-standard-deviation units.
    pub huber: f64,
    /// Maximum number of IRLS steps.
    pub max_iterations: usize,
    /// Absolute coefficient-change stopping tolerance.
    pub tolerance: f64,
}

/// Fitted log slope and auditable conditional reference information.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Log-intensity slope over one normalized half scan.
    pub coefficient: f64,
    /// Conditional variance, holding reference selection/robust weights fixed.
    pub variance: f64,
    /// Original input indices of strong repeated-family training references.
    pub reference_ids: Vec<usize>,
    /// Final robust inverse-log-variance weights, in reference order.
    pub reference_weights: Vec<f64>,
    /// Number of fit steps used.
    pub iterations: usize,
    /// Whether the declared coefficient-change criterion was satisfied.
    pub converged: bool,
}

fn moments(x: &[f64], y: &[f64], group: &[usize], w: &[f64], n: usize) -> Vec<[f64; 3]> {
    let mut sums = vec![[0.0; 3]; n];
    for (((&a, &b), &g), &weight) in x.iter().zip(y).zip(group).zip(w) {
        sums[g][0] += weight;
        sums[g][1] += weight * a;
        sums[g][2] += weight * b;
    }
    for v in &mut sums {
        if v[0] > 0.0 {
            v[1] /= v[0];
            v[2] /= v[0];
        }
    }
    sums
}

fn validate(
    intensity: &[f64],
    sigma: &[f64],
    x: &[f64],
    groups: &[usize],
    training: &[bool],
    options: Options,
) -> Result<(), SpotError> {
    let n = x.len();
    if n == 0
        || n > 1_000_000
        || intensity.len() != n
        || sigma.len() != n
        || groups.len() != n
        || training.len() != n
        || groups.iter().any(|&g| g >= n)
        || x.iter().any(|v| !v.is_finite() || v.abs() > 1.0 + 1e-12)
        || !options.minimum_snr.is_finite()
        || options.minimum_snr < 3.0
        || !options.relative_error_floor.is_finite()
        || !(0.0..=1.0).contains(&options.relative_error_floor)
        || !options.huber.is_finite()
        || options.huber <= 0.0
        || !(1..=1_000).contains(&options.max_iterations)
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
    {
        return Err(invalid("invalid scan-scaling arrays or options"));
    }
    Ok(())
}

fn references(
    intensity: &[f64],
    sigma: &[f64],
    x: &[f64],
    groups: &[usize],
    training: &[bool],
    minimum_snr: f64,
) -> Result<(Vec<usize>, usize), SpotError> {
    let n = x.len();
    let ng = groups.iter().copied().max().unwrap_or(0) + 1;
    let mut counts = vec![0_usize; ng];
    let mut low = vec![f64::INFINITY; ng];
    let mut high = vec![f64::NEG_INFINITY; ng];
    let mut eligible = vec![false; n];
    for i in 0..n {
        if training[i] {
            if !intensity[i].is_finite() || !sigma[i].is_finite() || sigma[i] <= 0.0 {
                return Err(invalid(
                    "training intensities and positive sigmas must be finite",
                ));
            }
            if intensity[i] / sigma[i] > minimum_snr {
                let g = groups[i];
                eligible[i] = true;
                counts[g] += 1;
                low[g] = low[g].min(x[i]);
                high[g] = high[g].max(x[i]);
            }
        }
    }
    let reference_ids: Vec<usize> = (0..n)
        .filter(|&i| {
            eligible[i] && counts[groups[i]] >= 2 && high[groups[i]] - low[groups[i]] > 1e-10
        })
        .collect();
    if reference_ids.len() < 3 {
        return Err(invalid(
            "insufficient repeated strong training references for scan scaling",
        ));
    }
    let families = counts
        .iter()
        .enumerate()
        .filter(|&(i, &count)| count >= 2 && high[i] - low[i] > 1e-10)
        .count();
    Ok((reference_ids, families))
}

/// Fit the single smooth scan-scale term without using external intensities.
///
/// `x` lies on the closed interval `[-1,1]`; group IDs must be nonnegative,
/// less than input length and label already-confirmed equivalent reflections
/// within the same physical crystal. True `training` observations must have
/// finite intensity and positive sigma. False rows may contain missing counts.
/// Groups need at least two strong training measurements with angular leverage.
///
/// # Errors
/// Returns errors for invalid arrays/options or no identifiable scan slope.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    intensity: &[f64],
    sigma: &[f64],
    x: &[f64],
    groups: &[usize],
    training: &[bool],
    options: Options,
) -> Result<Fit, SpotError> {
    validate(intensity, sigma, x, groups, training, options)?;
    let ng = groups.iter().copied().max().unwrap_or(0) + 1;
    let (reference_ids, families) =
        references(intensity, sigma, x, groups, training, options.minimum_snr)?;
    let xx: Vec<f64> = reference_ids.iter().map(|&i| x[i]).collect();
    let yy: Vec<f64> = reference_ids.iter().map(|&i| intensity[i].ln()).collect();
    let gg: Vec<usize> = reference_ids.iter().map(|&i| groups[i]).collect();
    let variance: Vec<f64> = reference_ids
        .iter()
        .map(|&i| (sigma[i] / intensity[i]).powi(2) + options.relative_error_floor.powi(2))
        .collect();
    let base: Vec<f64> = variance.iter().map(|v| 1.0 / v).collect();
    if base.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(invalid("overflow in scan reference weights"));
    }
    let mut weights = base.clone();
    let mut beta = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    let mut denominator = 0.0;
    let mut residual = vec![0.0; reference_ids.len()];
    for iteration in 0..options.max_iterations {
        let means = moments(&xx, &yy, &gg, &weights, ng);
        denominator = 0.0;
        let mut numerator = 0.0;
        let mut total_weight = 0.0;
        for i in 0..xx.len() {
            let dx = xx[i] - means[gg[i]][1];
            let dy = yy[i] - means[gg[i]][2];
            denominator += weights[i] * dx * dx;
            numerator += weights[i] * dx * dy;
            total_weight += weights[i];
        }
        if !denominator.is_finite() || denominator <= 1e-14 * total_weight {
            return Err(invalid(
                "scan slope is not identifiable from equivalent observations",
            ));
        }
        let next = numerator / denominator;
        if !next.is_finite() || next.abs() > 100.0 {
            return Err(invalid("nonfinite or unsupported scan-scaling coefficient"));
        }
        for i in 0..xx.len() {
            residual[i] = yy[i] - means[gg[i]][2] - next * (xx[i] - means[gg[i]][1]);
        }
        iterations = iteration + 1;
        if (next - beta).abs() < options.tolerance {
            beta = next;
            converged = true;
            break;
        }
        beta = next;
        if iteration + 1 < options.max_iterations {
            for i in 0..weights.len() {
                let z = residual[i].abs() / variance[i].sqrt();
                weights[i] = base[i] * (options.huber / z.max(1e-30)).min(1.0);
            }
        }
    }
    let dof = (reference_ids.len().saturating_sub(families + 1)).max(1) as f64;
    let cost: f64 = residual
        .iter()
        .zip(&weights)
        .map(|(&r, &w)| w * r * r)
        .sum();
    let conditional_variance = (cost / dof).max(1.0) / denominator;
    if !conditional_variance.is_finite() {
        return Err(invalid("overflow in scan-scaling covariance"));
    }
    Ok(Fit {
        coefficient: beta,
        variance: conditional_variance,
        reference_ids,
        reference_weights: weights,
        iterations,
        converged,
    })
}

/// Apply the fitted factor to signed intensities, evaluating partials in one pass.
///
/// Returns `[factor,I_corrected,sigma_conditional,dI/dbeta,dI/domega,sigma_model]`.
/// `sigma_model` is separate: model errors correlate different measurements.
/// No extrapolation beyond the declared closed angular support is allowed.
///
/// # Errors
/// Returns errors for invalid input or arithmetic overflow.
pub fn apply(
    intensity: &[f64],
    sigma: &[f64],
    omega: &[f64],
    range: [f64; 2],
    beta: f64,
    variance: f64,
) -> Result<Vec<[f64; 6]>, SpotError> {
    let width = (range[1] - range[0]) / 2.0;
    let mid = range[0] + width;
    if intensity.len() != sigma.len()
        || intensity.len() != omega.len()
        || !range.iter().all(|v| v.is_finite())
        || width <= 0.0
        || !width.is_finite()
        || !beta.is_finite()
        || !variance.is_finite()
        || variance < 0.0
    {
        return Err(invalid("invalid scan-scaling application parameters"));
    }
    intensity
        .iter()
        .zip(sigma)
        .zip(omega)
        .map(|((&i, &s), &om)| {
            if !i.is_finite()
                || !s.is_finite()
                || s <= 0.0
                || !om.is_finite()
                || om < range[0]
                || om > range[1]
            {
                return Err(invalid(
                    "finite intensity/positive sigma and in-range omega required",
                ));
            }
            let x = (om - mid) / width;
            let factor = (-beta * x).exp();
            let value = i * factor;
            let d_beta = -x * value;
            let row = [
                factor,
                value,
                s * factor,
                d_beta,
                -beta * value / width,
                d_beta.abs() * variance.sqrt(),
            ];
            if row.iter().all(|v| v.is_finite()) && row[0] > 0.0 && row[2] > 0.0 {
                Ok(row)
            } else {
                Err(invalid("overflow applying scan scaling"))
            }
        })
        .collect()
}
