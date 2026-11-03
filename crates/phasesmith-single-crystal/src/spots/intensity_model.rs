//! First-principles slab response and shared empirical intensity-model passes.
use crate::spots::{SpotError, invalid};

/// Slab absorption, relative correction, mean depth, then their cosine partials.
/// Lengths use the same unit; attenuation uses its inverse. Zero optical depth
/// returns the thin-slab limit (relative correction=c, mean depth=t/2).
/// # Errors
/// Nonfinite/nonphysical inputs or more than a million observations.
pub fn slab(cosines: &[f64], thickness: f64, mu: f64) -> Result<Vec<[f64; 6]>, SpotError> {
    if cosines.len() > 1_000_000
        || !thickness.is_finite()
        || thickness < 0.
        || !mu.is_finite()
        || mu < 0.
        || !(thickness * mu).is_finite()
        || cosines
            .iter()
            .any(|c| !c.is_finite() || *c <= 0. || *c > 1.)
    {
        return Err(invalid("invalid homogeneous slab response"));
    }
    let tau0 = thickness * mu;
    let eta0 = -(-tau0).exp_m1();
    let result: Vec<_> = cosines
        .iter()
        .map(|&c| {
            let tau = tau0 / c;
            if tau == 0. {
                return [0., c, thickness / 2., 0., 1., 0.];
            }
            if tau > 700. {
                return [1., eta0, c / mu, 0., 0., 1. / mu];
            }
            let e = (-tau).exp();
            let eta = -(-tau).exp_m1();
            let (depth, dh) = if tau < 1e-3 {
                let t2 = tau * tau;
                (
                    thickness * (0.5 - tau / 12. + tau * t2 / 720. - tau * t2 * t2 / 30240.),
                    thickness / c * (tau / 12. - tau * t2 / 240. + tau * t2 * t2 / 6048.),
                )
            } else {
                (
                    thickness * (1. / tau - e / eta),
                    thickness / c * (1. / tau - tau * e / (eta * eta)),
                )
            };
            let de = -tau * e / c;
            let relative = eta0 / eta;
            [eta, relative, depth, de, -relative * de / eta, dh]
        })
        .collect();
    if result.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("homogeneous slab arithmetic overflow"));
    }
    Ok(result)
}

/// Dense contiguous empirical design, partitioned by its parameter sectors.
pub struct Design<'a> {
    /// Number of observations.
    pub rows: usize,
    /// Free frame, detector-node and absorption-sector widths.
    pub widths: [usize; 3],
    /// Row-major design matrices in the same order.
    pub matrices: [&'a [f64]; 3],
}

/// Flattened log scales, raw D/A components and row-major parameter partials.
pub type Evaluation = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Log(F*D*A), raw D/A components and all analytical parameter partials.
/// F=exp(Xf*b), D=1+Xd*b, A=1+Xa*b. A positive floor explicitly
/// defines the optimizer's extension outside its physical positive domain.
/// # Errors
/// Invalid dimensions, nonfinite arrays, coefficient budget or arithmetic overflow.
pub fn evaluate(d: &Design<'_>, parameters: &[f64], floor: f64) -> Result<Evaluation, SpotError> {
    let p: usize = d.widths.iter().sum();
    if d.rows > 1_000_000
        || p == 0
        || p > 256
        || d.rows.saturating_mul(p) > 20_000_000
        || parameters.len() != p
        || !floor.is_finite()
        || floor <= 0.
        || parameters.iter().any(|v| !v.is_finite())
        || d.widths
            .iter()
            .zip(d.matrices)
            .any(|(&w, m)| m.len() != d.rows * w || m.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid("invalid empirical intensity design"));
    }
    let mut values = vec![0.; d.rows];
    let mut components = vec![0.; d.rows * 2];
    let mut jac = vec![0.; d.rows * p];
    for row in 0..d.rows {
        let mut offset = 0;
        for sector in 0..3 {
            let w = d.widths[sector];
            let x = &d.matrices[sector][row * w..(row + 1) * w];
            let dot: f64 = x
                .iter()
                .zip(&parameters[offset..offset + w])
                .map(|(a, b)| a * b)
                .sum();
            let derivative = if sector == 0 {
                values[row] += dot;
                1.
            } else {
                let component = 1. + dot;
                components[2 * row + sector - 1] = component;
                values[row] += component.max(floor).ln();
                if component > floor {
                    1. / component
                } else {
                    0.
                }
            };
            for (column, &v) in x.iter().enumerate() {
                jac[row * p + offset + column] = v * derivative;
            }
            offset += w;
        }
    }
    if values
        .iter()
        .chain(&components)
        .chain(&jac)
        .any(|v| !v.is_finite())
    {
        return Err(invalid("empirical intensity arithmetic overflow"));
    }
    Ok((values, components, jac))
}

/// Weighted within-family centering of a row-major matrix in deterministic order.
/// # Errors
/// Invalid positive weights, groups, shapes, nonfinite values or memory budget.
pub fn center(
    values: &[f64],
    columns: usize,
    groups: &[usize],
    weights: &[f64],
) -> Result<Vec<f64>, SpotError> {
    let n = groups.len();
    if columns == 0
        || columns > 257
        || n > 1_000_000
        || values.len() != n * columns
        || values.len() > 20_000_000
        || weights.len() != n
        || values.iter().any(|v| !v.is_finite())
        || weights.iter().any(|v| !v.is_finite() || *v <= 0.)
        || groups.iter().any(|v| *v >= n)
    {
        return Err(invalid("invalid family centering arrays"));
    }
    let count = groups.iter().max().map_or(0, |v| v + 1);
    let mut sums = vec![0.; count * columns];
    let mut totals = vec![0.; count];
    for row in 0..n {
        let g = groups[row];
        totals[g] += weights[row];
        for col in 0..columns {
            sums[g * columns + col] += weights[row] * values[row * columns + col];
        }
    }
    let mut out = values.to_vec();
    for row in 0..n {
        let g = groups[row];
        for col in 0..columns {
            out[row * columns + col] -= sums[g * columns + col] / totals[g];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(invalid("family centering arithmetic overflow"));
    }
    Ok(out)
}

/// Conditional GLS normal matrix, right hand side and y'Wy in one sample pass.
/// Background plane uncertainty is marginalized per group with Woodbury.
/// Caller supplies a frozen inverse independent-pixel variance; no intensity
/// threshold selects samples. Columns include amplitude profile and partials.
/// # Errors
/// Invalid dimensions, nonfinite inputs, or singular covariance update.
#[allow(clippy::too_many_arguments)]
pub fn normal(
    signal: &[f64],
    weights: &[f64],
    basis: &[[f64; 3]],
    offsets: &[usize],
    covariance: &[[f64; 9]],
    design: &[f64],
    columns: usize,
) -> Result<Vec<f64>, SpotError> {
    let n = signal.len();
    let g = covariance.len();
    if !(1..=6).contains(&columns)
        || n > 1_000_000
        || weights.len() != n
        || basis.len() != n
        || offsets.len() != g + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&n)
        || offsets.windows(2).any(|w| w[0] > w[1])
        || design.len() != n * columns
        || signal
            .iter()
            .chain(weights)
            .chain(design)
            .chain(basis.iter().flatten())
            .chain(covariance.iter().flatten())
            .any(|v| !v.is_finite())
        || weights.iter().any(|v| *v < 0.)
    {
        return Err(invalid("invalid conditional GLS arrays"));
    }
    // Augment the design with y; its Gram matrix contains all required terms.
    let q = columns + 1;
    let mut total = vec![0.; q * q];
    for group in 0..g {
        let mut cross = vec![[0.; 3]; q];
        let mut gram = [[0.; 3]; 3];
        let mut local = vec![0.; q * q];
        for i in offsets[group]..offsets[group + 1] {
            let mut z = [0.; 7];
            z[..columns].copy_from_slice(&design[i * columns..(i + 1) * columns]);
            z[columns] = signal[i];
            let w = weights[i];
            let x = basis[i];
            for a in 0..3 {
                for b in 0..3 {
                    gram[a][b] += w * x[a] * x[b];
                }
            }
            for a in 0..q {
                for b in 0..3 {
                    cross[a][b] += w * z[a] * x[b];
                }
                for b in 0..q {
                    local[a * q + b] += w * z[a] * z[b];
                }
            }
        }
        let c: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| covariance[group][3 * i + j]));
        let m: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                f64::from(i == j) + (0..3).map(|k| c[i][k] * gram[k][j]).sum::<f64>()
            })
        });
        let inv = crate::spots::profiles::inverse(m)
            .ok_or_else(|| invalid("singular background update"))?;
        let b: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| inv[i][k] * c[k][j]).sum::<f64>())
        });
        for a in 0..q {
            for d in 0..q {
                let correction: f64 = (0..3)
                    .map(|i| cross[a][i] * (0..3).map(|j| b[i][j] * cross[d][j]).sum::<f64>())
                    .sum();
                total[a * q + d] += local[a * q + d] - correction;
            }
        }
    }
    if total.iter().any(|v| !v.is_finite()) {
        return Err(invalid("GLS arithmetic overflow"));
    }
    Ok(total)
}

/// Signed fixed linear-profile amplitude with observed sandwich uncertainty.
/// Design weights are frozen independently of the measured signal, avoiding
/// noise-driven peak motion or width selection for weak reflections.
/// # Errors
/// Invalid GLS input or a nonidentifiable amplitude.
#[allow(clippy::too_many_arguments)]
pub fn fixed_filter(
    signal: &[f64],
    variance: &[f64],
    weights: &[f64],
    basis: &[[f64; 3]],
    offsets: &[usize],
    covariance: &[[f64; 9]],
    mass: &[f64],
) -> Result<[f64; 2], SpotError> {
    let gram = normal(signal, weights, basis, offsets, covariance, mass, 1)?;
    if variance.len() != signal.len()
        || variance.iter().any(|v| !v.is_finite() || *v < 0.)
        || gram[0] <= 0.
    {
        return Err(invalid("nonidentifiable fixed profile"));
    }
    let mut var = 0.;
    for g in 0..covariance.len() {
        let mut m = [[0.; 3]; 3];
        let mut z = [0.; 3];
        for i in offsets[g]..offsets[g + 1] {
            for a in 0..3 {
                z[a] += weights[i] * basis[i][a] * mass[i];
                for b in 0..3 {
                    m[a][b] += weights[i] * basis[i][a] * basis[i][b];
                }
            }
        }
        let c: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| covariance[g][3 * i + j]));
        let update: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                f64::from(i == j) + (0..3).map(|k| c[i][k] * m[k][j]).sum::<f64>()
            })
        });
        let inv = crate::spots::profiles::inverse(update)
            .ok_or_else(|| invalid("singular background update"))?;
        let cz: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| c[i][j] * z[j]).sum());
        let correction: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| inv[i][j] * cz[j]).sum());
        let mut plane = [0.; 3];
        for i in offsets[g]..offsets[g + 1] {
            let a = weights[i]
                * (mass[i] - (0..3).map(|j| basis[i][j] * correction[j]).sum::<f64>())
                / gram[0];
            var += a * a * variance[i];
            for j in 0..3 {
                plane[j] += basis[i][j] * a;
            }
        }
        var += (0..3)
            .map(|i| plane[i] * (0..3).map(|j| c[i][j] * plane[j]).sum::<f64>())
            .sum::<f64>();
    }
    if !var.is_finite() || var < 0. {
        return Err(invalid("invalid fixed profile variance"));
    }
    Ok([gram[1] / gram[0], var.sqrt()])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slab_limits_and_cosine_derivatives() {
        for mu in [0., 1e-8, 0.122_453_913, 9., 1000.] {
            let c = 0.83;
            let h = 1e-6;
            let a = slab(&[c], 0.75, mu).unwrap()[0];
            let lo = slab(&[c - h], 0.75, mu).unwrap()[0];
            let hi = slab(&[c + h], 0.75, mu).unwrap()[0];
            for k in 0..3 {
                assert!(((hi[k] - lo[k]) / (2. * h) - a[k + 3]).abs() < 2e-8);
            }
        }
        assert!(slab(&[0.], 0.75, 9.).is_err());
        assert_eq!(slab(&[1.], 0., 9.).unwrap()[0], [0., 1., 0., 0., 1., 0.]);
    }
    #[test]
    fn model_partials_and_weighted_centers() {
        let x = [1., 2., -1., 0.5];
        let a = [0.2, 0.5];
        let d = Design {
            rows: 2,
            widths: [2, 0, 1],
            matrices: [&x, &[], &a],
        };
        let b = [0.1, -0.2, 0.3];
        let (_, _, j) = evaluate(&d, &b, 1e-8).unwrap();
        for k in 0..3 {
            let mut lo = b;
            let mut hi = b;
            lo[k] -= 1e-6;
            hi[k] += 1e-6;
            let l = evaluate(&d, &lo, 1e-8).unwrap().0;
            let h = evaluate(&d, &hi, 1e-8).unwrap().0;
            for i in 0..2 {
                assert!(((h[i] - l[i]) / 2e-6 - j[i * 3 + k]).abs() < 1e-9);
            }
        }
        let c = center(&[1., 3., 7., 11.], 2, &[0, 0], &[1., 2.]).unwrap();
        for (a, b) in c.iter().zip([-4., -16. / 3., 2., 8. / 3.]) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}
