//! Signed GLS in a small empirical shape basis; intensity is the sum of components.
//! Unit-mass component grids make the remaining directions zero-integral shape modes.
use crate::{
    SpotError, invalid, local_background,
    profiles::{self, Grid, Observations},
};
use std::collections::HashMap;
type Four = [f64; 4];
type Matrix = [[f64; 4]; 4];

fn inverse(a: Matrix, k: usize) -> Option<Matrix> {
    let scale = std::array::from_fn::<_, 4, _>(|i| if i < k { a[i][i].sqrt() } else { 1. });
    if scale[..k].iter().any(|v| !v.is_finite() || *v <= 0.) {
        return None;
    }
    let mut rows = [[0.; 8]; 4];
    for i in 0..k {
        for j in 0..k {
            rows[i][j] = a[i][j] / (scale[i] * scale[j]);
        }
        rows[i][4 + i] = 1.;
    }
    for c in 0..k {
        let pivot = (c..k).max_by(|&i, &j| rows[i][c].abs().total_cmp(&rows[j][c].abs()))?;
        rows.swap(c, pivot);
        let d = rows[c][c];
        if !d.is_finite() || d.abs() < 1e-9 {
            return None;
        }
        for value in &mut rows[c] {
            *value /= d;
        }
        for r in 0..k {
            if r != c {
                let f = rows[r][c];
                let p = rows[c];
                for j in 0..8 {
                    rows[r][j] -= f * p[j];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < k && j < k {
                rows[i][4 + j] / (scale[i] * scale[j])
            } else {
                0.
            }
        })
    }))
}
// Enumerate all faces of the two signed mixture cones. Every shape remains
// nonnegative up to the sign of its total amplitude; weak I can stay negative.
fn physical_coefficients(normal: Matrix, rhs: Four, k: usize) -> Four {
    let mut best = [0.; 4];
    let mut cost = 0.;
    for mask in 1..(1_usize << k) {
        let ids: Vec<_> = (0..k).filter(|j| mask & (1 << j) != 0).collect();
        let active = ids.len();
        let matrix = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                if a < active && b < active {
                    normal[ids[a]][ids[b]]
                } else {
                    0.
                }
            })
        });
        let Some(inv) = inverse(matrix, active) else {
            continue;
        };
        let sub_rhs = std::array::from_fn(|a| if a < active { rhs[ids[a]] } else { 0. });
        let solution = mv(inv, sub_rhs);
        if !(solution[..active].iter().all(|v| *v >= 0.)
            || solution[..active].iter().all(|v| *v <= 0.))
        {
            continue;
        }
        let mut beta = [0.; 4];
        for (j, &id) in ids.iter().enumerate() {
            beta[id] = solution[j];
        }
        let q = dot(beta, mv(normal, beta)) - 2. * dot(beta, rhs);
        if q < cost {
            cost = q;
            best = beta;
        }
    }
    best
}
fn mv(a: Matrix, x: Four) -> Four {
    std::array::from_fn(|i| (0..4).map(|j| a[i][j] * x[j]).sum())
}
fn dot(a: Four, b: Four) -> f64 {
    (0..4).map(|i| a[i] * b[i]).sum()
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn mv3(a: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * x[j]).sum())
}

/// Conditional measurements, component coefficients and their joint covariance.
pub struct Result {
    /// I, sigma, minimum component coverage, reduced chi-square, count, correlation and gradients.
    pub values: Vec<[f64; 10]>,
    /// Measurement flags, including rank failure, partiality and model mismatch.
    pub flags: Vec<u32>,
    /// Up to four signed component amplitudes; unused entries are zero.
    pub coefficients: Vec<Four>,
    /// Row-major 4-by-4 conditional coefficient covariance.
    pub covariance: Vec<[f64; 16]>,
}

/// Fit a fixed set of normalised empirical shapes without clipping weak amplitudes.
/// # Errors
/// Invalid observations, models, bounded width factors or coefficient dimensions.
#[allow(clippy::too_many_arguments, clippy::too_many_lines, clippy::float_cmp)] // Exact parameter identities and zero fractions select algebraic branches.
pub fn fit(
    o: &Observations<'_>,
    profiles: &[f64],
    grid: Grid,
    assignments: &[usize],
    shifts: &[[f64; 3]],
    factors: &[[f64; 3]],
    fixed_fractions: Option<&[Four]>,
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
) -> std::result::Result<Result, SpotError> {
    profiles::validate_fit(
        o,
        profiles,
        grid,
        assignments,
        shifts,
        minimum_mass,
        gain,
        read_noise,
    )?;
    let k = factors.len();
    let n = o.centres.len();
    let voxels = grid.size();
    if !(2..=4).contains(&k)
        || factors[0] != [1.; 3]
        || factors
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || !(0.5..=2.5).contains(v))
    {
        return Err(invalid(
            "variation factors require 2..4 finite triples in [0.5,2.5], starting with ones",
        ));
    }
    if let Some(fixed) = fixed_fractions {
        if fixed.len() != n
            || fixed.iter().any(|f| {
                f.iter().any(|v| !v.is_finite() || *v < 0.)
                    || (f[..k].iter().sum::<f64>() - 1.).abs() > 1e-10
                    || f[k..].iter().any(|v| *v != 0.)
            })
        {
            return Err(invalid("fixed fractions must be nonnegative unit-sum rows"));
        }
    }
    let grids: Vec<_> = factors
        .iter()
        .map(|f| Grid {
            shape: grid.shape,
            step: std::array::from_fn(|j| grid.step[j] * f[j]),
        })
        .collect();
    let components = k;
    let k = if fixed_fractions.is_some() { 1 } else { k };
    let mut probability = vec![[0.; 4]; o.samples.len()];
    let mut masses = vec![[0.; 4]; n];
    let mut flags = vec![0; n];
    let mut occupancy = HashMap::<(u64, u64, usize, usize), u32>::new();
    for g in 0..o.ids.len() {
        let id = o.ids[g];
        let model = assignments[id];
        let centre = o.shifted(id, shifts[model]);
        for (i, p) in probability
            .iter_mut()
            .enumerate()
            .take(o.offsets[g + 1])
            .skip(o.offsets[g])
        {
            for j in 0..components {
                if fixed_fractions.is_some_and(|f| f[id][j] == 0.) {
                    continue;
                }
                o.weights(grids[j], o.bounds(g, i), centre, id, |v, w, _| {
                    p[j] += profiles[model * voxels + v] * w;
                });
            }
            if let Some(fixed) = fixed_fractions {
                *p = [dot(*p, fixed[id]), 0., 0., 0.];
            }
            if p[..k].iter().any(|v| *v > 0.) {
                let b = o.boxes[g];
                let local = i - o.offsets[g];
                let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
                *occupancy
                    .entry((
                        o.angles[g][0].to_bits(),
                        o.angles[g][1].to_bits(),
                        xy.0,
                        xy.1,
                    ))
                    .or_insert(0) += 1;
                flags[id] |= o.flags[g] & (crate::BACKGROUND | local_background::MODEL);
                if !o.valid[i] {
                    flags[id] |= o.flags[g] & (crate::MASKED | crate::SATURATED);
                }
            }
            if o.valid[i] {
                for (j, m) in masses[id].iter_mut().enumerate().take(k) {
                    *m += p[j];
                }
            }
        }
    }
    for g in 0..o.ids.len() {
        let id = o.ids[g];
        for (i, p) in probability
            .iter()
            .enumerate()
            .take(o.offsets[g + 1])
            .skip(o.offsets[g])
        {
            if p[..k].iter().all(|v| *v <= 0.) {
                continue;
            }
            let b = o.boxes[g];
            let local = i - o.offsets[g];
            let xy = (b.x0 + local % (b.x1 - b.x0), b.y0 + local / (b.x1 - b.x0));
            if occupancy[&(
                o.angles[g][0].to_bits(),
                o.angles[g][1].to_bits(),
                xy.0,
                xy.1,
            )] > 1
            {
                flags[id] |= crate::OVERLAP;
            }
        }
        if masses[id][..k].iter().any(|v| *v < 0.999) {
            flags[id] |= o.flags[g] & (crate::EDGE | 256);
        }
    }
    let mut coefficients = vec![[0.; 4]; n];
    let mut covariance = vec![[0.; 16]; n];
    let mut values = vec![[f64::NAN; 10]; n];
    let iterations = if expected_noise { 12 } else { 1 };
    for iteration in 0..iterations {
        let last = iteration + 1 == iterations;
        let mut normal = vec![[[0.; 4]; 4]; n];
        let mut rhs = vec![[0.; 4]; n];
        let mut data_cost = vec![0.; n];
        let mut counts = vec![0_usize; n];
        let mut shape_raw = vec![[[0.; 4]; 3]; if last { n } else { 0 }];
        let mut shape_normal = vec![[[[0.; 4]; 4]; 3]; if last { n } else { 0 }];
        let mut sums = vec![[0.; 4]; n];
        let mut raw_product = vec![[[0.; 4]; 4]; n];
        let mut raw_rhs = vec![[0.; 4]; n];
        let mut data_sums = vec![[0.; 2]; n];
        for g in 0..o.ids.len() {
            let id = o.ids[g];
            let model = assignments[id];
            let centre = o.shifted(id, shifts[model]);
            let background =
                std::array::from_fn(|i| std::array::from_fn(|j| o.covariance[g][i * 3 + j]));
            let zero = background.iter().flatten().all(|v| v.abs() < 1e-25);
            let mut middle = if zero {
                [[0.; 3]; 3]
            } else {
                let Some(inv) = profiles::inverse(background) else {
                    flags[id] |= profiles::FAILED;
                    continue;
                };
                inv
            };
            let mut pu = [[0.; 3]; 4];
            let mut du = [0.; 3];
            let mut pp = [[0.; 4]; 4];
            let mut pd = [0.; 4];
            let mut dd = 0.;
            let mut dp_u = [[[0.; 3]; 4]; 3];
            let mut dp_data = [[0.; 4]; 3];
            let mut dp_p = [[[0.; 4]; 4]; 3];
            for (i, &p) in probability
                .iter()
                .enumerate()
                .take(o.offsets[g + 1])
                .skip(o.offsets[g])
            {
                if !o.valid[i] || p[..k].iter().all(|v| *v <= 0.) {
                    continue;
                }
                let s = o.samples[i];
                let x = o.basis(g, i);
                let signal = dot(p, coefficients[id]);
                let variance = if expected_noise && iteration > 0 {
                    ((s[2] + signal.max(0.)).max(0.) / gain + read_noise.powi(2)).max(1.)
                } else {
                    s[1]
                };
                let w = 1. / variance;
                dd += w * s[0] * s[0];
                counts[id] += 1;
                data_sums[id][0] += s[0];
                data_sums[id][1] += s[0] * s[0];
                for a in 0..3 {
                    du[a] += w * x[a] * s[0];
                    for b in 0..3 {
                        if !zero {
                            middle[a][b] += w * x[a] * x[b];
                        }
                    }
                }
                for a in 0..k {
                    pd[a] += w * p[a] * s[0];
                    sums[id][a] += p[a];
                    raw_rhs[id][a] += p[a] * s[0];
                    for b in 0..k {
                        pp[a][b] += w * p[a] * p[b];
                        raw_product[id][a][b] += p[a] * p[b];
                    }
                    for b in 0..3 {
                        pu[a][b] += w * p[a] * x[b];
                    }
                    if last {
                        let mut derivative = [0.; 3];
                        if let Some(fixed) = fixed_fractions {
                            for j in 0..components {
                                if fixed[id][j] == 0. {
                                    continue;
                                }
                                o.weights(grids[j], o.bounds(g, i), centre, id, |v, _, d| {
                                    for axis in 0..3 {
                                        derivative[axis] +=
                                            fixed[id][j] * profiles[model * voxels + v] * d[axis];
                                    }
                                });
                            }
                        } else {
                            o.weights(grids[a], o.bounds(g, i), centre, id, |v, _, d| {
                                for axis in 0..3 {
                                    derivative[axis] += profiles[model * voxels + v] * d[axis];
                                }
                            });
                        }
                        for axis in 0..3 {
                            dp_data[axis][a] += w * derivative[axis] * s[0];
                            for b in 0..k {
                                dp_p[axis][a][b] += w * derivative[axis] * p[b];
                            }
                            for b in 0..3 {
                                dp_u[axis][a][b] += w * derivative[axis] * x[b];
                            }
                        }
                    }
                }
            }
            let inv = if zero {
                [[0.; 3]; 3]
            } else {
                let Some(inv) = profiles::inverse(middle) else {
                    flags[id] |= profiles::FAILED;
                    continue;
                };
                inv
            };
            let wd = mv3(inv, du);
            data_cost[id] += dd - dot3(du, wd);
            for a in 0..k {
                rhs[id][a] += pd[a] - dot3(pu[a], wd);
                for b in 0..k {
                    normal[id][a][b] += pp[a][b] - dot3(pu[a], mv3(inv, pu[b]));
                }
                if last {
                    for axis in 0..3 {
                        shape_raw[id][axis][a] += dp_data[axis][a] - dot3(dp_u[axis][a], wd);
                        for b in 0..k {
                            shape_normal[id][axis][a][b] +=
                                dp_p[axis][a][b] - dot3(dp_u[axis][a], mv3(inv, pu[b]));
                        }
                    }
                }
            }
        }
        for id in 0..n {
            let Some(inv) = inverse(normal[id], k) else {
                flags[id] |= profiles::NO_PROFILE;
                continue;
            };
            coefficients[id] = if fixed_fractions.is_some() {
                mv(inv, rhs[id])
            } else {
                physical_coefficients(normal[id], rhs[id], k)
            };
            let beta = coefficients[id];
            let intensity = beta[..k].iter().sum::<f64>();
            let variance = (0..k)
                .map(|a| (0..k).map(|b| inv[a][b]).sum::<f64>())
                .sum::<f64>();
            if variance <= 0. || !variance.is_finite() {
                flags[id] |= profiles::FAILED;
                continue;
            }
            covariance[id] = std::array::from_fn(|v| inv[v / 4][v % 4]);
            let cost =
                (data_cost[id] - 2. * dot(beta, rhs[id]) + dot(beta, mv(normal[id], beta))).max(0.);
            let count = counts[id] as f64;
            let mean_signal = dot(beta, sums[id]);
            let corr_num = dot(beta, raw_rhs[id]) - mean_signal * data_sums[id][0] / count;
            let corr_den = ((dot(beta, mv(raw_product[id], beta)) - mean_signal.powi(2) / count)
                .max(0.)
                * (data_sums[id][1] - data_sums[id][0].powi(2) / count).max(0.))
            .sqrt();
            let mut derivatives = [0.; 4];
            if last {
                derivatives[0] = if intensity == 0. {
                    0.
                } else {
                    2. * (dot(beta, mv(normal[id], beta)) - dot(beta, rhs[id])) / intensity
                };
                for axis in 0..3 {
                    derivatives[axis + 1] = -2.
                        * dot(
                            beta,
                            std::array::from_fn(|a| {
                                shape_raw[id][axis][a] - dot(shape_normal[id][axis][a], beta)
                            }),
                        );
                }
            }
            values[id] = [
                intensity,
                variance.sqrt(),
                if intensity == 0. {
                    masses[id][..k].iter().copied().fold(1., f64::min)
                } else {
                    dot(beta, masses[id]) / intensity
                },
                cost / counts[id]
                    .saturating_sub(if fixed_fractions.is_some() { 1 } else { k })
                    .max(1) as f64,
                count,
                intensity.signum() * corr_num / corr_den,
                derivatives[0],
                derivatives[1],
                derivatives[2],
                derivatives[3],
            ];
        }
    }
    for id in 0..n {
        if values[id][2] < minimum_mass {
            flags[id] |= profiles::PARTIAL;
        }
        if values[id][0] > 5. * values[id][1] && values[id][3] > 4. {
            flags[id] |= profiles::MISMATCH;
        }
    }
    if let Some(fixed) = fixed_fractions {
        for id in 0..n {
            let amplitude = coefficients[id][0];
            let variance = covariance[id][0];
            coefficients[id] = fixed[id].map(|v| v * amplitude);
            covariance[id] =
                std::array::from_fn(|v| variance * fixed[id][v / 4] * fixed[id][v % 4]);
        }
    }
    Ok(Result {
        values,
        flags,
        coefficients,
        covariance,
    })
}
