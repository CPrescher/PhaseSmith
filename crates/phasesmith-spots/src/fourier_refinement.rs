//! Bounded independent-vector Newton ascent of the positional cosine objective.
//! No convergence relaxation: twenty passes, the original four line scales,
//! spectral damping, 0.3-Angstrom step cap and initial-coordinate box.

use crate::fft::{M3, V3};
use crate::{SpotError, reciprocal};
use rayon::prelude::*;

const ITERATIONS: usize = 20;
const LINE_SCALES: [f64; 4] = [1., 0.5, 0.25, 0.125];
const MAX_QUERIES: usize = 1 + ITERATIONS * LINE_SCALES.len();

fn minimum_eigenvalue(h: M3) -> f64 {
    // Characteristic polynomial of a symmetric 3x3 matrix. Match eigvalsh's
    // lower-triangle convention; upper/lower entries may differ by one ulp.
    let scale = h.iter().flatten().fold(0_f64, |a, x| a.max(x.abs()));
    if scale == 0. {
        return 0.;
    }
    let a = h.map(|r| r.map(|v| v / scale));
    let mean = (a[0][0] + a[1][1] + a[2][2]) / 3.;
    let diagonal = [a[0][0] - mean, a[1][1] - mean, a[2][2] - mean];
    let off = a[1][0] * a[1][0] + a[2][0] * a[2][0] + a[2][1] * a[2][1];
    if off == 0. {
        return h[0][0].min(h[1][1]).min(h[2][2]);
    }
    let p = ((diagonal.iter().map(|v| v * v).sum::<f64>() + 2. * off) / 6.).sqrt();
    if p == 0. {
        // Subnormal off-diagonal squares can survive the first sum but
        // underflow on division by six. They are below diagonal precision.
        return h[0][0].min(h[1][1]).min(h[2][2]);
    }
    let d = diagonal.map(|v| v / p);
    let b = [a[1][0] / p, a[2][0] / p, a[2][1] / p];
    let determinant = d[0] * (d[1] * d[2] - b[2] * b[2]) - b[0] * (b[0] * d[2] - b[2] * b[1])
        + b[1] * (b[0] * b[2] - d[1] * b[1]);
    let angle = (0.5 * determinant).clamp(-1., 1.).acos() / 3.;
    (mean + 2. * p * (angle + 2. * std::f64::consts::PI / 3.).cos()) * scale
}

fn solve(mut a: M3, mut b: V3) -> Result<V3, SpotError> {
    for k in 0..3 {
        let mut pivot = k;
        for i in k + 1..3 {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        if a[pivot][k] == 0. {
            return Err(SpotError("singular Fourier Newton step".into()));
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..3 {
            let factor = a[i][k] / a[k][k];
            let pivot_row = a[k];
            for (j, value) in a[i].iter_mut().enumerate().skip(k + 1) {
                *value -= factor * pivot_row[j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = [0.; 3];
    for i in (0..3).rev() {
        let mut value = b[i];
        for j in i + 1..3 {
            value -= a[i][j] * x[j];
        }
        x[i] = value / a[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(SpotError("nonfinite Fourier Newton step".into()));
    }
    Ok(x)
}

fn step(m: &[f64; 13]) -> Result<V3, SpotError> {
    let mut h: M3 = std::array::from_fn(|i| std::array::from_fn(|j| -m[4 + 3 * i + j]));
    let damping = (0.01 - minimum_eigenvalue(h)).max(0.);
    for (i, row) in h.iter_mut().enumerate() {
        row[i] += damping;
    }
    let mut delta = solve(h, [m[1], m[2], m[3]])?;
    let norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = (0.3 / norm.max(1e-12)).min(1.);
    for v in &mut delta {
        *v *= scale;
    }
    Ok(delta)
}

/// Refined position, all fused moments and [unique evaluations, exact cache hits].
pub(crate) type Refined = (V3, [f64; 13], [usize; 2]);

/// Twenty iterations with four trials plus the initial query require at most
/// 81 distinct records. Separate contiguous keys make exact lookup cache-local.
struct TrialMemo {
    keys: [[u64; 3]; MAX_QUERIES],
    moments: [[f64; 13]; MAX_QUERIES],
    len: usize,
}
impl TrialMemo {
    fn new() -> Self {
        Self {
            keys: [[0; 3]; MAX_QUERIES],
            moments: [[0.; 13]; MAX_QUERIES],
            len: 0,
        }
    }
    fn get(&self, key: &[u64; 3]) -> Option<[f64; 13]> {
        self.keys[..self.len]
            .iter()
            .position(|old| old == key)
            .map(|i| self.moments[i])
    }
    fn insert(&mut self, key: [u64; 3], m: [f64; 13]) {
        self.keys[self.len] = key;
        self.moments[self.len] = m;
        self.len += 1;
    }
}

fn one(
    cloud: &reciprocal::PreparedFourier<'_>,
    initial: V3,
    radius: f64,
) -> Result<Refined, SpotError> {
    let mut memo = TrialMemo::new();
    let mut calls = 0;
    let mut hits = 0;
    let mut evaluate = |v: V3| -> Result<[f64; 13], SpotError> {
        let key = v.map(f64::to_bits);
        if let Some(m) = memo.get(&key) {
            hits += 1;
            return Ok(m);
        }
        let m = cloud.evaluate(v)?;
        memo.insert(key, m);
        calls += 1;
        Ok(m)
    };
    let mut v = initial;
    let mut m = evaluate(v)?;
    for _ in 0..ITERATIONS {
        let delta = step(&m)?;
        let mut improved = false;
        for scale in LINE_SCALES {
            let trial = std::array::from_fn(|j| {
                (v[j] + scale * delta[j]).clamp(initial[j] - radius, initial[j] + radius)
            });
            let next = evaluate(trial)?;
            if next[0] > m[0] {
                v = trial;
                m = next;
                improved = true;
                break;
            }
        }
        if !improved {
            break;
        }
    }
    Ok((v, m, [calls, hits]))
}

pub(crate) fn refine(
    pool: &rayon::ThreadPool,
    points: &[V3],
    initial: &[V3],
    radius: f64,
) -> Result<Vec<Refined>, SpotError> {
    reciprocal::check_fourier(points, initial)?;
    if points.len() > 100_000
        || initial.len() > 400
        || !radius.is_finite()
        || radius <= 0.
        || radius > 1000.
        || points
            .iter()
            .chain(initial)
            .flatten()
            .any(|v| v.abs() > 1000.)
    {
        return Err(SpotError(
            "invalid bounded Fourier refinement inputs".into(),
        ));
    }
    let cloud = reciprocal::PreparedFourier::new(points);
    pool.install(|| {
        initial
            .par_iter()
            .map(|&v| one(&cloud, v, radius))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    #[test]
    fn bounded_trials_keep_coordinate_and_moment_bits() {
        let mut memo = TrialMemo::new();
        let mut expected = HashMap::new();
        for i in 0_u32..81 {
            let key = [
                f64::from(i).to_bits(),
                (-f64::from(i)).to_bits(),
                if i % 2 == 0 {
                    0_f64.to_bits()
                } else {
                    (-0_f64).to_bits()
                },
            ];
            let m = std::array::from_fn(|j| {
                if j == 0 && i % 2 == 1 {
                    -0.
                } else {
                    f64::from(i) + j as f64
                }
            });
            assert!(memo.get(&key).is_none());
            memo.insert(key, m);
            expected.insert(key, m);
        }
        for (key, m) in expected {
            assert_eq!(
                memo.get(&key).unwrap().map(f64::to_bits),
                m.map(f64::to_bits)
            );
        }
        assert!(memo.get(&[u64::MAX; 3]).is_none());
    }

    #[test]
    fn diagonal_repeated_and_rotated_eigenvalues() {
        assert_eq!(
            minimum_eigenvalue([[3., 0., 0.], [0., -2., 0.], [0., 0., 7.]]),
            -2.
        );
        assert_eq!(
            minimum_eigenvalue([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]),
            2.
        );
        assert_eq!(
            minimum_eigenvalue([[-1., 1.6e-162, 0.], [1.6e-162, -1., 0.], [0., 0., -1.]]),
            -1.
        );
        let h = [[2., 1., 0.], [1., 2., 0.], [0., 0., 4.]];
        assert!((minimum_eigenvalue(h) - 1.).abs() < 2e-15);
    }
}
