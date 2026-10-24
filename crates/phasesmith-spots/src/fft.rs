//! Position-only FFT and lattice ownership engine reused from SFSCXRD.
//! See docs/sfscxrd-fft-provenance.json. No structure or material inputs.
// Numeric index conversions are bounded by the 256-cubed public grid budget.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::float_cmp // Integer-valued HKLs, exact ties and zero exclusions are observable.
)]
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;

/// Three reciprocal or direct Cartesian components.
pub type V3 = [f64; 3];
/// Three lattice basis row vectors.
pub type M3 = [V3; 3];

/// Bounded private worker pool for FFT and point ownership.
pub struct Engine {
    /// Number of workers (one through four).
    pub workers: usize,
    /// Shared FFT/ownership worker pool.
    pub pool: Arc<rayon::ThreadPool>,
}

/// Real part of the half-spectrum in (x,y,positive-z) order.
pub struct Field {
    /// Even grid edge.
    pub n: usize,
    /// Reciprocal-space deposition spacing.
    pub step: f64,
    /// Contiguous real cosine sums, length n*n*(n/2+1).
    pub values: Vec<f32>,
    /// Shared FFT/ownership worker pool.
    pub pool: Arc<rayon::ThreadPool>,
}

/// Cartesian dot product.
#[must_use]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Reciprocal row basis, with no 2*pi factor.
/// # Errors
/// Rejects singular or nonfinite determinants.
pub fn inverse_transpose(a: M3) -> Result<M3, String> {
    let cross = |x: V3, y: V3| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let b = [cross(a[1], a[2]), cross(a[2], a[0]), cross(a[0], a[1])];
    let det = dot(a[0], b[0]);
    if det == 0. || !det.is_finite() {
        return Err("Singular or nonfinite lattice basis".into());
    }
    Ok(b.map(|r| r.map(|v| v / det)))
}

/// `NumPy` rint uses ties-to-even, including negative half integers.
#[must_use]
pub fn indexed(q: V3, rows: M3, reciprocal: M3) -> (V3, f64) {
    let h = rows.map(|r| dot(q, r).round_ties_even());
    let predicted: V3 = std::array::from_fn(|j| {
        h[0] * reciprocal[0][j] + h[1] * reciprocal[1][j] + h[2] * reciprocal[2][j]
    });
    let d: V3 = std::array::from_fn(|j| q[j] - predicted[j]);
    (h, dot(d, d).sqrt())
}

impl Engine {
    /// Create a private pool.
    /// # Errors
    /// Rejects workers outside 1..=4 or pool creation failure.
    pub fn new(workers: usize) -> Result<Self, String> {
        if workers == 0 || workers > 4 {
            return Err("workers must lie in [1,4]".into());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            workers,
            pool: Arc::new(pool),
        })
    }

    /// Serial accumulation preserves the reference offset/point float32 order.
    /// FFT axes execute sequentially; each axis uses the same bounded pool.
    /// # Errors
    /// Rejects empty, nonfinite/out-of-grid points, invalid spacing or even grids outside 16..=256.
    #[allow(clippy::too_many_lines)] // Three explicit FFT axes preserve layout and accumulation order.
    pub fn field(&self, q: &[V3], n: usize, step: f64) -> Result<Field, String> {
        if n < 16 || n % 2 != 0 || !step.is_finite() || step <= 0. {
            return Err("grid size must be even and >=16, step positive and finite".into());
        }
        if q.is_empty()
            || q.iter()
                .flatten()
                .any(|x| !x.is_finite() || x.abs() / step >= n as f64 / 2. - 1.)
        {
            return Err("Nonfinite, empty or out-of-grid reciprocal coordinates".into());
        }
        let size = n
            .checked_mul(n)
            .and_then(|x| x.checked_mul(n))
            .ok_or("Grid size overflow")?;
        // Reject unreasonable allocations at the public boundary, before allocation.
        if size > 256usize.pow(3) {
            return Err("Grid exceeds supported maximum of 256 per axis".into());
        }
        let mut grid = vec![0f32; size];
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let off = [dx, dy, dz];
                    for p in q {
                        let x = p.map(|v| v / step);
                        let low = x.map(f64::floor);
                        let frac: V3 = std::array::from_fn(|i| x[i] - low[i]);
                        let ix: [usize; 3] = std::array::from_fn(|i| {
                            ((low[i] as i64 + off[i]).rem_euclid(n as i64)) as usize
                        });
                        let w: V3 =
                            std::array::from_fn(
                                |i| if off[i] == 1 { frac[i] } else { 1. - frac[i] },
                            );
                        grid[(ix[0] * n + ix[1]) * n + ix[2]] += (w[0] * w[1] * w[2]) as f32;
                    }
                }
            }
        }
        let nz = n / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let scratch_len = fft.get_inplace_scratch_len();
        let mut data = vec![Complex32::default(); n * n * nz];
        self.pool.install(|| {
            data.par_chunks_mut(nz)
                .zip(grid.par_chunks(n))
                .for_each_init(
                    || {
                        (
                            vec![Complex32::default(); n],
                            vec![Complex32::default(); scratch_len],
                        )
                    },
                    |(line, scratch), (out, input)| {
                        for (z, &v) in line.iter_mut().zip(input) {
                            *z = Complex32::new(v, 0.);
                        }
                        fft.process_with_scratch(line, scratch);
                        out.copy_from_slice(&line[..nz]);
                    },
                );
        });
        drop(grid);
        // Layout (x,z,y) makes the second transform contiguous.
        let mut other = vec![Complex32::default(); data.len()];
        self.pool.install(|| {
            other.par_chunks_mut(n).enumerate().for_each_init(
                || vec![Complex32::default(); scratch_len],
                |scratch, (line, out)| {
                    let x = line / nz;
                    let z = line % nz;
                    for (y, v) in out.iter_mut().enumerate() {
                        *v = data[(x * n + y) * nz + z];
                    }
                    fft.process_with_scratch(out, scratch);
                },
            );
        });
        // Layout (y,z,x) makes the third transform contiguous.
        self.pool.install(|| {
            data.par_chunks_mut(n).enumerate().for_each_init(
                || vec![Complex32::default(); scratch_len],
                |scratch, (line, out)| {
                    let y = line / nz;
                    let z = line % nz;
                    for (x, v) in out.iter_mut().enumerate() {
                        *v = other[(x * nz + z) * n + y];
                    }
                    fft.process_with_scratch(out, scratch);
                },
            );
        });
        drop(other);
        let mut values = vec![0f32; data.len()];
        self.pool.install(|| {
            values
                .par_chunks_mut(nz)
                .enumerate()
                .for_each(|(line, out)| {
                    let x = line / n;
                    let y = line % n;
                    for (z, v) in out.iter_mut().enumerate() {
                        *v = data[(y * nz + z) * n + x].re;
                    }
                });
        });
        Ok(Field {
            n,
            step,
            values,
            pool: self.pool.clone(),
        })
    }

    /// Nearest integer coordinates and Cartesian residual for each point.
    #[must_use]
    pub fn score(&self, q: &[V3], rows: M3, reciprocal: M3) -> Vec<(V3, f64)> {
        self.pool.install(|| {
            q.par_iter()
                .map(|&p| indexed(p, rows, reciprocal))
                .collect()
        })
    }

    /// Stable model order, strict distance/ambiguity cuts and origin exclusion.
    /// Does not allocate an N-points by N-models distance/sort matrix.
    #[must_use]
    pub fn assign(
        &self,
        q: &[V3],
        models: &[(M3, M3)],
        tol: f64,
        ratio: f64,
    ) -> (Vec<i32>, usize, usize) {
        let records: Vec<(i32, bool, bool)> = self.pool.install(|| {
            q.par_iter()
                .map(|&p| {
                    let (mut best, mut second, mut winner, mut count) =
                        (f64::INFINITY, f64::INFINITY, 0, 0);
                    for (i, &(rows, recip)) in models.iter().enumerate() {
                        let (h, d) = indexed(p, rows, recip);
                        let d = if h == [0.; 3] { f64::INFINITY } else { d };
                        if d < tol {
                            count += 1;
                        }
                        if d < best {
                            second = best;
                            best = d;
                            winner = i as i32 + 1;
                        } else if d < second {
                            second = d;
                        }
                    }
                    let inside = best < tol;
                    let ambiguous = inside
                        && models.len() > 1
                        && (best > ratio * second || (best == 0. && second == 0.));
                    (
                        if inside && !ambiguous { winner } else { 0 },
                        ambiguous,
                        count > 1,
                    )
                })
                .collect()
        });
        let ambiguous = records.iter().filter(|r| r.1).count();
        let multiple = records.iter().filter(|r| r.2).count();
        (
            records.into_iter().map(|r| r.0).collect(),
            ambiguous,
            multiple,
        )
    }
}

impl Field {
    /// Deterministically rank 26-neighbour maxima inside inclusive radius bounds.
    /// Periodic x/y and clamped z edges; ties retain all sites before rank truncation.
    #[must_use]
    pub fn maxima(&self, min: f64, max: f64, limit: usize) -> (Vec<V3>, Vec<f32>, usize) {
        let n = self.n;
        let nz = n / 2 + 1;
        let step = 1. / (n as f64 * self.step);
        let freq = |i: usize| {
            if i < n / 2 {
                i as f64 * step
            } else {
                (i as f64 - n as f64) * step
            }
        };
        let mut scores = self.values.clone();
        self.pool.install(|| {
            scores.par_iter_mut().enumerate().for_each(|(i, s)| {
                let x = i / (n * nz);
                let y = (i / nz) % n;
                let z = i % nz;
                let length = freq(x).powi(2) + freq(y).powi(2) + (z as f64 * step).powi(2);
                if length < min * min || length > max * max {
                    *s = f32::NEG_INFINITY;
                }
            });
        });
        let chunks: Vec<Vec<usize>> = self.pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|x| {
                    let mut local = Vec::new();
                    for y in 0..n {
                        for z in 0..nz {
                            let i = (x * n + y) * nz + z;
                            let s = scores[i];
                            if !s.is_finite() {
                                continue;
                            }
                            let mut keep = true;
                            'neighbors: for dx in -1isize..=1 {
                                for dy in -1isize..=1 {
                                    for dz in -1isize..=1 {
                                        let xx = (x as isize + dx).rem_euclid(n as isize) as usize;
                                        let yy = (y as isize + dy).rem_euclid(n as isize) as usize;
                                        let zz =
                                            (z as isize + dz).clamp(0, nz as isize - 1) as usize;
                                        if scores[(xx * n + yy) * nz + zz] > s {
                                            keep = false;
                                            break 'neighbors;
                                        }
                                    }
                                }
                            }
                            if keep {
                                local.push(i);
                            }
                        }
                    }
                    local
                })
                .collect()
        });
        let mut ids: Vec<usize> = chunks.into_iter().flatten().collect();
        let total = ids.len();
        ids.sort_unstable_by(|&i, &j| scores[j].total_cmp(&scores[i]).then(i.cmp(&j)));
        ids.truncate(limit);
        let vectors = ids
            .iter()
            .map(|&i| {
                [
                    freq(i / (n * nz)),
                    freq((i / nz) % n),
                    (i % nz) as f64 * step,
                ]
            })
            .collect();
        let values = ids.iter().map(|&i| scores[i]).collect();
        (vectors, values, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding_and_origin() {
        let e = Engine::new(4).unwrap();
        let a = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let q = [[0.5, 1.5, -2.5], [1., 0., 0.], [0.; 3]];
        assert_eq!(e.score(&q, a, a)[0].0, [0., 2., -2.]);
        let (labels, amb, multiple) = e.assign(&q, &[(a, a), (a, a)], 0.01, 0.8);
        assert_eq!(labels, vec![0, 0, 0]);
        assert_eq!((amb, multiple), (1, 1));
    }
    #[test]
    fn dc_and_sign() {
        let e = Engine::new(2).unwrap();
        let q = [[0.; 3], [0.1, 0.2, -0.1]];
        let f = e.field(&q, 32, 0.05).unwrap();
        assert!((f.values[0] - 2.).abs() < 1e-6);
    }
    #[test]
    fn nonorthogonal_inverse() {
        let a = [[5., 1., 0.], [0., 6., 1.], [1., 0., 7.]];
        let recip = inverse_transpose(a).unwrap();
        let p = std::array::from_fn(|j| 2. * recip[0][j] - recip[1][j] + recip[2][j]);
        let (h, d) = indexed(p, a, recip);
        assert_eq!(h, [2., -1., 1.]);
        assert!(d < 1e-14);
    }
}
