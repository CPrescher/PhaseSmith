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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Three reciprocal or direct Cartesian components.
pub type V3 = [f64; 3];
/// Three lattice basis row vectors.
pub type M3 = [V3; 3];
/// Refined position, fused moments and unique-evaluation/cache-hit counts.
pub type FourierRefinement = (V3, [f64; 13], [usize; 2]);

/// Fixed ownership tolerances in reciprocal and observation coordinates.
#[derive(Clone, Copy)]
pub struct ObservationOwnership {
    /// Strict Cartesian reciprocal cutoff, inverse angstroms.
    pub tolerance: f64,
    /// Best/second distance ambiguity ratio.
    pub ambiguity_ratio: f64,
    /// Pixel scale for dimensionless residual ranking.
    pub xy_scale: f64,
    /// Degree scale for dimensionless residual ranking.
    pub omega_scale: f64,
    /// Inclusive pixel-radius eligibility bound.
    pub xy_gate: f64,
    /// Inclusive absolute-degree eligibility bound.
    pub omega_gate: f64,
}

/// Bounded private worker pool for FFT and point ownership.
pub struct Engine {
    /// Number of workers (one through four).
    pub workers: usize,
    plans: Mutex<HashMap<usize, Arc<dyn rustfft::Fft<f32>>>>,
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

/// One selected observation in its original input order.
pub struct SelectedPoint {
    /// Original point index.
    pub index: usize,
    /// Nearest nonzero integer indices, represented as f64 as in `score`.
    pub hkl: V3,
    /// Original reciprocal point.
    pub point: V3,
    /// Original Cartesian reciprocal residual.
    pub error: f64,
}

/// Immutable ordered point snapshot for repeated least-squares selection.
pub struct FittingCloud {
    points: Vec<V3>,
    pool: Arc<rayon::ThreadPool>,
}
impl FittingCloud {
    /// Return only nonzero HKLs strictly inside the original reciprocal cutoff.
    /// All point arithmetic is the original `indexed`; no solver or gate changes.
    /// # Errors
    /// Rejects nonfinite bases or a nonpositive/nonfinite tolerance.
    pub fn selected(
        &self,
        rows: M3,
        reciprocal: M3,
        tolerance: f64,
    ) -> Result<Vec<SelectedPoint>, crate::SpotError> {
        if !tolerance.is_finite()
            || tolerance <= 0.
            || rows
                .iter()
                .chain(&reciprocal)
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err(crate::SpotError("invalid fitting selection inputs".into()));
        }
        let evaluate = |(i, &point): (usize, &V3)| {
            let (hkl, error) = indexed(point, rows, reciprocal);
            (hkl != [0.; 3] && error < tolerance).then_some(SelectedPoint {
                index: i,
                hkl,
                point,
                error,
            })
        };
        if self.points.len() <= 2048 {
            return Ok(self
                .points
                .iter()
                .enumerate()
                .filter_map(evaluate)
                .collect());
        }
        // Indexed map/collect retains original ordering before flattening.
        let chunks: Vec<Vec<_>> = self.pool.install(|| {
            self.points
                .par_chunks(2048)
                .enumerate()
                .map(|(chunk, q)| {
                    q.iter()
                        .enumerate()
                        .filter_map(|(i, p)| evaluate((chunk * 2048 + i, p)))
                        .collect()
                })
                .collect()
        });
        Ok(chunks.into_iter().flatten().collect())
    }
}

/// Owned integer-phase memberships, packed into exact 64-bit words.
pub struct SupportBits {
    /// One bit row per Fourier vector.
    pub rows: Vec<Vec<u64>>,
    /// Number of observations, excluding padding bits.
    pub count: usize,
    /// Same bounded worker pool as the search engine.
    pub pool: Arc<rayon::ThreadPool>,
}

/// Owned, invariant coefficients for repeated fused Fourier moment evaluations.
pub struct FourierCloud {
    points: Vec<V3>,
    coefficients: Vec<[f64; 12]>,
    scale: f64,
    pool: Arc<rayon::ThreadPool>,
}

impl FourierCloud {
    /// Evaluate the original fused moment equations, with ordered point sums.
    /// # Errors
    /// Rejects invalid vectors/work or nonfinite output, as `Engine::fourier`.
    pub fn evaluate(&self, vectors: &[V3]) -> Result<Vec<[f64; 13]>, crate::SpotError> {
        if vectors.iter().flatten().any(|v| !v.is_finite())
            || self.points.len().saturating_mul(vectors.len()) > 200_000_000
        {
            return Err(crate::SpotError(
                "invalid Fourier arrays or work budget exceeded".into(),
            ));
        }
        let evaluate = |&v: &V3| {
            let mut out = [0.; 13];
            for (p, coefficients) in self.points.iter().zip(&self.coefficients) {
                let phase =
                    std::f64::consts::TAU * p.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
                let (s, c) = phase.sin_cos();
                out[0] += c;
                for j in 0..3 {
                    out[1 + j] -= coefficients[j] * s * self.scale;
                }
                for j in 0..9 {
                    out[4 + j] -= coefficients[3 + j] * c * self.scale;
                }
            }
            out[0] *= self.scale;
            if out.iter().any(|x| !x.is_finite()) {
                return Err(crate::SpotError("Fourier overflow".into()));
            }
            Ok(out)
        };
        if self.points.len().saturating_mul(vectors.len()) <= 32768 {
            vectors.iter().map(evaluate).collect()
        } else {
            self.pool
                .install(|| vectors.par_iter().map(evaluate).collect())
        }
    }
}

/// Original offset/point-ordered cloud-in-cell deposition, shared by both FFT paths.
fn deposit(q: &[V3], n: usize, step: f64) -> Result<Vec<f32>, String> {
    if !(16..=256).contains(&n) || !n.is_multiple_of(2) || !step.is_finite() || step <= 0. {
        return Err("grid size must be even in 16..=256, step positive and finite".into());
    }
    if q.is_empty()
        || q.iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() / step >= n as f64 / 2. - 1.)
    {
        return Err("Nonfinite, empty or out-of-grid reciprocal coordinates".into());
    }
    let mut grid = vec![0f32; n * n * n];
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
                        std::array::from_fn(|i| if off[i] == 1 { frac[i] } else { 1. - frac[i] });
                    grid[(ix[0] * n + ix[1]) * n + ix[2]] += (w[0] * w[1] * w[2]) as f32;
                }
            }
        }
    }
    Ok(grid)
}

/// Dense indexing of occupied XY columns only. Offset/point float32 additions
/// remain identical to `deposit`; absent columns are exactly positive zero.
struct Columns {
    index: Vec<usize>,
    values: Vec<f32>,
    active_x: Vec<bool>,
}

fn deposit_columns(q: &[V3], n: usize, step: f64) -> Result<Columns, String> {
    if !(16..=256).contains(&n) || !n.is_multiple_of(2) || !step.is_finite() || step <= 0. {
        return Err("grid size must be even in 16..=256, step positive and finite".into());
    }
    if q.is_empty()
        || q.iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() / step >= n as f64 / 2. - 1.)
    {
        return Err("Nonfinite, empty or out-of-grid reciprocal coordinates".into());
    }
    // Coordinate arithmetic is invariant across the eight deposition offsets.
    let cells: Vec<([usize; 3], V3)> = q
        .iter()
        .map(|p| {
            let x = p.map(|v| v / step);
            let low = x.map(f64::floor);
            let frac = std::array::from_fn(|i| x[i] - low[i]);
            (low.map(|v| (v as i64).rem_euclid(n as i64) as usize), frac)
        })
        .collect();
    let mut result = Columns {
        index: vec![usize::MAX; n * n],
        values: Vec::new(),
        active_x: vec![false; n],
    };
    let mut count = 0;
    // Assign each touched XY column once, then allocate one contiguous block.
    for dx in 0..2 {
        for dy in 0..2 {
            for &(low, _) in &cells {
                let x = (low[0] + dx) % n;
                let y = (low[1] + dy) % n;
                let slot = &mut result.index[x * n + y];
                if *slot == usize::MAX {
                    *slot = count;
                    count += 1;
                    result.active_x[x] = true;
                }
            }
        }
    }
    result.values = vec![0.; count * n];
    for dx in 0..2 {
        for dy in 0..2 {
            for dz in 0..2 {
                let off = [dx, dy, dz];
                for &(low, frac) in &cells {
                    let ix: [usize; 3] = std::array::from_fn(|i| (low[i] + off[i]) % n);
                    let w: V3 =
                        std::array::from_fn(|i| if off[i] == 1 { frac[i] } else { 1. - frac[i] });
                    let slot = result.index[ix[0] * n + ix[1]];
                    result.values[slot * n + ix[2]] += (w[0] * w[1] * w[2]) as f32;
                }
            }
        }
    }
    Ok(result)
}

impl SupportBits {
    /// Unique lexicographic triplets generated by the same strongest pair prefix.
    /// Pair ties use their original (i,j) order; no geometry is screened here.
    #[must_use]
    pub fn candidate_keys(&self, max_pairs: usize) -> Vec<[usize; 3]> {
        let counts = self.pairs();
        let mut pairs = Vec::new();
        for (i, row) in counts.iter().enumerate() {
            for (j, &count) in row.iter().enumerate().skip(i + 1) {
                pairs.push((count, i, j));
            }
        }
        pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        pairs.truncate(max_pairs);
        let mut keys = Vec::with_capacity(pairs.len() * self.rows.len());
        for (_, i, j) in pairs {
            for k in 0..self.rows.len() {
                if k == i || k == j {
                    continue;
                }
                let mut key = [i, j, k];
                key.sort_unstable();
                keys.push(key);
            }
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Exact support counts and their count/volume/key ranking permutation.
    /// Geometry volumes come from the caller's unchanged array conventions.
    /// # Panics
    /// Panics for mismatched lengths or invalid row indices.
    #[must_use]
    pub fn rank_triplets(&self, indices: &[[usize; 3]], volumes: &[f64]) -> (Vec<usize>, Vec<u32>) {
        assert_eq!(indices.len(), volumes.len());
        let counts = self.triples(indices);
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_unstable_by(|&i, &j| {
            counts[j]
                .cmp(&counts[i])
                .then(volumes[i].total_cmp(&volumes[j]))
                .then(indices[i].cmp(&indices[j]))
                .then(i.cmp(&j))
        });
        (order, counts)
    }
    /// Exact intersection sizes, in stable row-major order.
    #[must_use]
    pub fn pairs(&self) -> Vec<Vec<u32>> {
        self.pool.install(|| {
            self.rows
                .par_iter()
                .map(|a| {
                    self.rows
                        .iter()
                        .map(|b| a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum())
                        .collect()
                })
                .collect()
        })
    }
    /// Exact triple intersections; caller validates indices.
    /// # Panics
    /// Panics if any index is outside the packed row array.
    #[must_use]
    pub fn triples(&self, indices: &[[usize; 3]]) -> Vec<u32> {
        self.pool.install(|| {
            indices
                .par_iter()
                .map(|&[i, j, k]| {
                    self.rows[i]
                        .iter()
                        .zip(&self.rows[j])
                        .zip(&self.rows[k])
                        .map(|((a, b), c)| (a & b & c).count_ones())
                        .sum()
                })
                .collect()
        })
    }
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

/// One model's contiguous positional evidence and eligibility fields.
pub struct PhysicalField {
    /// Gaussian-plus-uniform evidence on physically eligible points.
    pub evidence: Vec<f64>,
    /// Nonzero HKLs strictly inside one quarter of the reciprocal cutoff.
    pub core: Vec<bool>,
    /// Strict reciprocal and inclusive detector-coordinate eligibility.
    pub physical: Vec<bool>,
}

impl Engine {
    /// Bounded per-vector Newton ascent, retaining all fused Fourier moments.
    /// # Errors
    /// Rejects invalid arrays, radius, work budgets and nonfinite arithmetic.
    pub fn refine_fourier(
        &self,
        points: &[V3],
        initial: &[V3],
        radius: f64,
    ) -> Result<Vec<FourierRefinement>, crate::SpotError> {
        crate::fourier_refinement::refine(&self.pool, points, initial, radius)
    }
    /// Snapshot at most 100,000 finite points for repeated fitting selection.
    /// # Errors
    /// Rejects nonfinite points or an excessive point count; empty is allowed.
    pub fn prepare_fitting(&self, points: &[V3]) -> Result<FittingCloud, crate::SpotError> {
        if points.len() > 100_000 || points.iter().flatten().any(|x| !x.is_finite()) {
            return Err(crate::SpotError("invalid fitting point cloud".into()));
        }
        Ok(FittingCloud {
            points: points.to_vec(),
            pool: self.pool.clone(),
        })
    }
    /// First-cycle nonzero-HKL support, without allocating per-point outputs.
    /// Each point uses exactly the original indexed residual and strict cutoff.
    #[must_use]
    pub fn initial_support(&self, q: &[V3], models: &[(M3, M3)], tolerance: f64) -> Vec<u32> {
        self.pool.install(|| {
            models
                .par_iter()
                .map(|&(a, b)| {
                    q.iter()
                        .filter(|&&p| {
                            let (h, error) = indexed(p, a, b);
                            h != [0.; 3] && error < tolerance
                        })
                        .count() as u32
                })
                .collect()
        })
    }
    /// Snapshot one point cloud and reuse its unchanged derivative prefactors.
    /// # Errors
    /// Rejects the same invalid point arrays as `fourier`.
    pub fn prepare_fourier(&self, points: &[V3]) -> Result<FourierCloud, crate::SpotError> {
        crate::reciprocal::check_fourier(points, &[])?;
        let tau = std::f64::consts::TAU;
        let coefficients = points
            .iter()
            .map(|q| {
                std::array::from_fn(|j| {
                    if j < 3 {
                        tau * q[j]
                    } else {
                        let k = j - 3;
                        tau * tau * q[k / 3] * q[k % 3]
                    }
                })
            })
            .collect();
        Ok(FourierCloud {
            points: points.to_vec(),
            coefficients,
            scale: 1. / points.len() as f64,
            pool: self.pool.clone(),
        })
    }
    /// Fused fields for observation-consistent training proposals.
    /// Arrays and normalisation measures are validated by the Python boundary.
    /// Each model keeps ordered point arithmetic; models run independently.
    /// # Errors
    /// Rejects a nonpositive/nonfinite Gaussian normalisation at an eligible point.
    /// # Panics
    /// Panics if chart, enabled or measure counts differ from the point count.
    #[allow(clippy::too_many_arguments)]
    pub fn physical_fields(
        &self,
        q: &[V3],
        models: &[(M3, M3, f64)],
        charts: &[M3],
        enabled: &[bool],
        measure: &[f64],
        parameters: [f64; 5],
    ) -> Result<Vec<PhysicalField>, crate::SpotError> {
        assert_eq!(q.len(), charts.len());
        assert_eq!(q.len(), enabled.len());
        assert_eq!(q.len(), measure.len());
        let [tolerance, xy_scale, omega_scale, xy_gate, omega_gate] = parameters;
        self.pool.install(|| {
            models
                .par_iter()
                .map(|&(rows, reciprocal, volume)| {
                    let mut out = PhysicalField {
                        evidence: vec![0.; q.len()],
                        core: vec![false; q.len()],
                        physical: vec![false; q.len()],
                    };
                    for (i, &p) in q.iter().enumerate() {
                        let h = rows.map(|r| dot(p, r).round_ties_even());
                        if h == [0.; 3] {
                            continue;
                        }
                        let delta: V3 = std::array::from_fn(|j| {
                            p[j] - (h[0] * reciprocal[0][j]
                                + h[1] * reciprocal[1][j]
                                + h[2] * reciprocal[2][j])
                        });
                        let error = dot(delta, delta).sqrt();
                        out.core[i] = error < tolerance / 4.;
                        if !enabled[i] || !error.is_finite() || error >= tolerance {
                            continue;
                        }
                        let residual = charts[i].map(|r| dot(r, delta));
                        if residual.iter().any(|x| !x.is_finite())
                            || (residual[0] * residual[0] + residual[1] * residual[1]).sqrt()
                                > xy_gate
                            || residual[2].abs() > omega_gate
                        {
                            continue;
                        }
                        let normalisation = volume * measure[i];
                        if !normalisation.is_finite() || normalisation <= 0. {
                            return Err(crate::SpotError(
                                "physical evidence normalisation overflow".into(),
                            ));
                        }
                        let d2 = (residual[0] / xy_scale).powi(2)
                            + (residual[1] / xy_scale).powi(2)
                            + (residual[2] / omega_scale).powi(2);
                        let x = -0.5 * d2 - normalisation.ln();
                        out.evidence[i] = if x > 0. {
                            x + (-x).exp().ln_1p()
                        } else {
                            x.exp().ln_1p()
                        };
                        out.physical[i] = true;
                    }
                    Ok(out)
                })
                .collect()
        })
    }

    /// Three reciprocal cutoffs share one nearest-HKL/residual evaluation.
    /// Each output triple is count, summed squared q error, physical count.
    /// Caller validates array lengths and positive finite cutoffs/gates.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn candidate_counts(
        &self,
        q: &[V3],
        models: &[(M3, M3)],
        charts: &[M3],
        enabled: &[bool],
        tolerances: V3,
        xy_gate: f64,
        omega_gate: f64,
    ) -> Vec<[f64; 9]> {
        self.pool.install(|| {
            models
                .par_iter()
                .map(|&(rows, reciprocal)| {
                    let mut out = [0.; 9];
                    for ((&p, chart), &valid) in q.iter().zip(charts).zip(enabled) {
                        let h = rows.map(|r| dot(p, r).round_ties_even());
                        if h == [0.; 3] {
                            continue;
                        }
                        let delta: V3 = std::array::from_fn(|j| {
                            p[j] - (h[0] * reciprocal[0][j]
                                + h[1] * reciprocal[1][j]
                                + h[2] * reciprocal[2][j])
                        });
                        let error = dot(delta, delta).sqrt();
                        if error >= tolerances[2] {
                            continue;
                        }
                        let residual = chart.map(|r| dot(r, delta));
                        let physical = valid
                            && residual[0].hypot(residual[1]) <= xy_gate
                            && residual[2].abs() <= omega_gate;
                        for k in 0..3 {
                            if error < tolerances[k] {
                                out[3 * k] += 1.;
                                out[3 * k + 1] += error * error;
                                if physical {
                                    out[3 * k + 2] += 1.;
                                }
                            }
                        }
                    }
                    out
                })
                .collect()
        })
    }

    /// Parallel vectors, with unchanged ordered point accumulation within each.
    /// Values, gradients and Hessians remain fused in a single point pass.
    /// # Errors
    /// Rejects the same arrays, work budget and overflow as the serial oracle.
    pub fn fourier(
        &self,
        points: &[V3],
        vectors: &[V3],
    ) -> Result<Vec<[f64; 13]>, crate::SpotError> {
        crate::reciprocal::check_fourier(points, vectors)?;
        if vectors.len() >= 64 && points.len().saturating_mul(vectors.len()) > 32768 {
            return self.fourier_tiled(points, vectors);
        }
        if points.len().saturating_mul(vectors.len()) <= 32768 {
            return vectors
                .iter()
                .map(|&v| crate::reciprocal::fourier_one(points, v))
                .collect();
        }
        self.pool.install(|| {
            vectors
                .par_iter()
                .map(|&vector| crate::reciprocal::fourier_one(points, vector))
                .collect()
        })
    }
    /// Two-lane fused moments with the original per-vector arithmetic order.
    /// # Errors
    /// Uses the same validation and work/overflow boundaries as `fourier`.
    pub fn fourier_tiled(
        &self,
        points: &[V3],
        vectors: &[V3],
    ) -> Result<Vec<[f64; 13]>, crate::SpotError> {
        crate::reciprocal::check_fourier(points, vectors)?;
        let chunks: Result<Vec<Vec<[f64; 13]>>, _> = self.pool.install(|| {
            vectors
                .par_chunks(2)
                .map(|v| {
                    if v.len() == 2 {
                        Ok(crate::reciprocal::fourier_two(points, [v[0], v[1]])?.to_vec())
                    } else {
                        Ok(vec![crate::reciprocal::fourier_one(points, v[0])?])
                    }
                })
                .collect()
        });
        Ok(chunks?.into_iter().flatten().collect())
    }

    /// Fused robust observation-coordinate cost, gradient and 9x9 GLS normal.
    /// HKLs, mask and local charts are fixed during an inner basis fit.
    /// Serial accumulation deliberately preserves worker-independent sums.
    /// # Panics
    /// Panics when the per-observation arrays have inconsistent lengths.
    #[allow(clippy::too_many_arguments)] // Fixed arrays mirror the Python validation boundary.
    #[must_use]
    pub fn observation_normal(
        &self,
        q: &[V3],
        h: &[V3],
        charts: &[M3],
        enabled: &[bool],
        reciprocal: M3,
        scales: V3,
        huber: f64,
        background: Option<&[f64]>,
    ) -> ([f64; 81], [f64; 9], f64, usize) {
        assert_eq!(q.len(), h.len());
        assert_eq!(q.len(), charts.len());
        assert_eq!(q.len(), enabled.len());
        if let Some(noise) = background {
            assert_eq!(q.len(), noise.len());
        }
        let mut normal = [0.; 81];
        let mut gradient = [0.; 9];
        let mut cost = 0.;
        let mut count = 0;
        for (row, (((&p, &indices), chart), &valid)) in
            q.iter().zip(h).zip(charts).zip(enabled).enumerate()
        {
            if !valid {
                continue;
            }
            let residual: V3 = std::array::from_fn(|j| {
                p[j] - (indices[0] * reciprocal[0][j]
                    + indices[1] * reciprocal[1][j]
                    + indices[2] * reciprocal[2][j])
            });
            let r: V3 = std::array::from_fn(|j| dot(chart[j], residual) / scales[j]);
            let length = dot(r, r).sqrt();
            let weight = if let Some(noise) = background {
                let likelihood = (-0.5 * length * length).exp();
                cost += ((1. - likelihood) / (noise[row] + likelihood)).ln_1p();
                likelihood / (noise[row] + likelihood)
            } else {
                cost += if length <= huber {
                    0.5 * length * length
                } else {
                    huber * (length - 0.5 * huber)
                };
                if length <= huber { 1. } else { huber / length }
            };
            for u in 0..3 {
                let jac: [f64; 9] =
                    std::array::from_fn(|k| -indices[k / 3] * chart[u][k % 3] / scales[u]);
                for a in 0..9 {
                    gradient[a] += weight * jac[a] * r[u];
                    for b in 0..9 {
                        normal[a * 9 + b] += weight * jac[a] * jac[b];
                    }
                }
            }
            count += 1;
        }
        (normal, gradient, cost, count)
    }

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
            plans: Mutex::new(HashMap::new()),
            pool: Arc::new(pool),
        })
    }

    fn fft_plan(&self, n: usize) -> Result<Arc<dyn rustfft::Fft<f32>>, String> {
        let mut plans = self.plans.lock().map_err(|e| e.to_string())?;
        Ok(plans
            .entry(n)
            .or_insert_with(|| FftPlanner::<f32>::new().plan_fft_forward(n))
            .clone())
    }

    /// Serial accumulation preserves the reference offset/point float32 order.
    /// FFT axes execute sequentially; each axis uses the same bounded pool.
    /// # Errors
    /// Rejects empty, nonfinite/out-of-grid points, invalid spacing or even grids outside 16..=256.
    #[allow(clippy::too_many_lines)] // Three explicit FFT axes preserve layout and accumulation order.
    pub fn field(&self, q: &[V3], n: usize, step: f64) -> Result<Field, String> {
        let grid = deposit(q, n, step)?;
        let nz = n / 2 + 1;
        let fft = self.fft_plan(n)?;
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

    /// Exact radius-bounded FFT maxima without constructing the unused spectrum.
    /// Z transforms retain only needed frequencies; Y transforms retain only
    /// needed Y outputs before X transforms. Each retained line executes the
    /// original length-N FFT, in the original axis order. Empty deposited Z
    /// lines reuse one actual zero-line transform, including its signed zeros.
    /// # Errors
    /// Rejects the same invalid grids/points as `field`, and invalid radius/limit.
    // Explicit indices retain the audited original voxel/neighbor traversal.
    #[allow(clippy::too_many_lines, clippy::needless_range_loop)]
    pub fn windowed_maxima(
        &self,
        q: &[V3],
        n: usize,
        step: f64,
        min: f64,
        max: f64,
        limit: usize,
    ) -> Result<(Vec<V3>, Vec<f32>, usize), String> {
        if !min.is_finite()
            || !max.is_finite()
            || min <= 0.
            || max < min
            || limit == 0
            || limit > 1000
        {
            return Err("invalid maxima bounds or limit".into());
        }
        let columns = deposit_columns(q, n, step)?;
        let ds = 1. / (n as f64 * step);
        if !ds.is_finite() {
            return Err("nonfinite direct-grid spacing".into());
        }
        let freq = |i: usize| {
            if i < n / 2 {
                i as f64 * ds
            } else {
                (i as f64 - n as f64) * ds
            }
        };
        let axes: Vec<usize> = (0..n).filter(|&i| freq(i).powi(2) <= max * max).collect();
        let nz = n / 2 + 1;
        let zlen = (0..nz)
            .take_while(|&i| (i as f64 * ds).powi(2) <= max * max)
            .count();
        let width = axes.len();
        let mut index = vec![usize::MAX; n];
        for (i, &v) in axes.iter().enumerate() {
            index[v] = i;
        }
        let fft = self.fft_plan(n)?;
        let scratch_len = fft.get_inplace_scratch_len();
        let mut zero = vec![Complex32::default(); n];
        fft.process_with_scratch(&mut zero, &mut vec![Complex32::default(); scratch_len]);
        let mut data = vec![Complex32::default(); (columns.values.len() / n) * zlen];
        self.pool.install(|| {
            data.par_chunks_mut(zlen)
                .zip(columns.values.par_chunks(n))
                .for_each_init(
                    || {
                        (
                            vec![Complex32::default(); n],
                            vec![Complex32::default(); scratch_len],
                        )
                    },
                    |(line, scratch), (out, input)| {
                        if input.iter().all(|&v| v == 0.) {
                            out.copy_from_slice(&zero[..zlen]);
                            return;
                        }
                        for (z, &v) in line.iter_mut().zip(input) {
                            *z = Complex32::new(v, 0.);
                        }
                        fft.process_with_scratch(line, scratch);
                        out.copy_from_slice(&line[..zlen]);
                    },
                );
        });
        // Keep the small XY map; deposited columns are no longer needed.
        drop(columns.values);
        // Actual all-empty Y transforms preserve signed-zero conventions.
        let mut empty_y = vec![Complex32::default(); zlen * width];
        let mut line = vec![Complex32::default(); n];
        let mut scratch = vec![Complex32::default(); scratch_len];
        for z in 0..zlen {
            line.fill(zero[z]);
            fft.process_with_scratch(&mut line, &mut scratch);
            for (i, &y) in axes.iter().enumerate() {
                empty_y[z * width + i] = line[y];
            }
        }
        // X,Z,Y; full Y FFT with only its radius-relevant outputs stored.
        let mut other = vec![Complex32::default(); n * zlen * width];
        self.pool.install(|| {
            other.par_chunks_mut(width).enumerate().for_each_init(
                || {
                    (
                        vec![Complex32::default(); n],
                        vec![Complex32::default(); scratch_len],
                    )
                },
                |(line, scratch), (row, out)| {
                    let x = row / zlen;
                    let z = row % zlen;
                    if !columns.active_x[x] {
                        out.copy_from_slice(&empty_y[z * width..(z + 1) * width]);
                        return;
                    }
                    for (y, v) in line.iter_mut().enumerate() {
                        let slot = columns.index[x * n + y];
                        *v = if slot == usize::MAX {
                            zero[z]
                        } else {
                            data[slot * zlen + z]
                        };
                    }
                    fft.process_with_scratch(line, scratch);
                    for (i, &y) in axes.iter().enumerate() {
                        out[i] = line[y];
                    }
                },
            );
        });
        drop(data);
        // Selected Y,Z,X; original full X FFT, radius-relevant outputs only.
        let mut last = vec![Complex32::default(); width * zlen * width];
        self.pool.install(|| {
            last.par_chunks_mut(width).enumerate().for_each_init(
                || {
                    (
                        vec![Complex32::default(); n],
                        vec![Complex32::default(); scratch_len],
                    )
                },
                |(line, scratch), (row, out)| {
                    let y = row / zlen;
                    let z = row % zlen;
                    // This entire X line lies outside the spherical window.
                    // No value on it participates in maxima comparisons.
                    if freq(axes[y]).powi(2) + (z as f64 * ds).powi(2) > max * max {
                        return;
                    }
                    for (x, v) in line.iter_mut().enumerate() {
                        *v = other[(x * zlen + z) * width + y];
                    }
                    fft.process_with_scratch(line, scratch);
                    for (i, &x) in axes.iter().enumerate() {
                        out[i] = line[x];
                    }
                },
            );
        });
        drop(other);
        let mut scores = vec![f32::NEG_INFINITY; width * width * zlen];
        self.pool.install(|| {
            scores
                .par_chunks_mut(zlen)
                .enumerate()
                .for_each(|(row, out)| {
                    let x = row / width;
                    let y = row % width;
                    for (z, value) in out.iter_mut().enumerate() {
                        let length =
                            freq(axes[x]).powi(2) + freq(axes[y]).powi(2) + (z as f64 * ds).powi(2);
                        if length >= min * min && length <= max * max {
                            *value = last[(y * zlen + z) * width + x].re;
                        }
                    }
                });
        });
        drop(last);
        // Neighborhood topology is invariant across all voxel comparisons.
        let adjacent: Vec<[usize; 3]> = axes
            .iter()
            .map(|&v| {
                std::array::from_fn(|k| {
                    index[(v as isize + k as isize - 1).rem_euclid(n as isize) as usize]
                })
            })
            .collect();
        let adjacent_z: Vec<[usize; 3]> = (0..zlen)
            .map(|z| {
                std::array::from_fn(|k| {
                    (z as isize + k as isize - 1).clamp(0, nz as isize - 1) as usize
                })
            })
            .collect();
        let chunks: Vec<Vec<usize>> = self.pool.install(|| {
            (0..width)
                .into_par_iter()
                .map(|x| {
                    let mut local = Vec::new();
                    for y in 0..width {
                        for z in 0..zlen {
                            let i = (x * width + y) * zlen + z;
                            let s = scores[i];
                            if !s.is_finite() {
                                continue;
                            }
                            let mut keep = true;
                            'neighbors: for &xx in &adjacent[x] {
                                if xx == usize::MAX {
                                    continue;
                                }
                                for &yy in &adjacent[y] {
                                    if yy == usize::MAX {
                                        continue;
                                    }
                                    let base = (xx * width + yy) * zlen;
                                    for &zz in &adjacent_z[z] {
                                        if zz < zlen && scores[base + zz] > s {
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
        let compare = |&i: &usize, &j: &usize| scores[j].total_cmp(&scores[i]).then(i.cmp(&j));
        if ids.len() > limit {
            ids.select_nth_unstable_by(limit, compare);
        }
        ids.truncate(limit);
        ids.sort_unstable_by(compare);
        let vectors = ids
            .iter()
            .map(|&i| {
                [
                    freq(axes[i / (width * zlen)]),
                    freq(axes[(i / zlen) % width]),
                    (i % zlen) as f64 * ds,
                ]
            })
            .collect();
        let values = ids.iter().map(|&i| scores[i]).collect();
        Ok((vectors, values, total))
    }

    /// Nearest integer coordinates and Cartesian residual for each point.
    #[must_use]
    pub fn score(&self, q: &[V3], rows: M3, reciprocal: M3) -> Vec<(V3, f64)> {
        // Conditional subsets are small: scheduling point tasks costs more
        // than their independent dot products. Each point's arithmetic and
        // output position are unchanged on either path.
        if q.len() <= 2048 {
            q.iter().map(|&p| indexed(p, rows, reciprocal)).collect()
        } else {
            self.pool.install(|| {
                q.par_iter()
                    .map(|&p| indexed(p, rows, reciprocal))
                    .collect()
            })
        }
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

    /// Observation-consistent ownership with a per-point inverse geometry Jacobian.
    /// Charts map reciprocal residual columns to pixel/pixel/degree residuals.
    /// Stable ties, nonzero HKLs and the strict reciprocal cutoff match `assign`.
    /// Both physical gates are inclusive. No point-by-model matrix is allocated.
    /// # Panics
    /// Panics when the chart or enabled arrays differ from the point count.
    #[must_use]
    pub fn assign_observed(
        &self,
        q: &[V3],
        models: &[(M3, M3)],
        charts: &[M3],
        enabled: &[bool],
        options: ObservationOwnership,
    ) -> (Vec<i32>, usize, usize) {
        assert_eq!(q.len(), charts.len());
        assert_eq!(q.len(), enabled.len());
        let records: Vec<(i32, bool, bool)> = self.pool.install(|| {
            q.par_iter()
                .zip(charts)
                .zip(enabled)
                .map(|((&p, chart), &valid)| {
                    if !valid {
                        return (0, false, false);
                    }
                    let (mut best, mut second, mut winner, mut count) =
                        (f64::INFINITY, f64::INFINITY, 0, 0);
                    for (i, &(rows, reciprocal)) in models.iter().enumerate() {
                        let h = rows.map(|r| dot(p, r).round_ties_even());
                        if h == [0.; 3] {
                            continue;
                        }
                        let residual: V3 = std::array::from_fn(|j| {
                            p[j] - (h[0] * reciprocal[0][j]
                                + h[1] * reciprocal[1][j]
                                + h[2] * reciprocal[2][j])
                        });
                        if dot(residual, residual).sqrt() >= options.tolerance {
                            continue;
                        }
                        let observed = chart.map(|r| dot(r, residual));
                        if observed[0].hypot(observed[1]) > options.xy_gate
                            || observed[2].abs() > options.omega_gate
                        {
                            continue;
                        }
                        let distance = ((observed[0] / options.xy_scale).powi(2)
                            + (observed[1] / options.xy_scale).powi(2)
                            + (observed[2] / options.omega_scale).powi(2))
                        .sqrt();
                        if !distance.is_finite() {
                            continue;
                        }
                        count += 1;
                        if distance < best {
                            second = best;
                            best = distance;
                            winner = i as i32 + 1;
                        } else if distance < second {
                            second = distance;
                        }
                    }
                    let inside = best.is_finite();
                    let ambiguous = inside
                        && count > 1
                        && (best > options.ambiguity_ratio * second
                            || (best == 0. && second == 0.));
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
        let compare = |&i: &usize, &j: &usize| scores[j].total_cmp(&scores[i]).then(i.cmp(&j));
        // The score/index order is total, including tied amplitudes. Selecting
        // its prefix before sorting returns exactly the original top maxima.
        if limit > 0 && ids.len() > limit {
            ids.select_nth_unstable_by(limit, compare);
        }
        ids.truncate(limit);
        ids.sort_unstable_by(compare);
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
    fn occupied_columns_match_every_dense_deposition_bit() {
        for n in [16, 30, 64] {
            let step = 2.31 / (n - 4) as f64;
            let mut q: Vec<V3> = (0..2001)
                .map(|i| {
                    [
                        (f64::from(i) * 1.27).sin(),
                        (f64::from(i) * 0.71).cos(),
                        (f64::from(i) * 2.03).sin(),
                    ]
                })
                .collect();
            q.extend([[0., -0., 0.], [step, -step, 2. * step]]);
            let dense = deposit(&q, n, step).unwrap();
            let columns = deposit_columns(&q, n, step).unwrap();
            for xy in 0..n * n {
                let slot = columns.index[xy];
                for z in 0..n {
                    let v = if slot == usize::MAX {
                        0.
                    } else {
                        columns.values[slot * n + z]
                    };
                    assert_eq!(v.to_bits(), dense[xy * n + z].to_bits());
                }
            }
        }
    }

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
