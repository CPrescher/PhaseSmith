//! Independent arPLS along uniformly spaced omega samples, with natural boundaries.
//!
//! Baek et al., Analyst 140 (2015), 250–257, doi:10.1039/C4AN01061B.
//! Application inspired by Konstantin Glazyrin's background-sc; no upstream code reused.
//! File I/O, mask estimation, angular provenance and export belong to the caller.
#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
use rayon::prelude::*;

/// Explicit numerical safeguards and stopping policy.
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    /// Second-difference penalty multiplier, 1e-6 through 1e9.
    pub smoothness: f64,
    /// Relative L2 weight-change tolerance, strictly between zero and one.
    pub tolerance: f64,
    /// Maximum number of banded solves, one through 1000.
    pub max_iterations: i32,
}
impl Parameters {
    fn validate(self) -> Result<(), String> {
        if !self.smoothness.is_finite()
            || !(1e-6..=1e9).contains(&self.smoothness)
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.
            || self.tolerance >= 1.
            || !(1..=1000).contains(&self.max_iterations)
        {
            return Err("Invalid arPLS parameters".into());
        }
        Ok(())
    }
}
/// Pixel-major baseline and per-pixel diagnostics.
pub struct ResultArrays {
    /// Float64 baseline; excluded/short intervals are NaN.
    pub baseline: Vec<f64>,
    /// Maximum solves over all fitted intervals of each pixel.
    pub iterations: Vec<i32>,
    /// True only if at least one interval was fitted and all fitted runs converged.
    pub converged: Vec<bool>,
}
/// Factorization of the same positive-definite, natural-boundary system.
#[derive(Clone, Copy, Debug)]
pub enum Solver {
    /// Original banded Cholesky arithmetic, retained as the library default.
    Cholesky,
    /// Unit-lower `LDLᵀ` with reciprocal diagonal; different rounding, fewer divisions.
    Ldlt,
}
/// Reusable bounded pool; no process-global scheduler or nested pixel pools.
pub struct Engine {
    pool: Option<rayon::ThreadPool>,
    solver: Solver,
}
impl Engine {
    /// Create a serial engine or a private pool of at most 128 workers.
    /// # Errors
    /// Invalid count or failure to create the worker pool.
    pub fn new(workers: usize) -> Result<Self, String> {
        Self::with_solver(workers, Solver::Cholesky)
    }
    /// Create a bounded engine with an explicit linear solver.
    /// # Errors
    /// Invalid count or failure to create the worker pool.
    pub fn with_solver(workers: usize, solver: Solver) -> Result<Self, String> {
        if !(1..=128).contains(&workers) {
            return Err("Workers must lie in [1,128]".into());
        }
        Ok(Self {
            solver,
            pool: if workers == 1 {
                None
            } else {
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .map_err(|e| e.to_string())?,
                )
            },
        })
    }
    /// Fit independent contiguous intervals, never interpolating missing samples.
    ///
    /// Inputs have row-major `(pixels, frames)` layout. An optional mask uses
    /// true for unavailable. Uniform angular spacing must be checked by the caller.
    /// # Errors
    /// Invalid dimensions/parameters, nonfinite fitted samples or loss of positive definiteness.
    pub fn baselines(
        &self,
        data: &[f64],
        frames: usize,
        excluded: Option<&[bool]>,
        p: Parameters,
        minimum_frames: usize,
    ) -> Result<ResultArrays, String> {
        p.validate()?;
        if frames < 3
            || !data.len().is_multiple_of(frames)
            || minimum_frames < 3
            || excluded.is_some_and(|m| m.len() != data.len())
        {
            return Err("Invalid pixel/frame dimensions or minimum interval".into());
        }
        let rows = data.len() / frames;
        let mut result = ResultArrays {
            baseline: vec![f64::NAN; data.len()],
            iterations: vec![0; rows],
            converged: vec![false; rows],
        };
        let solve_row: RowFit = match self.solver {
            Solver::Cholesky => fit_row::<false>,
            Solver::Ldlt => fit_row::<true>,
        };
        let fit =
            |work: &mut Vec<f64>, row: usize, out: &mut [f64], steps: &mut i32, ok: &mut bool| {
                let input = &data[row * frames..(row + 1) * frames];
                let mask = excluded.map(|m| &m[row * frames..(row + 1) * frames]);
                let mut start = 0;
                let mut all_ok = true;
                let mut fitted = false;
                while start < frames {
                    while start < frames && mask.is_some_and(|m| m[start]) {
                        start += 1;
                    }
                    let mut end = start;
                    while end < frames && !mask.is_some_and(|m| m[end]) {
                        end += 1;
                    }
                    if end - start >= minimum_frames {
                        if input[start..end].iter().any(|v| !v.is_finite()) {
                            return Err("Fitted arPLS samples must be finite".to_owned());
                        }
                        let (it, converged) =
                            solve_row(&input[start..end], &mut out[start..end], p, work)?;
                        *steps = (*steps).max(it);
                        all_ok &= converged;
                        fitted = true;
                    }
                    start = end;
                }
                *ok = fitted && all_ok;
                Ok(())
            };
        if let Some(pool) = &self.pool {
            pool.install(|| {
                result
                    .baseline
                    .par_chunks_mut(frames)
                    .zip(result.iterations.par_iter_mut())
                    .zip(result.converged.par_iter_mut())
                    .enumerate()
                    .try_for_each_init(
                        || vec![0.; 6 * frames],
                        |work, (row, ((out, steps), ok))| fit(work, row, out, steps, ok),
                    )
            })?;
        } else {
            let mut work = vec![0.; 6 * frames];
            for row in 0..rows {
                fit(
                    &mut work,
                    row,
                    &mut result.baseline[row * frames..(row + 1) * frames],
                    &mut result.iterations[row],
                    &mut result.converged[row],
                )?;
            }
        }
        Ok(result)
    }
}

// Scalar operation order follows the preserved Dioptas CPU baseline. Parallelism
// distributes pixels, never reductions within a pixel or interval.
#[allow(clippy::too_many_lines)]
fn fit_row<const LDLT: bool>(
    input: &[f64],
    output: &mut [f64],
    p: Parameters,
    work: &mut [f64],
) -> Result<(i32, bool), String> {
    let n = input.len();
    let (y, rest) = work.split_at_mut(n);
    let (weights, rest) = rest.split_at_mut(n);
    let (diag, rest) = rest.split_at_mut(n);
    let (sub1, rest) = rest.split_at_mut(n);
    let (sub2, rest) = rest.split_at_mut(n);
    let z = &mut rest[..n];
    let offset = input[0];
    let mut scale: f64 = 0.;
    for &v in input {
        scale = scale.max((v - offset).abs());
    }
    if scale == 0. {
        output.copy_from_slice(input);
        return Ok((0, true));
    }
    for i in 0..n {
        y[i] = (input[i] - offset) / scale;
        weights[i] = 1.;
    }
    let mut converged = false;
    let mut steps = 0;
    for step in 0..p.max_iterations {
        if LDLT {
            solve_ldlt(y, weights, diag, sub1, sub2, z, p.smoothness)?;
        } else {
            solve_cholesky(y, weights, diag, sub1, sub2, z, p.smoothness)?;
        }
        steps = step + 1;
        let mut count = 0;
        let (mut mean, mut variance_sum, mut max_residual): (f64, f64, f64) = (0., 0., 0.);
        for i in 0..n {
            let residual = y[i] - z[i];
            max_residual = max_residual.max(residual.abs());
            if residual < 0. {
                count += 1;
                let delta = residual - mean;
                mean += delta / f64::from(count);
                variance_sum += delta * (residual - mean);
            }
        }
        let sigma = if count > 0 {
            (variance_sum / f64::from(count)).max(0.).sqrt()
        } else {
            0.
        };
        if count < 2 || max_residual < 1e-10 {
            converged = true;
            break;
        }
        let sigma = sigma.max(1e-8);
        let (mut difference2, mut magnitude2) = (0., 0.);
        for i in 0..n {
            let argument = (2. * (y[i] - z[i] - (2. * sigma - mean)) / sigma).clamp(-40., 40.);
            let next = (1. / (1. + argument.exp())).max(1e-8);
            let difference = next - weights[i];
            difference2 += difference * difference;
            magnitude2 += weights[i] * weights[i];
            weights[i] = next;
        }
        if (difference2 / magnitude2).sqrt() < p.tolerance {
            converged = true;
            break;
        }
    }
    for i in 0..n {
        output[i] = z[i] * scale + offset;
    }
    Ok((steps, converged))
}

type RowFit = fn(&[f64], &mut [f64], Parameters, &mut [f64]) -> Result<(i32, bool), String>;

// Specialize the row's solver statically; do not dispatch indirectly at every
// nonlinear iteration or prevent the compiler from seeing equal slice lengths.
#[inline]
fn solve_cholesky(
    y: &[f64],
    weights: &[f64],
    diag: &mut [f64],
    sub1: &mut [f64],
    sub2: &mut [f64],
    z: &mut [f64],
    smoothness: f64,
) -> Result<(), String> {
    let n = y.len();
    let weights = &weights[..n];
    let diag = &mut diag[..n];
    let sub1 = &mut sub1[..n];
    let sub2 = &mut sub2[..n];
    let z = &mut z[..n];
    for i in 0..n {
        let mut penalty = 0.;
        if i < n - 2 {
            penalty += 1.;
        }
        if i > 0 && i < n - 1 {
            penalty += 4.;
        }
        if i > 1 {
            penalty += 1.;
        }
        sub2[i] = if i > 1 { smoothness / diag[i - 2] } else { 0. };
        let adjacent = if i > 0 {
            -2. * smoothness * (f64::from(i < n - 1) + f64::from(i > 1))
        } else {
            0.
        };
        sub1[i] = if i > 0 {
            (adjacent - sub2[i] * sub1[i - 1]) / diag[i - 1]
        } else {
            0.
        };
        let pivot = weights[i] + smoothness * penalty - sub1[i] * sub1[i] - sub2[i] * sub2[i];
        if pivot <= 0. || !pivot.is_finite() {
            return Err("arPLS system lost positive definiteness; reduce smoothness".into());
        }
        diag[i] = pivot.sqrt();
        let mut rhs = weights[i] * y[i];
        if i > 0 {
            rhs -= sub1[i] * z[i - 1];
        }
        if i > 1 {
            rhs -= sub2[i] * z[i - 2];
        }
        z[i] = rhs / diag[i];
    }
    for i in (0..n).rev() {
        let mut rhs = z[i];
        if i + 1 < n {
            rhs -= sub1[i + 1] * z[i + 1];
        }
        if i + 2 < n {
            rhs -= sub2[i + 2] * z[i + 2];
        }
        z[i] = rhs / diag[i];
    }
    Ok(())
}

#[inline]
fn solve_ldlt(
    y: &[f64],
    weights: &[f64],
    diag: &mut [f64],
    sub1: &mut [f64],
    sub2: &mut [f64],
    z: &mut [f64],
    smoothness: f64,
) -> Result<(), String> {
    let n = y.len();
    let weights = &weights[..n];
    let diag = &mut diag[..n];
    let sub1 = &mut sub1[..n];
    let sub2 = &mut sub2[..n];
    let z = &mut z[..n];
    for i in 0..n {
        let mut penalty = 0.;
        if i < n - 2 {
            penalty += 1.;
        }
        if i > 0 && i < n - 1 {
            penalty += 4.;
        }
        if i > 1 {
            penalty += 1.;
        }
        let adjacent = if i > 0 {
            -2. * smoothness * (f64::from(i < n - 1) + f64::from(i > 1))
        } else {
            0.
        };
        sub2[i] = if i > 1 { smoothness * diag[i - 2] } else { 0. };
        let e1 = if i > 0 {
            adjacent - if i > 1 { smoothness * sub1[i - 1] } else { 0. }
        } else {
            0.
        };
        sub1[i] = if i > 0 { e1 * diag[i - 1] } else { 0. };
        let pivot = weights[i] + smoothness * penalty - e1 * sub1[i] - smoothness * sub2[i];
        if pivot <= 0. || !pivot.is_finite() {
            return Err("arPLS system lost positive definiteness; reduce smoothness".into());
        }
        diag[i] = 1. / pivot;
        let mut rhs = weights[i] * y[i];
        if i > 0 {
            rhs -= e1 * z[i - 1];
        }
        if i > 1 {
            rhs -= smoothness * z[i - 2];
        }
        z[i] = rhs * diag[i];
    }
    for i in (0..n).rev() {
        let mut rhs = z[i];
        if i + 1 < n {
            rhs -= sub1[i + 1] * z[i + 1];
        }
        if i + 2 < n {
            rhs -= sub2[i + 2] * z[i + 2];
        }
        z[i] = rhs;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_runs_and_workers_are_exact() {
        let p = Parameters {
            smoothness: 1e4,
            tolerance: 1e-3,
            max_iterations: 30,
        };
        let mut data: Vec<_> = (0..400).map(|i| f64::from(i % 7)).collect();
        let mut mask = vec![false; 400];
        mask[12..20].fill(true);
        data[12..20].fill(f64::NAN);
        let a = Engine::new(1)
            .unwrap()
            .baselines(&data, 40, Some(&mask), p, 12)
            .unwrap();
        let b = Engine::new(4)
            .unwrap()
            .baselines(&data, 40, Some(&mask), p, 12)
            .unwrap();
        assert_eq!(a.iterations, b.iterations);
        assert_eq!(a.converged, b.converged);
        for (x, y) in a.baseline.iter().zip(b.baseline) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert!(a.baseline[12..20].iter().all(|v| v.is_nan()));
    }
}
