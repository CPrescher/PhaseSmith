//! Diagonally scaled CG for a damped Gauss–Newton system; no dense normal matrix.
use crate::{Cancellation, SolutionError, error};
use std::time::Instant;

/// Actual execution of a guarded normal-cache solve.
pub struct CachedNormalStep {
    /// Step in the original parameter units.
    pub step: Vec<f64>,
    /// CG iterations; zero for a direct solve.
    pub iterations: usize,
    /// Whether the normal cache returned the accepted linear step.
    pub used_cache: bool,
    /// Whether a cache attempt fell back to the original matrix-free solver.
    pub fell_back: bool,
}

/// Form a blocked Gram matrix and solve with the same checked CG equations.
/// A failed original-Jacobian residual check retries legacy Gram formation and
/// then the existing matrix-free solve. Unsuitable shapes use matrix-free CG.
/// # Errors
/// Rejects malformed, stopped and unconverged systems as `cached_normal_step`.
pub fn blocked_normal_step(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<CachedNormalStep, SolutionError> {
    NormalReuse {
        blocked: true,
        ..Default::default()
    }
    .step(jacobian, residual, damping, deadline, cancellation)
}

const NORMAL_CACHE_BYTES: usize = 8 * 1024 * 1024;

// Fixed eight-lane reduction allows vectorization without fast-math or FMA.
fn lane_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sums = [0.; 8];
    let mut aa = a.chunks_exact(8);
    let mut bb = b.chunks_exact(8);
    for (x, y) in aa.by_ref().zip(bb.by_ref()) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    let mut result = sums.into_iter().sum::<f64>();
    for (x, y) in aa.remainder().iter().zip(bb.remainder()) {
        result += x * y;
    }
    result
}

/// Cache `S J^T J S`, solve by CG, then check against the original J operator.
/// Keeps the original damping, iteration ceiling and residual tolerances. Uses
/// at most 8 MiB; unsuitable shapes and numerical failures use the original path.
/// # Errors
/// Rejects the same malformed inputs, stopped or failed systems as the original.
pub fn cached_normal_step(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<CachedNormalStep, SolutionError> {
    cached_normal_step_with_memory_budget(
        jacobian,
        residual,
        damping,
        deadline,
        cancellation,
        crate::DenseMemoryBudget::default(),
    )
}

pub(crate) fn cached_normal_step_with_memory_budget(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
    memory: crate::DenseMemoryBudget,
) -> Result<CachedNormalStep, SolutionError> {
    memory.check(1, jacobian.len())?;
    validate(jacobian, residual, damping)?;
    let n = jacobian.len() / residual.len();
    let eligible = cache_shape(n, residual.len());
    if eligible {
        if let Ok((step, iterations)) =
            cached_solve(jacobian, residual, damping, deadline, cancellation)
        {
            return Ok(CachedNormalStep {
                step,
                iterations,
                used_cache: true,
                fell_back: false,
            });
        }
    }
    let (step, iterations) = iterative_normal_step_with_memory_budget(
        jacobian,
        residual,
        damping,
        deadline,
        cancellation,
        memory,
    )?;
    Ok(CachedNormalStep {
        step,
        iterations,
        used_cache: false,
        fell_back: eligible,
    })
}

#[allow(clippy::too_many_lines)]
fn cached_solve(
    jacobian: &[f64],
    observed_residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<(Vec<f64>, usize), SolutionError> {
    let mut prepared =
        PreparedNormalSystem::new(jacobian, observed_residual, deadline, cancellation)?;
    let rhs = std::mem::take(&mut prepared.rhs);
    prepared.solve_with_rhs(
        jacobian,
        observed_residual,
        damping,
        deadline,
        cancellation,
        rhs,
    )
}

struct PreparedNormalSystem {
    scales: Vec<f64>,
    rhs: Vec<f64>,
    norm: f64,
    matrix: Vec<f64>,
    blocked: bool,
}
impl PreparedNormalSystem {
    fn new(
        jacobian: &[f64],
        observed_residual: &[f64],
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<Self, SolutionError> {
        Self::prepare(jacobian, observed_residual, deadline, cancellation, false)
    }
    fn prepare(
        jacobian: &[f64],
        observed_residual: &[f64],
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        blocked: bool,
    ) -> Result<Self, SolutionError> {
        let reflections = observed_residual.len();
        let parameters = jacobian.len() / reflections;
        let scales: Vec<_> = jacobian
            .chunks_exact(reflections)
            .map(|row| 1. / row.iter().map(|v| v * v).sum::<f64>().max(1e-8).sqrt())
            .collect();
        let rhs: Vec<_> = jacobian
            .chunks_exact(reflections)
            .zip(&scales)
            .map(|(row, scale)| {
                -scale
                    * row
                        .iter()
                        .zip(observed_residual)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
            })
            .collect();
        let norm: f64 = rhs.iter().map(|v| v * v).sum();
        let mut matrix = vec![
            0.;
            if norm == 0. {
                0
            } else {
                parameters * parameters
            }
        ];
        if blocked && norm != 0. {
            blocked_gram(jacobian, reflections, &mut matrix, deadline, cancellation)?;
        }
        for i in 0..if norm == 0. { 0 } else { parameters } {
            if crate::stopped(deadline, cancellation) {
                return Err(error("normal cache stopped"));
            }
            for k in 0..=i {
                let value = scales[i]
                    * scales[k]
                    * if blocked {
                        matrix[i * parameters + k]
                    } else {
                        lane_dot(
                            &jacobian[i * reflections..(i + 1) * reflections],
                            &jacobian[k * reflections..(k + 1) * reflections],
                        )
                    };
                matrix[i * parameters + k] = value;
                matrix[k * parameters + i] = value;
            }
        }
        Ok(Self {
            scales,
            rhs,
            norm,
            matrix,
            blocked,
        })
    }
    #[allow(clippy::too_many_lines)]
    fn solve(
        &self,
        jacobian: &[f64],
        observed_residual: &[f64],
        damping: f64,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<(Vec<f64>, usize), SolutionError> {
        self.solve_with_rhs(
            jacobian,
            observed_residual,
            damping,
            deadline,
            cancellation,
            self.rhs.clone(),
        )
    }
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn solve_with_rhs(
        &self,
        jacobian: &[f64],
        observed_residual: &[f64],
        damping: f64,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        rhs: Vec<f64>,
    ) -> Result<(Vec<f64>, usize), SolutionError> {
        let parameters = self.scales.len();
        let scales = &self.scales;
        let matrix = &self.matrix;
        let norm = self.norm;
        if norm == 0. {
            return Ok((vec![0.; parameters], 0));
        }
        let mut scaled_step = vec![0.; parameters];
        let mut residual = rhs;
        let mut direction = residual.clone();
        let mut rr = norm;
        let mut product = vec![0.; parameters];
        let limit = (2 * parameters).min(256);
        for iteration in 0..limit {
            if crate::stopped(deadline, cancellation) {
                return Err(error("normal cache stopped"));
            }
            for (i, row) in matrix.chunks_exact(parameters).enumerate() {
                product[i] = lane_dot(row, &direction) + damping * direction[i];
            }
            let curvature: f64 = direction.iter().zip(&product).map(|(a, b)| a * b).sum();
            if !curvature.is_finite() || curvature <= 0. {
                return Err(error("nonpositive cached curvature"));
            }
            let alpha = rr / curvature;
            for i in 0..parameters {
                scaled_step[i] += alpha * direction[i];
                residual[i] -= alpha * product[i];
            }
            let next: f64 = residual.iter().map(|v| v * v).sum();
            if next <= norm * 1e-16 || iteration + 1 == limit {
                // Reapply the original matrix-free operator, not the cached Gram matrix.
                let actual = original_residual_squared(
                    jacobian,
                    observed_residual,
                    damping,
                    scales,
                    &scaled_step,
                );
                let tolerance = if iteration + 1 == limit { 1e-10 } else { 1e-16 };
                let step: Vec<_> = scaled_step.iter().zip(scales).map(|(a, b)| a * b).collect();
                if actual <= norm * tolerance && step.iter().all(|v| v.is_finite()) {
                    return Ok((step, iteration + 1));
                }
                return Err(error("cached step fails original-operator residual check"));
            }
            for i in 0..parameters {
                direction[i] = residual[i] + next / rr * direction[i];
            }
            rr = next;
        }
        Err(error("cached solve did not converge"))
    }
}

/// A single fit's linearization cache; never shared between candidates or fits.
#[derive(Default)]
pub(crate) struct NormalReuse {
    pub(crate) memory_budget: crate::DenseMemoryBudget,
    prepared: Option<PreparedNormalSystem>,
    pub(crate) builds: usize,
    pub(crate) reuses: usize,
    pub(crate) blocked: bool,
    pub(crate) blocked_builds: usize,
    pub(crate) blocked_fallbacks: usize,
}
impl NormalReuse {
    pub(crate) fn clear(&mut self) {
        self.prepared = None;
    }
    pub(crate) fn step(
        &mut self,
        jacobian: &[f64],
        residual: &[f64],
        damping: f64,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<CachedNormalStep, SolutionError> {
        self.memory_budget.check(1, jacobian.len())?;
        validate(jacobian, residual, damping)?;
        let eligible = cache_shape(jacobian.len() / residual.len(), residual.len());
        if eligible {
            if self.prepared.is_some() {
                self.reuses += 1;
            } else if let Ok(p) = PreparedNormalSystem::prepare(
                jacobian,
                residual,
                deadline,
                cancellation,
                self.blocked,
            ) {
                self.prepared = Some(p);
                self.builds += 1;
                self.blocked_builds += usize::from(self.blocked);
            }
            if let Some(p) = &self.prepared {
                if let Ok((step, iterations)) =
                    p.solve(jacobian, residual, damping, deadline, cancellation)
                {
                    return Ok(CachedNormalStep {
                        step,
                        iterations,
                        used_cache: true,
                        fell_back: false,
                    });
                }
            }
        }
        // A changed Gram summation order must still satisfy the original J
        // operator. On failure retain the legacy Gram for this linearization.
        if self.prepared.as_ref().is_some_and(|p| p.blocked)
            && !crate::stopped(deadline, cancellation)
        {
            self.blocked_fallbacks += 1;
            self.prepared =
                PreparedNormalSystem::new(jacobian, residual, deadline, cancellation).ok();
            self.builds += usize::from(self.prepared.is_some());
            if let Some(p) = &self.prepared {
                if let Ok((step, iterations)) =
                    p.solve(jacobian, residual, damping, deadline, cancellation)
                {
                    return Ok(CachedNormalStep {
                        step,
                        iterations,
                        used_cache: true,
                        fell_back: false,
                    });
                }
            }
        }
        let (step, iterations) = iterative_normal_step_with_memory_budget(
            jacobian,
            residual,
            damping,
            deadline,
            cancellation,
            self.memory_budget,
        )?;
        Ok(CachedNormalStep {
            step,
            iterations,
            used_cache: false,
            fell_back: eligible,
        })
    }
}

// J is parameter-major. Only lower block tiles are needed; the caller scales
// and mirrors the lower triangle. Views borrow existing storage; matrixmultiply
// packs panels internally and uses its single-threaded portable SIMD backend.
fn blocked_gram(
    jacobian: &[f64],
    reflections: usize,
    matrix: &mut [f64],
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<(), SolutionError> {
    use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder, linalg::general_mat_mul};
    const TILE: usize = 32;
    let parameters = jacobian.len() / reflections;
    for i in (0..parameters).step_by(TILE) {
        let rows = TILE.min(parameters - i);
        let a = ArrayView2::from_shape(
            (rows, reflections),
            &jacobian[i * reflections..(i + rows) * reflections],
        )
        .map_err(|e| error(e.to_string()))?;
        for k in (0..=i).step_by(TILE) {
            if crate::stopped(deadline, cancellation) {
                return Err(error("blocked normal cache stopped"));
            }
            let cols = TILE.min(parameters - k);
            let b = ArrayView2::from_shape(
                (cols, reflections),
                &jacobian[k * reflections..(k + cols) * reflections],
            )
            .map_err(|e| error(e.to_string()))?;
            let start = i * parameters + k;
            let end = start + (rows - 1) * parameters + cols;
            let mut c = ArrayViewMut2::from_shape(
                (rows, cols).strides((parameters, 1)),
                &mut matrix[start..end],
            )
            .map_err(|e| error(e.to_string()))?;
            general_mat_mul(1., &a, &b.t(), 0., &mut c);
        }
    }
    Ok(())
}

fn cache_shape(parameters: usize, reflections: usize) -> bool {
    parameters >= 32
        && reflections >= 2 * parameters
        && parameters
            .checked_mul(parameters)
            .and_then(|v| v.checked_mul(size_of::<f64>()))
            .is_some_and(|v| v <= NORMAL_CACHE_BYTES)
}

/// Blocked normal formation for small final models; retain their Cholesky solve.
pub(crate) fn cached_dense_step(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
    memory: crate::DenseMemoryBudget,
) -> Result<CachedNormalStep, SolutionError> {
    memory.check(1, jacobian.len())?;
    validate(jacobian, residual, damping)?;
    let reflections = residual.len();
    let count = jacobian.len() / reflections;
    if !(12..32).contains(&count) {
        return Err(error("invalid blocked dense shape"));
    }
    let diagonal: Vec<_> = jacobian
        .chunks_exact(reflections)
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().max(1e-8))
        .collect();
    let scales: Vec<_> = diagonal.iter().map(|v| 1. / v.sqrt()).collect();
    let rhs: Vec<_> = jacobian
        .chunks_exact(reflections)
        .map(|row| -row.iter().zip(residual).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let norm: f64 = rhs.iter().zip(&scales).map(|(v, s)| (v * s).powi(2)).sum();
    for blocked in [true, false] {
        let mut normal = vec![0.; count * count];
        for i in 0..count {
            if crate::stopped(deadline, cancellation) {
                return Err(error("blocked dense solve stopped"));
            }
            let row = &jacobian[i * reflections..(i + 1) * reflections];
            for k in 0..=i {
                let other = &jacobian[k * reflections..(k + 1) * reflections];
                let value = if blocked {
                    lane_dot(row, other)
                } else {
                    row.iter().zip(other).map(|(a, b)| a * b).sum()
                };
                normal[i * count + k] = value;
                normal[k * count + i] = value;
            }
            normal[i * count + i] += damping * diagonal[i];
        }
        if let Some(step) = crate::model::cholesky_solve(&normal, &rhs) {
            if !blocked {
                return Ok(CachedNormalStep {
                    step,
                    iterations: 0,
                    used_cache: false,
                    fell_back: true,
                });
            }
            let candidate: Vec<_> = step.iter().zip(&scales).map(|(v, s)| v / s).collect();
            let actual =
                original_residual_squared(jacobian, residual, damping, &scales, &candidate);
            if actual <= norm * 1e-16 && step.iter().all(|v| v.is_finite()) {
                return Ok(CachedNormalStep {
                    step,
                    iterations: 0,
                    used_cache: true,
                    fell_back: false,
                });
            }
        }
    }
    Err(error("blocked dense solve failed"))
}

fn original_residual_squared(
    jacobian: &[f64],
    observed_residual: &[f64],
    damping: f64,
    scales: &[f64],
    scaled_step: &[f64],
) -> f64 {
    let reflections = observed_residual.len();
    let mut jp = vec![0.; reflections];
    for ((row, scale), value) in jacobian
        .chunks_exact(reflections)
        .zip(scales)
        .zip(scaled_step)
    {
        for (out, v) in jp.iter_mut().zip(row) {
            *out += v * scale * value;
        }
    }
    jacobian
        .chunks_exact(reflections)
        .zip(scales)
        .zip(scaled_step)
        .map(|((row, scale), value)| {
            let rhs = -scale
                * row
                    .iter()
                    .zip(observed_residual)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            let ax = scale * row.iter().zip(&jp).map(|(a, b)| a * b).sum::<f64>() + damping * value;
            (rhs - ax).powi(2)
        })
        .sum()
}

fn validate(jacobian: &[f64], residual: &[f64], damping: f64) -> Result<(), SolutionError> {
    let m = residual.len();
    if m == 0
        || jacobian.is_empty()
        || jacobian.len() % m != 0
        || !damping.is_finite()
        || damping <= 0.
        || residual.iter().chain(jacobian).any(|x| !x.is_finite())
    {
        return Err(error("invalid iterative normal system"));
    }
    Ok(())
}

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn explicit_linear_budget_reaches_all_operator_paths() {
        let rows = 62_501;
        let parameters = 32;
        let jacobian = vec![1.; rows * parameters];
        let residual = vec![0.; rows];
        let memory = crate::DenseMemoryBudget::new(64 * 1024 * 1024).unwrap();
        assert!(iterative_normal_step(&jacobian, &residual, 0.1, None, None).is_err());
        assert!(cached_normal_step(&jacobian, &residual, 0.1, None, None).is_err());
        assert!(blocked_normal_step(&jacobian, &residual, 0.1, None, None).is_err());
        let (step, _) =
            iterative_normal_step_with_memory_budget(&jacobian, &residual, 0.1, None, None, memory)
                .unwrap();
        assert_eq!(step, vec![0.; parameters]);
        let step =
            cached_normal_step_with_memory_budget(&jacobian, &residual, 0.1, None, None, memory)
                .unwrap();
        assert_eq!(step.step, vec![0.; parameters]);
        let mut reuse = NormalReuse {
            memory_budget: memory,
            blocked: true,
            ..Default::default()
        };
        assert_eq!(
            reuse
                .step(&jacobian, &residual, 0.1, None, None)
                .unwrap()
                .step,
            vec![0.; parameters]
        );
        // Small-model dense/Cholesky path has the same explicit buffer admission.
        let dense = vec![1.; 15 * 140_000];
        assert!(
            cached_dense_step(
                &dense,
                &vec![0.; 140_000],
                0.1,
                None,
                None,
                crate::DenseMemoryBudget::default()
            )
            .is_err()
        );
        assert_eq!(
            cached_dense_step(&dense, &vec![0.; 140_000], 0.1, None, None, memory)
                .unwrap()
                .step,
            vec![0.; 15]
        );
    }
}

/// Solve `(JᵀJ + damping*D) step = -Jᵀr`, `D=max(diag(JᵀJ),1e-8)`.
/// J is parameter-major; this retains J but never constructs `JᵀJ`.
/// Returns the step and actual CG iterations. A stop is an explicit error.
/// # Errors
/// Rejects malformed, nonfinite arrays, invalid damping, stops and failed solves.
#[allow(clippy::too_many_lines)]
pub fn iterative_normal_step(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
) -> Result<(Vec<f64>, usize), SolutionError> {
    iterative_normal_step_with_memory_budget(
        jacobian,
        residual,
        damping,
        deadline,
        cancellation,
        crate::DenseMemoryBudget::default(),
    )
}

pub(crate) fn iterative_normal_step_with_memory_budget(
    jacobian: &[f64],
    residual: &[f64],
    damping: f64,
    deadline: Option<Instant>,
    cancellation: Option<&Cancellation>,
    memory: crate::DenseMemoryBudget,
) -> Result<(Vec<f64>, usize), SolutionError> {
    memory.check(1, jacobian.len())?;
    let m = residual.len();
    validate(jacobian, residual, damping)?;
    let n = jacobian.len() / m;
    let inverse_scale: Vec<_> = jacobian
        .chunks_exact(m)
        .map(|row| 1. / row.iter().map(|v| v * v).sum::<f64>().max(1e-8).sqrt())
        .collect();
    let rhs: Vec<_> = jacobian
        .chunks_exact(m)
        .zip(&inverse_scale)
        .map(|(row, scale)| -scale * row.iter().zip(residual).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let mut x = vec![0.; n];
    let mut r = rhs.clone();
    let mut direction = r.clone();
    let norm: f64 = rhs.iter().map(|v| v * v).sum();
    if norm == 0. {
        return Ok((x, 0));
    }
    let mut rr = norm;
    let mut jp = vec![0.; m];
    for iteration in 0..(2 * n).min(256) {
        if crate::stopped(deadline, cancellation) {
            return Err(error("iterative normal solve stopped"));
        }
        jp.fill(0.);
        for ((row, scale), p) in jacobian.chunks_exact(m).zip(&inverse_scale).zip(&direction) {
            for (out, v) in jp.iter_mut().zip(row) {
                *out += v * scale * p;
            }
        }
        let product: Vec<_> = jacobian
            .chunks_exact(m)
            .zip(&inverse_scale)
            .zip(&direction)
            .map(|((row, scale), p)| {
                scale * row.iter().zip(&jp).map(|(a, b)| a * b).sum::<f64>() + damping * p
            })
            .collect();
        let curvature: f64 = direction.iter().zip(&product).map(|(a, b)| a * b).sum();
        if !curvature.is_finite() || curvature <= 0. {
            return Err(error("nonpositive iterative curvature"));
        }
        let alpha = rr / curvature;
        for i in 0..n {
            x[i] += alpha * direction[i];
            r[i] -= alpha * product[i];
        }
        let next: f64 = r.iter().map(|v| v * v).sum();
        if next <= norm * 1e-16 {
            let step: Vec<_> = x.iter().zip(&inverse_scale).map(|(a, b)| a * b).collect();
            if step.iter().any(|v| !v.is_finite()) {
                return Err(error("nonfinite iterative step"));
            }
            return Ok((step, iteration + 1));
        }
        for i in 0..n {
            direction[i] = r[i] + next / rr * direction[i];
        }
        rr = next;
    }
    // An inexact step is admissible only with a separately checked residual bound.
    if rr <= norm * 1e-10 {
        Ok((
            x.iter().zip(&inverse_scale).map(|(a, b)| a * b).collect(),
            (2 * n).min(256),
        ))
    } else {
        Err(error("iterative normal solve did not converge"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn blocked_tiles_match_legacy_with_partial_edges_and_preserve_guards() {
        for (parameters, rows) in [(32, 67), (33, 131), (69, 333)] {
            let j: Vec<_> = (0..parameters * rows)
                .map(|i| (i as f64 * 0.731).sin() + (i as f64 * 0.017).cos())
                .collect();
            let r: Vec<_> = (0..rows).map(|i| (i as f64 * 0.379).cos()).collect();
            let legacy = PreparedNormalSystem::new(&j, &r, None, None).unwrap();
            let blocked = PreparedNormalSystem::prepare(&j, &r, None, None, true).unwrap();
            assert_eq!(legacy.scales, blocked.scales);
            assert_eq!(legacy.rhs, blocked.rhs);
            for (a, b) in legacy.matrix.iter().zip(&blocked.matrix) {
                assert!((a - b).abs() <= 3e-14);
            }
            let mut cache = NormalReuse {
                blocked: true,
                ..Default::default()
            };
            cache.step(&j, &r, 0.1, None, None).unwrap();
            assert_eq!(cache.blocked_builds, 1);
            // Corrupt only the cached matrix. A failed original-operator guard
            // must rebuild the legacy matrix, never accept this changed system.
            cache.prepared.as_mut().unwrap().matrix.fill(0.);
            let result = cache.step(&j, &r, 0.1, None, None).unwrap();
            assert_eq!(cache.blocked_fallbacks, 1);
            assert!(result.used_cache);
            assert_eq!(
                result.step,
                cached_normal_step(&j, &r, 0.1, None, None).unwrap().step
            );
            assert!(!cache.prepared.as_ref().unwrap().blocked);
            cache.step(&j, &r, 1., None, None).unwrap();
            assert_eq!(cache.blocked_builds, 1);
            cache.clear();
            cache.step(&j, &r, 1., None, None).unwrap();
            assert_eq!(cache.blocked_builds, 2);
            assert!(blocked_normal_step(&j, &r, 0.1, Some(Instant::now()), None).is_err());
            let token = Cancellation::default();
            let _ = token.cancel();
            assert!(blocked_normal_step(&j, &r, 0.1, None, Some(&token)).is_err());
            let z = blocked_normal_step(&j, &vec![0.; rows], 0.1, None, None).unwrap();
            assert_eq!(z.step, vec![0.; parameters]);
            assert_eq!(z.iterations, 0);
        }
    }
    #[test]
    fn blocked_dense_step_satisfies_original_system_and_stops() {
        let rows = 197;
        let count = 15;
        let jacobian: Vec<_> = (0..rows * count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f64;
                (x * 0.123).sin() + (x * 0.331).cos()
            })
            .collect();
        let residual = vec![0.21; rows];
        let out = cached_dense_step(
            &jacobian,
            &residual,
            0.1,
            None,
            None,
            crate::DenseMemoryBudget::default(),
        )
        .unwrap();
        assert!(out.used_cache);
        let scales: Vec<_> = jacobian
            .chunks_exact(rows)
            .map(|r| 1. / r.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();
        let x: Vec<_> = out.step.iter().zip(&scales).map(|(v, s)| v / s).collect();
        assert!(original_residual_squared(&jacobian, &residual, 0.1, &scales, &x) < 1e-20);
        assert!(
            cached_dense_step(
                &jacobian,
                &residual,
                0.1,
                Some(Instant::now()),
                None,
                crate::DenseMemoryBudget::default()
            )
            .is_err()
        );
    }
    #[test]
    fn cache_limits_and_original_operator_guard() {
        assert!(cache_shape(32, 64));
        assert!(!cache_shape(31, 64));
        assert!(!cache_shape(32, 63));
        assert!(!cache_shape(1025, 2050));
        let jacobian = [1., 0., 0., 1.];
        let residual = [1., 2.];
        let scales = [1., 1.];
        assert_eq!(
            original_residual_squared(&jacobian, &residual, 1., &scales, &[-0.5, -1.]),
            0.
        );
        assert!(original_residual_squared(&jacobian, &residual, 1., &scales, &[0., 0.]) > 1.);
        assert!(
            !original_residual_squared(&jacobian, &residual, 1., &scales, &[f64::NAN, 0.])
                .is_finite()
        );
    }
    #[test]
    fn cached_systems_honor_cancellation_and_deadline() {
        let token = Cancellation::default();
        let _ = token.cancel();
        let j = vec![1.; 32 * 64];
        let r = vec![1.; 64];
        assert!(cached_normal_step(&j, &r, 0.1, None, Some(&token)).is_err());
        assert!(cached_normal_step(&j, &r, 0.1, Some(Instant::now()), None).is_err());
    }
    #[test]
    fn stopped_systems_and_invalid_shapes_fail_explicitly() {
        let token = Cancellation::default();
        let _ = token.cancel();
        assert!(
            iterative_normal_step(&[1., 0., 0., 1.], &[1., 1.], 0.001, None, Some(&token)).is_err()
        );
        assert!(
            iterative_normal_step(
                &[1., 0., 0., 1.],
                &[1., 1.],
                0.001,
                Some(Instant::now()),
                None
            )
            .is_err()
        );
        assert!(iterative_normal_step(&[1., 2., 3.], &[1., 1.], 0.001, None, None).is_err());
        assert_eq!(
            iterative_normal_step(&[1., 0., 0., 1.], &[0., 0.], 0.001, None, None)
                .unwrap()
                .0,
            vec![0.; 2]
        );
    }
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn retained_systems_preserve_bits_for_damping_changes_and_clear_on_new_data() {
        for (parameters, rows) in [(8, 100), (32, 63), (32, 131), (69, 333)] {
            let mut jacobian: Vec<_> = (0..parameters * rows)
                .map(|i| ((i as f64) * 0.731).sin())
                .collect();
            let residual: Vec<_> = (0..rows).map(|i| ((i as f64) * 0.379).cos()).collect();
            let mut cache = NormalReuse::default();
            for damping in [0.001, 0.01, 0.1, 1., 10.] {
                let a = cached_normal_step(&jacobian, &residual, damping, None, None).unwrap();
                let b = cache
                    .step(&jacobian, &residual, damping, None, None)
                    .unwrap();
                assert_eq!(a.iterations, b.iterations);
                assert_eq!(a.used_cache, b.used_cache);
                assert_eq!(a.fell_back, b.fell_back);
                assert_eq!(
                    a.step.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.step.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            let eligible = cache_shape(parameters, rows);
            assert_eq!(cache.builds, usize::from(eligible));
            assert_eq!(cache.reuses, if eligible { 4 } else { 0 });
            cache.clear();
            jacobian[0] += 0.1;
            let a = cached_normal_step(&jacobian, &residual, 0.1, None, None).unwrap();
            let b = cache.step(&jacobian, &residual, 0.1, None, None).unwrap();
            assert_eq!(a.step, b.step);
            assert_eq!(cache.builds, if eligible { 2 } else { 0 });
        }
    }
    #[test]
    fn retained_systems_honor_stops_invalid_inputs_and_zero_rhs() {
        let j = vec![1.; 32 * 131];
        let r = vec![1.; 131];
        let mut cache = NormalReuse::default();
        cache.step(&j, &r, 0.1, None, None).unwrap();
        assert!(cache.step(&j, &r, 0.1, Some(Instant::now()), None).is_err());
        let token = Cancellation::default();
        let _ = token.cancel();
        assert!(cache.step(&j, &r, 0.1, None, Some(&token)).is_err());
        assert!(cache.step(&j, &r, 0., None, None).is_err());
        assert!(cache.step(&j[..10], &r, 0.1, None, None).is_err());
        cache.clear();
        let zero = cache.step(&j, &vec![0.; 131], 0.1, None, None).unwrap();
        assert_eq!(zero.iterations, 0);
        assert_eq!(zero.step, vec![0.; 32]);
        assert_eq!(cache.prepared.as_ref().unwrap().matrix, Vec::<f64>::new());
    }
}
