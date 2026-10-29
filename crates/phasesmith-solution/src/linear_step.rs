//! Diagonally scaled CG for a damped Gauss–Newton system; no dense normal matrix.
use crate::{Cancellation, SolutionError, error};
use std::time::Instant;

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
    let m = residual.len();
    if m == 0
        || jacobian.is_empty()
        || jacobian.len() % m != 0
        || jacobian.len() > 2_000_000
        || !damping.is_finite()
        || damping <= 0.
        || residual.iter().chain(jacobian).any(|x| !x.is_finite())
    {
        return Err(error("invalid iterative normal system"));
    }
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
}
