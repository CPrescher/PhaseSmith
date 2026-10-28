//! Bounded P1 coordinate polishing using existing fused analytical derivatives.

use std::time::Instant;

use phasesmith_crystallography::{
    P1BatchView, PreparedXrayScattering, UnitCell, calculate_p1_dense,
};

use crate::{SolutionError, error};

/// Profiled amplitude objective and its parameter-major analytical Jacobian.
#[derive(Clone, Debug)]
pub struct ModelEvaluation {
    /// Sum of squared residuals divided by sum Fo squared.
    pub objective: f64,
    /// sum |k |Fc| - Fo| / sum Fo, on the supplied expanded reflection set.
    pub r1: f64,
    /// Positive amplitude scale (intensity scale is its square).
    pub amplitude_scale: f64,
    /// Normalized amplitude residuals.
    pub residual: Vec<f64>,
    /// Parameter-major derivatives: coordinates of sites 1..N, then common U.
    pub jacobian: Vec<f64>,
}

/// Last accepted coordinate-polishing state.
#[derive(Clone, Debug)]
pub struct PolishedModel {
    /// Full-cell fractional coordinates.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Common isotropic displacement parameter.
    pub u_iso: f64,
    /// Objective, residuals, and analytical Jacobian at this state.
    pub evaluation: ModelEvaluation,
    /// Number of accepted polishing steps.
    pub accepted_steps: usize,
}

/// Evaluate the amplitude objective and analytical derivatives in one native pass.
///
/// Coordinates of site zero fix the translational gauge. At exactly zero Fc the
/// amplitude derivative is assigned zero; derivative tests avoid that cusp.
///
/// # Errors
/// Rejects invalid cell/species, nonfinite amplitudes/U, incompatible shapes,
/// or a dense derivative allocation beyond two million elements per array.
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
pub fn evaluate_model(
    cell: UnitCell,
    hkl: &[[i32; 3]],
    observed: &[f64],
    species: &[String],
    xyz: &[[f64; 3]],
    u: f64,
) -> Result<ModelEvaluation, SolutionError> {
    let scattering = prepare_scattering(cell, hkl, species)?;
    evaluate_prepared(cell, hkl, observed, xyz, u, &scattering)
}

fn prepare_scattering(
    cell: UnitCell,
    hkl: &[[i32; 3]],
    species: &[String],
) -> Result<Vec<f64>, SolutionError> {
    if species.is_empty()
        || species.len() > 128
        || hkl.is_empty()
        || hkl.len() > 100_000
        || (5 * species.len() + 7) * hkl.len() > 2_000_000
    {
        return Err(error(
            "model exceeds atom/reflection/dense-derivative limits",
        ));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let s: Vec<_> = hkl
        .iter()
        .map(|&h| geometry.d_spacing_and_derivatives(h).map(|(d, _)| 0.5 / d))
        .collect::<Result<_, _>>()
        .map_err(|e| error(e.to_string()))?;
    PreparedXrayScattering::new(species)
        .and_then(|p| p.evaluate(&s))
        .map(|b| b.real)
        .map_err(|e| error(e.to_string()))
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)] // Equation-local a,k,p,r notation.
fn evaluate_prepared(
    cell: UnitCell,
    hkl: &[[i32; 3]],
    observed: &[f64],
    xyz: &[[f64; 3]],
    u: f64,
    scattering: &[f64],
) -> Result<ModelEvaluation, SolutionError> {
    let n = xyz.len();
    let r = hkl.len();
    if n == 0
        || observed.len() != r
        || r == 0
        || scattering.len() != n * r
        || !u.is_finite()
        || !(0.0..=0.2).contains(&u)
        || observed
            .iter()
            .any(|&a| !a.is_finite() || a < 0.0 || a > 1e100)
    {
        return Err(error(
            "invalid model arrays, amplitudes, or U (must lie in [0,0.2])",
        ));
    }
    let norm = observed.iter().map(|a| a * a).sum::<f64>().sqrt();
    let observed_sum: f64 = observed.iter().sum();
    if norm <= 0.0 || !norm.is_finite() {
        return Err(error("positive finite observed amplitude norm required"));
    }
    let occupancy = vec![1.0; n];
    let u_iso = vec![u; n];
    let imaginary = vec![0.; n * r];
    let result = calculate_p1_dense(
        cell,
        P1BatchView {
            hkl,
            fractional_xyz: xyz,
            occupancy: &occupancy,
            u_iso_angstrom2: &u_iso,
            scattering_real: scattering,
            scattering_imag: &imaginary,
            scale: 1.0,
        },
    )
    .map_err(|e| error(e.to_string()))?;
    let a: Vec<_> = result.values.intensity.iter().map(|v| v.sqrt()).collect();
    let denominator: f64 = a.iter().map(|v| v * v).sum();
    if denominator <= 1e-100 || !denominator.is_finite() {
        return Err(error("zero or nonfinite calculated amplitude norm"));
    }
    let k: f64 = a.iter().zip(observed).map(|(a, b)| a * b).sum::<f64>() / denominator;
    let residual: Vec<_> = a
        .iter()
        .zip(observed)
        .map(|(a, b)| (k * a - b) / norm)
        .collect();
    let p = 3 * (n - 1) + 1;
    let mut derivative = vec![0.; p * r];
    for reflection in 0..r {
        if a[reflection] <= 1e-14 {
            continue;
        }
        let re = result.values.f_real[reflection];
        let im = result.values.f_imag[reflection];
        for site in 1..n {
            for axis in 0..3 {
                let row = result.layout.coordinate(site, axis) * r + reflection;
                derivative[(3 * (site - 1) + axis) * r + reflection] =
                    (re * result.d_f_real[row] + im * result.d_f_imag[row]) / a[reflection];
            }
        }
        let mut value = 0.0;
        for site in 0..n {
            let row = result.layout.u_iso(site) * r + reflection;
            value += (re * result.d_f_real[row] + im * result.d_f_imag[row]) / a[reflection];
        }
        derivative[(p - 1) * r + reflection] = value;
    }
    for row in derivative.chunks_exact_mut(r) {
        let dk: f64 = row
            .iter()
            .zip(observed)
            .zip(&a)
            .map(|((&d, &o), &a)| d * (o - 2. * k * a))
            .sum::<f64>()
            / denominator;
        for (d, &amplitude) in row.iter_mut().zip(&a) {
            *d = (k * *d + amplitude * dk) / norm;
        }
    }
    let objective = residual.iter().map(|v| v * v).sum();
    let r1 = a
        .iter()
        .zip(observed)
        .map(|(a, b)| (k * a - b).abs())
        .sum::<f64>()
        / observed_sum;
    if !k.is_finite() || !r1.is_finite() || !derivative.iter().all(|v| v.is_finite()) {
        return Err(error("nonfinite model evaluation"));
    }
    Ok(ModelEvaluation {
        objective,
        r1,
        amplitude_scale: k,
        residual,
        jacobian: derivative,
    })
}

/// Polish coordinates and a common bounded U, retaining only improving states.
///
/// # Errors
/// Returns the same model/input errors as [`evaluate_model`], or invalid budgets.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn polish_model(
    cell: UnitCell,
    hkl: &[[i32; 3]],
    observed: &[f64],
    species: &[String],
    initial: &[[f64; 3]],
    initial_u: f64,
    iterations: usize,
    deadline: Option<Instant>,
    cancellation: Option<&crate::Cancellation>,
) -> Result<PolishedModel, SolutionError> {
    if initial.len() != species.len() || iterations > 200 {
        return Err(error("invalid model shape or polishing budget"));
    }
    let scattering = prepare_scattering(cell, hkl, species)?;
    let mut xyz = initial.to_vec();
    let mut u = initial_u;
    let mut state = evaluate_prepared(cell, hkl, observed, &xyz, u, &scattering)?;
    let p = 3 * (xyz.len() - 1) + 1;
    let r = hkl.len();
    let mut damping: f64 = 1e-3;
    let mut completed = 0;
    for _ in 0..iterations {
        if crate::stopped(deadline, cancellation) {
            break;
        }
        let mut normal = vec![0.; p * p];
        let mut gradient = vec![0.; p];
        for i in 0..p {
            let row = &state.jacobian[i * r..(i + 1) * r];
            gradient[i] = row.iter().zip(&state.residual).map(|(a, b)| a * b).sum();
            for j in 0..=i {
                let value = row
                    .iter()
                    .zip(&state.jacobian[j * r..(j + 1) * r])
                    .map(|(a, b)| a * b)
                    .sum();
                normal[i * p + j] = value;
                normal[j * p + i] = value;
            }
        }
        if gradient.iter().fold(0.0_f64, |a, b| a.max(b.abs())) < 1e-10 {
            break;
        }
        for i in 0..p {
            normal[i * p + i] += damping * normal[i * p + i].max(1e-6);
            gradient[i] = -gradient[i];
        }
        let Some(mut step) = cholesky_solve(&normal, &gradient) else {
            damping *= 10.;
            continue;
        };
        let maximum = step[..p - 1].iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let factor = (0.1 / maximum.max(1e-30))
            .min(1.0)
            .min(0.02 / step[p - 1].abs().max(1e-30));
        for v in &mut step {
            *v *= factor;
        }
        let mut accepted = false;
        for backtrack in 0_i32..10 {
            if crate::stopped(deadline, cancellation) {
                break;
            }
            let alpha = 0.5_f64.powi(backtrack);
            let mut trial = xyz.clone();
            for (site, point) in trial.iter_mut().enumerate().skip(1) {
                for axis in 0..3 {
                    point[axis] =
                        (point[axis] + alpha * step[3 * (site - 1) + axis]).rem_euclid(1.0);
                }
            }
            let trial_u = (u + alpha * step[p - 1]).clamp(0., 0.2);
            let candidate = evaluate_prepared(cell, hkl, observed, &trial, trial_u, &scattering)?;
            if candidate.objective < state.objective {
                xyz = trial;
                u = trial_u;
                state = candidate;
                damping = (damping * 0.3).max(1e-9);
                accepted = true;
                completed += 1;
                break;
            }
        }
        if !accepted {
            damping *= 10.;
            if damping > 1e10 {
                break;
            }
        }
    }
    Ok(PolishedModel {
        fractional_xyz: xyz,
        u_iso: u,
        evaluation: state,
        accepted_steps: completed,
    })
}

pub(crate) fn cholesky_solve(matrix: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    let mut lower = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j];
            for k in 0..j {
                value -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if value <= 0.0 || !value.is_finite() {
                    return None;
                }
                lower[i * n + j] = value.sqrt();
            } else {
                lower[i * n + j] = value / lower[j * n + j];
            }
        }
    }
    let mut x = rhs.to_vec();
    for i in 0..n {
        for j in 0..i {
            x[i] -= lower[i * n + j] * x[j];
        }
        x[i] /= lower[i * n + i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lower[j * n + i] * x[j];
        }
        x[i] /= lower[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profiled_scale_jacobian_matches_centered_differences() {
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 6.,
            c_angstrom: 7.,
            alpha_deg: 90.,
            beta_deg: 103.,
            gamma_deg: 90.,
        };
        let hkl = vec![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 2, 1],
            [-2, 1, 3],
            [3, 1, 2],
        ];
        let species = vec!["Si".into(), "O".into()];
        let xyz = vec![[0.13, 0.24, 0.35], [0.46, 0.51, 0.72]];
        let observed = vec![10., 12., 9., 7., 8., 4.];
        let a = evaluate_model(cell, &hkl, &observed, &species, &xyz, 0.01).unwrap();
        let step = 1e-7;
        for parameter in 0..4 {
            let mut left = xyz.clone();
            let mut right = xyz.clone();
            let (mut ul, mut ur) = (0.01, 0.01);
            if parameter < 3 {
                left[1][parameter] -= step;
                right[1][parameter] += step;
            } else {
                ul -= step;
                ur += step;
            }
            let l = evaluate_model(cell, &hkl, &observed, &species, &left, ul).unwrap();
            let r = evaluate_model(cell, &hkl, &observed, &species, &right, ur).unwrap();
            for j in 0..hkl.len() {
                let fd = (r.residual[j] - l.residual[j]) / (2. * step);
                assert!(
                    (fd - a.jacobian[parameter * hkl.len() + j]).abs() < 2e-7,
                    "parameter {parameter}, reflection {j}"
                );
            }
        }
    }
}
