//! Analytic spherical integration of a band-limited measured-amplitude map.
use crate::fourier::{Transform, grid_index, grid_len};
use crate::math3::{dot, mv};
use crate::{SolutionError, error};
use phasesmith_crystallography::UnitCell;
use rustfft::num_complex::Complex;

/// Periodic integrated map and its analytic radius derivative.
#[derive(Clone, Debug)]
pub struct IntegratedPeakDensity {
    shape: [usize; 3],
    values: Vec<f64>,
    radial: Vec<f64>,
    /// Number of measured coefficients with an available source-map phase.
    pub phased_reflections: usize,
}
/// Fused interpolation result in arbitrary measured-amplitude units.
#[derive(Clone, Copy, Debug)]
pub struct PeakIntegral {
    /// Signed spherical integral.
    pub value: f64,
    /// Derivatives with respect to fractional center coordinates.
    pub gradient: [f64; 3],
    /// Derivative with respect to the Cartesian radius in angstrom.
    pub radius_derivative: f64,
}

/// Cartesian ball Fourier transform and its radius derivative.
/// # Errors
/// Rejects negative/nonfinite frequency or invalid radius.
pub fn ball_transform(q: f64, radius: f64) -> Result<(f64, f64), SolutionError> {
    if !q.is_finite() || q < 0. || !radius.is_finite() || !(1e-4..=2.).contains(&radius) {
        return Err(error(
            "invalid ball frequency or radius (allowed radius [1e-4,2] A)",
        ));
    }
    let t = std::f64::consts::TAU * q * radius;
    if !t.is_finite() {
        return Err(error("nonfinite ball transform argument"));
    }
    let t2 = t * t;
    let (window, sinc) = if t.abs() < 0.1 {
        (
            1. - t2 / 10. + t2 * t2 / 280. - t2 * t2 * t2 / 15120. + t2.powi(4) / 1_330_560.,
            1. - t2 / 6. + t2 * t2 / 120. - t2 * t2 * t2 / 5040. + t2.powi(4) / 362_880.,
        )
    } else {
        let (s, c) = t.sin_cos();
        (3. * (s / t - c) / (t * t), s / t)
    };
    Ok((
        4. * std::f64::consts::PI * radius.powi(3) / 3. * window,
        4. * std::f64::consts::PI * radius * radius * sinc,
    ))
}

impl IntegratedPeakDensity {
    /// Prepare an unsharpened Fo map integrated over a Cartesian ball.
    /// # Errors
    /// Rejects invalid arrays, radius, aliasing, or non-Hermitian observations.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    pub fn new(
        cell: UnitCell,
        shape: [usize; 3],
        hkl: &[[i32; 3]],
        observed: &[f64],
        source_density: &[f64],
        radius: f64,
    ) -> Result<Self, SolutionError> {
        let n = grid_len(shape)?;
        ball_transform(0., radius)?;
        if hkl.is_empty()
            || hkl.len() > 100_000
            || hkl.len() != observed.len()
            || source_density.len() != n
            || source_density
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 1e100)
            || hkl.windows(2).any(|w| w[0] >= w[1])
            || observed
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1e100)
            || !observed.iter().any(|v| *v > 0.)
        {
            return Err(error("invalid integrated-density arrays"));
        }
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let mut fft = Transform::new(shape);
        let mut phases: Vec<_> = source_density
            .iter()
            .map(|&v| Complex::new(v, 0.))
            .collect();
        fft.apply(&mut phases, true);
        let threshold = phases.iter().map(|v| v.norm()).fold(0., f64::max) * 1e-12;
        let mut values = vec![Complex::default(); n];
        let mut radial = values.clone();
        let mut available = 0;
        for (row, &h) in hkl.iter().enumerate() {
            if (0..3).any(|i| 2 * i64::from(h[i]).abs() >= i64::try_from(shape[i]).unwrap_or(0)) {
                return Err(error("integrated-density HKL aliases a Nyquist boundary"));
            }
            let negative = h.map(|v| v.checked_neg().unwrap_or(i32::MAX));
            let mate = hkl
                .binary_search(&negative)
                .map_err(|_| error("integrated-density Friedel mate missing"))?;
            if (observed[row] - observed[mate]).abs() > 1e-10 * (1. + observed[row]) {
                return Err(error("integrated-density Friedel amplitudes disagree"));
            }
            if h > negative {
                continue;
            }
            let index = grid_index(h, shape);
            let phase = phases[index];
            if phase.norm() <= threshold || phase.norm() <= 1e-100 {
                continue;
            }
            let q = mv(geometry.reciprocal_basis, h.map(f64::from));
            let (k, dr) = ball_transform(dot(q, q).sqrt(), radius)?;
            let coefficient = phase / phase.norm() * observed[row] / geometry.volume_angstrom3;
            values[index] = coefficient * k;
            radial[index] = coefficient * dr;
            available += 1;
            if h != negative {
                let j = grid_index(negative, shape);
                values[j] = values[index].conj();
                radial[j] = radial[index].conj();
                available += 1;
            }
        }
        fft.apply(&mut values, false);
        fft.apply(&mut radial, false);
        let values: Vec<_> = values.into_iter().map(|v| v.re).collect();
        let radial: Vec<_> = radial.into_iter().map(|v| v.re).collect();
        if values.iter().chain(&radial).any(|v| !v.is_finite()) {
            return Err(error("nonfinite integrated density"));
        }
        Ok(Self {
            shape,
            values,
            radial,
            phased_reflections: available,
        })
    }

    /// Value/center-gradient/radius-derivative in one periodic eight-node pass.
    /// # Errors
    /// Rejects nonfinite centers.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn evaluate(&self, point: [f64; 3]) -> Result<PeakIntegral, SolutionError> {
        if point.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid integrated-density center"));
        }
        let grid =
            std::array::from_fn::<_, 3, _>(|i| point[i].rem_euclid(1.) * self.shape[i] as f64);
        let origin = grid.map(|v| v.floor() as i32);
        let t = std::array::from_fn::<_, 3, _>(|i| grid[i] - f64::from(origin[i]));
        let mut result = PeakIntegral {
            value: 0.,
            gradient: [0.; 3],
            radius_derivative: 0.,
        };
        for x in 0..=1 {
            for y in 0..=1 {
                for z in 0..=1 {
                    let bits = [x, y, z];
                    let w =
                        std::array::from_fn::<_, 3, _>(
                            |i| if bits[i] == 0 { 1. - t[i] } else { t[i] },
                        );
                    let index =
                        grid_index(std::array::from_fn(|i| origin[i] + bits[i]), self.shape);
                    let weight = w[0] * w[1] * w[2];
                    result.value += weight * self.values[index];
                    result.radius_derivative += weight * self.radial[index];
                    for axis in 0..3 {
                        let derivative = if bits[axis] == 0 { -1. } else { 1. }
                            * self.shape[axis] as f64
                            * w[(axis + 1) % 3]
                            * w[(axis + 2) % 3];
                        result.gradient[axis] += derivative * self.values[index];
                    }
                }
            }
        }
        if !result.value.is_finite()
            || !result.radius_derivative.is_finite()
            || result.gradient.iter().any(|v| !v.is_finite())
        {
            return Err(error("nonfinite integrated-density interpolation"));
        }
        Ok(result)
    }
}

/// Conservative relative-mass label admissibility, with all labels on weak evidence.
/// # Errors
/// Rejects invalid neutral atomic numbers or nonfinite mass/scale.
pub fn density_label_mask(
    mass: f64,
    scale: Option<f64>,
    atomic_numbers: &[u8],
) -> Result<Vec<bool>, SolutionError> {
    if !mass.is_finite()
        || atomic_numbers.is_empty()
        || atomic_numbers.len() > 8
        || atomic_numbers.iter().any(|z| !(2..=118).contains(z))
        || scale.is_some_and(|s| !s.is_finite() || s <= 0.)
    {
        return Err(error("invalid density label evidence"));
    }
    let Some(scale) = scale.filter(|_| mass > 0.) else {
        return Ok(vec![true; atomic_numbers.len()]);
    };
    let estimate = mass / scale;
    if !estimate.is_finite() || estimate <= 0. {
        return Ok(vec![true; atomic_numbers.len()]);
    }
    let errors: Vec<_> = atomic_numbers
        .iter()
        .map(|&z| (estimate / f64::from(z)).ln().abs())
        .collect();
    let best = errors.iter().copied().fold(f64::INFINITY, f64::min);
    // Preserve near neighbors and a broad 50% relative band, independent of counts.
    Ok(errors
        .iter()
        .map(|&e| e <= best + 1.25_f64.ln() || e <= 1.5_f64.ln())
        .collect())
}
