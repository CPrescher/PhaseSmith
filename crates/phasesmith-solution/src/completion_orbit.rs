//! Partial-model difference Fourier and fixed-topology orbit value/Jacobian.
use crate::fourier::{Transform, grid_index, grid_len};
use crate::math3::{
    IDENTITY, add, dot, multiply, mv, periodic_difference, pseudo_solve, sub, transpose,
};
use crate::{SolutionError, error};
use phasesmith_crystallography::{
    CellGeometry, PreparedXrayScattering, SpaceGroup, SymmetryOperation, UnitCell,
};
use rustfft::num_complex::Complex;

/// Prepared measured amplitudes and a fixed partial framework.
#[derive(Clone, Debug)]
pub struct PreparedOrbitSearch {
    /// Validated cell geometry.
    pub geometry: CellGeometry,
    /// Supplied exact setting.
    pub group: SpaceGroup,
    /// Projected full-cell framework positions.
    pub fixed_xyz: Vec<[f64; 3]>,
    pub(crate) hkl: Vec<[i32; 3]>,
    pub(crate) observed: Vec<f64>,
    pub(crate) base: Vec<Complex<f64>>,
    added_factor: Vec<f64>,
}

/// Common full-model amplitude score and fused derivatives for an added orbit.
#[derive(Clone, Debug)]
pub struct OrbitEvaluation {
    /// Normalized profiled amplitude squared residual.
    pub objective: f64,
    /// Unweighted amplitude R1.
    pub r1: f64,
    /// Profiled amplitude scale.
    pub amplitude_scale: f64,
    /// Gradient with respect to the representative fractional position.
    pub gradient: [f64; 3],
    /// Normalized residuals.
    pub residual: Vec<f64>,
    /// Three parameter-major normalized residual derivative rows.
    pub jacobian: Vec<f64>,
    /// Unique physical images.
    pub fractional_xyz: Vec<[f64; 3]>,
}

/// Difference-map diagnostic without treating unknown partial phases as observed.
#[derive(Clone, Debug)]
pub struct DifferenceMap {
    /// Flattened signed map in C xyz order, arbitrary model units.
    pub density: Vec<f64>,
    /// Profiled partial-model amplitude scale.
    pub amplitude_scale: f64,
    /// Number of measured coefficients with a defined partial-model phase.
    pub phased_reflections: usize,
}

impl PreparedOrbitSearch {
    /// Prepare a fixed full-cell framework and one new orbit species.
    ///
    /// # Errors
    /// Rejects invalid geometry/setting, arrays/species, nonfinite values or budgets.
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    pub fn new(
        cell: UnitCell,
        group: &SpaceGroup,
        fixed_species: &[String],
        fixed_xyz: &[[f64; 3]],
        added_species: &str,
        hkl: &[[i32; 3]],
        observed: &[f64],
        u_iso: f64,
    ) -> Result<Self, SolutionError> {
        if fixed_xyz.is_empty()
            || fixed_xyz.len() > 128
            || fixed_xyz.len() != fixed_species.len()
            || fixed_xyz.iter().flatten().any(|v| !v.is_finite())
            || hkl.is_empty()
            || hkl.len() > 100_000
            || hkl.len() != observed.len()
            || hkl.windows(2).any(|w| w[0] >= w[1])
            || hkl
                .iter()
                .any(|h| h.iter().any(|v| v.unsigned_abs() > 1024))
            || observed
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 1e100)
            || !observed.iter().any(|v| *v > 0.)
            || !u_iso.is_finite()
            || !(0.0..=0.2).contains(&u_iso)
            || group.operations().len() > 192
            || hkl.len() * (fixed_xyz.len() + 1) > 2_000_000
        {
            return Err(error("invalid orbit-search arrays, U or resource bounds"));
        }
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        for op in group.operations() {
            let q = multiply(
                multiply(
                    geometry.direct_basis,
                    op.rotation().map(|r| r.map(f64::from)),
                ),
                transpose(geometry.reciprocal_basis),
            );
            let check = multiply(transpose(q), q);
            if check
                .iter()
                .flatten()
                .zip(IDENTITY.iter().flatten())
                .any(|(a, b)| (a - b).abs() > 1e-8)
            {
                return Err(error(
                    "cell metric is incompatible with the supplied symmetry",
                ));
            }
        }
        let fixed_xyz = fixed_xyz
            .iter()
            .map(|p| p.map(|v| v.rem_euclid(1.)))
            .collect::<Vec<_>>();
        let s = hkl
            .iter()
            .map(|h| {
                let q = mv(geometry.reciprocal_basis, h.map(f64::from));
                dot(q, q).sqrt() / 2.
            })
            .collect::<Vec<_>>();
        let mut species = fixed_species.to_vec();
        species.push(added_species.to_owned());
        let factors = PreparedXrayScattering::new(&species)
            .and_then(|p| p.evaluate(&s))
            .map_err(|e| error(e.to_string()))?
            .real;
        let mut base = Vec::with_capacity(hkl.len());
        let mut added_factor = Vec::with_capacity(hkl.len());
        for (index, &h) in hkl.iter().enumerate() {
            let row = &factors[index * species.len()..(index + 1) * species.len()];
            let dw = (-8. * std::f64::consts::PI.powi(2) * u_iso * s[index].powi(2)).exp();
            let mut total = Complex::default();
            for (&x, &f) in fixed_xyz.iter().zip(row) {
                total +=
                    Complex::from_polar(f * dw, std::f64::consts::TAU * dot(h.map(f64::from), x));
            }
            base.push(total);
            added_factor.push(row[fixed_xyz.len()] * dw);
        }
        Ok(Self {
            geometry,
            group: group.clone(),
            fixed_xyz,
            hkl: hkl.to_vec(),
            observed: observed.to_vec(),
            base,
            added_factor,
        })
    }

    /// Fixed-framework complex structure factors for independent diagnostics.
    #[must_use]
    pub fn framework_f(&self) -> Vec<[f64; 2]> {
        self.base.iter().map(|v| [v.re, v.im]).collect()
    }

    /// Evaluate one unique orbit and all three analytic coordinate derivatives.
    /// Derivatives assume fixed topology and site-stabilizer-compatible directions.
    ///
    /// # Errors
    /// Rejects invalid representatives or a nonfinite/zero calculated norm.
    pub fn evaluate(&self, point: [f64; 3]) -> Result<OrbitEvaluation, SolutionError> {
        if point.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid orbit representative"));
        }
        let point = point.map(|v| v.rem_euclid(1.));
        let expansion = self
            .group
            .expand_sites(&[point], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        if expansion.fractional_xyz.len() + self.fixed_xyz.len() > 128 {
            return Err(error("completed orbit exceeds 128 full-cell atoms"));
        }
        let count = self.hkl.len();
        let mut amplitude = vec![0.; count];
        let mut derivative = vec![0.; 3 * count];
        for (row, &h) in self.hkl.iter().enumerate() {
            let h = h.map(f64::from);
            let mut total = self.base[row];
            let mut jet = [Complex::default(); 3];
            for (&x, &rotation) in expansion
                .fractional_xyz
                .iter()
                .zip(&expansion.representative_rotation)
            {
                let value =
                    Complex::from_polar(self.added_factor[row], std::f64::consts::TAU * dot(h, x));
                total += value;
                for axis in 0..3 {
                    let phase = std::f64::consts::TAU
                        * (0..3)
                            .map(|j| h[j] * f64::from(rotation[j][axis]))
                            .sum::<f64>();
                    jet[axis] += Complex::new(-value.im * phase, value.re * phase);
                }
            }
            let a = total.norm();
            amplitude[row] = a;
            if a > 1e-14 {
                for axis in 0..3 {
                    derivative[axis * count + row] = (total.conj() * jet[axis]).re / a;
                }
            }
        }
        let norm: f64 = amplitude.iter().map(|v| v * v).sum();
        let obs_norm = self.observed.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-100 || !norm.is_finite() {
            return Err(error("zero/nonfinite orbit amplitude norm"));
        }
        let scale = amplitude
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| a * o)
            .sum::<f64>()
            / norm;
        let residual = amplitude
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| (scale * a - o) / obs_norm)
            .collect::<Vec<_>>();
        let mut gradient = [0.; 3];
        for (axis, row) in derivative.chunks_exact_mut(count).enumerate() {
            let dk = row
                .iter()
                .zip(&self.observed)
                .zip(&amplitude)
                .map(|((&d, &o), &a)| d * (o - 2. * scale * a))
                .sum::<f64>()
                / norm;
            for (d, &a) in row.iter_mut().zip(&amplitude) {
                *d = (scale * *d + dk * a) / obs_norm;
            }
            gradient[axis] = 2. * row.iter().zip(&residual).map(|(d, r)| d * r).sum::<f64>();
        }
        let objective = residual.iter().map(|r| r * r).sum();
        let r1 = amplitude
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| (scale * a - o).abs())
            .sum::<f64>()
            / self.observed.iter().sum::<f64>();
        if !r1.is_finite() || derivative.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite orbit objective"));
        }
        Ok(OrbitEvaluation {
            objective,
            r1,
            amplitude_scale: scale,
            gradient,
            residual,
            jacobian: derivative,
            fractional_xyz: expansion.fractional_xyz,
        })
    }

    /// Signed Fo-Fp Fourier map with explicit zeros for undefined partial phases.
    ///
    /// # Errors
    /// Rejects aliased indices, missing/nonmatching Friedel mates or zero scale.
    #[allow(clippy::cast_precision_loss)]
    pub fn difference_map(&self, shape: [usize; 3]) -> Result<DifferenceMap, SolutionError> {
        let n = grid_len(shape)?;
        let norm = self.base.iter().map(Complex::norm_sqr).sum::<f64>();
        let scale = self
            .base
            .iter()
            .zip(&self.observed)
            .map(|(f, o)| f.norm() * o)
            .sum::<f64>()
            / norm;
        if !scale.is_finite() || scale <= 1e-100 {
            return Err(error("zero/nonfinite partial-model scale"));
        }
        let threshold = self.base.iter().map(|f| f.norm()).fold(0.0_f64, f64::max) * 1e-10;
        let mut coefficients = vec![Complex::default(); n];
        let mut phased = 0;
        for (row, &h) in self.hkl.iter().enumerate() {
            if h == [0; 3] {
                continue;
            }
            if (0..3).any(|i| 2 * i64::from(h[i]).abs() >= i64::try_from(shape[i]).unwrap_or(0)) {
                return Err(error("difference-map HKL aliases a Nyquist/grid boundary"));
            }
            let negative = h.map(|v| v.checked_neg().unwrap_or(i32::MAX));
            let mate = self
                .hkl
                .binary_search(&negative)
                .map_err(|_| error("missing difference-map Friedel mate"))?;
            if (self.observed[row] - self.observed[mate]).abs() > 1e-10 * (1. + self.observed[row])
            {
                return Err(error("difference-map Friedel amplitudes disagree"));
            }
            if self
                .group
                .is_systematically_absent(h)
                .map_err(|e| error(e.to_string()))?
            {
                continue;
            }
            let magnitude = self.base[row].norm();
            if magnitude > threshold {
                coefficients[grid_index(h, shape)] =
                    self.base[row] * ((self.observed[row] / scale - magnitude) / magnitude);
                phased += 1;
            }
        }
        Transform::new(shape).apply(&mut coefficients, false);
        let density = coefficients
            .iter()
            .map(|v| v.re / n as f64)
            .collect::<Vec<_>>();
        if density.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite difference map"));
        }
        Ok(DifferenceMap {
            density,
            amplitude_scale: scale,
            phased_reflections: phased,
        })
    }
}

/// Cartesian minimum-norm projection onto simultaneous nearby affine fixed sets.
///
/// # Errors
/// Rejects nonfinite input or invalid tolerance; returns None for incompatible or
/// distant constraints rather than labelling an approximate point as special.
pub fn project_fixed_sets(
    geometry: &CellGeometry,
    seed: [f64; 3],
    operations: &[SymmetryOperation],
    tolerance: f64,
) -> Result<Option<[f64; 3]>, SolutionError> {
    if seed.iter().any(|v| !v.is_finite())
        || !tolerance.is_finite()
        || !(1e-5..=0.5).contains(&tolerance)
        || operations.len() > 3
    {
        return Err(error("invalid affine projection boundary"));
    }
    let mut normal = [[0.; 3]; 3];
    let mut gradient = [0.; 3];
    let mut equations = Vec::new();
    for &op in operations {
        let r = op.rotation().map(|r| r.map(f64::from));
        let fractional = std::array::from_fn(|i| std::array::from_fn(|j| r[i][j] - IDENTITY[i][j]));
        let jac = multiply(
            multiply(geometry.direct_basis, fractional),
            transpose(geometry.reciprocal_basis),
        );
        let delta = periodic_difference(geometry, sub(op.apply_fractional(seed), seed))?;
        let residual = mv(geometry.direct_basis, delta);
        crate::math3::matrix_add(&mut normal, multiply(transpose(jac), jac));
        gradient = add(gradient, mv(transpose(jac), residual));
        equations.push((jac, residual));
    }
    let correction = pseudo_solve(normal, gradient.map(|v| -v));
    if dot(correction, correction) > tolerance * tolerance * (1. + 1e-12)
        || equations.iter().any(|&(jac, residual)| {
            let r = add(mv(jac, correction), residual);
            dot(r, r) > 1e-16
        })
    {
        return Ok(None);
    }
    let shift = mv(transpose(geometry.reciprocal_basis), correction);
    Ok(Some(add(seed, shift).map(|v| v.rem_euclid(1.))))
}
