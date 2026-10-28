//! Cached parent phases and one orbit phase jet shared by alternative elements.
use crate::math3::{dot, mv};
use crate::{OrbitEvaluation, PreparedOrbitSearch, SolutionError, error};
use phasesmith_crystallography::{PreparedXrayScattering, SpaceGroup, UnitCell};
use rustfft::num_complex::Complex;

/// A parent framework with prepared alternative-element scattering columns.
#[derive(Clone, Debug)]
pub struct PreparedElementPool {
    /// Validated parent framework and shared difference-map preparation.
    pub parent: PreparedOrbitSearch,
    /// Alternative label order, identical to returned evaluations.
    pub elements: Vec<String>,
    factors: Vec<Vec<f64>>,
}

impl PreparedElementPool {
    /// Prepare all element labels once, without repeating parent phase sums.
    /// # Errors
    /// Rejects invalid orbit-search inputs, empty/duplicate labels or excess memory.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cell: UnitCell,
        group: &SpaceGroup,
        fixed_species: &[String],
        fixed_xyz: &[[f64; 3]],
        elements: &[String],
        hkl: &[[i32; 3]],
        observed: &[f64],
        u_iso: f64,
    ) -> Result<Self, SolutionError> {
        if elements.is_empty()
            || elements.len() > 8
            || elements
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != elements.len()
            || elements.len().saturating_mul(hkl.len()).saturating_mul(4) > 2_000_000
        {
            return Err(error("invalid shared element labels or memory bound"));
        }
        let parent = PreparedOrbitSearch::new(
            cell,
            group,
            fixed_species,
            fixed_xyz,
            &elements[0],
            hkl,
            observed,
            u_iso,
        )?;
        let s: Vec<_> = hkl
            .iter()
            .map(|h| {
                let q = mv(parent.geometry.reciprocal_basis, h.map(f64::from));
                dot(q, q).sqrt() / 2.
            })
            .collect();
        let f = PreparedXrayScattering::new(elements)
            .and_then(|p| p.evaluate(&s))
            .map_err(|e| error(e.to_string()))?
            .real;
        let factors = (0..elements.len())
            .map(|e| {
                s.iter()
                    .enumerate()
                    .map(|(h, &s)| {
                        f[h * elements.len() + e]
                            * (-8. * std::f64::consts::PI.powi(2) * u_iso * s * s).exp()
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            parent,
            elements: elements.to_vec(),
            factors,
        })
    }

    /// Score all alternative labels, reusing each image's phase and phase jet.
    /// # Errors
    /// Rejects invalid representatives, excessive full-cell count or zero norms.
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, point: [f64; 3]) -> Result<Vec<OrbitEvaluation>, SolutionError> {
        if point.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid shared orbit point"));
        }
        let expansion = self
            .parent
            .group
            .expand_sites(&[point.map(|v| v.rem_euclid(1.))], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        if expansion.fractional_xyz.len() + self.parent.fixed_xyz.len() > 128 {
            return Err(error("shared orbit exceeds 128 atoms"));
        }
        let n = self.parent.hkl.len();
        let mut amplitudes = vec![vec![0.; n]; self.elements.len()];
        let mut derivatives = vec![vec![0.; 3 * n]; self.elements.len()];
        for (h, &index) in self.parent.hkl.iter().enumerate() {
            let index = index.map(f64::from);
            let mut phase_sum = Complex::default();
            let mut jet = [Complex::default(); 3];
            for (&x, &rotation) in expansion
                .fractional_xyz
                .iter()
                .zip(&expansion.representative_rotation)
            {
                let value = Complex::from_polar(1., std::f64::consts::TAU * dot(index, x));
                phase_sum += value;
                for axis in 0..3 {
                    let phase = std::f64::consts::TAU
                        * (0..3)
                            .map(|j| index[j] * f64::from(rotation[j][axis]))
                            .sum::<f64>();
                    jet[axis] += Complex::new(-value.im * phase, value.re * phase);
                }
            }
            for e in 0..self.elements.len() {
                let total = self.parent.base[h] + phase_sum * self.factors[e][h];
                let a = total.norm();
                amplitudes[e][h] = a;
                if a > 1e-14 {
                    for axis in 0..3 {
                        derivatives[e][axis * n + h] =
                            (total.conj() * jet[axis]).re * self.factors[e][h] / a;
                    }
                }
            }
        }
        amplitudes
            .into_iter()
            .zip(derivatives)
            .map(|(a, d)| {
                finish(
                    &self.parent.observed,
                    &a,
                    d,
                    expansion.fractional_xyz.clone(),
                )
            })
            .collect()
    }
}

fn finish(
    observed: &[f64],
    amplitude: &[f64],
    mut derivative: Vec<f64>,
    fractional_xyz: Vec<[f64; 3]>,
) -> Result<OrbitEvaluation, SolutionError> {
    let norm = amplitude.iter().map(|v| v * v).sum::<f64>();
    let obs_norm = observed.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-100 || !norm.is_finite() {
        return Err(error("zero/nonfinite shared norm"));
    }
    let scale = amplitude
        .iter()
        .zip(observed)
        .map(|(a, o)| a * o)
        .sum::<f64>()
        / norm;
    let residual: Vec<_> = amplitude
        .iter()
        .zip(observed)
        .map(|(a, o)| (scale * a - o) / obs_norm)
        .collect();
    let mut gradient = [0.; 3];
    for (axis, row) in derivative.chunks_exact_mut(observed.len()).enumerate() {
        let dk = row
            .iter()
            .zip(observed)
            .zip(amplitude)
            .map(|((&d, &o), &a)| d * (o - 2. * scale * a))
            .sum::<f64>()
            / norm;
        for (d, &a) in row.iter_mut().zip(amplitude) {
            *d = (scale * *d + dk * a) / obs_norm;
        }
        gradient[axis] = 2. * row.iter().zip(&residual).map(|(d, r)| d * r).sum::<f64>();
    }
    let objective = residual.iter().map(|r| r * r).sum();
    let r1 = amplitude
        .iter()
        .zip(observed)
        .map(|(a, o)| (scale * a - o).abs())
        .sum::<f64>()
        / observed.iter().sum::<f64>();
    if !r1.is_finite() || derivative.iter().any(|v| !v.is_finite()) {
        return Err(error("nonfinite shared fit"));
    }
    Ok(OrbitEvaluation {
        objective,
        r1,
        amplitude_scale: scale,
        gradient,
        residual,
        jacobian: derivative,
        fractional_xyz,
    })
}
