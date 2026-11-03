//! Canonical isotropic cached geometry, with the original per-image reductions.
use super::{AdpMode, GeometryCache, PreparedSymmetricModel, canonical_isotropic};
use crate::solution::math3::dot;
use phasesmith_crystallography::Matrix3;
use std::collections::BTreeMap;

/// Exact repeated reciprocal weights; no resolution rounding or family assumptions.
#[derive(Clone, Debug)]
pub(super) struct WeightPlan {
    weights: Vec<f64>,
    ids: Vec<usize>,
}
impl WeightPlan {
    pub(super) fn new(reciprocal: &[[f64; 3]]) -> Option<Self> {
        let mut map = BTreeMap::new();
        let mut weights = Vec::new();
        let ids = reciprocal
            .iter()
            .map(|&q| {
                let weight = -2. * std::f64::consts::PI.powi(2) * dot(q, q);
                *map.entry(weight.to_bits()).or_insert_with(|| {
                    let index = weights.len();
                    weights.push(weight);
                    index
                })
            })
            .collect();
        // Avoid a temporary table unless at least half the exponentials repeat.
        (weights.len() <= reciprocal.len() / 2).then_some(Self { weights, ids })
    }
    pub(super) fn displacements(&self, tensors: &[Matrix3]) -> Vec<f64> {
        self.weights
            .iter()
            .flat_map(|&weight| tensors.iter().map(move |u| (weight * u[0][0]).exp()))
            .collect()
    }
}

pub(super) fn eligible(model: &PreparedSymmetricModel, geometry: Option<&GeometryCache>) -> bool {
    model.specialized_isotropic_evaluation
        && geometry.is_some()
        && model.mode == AdpMode::Isotropic
        && model
            .tensor_bases
            .iter()
            .all(|basis| canonical_isotropic(basis))
}

// Values and all coordinate/log-U derivatives are accumulated in the same pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    model: &PreparedSymmetricModel,
    start: usize,
    parameters: &[f64],
    tensors: &[Matrix3],
    tensor_derivatives: &[Vec<Matrix3>],
    geometry: &GeometryCache,
    displacements: Option<&[f64]>,
    amplitudes: &mut [f64],
    jacobian: &mut [f64],
) {
    let count = amplitudes.len();
    let mut dre = vec![0.; model.parameter_count];
    let mut dim = dre.clone();
    for local in 0..count {
        let reflection = start + local;
        let q = model.reciprocal[reflection];
        dre.fill(0.);
        dim.fill(0.);
        let mut real = 0.;
        let mut imaginary = 0.;
        let mut image_index = reflection * model.symmetry.full_count;
        let entry = model
            .isotropic_weights
            .as_ref()
            .map(|p| (p, p.ids[reflection]));
        let weight = entry.map_or_else(
            || -2. * std::f64::consts::PI.powi(2) * dot(q, q),
            |(plan, id)| plan.weights[id],
        );
        for (site_index, site) in model.symmetry.sites.iter().enumerate() {
            let (coordinate_offset, tensor_offset) = model.offsets[site_index];
            let displacement = displacements.map_or_else(
                || (weight * tensors[site_index][0][0]).exp(),
                |table| {
                    table[entry.expect("prepared displacement plan").1 * model.symmetry.sites.len()
                        + site_index]
                },
            );
            let du = weight * tensor_derivatives[site_index][0][0][0];
            let factor = model.factors[reflection * model.symmetry.sites.len() + site_index];
            for _ in &site.images {
                let g = &geometry.entries[image_index];
                image_index += 1;
                let mut phase = g.phase;
                for i in 0..site.coordinate_basis.len() {
                    phase += g.coordinate_phase[i] * parameters[coordinate_offset + i];
                }
                let (sin, cos) = g.sin_cos(phase, site.coordinate_basis.len());
                let re = factor * displacement * cos;
                let im = factor * displacement * sin;
                real += re;
                imaginary += im;
                for i in 0..site.coordinate_basis.len() {
                    let derivative = g.coordinate_phase[i];
                    dre[coordinate_offset + i] -= derivative * im;
                    dim[coordinate_offset + i] += derivative * re;
                }
                dre[tensor_offset] += du * re;
                dim[tensor_offset] += du * im;
            }
        }
        let raw = real.hypot(imaginary);
        let family_weight = model
            .families
            .as_ref()
            .map_or(1., |p| p.square_roots[reflection]);
        amplitudes[local] = raw * family_weight;
        if raw > 1e-14 {
            for parameter in 0..model.parameter_count {
                jacobian[parameter * count + local] =
                    ((real * dre[parameter] + imaginary * dim[parameter]) / raw) * family_weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_plan_only_reuses_identical_bit_patterns() {
        let next = f64::from_bits(1_f64.to_bits() + 1);
        let plan = WeightPlan::new(&[[1., 0., 0.], [-1., 0., 0.], [next, 0., 0.], [-next, 0., 0.]])
            .unwrap();
        assert_eq!(plan.ids[0], plan.ids[1]);
        assert_eq!(plan.ids[2], plan.ids[3]);
        assert_ne!(plan.ids[0], plan.ids[2]);
        assert_eq!(plan.weights.len(), 2);
        assert!(WeightPlan::new(&[[1., 0., 0.], [2., 0., 0.]]).is_none());
    }
}
