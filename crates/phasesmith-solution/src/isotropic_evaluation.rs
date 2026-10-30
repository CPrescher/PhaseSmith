//! Canonical isotropic cached geometry, with the original per-image reductions.
use super::{AdpMode, GeometryCache, PreparedSymmetricModel, canonical_isotropic};
use crate::math3::dot;
use phasesmith_crystallography::Matrix3;

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
        let weight = -2. * std::f64::consts::PI.powi(2) * dot(q, q);
        for (site_index, site) in model.symmetry.sites.iter().enumerate() {
            let (coordinate_offset, tensor_offset) = model.offsets[site_index];
            let displacement = (weight * tensors[site_index][0][0]).exp();
            let du = weight * tensor_derivatives[site_index][0][0][0];
            let factor = model.factors[reflection * model.symmetry.sites.len() + site_index];
            for _ in &site.images {
                let g = &geometry.entries[image_index];
                image_index += 1;
                let mut phase = g.phase;
                for i in 0..site.coordinate_basis.len() {
                    phase += g.coordinate_phase[i] * parameters[coordinate_offset + i];
                }
                let (sin, cos) = phase.sin_cos();
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
