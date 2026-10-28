//! Symmetry-allowed log-SPD displacement tensors and fused analytical derivatives.

use crate::math3::{
    IDENTITY, add, conjugate, dot, exponential, frobenius, matrix_add, mv, scaled, transpose,
};
use crate::symmetry_model::SymmetryModel;
use crate::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{Matrix3, PreparedXrayScattering};
use std::time::Instant;

/// Displacement model on each asymmetric site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdpMode {
    /// One independently fitted isotropic displacement per asymmetric site.
    Isotropic,
    /// All independent site-symmetry-allowed Cartesian tensor components.
    Anisotropic,
}

/// Frozen coordinates/orbits and bounded reflection/scattering preparation.
#[derive(Clone, Debug)]
pub struct PreparedSymmetricModel {
    /// Recovered supplied symmetry and coordinate tangents.
    pub symmetry: SymmetryModel,
    /// Independent invariant log-tensor bases per asymmetric site.
    pub tensor_bases: Vec<Vec<Matrix3>>,
    /// Coordinate and then tensor parameter offsets for each site.
    pub offsets: Vec<(usize, usize)>,
    /// Total coordinate/log-tensor parameter count, excluding profiled scale.
    pub parameter_count: usize,
    hkl: Vec<[i32; 3]>,
    observed: Vec<f64>,
    factors: Vec<f64>,
    reciprocal: Vec<[f64; 3]>,
}

/// Values and parameter-major derivatives from one fused reflection/image pass.
#[derive(Clone, Debug)]
pub struct SymmetricEvaluation {
    /// Normalized unweighted amplitude least-squares objective.
    pub objective: f64,
    /// Unweighted amplitude residual factor on the supplied observations.
    pub r1: f64,
    /// Profiled amplitude scale.
    pub amplitude_scale: f64,
    /// Normalized amplitude residuals.
    pub residual: Vec<f64>,
    /// Parameter-major derivatives of the normalized residual.
    pub jacobian: Vec<f64>,
    /// Refined asymmetric coordinates in the supplied setting.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Physical Cartesian asymmetric tensors in angstrom squared.
    pub u_cartesian: Vec<Matrix3>,
}

/// Last accepted constrained refinement, including a truthful stop reason.
#[derive(Clone, Debug)]
pub struct SymmetricRefinement {
    /// Frozen setting and topology.
    pub model: PreparedSymmetricModel,
    /// Last accepted independent parameter values.
    pub parameters: Vec<f64>,
    /// Last accepted objective and tensors.
    pub evaluation: SymmetricEvaluation,
    /// Count of improving accepted steps.
    pub accepted_steps: usize,
    /// `gradient`, `max_iterations`, `stagnated`, `cancelled`, or `max_runtime`.
    pub termination: &'static str,
}

/// Lightweight final candidate; excludes prepared scattering and dense Jacobians.
#[derive(Clone, Debug)]
pub struct RefinementSummary {
    /// One neutral species per asymmetric site.
    pub species: Vec<String>,
    /// Asymmetric coordinates in the supplied symmetry setting.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Asymmetric Cartesian tensors in angstrom squared.
    pub u_cartesian: Vec<Matrix3>,
    /// Asymmetric CIF tensors in order 11,22,33,23,13,12.
    pub u_cif: Vec<[f64; 6]>,
    /// Asymmetric `U_equiv` values from Cartesian tensor traces.
    pub u_equiv: Vec<f64>,
    /// Full-cell Cartesian tensors in the original candidate atom order.
    pub full_u_cartesian: Vec<Matrix3>,
    /// Full-cell CIF tensors in the original candidate atom order.
    pub full_u_cif: Vec<[f64; 6]>,
    /// Full-cell `U_equiv` values from Cartesian tensor traces.
    pub full_u_equiv: Vec<f64>,
    /// Aligned coordinate = hand * density-frame coordinate + `origin_shift`.
    pub hand: i32,
    /// Shift to the supplied symmetry setting.
    pub origin_shift: [f64; 3],
    /// Maximum pre-projection same-element matching error in angstrom.
    pub maximum_matching_error: f64,
    /// Number of fitted coordinate and log-tensor parameters, excluding scale.
    pub parameter_count: usize,
    /// Number of independent fitted coordinate parameters.
    pub coordinate_parameter_count: usize,
    /// Removed continuous origin-gauge dimension.
    pub origin_gauge_dimension: usize,
    /// Isotropic or anisotropic site model.
    pub mode: &'static str,
    /// Last accepted numerical stop reason.
    pub termination: &'static str,
    /// Accepted constrained-refinement steps.
    pub accepted_steps: usize,
    /// Original P1/common-U fit metrics, before changing the model.
    pub p1_r1: f64,
    /// Original P1/common-U normalized amplitude objective.
    pub p1_objective: f64,
}

fn tensor_cif(geometry: &phasesmith_crystallography::CellGeometry, u: Matrix3) -> [f64; 6] {
    let inverse = transpose(geometry.reciprocal_basis);
    let conversion =
        std::array::from_fn(|i| inverse[i].map(|v| v / geometry.reciprocal_metric[i][i].sqrt()));
    let tensor = conjugate(conversion, u);
    [
        tensor[0][0],
        tensor[1][1],
        tensor[2][2],
        tensor[1][2],
        tensor[0][2],
        tensor[0][1],
    ]
}

impl SymmetricRefinement {
    /// Produce output arrays without retaining the large prepared work buffers.
    /// Returns full-cell coordinates in the original density/P1 frame.
    #[must_use]
    pub fn summarize(
        &self,
        mode: AdpMode,
        p1_r1: f64,
        p1_objective: f64,
    ) -> (RefinementSummary, Vec<[f64; 3]>) {
        let symmetry = &self.model.symmetry;
        let mut full_xyz = vec![[0.; 3]; symmetry.full_count];
        let mut full_u_cartesian = vec![[[0.; 3]; 3]; symmetry.full_count];
        for (site_index, site) in symmetry.sites.iter().enumerate() {
            for image in &site.images {
                let aligned = image
                    .operation
                    .apply_fractional(self.evaluation.fractional_xyz[site_index]);
                full_xyz[image.source_index] = std::array::from_fn(|i| {
                    ((aligned[i] - symmetry.origin_shift[i]) * f64::from(symmetry.hand))
                        .rem_euclid(1.)
                });
                full_u_cartesian[image.source_index] = conjugate(
                    image.cartesian_rotation,
                    self.evaluation.u_cartesian[site_index],
                );
            }
        }
        let summary = RefinementSummary {
            species: symmetry.sites.iter().map(|s| s.species.clone()).collect(),
            fractional_xyz: self.evaluation.fractional_xyz.clone(),
            u_cartesian: self.evaluation.u_cartesian.clone(),
            u_cif: self
                .evaluation
                .u_cartesian
                .iter()
                .map(|&u| tensor_cif(&symmetry.geometry, u))
                .collect(),
            full_u_cif: full_u_cartesian
                .iter()
                .map(|&u| tensor_cif(&symmetry.geometry, u))
                .collect(),
            u_equiv: self
                .evaluation
                .u_cartesian
                .iter()
                .map(|u| (u[0][0] + u[1][1] + u[2][2]) / 3.)
                .collect(),
            full_u_equiv: full_u_cartesian
                .iter()
                .map(|u| (u[0][0] + u[1][1] + u[2][2]) / 3.)
                .collect(),
            full_u_cartesian,
            hand: symmetry.hand,
            origin_shift: symmetry.origin_shift,
            maximum_matching_error: symmetry.maximum_matching_error,
            parameter_count: self.model.parameter_count,
            coordinate_parameter_count: symmetry
                .sites
                .iter()
                .map(|s| s.coordinate_basis.len())
                .sum(),
            origin_gauge_dimension: symmetry.origin_gauge_dimension,
            mode: if mode == AdpMode::Isotropic {
                "isotropic"
            } else {
                "anisotropic"
            },
            termination: self.termination,
            accepted_steps: self.accepted_steps,
            p1_r1,
            p1_objective,
        };
        (summary, full_xyz)
    }
}

#[allow(clippy::cast_precision_loss)] // Site stabilizer has at most 192 operations.
fn invariant_basis(stabilizer: &[Matrix3], mode: AdpMode) -> Vec<Matrix3> {
    if mode == AdpMode::Isotropic {
        return vec![IDENTITY.map(|r| r.map(|v| v / 3.0_f64.sqrt()))];
    }
    let components = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut basis = Vec::new();
    for (i, j) in components {
        let mut unit = [[0.; 3]; 3];
        unit[i][j] = if i == j { 1. } else { 1. / 2.0_f64.sqrt() };
        unit[j][i] = unit[i][j];
        let mut projected = [[0.; 3]; 3];
        for &q in stabilizer {
            matrix_add(
                &mut projected,
                conjugate(q, unit).map(|r| r.map(|v| v / stabilizer.len() as f64)),
            );
        }
        for &previous in &basis {
            let coefficient = frobenius(projected, previous);
            for a in 0..3 {
                for b in 0..3 {
                    projected[a][b] -= coefficient * previous[a][b];
                }
            }
        }
        let norm = frobenius(projected, projected).sqrt();
        if norm > 1e-9 {
            basis.push(projected.map(|r| r.map(|v| v / norm)));
        }
    }
    basis
}

impl PreparedSymmetricModel {
    /// Prepare site-symmetry constraints and non-resonant scattering amplitudes.
    ///
    /// # Errors
    /// Rejects invalid/empty observations, bad scattering species or a dense
    /// residual Jacobian exceeding two million elements.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        symmetry: SymmetryModel,
        hkl: &[[i32; 3]],
        observed: &[f64],
        mode: AdpMode,
    ) -> Result<Self, SolutionError> {
        if hkl.is_empty()
            || hkl.len() != observed.len()
            || hkl.len() > 100_000
            || observed
                .iter()
                .any(|&v| !v.is_finite() || v < 0. || v > 1e100)
            || !observed.iter().any(|&v| v > 0.)
        {
            return Err(error("invalid symmetric refinement observations"));
        }
        let tensor_bases: Vec<_> = symmetry
            .sites
            .iter()
            .map(|site| invariant_basis(&site.stabilizer, mode))
            .collect();
        let mut count = 0;
        let mut offsets = Vec::new();
        for (site, basis) in symmetry.sites.iter().zip(&tensor_bases) {
            let coordinate = count;
            count += site.coordinate_basis.len();
            let tensor = count;
            count += basis.len();
            offsets.push((coordinate, tensor));
        }
        if count == 0 || count * hkl.len() > 2_000_000 {
            return Err(error(
                "symmetric refinement exceeds the dense Jacobian memory budget",
            ));
        }
        let reciprocal: Vec<_> = hkl
            .iter()
            .map(|h| mv(symmetry.geometry.reciprocal_basis, h.map(f64::from)))
            .collect();
        let s: Vec<_> = reciprocal.iter().map(|&q| dot(q, q).sqrt() / 2.).collect();
        let factors = PreparedXrayScattering::new(symmetry.sites.iter().map(|site| &site.species))
            .and_then(|p| p.evaluate(&s))
            .map_err(|e| error(e.to_string()))?
            .real;
        Ok(Self {
            symmetry,
            tensor_bases,
            offsets,
            parameter_count: count,
            hkl: hkl.to_vec(),
            observed: observed.to_vec(),
            factors,
            reciprocal,
        })
    }

    /// Initialize zero coordinate motion and an isotropic physical U.
    ///
    /// # Errors
    /// Rejects an initial U outside the strictly positive physical interval.
    pub fn initial_parameters(&self, u_iso: f64) -> Result<Vec<f64>, SolutionError> {
        if !u_iso.is_finite() || !(1e-5..=0.2).contains(&u_iso) {
            return Err(error("initial U must lie in [1e-5,0.2]"));
        }
        let h = IDENTITY.map(|r| r.map(|v| v * u_iso.ln()));
        let mut parameters = vec![0.; self.parameter_count];
        for (site, basis) in self.tensor_bases.iter().enumerate() {
            for (i, &b) in basis.iter().enumerate() {
                parameters[self.offsets[site].1 + i] = frobenius(h, b);
            }
        }
        Ok(parameters)
    }

    /// Evaluate values and analytical coordinate/log-tensor derivatives together.
    ///
    /// # Errors
    /// Rejects nonfinite/mismatched parameters, ADPs outside the eigenvalue
    /// bounds, a changed special-position topology, or a zero calculated norm.
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, parameters: &[f64]) -> Result<SymmetricEvaluation, SolutionError> {
        if parameters.len() != self.parameter_count || parameters.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid symmetric parameter vector"));
        }
        let mut positions = Vec::new();
        let mut tensors = Vec::new();
        let mut tensor_derivatives = Vec::new();
        for (index, site) in self.symmetry.sites.iter().enumerate() {
            let mut position = site.fractional_xyz;
            for (i, &b) in site.coordinate_basis.iter().enumerate() {
                position = add(position, scaled(b, parameters[self.offsets[index].0 + i]));
            }
            let expansion = self
                .symmetry
                .group
                .expand_sites(&[position], 1e-8)
                .map_err(|e| error(e.to_string()))?;
            if expansion.fractional_xyz.len() != site.images.len() {
                return Err(error(
                    "coordinate trial changes special-position multiplicity",
                ));
            }
            positions.push(position);
            let mut h = [[0.; 3]; 3];
            for (i, &b) in self.tensor_bases[index].iter().enumerate() {
                matrix_add(
                    &mut h,
                    b.map(|r| r.map(|v| v * parameters[self.offsets[index].1 + i])),
                );
            }
            let (u, derivatives) = exponential(h, &self.tensor_bases[index])?;
            tensors.push(u);
            tensor_derivatives.push(derivatives);
        }
        let reflection_count = self.hkl.len();
        let site_count = self.symmetry.sites.len();
        let mut amplitudes = vec![0.; reflection_count];
        let mut jacobian = vec![0.; self.parameter_count * reflection_count];
        let mut dre = vec![0.; self.parameter_count];
        let mut dim = vec![0.; self.parameter_count];
        for (reflection, (&hkl, &q)) in self.hkl.iter().zip(&self.reciprocal).enumerate() {
            dre.fill(0.);
            dim.fill(0.);
            let mut real = 0.;
            let mut imaginary = 0.;
            for (site_index, site) in self.symmetry.sites.iter().enumerate() {
                let factor = self.factors[reflection * site_count + site_index];
                for image in &site.images {
                    let position = add(
                        mv(
                            image.operation.rotation().map(|r| r.map(f64::from)),
                            positions[site_index],
                        ),
                        image
                            .operation
                            .translation()
                            .map(phasesmith_crystallography::Rational::as_f64),
                    );
                    let phase = std::f64::consts::TAU * dot(hkl.map(f64::from), position);
                    let vector = mv(transpose(image.cartesian_rotation), q);
                    let displacement = (-2.
                        * std::f64::consts::PI.powi(2)
                        * dot(vector, mv(tensors[site_index], vector)))
                    .exp();
                    let (sin, cos) = phase.sin_cos();
                    let re = factor * displacement * cos;
                    let im = factor * displacement * sin;
                    real += re;
                    imaginary += im;
                    for (i, &basis) in site.coordinate_basis.iter().enumerate() {
                        let derivative = std::f64::consts::TAU
                            * dot(
                                hkl.map(f64::from),
                                mv(image.operation.rotation().map(|r| r.map(f64::from)), basis),
                            );
                        let parameter = self.offsets[site_index].0 + i;
                        dre[parameter] -= derivative * im;
                        dim[parameter] += derivative * re;
                    }
                    for (i, &derivative) in tensor_derivatives[site_index].iter().enumerate() {
                        let value = -2.
                            * std::f64::consts::PI.powi(2)
                            * dot(vector, mv(derivative, vector));
                        let parameter = self.offsets[site_index].1 + i;
                        dre[parameter] += value * re;
                        dim[parameter] += value * im;
                    }
                }
            }
            let amplitude = real.hypot(imaginary);
            amplitudes[reflection] = amplitude;
            if amplitude > 1e-14 {
                for i in 0..self.parameter_count {
                    jacobian[i * reflection_count + reflection] =
                        (real * dre[i] + imaginary * dim[i]) / amplitude;
                }
            }
        }
        let denominator: f64 = amplitudes.iter().map(|a| a * a).sum();
        let norm = self.observed.iter().map(|a| a * a).sum::<f64>().sqrt();
        if denominator <= 1e-100 || !denominator.is_finite() {
            return Err(error("zero/nonfinite symmetric calculated amplitude norm"));
        }
        let scale = amplitudes
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| a * o)
            .sum::<f64>()
            / denominator;
        let residual: Vec<_> = amplitudes
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| (scale * a - o) / norm)
            .collect();
        for row in jacobian.chunks_exact_mut(reflection_count) {
            let dk = row
                .iter()
                .zip(&self.observed)
                .zip(&amplitudes)
                .map(|((&d, &o), &a)| d * (o - 2. * scale * a))
                .sum::<f64>()
                / denominator;
            for (d, &a) in row.iter_mut().zip(&amplitudes) {
                *d = (scale * *d + a * dk) / norm;
            }
        }
        let r1 = amplitudes
            .iter()
            .zip(&self.observed)
            .map(|(a, o)| (scale * a - o).abs())
            .sum::<f64>()
            / self.observed.iter().sum::<f64>();
        if !r1.is_finite() || !jacobian.iter().all(|v| v.is_finite()) {
            return Err(error("nonfinite symmetric objective"));
        }
        Ok(SymmetricEvaluation {
            objective: residual.iter().map(|r| r * r).sum(),
            r1,
            amplitude_scale: scale,
            residual,
            jacobian,
            fractional_xyz: positions
                .iter()
                .map(|p| p.map(|v| v.rem_euclid(1.)))
                .collect(),
            u_cartesian: tensors,
        })
    }

    /// Bounded damped Gauss–Newton polishing, retaining improving accepted states.
    ///
    /// # Errors
    /// Rejects invalid controls or an invalid initial parameter vector.
    #[allow(clippy::too_many_lines)]
    pub fn refine(
        self,
        initial: &[f64],
        iterations: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
    ) -> Result<SymmetricRefinement, SolutionError> {
        if iterations > 200 {
            return Err(error("symmetric iteration budget must not exceed 200"));
        }
        let mut parameters = initial.to_vec();
        let mut evaluation = self.evaluate(&parameters)?;
        let count = self.parameter_count;
        let reflections = self.hkl.len();
        let mut damping = 1e-3_f64;
        let mut accepted = 0;
        let mut termination = "max_iterations";
        for _ in 0..iterations {
            if crate::stopped(deadline, cancellation) {
                termination = if cancellation.is_some_and(Cancellation::is_cancelled) {
                    "cancelled"
                } else {
                    "max_runtime"
                };
                break;
            }
            let mut normal = vec![0.; count * count];
            let mut gradient = vec![0.; count];
            for i in 0..count {
                let row = &evaluation.jacobian[i * reflections..(i + 1) * reflections];
                gradient[i] = row
                    .iter()
                    .zip(&evaluation.residual)
                    .map(|(a, b)| a * b)
                    .sum();
                for j in 0..=i {
                    let value = row
                        .iter()
                        .zip(&evaluation.jacobian[j * reflections..(j + 1) * reflections])
                        .map(|(a, b)| a * b)
                        .sum();
                    normal[i * count + j] = value;
                    normal[j * count + i] = value;
                }
            }
            if gradient.iter().fold(0.0_f64, |a, b| a.max(b.abs())) < 1e-9 {
                termination = "gradient";
                break;
            }
            for i in 0..count {
                normal[i * count + i] += damping * normal[i * count + i].max(1e-8);
                gradient[i] = -gradient[i];
            }
            let Some(mut step) = crate::model::cholesky_solve(&normal, &gradient) else {
                damping *= 10.;
                continue;
            };
            let mut factor = 1.0_f64;
            for (site_index, site) in self.symmetry.sites.iter().enumerate() {
                for i in 0..site.coordinate_basis.len() {
                    factor =
                        factor.min(0.15 / step[self.offsets[site_index].0 + i].abs().max(1e-30));
                }
                for i in 0..self.tensor_bases[site_index].len() {
                    factor =
                        factor.min(0.5 / step[self.offsets[site_index].1 + i].abs().max(1e-30));
                }
            }
            for v in &mut step {
                *v *= factor;
            }
            let mut improved = false;
            for backtrack in 0_i32..12 {
                if crate::stopped(deadline, cancellation) {
                    break;
                }
                let alpha = 0.5_f64.powi(backtrack);
                let trial: Vec<_> = parameters
                    .iter()
                    .zip(&step)
                    .map(|(a, b)| a + alpha * b)
                    .collect();
                if let Ok(next) = self.evaluate(&trial) {
                    if next.objective < evaluation.objective {
                        parameters = trial;
                        evaluation = next;
                        accepted += 1;
                        improved = true;
                        damping = (damping * 0.3).max(1e-10);
                        break;
                    }
                }
            }
            if !improved {
                damping *= 10.;
                if damping > 1e10 {
                    termination = "stagnated";
                    break;
                }
            }
        }
        if crate::stopped(deadline, cancellation) {
            termination = if cancellation.is_some_and(Cancellation::is_cancelled) {
                "cancelled"
            } else {
                "max_runtime"
            };
        }
        Ok(SymmetricRefinement {
            model: self,
            parameters,
            evaluation,
            accepted_steps: accepted,
            termination,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cif_conversion_preserves_nonorthogonal_anisotropic_exponent() {
        use phasesmith_crystallography::UnitCell;
        let geometry = UnitCell {
            a_angstrom: 4.7,
            b_angstrom: 5.3,
            c_angstrom: 6.9,
            alpha_deg: 79.,
            beta_deg: 98.,
            gamma_deg: 113.,
        }
        .geometry()
        .unwrap();
        let u = [
            [0.012, 0.002, -0.001],
            [0.002, 0.019, 0.003],
            [-0.001, 0.003, 0.023],
        ];
        let cif = tensor_cif(&geometry, u);
        let matrix = [
            [cif[0], cif[5], cif[4]],
            [cif[5], cif[1], cif[3]],
            [cif[4], cif[3], cif[2]],
        ];
        let h = [2., -3., 4.];
        let q = mv(geometry.reciprocal_basis, h);
        let reciprocal_lengths =
            std::array::from_fn(|i| h[i] * geometry.reciprocal_metric[i][i].sqrt());
        assert!(
            (dot(q, mv(u, q)) - dot(reciprocal_lengths, mv(matrix, reciprocal_lengths))).abs()
                < 1e-14
        );
        let isotropic = tensor_cif(&geometry, IDENTITY.map(|r| r.map(|v| v * 0.01)));
        for value in &isotropic[..3] {
            assert!((value - 0.01).abs() < 1e-15);
        }
        assert!(isotropic[5].abs() > 1e-3);
    }
}
