//! Symmetry-allowed log-SPD displacement tensors and fused analytical derivatives.

use crate::solution::math3::{
    IDENTITY, add, conjugate, dot, exponential, frobenius, matrix_add, mv, scaled, transpose,
};
use crate::solution::symmetry_model::SymmetryModel;
use crate::solution::{Cancellation, SolutionError, error};
use phasesmith_crystallography::{Matrix3, PreparedXrayScattering};
use std::time::Instant;
#[path = "isotropic_evaluation.rs"]
mod isotropic;

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
    geometry: Option<GeometryCache>,
    mode: AdpMode,
    families: Option<crate::solution::refinement_families::ReflectionFamilies>,
    specialized_isotropic_evaluation: bool,
    isotropic_weights: Option<isotropic::WeightPlan>,
    blocked_normal_matrices: bool,
    assignment_convergence_stopping: bool,
    dense_memory_budget: crate::solution::DenseMemoryBudget,
}

// Cached reciprocal/image geometry, in reflection/site/image order.
#[derive(Clone, Debug)]
struct ImageGeometry {
    phase: f64,
    // Free sites: coordinate phase derivatives. Fixed sites: unused lanes hold
    // sin(phase), cos(phase), 0. The owning site's unchanged basis guards the tag.
    // This keeps the existing 80-byte image entry and 32 MiB cache admission.
    coordinate_phase: [f64; 3],
    tensor_weight: [f64; 6],
}
impl ImageGeometry {
    fn sin_cos(&self, phase: f64, coordinate_count: usize) -> (f64, f64) {
        if coordinate_count == 0 {
            (self.coordinate_phase[0], self.coordinate_phase[1])
        } else {
            phase.sin_cos()
        }
    }
}
#[derive(Clone, Debug)]
struct SiteGeometryKey {
    position: [f64; 3],
    tangents: Vec<[f64; 3]>,
    images: Vec<(phasesmith_crystallography::SymmetryOperation, Matrix3)>,
    orbit_count: Option<usize>,
}
#[derive(Clone, Debug)]
struct GeometryCache {
    entries: Vec<ImageGeometry>,
    sites: Vec<SiteGeometryKey>,
    full_count: usize,
    operations: Vec<phasesmith_crystallography::SymmetryOperation>,
}
impl GeometryCache {
    // Exact identity checks guard caches when public Rust geometry is modified.
    #[allow(clippy::float_cmp)]
    fn matches(&self, symmetry: &SymmetryModel) -> bool {
        self.full_count == symmetry.full_count
            && self.operations == symmetry.group.operations()
            && self.sites.len() == symmetry.sites.len()
            && self.sites.iter().zip(&symmetry.sites).all(|(key, site)| {
                key.position == site.fractional_xyz
                    && key.tangents == site.coordinate_basis
                    && key.images.len() == site.images.len()
                    && key
                        .images
                        .iter()
                        .zip(&site.images)
                        .all(|((operation, rotation), image)| {
                            *operation == image.operation && *rotation == image.cartesian_rotation
                        })
            })
    }

    #[allow(clippy::float_cmp)] // Exact reuse only; changed trials run the original topology check.
    fn orbit_count(&self, site: usize, position: [f64; 3]) -> Option<usize> {
        (self.sites[site].position == position)
            .then_some(self.sites[site].orbit_count)
            .flatten()
    }
}
#[allow(clippy::float_cmp)]
fn canonical_isotropic(basis: &[Matrix3]) -> bool {
    basis.len() == 1 && basis[0] == IDENTITY.map(|r| r.map(|v| v / 3.0_f64.sqrt()))
}

const GEOMETRY_CACHE_BYTES: usize = 32 * 1024 * 1024;

fn tensor_contraction(w: [f64; 6], u: Matrix3) -> f64 {
    w[0] * u[0][0]
        + w[1] * u[1][1]
        + w[2] * u[2][2]
        + w[3] * (u[1][2] + u[2][1])
        + w[4] * (u[0][2] + u[2][0])
        + w[5] * (u[0][1] + u[1][0])
}

fn prepare_geometry(
    symmetry: &SymmetryModel,
    hkl: &[[i32; 3]],
    reciprocal: &[[f64; 3]],
    budget: usize,
) -> Option<GeometryCache> {
    let size = hkl.len().checked_mul(symmetry.full_count)?;
    if size.checked_mul(std::mem::size_of::<ImageGeometry>())? > budget {
        return None;
    }
    let mut cache = Vec::with_capacity(size);
    let scale = -2. * std::f64::consts::PI.powi(2);
    for (&h, &q) in hkl.iter().zip(reciprocal) {
        for site in &symmetry.sites {
            for image in &site.images {
                let rotation = image.operation.rotation().map(|r| r.map(f64::from));
                let position = add(
                    mv(rotation, site.fractional_xyz),
                    image
                        .operation
                        .translation()
                        .map(phasesmith_crystallography::Rational::as_f64),
                );
                let mut coordinate_phase = [0.; 3];
                for (i, &b) in site.coordinate_basis.iter().enumerate() {
                    coordinate_phase[i] =
                        std::f64::consts::TAU * dot(h.map(f64::from), mv(rotation, b));
                }
                let phase = std::f64::consts::TAU * dot(h.map(f64::from), position);
                if site.coordinate_basis.is_empty() {
                    let (sin, cos) = phase.sin_cos();
                    coordinate_phase = [sin, cos, 0.];
                }
                let v = mv(transpose(image.cartesian_rotation), q);
                cache.push(ImageGeometry {
                    phase,
                    coordinate_phase,
                    tensor_weight: [
                        v[0] * v[0],
                        v[1] * v[1],
                        v[2] * v[2],
                        v[1] * v[2],
                        v[0] * v[2],
                        v[0] * v[1],
                    ]
                    .map(|x| scale * x),
                });
            }
        }
    }
    Some(GeometryCache {
        entries: cache,
        full_count: symmetry.full_count,
        operations: symmetry.group.operations().to_vec(),
        sites: symmetry
            .sites
            .iter()
            .map(|site| SiteGeometryKey {
                orbit_count: symmetry
                    .group
                    .expand_sites(&[site.fractional_xyz], 1e-8)
                    .ok()
                    .map(|e| e.fractional_xyz.len()),
                position: site.fractional_xyz,
                tangents: site.coordinate_basis.clone(),
                images: site
                    .images
                    .iter()
                    .map(|image| (image.operation, image.cartesian_rotation))
                    .collect(),
            })
            .collect(),
    })
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
    /// Actual linear steps accepted from the bounded normal cache.
    pub cached_normal_solves: usize,
    /// Cache attempts that required the original matrix-free solver.
    pub normal_cache_fallbacks: usize,
    /// Complete bounded normal-matrix preparations in the reuse path.
    pub normal_matrix_builds: usize,
    /// Attempts using a retained matrix after a rejected step.
    pub normal_matrix_reuses: usize,
    /// Normal matrices formed by the blocked backend.
    pub blocked_normal_builds: usize,
    /// Blocked systems replaced by the legacy Gram after a failed checked solve.
    pub blocked_normal_fallbacks: usize,
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
        Self::new_with_geometry_cache(symmetry, hkl, observed, mode, true)
    }

    /// Prepare the same model with an explicit cached/direct geometry choice.
    /// # Errors
    /// Applies the same input and dense-Jacobian bounds as `new`.
    #[allow(clippy::cast_precision_loss)]
    pub fn new_with_geometry_cache(
        symmetry: SymmetryModel,
        hkl: &[[i32; 3]],
        observed: &[f64],
        mode: AdpMode,
        cache_geometry: bool,
    ) -> Result<Self, SolutionError> {
        Self::new_with_family_weighting(symmetry, hkl, observed, mode, cache_geometry, false)
    }

    /// Evaluate equivalent supplied rows once with multiplicity weights when requested.
    /// Original input/Jacobian admissibility limits apply before reduction.
    /// # Errors
    /// Also rejects nonidentical amplitudes within a reduced symmetry/Friedel family.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn new_with_family_weighting(
        symmetry: SymmetryModel,
        hkl: &[[i32; 3]],
        observed: &[f64],
        mode: AdpMode,
        cache_geometry: bool,
        weight_families: bool,
    ) -> Result<Self, SolutionError> {
        Self::new_with_memory_budget(
            symmetry,
            hkl,
            observed,
            mode,
            cache_geometry,
            weight_families,
            crate::solution::DenseMemoryBudget::default(),
        )
    }

    /// Prepare the identical model with an explicit per-dense-buffer byte bound.
    /// Original-row admission is checked before optional family reduction.
    /// # Errors
    /// Applies the same data/symmetry validation and rejects excess dense allocations.
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        clippy::too_many_arguments
    )]
    pub fn new_with_memory_budget(
        symmetry: SymmetryModel,
        hkl: &[[i32; 3]],
        observed: &[f64],
        mode: AdpMode,
        cache_geometry: bool,
        weight_families: bool,
        dense_memory_budget: crate::solution::DenseMemoryBudget,
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
        if symmetry.full_count != symmetry.sites.iter().map(|s| s.images.len()).sum::<usize>()
            || symmetry.sites.iter().any(|s| s.coordinate_basis.len() > 3)
        {
            return Err(error("inconsistent symmetric image/coordinate topology"));
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
        if count == 0 {
            return Err(error(
                "symmetric refinement exceeds the dense Jacobian memory budget",
            ));
        }
        dense_memory_budget.check(count, hkl.len()).map_err(|e| {
            error(format!(
                "symmetric refinement exceeds the dense Jacobian memory budget: {e}"
            ))
        })?;
        let families = weight_families
            .then(|| {
                crate::solution::refinement_families::ReflectionFamilies::new(
                    &symmetry.group,
                    hkl,
                    observed,
                )
            })
            .transpose()?;
        let reduced_hkl;
        let reduced_observed;
        let (hkl, observed) = if let Some(plan) = &families {
            reduced_hkl = plan
                .representatives
                .iter()
                .map(|&i| hkl[i])
                .collect::<Vec<_>>();
            reduced_observed = plan
                .representatives
                .iter()
                .zip(&plan.square_roots)
                .map(|(&i, &w)| observed[i] * w)
                .collect::<Vec<_>>();
            (reduced_hkl.as_slice(), reduced_observed.as_slice())
        } else {
            (hkl, observed)
        };
        let reciprocal: Vec<_> = hkl
            .iter()
            .map(|h| mv(symmetry.geometry.reciprocal_basis, h.map(f64::from)))
            .collect();
        let s: Vec<_> = reciprocal.iter().map(|&q| dot(q, q).sqrt() / 2.).collect();
        let factors = PreparedXrayScattering::new(symmetry.sites.iter().map(|site| &site.species))
            .and_then(|p| p.evaluate(&s))
            .map_err(|e| error(e.to_string()))?
            .real;
        let geometry = cache_geometry
            .then(|| prepare_geometry(&symmetry, hkl, &reciprocal, GEOMETRY_CACHE_BYTES))
            .flatten();
        Ok(Self {
            dense_memory_budget,
            families,
            specialized_isotropic_evaluation: false,
            isotropic_weights: None,
            blocked_normal_matrices: false,
            assignment_convergence_stopping: false,
            mode,
            geometry,
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

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn evaluate_rows(
        &self,
        parameters: &[f64],
        positions: &[[f64; 3]],
        tensors: &[Matrix3],
        tensor_derivatives: &[Vec<Matrix3>],
        geometry: Option<&GeometryCache>,
        isotropic_displacements: Option<&[f64]>,
        start: usize,
        amplitudes: &mut [f64],
        jacobian: &mut [f64],
    ) {
        let reflection_count = amplitudes.len();
        let site_count = self.symmetry.sites.len();
        if let Some(block_geometry) = geometry.filter(|_| isotropic::eligible(self, geometry)) {
            isotropic::evaluate(
                self,
                start,
                parameters,
                tensors,
                tensor_derivatives,
                block_geometry,
                isotropic_displacements,
                amplitudes,
                jacobian,
            );
        } else {
            let mut dre = vec![0.; self.parameter_count];
            let mut dim = vec![0.; self.parameter_count];
            for (local, reflection) in (start..start + reflection_count).enumerate() {
                let hkl = self.hkl[reflection];
                let q = self.reciprocal[reflection];
                dre.fill(0.);
                dim.fill(0.);
                let mut real = 0.;
                let mut imaginary = 0.;
                let mut image_index = reflection * self.symmetry.full_count;
                for (site_index, site) in self.symmetry.sites.iter().enumerate() {
                    let factor = self.factors[reflection * site_count + site_index];
                    // For U=uI every orthogonal symmetry image has the same exponent.
                    let isotropic = (self.mode == AdpMode::Isotropic
                        && geometry.is_some()
                        && canonical_isotropic(&self.tensor_bases[site_index]))
                    .then(|| {
                        let weight = -2. * std::f64::consts::PI.powi(2) * dot(q, q);
                        (
                            (weight * tensors[site_index][0][0]).exp(),
                            weight * tensor_derivatives[site_index][0][0][0],
                        )
                    });
                    for image in &site.images {
                        let cached = geometry.map(|g| &g.entries[image_index]);
                        image_index += 1;
                        let (phase, displacement, vector) = if let Some(g) = cached {
                            let mut phase = g.phase;
                            for i in 0..site.coordinate_basis.len() {
                                phase += g.coordinate_phase[i]
                                    * parameters[self.offsets[site_index].0 + i];
                            }
                            (
                                phase,
                                isotropic.map_or_else(
                                    || {
                                        tensor_contraction(g.tensor_weight, tensors[site_index])
                                            .exp()
                                    },
                                    |v| v.0,
                                ),
                                [0.; 3],
                            )
                        } else {
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
                            (phase, displacement, vector)
                        };
                        let (sin, cos) = cached.map_or_else(
                            || phase.sin_cos(),
                            |g| g.sin_cos(phase, site.coordinate_basis.len()),
                        );
                        let re = factor * displacement * cos;
                        let im = factor * displacement * sin;
                        real += re;
                        imaginary += im;
                        for (i, &basis) in site.coordinate_basis.iter().enumerate() {
                            let derivative = cached.map_or_else(
                                || {
                                    std::f64::consts::TAU
                                        * dot(
                                            hkl.map(f64::from),
                                            mv(
                                                image
                                                    .operation
                                                    .rotation()
                                                    .map(|r| r.map(f64::from)),
                                                basis,
                                            ),
                                        )
                                },
                                |g| g.coordinate_phase[i],
                            );
                            let parameter = self.offsets[site_index].0 + i;
                            dre[parameter] -= derivative * im;
                            dim[parameter] += derivative * re;
                        }
                        for (i, &derivative) in tensor_derivatives[site_index].iter().enumerate() {
                            let value = isotropic.map_or_else(
                                || {
                                    cached.map_or_else(
                                        || {
                                            -2. * std::f64::consts::PI.powi(2)
                                                * dot(vector, mv(derivative, vector))
                                        },
                                        |g| tensor_contraction(g.tensor_weight, derivative),
                                    )
                                },
                                |v| v.1,
                            );
                            let parameter = self.offsets[site_index].1 + i;
                            dre[parameter] += value * re;
                            dim[parameter] += value * im;
                        }
                    }
                }
                let amplitude = real.hypot(imaginary);
                let weight = self
                    .families
                    .as_ref()
                    .map_or(1., |p| p.square_roots[reflection]);
                amplitudes[local] = amplitude * weight;
                if amplitude > 1e-14 {
                    for i in 0..self.parameter_count {
                        jacobian[i * reflection_count + local] =
                            ((real * dre[i] + imaginary * dim[i]) / amplitude) * weight;
                    }
                }
            }
        }
    }

    /// Enable canonical isotropic cached evaluation with generic fallback.
    #[must_use]
    pub fn with_specialized_isotropic_evaluation(mut self, enabled: bool) -> Self {
        self.specialized_isotropic_evaluation = enabled;
        self.isotropic_weights =
            if enabled && self.mode == AdpMode::Isotropic && self.geometry.is_some() {
                isotropic::WeightPlan::new(&self.reciprocal)
            } else {
                None
            };
        self
    }
    /// Stop short assignment fits after repeated negligible accepted improvements.
    #[must_use]
    pub fn with_assignment_convergence_stopping(mut self, enabled: bool) -> Self {
        self.assignment_convergence_stopping = enabled;
        self
    }

    /// Use blocked Gram formation in eligible reused normal solves.
    #[must_use]
    pub fn with_blocked_normal_matrices(mut self, enabled: bool) -> Self {
        self.blocked_normal_matrices = enabled;
        self
    }

    /// Whether current geometry and tensor bases permit isotropic specialization.
    #[must_use]
    pub fn uses_specialized_isotropic_evaluation(&self) -> bool {
        isotropic::eligible(
            self,
            self.geometry.as_ref().filter(|g| g.matches(&self.symmetry)),
        )
    }
    /// Number of supplied rows before optional family reduction.
    #[must_use]
    pub fn input_reflection_count(&self) -> usize {
        self.families
            .as_ref()
            .map_or(self.hkl.len(), |p| p.input_rows)
    }
    /// Number of reflection rows evaluated per value/derivative pass.
    #[must_use]
    pub fn evaluated_reflection_count(&self) -> usize {
        self.hkl.len()
    }
    /// Supplied-row representative indices when reduction is enabled.
    #[must_use]
    pub fn reflection_representatives(&self) -> Option<&[usize]> {
        self.families.as_ref().map(|p| p.representatives.as_slice())
    }
    /// Actual supplied multiplicities in representative order.
    #[must_use]
    pub fn reflection_multiplicities(&self) -> Option<&[usize]> {
        self.families.as_ref().map(|p| p.multiplicities.as_slice())
    }

    /// Allocated reciprocal/image geometry bytes, zero for direct fallback.
    #[must_use]
    pub fn geometry_cache_bytes(&self) -> usize {
        self.geometry.as_ref().map_or(0, |g| {
            g.entries.len() * std::mem::size_of::<ImageGeometry>()
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

    /// Embed independent isotropic site U values in the invariant log-tensor basis.
    /// # Errors
    /// Rejects mismatched, nonfinite or out-of-range U values.
    pub fn initial_site_parameters(&self, site_u: &[f64]) -> Result<Vec<f64>, SolutionError> {
        if site_u.len() != self.symmetry.sites.len()
            || site_u
                .iter()
                .any(|u| !u.is_finite() || !(1e-5..=0.2).contains(u))
        {
            return Err(error(
                "site U values must match asymmetric sites and lie in [1e-5,0.2]",
            ));
        }
        let mut parameters = vec![0.; self.parameter_count];
        for (site, (&u, basis)) in site_u.iter().zip(&self.tensor_bases).enumerate() {
            let h = IDENTITY.map(|r| r.map(|v| v * u.ln()));
            for (i, &b) in basis.iter().enumerate() {
                parameters[self.offsets[site].1 + i] = frobenius(h, b);
            }
        }
        Ok(parameters)
    }

    /// Embed physical Cartesian site tensors in the invariant log-SPD basis.
    /// # Errors
    /// Rejects invalid eigenvalues or tensors incompatible with the model symmetry/mode.
    pub fn initial_tensor_parameters(
        &self,
        tensors: &[Matrix3],
    ) -> Result<Vec<f64>, SolutionError> {
        if tensors.len() != self.symmetry.sites.len() {
            return Err(error("tensor seed length differs"));
        }
        let mut parameters = vec![0.; self.parameter_count];
        for (site, &tensor) in tensors.iter().enumerate() {
            if tensor.iter().flatten().any(|x| !x.is_finite())
                || (0..3).any(|i| (0..3).any(|j| (tensor[i][j] - tensor[j][i]).abs() > 1e-12))
            {
                return Err(error("invalid tensor seed"));
            }
            let (values, vectors) = crate::solution::math3::eigen(tensor);
            if values
                .iter()
                .any(|x| !(1e-5 - 1e-12..=0.2 + 1e-12).contains(x))
            {
                return Err(error("tensor seed eigenvalues outside bounds"));
            }
            let diagonal = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i == j {
                        values[i].clamp(1e-5, 0.2).ln()
                    } else {
                        0.
                    }
                })
            });
            let log = conjugate(vectors, diagonal);
            let mut reconstructed = [[0.; 3]; 3];
            for (i, &basis) in self.tensor_bases[site].iter().enumerate() {
                let value = frobenius(log, basis);
                parameters[self.offsets[site].1 + i] = value;
                matrix_add(&mut reconstructed, basis.map(|r| r.map(|x| x * value)));
            }
            if reconstructed
                .iter()
                .flatten()
                .zip(log.iter().flatten())
                .any(|(a, b)| (a - b).abs() > 1e-9)
            {
                return Err(error(
                    "tensor seed violates site symmetry/displacement mode",
                ));
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
        if self
            .families
            .as_ref()
            .is_some_and(|p| p.group.operations() != self.symmetry.group.operations())
        {
            return Err(error("multiplicity plan does not match the model symmetry"));
        }
        if parameters.len() != self.parameter_count || parameters.iter().any(|v| !v.is_finite()) {
            return Err(error("invalid symmetric parameter vector"));
        }
        let geometry = self.geometry.as_ref().filter(|g| g.matches(&self.symmetry));
        let mut positions = Vec::new();
        let mut tensors = Vec::new();
        let mut tensor_derivatives = Vec::new();
        for (index, site) in self.symmetry.sites.iter().enumerate() {
            let mut position = site.fractional_xyz;
            for (i, &b) in site.coordinate_basis.iter().enumerate() {
                position = add(position, scaled(b, parameters[self.offsets[index].0 + i]));
            }
            let orbit_count = geometry
                .and_then(|g| g.orbit_count(index, position))
                .map_or_else(
                    || {
                        self.symmetry
                            .group
                            .expand_sites(&[position], 1e-8)
                            .map(|e| e.fractional_xyz.len())
                            .map_err(|e| error(e.to_string()))
                    },
                    Ok,
                )?;
            if orbit_count != site.images.len() {
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
        let displacements = if isotropic::eligible(self, geometry) {
            self.isotropic_weights
                .as_ref()
                .map(|p| p.displacements(&tensors))
        } else {
            None
        };
        let mut amplitudes = vec![0.; reflection_count];
        let mut jacobian = vec![0.; self.parameter_count * reflection_count];
        let tasks = crate::solution::execution::current().map(|workers| {
            let tasks = workers
                .count()
                .min(reflection_count / 128)
                .min(reflection_count * self.symmetry.full_count / 8192);
            (workers, tasks)
        });
        if let Some((workers, tasks)) = tasks.filter(|(_, tasks)| *tasks >= 2) {
            let width = reflection_count.div_ceil(tasks);
            let tiles = workers.map(tasks, |i| {
                let start = i * width;
                let end = (start + width).min(reflection_count);
                let mut a = vec![0.; end - start];
                let mut j = vec![0.; self.parameter_count * (end - start)];
                self.evaluate_rows(
                    parameters,
                    &positions,
                    &tensors,
                    &tensor_derivatives,
                    geometry,
                    displacements.as_deref(),
                    start,
                    &mut a,
                    &mut j,
                );
                (start, a, j)
            })?;
            for (start, a, j) in tiles {
                amplitudes[start..start + a.len()].copy_from_slice(&a);
                for parameter in 0..self.parameter_count {
                    jacobian[parameter * reflection_count + start
                        ..parameter * reflection_count + start + a.len()]
                        .copy_from_slice(&j[parameter * a.len()..(parameter + 1) * a.len()]);
                }
            }
        } else {
            self.evaluate_rows(
                parameters,
                &positions,
                &tensors,
                &tensor_derivatives,
                geometry,
                displacements.as_deref(),
                0,
                &mut amplitudes,
                &mut jacobian,
            );
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
        let r1 = if let Some(plan) = &self.families {
            amplitudes
                .iter()
                .zip(&self.observed)
                .zip(&plan.square_roots)
                .map(|((&a, &o), &w)| w * (scale * a - o).abs())
                .sum::<f64>()
                / self
                    .observed
                    .iter()
                    .zip(&plan.square_roots)
                    .map(|(&o, &w)| o * w)
                    .sum::<f64>()
        } else {
            amplitudes
                .iter()
                .zip(&self.observed)
                .map(|(a, o)| (scale * a - o).abs())
                .sum::<f64>()
                / self.observed.iter().sum::<f64>()
        };
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
        self.refine_with_solver(initial, iterations, deadline, cancellation, false)
    }

    /// Damped polishing with optional diagonally scaled CG for 32 or more parameters.
    /// The original dense Jacobian budget and line search remain unchanged.
    /// # Errors
    /// Rejects the same invalid inputs as `refine`.
    #[allow(clippy::too_many_lines)]
    pub fn refine_with_solver(
        self,
        initial: &[f64],
        iterations: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        iterative: bool,
    ) -> Result<SymmetricRefinement, SolutionError> {
        self.refine_with_normal_cache(
            initial,
            iterations,
            deadline,
            cancellation,
            iterative,
            false,
        )
    }

    /// Optional bounded preconditioned normal cache with original-operator checks.
    /// All observations, derivatives, line searches and stopping gates are retained.
    /// # Errors
    /// Rejects the same invalid inputs as `refine_with_solver`.
    #[allow(clippy::too_many_lines)]
    pub fn refine_with_normal_cache(
        self,
        initial: &[f64],
        iterations: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        iterative: bool,
        cached_normal: bool,
    ) -> Result<SymmetricRefinement, SolutionError> {
        self.refine_with_normal_reuse(
            initial,
            iterations,
            deadline,
            cancellation,
            iterative,
            cached_normal,
            false,
        )
    }

    /// Reuse a checked normal linearization only while its accepted model is unchanged.
    /// All damping, line searches, residual checks and matrix-free fallback are retained.
    /// # Errors
    /// Rejects the same inputs and stopped/failed systems as `refine_with_normal_cache`.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools
    )]
    pub fn refine_with_normal_reuse(
        self,
        initial: &[f64],
        iterations: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        iterative: bool,
        cached_normal: bool,
        reuse_normal: bool,
    ) -> Result<SymmetricRefinement, SolutionError> {
        if iterations > 200 {
            return Err(error("symmetric iteration budget must not exceed 200"));
        }
        let evaluation = self.evaluate(initial)?;
        self.refine_with_initial_evaluation(
            initial,
            evaluation,
            iterations,
            deadline,
            cancellation,
            iterative,
            cached_normal,
            reuse_normal,
        )
    }

    // Trusted crate-internal reuse: the caller produced this evaluation from this
    // exact model/initial vector immediately before calling. Public entry points
    // always evaluate it themselves. Keeps initial-failure audit semantics intact.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::fn_params_excessive_bools
    )]
    pub(crate) fn refine_with_initial_evaluation(
        self,
        initial: &[f64],
        mut evaluation: SymmetricEvaluation,
        iterations: usize,
        deadline: Option<Instant>,
        cancellation: Option<&Cancellation>,
        iterative: bool,
        cached_normal: bool,
        reuse_normal: bool,
    ) -> Result<SymmetricRefinement, SolutionError> {
        if iterations > 200 {
            return Err(error("symmetric iteration budget must not exceed 200"));
        }
        debug_assert_eq!(initial.len(), self.parameter_count);
        debug_assert_eq!(
            evaluation.jacobian.len(),
            self.parameter_count * self.hkl.len()
        );
        let mut parameters = initial.to_vec();
        let count = self.parameter_count;
        let reflections = self.hkl.len();
        let mut damping = 1e-3_f64;
        let mut accepted = 0;
        let mut termination = "max_iterations";
        let mut cached_normal_solves = 0;
        let mut normal_cache_fallbacks = 0;
        let mut linear_cache = crate::solution::linear_step::NormalReuse::default();
        linear_cache.memory_budget = self.dense_memory_budget;
        linear_cache.blocked = self.blocked_normal_matrices;
        let mut small_improvements = 0;
        for iteration in 0..iterations {
            if crate::solution::stopped(deadline, cancellation) {
                termination = if cancellation.is_some_and(Cancellation::is_cancelled) {
                    "cancelled"
                } else {
                    "max_runtime"
                };
                break;
            }
            let use_iterative = iterative && count >= 32;
            let use_cached = cached_normal && iterative && count >= 12;
            let mut normal = vec![
                0.;
                if use_iterative || use_cached {
                    0
                } else {
                    count * count
                }
            ];
            let mut gradient = vec![0.; count];
            for i in 0..count {
                let row = &evaluation.jacobian[i * reflections..(i + 1) * reflections];
                gradient[i] = row
                    .iter()
                    .zip(&evaluation.residual)
                    .map(|(a, b)| a * b)
                    .sum();
                for j in 0..if use_iterative || use_cached {
                    0
                } else {
                    i + 1
                } {
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
                if !use_iterative && !use_cached {
                    normal[i * count + i] += damping * normal[i * count + i].max(1e-8);
                }
                gradient[i] = -gradient[i];
            }
            let solved = if use_cached {
                let linear = if use_iterative && reuse_normal {
                    linear_cache.step(
                        &evaluation.jacobian,
                        &evaluation.residual,
                        damping,
                        deadline,
                        cancellation,
                    )
                } else if use_iterative {
                    crate::solution::linear_step::cached_normal_step_with_memory_budget(
                        &evaluation.jacobian,
                        &evaluation.residual,
                        damping,
                        deadline,
                        cancellation,
                        self.dense_memory_budget,
                    )
                } else {
                    crate::solution::linear_step::cached_dense_step(
                        &evaluation.jacobian,
                        &evaluation.residual,
                        damping,
                        deadline,
                        cancellation,
                        self.dense_memory_budget,
                    )
                };
                linear.ok().map(|v| {
                    cached_normal_solves += usize::from(v.used_cache);
                    normal_cache_fallbacks += usize::from(v.fell_back);
                    v.step
                })
            } else if use_iterative {
                crate::solution::linear_step::iterative_normal_step_with_memory_budget(
                    &evaluation.jacobian,
                    &evaluation.residual,
                    damping,
                    deadline,
                    cancellation,
                    self.dense_memory_budget,
                )
                .ok()
                .map(|v| v.0)
            } else {
                crate::solution::model::cholesky_solve(&normal, &gradient)
            };
            let Some(mut step) = solved else {
                small_improvements = 0;
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
                if crate::solution::stopped(deadline, cancellation) {
                    break;
                }
                let alpha = 0.5_f64.powi(backtrack);
                let trial: Vec<_> = parameters
                    .iter()
                    .zip(&step)
                    .map(|(a, b)| a + alpha * b)
                    .collect();
                if let Ok(next) = self.evaluate(&trial)
                    && next.objective < evaluation.objective
                {
                    if self.assignment_convergence_stopping {
                        let small = (evaluation.objective - next.objective)
                            <= 1e-6 * evaluation.objective.max(1e-30)
                            && (next.r1 - evaluation.r1).abs() <= 1e-6;
                        small_improvements = if small { small_improvements + 1 } else { 0 };
                    }
                    parameters = trial;
                    linear_cache.clear();
                    evaluation = next;
                    accepted += 1;
                    improved = true;
                    damping = (damping * 0.3).max(1e-10);
                    break;
                }
            }
            if self.assignment_convergence_stopping
                && improved
                && iteration + 1 >= 4
                && iteration + 1 < iterations
                && small_improvements >= 2
            {
                termination = "assignment_converged";
                break;
            }
            if !improved {
                small_improvements = 0;
                damping *= 10.;
                if damping > 1e10 {
                    termination = "stagnated";
                    break;
                }
            }
        }
        if crate::solution::stopped(deadline, cancellation) {
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
            cached_normal_solves,
            normal_cache_fallbacks,
            normal_matrix_builds: linear_cache.builds,
            normal_matrix_reuses: linear_cache.reuses,
            blocked_normal_builds: linear_cache.blocked_builds,
            blocked_normal_fallbacks: linear_cache.blocked_fallbacks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_precision_loss)] // Deterministic test indices are below 2,000.
    fn mixed_fixed_free_model() -> (PreparedSymmetricModel, Vec<f64>) {
        use phasesmith_crystallography::{Rational, SpaceGroup, SymmetryOperation, UnitCell};
        let group = SpaceGroup::new(vec![
            SymmetryOperation::identity(),
            SymmetryOperation::new([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [Rational::zero(); 3])
                .unwrap(),
        ])
        .unwrap();
        let cell = UnitCell {
            a_angstrom: 7.,
            b_angstrom: 8.,
            c_angstrom: 9.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let points = [
            [0., 0., 0.],
            [0.5, 0.5, 0.5],
            [0.13, 0.26, 0.37],
            [0.87, 0.74, 0.63],
        ];
        let labels = ["C", "Si", "O", "O"].map(str::to_owned);
        let symmetry =
            crate::solution::recover_symmetry(cell, &group, &labels, &points, 0.01, None, None)
                .unwrap();
        let hkl: Vec<_> = (-5i32..=5)
            .flat_map(|h| {
                (-5..=5).flat_map(move |k| {
                    (-5..=5).filter_map(move |l| ([h, k, l] != [0; 3]).then_some([h, k, l]))
                })
            })
            .collect();
        let observed: Vec<_> = (0..hkl.len())
            .map(|i| 8. + (i as f64 * 0.17).sin())
            .collect();
        let model = PreparedSymmetricModel::new(symmetry, &hkl, &observed, AdpMode::Isotropic)
            .unwrap()
            .with_specialized_isotropic_evaluation(true);
        let start = model.initial_parameters(0.015).unwrap();
        (model, start)
    }
    fn same_evaluation(a: &SymmetricEvaluation, b: &SymmetricEvaluation) {
        assert_eq!(a.objective.to_bits(), b.objective.to_bits());
        assert_eq!(a.r1.to_bits(), b.r1.to_bits());
        assert_eq!(a.amplitude_scale.to_bits(), b.amplitude_scale.to_bits());
        assert_eq!(
            a.residual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.residual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.jacobian.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.jacobian.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    #[allow(clippy::cast_precision_loss)] // Deterministic test indices are below 2,000.
    fn repeated_weights_fixed_phases_and_topology_reuse_are_bitwise_exact() {
        let (model, start) = mixed_fixed_free_model();
        assert!(model.isotropic_weights.is_some());
        assert_eq!(std::mem::size_of::<ImageGeometry>(), 80);
        let mut scalar = model.clone();
        scalar.isotropic_weights = None;
        for site in &mut scalar.geometry.as_mut().unwrap().sites {
            site.orbit_count = None;
        }
        for i in 0..16 {
            let mut p = start.clone();
            for (j, v) in p.iter_mut().enumerate() {
                *v += 0.005 * ((i * 13 + j) as f64).sin();
            }
            same_evaluation(&model.evaluate(&p).unwrap(), &scalar.evaluate(&p).unwrap());
        }
        let geometry = model.geometry.as_ref().unwrap();
        for row in geometry.entries.chunks_exact(model.symmetry.full_count) {
            let mut offset = 0;
            for site in &model.symmetry.sites {
                if site.coordinate_basis.is_empty() {
                    for image in &row[offset..offset + site.images.len()] {
                        let actual = image.sin_cos(image.phase, 0);
                        let expected = image.phase.sin_cos();
                        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
                        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
                    }
                }
                offset += site.images.len();
            }
        }
        let baseline = model.evaluate(&start).unwrap();
        let a = model
            .clone()
            .refine_with_normal_reuse(&start, 3, None, None, true, true, true)
            .unwrap();
        let b = model
            .refine_with_initial_evaluation(&start, baseline, 3, None, None, true, true, true)
            .unwrap();
        assert_eq!(a.parameters, b.parameters);
        assert_eq!(a.termination, b.termination);
        assert_eq!(a.accepted_steps, b.accepted_steps);
        same_evaluation(&a.evaluation, &b.evaluation);
    }
    #[test]
    fn cached_topology_still_rejects_changed_special_positions_and_groups() {
        let (mut model, start) = mixed_fixed_free_model();
        let fixed = model
            .symmetry
            .sites
            .iter()
            .position(|s| s.coordinate_basis.is_empty())
            .unwrap();
        model.symmetry.sites[fixed].fractional_xyz[0] += 0.002;
        assert!(
            model
                .evaluate(&start)
                .unwrap_err()
                .to_string()
                .contains("multiplicity")
        );
        let (mut model, start) = mixed_fixed_free_model();
        model.symmetry.group = phasesmith_crystallography::SpaceGroup::new(vec![
            phasesmith_crystallography::SymmetryOperation::identity(),
        ])
        .unwrap();
        assert!(
            model
                .evaluate(&start)
                .unwrap_err()
                .to_string()
                .contains("multiplicity")
        );
        let (model, mut start) = mixed_fixed_free_model();
        let free = model
            .symmetry
            .sites
            .iter()
            .position(|s| s.coordinate_basis.len() == 3)
            .unwrap();
        let site = &model.symmetry.sites[free];
        for (i, b) in site.coordinate_basis.iter().enumerate() {
            let axis = b.iter().position(|v| v.abs() > 1e-8).unwrap();
            start[model.offsets[free].0 + i] = -site.fractional_xyz[axis] / b[axis];
        }
        assert!(
            model
                .evaluate(&start)
                .unwrap_err()
                .to_string()
                .contains("multiplicity")
        );
    }

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

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use phasesmith_crystallography::{SpaceGroup, SymmetryOperation, UnitCell};
    #[test]
    fn explicit_budget_prepares_jacobian_above_old_limit() {
        let cell = UnitCell {
            a_angstrom: 20.,
            b_angstrom: 21.,
            c_angstrom: 22.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let labels = vec!["O".into(); 16];
        let points: Vec<_> = (0..16)
            .map(|i| {
                let x = f64::from(i);
                [
                    (x * 0.137).fract(),
                    (x * 0.241).fract(),
                    (x * 0.319).fract(),
                ]
            })
            .collect();
        let symmetry =
            crate::solution::recover_symmetry(cell, &group, &labels, &points, 0.01, None, None)
                .unwrap();
        let hkl: Vec<_> = (0_i32..16000)
            .map(|i| [1 + i % 25, (i / 25) % 25, i / 625])
            .collect();
        let observed = vec![1.; hkl.len()];
        assert!(
            PreparedSymmetricModel::new(symmetry.clone(), &hkl, &observed, AdpMode::Anisotropic)
                .unwrap_err()
                .0
                .contains("dense memory budget")
        );
        let prepared = PreparedSymmetricModel::new_with_memory_budget(
            symmetry,
            &hkl,
            &observed,
            AdpMode::Anisotropic,
            false,
            false,
            crate::solution::DenseMemoryBudget::new(64 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        let out = prepared
            .evaluate(&prepared.initial_parameters(0.01).unwrap())
            .unwrap();
        assert!(out.jacobian.len() > 2_000_000);
        assert_eq!(out.jacobian.len(), prepared.parameter_count * hkl.len());
        assert!(out.jacobian.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn cache_budget_boundary_and_inconsistent_topology() {
        let cell = UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 100.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let labels = vec!["Si".into(), "O".into()];
        let mut symmetry = crate::solution::recover_symmetry(
            cell,
            &group,
            &labels,
            &[[0.137, 0.259, 0.382], [0.421, 0.148, 0.723]],
            0.01,
            None,
            None,
        )
        .unwrap();
        let hkl = [[1, 0, 0], [0, 1, 0]];
        let q: Vec<_> = hkl
            .iter()
            .map(|h| mv(symmetry.geometry.reciprocal_basis, h.map(f64::from)))
            .collect();
        let exact = 4 * std::mem::size_of::<ImageGeometry>();
        assert!(prepare_geometry(&symmetry, &hkl, &q, exact - 1).is_none());
        assert_eq!(
            prepare_geometry(&symmetry, &hkl, &q, exact)
                .unwrap()
                .entries
                .len(),
            4
        );
        symmetry.full_count += 1;
        assert!(
            PreparedSymmetricModel::new(symmetry, &hkl, &[1., 2.], AdpMode::Isotropic).is_err()
        );
    }
    #[test]
    fn changed_public_geometry_uses_direct_values_and_changed_tensor_basis_is_not_isotropic() {
        let cell = UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 100.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let labels = vec!["Si".into(), "O".into()];
        let symmetry = crate::solution::recover_symmetry(
            cell,
            &group,
            &labels,
            &[[0.137, 0.259, 0.382], [0.421, 0.148, 0.723]],
            0.01,
            None,
            None,
        )
        .unwrap();
        let hkl = [[1, 0, 0], [0, 1, 0], [1, 2, -1]];
        let observed = [1., 2., 3.];
        let mut cached =
            PreparedSymmetricModel::new(symmetry, &hkl, &observed, AdpMode::Isotropic).unwrap();
        cached = cached.with_specialized_isotropic_evaluation(true);
        assert!(cached.uses_specialized_isotropic_evaluation());
        let parameters = cached.initial_parameters(0.015).unwrap();
        cached.tensor_bases[1][0][0][1] = 0.02;
        cached.tensor_bases[1][0][1][0] = 0.02;
        assert!(!cached.uses_specialized_isotropic_evaluation());
        let mut direct = cached.clone();
        direct.geometry = None;
        let a = cached.evaluate(&parameters).unwrap();
        let b = direct.evaluate(&parameters).unwrap();
        assert!((a.objective - b.objective).abs() < 1e-13);
        for (x, y) in a.jacobian.iter().zip(&b.jacobian) {
            assert!((x - y).abs() < 1e-12);
        }
        cached.symmetry.sites[1].fractional_xyz[0] += 0.023;
        let mut direct = cached.clone();
        direct.geometry = None;
        let a = cached.evaluate(&parameters).unwrap();
        let b = direct.evaluate(&parameters).unwrap();
        assert_eq!(a.jacobian, b.jacobian);
        assert_eq!(a.residual, b.residual);
    }
}
