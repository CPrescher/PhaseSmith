//! Explicit experimental symmetry/ADP model boundary with owned detached inputs.

use super::{NativeCancellation, array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::{
    AdpMode, PreparedSymmetricModel, RefinementSummary, SymmetricEvaluation, SymmetricRefinement,
    recover_symmetry,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use std::time::{Duration, Instant};

pub(super) fn summary<'py>(py: Python<'py>, s: &RefinementSummary) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("species", s.species.clone())?;
    out.set_item("fractional_xyz", array2(py, s.fractional_xyz.clone()))?;
    out.set_item("u_cartesian", s.u_cartesian.clone())?;
    out.set_item("u_cif", s.u_cif.clone())?;
    out.set_item("u_equiv", s.u_equiv.clone())?;
    out.set_item("full_u_cartesian", s.full_u_cartesian.clone())?;
    out.set_item("full_u_cif", s.full_u_cif.clone())?;
    out.set_item("full_u_equiv", s.full_u_equiv.clone())?;
    out.set_item("hand", s.hand)?;
    out.set_item("origin_shift", s.origin_shift)?;
    out.set_item("maximum_matching_error", s.maximum_matching_error)?;
    out.set_item("parameter_count", s.parameter_count)?;
    out.set_item("coordinate_parameter_count", s.coordinate_parameter_count)?;
    out.set_item("origin_gauge_dimension", s.origin_gauge_dimension)?;
    out.set_item("mode", s.mode)?;
    out.set_item("termination", s.termination)?;
    out.set_item("accepted_steps", s.accepted_steps)?;
    out.set_item("p1_r1", s.p1_r1)?;
    out.set_item("p1_objective", s.p1_objective)?;
    Ok(out)
}

pub(super) fn evaluation<'py>(
    py: Python<'py>,
    state: &SymmetricEvaluation,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("objective", state.objective)?;
    out.set_item("r1", state.r1)?;
    out.set_item("amplitude_scale", state.amplitude_scale)?;
    out.set_item("fractional_xyz", array2(py, state.fractional_xyz.clone()))?;
    out.set_item("u_cartesian", state.u_cartesian.clone())?;
    out.set_item("residual", state.residual.clone().into_pyarray(py))?;
    out.set_item("jacobian", state.jacobian.clone().into_pyarray(py))?;
    Ok(out)
}

pub(super) fn metadata<'py>(
    py: Python<'py>,
    model: &PreparedSymmetricModel,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let symmetry = &model.symmetry;
    out.set_item("hand", symmetry.hand)?;
    out.set_item("origin_shift", symmetry.origin_shift)?;
    out.set_item("maximum_matching_error", symmetry.maximum_matching_error)?;
    out.set_item("origin_gauge_dimension", symmetry.origin_gauge_dimension)?;
    out.set_item("full_count", symmetry.full_count)?;
    out.set_item("parameter_count", model.parameter_count)?;
    out.set_item(
        "specialized_isotropic_evaluation",
        model.uses_specialized_isotropic_evaluation(),
    )?;
    out.set_item("geometry_cache_bytes", model.geometry_cache_bytes())?;
    out.set_item("input_reflection_count", model.input_reflection_count())?;
    out.set_item(
        "evaluated_reflection_count",
        model.evaluated_reflection_count(),
    )?;
    out.set_item(
        "reflection_representatives",
        model.reflection_representatives(),
    )?;
    out.set_item(
        "reflection_multiplicities",
        model.reflection_multiplicities(),
    )?;
    out.set_item("offsets", model.offsets.clone())?;
    out.set_item("tensor_bases", model.tensor_bases.clone())?;
    let sites = PyList::empty(py);
    for site in &symmetry.sites {
        let item = PyDict::new(py);
        item.set_item("species", &site.species)?;
        item.set_item("fractional_xyz", site.fractional_xyz)?;
        item.set_item("coordinate_basis", site.coordinate_basis.clone())?;
        item.set_item("stabilizer", site.stabilizer.clone())?;
        let images = PyList::empty(py);
        for image in &site.images {
            let record = PyDict::new(py);
            record.set_item("rotation", image.operation.rotation())?;
            record.set_item(
                "translation",
                image
                    .operation
                    .translation()
                    .map(phasesmith_engine::crystallography::Rational::as_f64),
            )?;
            record.set_item("cartesian_rotation", image.cartesian_rotation)?;
            record.set_item("source_index", image.source_index)?;
            images.append(record)?;
        }
        item.set_item("images", images)?;
        sites.append(item)?;
    }
    out.set_item("sites", sites)?;
    Ok(out)
}

pub(super) fn result<'py>(
    py: Python<'py>,
    state: &SymmetricRefinement,
) -> PyResult<Bound<'py, PyDict>> {
    let out = metadata(py, &state.model)?;
    out.set_item("evaluation", evaluation(py, &state.evaluation)?)?;
    out.set_item("parameters", state.parameters.clone().into_pyarray(py))?;
    out.set_item("accepted_steps", state.accepted_steps)?;
    out.set_item("cached_normal_solves", state.cached_normal_solves)?;
    out.set_item("normal_cache_fallbacks", state.normal_cache_fallbacks)?;
    out.set_item("normal_matrix_builds", state.normal_matrix_builds)?;
    out.set_item("normal_matrix_reuses", state.normal_matrix_reuses)?;
    out.set_item("blocked_normal_builds", state.blocked_normal_builds)?;
    out.set_item("blocked_normal_fallbacks", state.blocked_normal_fallbacks)?;
    out.set_item("termination", state.termination)?;
    Ok(out)
}

#[pyclass(name = "_SymmetricSolutionModel", frozen)]
struct NativeModel {
    inner: PreparedSymmetricModel,
}

#[pymethods]
impl NativeModel {
    #[new]
    #[pyo3(signature = (cell_parameters, group, species, xyz, hkl, observed, mode, tolerance, cache_geometry=true, weight_families=false, specialized_isotropic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        species: Vec<String>,
        xyz: PyReadonlyArray1<'_, f64>,
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        mode: String,
        tolerance: f64,
        cache_geometry: bool,
        weight_families: bool,
        specialized_isotropic: bool,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let coordinates = xyz.as_slice()?;
        if coordinates.len() != 3 * species.len() {
            return Err(invalid("coordinate/species lengths differ"));
        }
        let points: Vec<_> = coordinates
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let hkl = hkl_rows(&hkl)?;
        let observed = observed.as_slice()?.to_vec();
        let mode = match mode.as_str() {
            "isotropic" => AdpMode::Isotropic,
            "anisotropic" => AdpMode::Anisotropic,
            _ => return Err(invalid("unknown ADP mode")),
        };
        let inner = py
            .detach(move || {
                let symmetry = recover_symmetry(
                    cell,
                    &group,
                    &species,
                    &points,
                    tolerance,
                    Some(Instant::now() + Duration::from_secs(60)),
                    None,
                )?;
                PreparedSymmetricModel::new_with_family_weighting(
                    symmetry,
                    &hkl,
                    &observed,
                    mode,
                    cache_geometry,
                    weight_families,
                )
                .map(|p| p.with_specialized_isotropic_evaluation(specialized_isotropic))
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata(py, &self.inner)
    }
    fn initial_parameters<'py>(
        &self,
        py: Python<'py>,
        u_iso: f64,
    ) -> PyResult<Bound<'py, npy::PyArray1<f64>>> {
        Ok(self
            .inner
            .initial_parameters(u_iso)
            .map_err(invalid)?
            .into_pyarray(py))
    }
    fn initial_site_parameters<'py>(
        &self,
        py: Python<'py>,
        site_u: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, npy::PyArray1<f64>>> {
        Ok(self
            .inner
            .initial_site_parameters(site_u.as_slice()?)
            .map_err(invalid)?
            .into_pyarray(py))
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let parameters = parameters.as_slice()?.to_vec();
        let state = py
            .detach(|| self.inner.evaluate(&parameters))
            .map_err(invalid)?;
        evaluation(py, &state)
    }
    #[pyo3(signature=(parameters, iterations, max_seconds, cancellation, iterative=false, cached_normal=false, reuse_normal=false, blocked_normal=false, assignment_stopping=false))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn refine<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'_, f64>,
        iterations: usize,
        max_seconds: f64,
        cancellation: Option<PyRef<'_, NativeCancellation>>,
        iterative: bool,
        cached_normal: bool,
        reuse_normal: bool,
        blocked_normal: bool,
        assignment_stopping: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !max_seconds.is_finite() || !(0.0..=3600.).contains(&max_seconds) || max_seconds == 0. {
            return Err(PyValueError::new_err("invalid refinement runtime"));
        }
        let parameters = parameters.as_slice()?.to_vec();
        let cancellation = cancellation.map(|c| c.inner.clone());
        let model = self
            .inner
            .clone()
            .with_blocked_normal_matrices(blocked_normal)
            .with_assignment_convergence_stopping(assignment_stopping);
        let state = py
            .detach(move || {
                model.refine_with_normal_reuse(
                    &parameters,
                    iterations,
                    Some(Instant::now() + Duration::from_secs_f64(max_seconds)),
                    cancellation.as_ref(),
                    iterative,
                    cached_normal,
                    reuse_normal,
                )
            })
            .map_err(invalid)?;
        result(py, &state)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeModel>()?;
    Ok(())
}
