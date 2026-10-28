//! Owned detached boundary for completion equations and trial provenance.
use super::{cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::{CompletionSummary, PreparedOrbitSearch, project_fixed_sets};
use pyo3::{prelude::*, types::PyDict};

pub(super) fn summary<'py>(py: Python<'py>, s: &CompletionSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("applied", s.applied)?;
    d.set_item("status", s.status)?;
    d.set_item("diagnostic", &s.diagnostic)?;
    d.set_item("framework_species", &s.framework_species)?;
    d.set_item("framework_atoms", s.framework_atoms)?;
    d.set_item("added_species", &s.added_species)?;
    d.set_item("added_atoms", s.added_atoms)?;
    d.set_item("seed_count", s.seed_count)?;
    d.set_item("projection_trials", s.projection_trials)?;
    d.set_item("orbit_trials", s.orbit_trials)?;
    d.set_item("refinements", s.refinements)?;
    d.set_item("framework_maximum_error", s.framework_maximum_error)?;
    d.set_item("framework_hypotheses", s.framework_hypotheses)?;
    d.set_item("selected_framework", s.selected_framework)?;
    d.set_item("initial_objective", s.initial_objective)?;
    d.set_item("best_objective", s.best_objective)?;
    Ok(d)
}

#[pyclass(name = "_CompletionSearch", frozen)]
struct NativeSearch {
    inner: PreparedOrbitSearch,
}
#[pymethods]
impl NativeSearch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        fixed_species: Vec<String>,
        fixed_xyz: PyReadonlyArray1<'_, f64>,
        added_species: String,
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        u_iso: f64,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let xyz = fixed_xyz.as_slice()?;
        if xyz.len() != fixed_species.len() * 3 {
            return Err(invalid("framework coordinate/species lengths differ"));
        }
        let points = xyz
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect::<Vec<_>>();
        let hkl = hkl_rows(&hkl)?;
        let observed = observed.as_slice()?.to_vec();
        let inner = py
            .detach(move || {
                PreparedOrbitSearch::new(
                    cell,
                    &group,
                    &fixed_species,
                    &points,
                    &added_species,
                    &hkl,
                    &observed,
                    u_iso,
                )
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn framework_f(&self) -> Vec<[f64; 2]> {
        self.inner.framework_f()
    }
    fn evaluate<'py>(&self, py: Python<'py>, point: [f64; 3]) -> PyResult<Bound<'py, PyDict>> {
        let e = py.detach(|| self.inner.evaluate(point)).map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("objective", e.objective)?;
        d.set_item("r1", e.r1)?;
        d.set_item("amplitude_scale", e.amplitude_scale)?;
        d.set_item("gradient", e.gradient)?;
        d.set_item("residual", e.residual.into_pyarray(py))?;
        d.set_item("jacobian", e.jacobian.into_pyarray(py))?;
        d.set_item("fractional_xyz", super::array2(py, e.fractional_xyz))?;
        Ok(d)
    }
    fn difference_map<'py>(
        &self,
        py: Python<'py>,
        shape: [usize; 3],
    ) -> PyResult<Bound<'py, PyDict>> {
        let m = py
            .detach(|| self.inner.difference_map(shape))
            .map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("density", m.density.into_pyarray(py))?;
        d.set_item("amplitude_scale", m.amplitude_scale)?;
        d.set_item("phased_reflections", m.phased_reflections)?;
        Ok(d)
    }
    fn project(
        &self,
        point: [f64; 3],
        operation_indices: Vec<usize>,
        tolerance: f64,
    ) -> PyResult<Option<[f64; 3]>> {
        let ops = operation_indices
            .iter()
            .map(|&i| {
                self.inner
                    .group
                    .operations()
                    .get(i)
                    .copied()
                    .ok_or_else(|| invalid("operation index out of range"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        project_fixed_sets(&self.inner.geometry, point, &ops, tolerance).map_err(invalid)
    }
}

#[pyfunction(name = "_completion_frameworks")]
#[allow(clippy::too_many_arguments)]
fn frameworks<'py>(
    py: Python<'py>,
    cell_parameters: Vec<f64>,
    group: PyRef<'_, NativePreparedReflectionGenerator>,
    species: Vec<String>,
    xyz: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let cell = cell(&cell_parameters)?;
    let group = group.space_group().clone();
    let xyz = xyz.as_slice()?;
    if xyz.len() != 3 * species.len() {
        return Err(invalid("framework coordinate/species lengths differ"));
    }
    let points = xyz
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect::<Vec<_>>();
    let models = py
        .detach(move || {
            phasesmith_solution::recover_symmetry_hypotheses(
                cell, &group, &species, &points, tolerance, None, None, limit,
            )
        })
        .map_err(invalid)?;
    let out = pyo3::types::PyList::empty(py);
    for model in models {
        let d = PyDict::new(py);
        d.set_item("hand", model.hand)?;
        d.set_item("origin_shift", model.origin_shift)?;
        d.set_item(
            "asymmetric_species",
            model
                .sites
                .iter()
                .map(|s| s.species.clone())
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "asymmetric_xyz",
            model
                .sites
                .iter()
                .map(|s| s.fractional_xyz)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "multiplicities",
            model
                .sites
                .iter()
                .map(|s| s.images.len())
                .collect::<Vec<_>>(),
        )?;
        out.append(d)?;
    }
    Ok(out)
}

#[pyfunction(name = "_solution_complete_candidate")]
#[allow(clippy::too_many_arguments)]
fn complete<'py>(
    py: Python<'py>,
    cell_parameters: Vec<f64>,
    group: PyRef<'_, NativePreparedReflectionGenerator>,
    species: Vec<String>,
    xyz: PyReadonlyArray1<'_, f64>,
    hkl: PyReadonlyArray1<'_, i64>,
    observed: PyReadonlyArray1<'_, f64>,
    shape: [usize; 3],
    max_seconds: f64,
    cancellation: Option<PyRef<'_, super::NativeCancellation>>,
) -> PyResult<Bound<'py, PyDict>> {
    if !max_seconds.is_finite() || !(1e-12..=60.).contains(&max_seconds) {
        return Err(invalid("invalid completion runtime"));
    }
    let cell = cell(&cell_parameters)?;
    let group = group.space_group().clone();
    let hkl = hkl_rows(&hkl)?;
    let observed = observed.as_slice()?.to_vec();
    let xyz = xyz.as_slice()?;
    if xyz.len() != 3 * species.len()
        || observed.len() != hkl.len()
        || observed
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || *v > 1e50)
    {
        return Err(invalid("invalid candidate boundary"));
    }
    let points = xyz
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect::<Vec<_>>();
    let token = cancellation.map(|t| t.inner.clone());
    let (candidate, audit) = py
        .detach(move || -> Result<_, phasesmith_solution::SolutionError> {
            let observations = hkl
                .iter()
                .zip(&observed)
                .map(|(&h, &f)| phasesmith_solution::Observation {
                    hkl: h,
                    intensity: f * f,
                    sigma: 1.,
                })
                .collect::<Vec<_>>();
            let data = phasesmith_solution::prepare_data(&observations, cell, &group)?;
            let evaluation = phasesmith_solution::evaluate_model(
                cell,
                &data.hkl,
                &data.amplitudes,
                &species,
                &points,
                0.01,
            )?;
            let candidate = phasesmith_solution::Candidate {
                species,
                fractional_xyz: points,
                u_iso: 0.01,
                amplitude_scale: evaluation.amplitude_scale,
                r1: evaluation.r1,
                objective: evaluation.objective,
                trial: 0,
                polish_steps: 0,
                element_swaps: 0,
                refinement: None,
                refinement_diagnostic: None,
                completion: None,
            };
            let options = phasesmith_solution::SolveOptions {
                cancellation: token,
                ..Default::default()
            };
            Ok(phasesmith_solution::complete_candidate(
                cell,
                &group,
                &data,
                shape,
                &candidate,
                phasesmith_solution::AdpMode::Anisotropic,
                &options,
                std::time::Instant::now() + std::time::Duration::from_secs_f64(max_seconds),
            ))
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("completion", summary(py, &audit)?)?;
    if let Some(c) = candidate {
        let d = PyDict::new(py);
        d.set_item("species", c.species)?;
        d.set_item("fractional_xyz", super::array2(py, c.fractional_xyz))?;
        d.set_item("r1", c.r1)?;
        d.set_item("objective", c.objective)?;
        d.set_item("amplitude_scale", c.amplitude_scale)?;
        d.set_item("u_iso", c.u_iso)?;
        d.set_item(
            "refinement",
            super::symmetry_adp::summary(
                py,
                c.refinement
                    .as_ref()
                    .expect("completed candidate is refined"),
            )?,
        )?;
        out.set_item("candidate", d)?;
    } else {
        out.set_item("candidate", py.None())?;
    }
    Ok(out)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSearch>()?;
    module.add_function(wrap_pyfunction!(frameworks, module)?)?;
    module.add_function(wrap_pyfunction!(complete, module)?)?;
    Ok(())
}
