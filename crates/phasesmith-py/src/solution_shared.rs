//! Detached shared-search diagnostics and equation-test boundary.
use super::{array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::{PreparedElementPool, SharedSearchSummary};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

pub(super) fn summary<'py>(
    py: Python<'py>,
    s: &SharedSearchSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("elements", &s.elements)?;
    d.set_item("phasing_starts", s.phasing_starts)?;
    d.set_item("framework_hypotheses", s.framework_hypotheses)?;
    d.set_item("initial_prefix", s.initial_prefix)?;
    d.set_item("rounds", s.rounds)?;
    d.set_item("orbit_proposals", s.orbit_proposals)?;
    d.set_item("element_scores", s.element_scores)?;
    d.set_item("refinements", s.refinements)?;
    d.set_item("deletion_trials", s.deletion_trials)?;
    d.set_item("relabel_trials", s.relabel_trials)?;
    d.set_item("maximum_atoms", s.maximum_atoms)?;
    d.set_item("beam_width", s.beam_width)?;
    d.set_item("required_elements_present", s.required_elements_present)?;
    let moves = PyList::empty(py);
    for m in &s.accepted_moves {
        let x = PyDict::new(py);
        x.set_item("kind", m.kind)?;
        x.set_item("element", &m.element)?;
        x.set_item("atoms", m.atoms)?;
        x.set_item("r1", m.r1)?;
        x.set_item("score", m.score)?;
        moves.append(x)?;
    }
    d.set_item("accepted_moves", moves)?;
    Ok(d)
}
#[pyclass(name = "_SharedOrbitPool", frozen)]
struct NativePool {
    inner: PreparedElementPool,
}
#[pymethods]
impl NativePool {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        fixed_species: Vec<String>,
        fixed_xyz: PyReadonlyArray1<'_, f64>,
        elements: Vec<String>,
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        u_iso: f64,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let xyz = fixed_xyz.as_slice()?;
        if xyz.len() != fixed_species.len() * 3 {
            return Err(invalid("shared coordinate/species lengths differ"));
        }
        let points: Vec<_> = xyz.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let hkl = hkl_rows(&hkl)?;
        let observed = observed.as_slice()?.to_vec();
        let inner = py
            .detach(move || {
                PreparedElementPool::new(
                    cell,
                    &group,
                    &fixed_species,
                    &points,
                    &elements,
                    &hkl,
                    &observed,
                    u_iso,
                )
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn evaluate<'py>(&self, py: Python<'py>, point: [f64; 3]) -> PyResult<Bound<'py, PyList>> {
        let evaluations = py.detach(|| self.inner.evaluate(point)).map_err(invalid)?;
        let result = PyList::empty(py);
        for (element, e) in self.inner.elements.iter().zip(evaluations) {
            let d = PyDict::new(py);
            d.set_item("element", element)?;
            d.set_item("objective", e.objective)?;
            d.set_item("r1", e.r1)?;
            d.set_item("amplitude_scale", e.amplitude_scale)?;
            d.set_item("gradient", e.gradient)?;
            d.set_item("residual", e.residual.into_pyarray(py))?;
            d.set_item("jacobian", e.jacobian.into_pyarray(py))?;
            d.set_item("fractional_xyz", array2(py, e.fractional_xyz))?;
            result.append(d)?;
        }
        Ok(result)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePool>()
}
