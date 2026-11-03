//! Owned-array boundary for independent phase-invariant and complete-step checks.
use super::{array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_single_crystal::solution::{PreparedTangent, TangentAudit};
use pyo3::{prelude::*, types::PyDict};

pub(super) fn audit<'py>(py: Python<'py>, a: &TangentAudit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("trial", a.trial)?;
    d.set_item("variables", a.variables)?;
    d.set_item("strong", a.strong)?;
    d.set_item("triplets", a.triplets)?;
    d.set_item("quartets", a.quartets)?;
    d.set_item("truncated", a.truncated)?;
    d.set_item("steps", a.steps)?;
    d.set_item("annealed_steps", a.annealed_steps)?;
    d.set_item("proposals", a.proposals)?;
    d.set_item("changed", a.changed)?;
    d.set_item("declined_steps", a.declined_steps)?;
    d.set_item("symmetry_restorations", a.symmetry_restorations)?;
    d.set_item("symmetry_declined_steps", a.symmetry_declined_steps)?;
    d.set_item("quartet_before", &a.quartet_before)?;
    d.set_item("quartet_after", &a.quartet_after)?;
    Ok(d)
}
#[pyclass(name = "_PreparedTangent", frozen)]
pub(super) struct NativeTangent {
    pub(super) inner: PreparedTangent,
}
#[pymethods]
impl NativeTangent {
    #[new]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        hkl: PyReadonlyArray1<'_, i64>,
        intensity: PyReadonlyArray1<'_, f64>,
        sigma: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let hkl = hkl_rows(&hkl)?;
        let intensity = intensity.as_slice()?.to_vec();
        let sigma = sigma.as_slice()?.to_vec();
        let inner = py
            .detach(move || {
                PreparedTangent::new(cell, &group, &hkl, &intensity, &sigma, None, None)
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let (e, upper) = self.inner.normalized_values();
        let (centers, means) = self.inner.shells();
        d.set_item("hkl", array2(py, self.inner.independent_hkl().to_vec()))?;
        d.set_item("normalized", e.to_vec().into_pyarray(py))?;
        d.set_item("upper", upper.to_vec().into_pyarray(py))?;
        d.set_item("strong", self.inner.selected().to_vec())?;
        d.set_item("shell_centers", centers.to_vec())?;
        d.set_item("shell_means", means.to_vec())?;
        let triplets: Vec<_> = self
            .inner
            .triplet_relations()
            .iter()
            .map(|t| (t.target, t.partners.map(|p| (p.index, p.sign)), t.weight))
            .collect();
        let quartets: Vec<_> = self
            .inner
            .quartet_relations()
            .iter()
            .map(|q| (q.terms.map(|p| (p.index, p.sign)), q.weight))
            .collect();
        let mapping: Vec<_> = self
            .inner
            .mapping()
            .iter()
            .map(|t| (t.index, t.sign))
            .collect();
        d.set_item("triplets", triplets)?;
        d.set_item("quartets", quartets)?;
        d.set_item("mapping", mapping)?;
        d.set_item("audit", audit(py, &self.inner.audit())?)?;
        Ok(d)
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let phases = phases.as_slice()?.to_vec();
        let e = py
            .detach(|| self.inner.evaluate(&phases))
            .map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("triplet", e.triplet)?;
        d.set_item("negative_quartet", e.negative_quartet)?;
        d.set_item("triplet_gradient", e.triplet_gradient.into_pyarray(py))?;
        d.set_item("quartet_gradient", e.quartet_gradient.into_pyarray(py))?;
        Ok(d)
    }
    fn step<'py>(
        &self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'_, f64>,
        magnitudes: PyReadonlyArray1<'_, f64>,
        cycle: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let phases = phases.as_slice()?.to_vec();
        let magnitudes = magnitudes.as_slice()?.to_vec();
        let s = py
            .detach(|| self.inner.step(&phases, &magnitudes, cycle, seed))
            .map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("phases", s.phases.into_pyarray(py))?;
        d.set_item("temperature", s.temperature)?;
        d.set_item("proposals", s.proposals)?;
        d.set_item("changed", s.changed)?;
        d.set_item("blend", s.blend)?;
        d.set_item("quartet_before", s.quartet_before)?;
        d.set_item("quartet_after", s.quartet_after)?;
        Ok(d)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTangent>()?;
    Ok(())
}
