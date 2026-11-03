//! Owned test/diagnostic boundary for normalization and measured-only phase quality.
use super::{array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_single_crystal::solution::{PreparedPhaseQuality, WilsonAudit, WilsonNormalization};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

pub(super) fn normalization_record<'py>(
    py: Python<'py>,
    a: &WilsonAudit,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("families", a.families)?;
    d.set_item("nonpositive_families", a.nonpositive_families)?;
    d.set_item("log_scale", a.log_scale)?;
    d.set_item("slope", a.slope)?;
    d.set_item("mean_e_squared", a.mean_e_squared)?;
    let shells = PyList::empty(py);
    for s in &a.shells {
        let item = PyDict::new(py);
        item.set_item("s_squared", s.s_squared)?;
        item.set_item("families", s.families)?;
        item.set_item("reduced_mean", s.reduced_mean)?;
        item.set_item("log_residual", s.log_residual)?;
        shells.append(item)?;
    }
    d.set_item("shells", shells)?;
    Ok(d)
}
pub(super) fn trace_records<'py>(
    py: Python<'py>,
    traces: &[phasesmith_single_crystal::solution::PhaseQualityTrace],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for a in traces {
        let d = PyDict::new(py);
        d.set_item("trial", a.trial)?;
        d.set_item("allowed_families", a.allowed_families)?;
        d.set_item("absent_families", a.absent_families)?;
        d.set_item("weak_families", a.weak_families)?;
        d.set_item("correlation", &a.correlation)?;
        d.set_item("r_weak", &a.r_weak)?;
        d.set_item("cfom", &a.cfom)?;
        d.set_item("deferred_handoffs", a.deferred_handoffs)?;
        d.set_item("alternative_handoffs", a.alternative_handoffs)?;
        d.set_item("calculated_shells", a.calculated_shells)?;
        out.append(d)?;
    }
    Ok(out)
}
#[pyclass(name = "_PhaseQualityPlan", frozen)]
struct NativeQuality {
    norm: WilsonNormalization,
    plan: PreparedPhaseQuality,
    hkl: Vec<[i32; 3]>,
}
#[pymethods]
impl NativeQuality {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(cell_parameters,group,hkl,intensity,sigma,elements,wilson=true,calculated_shell_normalization=true))]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        hkl: PyReadonlyArray1<'_, i64>,
        intensity: PyReadonlyArray1<'_, f64>,
        sigma: PyReadonlyArray1<'_, f64>,
        elements: Vec<String>,
        wilson: bool,
        calculated_shell_normalization: bool,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let hkl = hkl_rows(&hkl)?;
        let intensity = intensity.as_slice()?.to_vec();
        let sigma = sigma.as_slice()?.to_vec();
        if intensity.len() != hkl.len() || sigma.len() != hkl.len() {
            return Err(invalid("reflection array lengths differ"));
        }
        let raw: Vec<_> = hkl
            .into_iter()
            .zip(intensity)
            .zip(sigma)
            .map(
                |((hkl, intensity), sigma)| phasesmith_single_crystal::solution::Observation {
                    hkl,
                    intensity,
                    sigma,
                },
            )
            .collect();
        py.detach(move || {
            let data = phasesmith_single_crystal::solution::prepare_data(&raw, cell, &group)?;
            let norm = phasesmith_single_crystal::solution::normalize_wilson(
                cell,
                &group,
                &data.hkl,
                &data.intensity,
                &elements,
            )?;
            let plan = PreparedPhaseQuality::new(
                cell,
                &group,
                &data,
                &raw,
                &elements,
                &norm,
                wilson,
                calculated_shell_normalization,
            )?;
            Ok(Self {
                norm,
                plan,
                hkl: data.hkl,
            })
        })
        .map_err(|e: phasesmith_single_crystal::solution::SolutionError| invalid(e))
    }
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("hkl", array2(py, self.hkl.clone()))?;
        d.set_item(
            "expected_intensity",
            self.norm.expected_intensity.clone().into_pyarray(py),
        )?;
        d.set_item("e_values", self.norm.e_values.clone().into_pyarray(py))?;
        d.set_item("epsilon", &self.norm.epsilon)?;
        d.set_item("representatives", &self.norm.representatives)?;
        d.set_item("normalization", normalization_record(py, &self.norm.audit)?)?;
        d.set_item("quality_hkl", array2(py, self.plan.hkl.clone()))?;
        d.set_item("target", &self.plan.target)?;
        d.set_item("e_weights", &self.plan.e_weights)?;
        d.set_item("calculated_shell", &self.plan.calculated_shell)?;
        d.set_item("weak", &self.plan.weak)?;
        d.set_item("allowed", self.plan.allowed)?;
        Ok(d)
    }
    #[pyo3(signature=(calculated, rows=1))]
    fn evaluate_values<'py>(
        &self,
        py: Python<'py>,
        calculated: PyReadonlyArray1<'_, f64>,
        rows: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let values = calculated.as_slice()?.to_vec();
        if rows == 0 || values.len() != rows.checked_mul(self.plan.hkl.len()).unwrap_or(0) {
            return Err(invalid("invalid score-only batch shape"));
        }
        let result = py
            .detach(|| {
                let mut scorer =
                    phasesmith_single_crystal::solution::PreparedPhaseScore::new(&self.plan)?;
                values
                    .chunks_exact(self.plan.hkl.len())
                    .map(|v| scorer.evaluate(v))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(invalid)?;
        let out = pyo3::types::PyList::empty(py);
        for r in result {
            let d = PyDict::new(py);
            d.set_item("correlation", r.correlation)?;
            d.set_item("r_weak", r.r_weak)?;
            d.set_item("cfom", r.cfom)?;
            d.set_item("correlation_defined", r.correlation_defined)?;
            out.append(d)?;
        }
        Ok(out)
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        calculated: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v = calculated.as_slice()?.to_vec();
        let r = py.detach(|| self.plan.evaluate(&v)).map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("correlation", r.correlation)?;
        d.set_item("r_weak", r.r_weak)?;
        d.set_item("cfom", r.cfom)?;
        d.set_item("correlation_defined", r.correlation_defined)?;
        d.set_item("gradient", r.gradient.into_pyarray(py))?;
        Ok(d)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeQuality>()
}
