//! Owned prediction diagnostic boundary; generated coefficients remain separate.
use super::{array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::Complex64 as Complex;
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_single_crystal::solution::{PredictionAudit, PreparedReflectionPrediction};
use pyo3::{prelude::*, types::PyDict};

pub(super) fn audit<'py>(py: Python<'py>, a: &PredictionAudit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("shape", a.shape)?;
    d.set_item("measured_rows", a.measured_rows)?;
    d.set_item("missing_pairs", a.missing_pairs)?;
    d.set_item("extended_pairs", a.extended_pairs)?;
    d.set_item("observed_d_min", a.observed_d_min)?;
    d.set_item("requested_extension", a.requested_extension)?;
    d.set_item("effective_extension", a.effective_extension)?;
    d.set_item("effective_d_min", a.effective_d_min)?;
    Ok(d)
}
#[pyclass(name = "_ReflectionPredictionPlan", frozen)]
struct NativePrediction {
    inner: PreparedReflectionPrediction,
    cell: phasesmith_engine::crystallography::UnitCell,
    hkl: Vec<[i32; 3]>,
    amplitudes: Vec<f64>,
    intensity: Vec<f64>,
}
#[pymethods]
impl NativePrediction {
    #[new]
    #[allow(clippy::too_many_arguments, clippy::float_cmp)]
    #[pyo3(signature=(shape,cell_parameters,group,hkl,intensity,sigma,elements,missing=true,extension=1.2,power=1.))]
    fn new(
        py: Python<'_>,
        shape: [usize; 3],
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        hkl: PyReadonlyArray1<'_, i64>,
        intensity: PyReadonlyArray1<'_, f64>,
        sigma: PyReadonlyArray1<'_, f64>,
        elements: Vec<String>,
        missing: bool,
        extension: f64,
        power: f64,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let hkl = hkl_rows(&hkl)?;
        let intensity = intensity.as_slice()?.to_vec();
        let sigma = sigma.as_slice()?.to_vec();
        if hkl.len() != intensity.len() || hkl.len() != sigma.len() {
            return Err(invalid("prediction observation lengths differ"));
        }
        let rows: Vec<_> = hkl
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
            let data = phasesmith_single_crystal::solution::prepare_data(&rows, cell, &group)?;
            let norm = phasesmith_single_crystal::solution::normalize_wilson(
                cell,
                &group,
                &data.hkl,
                &data.intensity,
                &elements,
            )?;
            let amplitudes = if power == 0.5 {
                norm.e_values
                    .iter()
                    .zip(&data.amplitudes)
                    .map(|(e, f)| e.sqrt() * f.sqrt())
                    .collect()
            } else {
                data.amplitudes.clone()
            };
            let inner = PreparedReflectionPrediction::new(
                shape,
                cell,
                &group,
                &data,
                &norm,
                &elements,
                &amplitudes,
                power,
                missing,
                extension,
            )?;
            Ok(Self {
                inner,
                cell,
                hkl: data.hkl,
                intensity: data.intensity,
                amplitudes,
            })
        })
        .map_err(|e: phasesmith_single_crystal::solution::SolutionError| invalid(e))
    }
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = audit(py, &self.inner.audit)?;
        d.set_item("support", self.inner.support.clone().into_pyarray(py))?;
        d.set_item("observed_rms", self.inner.observed_rms)?;
        d.set_item("hkl", array2(py, self.hkl.clone()))?;
        d.set_item("amplitudes", self.amplitudes.clone().into_pyarray(py))?;
        d.set_item("intensity", self.intensity.clone().into_pyarray(py))?;
        d.set_item("predicted", self.inner.predicted_columns())?;
        d.set_item("measured", self.inner.measured_columns())?;
        Ok(d)
    }
    #[pyo3(signature=(real,imag,cycle,unit_scale=1.))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        real: PyReadonlyArray1<'_, f64>,
        imag: PyReadonlyArray1<'_, f64>,
        cycle: usize,
        unit_scale: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let real = real.as_slice()?.to_vec();
        let imag = imag.as_slice()?.to_vec();
        if real.len() != imag.len() {
            return Err(invalid("prediction complex array lengths differ"));
        }
        let mut coefficients: Vec<_> = real
            .into_iter()
            .zip(imag)
            .map(|(a, b)| Complex::new(a, b))
            .collect();
        let (stats, step) = py
            .detach(|| {
                let stats = self.inner.statistics(&coefficients, unit_scale)?;
                let step = self.inner.apply_with_statistics(
                    &mut coefficients,
                    unit_scale,
                    cycle,
                    &stats,
                )?;
                Ok((stats, step))
            })
            .map_err(|e: phasesmith_single_crystal::solution::SolutionError| invalid(e))?;
        let d = PyDict::new(py);
        d.set_item(
            "real",
            coefficients
                .iter()
                .map(|v| v.re)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        d.set_item(
            "imag",
            coefficients
                .iter()
                .map(|v| v.im)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        d.set_item("statistics", stats)?;
        d.set_item("updated", step.updated)?;
        d.set_item("clipped", step.clipped)?;
        d.set_item("reliability", step.reliability)?;
        Ok(d)
    }
    #[pyo3(signature=(phases,cycles=16,count=4,seed=1))]
    fn dual<'py>(
        &self,
        py: Python<'py>,
        phases: PyReadonlyArray1<'_, f64>,
        cycles: usize,
        count: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let phases = phases.as_slice()?.to_vec();
        let state = py
            .detach(|| {
                phasesmith_single_crystal::solution::dual_space_with_prediction(
                    self.inner.audit.shape,
                    self.cell,
                    &self.hkl,
                    &self.amplitudes,
                    &phases,
                    &self.inner.support,
                    &phasesmith_single_crystal::solution::DualSpaceOptions {
                        cycles,
                        peak_selection: phasesmith_single_crystal::solution::PeakSelection::Fixed(
                            count,
                        ),
                        mask_sigma: 0.25,
                        hybrid_m: 3.,
                        omit_fraction: 0.1,
                        seed,
                        patterson_rank: None,
                        cancellation: None,
                    },
                    None,
                    &self.inner,
                )
            })
            .map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("density", state.trial.density.into_pyarray(py))?;
        d.set_item("history", state.trial.history.into_pyarray(py))?;
        d.set_item("best_cycle", state.trial.best_cycle)?;
        d.set_item(
            "updates",
            state.prediction.iter().map(|s| s.updated).sum::<usize>(),
        )?;
        Ok(d)
    }
}
#[pyfunction(name = "_prediction_coefficient")]
fn coefficient(
    py: Python<'_>,
    value: [f64; 2],
    expected: f64,
    moment: f64,
    correlation: f64,
    gain: f64,
) -> PyResult<Bound<'_, PyDict>> {
    let result = phasesmith_single_crystal::solution::prediction_coefficient(
        value,
        expected,
        moment,
        correlation,
        gain,
    )
    .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("value", result.value)?;
    out.set_item("jacobian", result.jacobian)?;
    out.set_item("clipped", result.clipped)?;
    Ok(out)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePrediction>()?;
    module.add_function(wrap_pyfunction!(coefficient, module)?)?;
    Ok(())
}
