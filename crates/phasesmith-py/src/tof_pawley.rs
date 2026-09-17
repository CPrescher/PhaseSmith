//! Thin detached native TOF Pawley boundary.
use crate::pawley::{arrays, result_arrays};
use crate::rietveld::NativeRietveldCancellation;
use phasesmith_persistence::{
    TofPawleyProject, decode_tof_pawley_project, encode_tof_pawley_project, save_tof_pawley_project,
};
use phasesmith_workflows::{
    ConstraintTransform, RefinementLimits, RefinementRuntime, evaluate_tof_pawley,
    refine_tof_pawley_with_runtime,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
fn error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
const MAX_BYTES: usize = 256 * 1024 * 1024;
#[pyfunction]
fn _tof_pawley_prepare(py: Python<'_>, record: String) -> PyResult<String> {
    py.detach(move || encode_tof_pawley_project(&decode_tof_pawley_project(&record, MAX_BYTES)?))
        .map_err(error)
}
#[pyfunction]
fn _tof_pawley_save(py: Python<'_>, record: String, path: String) -> PyResult<()> {
    py.detach(move || {
        save_tof_pawley_project(path, &decode_tof_pawley_project(&record, MAX_BYTES)?)
    })
    .map_err(error)
}
#[pyfunction]
fn _tof_pawley_calculate(py: Python<'_>, record: String) -> PyResult<Bound<'_, PyDict>> {
    let evaluation = py
        .detach(move || {
            let mut p = decode_tof_pawley_project(&record, MAX_BYTES)?;
            for b in &mut p.input.banks {
                if b.pattern.observed_y.is_none() {
                    b.pattern.observed_y = Some(vec![0.0; b.pattern.sample_count()]);
                }
            }
            let t =
                ConstraintTransform::new(p.input.parameters.clone(), p.input.constraints.clone())
                    .map_err(|e| phasesmith_workflows::PawleyError(e.to_string()))?;
            let free = p
                .checkpoint
                .as_ref()
                .map_or_else(|| t.pack(), |cp| Ok(cp.free.clone()))
                .map_err(|e| phasesmith_workflows::PawleyError(e.to_string()))?;
            evaluate_tof_pawley(&p.input, &free, &p.options)
        })
        .map_err(error)?;
    arrays(py, &evaluation)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _tof_pawley_refine<'py>(
    py: Python<'py>,
    record: String,
    max_iterations: usize,
    max_evaluations: usize,
    max_rejections: usize,
    max_seconds: Option<f64>,
    cancellation: Option<&NativeRietveldCancellation>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let token = cancellation.map(|t| t.token.clone());
    let result = py
        .detach(move || -> Result<_, String> {
            let p = decode_tof_pawley_project(&record, MAX_BYTES).map_err(|e| e.to_string())?;
            let limits =
                RefinementLimits::new(max_iterations, max_evaluations, max_seconds, max_rejections)
                    .map_err(|e| e.to_string())?;
            let mut runtime = RefinementRuntime::new(limits, token).map_err(|e| e.to_string())?;
            if let Some(callback) = progress {
                runtime.set_event_sink(move |event: &phasesmith_workflows::RefinementEvent| {
                    Python::attach(|py| -> PyResult<()> {
                        let d = PyDict::new(py);
                        d.set_item("kind", event.kind().as_str())?;
                        d.set_item("message", event.message())?;
                        d.set_item("accepted_iterations", event.accepted_iterations())?;
                        callback.call1(py, (d,))?;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())
                });
            }
            refine_tof_pawley_with_runtime(
                &p.input,
                &p.options,
                p.checkpoint.as_ref(),
                &mut runtime,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(error)?;
    let record = encode_tof_pawley_project(&TofPawleyProject {
        input: result.checkpoint.input.clone(),
        options: result.checkpoint.options.clone(),
        checkpoint: Some(result.checkpoint.clone()),
    })
    .map_err(error)?;
    result_arrays(py, result, record)
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_tof_pawley_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(_tof_pawley_calculate, m)?)?;
    m.add_function(wrap_pyfunction!(_tof_pawley_refine, m)?)?;
    m.add_function(wrap_pyfunction!(_tof_pawley_save, m)?)?;
    Ok(())
}
