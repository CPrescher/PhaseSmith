//! Thin Python translation for the native file coordinator; numerical work releases the GIL.
use crate::solution::NativeCancellation;
use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyfunction]
fn _single_crystal_workflow_protocol() -> PyResult<String> {
    phasesmith::single_crystal_files::protocol_v1()
        .map(|v| v.to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (request_json, output, cancellation, progress=None))]
fn _run_single_crystal_workflow(
    py: Python<'_>,
    request_json: String,
    output: PathBuf,
    cancellation: PyRef<'_, NativeCancellation>,
    progress: Option<Py<PyAny>>,
) -> PyResult<String> {
    if progress.as_ref().is_some_and(|p| !p.bind(py).is_callable()) {
        return Err(PyValueError::new_err("progress must be callable"));
    }
    let token = cancellation.inner.clone();
    let stop = token.clone();
    let result = py.detach(move || -> Result<String, String> {
        let packet = serde_json::from_str(&request_json).map_err(|e| e.to_string())?;
        let mut callback =
            |name: &str, seconds: f64| -> phasesmith::single_crystal_files::Result<()> {
                if let Some(p) = &progress {
                    Python::attach(|py| p.call1(py, (name, seconds)))?;
                }
                Ok(())
            };
        phasesmith::single_crystal_files::run_packet(packet, output, token, &mut callback)
            .map(|v| v.to_string())
            .map_err(|e| e.to_string())
    });
    result.map_err(|e| {
        if stop.is_cancelled() {
            PyInterruptedError::new_err(e)
        } else {
            PyValueError::new_err(e)
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_single_crystal_workflow_protocol, module)?)?;
    module.add_function(wrap_pyfunction!(_run_single_crystal_workflow, module)?)?;
    Ok(())
}
