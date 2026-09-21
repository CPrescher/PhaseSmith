//! Thin adapter for read-only native fit evidence.

use npy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

type Region = (f64, f64, usize, f64, f64, f64);
type Report = (usize, usize, f64, f64, f64, Option<f64>, Vec<Region>);

#[pyfunction]
fn residual_diagnostics(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    residual: PyReadonlyArray1<'_, f64>,
    weighted: PyReadonlyArray1<'_, f64>,
    included: PyReadonlyArray1<'_, bool>,
    region_count: usize,
) -> PyResult<Report> {
    let x = x.as_slice()?;
    let residual = residual.as_slice()?;
    let weighted = weighted.as_slice()?;
    let included = included.as_slice()?;
    let r = py
        .detach(|| {
            phasesmith_workflows::diagnose_residuals(x, residual, weighted, included, region_count)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        r.included_count,
        r.adjacent_pair_count,
        r.chi_square,
        r.mean,
        r.weighted_rms,
        r.durbin_watson,
        r.regions
            .into_iter()
            .map(|v| {
                (
                    v.lower,
                    v.upper,
                    v.count,
                    v.chi_square,
                    v.chi_square_fraction,
                    v.maximum_absolute_weighted_residual,
                )
            })
            .collect(),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(residual_diagnostics, module)?)
}
