//! `NumPy` boundary for opt-in symmetry-based scan scaling.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use phasesmith_spots::scan_scaling as kernel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register opt-in scaling functions alongside the other experimental kernels.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_scan_scale_fit, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_scan_scale_apply, module)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn experimental_scan_scale_fit<'py>(
    py: Python<'py>,
    intensity: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    normalized_omega: PyReadonlyArray1<'py, f64>,
    families: PyReadonlyArray1<'py, i64>,
    training: PyReadonlyArray1<'py, bool>,
    minimum_snr: f64,
    relative_error_floor: f64,
    huber: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<(
    f64,
    f64,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    bool,
)> {
    let values = intensity.as_slice()?.to_vec();
    let sigma = sigma.as_slice()?.to_vec();
    let x = normalized_omega.as_slice()?.to_vec();
    let groups: Vec<usize> = families
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err("negative scaling family")))
        .collect::<PyResult<_>>()?;
    let training = training.as_slice()?.to_vec();
    let options = kernel::Options {
        minimum_snr,
        relative_error_floor,
        huber,
        max_iterations,
        tolerance,
    };
    let fit = py
        .detach(move || kernel::fit(&values, &sigma, &x, &groups, &training, options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ids: Vec<i64> = fit
        .reference_ids
        .iter()
        .map(|&v| i64::try_from(v).expect("one-million input bound"))
        .collect();
    Ok((
        fit.coefficient,
        fit.variance,
        ids.into_pyarray(py),
        fit.reference_weights.into_pyarray(py),
        fit.iterations,
        fit.converged,
    ))
}

#[pyfunction]
fn experimental_scan_scale_apply<'py>(
    py: Python<'py>,
    intensity: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    omega: PyReadonlyArray1<'py, f64>,
    range: [f64; 2],
    coefficient: f64,
    variance: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = intensity.as_slice()?.to_vec();
    let sigma = sigma.as_slice()?.to_vec();
    let omega = omega.as_slice()?.to_vec();
    let rows = py
        .detach(move || kernel::apply(&values, &sigma, &omega, range, coefficient, variance))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = rows.len();
    let values: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((n, 6), values)
        .expect("six columns per output")
        .into_pyarray(py))
}
