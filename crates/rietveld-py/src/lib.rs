//! Python bindings for the Rietveld Engine numerical core.

#![allow(clippy::needless_pass_by_value)] // PyO3 extracts owned argument guards.

use npy::ndarray::Array3;
use npy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rietveld_core::{Peak, accumulate_peaks, symmetric_pseudo_voigt};

type ProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type AccumulationArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray3<f64>>);

/// Vectorized scalar profile evaluation used by the public Python wrapper.
#[pyfunction]
fn profile<'py>(
    py: Python<'py>,
    delta: PyReadonlyArray1<'py, f64>,
    fwhm: f64,
    eta: f64,
) -> PyResult<ProfileArrays<'py>> {
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return Err(PyValueError::new_err("fwhm must be positive and finite"));
    }
    if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
        return Err(PyValueError::new_err(
            "eta must be finite and within [0, 1]",
        ));
    }
    let delta = delta
        .as_slice()
        .map_err(|_| PyValueError::new_err("delta must be a contiguous one-dimensional array"))?;
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "delta must contain only finite values",
        ));
    }

    let mut value = Vec::with_capacity(delta.len());
    let mut d_delta = Vec::with_capacity(delta.len());
    let mut d_fwhm = Vec::with_capacity(delta.len());
    let mut d_eta = Vec::with_capacity(delta.len());
    for coordinate in delta.iter().copied() {
        let point = symmetric_pseudo_voigt(coordinate, fwhm, eta);
        value.push(point.value);
        d_delta.push(point.d_delta);
        d_fwhm.push(point.d_fwhm);
        d_eta.push(point.d_eta);
    }
    Ok((
        value.into_pyarray(py),
        d_delta.into_pyarray(py),
        d_fwhm.into_pyarray(py),
        d_eta.into_pyarray(py),
    ))
}

/// Fused peak accumulation returning `(y, jacobian)`.
#[pyfunction]
fn accumulate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    fwhms: PyReadonlyArray1<'py, f64>,
    etas: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let fwhms = contiguous_slice(&fwhms, "fwhms")?;
    let etas = contiguous_slice(&etas, "etas")?;

    let peak_count = positions.len();
    if intensities.len() != peak_count || fwhms.len() != peak_count || etas.len() != peak_count {
        return Err(PyValueError::new_err(
            "positions, intensities, fwhms, and etas must have equal length",
        ));
    }
    let peaks: Vec<Peak> = (0..peak_count)
        .map(|index| Peak {
            position: positions[index],
            intensity: intensities[index],
            fwhm: fwhms[index],
            eta: etas[index],
        })
        .collect();

    let accumulation = accumulate_peaks(x, &peaks, support_fwhm)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let jacobian = Array3::from_shape_vec(
        (accumulation.peak_count, 4, accumulation.sample_count),
        accumulation.jacobian,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((accumulation.y.into_pyarray(py), jacobian.into_pyarray(py)))
}

fn contiguous_slice<'array>(
    array: &'array PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'array [f64]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous one-dimensional array"))
    })
}

/// Native Python module.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(profile, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate, module)?)?;
    module.add("PARAMETER_ORDER", ("intensity", "position", "fwhm", "eta"))?;
    module.add(
        "BUILD_MODE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
