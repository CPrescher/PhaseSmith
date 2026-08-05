//! Python bindings for the Rietveld Engine numerical core.

#![allow(clippy::needless_pass_by_value)] // PyO3 extracts owned argument guards.

use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rietveld_core::{
    Accumulation, GridView, PeakBatchView, SupportPolicy, TchPeakBatchView, TchShape, TchWidths,
    accumulate_batch, accumulate_tch_batch, accumulate_values_batch, symmetric_pseudo_voigt,
};

type ProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type AccumulationArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
);

type TchShapeValues = (f64, f64, f64, f64, f64, f64);

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

/// Transform component FWHMs into TCH total width, eta, and derivatives.
#[pyfunction]
fn tch_shape_from_fwhm(gaussian_fwhm: f64, lorentzian_fwhm: f64) -> PyResult<TchShapeValues> {
    let shape = TchShape::from_component_fwhm(TchWidths {
        gaussian_fwhm,
        lorentzian_fwhm,
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        shape.total_fwhm,
        shape.eta,
        shape.d_total_fwhm_d_gaussian_fwhm,
        shape.d_total_fwhm_d_lorentzian_fwhm,
        shape.d_eta_d_gaussian_fwhm,
        shape.d_eta_d_lorentzian_fwhm,
    ))
}

/// Vectorized TCH profile evaluation with component-width derivatives.
#[pyfunction]
fn profile_tch<'py>(
    py: Python<'py>,
    delta: PyReadonlyArray1<'py, f64>,
    gaussian_fwhm: f64,
    lorentzian_fwhm: f64,
) -> PyResult<ProfileArrays<'py>> {
    let delta = contiguous_slice(&delta, "delta")?;
    if delta.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "delta must contain only finite values",
        ));
    }
    let widths = TchWidths {
        gaussian_fwhm,
        lorentzian_fwhm,
    };
    let shape = TchShape::from_component_fwhm(widths)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut value = Vec::with_capacity(delta.len());
    let mut d_delta = Vec::with_capacity(delta.len());
    let mut d_gaussian = Vec::with_capacity(delta.len());
    let mut d_lorentzian = Vec::with_capacity(delta.len());
    for coordinate in delta.iter().copied() {
        let point = shape.evaluate(coordinate);
        value.push(point.value);
        d_delta.push(point.d_delta);
        d_gaussian.push(point.d_gaussian_fwhm);
        d_lorentzian.push(point.d_lorentzian_fwhm);
    }
    Ok((
        value.into_pyarray(py),
        d_delta.into_pyarray(py),
        d_gaussian.into_pyarray(py),
        d_lorentzian.into_pyarray(py),
    ))
}

/// Fused peak accumulation returning dense values and support-sparse derivatives.
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

    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = PeakBatchView::new(positions, intensities, fwhms, etas)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Fused TCH peak accumulation with support-sparse direct-input derivatives.
#[pyfunction]
fn accumulate_tch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    gaussian_fwhms: PyReadonlyArray1<'py, f64>,
    lorentzian_fwhms: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let gaussian_fwhms = contiguous_slice(&gaussian_fwhms, "gaussian_fwhms")?;
    let lorentzian_fwhms = contiguous_slice(&lorentzian_fwhms, "lorentzian_fwhms")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = TchPeakBatchView::new(positions, intensities, gaussian_fwhms, lorentzian_fwhms)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_tch_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

fn accumulation_to_numpy(
    py: Python<'_>,
    accumulation: Accumulation,
) -> PyResult<AccumulationArrays<'_>> {
    let local = accumulation.derivatives.local;
    let value_rows = local.active_sample_count();
    let starts = indices_to_i64(local.starts, "support starts")?;
    let offsets = indices_to_i64(local.offsets, "support offsets")?;
    let values = Array2::from_shape_vec((value_rows, 4), local.values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        accumulation.y.into_pyarray(py),
        starts.into_pyarray(py),
        offsets.into_pyarray(py),
        values.into_pyarray(py),
    ))
}

/// Fused peak accumulation returning calculated values without derivatives.
#[pyfunction]
fn accumulate_values<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    fwhms: PyReadonlyArray1<'py, f64>,
    etas: PyReadonlyArray1<'py, f64>,
    support_fwhm: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = contiguous_slice(&x, "x")?;
    let positions = contiguous_slice(&positions, "positions")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let fwhms = contiguous_slice(&fwhms, "fwhms")?;
    let etas = contiguous_slice(&etas, "etas")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let peaks = PeakBatchView::new(positions, intensities, fwhms, etas)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let y = accumulate_values_batch(grid, peaks, SupportPolicy::FwhmMultiple(support_fwhm))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(y.into_pyarray(py))
}

fn indices_to_i64(indices: Vec<usize>, name: &str) -> PyResult<Vec<i64>> {
    indices
        .into_iter()
        .map(|index| {
            i64::try_from(index)
                .map_err(|_| PyValueError::new_err(format!("{name} exceed NumPy int64 range")))
        })
        .collect()
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
    module.add_function(wrap_pyfunction!(tch_shape_from_fwhm, module)?)?;
    module.add_function(wrap_pyfunction!(profile_tch, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_tch, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_values, module)?)?;
    module.add("PARAMETER_ORDER", ("intensity", "position", "fwhm", "eta"))?;
    module.add(
        "TCH_PARAMETER_ORDER",
        ("intensity", "position", "gaussian_fwhm", "lorentzian_fwhm"),
    )?;
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
