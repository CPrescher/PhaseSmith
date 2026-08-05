//! Python bindings for the Rietveld Engine numerical core.

#![allow(clippy::needless_pass_by_value)] // PyO3 extracts owned argument guards.

use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rietveld_core::{
    Accumulation, ConstantWavelengthInstrument, CwProfileParameters, CwReflectionBatchView,
    FcjGeometry, FcjProfile, GridView, PeakBatchView, SupportPolicy, TchPeakBatchView, TchShape,
    TchWidths, accumulate_batch, accumulate_cw_batch, accumulate_cw_fcj_batch,
    accumulate_tch_batch, accumulate_values_batch, symmetric_pseudo_voigt,
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
    Bound<'py, PyArray2<f64>>,
);

type TchShapeValues = (f64, f64, f64, f64, f64, f64);

type CwProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

type FcjProfileArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

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

/// Vectorized FCJ-convolved TCH evaluation with direct-input derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn profile_fcj<'py>(
    py: Python<'py>,
    x_deg: PyReadonlyArray1<'py, f64>,
    position_deg: f64,
    gaussian_fwhm_deg: f64,
    lorentzian_fwhm_deg: f64,
    sample_over_radius: f64,
    detector_over_radius: f64,
) -> PyResult<FcjProfileArrays<'py>> {
    let x_deg = contiguous_slice(&x_deg, "x_deg")?;
    if x_deg.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "x_deg must contain only finite values",
        ));
    }
    let profile = FcjProfile::new(
        position_deg,
        TchWidths {
            gaussian_fwhm: gaussian_fwhm_deg,
            lorentzian_fwhm: lorentzian_fwhm_deg,
        },
        FcjGeometry {
            sample_over_radius,
            detector_over_radius,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut value = Vec::with_capacity(x_deg.len());
    let mut d_position = Vec::with_capacity(x_deg.len());
    let mut d_gaussian = Vec::with_capacity(x_deg.len());
    let mut d_lorentzian = Vec::with_capacity(x_deg.len());
    let mut d_sample = Vec::with_capacity(x_deg.len());
    let mut d_detector = Vec::with_capacity(x_deg.len());
    for coordinate in x_deg.iter().copied() {
        let point = profile.evaluate(coordinate);
        value.push(point.value);
        d_position.push(point.d_position);
        d_gaussian.push(point.d_gaussian_fwhm);
        d_lorentzian.push(point.d_lorentzian_fwhm);
        d_sample.push(point.d_sample_over_radius);
        d_detector.push(point.d_detector_over_radius);
    }
    Ok((
        value.into_pyarray(py),
        d_position.into_pyarray(py),
        d_gaussian.into_pyarray(py),
        d_lorentzian.into_pyarray(py),
        d_sample.into_pyarray(py),
        d_detector.into_pyarray(py),
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
    let local_parameter_count = local.parameter_count;
    let starts = indices_to_i64(local.starts, "support starts")?;
    let offsets = indices_to_i64(local.offsets, "support offsets")?;
    let values = Array2::from_shape_vec((value_rows, local_parameter_count), local.values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let global = if let Some(global) = accumulation.derivatives.global {
        if global.sample_count != accumulation.sample_count {
            return Err(PyValueError::new_err(
                "global Jacobian sample count does not match accumulation",
            ));
        }
        Array2::from_shape_vec((global.parameter_count, global.sample_count), global.values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
    } else {
        Array2::zeros((0, accumulation.sample_count))
    };
    Ok((
        accumulation.y.into_pyarray(py),
        starts.into_pyarray(py),
        offsets.into_pyarray(py),
        values.into_pyarray(py),
        global.into_pyarray(py),
    ))
}

/// Derive CW component widths, TCH shape, and width derivatives for reflections.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cw_profile_parameters<'py>(
    py: Python<'py>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
) -> PyResult<CwProfileArrays<'py>> {
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let instrument = cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg);
    instrument
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut variance = Vec::with_capacity(two_theta_deg.len());
    let mut gaussian = Vec::with_capacity(two_theta_deg.len());
    let mut lorentzian = Vec::with_capacity(two_theta_deg.len());
    let mut total = Vec::with_capacity(two_theta_deg.len());
    let mut eta = Vec::with_capacity(two_theta_deg.len());
    let mut d_gaussian = Vec::with_capacity(two_theta_deg.len() * 5);
    let mut d_lorentzian = Vec::with_capacity(two_theta_deg.len() * 5);
    let mut d_position = Vec::with_capacity(two_theta_deg.len() * 2);
    for position in two_theta_deg.iter().copied() {
        let profile = CwProfileParameters::from_instrument(position, instrument)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        variance.push(profile.gaussian_variance_deg2);
        gaussian.push(profile.gaussian_fwhm_deg);
        lorentzian.push(profile.lorentzian_fwhm_deg);
        total.push(profile.tch.total_fwhm);
        eta.push(profile.tch.eta);
        d_gaussian.extend_from_slice(&profile.d_gaussian_fwhm_d_instrument);
        d_lorentzian.extend_from_slice(&profile.d_lorentzian_fwhm_d_instrument);
        d_position.push(profile.d_gaussian_fwhm_d_two_theta);
        d_position.push(profile.d_lorentzian_fwhm_d_two_theta);
    }
    let count = two_theta_deg.len();
    Ok((
        variance.into_pyarray(py),
        gaussian.into_pyarray(py),
        lorentzian.into_pyarray(py),
        total.into_pyarray(py),
        eta.into_pyarray(py),
        Array2::from_shape_vec((count, 5), d_gaussian)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
        Array2::from_shape_vec((count, 5), d_lorentzian)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
        Array2::from_shape_vec((count, 2), d_position)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_pyarray(py),
    ))
}

/// Accumulate a CW reflection batch with local and global derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflections = CwReflectionBatchView::new(two_theta_deg, intensities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_cw_batch(
        grid,
        reflections,
        cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg),
        SupportPolicy::FwhmMultiple(support_fwhm),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

/// Accumulate an FCJ-asymmetric CW reflection batch with derivatives.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn accumulate_cw_fcj<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    two_theta_deg: PyReadonlyArray1<'py, f64>,
    intensities: PyReadonlyArray1<'py, f64>,
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
    sample_over_radius: f64,
    detector_over_radius: f64,
    support_fwhm: f64,
) -> PyResult<AccumulationArrays<'py>> {
    let x = contiguous_slice(&x, "x")?;
    let two_theta_deg = contiguous_slice(&two_theta_deg, "two_theta_deg")?;
    let intensities = contiguous_slice(&intensities, "intensities")?;
    let grid = GridView::new(x).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let reflections = CwReflectionBatchView::new(two_theta_deg, intensities)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let accumulation = accumulate_cw_fcj_batch(
        grid,
        reflections,
        cw_instrument(wavelength_angstrom, u_deg2, v_deg2, w_deg2, x_deg, y_deg),
        FcjGeometry {
            sample_over_radius,
            detector_over_radius,
        },
        SupportPolicy::FwhmMultiple(support_fwhm),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    accumulation_to_numpy(py, accumulation)
}

const fn cw_instrument(
    wavelength_angstrom: f64,
    u_deg2: f64,
    v_deg2: f64,
    w_deg2: f64,
    x_deg: f64,
    y_deg: f64,
) -> ConstantWavelengthInstrument {
    ConstantWavelengthInstrument {
        wavelength_angstrom,
        u_deg2,
        v_deg2,
        w_deg2,
        x_deg,
        y_deg,
    }
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
    module.add_function(wrap_pyfunction!(profile_fcj, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_tch, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_values, module)?)?;
    module.add_function(wrap_pyfunction!(cw_profile_parameters, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw, module)?)?;
    module.add_function(wrap_pyfunction!(accumulate_cw_fcj, module)?)?;
    module.add("PARAMETER_ORDER", ("intensity", "position", "fwhm", "eta"))?;
    module.add(
        "TCH_PARAMETER_ORDER",
        ("intensity", "position", "gaussian_fwhm", "lorentzian_fwhm"),
    )?;
    module.add("CW_LOCAL_PARAMETER_ORDER", ("intensity", "position"))?;
    module.add("CW_GLOBAL_PARAMETER_ORDER", ("u", "v", "w", "x", "y"))?;
    module.add(
        "CW_FCJ_GLOBAL_PARAMETER_ORDER",
        (
            "u",
            "v",
            "w",
            "x",
            "y",
            "sample_over_radius",
            "detector_over_radius",
        ),
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
