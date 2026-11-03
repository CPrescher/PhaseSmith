//! Array-boundary adapters for native shadow kernels; no numerical callbacks.

use npy::ndarray::{Array2, Array3};
use npy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use phasesmith_single_crystal::spots::{frame_validity as kernel, reciprocal::Geometry};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(phasesmith, ExperimentalSeatShadowUnavailable, PyValueError);

fn error(value: kernel::Error) -> PyErr {
    match value {
        kernel::Error::Invalid(message) => PyValueError::new_err(message),
        kernel::Error::Unavailable(message) => ExperimentalSeatShadowUnavailable::new_err(message),
    }
}

#[pyfunction]
fn experimental_shadow_coarse<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    block: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, bool)> {
    let (height, width) = image.as_array().dim();
    if mask.as_array().dim() != (height, width) {
        return Err(PyValueError::new_err("frame/mask shape mismatch"));
    }
    let shape = kernel::Shape {
        height,
        width,
        block,
    };
    let [h, w] = shape.coarse().map_err(error)?;
    let values = image.as_slice()?.to_vec();
    let excluded = mask.as_slice()?.to_vec();
    let result = py
        .detach(move || kernel::coarsen(&values, &excluded, shape))
        .map_err(error)?;
    Ok((
        Array2::from_shape_vec((h, w), result.values)
            .expect("validated coarse shape")
            .into_pyarray(py),
        result.available,
    ))
}

type EstimateArrays<'py> = (
    Bound<'py, PyArray3<bool>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Typed array boundary, not a core model record.
#[pyo3(signature=(coarse, available, height, width, block, threshold, guard_pixels, reference_first=None, reference_last=None))]
fn experimental_shadow_estimate<'py>(
    py: Python<'py>,
    coarse: PyReadonlyArray3<'py, f64>,
    available: PyReadonlyArray1<'py, bool>,
    height: usize,
    width: usize,
    block: usize,
    threshold: f64,
    guard_pixels: usize,
    reference_first: Option<usize>,
    reference_last: Option<usize>,
) -> PyResult<EstimateArrays<'py>> {
    let shape = kernel::Shape {
        height,
        width,
        block,
    };
    let [h, w] = shape.coarse().map_err(error)?;
    let frames = available.len()?;
    if coarse.as_array().dim() != (frames, h, w) {
        return Err(PyValueError::new_err("coarse scan shape mismatch"));
    }
    let values = coarse.as_slice()?.to_vec();
    let available = available.as_slice()?.to_vec();
    let result = py
        .detach(move || {
            kernel::estimate(
                &values,
                &available,
                shape,
                kernel::SeatOptions {
                    reference_first,
                    reference_last,
                    threshold,
                    guard_pixels,
                },
            )
        })
        .map_err(error)?;
    let indices = |values: Vec<usize>| {
        values
            .into_iter()
            .map(|v| i64::try_from(v).expect("bounded frame index"))
            .collect::<Vec<_>>()
    };
    Ok((
        Array3::from_shape_vec((frames, h, w), result.excluded)
            .expect("validated scan shape")
            .into_pyarray(py),
        result.detection_scales.into_pyarray(py),
        indices(result.reference_indices).into_pyarray(py),
        indices(result.known_unavailable_frames).into_pyarray(py),
    ))
}

fn triples(array: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    if array.as_array().ncols() != 3 {
        return Err(PyValueError::new_err("expected (n,3) triples"));
    }
    if array.as_array().iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("triples must be finite"));
    }
    Ok(array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Typed array boundary; core uses Shape/DacOptions.
fn experimental_dac_mask<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    bounds: (f64, f64),
    height: usize,
    width: usize,
    block: usize,
    axis_at_zero: [f64; 3],
    half_opening_degrees: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let v = triples(vectors)?;
    if v.len() != 5 {
        return Err(PyValueError::new_err("expected five detector vectors"));
    }
    let geometry = Geometry {
        origin: v[0],
        fast: v[1],
        slow: v[2],
        beam: v[3],
        axis: v[4],
        wavelength: 1.,
    };
    let shape = kernel::Shape {
        height,
        width,
        block,
    };
    let [h, w] = shape.coarse().map_err(error)?;
    let result = py
        .detach(move || {
            kernel::dac_mask(
                geometry,
                [bounds.0, bounds.1],
                shape,
                kernel::DacOptions {
                    axis_at_zero,
                    half_opening_degrees,
                },
            )
        })
        .map_err(error)?;
    Ok(Array2::from_shape_vec((h, w), result.excluded)
        .expect("validated DAC shape")
        .into_pyarray(py))
}

#[pyfunction]
fn experimental_mask_dot_range<'py>(
    py: Python<'py>,
    rays: PyReadonlyArray2<'py, f64>,
    axis: [f64; 3],
    normal: [f64; 3],
    low: f64,
    high: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !low.is_finite()
        || !high.is_finite()
        || high < low
        || axis.iter().chain(&normal).any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "invalid dot-range vectors or angular interval",
        ));
    }
    let rays = triples(rays)?;
    let result: Vec<_> = py.detach(move || {
        rays.into_iter()
            .flat_map(|r| kernel::dot_range(r, axis, normal, low, high))
            .collect()
    });
    Ok(Array2::from_shape_vec((result.len() / 2, 2), result)
        .expect("range pairs")
        .into_pyarray(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "ExperimentalSeatShadowUnavailable",
        module.py().get_type::<ExperimentalSeatShadowUnavailable>(),
    )?;
    module.add_function(wrap_pyfunction!(experimental_shadow_coarse, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_shadow_estimate, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_dac_mask, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_mask_dot_range, module)?)
}
