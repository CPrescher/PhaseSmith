//! Optional detector measurement boundary. Owned snapshots cross the GIL boundary.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use phasesmith_spots::{Bounds, Detection, Image, Options};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Parameters = (
    f64,
    f64,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type Arrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u32>>,
    usize,
);
fn options(p: Parameters) -> Options {
    Options {
        threshold_sigma: p.0,
        threshold_absolute: p.1,
        inner_radius: p.2,
        outer_radius: p.3,
        min_pixels: p.4,
        max_pixels: p.5,
        padding: p.6,
        background_margin: p.7,
        min_background: p.8,
        max_spots: p.9,
    }
}
fn arrays(py: Python<'_>, result: Detection) -> PyResult<Arrays<'_>> {
    let n = result.spots.len();
    let mut boxes = Vec::with_capacity(4 * n);
    let mut values = Vec::with_capacity(11 * n);
    let mut flags = Vec::with_capacity(n);
    for spot in result.spots {
        for v in [
            spot.bounds.x0,
            spot.bounds.y0,
            spot.bounds.x1,
            spot.bounds.y1,
        ] {
            boxes.push(i64::try_from(v).map_err(|_| PyValueError::new_err("box overflow"))?);
        }
        values.extend(spot.values);
        flags.push(spot.flags);
    }
    Ok((
        Array2::from_shape_vec((n, 4), boxes)
            .unwrap()
            .into_pyarray(py),
        Array2::from_shape_vec((n, 11), values)
            .unwrap()
            .into_pyarray(py),
        flags.into_pyarray(py),
        result.rejected_small,
    ))
}

#[pyfunction]
#[pyo3(signature = (data, height, width, mask, parameters, gain, read_noise, saturation=None, variance=None, boxes=None))]
#[allow(clippy::too_many_arguments)] // Flat, explicit array boundary; no Python in the numerical pass.
fn experimental_spots<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    height: usize,
    width: usize,
    mask: PyReadonlyArray1<'py, bool>,
    parameters: Parameters,
    gain: f64,
    read_noise: f64,
    saturation: Option<f64>,
    variance: Option<PyReadonlyArray1<'py, f64>>,
    boxes: Option<PyReadonlyArray2<'py, i64>>,
) -> PyResult<Arrays<'py>> {
    let data = data.as_slice()?.to_vec();
    let mask = mask.as_slice()?.to_vec();
    let variance = variance
        .map(|v| v.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let boxes = boxes
        .map(|b| -> PyResult<Vec<Bounds>> {
            let a = b.as_array();
            if a.ncols() != 4 {
                return Err(PyValueError::new_err("boxes must have shape (n,4)"));
            }
            a.rows()
                .into_iter()
                .map(|r| {
                    let mut v = [0; 4];
                    for (out, &input) in v.iter_mut().zip(r) {
                        *out = usize::try_from(input)
                            .map_err(|_| PyValueError::new_err("negative box bounds"))?;
                    }
                    Ok(Bounds {
                        x0: v[0],
                        y0: v[1],
                        x1: v[2],
                        y1: v[3],
                    })
                })
                .collect()
        })
        .transpose()?;
    let result = py
        .detach(move || {
            let image = Image {
                values: &data,
                height,
                width,
                mask: &mask,
                variance: variance.as_deref(),
                gain,
                read_noise,
                saturation,
            };
            let options = options(parameters);
            match boxes {
                None => phasesmith_spots::detect(&image, options),
                Some(boxes) => {
                    phasesmith_spots::extract(&image, &boxes, options).map(|spots| Detection {
                        spots,
                        rejected_small: 0,
                    })
                }
            }
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    arrays(py, result)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_spots, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_rotation_link, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_rotation_centroid, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_reciprocal, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_predict_spots, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_lattice_fourier, module)?)
}

#[pyfunction]
#[allow(clippy::type_complexity)] // Flat owned numerical snapshots.
fn experimental_predict_spots<'py>(
    py: Python<'py>,
    hkl: PyReadonlyArray2<'py, i64>,
    basis: PyReadonlyArray2<'py, f64>,
    hints: PyReadonlyArray1<'py, f64>,
    parameters: PyReadonlyArray2<'py, f64>,
    wavelength: f64,
    other: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u32>>,
)> {
    if hkl.as_array().ncols() != 3 {
        return Err(PyValueError::new_err("hkl must be (n,3)"));
    }
    let hkl: Vec<[i64; 3]> = hkl
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let hints = hints.as_slice()?.to_vec();
    let b = triples(basis)?;
    let p = triples(parameters)?;
    if b.len() != 3 || p.len() != 5 {
        return Err(PyValueError::new_err(
            "basis (3,3), geometry (5,3) required",
        ));
    }
    let geometry = phasesmith_spots::reciprocal::Geometry {
        origin: p[0],
        fast: p[1],
        slow: p[2],
        beam: p[3],
        axis: p[4],
        wavelength,
    };
    let result = py
        .detach(move || {
            phasesmith_spots::prediction::predict(geometry, [b[0], b[1], b[2]], &hkl, &hints, other)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = result.len();
    let mut values = Vec::with_capacity(n * 3);
    let mut jacobian = Vec::with_capacity(n * 27);
    let mut flags = Vec::with_capacity(n);
    for r in result {
        values.extend(r.values);
        jacobian.extend(r.jacobian);
        flags.push(r.flags);
    }
    Ok((
        Array2::from_shape_vec((n, 3), values)
            .unwrap()
            .into_pyarray(py),
        Array2::from_shape_vec((n, 27), jacobian)
            .unwrap()
            .into_pyarray(py),
        flags.into_pyarray(py),
    ))
}

fn triples(array: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    if array.as_array().ncols() != 3 {
        return Err(PyValueError::new_err("expected shape (n,3)"));
    }
    Ok(array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

#[pyfunction]
fn experimental_reciprocal<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    parameters: PyReadonlyArray2<'py, f64>,
    wavelength: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let samples = triples(samples)?;
    let p = triples(parameters)?;
    if p.len() != 5 {
        return Err(PyValueError::new_err("geometry must be (5,3)"));
    }
    let g = phasesmith_spots::reciprocal::Geometry {
        origin: p[0],
        fast: p[1],
        slow: p[2],
        beam: p[3],
        axis: p[4],
        wavelength,
    };
    let result = py
        .detach(move || g.convert(&samples))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(
        Array2::from_shape_vec((result.len(), 3), result.into_iter().flatten().collect())
            .unwrap()
            .into_pyarray(py),
    )
}

#[pyfunction]
fn experimental_lattice_fourier<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    vectors: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let points = triples(points)?;
    let vectors = triples(vectors)?;
    let result = py
        .detach(move || phasesmith_spots::reciprocal::fourier(&points, &vectors))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(
        Array2::from_shape_vec((result.len(), 13), result.into_iter().flatten().collect())
            .unwrap()
            .into_pyarray(py),
    )
}

#[pyfunction]
fn experimental_rotation_link<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray1<'py, u32>,
    xy: PyReadonlyArray2<'py, f64>,
    segments: PyReadonlyArray1<'py, u32>,
    radius: f64,
    max_candidates: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    if xy.as_array().ncols() != 2 {
        return Err(PyValueError::new_err("xy must have shape (n,2)"));
    }
    let frames = frames.as_slice()?.to_vec();
    let xy: Vec<[f64; 2]> = xy
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect();
    let segments = segments.as_slice()?.to_vec();
    let result = py
        .detach(move || {
            phasesmith_spots::rotation::link(&frames, &xy, &segments, radius, max_candidates)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn experimental_rotation_centroid<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if samples.as_array().ncols() != 4 {
        return Err(PyValueError::new_err("samples must have shape (n,4)"));
    }
    let samples: Vec<[f64; 4]> = samples
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();
    let result = py
        .detach(move || phasesmith_spots::rotation::summarize(&samples))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result.to_vec().into_pyarray(py))
}
