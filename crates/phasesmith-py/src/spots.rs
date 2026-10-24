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
    module.add_function(wrap_pyfunction!(experimental_spots, module)?)
}
