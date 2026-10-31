//! Typed `NumPy` boundary for the native indexed-spot geometry fitter.
use npy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};
use phasesmith_spots::{
    SpotError, geometry_refinement as kernel, prediction::Prediction, reciprocal::Geometry,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn triples(a: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    if a.as_array().ncols() != 3 {
        return Err(error("expected triples (n,3)"));
    }
    Ok(a.as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}
fn basis(a: PyReadonlyArray2<'_, f64>) -> PyResult<[[f64; 3]; 3]> {
    triples(a)?
        .try_into()
        .map_err(|_| error("basis must be (3,3)"))
}
fn geometry(a: PyReadonlyArray2<'_, f64>, wavelength: f64) -> PyResult<Geometry> {
    let v: [[f64; 3]; 5] = triples(a)?
        .try_into()
        .map_err(|_| error("geometry must be (5,3)"))?;
    Ok(Geometry {
        origin: v[0],
        fast: v[1],
        slow: v[2],
        beam: v[3],
        axis: v[4],
        wavelength,
    })
}
fn matrix(
    py: Python<'_>,
    values: Vec<f64>,
    rows: usize,
    columns: usize,
) -> Bound<'_, npy::PyArray2<f64>> {
    Array2::from_shape_vec((rows, columns), values)
        .expect("native validated matrix shape")
        .into_pyarray(py)
}
fn vectors(g: Geometry) -> Vec<f64> {
    [g.origin, g.fast, g.slow, g.beam, g.axis].concat()
}
fn predictions(py: Python<'_>, records: Vec<Prediction>) -> PyResult<Bound<'_, PyDict>> {
    let count = records.len();
    let mut values = Vec::with_capacity(count * 3);
    let mut jac = Vec::with_capacity(count * 27);
    let mut flags = Vec::with_capacity(count);
    for p in records {
        values.extend(p.values);
        jac.extend(p.jacobian);
        flags.push(p.flags);
    }
    let result = PyDict::new(py);
    result.set_item("values", matrix(py, values, count, 3))?;
    result.set_item("jacobian", matrix(py, jac, count, 27))?;
    result.set_item("flags", flags.into_pyarray(py))?;
    Ok(result)
}

#[pyfunction]
fn experimental_geometry_rotation(py: Python<'_>, angles: [f64; 3]) -> PyResult<Bound<'_, PyDict>> {
    if angles.iter().any(|v| !v.is_finite()) {
        return Err(error("rotation angles must be finite"));
    }
    let (r, d) = kernel::rotation(angles);
    let result = PyDict::new(py);
    result.set_item("rotation", matrix(py, r.concat(), 3, 3))?;
    result.set_item(
        "partials",
        matrix(py, d.into_iter().flat_map(|a| a.concat()).collect(), 9, 3),
    )?;
    Ok(result)
}
#[pyfunction]
fn experimental_geometry_reference_basis<'py>(
    py: Python<'py>,
    b: PyReadonlyArray2<'py, f64>,
    cell: [f64; 6],
) -> PyResult<Bound<'py, npy::PyArray2<f64>>> {
    Ok(matrix(
        py,
        kernel::reference_basis(basis(b)?, cell)
            .map_err(error)?
            .concat(),
        3,
        3,
    ))
}
#[pyfunction]
fn experimental_geometry_state<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    wavelength: f64,
    pivot: [f64; 3],
    lengths: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = parameters.as_slice()?;
    let state = kernel::parameter_state(p, basis(b)?, geometry(v, wavelength)?, pivot, lengths)
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item("basis", matrix(py, state.basis.concat(), 3, 3))?;
    result.set_item("vectors", matrix(py, vectors(state.geometry), 5, 3))?;
    result.set_item(
        "basis_jacobian",
        matrix(py, state.basis_jacobian, 9, p.len()),
    )?;
    result.set_item(
        "detector_jacobian",
        matrix(py, state.detector_jacobian, 9, p.len()),
    )?;
    Ok(result)
}

// Optional application-owned mapping bridge. The default fit and native Rust
// map implementations have no Python callbacks; only this explicit adapter does.
struct HostMap(Option<Py<PyAny>>);
impl kernel::PixelMap for HostMap {
    fn is_identity(&self) -> bool {
        self.0.is_none()
    }
    fn inverse(&self, plane: &[[f64; 2]]) -> Result<Vec<kernel::MappedPoint>, SpotError> {
        let Some(object) = &self.0 else {
            return kernel::PixelMap::inverse(&kernel::IdentityMap, plane);
        };
        Python::attach(|py| -> PyResult<Vec<kernel::MappedPoint>> {
            let input = matrix(
                py,
                plane.iter().flatten().copied().collect(),
                plane.len(),
                2,
            );
            input.call_method1("setflags", (false,))?;
            let image = object.call_method1(py, "from_plane", (&input,))?;
            let jac = object.call_method1(py, "inverse_jacobian", (&input,))?;
            let numpy = pyo3::types::PyModule::import(py, "numpy")?;
            for value in [&image, &jac] {
                let kind: String = value
                    .getattr(py, "dtype")?
                    .getattr(py, "kind")?
                    .extract(py)?;
                if !["f", "i", "u"].contains(&kind.as_str()) {
                    return Err(error(
                        "pixel inverse map/Jacobian must be finite real arrays",
                    ));
                }
            }
            let image = numpy
                .getattr("ascontiguousarray")?
                .call1((image, numpy.getattr("float64")?))?;
            let jac = numpy
                .getattr("ascontiguousarray")?
                .call1((jac, numpy.getattr("float64")?))?;
            let image = image.extract::<PyReadonlyArray2<'_, f64>>()?;
            let jac = jac.extract::<npy::PyReadonlyArray3<'_, f64>>()?;
            if image.as_array().dim() != (plane.len(), 2)
                || jac.as_array().dim() != (plane.len(), 2, 2)
            {
                return Err(error("pixel inverse map/Jacobian shape differs from input"));
            }
            let xy = image.as_slice()?;
            let j = jac.as_slice()?;
            Ok((0..plane.len())
                .map(|i| kernel::MappedPoint {
                    image: [xy[2 * i], xy[2 * i + 1]],
                    jacobian: [[j[4 * i], j[4 * i + 1]], [j[4 * i + 2], j[4 * i + 3]]],
                })
                .collect())
        })
        .map_err(|e| SpotError(format!("pixel coordinate map failed: {e}")))
    }
}

type OptionsTuple = (f64, f64, f64, usize, f64, f64, f64, f64, f64);
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)] // Flat typed boundary; native kernel takes Input/Options records.
#[pyo3(signature=(observed,hkl,b,v,wavelength,image_shape,training,eligible,reference_cell,lengths,options,progress=None,pixel_map=None))]
fn experimental_geometry_fit<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    hkl: PyReadonlyArray2<'py, i64>,
    b: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    wavelength: f64,
    image_shape: [usize; 2],
    training: PyReadonlyArray1<'py, bool>,
    eligible: PyReadonlyArray1<'py, bool>,
    reference_cell: Option<[f64; 6]>,
    lengths: bool,
    options: OptionsTuple,
    progress: Option<Py<PyAny>>,
    pixel_map: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    if hkl.as_array().ncols() != 3 {
        return Err(error("hkl must be (n,3)"));
    }
    if progress.as_ref().is_some_and(|p| !p.bind(py).is_callable()) {
        return Err(error("progress must be callable"));
    }
    let h: Vec<_> = hkl
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let obs = triples(observed)?;
    let b = basis(b)?;
    let g = geometry(v, wavelength)?;
    let train = training.as_slice()?.to_vec();
    let use_ = eligible.as_slice()?.to_vec();
    let options = kernel::Options {
        xy_scale: options.0,
        omega_scale: options.1,
        huber: options.2,
        max_evaluations: options.3,
        translation_limit_mm: options.4,
        rotation_limit_deg: options.5,
        log_length_limit: options.6,
        initial_xy_gate: options.7,
        initial_omega_gate: options.8,
    };
    let fit = py
        .detach(move || {
            kernel::fit_with_map(
                &kernel::Input {
                    observed: &obs,
                    hkl: &h,
                    training: &train,
                    eligible: &use_,
                    basis: b,
                    geometry: g,
                    image_shape,
                    reference_cell,
                    refine_lengths: lengths,
                },
                options,
                &HostMap(pixel_map),
                |done, total| {
                    if let Some(callback) = &progress {
                        Python::attach(|py| {
                            callback.call1(py, (done, total)).map(|_| ()).map_err(|e| {
                                SpotError(format!("geometry progress callback failed: {e}"))
                            })
                        })
                    } else {
                        Ok(())
                    }
                },
            )
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    let n = fit.parameters.len();
    result.set_item("parameters", fit.parameters.into_pyarray(py))?;
    result.set_item("basis", matrix(py, fit.state.basis.concat(), 3, 3))?;
    result.set_item("vectors", matrix(py, vectors(fit.state.geometry), 5, 3))?;
    result.set_item("initial", predictions(py, fit.initial)?)?;
    result.set_item("predictions", predictions(py, fit.predictions)?)?;
    result.set_item("eligible", fit.eligible.into_pyarray(py))?;
    result.set_item("converged", fit.converged)?;
    result.set_item("at_bounds", fit.at_bounds)?;
    result.set_item("condition", fit.scaled_condition)?;
    result.set_item("correlation", fit.maximum_parameter_correlation)?;
    result.set_item("covariance", matrix(py, fit.covariance, n, n))?;
    result.set_item("evaluations", fit.evaluations)?;
    result.set_item("history", fit.history.into_pyarray(py))?;
    result.set_item("counts", fit.counts)?;
    result.set_item("rms", fit.rms)?;
    Ok(result)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_geometry_rotation, module)?)?;
    module.add_function(wrap_pyfunction!(
        experimental_geometry_reference_basis,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(experimental_geometry_state, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_geometry_fit, module)?)
}
