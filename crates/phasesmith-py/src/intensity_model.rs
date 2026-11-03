//! `NumPy` boundary for deterministic experimental intensity kernels.
#![allow(clippy::many_single_char_names)] // Conventional numerical array names at this small boundary.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use phasesmith_single_crystal::spots::intensity_model as kernel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_fixed_profile, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_normal, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_slab_response, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_empirical_model, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_family_center, module)?)
}

#[pyfunction]
fn experimental_slab_response<'py>(
    py: Python<'py>,
    cosines: PyReadonlyArray1<'py, f64>,
    thickness: f64,
    mu: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let c = cosines.as_slice()?.to_vec();
    let rows = py
        .detach(move || kernel::slab(&c, thickness, mu))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(
        Array2::from_shape_vec((rows.len(), 6), rows.into_iter().flatten().collect())
            .expect("six columns")
            .into_pyarray(py),
    )
}

type Evaluation<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);
#[pyfunction]
fn experimental_empirical_model<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray2<'py, f64>,
    detector: PyReadonlyArray2<'py, f64>,
    harmonics: PyReadonlyArray2<'py, f64>,
    parameters: PyReadonlyArray1<'py, f64>,
    floor: f64,
) -> PyResult<Evaluation<'py>> {
    let n = frames.as_array().nrows();
    if detector.as_array().nrows() != n || harmonics.as_array().nrows() != n {
        return Err(PyValueError::new_err("empirical design lengths differ"));
    }
    let widths = [
        frames.as_array().ncols(),
        detector.as_array().ncols(),
        harmonics.as_array().ncols(),
    ];
    let f = frames.as_slice()?.to_vec();
    let d = detector.as_slice()?.to_vec();
    let h = harmonics.as_slice()?.to_vec();
    let b = parameters.as_slice()?.to_vec();
    let (value, components, jac) = py
        .detach(move || {
            kernel::evaluate(
                &kernel::Design {
                    rows: n,
                    widths,
                    matrices: [&f, &d, &h],
                },
                &b,
                floor,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        value.into_pyarray(py),
        Array2::from_shape_vec((n, 2), components)
            .expect("two components")
            .into_pyarray(py),
        Array2::from_shape_vec((n, widths.iter().sum()), jac)
            .expect("parameter columns")
            .into_pyarray(py),
    ))
}

#[pyfunction]
fn experimental_family_center<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    groups: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = (values.as_array().nrows(), values.as_array().ncols());
    if groups.as_array().len() != shape.0 || weights.as_array().len() != shape.0 {
        return Err(PyValueError::new_err("family array lengths differ"));
    }
    let v = values.as_slice()?.to_vec();
    let w = weights.as_slice()?.to_vec();
    let g: Vec<usize> = groups
        .as_slice()?
        .iter()
        .map(|&a| usize::try_from(a).map_err(|_| PyValueError::new_err("negative family")))
        .collect::<PyResult<_>>()?;
    let out = py
        .detach(move || kernel::center(&v, shape.1, &g, &w))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Array2::from_shape_vec(shape, out)
        .expect("centered shape")
        .into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_profile_normal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    basis: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    covariance: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = signal.as_array().len();
    let cols = design.as_array().ncols();
    if basis.as_array().dim() != (n, 3)
        || design.as_array().nrows() != n
        || covariance.as_array().ncols() != 9
    {
        return Err(PyValueError::new_err("GLS shape mismatch"));
    }
    let y = signal.as_slice()?.to_vec();
    let w = weights.as_slice()?.to_vec();
    let x: Vec<[f64; 3]> = basis
        .as_slice()?
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let c: Vec<[f64; 9]> = covariance
        .as_slice()?
        .as_chunks::<9>()
        .0
        .iter()
        .map(|r| std::array::from_fn(|i| r[i]))
        .collect();
    let o: Vec<usize> = offsets
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err("negative offset")))
        .collect::<PyResult<_>>()?;
    let d = design.as_slice()?.to_vec();
    let out = py
        .detach(move || kernel::normal(&y, &w, &x, &o, &c, &d, cols))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Array2::from_shape_vec((cols + 1, cols + 1), out)
        .expect("GLS Gram shape")
        .into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_fixed_profile<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    variance: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    basis: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    covariance: PyReadonlyArray2<'py, f64>,
    mass: PyReadonlyArray1<'py, f64>,
) -> PyResult<[f64; 2]> {
    let n = signal.as_array().len();
    if basis.as_array().dim() != (n, 3) || covariance.as_array().ncols() != 9 {
        return Err(PyValueError::new_err("fixed profile shape mismatch"));
    }
    let y = signal.as_slice()?.to_vec();
    let v = variance.as_slice()?.to_vec();
    let w = weights.as_slice()?.to_vec();
    let p = mass.as_slice()?.to_vec();
    let x: Vec<[f64; 3]> = basis
        .as_slice()?
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let c: Vec<[f64; 9]> = covariance
        .as_slice()?
        .as_chunks::<9>()
        .0
        .iter()
        .map(|r| std::array::from_fn(|i| r[i]))
        .collect();
    let o: Vec<usize> = offsets
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err("negative offset")))
        .collect::<PyResult<_>>()?;
    py.detach(move || kernel::fixed_filter(&y, &v, &w, &x, &o, &c, &p))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
