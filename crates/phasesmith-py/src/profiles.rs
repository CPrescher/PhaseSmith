//! Typed `NumPy` snapshots for experimental empirical profiles; no Python in kernels.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use phasesmith_spots::{Bounds, Detection, Image, profiles as kernel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Input<'py> = (
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray1<'py, bool>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray2<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
);
struct Owned {
    samples: Vec<[f64; 3]>,
    valid: Vec<bool>,
    offsets: Vec<usize>,
    boxes: Vec<Bounds>,
    ids: Vec<usize>,
    angles: Vec<[f64; 2]>,
    covariance: Vec<[f64; 9]>,
    centres: Vec<[f64; 3]>,
    flags: Vec<u32>,
    scales: Vec<[f64; 3]>,
    planar: Vec<[f64; 4]>,
}
fn integers(a: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    a.as_slice()?
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err("negative index")))
        .collect()
}
fn rows<const N: usize>(a: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; N]>> {
    if a.as_array().ncols() != N {
        return Err(PyValueError::new_err("invalid numerical columns"));
    }
    Ok(a.as_array()
        .rows()
        .into_iter()
        .map(|r| std::array::from_fn(|j| r[j]))
        .collect())
}
fn boxes(a: PyReadonlyArray2<'_, i64>) -> PyResult<Vec<Bounds>> {
    if a.as_array().ncols() != 4 {
        return Err(PyValueError::new_err("boxes must be (n,4)"));
    }
    a.as_array()
        .rows()
        .into_iter()
        .map(|r| {
            let mut v = [0; 4];
            for (out, &input) in v.iter_mut().zip(r) {
                *out = usize::try_from(input).map_err(|_| PyValueError::new_err("negative box"))?;
            }
            Ok(Bounds {
                x0: v[0],
                y0: v[1],
                x1: v[2],
                y1: v[3],
            })
        })
        .collect()
}
impl Owned {
    fn new(a: Input<'_>) -> PyResult<Self> {
        Ok(Self {
            samples: rows(a.0)?,
            valid: a.1.as_slice()?.to_vec(),
            offsets: integers(a.2)?,
            boxes: boxes(a.3)?,
            ids: integers(a.4)?,
            angles: rows(a.5)?,
            covariance: rows(a.6)?,
            centres: rows(a.7)?,
            flags: a.8.as_slice()?.to_vec(),
            scales: rows(a.9)?,
            planar: rows(a.10)?,
        })
    }
    fn borrowed(&self) -> kernel::Observations<'_> {
        kernel::Observations {
            samples: &self.samples,
            valid: &self.valid,
            offsets: &self.offsets,
            boxes: &self.boxes,
            ids: &self.ids,
            angles: &self.angles,
            covariance: &self.covariance,
            centres: &self.centres,
            flags: &self.flags,
            scales: &self.scales,
            planar: &self.planar,
        }
    }
}
fn error(e: phasesmith_spots::SpotError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn matrix<const N: usize>(py: Python<'_>, a: Vec<[f64; N]>) -> Bound<'_, PyArray2<f64>> {
    Array2::from_shape_vec((a.len(), N), a.into_iter().flatten().collect())
        .unwrap()
        .into_pyarray(py)
}
#[pyfunction]
fn experimental_profile_evaluate<'py>(
    py: Python<'py>,
    profile: PyReadonlyArray1<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    bounds: PyReadonlyArray2<'py, f64>,
    centre: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let profile = profile.as_slice()?.to_vec();
    let bounds = rows(bounds)?;
    let result = py
        .detach(move || kernel::evaluate(&profile, kernel::Grid { shape, step }, &bounds, centre))
        .map_err(error)?;
    Ok(matrix(py, result))
}
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn experimental_profile_learn<'py>(
    py: Python<'py>,
    observations: Input<'py>,
    regions: PyReadonlyArray1<'py, i64>,
    references: PyReadonlyArray1<'py, bool>,
    shape: [usize; 3],
    step: [f64; 3],
    models: usize,
    minimum: usize,
    snr: f64,
    cut: f64,
    iterations: usize,
    variance_weighted: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let owned = Owned::new(observations)?;
    let regions = integers(regions)?;
    let references = references.as_slice()?.to_vec();
    let result = py
        .detach(move || {
            kernel::learn(
                &owned.borrowed(),
                &regions,
                &references,
                kernel::Grid { shape, step },
                models,
                minimum,
                snr,
                cut,
                iterations,
                variance_weighted,
            )
        })
        .map_err(error)?;
    let profiles =
        Array2::from_shape_vec((models, result.profiles.len() / models), result.profiles)
            .unwrap()
            .into_pyarray(py);
    let counts = result
        .counts
        .into_iter()
        .map(|v| i64::try_from(v).unwrap())
        .collect::<Vec<_>>()
        .into_pyarray(py);
    Ok((
        profiles,
        counts,
        matrix(py, result.shifts),
        result.used.into_pyarray(py),
        matrix(py, result.reference_centres),
    ))
}
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn experimental_profile_fit<'py>(
    py: Python<'py>,
    observations: Input<'py>,
    profiles: PyReadonlyArray2<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    assignments: PyReadonlyArray1<'py, i64>,
    shifts: PyReadonlyArray2<'py, f64>,
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<u32>>)> {
    let owned = Owned::new(observations)?;
    let profiles = profiles.as_slice()?.to_vec();
    let assignments = integers(assignments)?;
    let shifts = rows(shifts)?;
    let (values, flags) = py
        .detach(move || {
            kernel::fit(
                &owned.borrowed(),
                &profiles,
                kernel::Grid { shape, step },
                &assignments,
                &shifts,
                minimum_mass,
                expected_noise,
                gain,
                read_noise,
            )
        })
        .map_err(error)?;
    Ok((matrix(py, values), flags.into_pyarray(py)))
}
#[pyfunction]
#[pyo3(signature=(data,height,width,mask,box_array,gap,margin,minimum,clip,gain,read_noise,saturation=None,variance=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn experimental_profile_frame<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    height: usize,
    width: usize,
    mask: PyReadonlyArray1<'py, bool>,
    box_array: PyReadonlyArray2<'py, i64>,
    gap: usize,
    margin: usize,
    minimum: usize,
    clip: f64,
    gain: f64,
    read_noise: f64,
    saturation: Option<f64>,
    variance: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
)> {
    let data = data.as_slice()?.to_vec();
    let mask = mask.as_slice()?.to_vec();
    let boxes = boxes(box_array)?;
    let variance = variance
        .map(|v| v.as_slice().map(<[f64]>::to_vec))
        .transpose()?;
    let (spots, diagnostics, covariance, samples, valid) = py
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
            kernel::extract_frame(&image, &boxes, gap, margin, minimum, clip)
        })
        .map_err(error)?;
    let (bounds, values, flags, _) = super::spots::arrays(
        py,
        Detection {
            spots,
            rejected_small: 0,
        },
    )?;
    Ok((
        bounds,
        values,
        flags,
        matrix(py, diagnostics),
        matrix(py, covariance),
        matrix(py, samples),
        valid.into_pyarray(py),
    ))
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        experimental_profile_evaluate_shape,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(
        experimental_profile_evaluate_affine,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_learn, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_fit, module)?)?;
    module.add_function(wrap_pyfunction!(
        experimental_profile_fit_variations,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_frame, module)?)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_profile_evaluate_shape<'py>(
    py: Python<'py>,
    profile: PyReadonlyArray1<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    bounds: PyReadonlyArray2<'py, f64>,
    centre: [f64; 3],
    planar: [f64; 4],
    scales: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = profile.as_slice()?.to_vec();
    let b = rows(bounds)?;
    let out = py
        .detach(move || {
            kernel::evaluate_shape(&p, kernel::Grid { shape, step }, &b, centre, planar, scales)
        })
        .map_err(error)?;
    Ok(matrix(py, out))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_profile_evaluate_affine<'py>(
    py: Python<'py>,
    profile: PyReadonlyArray1<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    bounds: PyReadonlyArray2<'py, f64>,
    centre: [f64; 3],
    planar: [f64; 4],
    scales: [f64; 3],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = profile.as_slice()?.to_vec();
    let b = rows(bounds)?;
    let out = py
        .detach(move || {
            kernel::evaluate_affine(&p, kernel::Grid { shape, step }, &b, centre, planar, scales)
        })
        .map_err(error)?;
    Ok(matrix(py, out))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn experimental_profile_fit_variations<'py>(
    py: Python<'py>,
    observations: Input<'py>,
    profiles: PyReadonlyArray2<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    assignments: PyReadonlyArray1<'py, i64>,
    shifts: PyReadonlyArray2<'py, f64>,
    factors: PyReadonlyArray2<'py, f64>,
    fixed: PyReadonlyArray2<'py, f64>,
    minimum_mass: f64,
    expected_noise: bool,
    gain: f64,
    read_noise: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let owned = Owned::new(observations)?;
    let profiles = profiles.as_slice()?.to_vec();
    let assignments = integers(assignments)?;
    let shifts = rows(shifts)?;
    let factors = rows(factors)?;
    let fixed: Vec<[f64; 4]> = rows(fixed)?;
    let result = py
        .detach(move || {
            phasesmith_spots::profiles_variation::fit(
                &owned.borrowed(),
                &profiles,
                kernel::Grid { shape, step },
                &assignments,
                &shifts,
                &factors,
                if fixed.is_empty() { None } else { Some(&fixed) },
                minimum_mass,
                expected_noise,
                gain,
                read_noise,
            )
        })
        .map_err(error)?;
    Ok((
        matrix(py, result.values),
        result.flags.into_pyarray(py),
        matrix(py, result.coefficients),
        matrix(py, result.covariance),
    ))
}
