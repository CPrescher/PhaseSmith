//! Typed `NumPy` snapshots for experimental empirical profiles; no Python in kernels.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use phasesmith_spots::profile_refinement as refinement;
use phasesmith_spots::{Bounds, Detection, Image, profiles as kernel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
#[pyo3(signature=(observations,regions,references,shape,step,models,minimum,snr,cut,iterations,variance_weighted,workers=1,cache_bytes=67_108_864))]
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
    workers: usize,
    cache_bytes: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyDict>,
)> {
    let owned = Owned::new(observations)?;
    let regions = integers(regions)?;
    let references = references.as_slice()?.to_vec();
    let result = py
        .detach(move || {
            kernel::learn_with_execution(
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
                kernel::Execution {
                    workers,
                    geometry_cache_bytes: cache_bytes,
                },
            )
        })
        .map_err(error)?;
    let stats = PyDict::new(py);
    stats.set_item("workers", result.execution.workers)?;
    stats.set_item("cache_budget_bytes", result.execution.cache_budget_bytes)?;
    stats.set_item("cache_bytes", result.execution.cache_bytes)?;
    stats.set_item("cache_rows", result.execution.cache_rows)?;
    stats.set_item("cache_entries", result.execution.cache_entries)?;
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
        stats,
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
    module.add_class::<NativeProfileSpot>()?;
    module.add_function(wrap_pyfunction!(experimental_profile_refine, module)?)?;
    module.add_function(wrap_pyfunction!(experimental_profile_centroids, module)?)?;
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

// Small typed controls keep the native numerical API independent of Python objects.
type ShapeOptions = (
    usize,
    usize,
    usize,
    f64,
    f64,
    [f64; 2],
    [f64; 2],
    [f64; 5],
    f64,
    f64,
);
fn shape_options(o: ShapeOptions) -> refinement::Options {
    refinement::Options {
        workers: o.0,
        iterations: o.1,
        maximum_iterations: o.2,
        centre_pixels: o.3,
        centre_degrees: o.4,
        spatial_range: o.5,
        rocking_range: o.6,
        regularizer: o.7,
        weak_design_counts: o.8,
        weak_spatial_width: o.9,
    }
}
/// Diagnostic adapter to the same native per-reflection objective used by the batch.
#[pyclass]
struct NativeProfileSpot {
    inner: refinement::PreparedSpot,
}
#[pymethods]
impl NativeProfileSpot {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        observations: Input<'_>,
        profiles: PyReadonlyArray2<'_, f64>,
        shape: [usize; 3],
        step: [f64; 3],
        bank: usize,
        shifts: PyReadonlyArray2<'_, f64>,
        minimum_mass: f64,
        expected: bool,
        gain: f64,
        read: f64,
    ) -> PyResult<Self> {
        let owned = Owned::new(observations)?;
        Ok(Self {
            inner: refinement::PreparedSpot::new(
                &owned.borrowed(),
                0,
                profiles.as_slice()?,
                kernel::Grid { shape, step },
                bank,
                &rows(shifts)?,
                minimum_mass,
                refinement::Noise {
                    expected,
                    gain,
                    read,
                },
            )
            .map_err(error)?,
        })
    }
    fn mass<'py>(
        &self,
        py: Python<'py>,
        parameters: [f64; 5],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(matrix(
            py,
            py.detach(|| self.inner.mass(parameters)).map_err(error)?,
        ))
    }
    fn set_noise(
        &mut self,
        mass: PyReadonlyArray1<'_, f64>,
        intensity: f64,
        foreground: bool,
    ) -> PyResult<()> {
        self.inner
            .set_noise(mass.as_slice()?, intensity, foreground)
            .map_err(error)
    }
    fn normal<'py>(
        &self,
        py: Python<'py>,
        design: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = design.as_array().ncols();
        let design = design.as_slice()?.to_vec();
        let out = py.detach(|| self.inner.normal(&design, n)).map_err(error)?;
        Ok(Array2::from_shape_vec((n + 1, n + 1), out)
            .expect("validated Gram")
            .into_pyarray(py))
    }
    fn objective<'py>(
        &self,
        py: Python<'py>,
        parameters: [f64; 5],
        regularizer: [f64; 5],
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let out = py
            .detach(|| self.inner.objective(parameters, regularizer))
            .map_err(error)?;
        Ok((out.cost, out.gradient.into_pyarray(py)))
    }
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.snapshot();
        let d = PyDict::new(py);
        d.set_item("signal", s.signal.to_vec().into_pyarray(py))?;
        d.set_item("weights", s.weights.to_vec().into_pyarray(py))?;
        d.set_item("basis", matrix(py, s.basis.to_vec()))?;
        d.set_item("bounds", matrix(py, s.bounds.to_vec()))?;
        d.set_item("cov", matrix(py, s.covariance.to_vec()))?;
        d.set_item(
            "offsets",
            s.offsets
                .iter()
                .map(|&v| i64::try_from(v).expect("sample index"))
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        Ok(d)
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_profile_centroids<'py>(
    py: Python<'py>,
    profiles: PyReadonlyArray2<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    shifts: PyReadonlyArray2<'py, f64>,
    assignments: PyReadonlyArray1<'py, i64>,
    scales: PyReadonlyArray2<'py, f64>,
    planar: PyReadonlyArray2<'py, f64>,
    log_width: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let profiles = profiles.as_slice()?.to_vec();
    let shifts = rows(shifts)?;
    let ids = integers(assignments)?;
    let scales = rows(scales)?;
    let planar = rows(planar)?;
    let widths = log_width.as_slice()?.to_vec();
    Ok(matrix(
        py,
        py.detach(|| {
            refinement::centroid_offsets(
                &profiles,
                kernel::Grid { shape, step },
                &shifts,
                &ids,
                &scales,
                &planar,
                &widths,
            )
        })
        .map_err(error)?,
    ))
}
#[pyfunction]
#[pyo3(signature=(observations,profiles,shape,step,assignments,shifts,minimum_mass,expected,gain,read,choices,weak,sensor_offsets,options,baseline_values,baseline_flags,frozen=None,progress=None))]
#[allow(clippy::too_many_arguments)]
fn experimental_profile_refine<'py>(
    py: Python<'py>,
    observations: Input<'py>,
    profiles: PyReadonlyArray2<'py, f64>,
    shape: [usize; 3],
    step: [f64; 3],
    assignments: PyReadonlyArray1<'py, i64>,
    shifts: PyReadonlyArray2<'py, f64>,
    minimum_mass: f64,
    expected: bool,
    gain: f64,
    read: f64,
    choices: PyReadonlyArray1<'py, u8>,
    weak: PyReadonlyArray1<'py, bool>,
    sensor_offsets: PyReadonlyArray2<'py, f64>,
    options: ShapeOptions,
    baseline_values: PyReadonlyArray2<'py, f64>,
    baseline_flags: PyReadonlyArray1<'py, u32>,
    frozen: Option<PyReadonlyArray2<'py, f64>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned = Owned::new(observations)?;
    let profiles = profiles.as_slice()?.to_vec();
    let assignments = integers(assignments)?;
    let shifts = rows(shifts)?;
    let choices = choices.as_slice()?.to_vec();
    let weak = weak.as_slice()?.to_vec();
    let sensor_offsets = rows(sensor_offsets)?;
    let baseline_values = rows(baseline_values)?;
    let baseline_flags = baseline_flags.as_slice()?.to_vec();
    let frozen = frozen.map(rows).transpose()?;
    let mut callback_error = None;
    let result = py.detach(|| {
        refinement::refine(
            &refinement::Input {
                observations: owned.borrowed(),
                profiles: &profiles,
                grid: kernel::Grid { shape, step },
                assignments: &assignments,
                shifts: &shifts,
                minimum_mass,
                noise: refinement::Noise {
                    expected,
                    gain,
                    read,
                },
                choices: &choices,
                weak: &weak,
                sensor_offsets: &sensor_offsets,
                frozen: frozen.as_deref(),
                baseline: Some(refinement::Baseline {
                    values: &baseline_values,
                    flags: &baseline_flags,
                }),
            },
            shape_options(options),
            |done, total| {
                if let Some(cb) = &progress {
                    if let Err(e) = Python::attach(|py| cb.call1(py, (done, total))) {
                        callback_error = Some(e);
                        return Err(phasesmith_spots::SpotError(
                            "profile refinement cancelled".into(),
                        ));
                    }
                }
                Ok(())
            },
        )
    });
    if let Some(e) = callback_error {
        return Err(e);
    }
    let result = result.map_err(error)?;
    let d = PyDict::new(py);
    d.set_item("values", matrix(py, result.values))?;
    d.set_item("flags", result.flags.into_pyarray(py))?;
    d.set_item("parameters", matrix(py, result.parameters))?;
    d.set_item("converged", result.converged.into_pyarray(py))?;
    d.set_item("evaluations", result.evaluations)?;
    Ok(d)
}
