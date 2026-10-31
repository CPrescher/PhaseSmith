//! Validated array boundary for the native empirical-scaling optimizer.
use npy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};
use phasesmith_spots::{empirical_fit as kernel, intensity_model::Design};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn experimental_empirical_fit<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray2<'py, f64>,
    detector: PyReadonlyArray2<'py, f64>,
    harmonics: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    intensity: PyReadonlyArray1<'py, f64>,
    variance: PyReadonlyArray1<'py, f64>,
    groups: PyReadonlyArray1<'py, i64>,
    references: PyReadonlyArray1<'py, bool>,
    lower: PyReadonlyArray1<'py, f64>,
    upper: PyReadonlyArray1<'py, f64>,
    start: PyReadonlyArray1<'py, f64>,
    strength: f64,
    huber: f64,
    iterations: usize,
    minimum_component: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let n = frames.as_array().nrows();
    let widths = [
        frames.as_array().ncols(),
        detector.as_array().ncols(),
        harmonics.as_array().ncols(),
    ];
    let p: usize = widths.iter().sum();
    if detector.as_array().nrows() != n
        || harmonics.as_array().nrows() != n
        || penalty.as_array().ncols() != p
    {
        return Err(error("scaling design shapes differ"));
    }
    let matrices = [
        frames.as_slice()?.to_vec(),
        detector.as_slice()?.to_vec(),
        harmonics.as_slice()?.to_vec(),
    ];
    let penalty = penalty.as_slice()?.to_vec();
    let intensity = intensity.as_slice()?.to_vec();
    let variance = variance.as_slice()?.to_vec();
    let references = references.as_slice()?.to_vec();
    let groups: Vec<_> = groups
        .as_slice()?
        .iter()
        .map(|&v| usize::try_from(v).map_err(error))
        .collect::<PyResult<_>>()?;
    let lower = lower.as_slice()?.to_vec();
    let upper = upper.as_slice()?.to_vec();
    let start = start.as_slice()?.to_vec();
    let fit = py
        .detach(move || {
            kernel::fit(
                &kernel::Input {
                    design: Design {
                        rows: n,
                        widths,
                        matrices: [&matrices[0], &matrices[1], &matrices[2]],
                    },
                    penalty: &penalty,
                    intensity: &intensity,
                    variance: &variance,
                    groups: &groups,
                    reference: &references,
                    lower: &lower,
                    upper: &upper,
                    start: &start,
                },
                kernel::Options {
                    strength,
                    huber,
                    iterations,
                    minimum_component,
                },
            )
        })
        .map_err(error)?;
    let result = PyDict::new(py);
    result.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    result.set_item("correction", fit.correction.into_pyarray(py))?;
    result.set_item(
        "components",
        Array2::from_shape_vec((n, 2), fit.components)
            .expect("validated components")
            .into_pyarray(py),
    )?;
    result.set_item("references", fit.references.into_pyarray(py))?;
    result.set_item("converged", fit.converged)?;
    let history: Vec<f64> = fit
        .history
        .iter()
        .flat_map(|h| {
            [
                h.parameter_change,
                h.weight_change,
                f64::from(h.inner_success),
            ]
        })
        .collect();
    result.set_item(
        "history",
        Array2::from_shape_vec((fit.history.len(), 3), history)
            .expect("iteration history")
            .into_pyarray(py),
    )?;
    result.set_item(
        "evaluations",
        fit.history
            .iter()
            .map(|h| h.evaluations)
            .collect::<Vec<_>>(),
    )?;
    Ok(result)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_empirical_fit, module)?)?;
    Ok(())
}
