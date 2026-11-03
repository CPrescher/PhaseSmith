//! Opt-in omega arPLS array boundary. Owned inputs allow GIL-free fitting.
use npy::ndarray::Array2;
use npy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use phasesmith_single_crystal::spots::omega_background::{Engine, Parameters, Solver};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Arrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<bool>>,
);
#[pyclass(module = "phasesmith._core")]
struct ExperimentalArPLSEngine {
    engine: Engine,
}
#[pymethods]
impl ExperimentalArPLSEngine {
    #[new]
    #[pyo3(signature = (workers, solver="cholesky"))]
    fn new(workers: usize, solver: &str) -> PyResult<Self> {
        let solver = match solver {
            "cholesky" => Solver::Cholesky,
            "ldlt" => Solver::Ldlt,
            _ => return Err(PyValueError::new_err("Solver must be cholesky or ldlt")),
        };
        Ok(Self {
            engine: Engine::with_solver(workers, solver).map_err(PyValueError::new_err)?,
        })
    }
    #[allow(clippy::too_many_arguments)] // Explicit flat numerical boundary.
    fn baselines<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        excluded: Option<PyReadonlyArray2<'py, bool>>,
        smoothness: f64,
        tolerance: f64,
        max_iterations: i32,
        minimum_frames: usize,
    ) -> PyResult<Arrays<'py>> {
        let shape = data.as_array().dim();
        if excluded
            .as_ref()
            .is_some_and(|m| m.as_array().dim() != shape)
        {
            return Err(PyValueError::new_err(
                "Mask shape differs from pixel/frame rows",
            ));
        }
        let data = data.as_slice()?.to_vec();
        let excluded = excluded
            .map(|m| m.as_slice().map(<[bool]>::to_vec))
            .transpose()?;
        let result = py
            .detach(|| {
                self.engine.baselines(
                    &data,
                    shape.1,
                    excluded.as_deref(),
                    Parameters {
                        smoothness,
                        tolerance,
                        max_iterations,
                    },
                    minimum_frames,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            Array2::from_shape_vec(shape, result.baseline)
                .unwrap()
                .into_pyarray(py),
            result.iterations.into_pyarray(py),
            result.converged.into_pyarray(py),
        ))
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ExperimentalArPLSEngine>()?;
    module.add("OMEGA_BACKGROUND_SOLVERS", ("cholesky", "ldlt"))
}
