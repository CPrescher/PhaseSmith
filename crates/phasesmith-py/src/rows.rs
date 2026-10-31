//! Array boundary for the unchanged independent row-grouping kernel.
use std::sync::atomic::AtomicBool;

use npy::PyReadonlyArray2;
use phasesmith_row_search::{Options, Point, run};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
fn experimental_row_search(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    options_json: &str,
) -> PyResult<String> {
    let view = points.as_array();
    if view.ncols() != 3 || view.nrows() > 100_000 {
        return Err(PyValueError::new_err(
            "expected at most 100000 points of shape (n,3)",
        ));
    }
    // Own the snapshot before releasing the GIL, including noncontiguous inputs.
    let points: Vec<Point> = view.outer_iter().map(|r| [r[0], r[1], r[2]]).collect();
    if points
        .iter()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1000.)
    {
        return Err(PyValueError::new_err(
            "points must be finite with |q|<=1000",
        ));
    }
    let options: Options =
        serde_json::from_str(options_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    options.validate().map_err(PyValueError::new_err)?;
    // Serialization is outside the numerical core and also releases the GIL.
    py.detach(|| {
        let result = run(&points, &options, &AtomicBool::new(false))?;
        serde_json::to_string(&result).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(experimental_row_search, module)?)
}
