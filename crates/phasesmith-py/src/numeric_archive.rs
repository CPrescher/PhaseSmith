use npy::PyReadonlyArray1;
use phasesmith_persistence::numeric_archive::{NumericArray, write_numeric_npz};
use pyo3::{
    exceptions::{PyIOError, PyValueError},
    prelude::*,
};
use std::path::PathBuf;

type ArrayInput<'py> = (String, String, Vec<usize>, PyReadonlyArray1<'py, u8>);

#[pyfunction]
fn native_write_npz(
    py: Python<'_>,
    path: PathBuf,
    arrays: Vec<ArrayInput<'_>>,
    level: u8,
) -> PyResult<()> {
    let mut total = 0_usize;
    let owned = arrays
        .iter()
        .map(|(name, dtype, shape, data)| {
            let data = data.as_slice()?;
            total = total
                .checked_add(data.len())
                .ok_or_else(|| PyValueError::new_err("archive size overflow"))?;
            if total > 1024 * 1024 * 1024 {
                return Err(PyValueError::new_err(
                    "numeric archive raw payload exceeds 1 GiB",
                ));
            }
            Ok((name.clone(), dtype.clone(), shape.clone(), data.to_vec()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| {
        let arrays: Vec<_> = owned
            .iter()
            .map(|(name, dtype, shape, bytes)| NumericArray {
                name,
                dtype,
                shape,
                bytes,
            })
            .collect();
        write_numeric_npz(&path, &arrays, level)
    })
    .map_err(|e| match e {
        phasesmith_persistence::PersistenceError::Io(e) => PyIOError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_write_npz, m)?)
}
