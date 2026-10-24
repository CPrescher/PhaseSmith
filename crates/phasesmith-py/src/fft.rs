//! Copies `NumPy` inputs before releasing the GIL. Rust never reads a Python
//! buffer concurrently with a possible mutation from another Python thread.
use npy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, ndarray};
use phasesmith_spots::fft::{Engine, Field, M3, V3};
use pyo3::{exceptions::PyValueError, prelude::*};

type ScoredPoints<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
type FourierMaxima<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f32>>, usize);

fn invalid(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn points(a: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<V3>> {
    let a = a.as_array();
    if a.ncols() != 3 || a.nrows() > 100_000 {
        return Err(invalid("Expected shape (N,3)"));
    }
    let v: Vec<V3> = a.outer_iter().map(|r| [r[0], r[1], r[2]]).collect();
    if v.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("Nonfinite coordinate"));
    }
    Ok(v)
}
fn matrix(a: PyReadonlyArray2<'_, f64>) -> PyResult<M3> {
    let v = points(a)?;
    v.try_into().map_err(|_| invalid("Expected shape (3,3)"))
}
fn matrices(a: PyReadonlyArray3<'_, f64>) -> PyResult<Vec<M3>> {
    let a = a.as_array();
    if a.shape()[1..] != [3, 3] || a.shape()[0] > 100 {
        return Err(invalid("Expected shape (M,3,3)"));
    }
    let v: Vec<M3> = a
        .outer_iter()
        .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
        .collect();
    if v.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("Nonfinite matrix"));
    }
    Ok(v)
}
fn array2(py: Python<'_>, v: Vec<V3>) -> Bound<'_, PyArray2<f64>> {
    let len = v.len();
    let flat = v.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((len, 3), flat)
        .unwrap()
        .into_pyarray(py)
}
fn positive(x: f64, name: &str) -> PyResult<()> {
    if !x.is_finite() || x <= 0. {
        return Err(invalid(format!("{name} must be positive and finite")));
    }
    Ok(())
}

#[pyclass(name = "ExperimentalLatticeEngine", frozen)]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature=(workers=4))]
    fn new(workers: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Engine::new(workers).map_err(invalid)?,
        })
    }
    #[getter]
    fn workers(&self) -> usize {
        self.inner.workers
    }

    fn build_field(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray2<'_, f64>,
        n: usize,
        step: f64,
    ) -> PyResult<PyField> {
        let q = points(q)?;
        let inner = py
            .detach(|| self.inner.field(&q, n, step))
            .map_err(invalid)?;
        Ok(PyField { inner })
    }
    fn field_from_array(&self, values: PyReadonlyArray3<'_, f32>, step: f64) -> PyResult<PyField> {
        positive(step, "step")?;
        let a = values.as_array();
        let n = a.shape()[0];
        if !(16..=256).contains(&n) || n % 2 != 0 || a.shape() != [n, n, n / 2 + 1] {
            return Err(invalid(
                "Expected real FFT field shape (N,N,N/2+1), N even and >=16",
            ));
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Nonfinite Fourier field"));
        }
        Ok(PyField {
            inner: Field {
                n,
                step,
                values: a.iter().copied().collect(),
                pool: self.inner.pool.clone(),
            },
        })
    }
    fn score<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray2<'_, f64>,
        reciprocal: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<ScoredPoints<'py>> {
        let q = points(q)?;
        let rows = matrix(rows)?;
        let recip = matrix(reciprocal)?;
        let r = py.detach(|| self.inner.score(&q, rows, recip));
        let (h, d): (Vec<_>, Vec<_>) = r.into_iter().unzip();
        Ok((array2(py, h), d.into_pyarray(py)))
    }
    fn assign<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray3<'_, f64>,
        reciprocals: PyReadonlyArray3<'_, f64>,
        tolerance: f64,
        ambiguity_ratio: f64,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, usize, usize)> {
        positive(tolerance, "tolerance")?;
        if !ambiguity_ratio.is_finite() || !(0. ..=1.).contains(&ambiguity_ratio) {
            return Err(invalid("ambiguity_ratio must lie in [0,1]"));
        }
        let q = points(q)?;
        let rows = matrices(rows)?;
        let recip = matrices(reciprocals)?;
        if rows.len() != recip.len() {
            return Err(invalid("Mismatched model counts"));
        }
        let models: Vec<_> = rows.into_iter().zip(recip).collect();
        let (labels, a, m) =
            py.detach(|| self.inner.assign(&q, &models, tolerance, ambiguity_ratio));
        Ok((labels.into_pyarray(py), a, m))
    }
}

#[pyclass(name = "ExperimentalFourierField", frozen)]
struct PyField {
    inner: Field,
}
#[pymethods]
impl PyField {
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let n = self.inner.n;
        ndarray::Array3::from_shape_vec((n, n, n / 2 + 1), self.inner.values.clone())
            .unwrap()
            .into_pyarray(py)
    }
    fn maxima<'py>(
        &self,
        py: Python<'py>,
        minimum: f64,
        maximum: f64,
        limit: usize,
    ) -> PyResult<FourierMaxima<'py>> {
        positive(minimum, "minimum")?;
        positive(maximum, "maximum")?;
        if maximum < minimum {
            return Err(invalid("maximum below minimum"));
        }
        if limit == 0 || limit > 1000 {
            return Err(invalid("maxima budget must lie in [1,1000]"));
        }
        let (v, s, total) = py.detach(|| self.inner.maxima(minimum, maximum, limit));
        Ok((array2(py, v), s.into_pyarray(py), total))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyField>()?;
    Ok(())
}
