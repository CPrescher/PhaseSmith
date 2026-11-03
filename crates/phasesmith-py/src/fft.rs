//! Copies `NumPy` inputs before releasing the GIL. Rust never reads a Python
//! buffer concurrently with a possible mutation from another Python thread.
use npy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, ndarray,
};
use phasesmith_single_crystal::spots::fft::{
    Engine, Field, FittingCloud, FourierCloud, M3, ObservationOwnership, SupportBits, V3,
};
use pyo3::{exceptions::PyValueError, prelude::*};

type ScoredPoints<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
type FourierMaxima<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f32>>, usize);
type RefinedFourier<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u64>>,
);
type FittingSelection<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

type RankedTriplets<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u32>>);

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
    fn fourier_tiled<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        vectors: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = points(q)?;
        let vectors = points(vectors)?;
        let rows = py
            .detach(|| self.inner.fourier_tiled(&q, &vectors))
            .map_err(invalid)?;
        Ok(
            ndarray::Array2::from_shape_vec((rows.len(), 13), rows.into_iter().flatten().collect())
                .unwrap()
                .into_pyarray(py),
        )
    }
    fn prepare_fitting(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyFittingCloud> {
        let q = points(q)?;
        Ok(PyFittingCloud {
            inner: py
                .detach(|| self.inner.prepare_fitting(&q))
                .map_err(invalid)?,
        })
    }
    fn initial_support<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray3<'_, f64>,
        reciprocals: PyReadonlyArray3<'_, f64>,
        tolerance: f64,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let q = points(q)?;
        let rows = matrices(rows)?;
        let reciprocal = matrices(reciprocals)?;
        positive(tolerance, "tolerance")?;
        if q.is_empty()
            || rows.len() != reciprocal.len()
            || rows.len().saturating_mul(q.len()) > 8_000_000
        {
            return Err(invalid(
                "requires nonempty points, matching bases, <=8million pairs",
            ));
        }
        let models: Vec<_> = rows.into_iter().zip(reciprocal).collect();
        Ok(py
            .detach(|| self.inner.initial_support(&q, &models, tolerance))
            .into_pyarray(py))
    }
    fn refine_fourier<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        initial: PyReadonlyArray2<'_, f64>,
        radius: f64,
    ) -> PyResult<RefinedFourier<'py>> {
        let q = points(q)?;
        let initial = points(initial)?;
        let rows = py
            .detach(|| self.inner.refine_fourier(&q, &initial, radius))
            .map_err(invalid)?;
        let n = rows.len();
        let mut vectors = Vec::with_capacity(n * 3);
        let mut moments = Vec::with_capacity(n * 13);
        let mut stats = Vec::with_capacity(n * 2);
        for (v, m, s) in rows {
            vectors.extend(v);
            moments.extend(m);
            stats.extend(s.map(|x| x as u64));
        }
        Ok((
            ndarray::Array2::from_shape_vec((n, 3), vectors)
                .unwrap()
                .into_pyarray(py),
            ndarray::Array2::from_shape_vec((n, 13), moments)
                .unwrap()
                .into_pyarray(py),
            ndarray::Array2::from_shape_vec((n, 2), stats)
                .unwrap()
                .into_pyarray(py),
        ))
    }
    fn prepare_fourier(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyFourierCloud> {
        let q = points(q)?;
        Ok(PyFourierCloud {
            inner: py
                .detach(|| self.inner.prepare_fourier(&q))
                .map_err(invalid)?,
        })
    }
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn physical_fields<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray3<'_, f64>,
        reciprocals: PyReadonlyArray3<'_, f64>,
        volumes: PyReadonlyArray1<'_, f64>,
        charts: PyReadonlyArray3<'_, f64>,
        enabled: PyReadonlyArray1<'_, bool>,
        measure: PyReadonlyArray1<'_, f64>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray2<bool>>,
    )> {
        let q = points(q)?;
        let copy = |data: PyReadonlyArray3<'_, f64>| -> PyResult<Vec<M3>> {
            let a = data.as_array();
            if a.shape()[1..] != [3, 3] || a.shape()[0] > 10_000 || a.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("expected finite (M,3,3), M<=10000"));
            }
            Ok(a.outer_iter()
                .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
                .collect())
        };
        let rows = copy(rows)?;
        let reciprocal = copy(reciprocals)?;
        if rows.len() != reciprocal.len() || rows.len().saturating_mul(q.len()) > 8_000_000 {
            return Err(invalid(
                "model counts differ or physical field budget exceeds 8 million",
            ));
        }
        let volumes: Vec<f64> = volumes.as_array().iter().copied().collect();
        if volumes.len() != rows.len() || volumes.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(invalid("volumes must be positive finite shape(M,)"));
        }
        let charts = charts.as_array();
        if charts.shape() != [q.len(), 3, 3] || charts.iter().any(|v| !v.is_finite()) {
            return Err(invalid("charts must be finite shape(N,3,3)"));
        }
        let charts: Vec<M3> = charts
            .outer_iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
            .collect();
        let enabled: Vec<bool> = enabled.as_array().iter().copied().collect();
        let measure: Vec<f64> = measure.as_array().iter().copied().collect();
        let p: Vec<f64> = parameters.as_array().iter().copied().collect();
        if enabled.len() != q.len()
            || measure.len() != q.len()
            || measure.iter().any(|v| !v.is_finite() || *v < 0.)
            || enabled.iter().zip(&measure).any(|(&on, &v)| on && v <= 0.)
            || p.len() != 5
            || p.iter().any(|v| !v.is_finite() || *v <= 0.)
            || p[0] >= 1.
        {
            return Err(invalid(
                "need enabled[N], positive active measure[N], and five positive finite parameters, tolerance<1",
            ));
        }
        let m = rows.len();
        let n = q.len();
        let models: Vec<_> = rows
            .into_iter()
            .zip(reciprocal)
            .zip(volumes)
            .map(|((a, b), v)| (a, b, v))
            .collect();
        let fields = py
            .detach(|| {
                self.inner.physical_fields(
                    &q,
                    &models,
                    &charts,
                    &enabled,
                    &measure,
                    [p[0], p[1], p[2], p[3], p[4]],
                )
            })
            .map_err(invalid)?;
        let mut evidence = Vec::with_capacity(m * n);
        let mut core = Vec::with_capacity(m * n);
        let mut physical = Vec::with_capacity(m * n);
        for row in fields {
            evidence.extend(row.evidence);
            core.extend(row.core);
            physical.extend(row.physical);
        }
        Ok((
            ndarray::Array2::from_shape_vec((m, n), evidence)
                .unwrap()
                .into_pyarray(py),
            ndarray::Array2::from_shape_vec((m, n), core)
                .unwrap()
                .into_pyarray(py),
            ndarray::Array2::from_shape_vec((m, n), physical)
                .unwrap()
                .into_pyarray(py),
        ))
    }

    fn pack_support(&self, support: PyReadonlyArray2<'_, bool>) -> PyResult<PySupportBits> {
        let a = support.as_array();
        let (vectors, count) = (a.nrows(), a.ncols());
        if vectors > 400
            || count == 0
            || count > 100_000
            || vectors.saturating_mul(count) > 20_000_000
        {
            return Err(invalid(
                "support requires(K,N), K<=400, 1<=N<=100000, <=20million bits",
            ));
        }
        let rows = a
            .outer_iter()
            .map(|row| {
                let mut words = vec![0u64; count.div_ceil(64)];
                for (j, &on) in row.iter().enumerate() {
                    if on {
                        words[j / 64] |= 1u64 << (j % 64);
                    }
                }
                words
            })
            .collect();
        Ok(PySupportBits {
            inner: SupportBits {
                rows,
                count,
                pool: self.inner.pool.clone(),
            },
        })
    }

    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn candidate_counts<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray3<'_, f64>,
        reciprocals: PyReadonlyArray3<'_, f64>,
        charts: PyReadonlyArray3<'_, f64>,
        enabled: PyReadonlyArray1<'_, bool>,
        tolerances: PyReadonlyArray1<'_, f64>,
        xy_gate: f64,
        omega_gate: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = points(q)?;
        let copy_models = |a: PyReadonlyArray3<'_, f64>| -> PyResult<Vec<M3>> {
            let a = a.as_array();
            if a.shape()[1..] != [3, 3] || a.shape()[0] > 50_000 || a.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("expected finite (M,3,3), M<=50000"));
            }
            Ok(a.outer_iter()
                .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
                .collect())
        };
        let rows = copy_models(rows)?;
        let reciprocal = copy_models(reciprocals)?;
        if rows.len() != reciprocal.len() || q.len().saturating_mul(rows.len()) > 500_000_000 {
            return Err(invalid("model counts differ or work exceeds500million"));
        }
        let charts = charts.as_array();
        if charts.shape() != [q.len(), 3, 3] || charts.iter().any(|v| !v.is_finite()) {
            return Err(invalid("charts must be finite(N,3,3)"));
        }
        let charts: Vec<M3> = charts
            .outer_iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
            .collect();
        let enabled: Vec<bool> = enabled.as_array().iter().copied().collect();
        let t: Vec<f64> = tolerances.as_array().iter().copied().collect();
        if enabled.len() != q.len()
            || t.len() != 3
            || t.iter().any(|v| !v.is_finite() || *v <= 0. || *v >= 1.)
            || !(t[0] < t[1] && t[1] < t[2])
        {
            return Err(invalid(
                "need enabled[N] and three strictly increasing cutoffs in(0,1)",
            ));
        }
        positive(xy_gate, "xy_gate")?;
        positive(omega_gate, "omega_gate")?;
        let models: Vec<_> = rows.into_iter().zip(reciprocal).collect();
        let result = py.detach(|| {
            self.inner.candidate_counts(
                &q,
                &models,
                &charts,
                &enabled,
                [t[0], t[1], t[2]],
                xy_gate,
                omega_gate,
            )
        });
        Ok(ndarray::Array2::from_shape_vec(
            (result.len(), 9),
            result.into_iter().flatten().collect(),
        )
        .unwrap()
        .into_pyarray(py))
    }

    fn fourier<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        vectors: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = points(q)?;
        let vectors = points(vectors)?;
        let result = py
            .detach(|| self.inner.fourier(&q, &vectors))
            .map_err(invalid)?;
        Ok(ndarray::Array2::from_shape_vec(
            (result.len(), 13),
            result.into_iter().flatten().collect(),
        )
        .unwrap()
        .into_pyarray(py))
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::many_single_char_names
    )]
    #[pyo3(signature=(q,hkl,charts,enabled,reciprocal,scales,huber,background=None))]
    fn observation_normal<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        hkl: PyReadonlyArray2<'_, f64>,
        charts: PyReadonlyArray3<'_, f64>,
        enabled: PyReadonlyArray1<'_, bool>,
        reciprocal: PyReadonlyArray2<'_, f64>,
        scales: PyReadonlyArray1<'_, f64>,
        huber: f64,
        background: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        f64,
        usize,
    )> {
        let q = points(q)?;
        let h = points(hkl)?;
        let b = matrix(reciprocal)?;
        if h.len() != q.len() || h.iter().flatten().any(|v| v.fract() != 0. || v.abs() > 1e6) {
            return Err(invalid("hkl must be bounded integer-valued shape (N,3)"));
        }
        let c = charts.as_array();
        if c.shape() != [q.len(), 3, 3] || c.iter().any(|v| !v.is_finite()) {
            return Err(invalid("charts must be finite (N,3,3)"));
        }
        let c: Vec<M3> = c
            .outer_iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
            .collect();
        let valid: Vec<bool> = enabled.as_array().iter().copied().collect();
        if valid.len() != q.len() {
            return Err(invalid("enabled must be (N,)"));
        }
        let s: Vec<f64> = scales.as_array().iter().copied().collect();
        if s.len() != 3 || s.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(invalid("scales must be three positive finite values"));
        }
        positive(huber, "huber")?;
        let background = background.map(|a| a.as_array().iter().copied().collect::<Vec<_>>());
        if background
            .as_ref()
            .is_some_and(|a| a.len() != q.len() || a.iter().any(|v| !v.is_finite() || *v <= 0.))
        {
            return Err(invalid("background must be positive finite (N,)"));
        }
        let (normal, gradient, cost, n) = py.detach(|| {
            self.inner.observation_normal(
                &q,
                &h,
                &c,
                &valid,
                b,
                [s[0], s[1], s[2]],
                huber,
                background.as_deref(),
            )
        });
        if !cost.is_finite() || normal.iter().chain(&gradient).any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite observation normal accumulation"));
        }
        Ok((
            ndarray::Array2::from_shape_vec((9, 9), normal.to_vec())
                .unwrap()
                .into_pyarray(py),
            gradient.to_vec().into_pyarray(py),
            cost,
            n,
        ))
    }

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
    #[allow(clippy::too_many_arguments)]
    fn windowed_maxima<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        n: usize,
        step: f64,
        minimum: f64,
        maximum: f64,
        limit: usize,
    ) -> PyResult<FourierMaxima<'py>> {
        let q = points(q)?;
        let (v, s, total) = py
            .detach(|| {
                self.inner
                    .windowed_maxima(&q, n, step, minimum, maximum, limit)
            })
            .map_err(invalid)?;
        Ok((array2(py, v), s.into_pyarray(py), total))
    }
    fn field_from_array(&self, values: PyReadonlyArray3<'_, f32>, step: f64) -> PyResult<PyField> {
        positive(step, "step")?;
        let a = values.as_array();
        let n = a.shape()[0];
        if !(16..=256).contains(&n) || !n.is_multiple_of(2) || a.shape() != [n, n, n / 2 + 1] {
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

    #[allow(clippy::too_many_arguments)] // Owned array boundary; numerical options are packed in Rust.
    fn assign_observed<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'_, f64>,
        rows: PyReadonlyArray3<'_, f64>,
        reciprocals: PyReadonlyArray3<'_, f64>,
        charts: PyReadonlyArray3<'_, f64>,
        enabled: PyReadonlyArray1<'_, bool>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, usize, usize)> {
        let q = points(q)?;
        let rows = matrices(rows)?;
        let reciprocal = matrices(reciprocals)?;
        if rows.len() != reciprocal.len() {
            return Err(invalid("Mismatched model counts"));
        }
        let charts = charts.as_array();
        if charts.shape() != [q.len(), 3, 3] || charts.iter().any(|x| !x.is_finite()) {
            return Err(invalid("charts must be finite shape (N,3,3)"));
        }
        let charts: Vec<M3> = charts
            .outer_iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
            .collect();
        let enabled: Vec<bool> = enabled.as_array().iter().copied().collect();
        if enabled.len() != q.len() {
            return Err(invalid("enabled must have shape (N,)"));
        }
        let p: Vec<f64> = parameters.as_array().iter().copied().collect();
        if p.len() != 6
            || p.iter().any(|x| !x.is_finite())
            || p[0] <= 0.
            || !(0. ..=1.).contains(&p[1])
            || p[2..].iter().any(|x| *x <= 0.)
        {
            return Err(invalid(
                "parameters are tolerance,ratio,xy_scale,omega_scale,xy_gate,omega_gate",
            ));
        }
        let options = ObservationOwnership {
            tolerance: p[0],
            ambiguity_ratio: p[1],
            xy_scale: p[2],
            omega_scale: p[3],
            xy_gate: p[4],
            omega_gate: p[5],
        };
        let models: Vec<_> = rows.into_iter().zip(reciprocal).collect();
        let (labels, a, m) = py.detach(|| {
            self.inner
                .assign_observed(&q, &models, &charts, &enabled, options)
        });
        Ok((labels.into_pyarray(py), a, m))
    }
}

#[pyclass(name = "ExperimentalSupportBits", frozen)]
struct PySupportBits {
    inner: SupportBits,
}
#[pymethods]
impl PySupportBits {
    fn candidate_keys<'py>(
        &self,
        py: Python<'py>,
        max_pairs: usize,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        if !(1..=256).contains(&max_pairs) {
            return Err(invalid("max_pairs must lie in [1,256]"));
        }
        let keys = py.detach(|| self.inner.candidate_keys(max_pairs));
        Ok(ndarray::Array2::from_shape_vec(
            (keys.len(), 3),
            keys.into_iter()
                .flatten()
                .map(|i| i64::try_from(i).expect("bounded vector index"))
                .collect(),
        )
        .unwrap()
        .into_pyarray(py))
    }
    fn rank_triplets<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray2<'_, i64>,
        volumes: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<RankedTriplets<'py>> {
        let a = indices.as_array();
        let volumes: Vec<f64> = volumes.as_array().iter().copied().collect();
        if a.ncols() != 3
            || a.nrows() != volumes.len()
            || a.nrows().saturating_mul(self.inner.count.div_ceil(64)) > 200_000_000
            || a.iter()
                .any(|&i| usize::try_from(i).map_or(true, |v| v >= self.inner.rows.len()))
            || volumes.iter().any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(invalid(
                "requires valid indices(M,3), positive finite volumes(M), bounded work",
            ));
        }
        let ids: Vec<[usize; 3]> = a
            .outer_iter()
            .map(|r| std::array::from_fn(|j| usize::try_from(r[j]).expect("validated index")))
            .collect();
        let (order, counts) = py.detach(|| self.inner.rank_triplets(&ids, &volumes));
        Ok((
            order
                .into_iter()
                .map(|i| i64::try_from(i).expect("bounded triplet index"))
                .collect::<Vec<_>>()
                .into_pyarray(py),
            counts.into_pyarray(py),
        ))
    }
    fn pair_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let result = py.detach(|| self.inner.pairs());
        let n = self.inner.rows.len();
        ndarray::Array2::from_shape_vec((n, n), result.into_iter().flatten().collect())
            .unwrap()
            .into_pyarray(py)
    }
    fn triplet_counts<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let a = indices.as_array();
        if a.ncols() != 3
            || a.nrows().saturating_mul(self.inner.count.div_ceil(64)) > 200_000_000
            || a.iter()
                .any(|&v| v < 0 || usize::try_from(v).map_or(true, |x| x >= self.inner.rows.len()))
        {
            return Err(invalid("indices must be valid(M,3), bounded work"));
        }
        let ids: Vec<[usize; 3]> = a
            .outer_iter()
            .map(|r| std::array::from_fn(|i| usize::try_from(r[i]).expect("validated index")))
            .collect();
        let result = py.detach(|| self.inner.triples(&ids));
        Ok(result.into_pyarray(py))
    }
}

#[pyclass(name = "ExperimentalFittingCloud", frozen)]
struct PyFittingCloud {
    inner: FittingCloud,
}
#[pymethods]
impl PyFittingCloud {
    fn selected<'py>(
        &self,
        py: Python<'py>,
        rows: PyReadonlyArray2<'_, f64>,
        reciprocal: PyReadonlyArray2<'_, f64>,
        tolerance: f64,
    ) -> PyResult<FittingSelection<'py>> {
        let rows = matrix(rows)?;
        let reciprocal = matrix(reciprocal)?;
        let result = py
            .detach(|| self.inner.selected(rows, reciprocal, tolerance))
            .map_err(invalid)?;
        let mut hkl = Vec::with_capacity(result.len());
        let mut selected = Vec::with_capacity(result.len());
        let mut ids = Vec::with_capacity(result.len());
        let mut errors = Vec::with_capacity(result.len());
        for row in result {
            hkl.push(row.hkl);
            selected.push(row.point);
            ids.push(i64::try_from(row.index).expect("point index bounded by 100000"));
            errors.push(row.error);
        }
        Ok((
            array2(py, hkl),
            array2(py, selected),
            ids.into_pyarray(py),
            errors.into_pyarray(py),
        ))
    }
}

#[pyclass(name = "ExperimentalFourierCloud", frozen)]
struct PyFourierCloud {
    inner: FourierCloud,
}
#[pymethods]
impl PyFourierCloud {
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vectors = points(vectors)?;
        let rows = py
            .detach(|| self.inner.evaluate(&vectors))
            .map_err(invalid)?;
        Ok(
            ndarray::Array2::from_shape_vec((rows.len(), 13), rows.into_iter().flatten().collect())
                .unwrap()
                .into_pyarray(py),
        )
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
    m.add_class::<PySupportBits>()?;
    m.add_class::<PyFourierCloud>()?;
    m.add_class::<PyFittingCloud>()?;
    Ok(())
}
