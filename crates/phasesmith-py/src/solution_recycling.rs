//! Owned-array boundary for independent complete recycling tests.
use super::{NativeCancellation, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyDict};
use std::time::{Duration, Instant};
#[pyfunction(name = "_solution_recycle_density", signature=(source, shape, group, hkl, amplitudes, support, hand, shift, cycles, max_seconds, cancellation=None, tangent=None, seed=0))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'_, f64>,
    shape: [usize; 3],
    group: PyRef<'_, NativePreparedReflectionGenerator>,
    hkl: PyReadonlyArray1<'_, i64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    support: PyReadonlyArray1<'_, u8>,
    hand: i32,
    shift: [f64; 3],
    cycles: usize,
    max_seconds: f64,
    cancellation: Option<PyRef<'_, NativeCancellation>>,
    tangent: Option<PyRef<'_, super::tangent::NativeTangent>>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if !max_seconds.is_finite() || !(0.0..=3600.).contains(&max_seconds) {
        return Err(invalid("invalid recycling time budget"));
    }
    let source = source.as_slice()?.to_vec();
    let hkl = hkl_rows(&hkl)?;
    let amplitudes = amplitudes.as_slice()?.to_vec();
    let support = support.as_slice()?.to_vec();
    let group = group.space_group().clone();
    let token = cancellation.map(|c| c.inner.clone());
    let tangent = tangent.map(|t| t.inner.clone());
    let result = py
        .detach(move || {
            phasesmith_solution::recycle_density_guided(
                shape,
                &hkl,
                &amplitudes,
                &support,
                &group,
                &source,
                hand,
                shift,
                cycles,
                Some(Instant::now() + Duration::from_secs_f64(max_seconds)),
                token.as_ref(),
                tangent.as_ref(),
                seed,
            )
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("density", result.density.into_pyarray(py))?;
    out.set_item("cycles", result.cycles)?;
    out.set_item("predicted_coefficients", result.predicted_coefficients)?;
    out.set_item("negative_fractions", result.negative_fractions)?;
    out.set_item("phase_fallbacks", result.phase_fallbacks)?;
    out.set_item(
        "tangent",
        result
            .tangent
            .as_ref()
            .map(|a| super::tangent::audit(py, a))
            .transpose()?,
    )?;
    Ok(out)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
