//! Thin detached adapters for native Pawley calculation, refinement and persistence.
use crate::rietveld::NativeRietveldCancellation;
use npy::{IntoPyArray, PyArray2};
use phasesmith_persistence::{PawleyProject, decode_pawley_project, encode_pawley_project};
use phasesmith_workflows::{
    ConstraintTransform, PawleyEvaluation, RefinementLimits, RefinementRuntime, evaluate_pawley,
    refine_pawley_with_runtime,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
const MAX_BYTES: usize = 256 * 1024 * 1024;
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
#[pyfunction]
fn _pawley_prepare(py: Python<'_>, record: String) -> PyResult<String> {
    py.detach(move || {
        let p = decode_pawley_project(&record, MAX_BYTES)?;
        encode_pawley_project(&p)
    })
    .map_err(error)
}
fn arrays<'py>(py: Python<'py>, e: &PawleyEvaluation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("calculated_y", e.calculated_y.clone().into_pyarray(py))?;
    d.set_item("background_y", e.background_y.clone().into_pyarray(py))?;
    d.set_item("intensities", e.intensities.clone().into_pyarray(py))?;
    d.set_item("positions", e.positions.clone().into_pyarray(py))?;
    let rows: Vec<Vec<f64>> = (0..e.jacobian.nrows())
        .map(|i| {
            (0..e.jacobian.ncols())
                .map(|j| e.jacobian[(i, j)])
                .collect()
        })
        .collect();
    d.set_item("jacobian", PyArray2::from_vec2(py, &rows)?)?;
    d.set_item("residual", e.residuals.residual.clone().into_pyarray(py))?;
    d.set_item(
        "weighted_residual",
        e.residuals.weighted_residual.clone().into_pyarray(py),
    )?;
    d.set_item("included", e.residuals.included.clone().into_pyarray(py))?;
    d.set_item("rp", e.residuals.rp)?;
    d.set_item("rwp", e.residuals.rwp)?;
    d.set_item("chi_square", e.residuals.chi_square)?;
    d.set_item("reduced_chi_square", e.residuals.reduced_chi_square)?;
    d.set_item("inactive_columns", e.inactive_columns.clone())?;
    d.set_item("unobserved_reflections", e.unobserved_reflections.clone())?;
    d.set_item(
        "coincident_groups",
        e.coincident_groups
            .iter()
            .map(|g| (g.members.clone(), g.total_intensity))
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}
#[pyfunction]
fn _pawley_calculate(py: Python<'_>, record: String) -> PyResult<Bound<'_, PyDict>> {
    let e = py
        .detach(move || {
            let mut p = decode_pawley_project(&record, MAX_BYTES)?;
            if p.input.pattern.observed_y.is_none() {
                p.input.pattern.observed_y = Some(vec![0.0; p.input.pattern.sample_count()]);
            }
            let t =
                ConstraintTransform::new(p.input.parameters.clone(), p.input.constraints.clone())
                    .map_err(|e| phasesmith_workflows::PawleyError(e.to_string()))?;
            let free = p
                .checkpoint
                .as_ref()
                .map_or_else(|| t.pack(), |cp| Ok(cp.free.clone()))
                .map_err(|e| phasesmith_workflows::PawleyError(e.to_string()))?;
            evaluate_pawley(
                &p.input,
                &free,
                p.options.support_fwhm,
                p.options.use_uncertainty,
                p.options.max_elements,
            )
        })
        .map_err(error)?;
    arrays(py, &e)
}
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _pawley_refine<'py>(
    py: Python<'py>,
    record: String,
    max_iterations: usize,
    max_evaluations: usize,
    max_rejections: usize,
    max_seconds: Option<f64>,
    cancellation: Option<&NativeRietveldCancellation>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let token = cancellation.map(|t| t.token.clone());
    let result = py
        .detach(move || -> Result<_, String> {
            let p = decode_pawley_project(&record, MAX_BYTES).map_err(|e| e.to_string())?;
            let limits =
                RefinementLimits::new(max_iterations, max_evaluations, max_seconds, max_rejections)
                    .map_err(|e| e.to_string())?;
            let mut runtime = RefinementRuntime::new(limits, token).map_err(|e| e.to_string())?;
            if let Some(callback) = progress {
                runtime.set_event_sink(move |event: &phasesmith_workflows::RefinementEvent| {
                    Python::attach(|py| -> PyResult<()> {
                        let d = PyDict::new(py);
                        d.set_item("kind", event.kind().as_str())?;
                        d.set_item("message", event.message())?;
                        d.set_item("accepted_iterations", event.accepted_iterations())?;
                        callback.call1(py, (d,))?;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())
                });
            }
            refine_pawley_with_runtime(&p.input, &p.options, p.checkpoint.as_ref(), &mut runtime)
                .map_err(|e| e.to_string())
        })
        .map_err(error)?;
    let d = arrays(py, &result.evaluation)?;
    d.set_item("termination_reason", result.termination_reason.as_str())?;
    d.set_item("rank", result.rank)?;
    let diagnostics = PyDict::new(py);
    diagnostics.set_item("evaluations", result.diagnostics.evaluations)?;
    diagnostics.set_item("linear_iterations", result.diagnostics.linear_iterations)?;
    diagnostics.set_item("evaluation_seconds", result.diagnostics.evaluation_seconds)?;
    diagnostics.set_item("qr_seconds", result.diagnostics.qr_seconds)?;
    diagnostics.set_item("face_seconds", result.diagnostics.face_seconds)?;
    diagnostics.set_item("multiplier_seconds", result.diagnostics.multiplier_seconds)?;
    diagnostics.set_item("diagnostic_seconds", result.diagnostics.diagnostic_seconds)?;
    diagnostics.set_item(
        "convergence_criterion",
        result.diagnostics.convergence_criterion,
    )?;
    diagnostics.set_item(
        "backtrack_rejections",
        result.diagnostics.backtrack_rejections,
    )?;
    diagnostics.set_item("last_step_norm", result.diagnostics.last_step_norm)?;
    diagnostics.set_item(
        "projected_gradient_norm",
        result.diagnostics.projected_gradient_norm,
    )?;
    diagnostics.set_item("infeasible_trials", result.diagnostics.infeasible_trials)?;
    diagnostics.set_item(
        "last_rejected_error",
        result.diagnostics.last_rejected_error,
    )?;
    d.set_item("diagnostics", diagnostics)?;
    d.set_item("observed_free_parameters", result.observed_free_parameters)?;
    d.set_item("active_bounds", result.active_bounds)?;
    d.set_item("active_width_bounds", result.active_width_bounds)?;
    d.set_item("covariance_limitation", result.covariance_limitation)?;
    if let Some(c) = result.covariance {
        let rows: Vec<Vec<f64>> = (0..c.nrows())
            .map(|i| (0..c.ncols()).map(|j| c[(i, j)]).collect())
            .collect();
        d.set_item("covariance", PyArray2::from_vec2(py, &rows)?)?;
    } else {
        d.set_item("covariance", py.None())?;
    }
    let checkpoint = result.checkpoint;
    d.set_item(
        "history",
        checkpoint.chi_square_history.clone().into_pyarray(py),
    )?;
    d.set_item("free", checkpoint.free.clone().into_pyarray(py))?;
    let project = PawleyProject {
        input: checkpoint.input.clone(),
        options: checkpoint.options.clone(),
        checkpoint: Some(checkpoint),
    };
    d.set_item(
        "checkpoint",
        encode_pawley_project(&project).map_err(error)?,
    )?;
    Ok(d)
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_pawley_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(_pawley_calculate, m)?)?;
    m.add_function(wrap_pyfunction!(_pawley_refine, m)?)?;
    Ok(())
}
