//! Detached shared-search diagnostics and equation-test boundary.
use super::{array2, cell, invalid};
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::{PreparedElementPool, SharedSearchSummary};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

#[allow(clippy::too_many_lines)]
pub(super) fn summary<'py>(
    py: Python<'py>,
    s: &SharedSearchSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("elements", &s.elements)?;
    d.set_item("phasing_starts", s.phasing_starts)?;
    d.set_item("dual_checkpoints", s.dual_checkpoints)?;
    d.set_item("cached_refinement", s.cached_refinement)?;
    d.set_item("symmetry_recycling", s.symmetry_recycling)?;
    d.set_item("recycling_cycles", s.recycling_cycles)?;
    let recycling = PyList::empty(py);
    for r in &s.recycling {
        let item = PyDict::new(py);
        item.set_item("trial", r.trial)?;
        item.set_item("lineage", r.lineage)?;
        item.set_item("hand", r.hand)?;
        item.set_item("origin_shift", r.origin_shift)?;
        item.set_item("cycles", r.cycles)?;
        item.set_item("predicted_coefficients", r.predicted_coefficients)?;
        item.set_item("peaks", r.peaks)?;
        item.set_item("batch_orbits", r.batch_orbits)?;
        item.set_item("label_scores", r.label_scores)?;
        item.set_item("label_changes", r.label_changes)?;
        item.set_item("batch_refined", r.batch_refined)?;
        item.set_item(
            "tangent",
            r.tangent
                .as_ref()
                .map(|a| super::tangent::audit(py, a))
                .transpose()?,
        )?;
        recycling.append(item)?;
    }
    d.set_item("recycling", recycling)?;
    d.set_item("tangent_refinement", s.tangent_refinement)?;
    d.set_item("tangent_in_recycling", s.tangent_in_recycling)?;
    d.set_item("adaptive_counts", s.adaptive_counts)?;
    d.set_item("wilson_normalization", s.wilson_normalization)?;
    d.set_item("phase_quality_selection", s.phase_quality_selection)?;
    d.set_item("phase_origin_selection", s.phase_origin_selection)?;
    d.set_item("adaptive_checkpoints", s.adaptive_checkpoints)?;
    d.set_item("score_only_evaluation", s.score_only_evaluation)?;
    d.set_item("reusable_fourier_workspaces", s.reusable_fourier_workspaces)?;
    d.set_item("work_sliced_phasing", s.work_sliced_phasing)?;
    d.set_item("filtered_peak_search", s.filtered_peak_search)?;
    d.set_item("batched_fft", s.batched_fft)?;
    d.set_item("batched_phase_transforms", s.batched_phase_transforms)?;
    d.set_item("filtered_peak_cycles", s.filtered_peak_cycles)?;
    d.set_item(
        "isotropic_assignment_refinement",
        s.isotropic_assignment_refinement,
    )?;
    let slices = PyList::empty(py);
    for a in &s.phase_slices {
        let row = PyDict::new(py);
        row.set_item("trial", a.trial)?;
        row.set_item("cycle_quota", a.cycle_quota)?;
        row.set_item("completed_cycles", a.completed_cycles)?;
        row.set_item("seconds", a.seconds)?;
        row.set_item("stop_reason", a.stop_reason)?;
        slices.append(row)?;
    }
    d.set_item("phase_slices", slices)?;
    d.set_item("score_only_evaluations", s.score_only_evaluations)?;
    d.set_item("fourier_workspace_cycles", s.fourier_workspace_cycles)?;
    d.set_item("independent_origin_sampling", s.independent_origin_sampling)?;
    let decisions = PyList::empty(py);
    for (trial, c) in &s.phase_checkpoints {
        let row = PyDict::new(py);
        row.set_item("trial", trial)?;
        row.set_item("cycles", c.cycles)?;
        row.set_item("cfom", c.cfom)?;
        row.set_item("best_cfom", c.best_cfom)?;
        row.set_item("score", c.score)?;
        row.set_item("map_cosine", c.map_cosine)?;
        row.set_item("selected", c.selected)?;
        row.set_item("reason", c.reason)?;
        decisions.append(row)?;
    }
    d.set_item("phase_checkpoints", decisions)?;
    d.set_item("chemical_assignment", s.chemical_assignment)?;
    d.set_item("iterative_refinement", s.iterative_refinement)?;
    d.set_item("cached_normal_refinement", s.cached_normal_refinement)?;
    d.set_item("cached_normal_solves", s.cached_normal_solves)?;
    d.set_item("normal_cache_fallbacks", s.normal_cache_fallbacks)?;
    d.set_item("missing_reflections", s.missing_reflections)?;
    d.set_item("resolution_extension", s.resolution_extension)?;
    d.set_item("prediction_steps", s.prediction_steps)?;
    d.set_item("prediction_updates", s.prediction_updates)?;
    d.set_item("prediction_clipped", s.prediction_clipped)?;
    d.set_item("prediction_seconds", s.prediction_seconds)?;
    d.set_item(
        "prediction",
        s.prediction
            .as_ref()
            .map(|a| super::prediction::audit(py, a))
            .transpose()?,
    )?;
    d.set_item(
        "phase_origins",
        super::origin::origin_records(py, &s.phase_origins)?,
    )?;
    d.set_item("phase_first_selection", s.phase_first_selection)?;
    d.set_item(
        "calculated_shell_normalization",
        s.calculated_shell_normalization,
    )?;
    d.set_item(
        "normalization",
        s.normalization
            .as_ref()
            .map(|n| super::quality::normalization_record(py, n))
            .transpose()?,
    )?;
    d.set_item("normalization_diagnostic", &s.normalization_diagnostic)?;
    d.set_item(
        "phase_quality",
        super::quality::trace_records(py, &s.phase_quality)?,
    )?;
    d.set_item("phase_model_trials", &s.phase_model_trials)?;
    d.set_item("count_prefix_screenings", s.count_prefix_screenings)?;
    d.set_item("bounded_map_handoffs", s.bounded_map_handoffs)?;
    let tangent_trials = PyList::empty(py);
    for trial in &s.tangent_trials {
        tangent_trials.append(super::tangent::audit(py, trial)?)?;
    }
    d.set_item("tangent_trials", tangent_trials)?;
    d.set_item("framework_hypotheses", s.framework_hypotheses)?;
    d.set_item("initial_prefix", s.initial_prefix)?;
    d.set_item("rounds", s.rounds)?;
    d.set_item("orbit_proposals", s.orbit_proposals)?;
    d.set_item("element_scores", s.element_scores)?;
    d.set_item("refinements", s.refinements)?;
    d.set_item("screening_refinements", s.screening_refinements)?;
    d.set_item("final_refinements", s.final_refinements)?;
    d.set_item("promoted_candidates", s.promoted_candidates)?;
    d.set_item("screened_out", s.screened_out)?;
    d.set_item("integrated_peak_density", s.integrated_peak_density)?;
    d.set_item(
        "peak_density_radius_angstrom",
        s.peak_density_radius_angstrom,
    )?;
    d.set_item("density_map_builds", s.density_map_builds)?;
    d.set_item("density_evaluations", s.density_evaluations)?;
    d.set_item("density_label_rejections", s.density_label_rejections)?;
    d.set_item("density_label_rescues", s.density_label_rescues)?;
    d.set_item("density_birth_skips", s.density_birth_skips)?;
    d.set_item("density_relabel_skips", s.density_relabel_skips)?;

    let calibrations = PyList::empty(py);
    for c in &s.density_calibrations {
        let item = PyDict::new(py);
        item.set_item("trial", c.trial)?;
        item.set_item("lineage", c.lineage)?;
        item.set_item("round", c.round)?;
        item.set_item("scale", c.scale)?;
        item.set_item("anchors", c.anchors)?;
        item.set_item("shortlist_limit", c.shortlist_limit)?;
        calibrations.append(item)?;
    }
    d.set_item("density_calibrations", calibrations)?;
    d.set_item("staged_refinement", s.staged_refinement)?;
    d.set_item("screening_iterations", s.screening_iterations)?;
    d.set_item("refinement_shortlist", s.refinement_shortlist)?;

    d.set_item("deletion_trials", s.deletion_trials)?;
    d.set_item("relabel_trials", s.relabel_trials)?;
    d.set_item("maximum_atoms", s.maximum_atoms)?;
    d.set_item("beam_width", s.beam_width)?;
    d.set_item("required_elements_present", s.required_elements_present)?;
    let moves = PyList::empty(py);
    for m in &s.accepted_moves {
        let x = PyDict::new(py);
        x.set_item("kind", m.kind)?;
        x.set_item("element", &m.element)?;
        x.set_item("atoms", m.atoms)?;
        x.set_item("r1", m.r1)?;
        x.set_item("score", m.score)?;
        moves.append(x)?;
    }
    d.set_item("accepted_moves", moves)?;
    Ok(d)
}
#[pyclass(name = "_SharedOrbitPool", frozen)]
struct NativePool {
    inner: PreparedElementPool,
}
#[pymethods]
impl NativePool {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        fixed_species: Vec<String>,
        fixed_xyz: PyReadonlyArray1<'_, f64>,
        elements: Vec<String>,
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        u_iso: f64,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let xyz = fixed_xyz.as_slice()?;
        if xyz.len() != fixed_species.len() * 3 {
            return Err(invalid("shared coordinate/species lengths differ"));
        }
        let points: Vec<_> = xyz.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let hkl = hkl_rows(&hkl)?;
        let observed = observed.as_slice()?.to_vec();
        let inner = py
            .detach(move || {
                PreparedElementPool::new(
                    cell,
                    &group,
                    &fixed_species,
                    &points,
                    &elements,
                    &hkl,
                    &observed,
                    u_iso,
                )
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn evaluate<'py>(&self, py: Python<'py>, point: [f64; 3]) -> PyResult<Bound<'py, PyList>> {
        let evaluations = py.detach(|| self.inner.evaluate(point)).map_err(invalid)?;
        let result = PyList::empty(py);
        for (element, e) in self.inner.elements.iter().zip(evaluations) {
            let d = PyDict::new(py);
            d.set_item("element", element)?;
            d.set_item("objective", e.objective)?;
            d.set_item("r1", e.r1)?;
            d.set_item("amplitude_scale", e.amplitude_scale)?;
            d.set_item("gradient", e.gradient)?;
            d.set_item("residual", e.residual.into_pyarray(py))?;
            d.set_item("jacobian", e.jacobian.into_pyarray(py))?;
            d.set_item("fractional_xyz", array2(py, e.fractional_xyz))?;
            result.append(d)?;
        }
        Ok(result)
    }
}
#[pyfunction(name = "_solution_screening_shortlist")]
fn shortlist(
    scores: PyReadonlyArray1<'_, f64>,
    birth_elements: Vec<Option<String>>,
    missing: Vec<String>,
    limit: usize,
) -> PyResult<Vec<usize>> {
    phasesmith_solution::screening_shortlist(scores.as_slice()?, &birth_elements, &missing, limit)
        .map_err(invalid)
}
#[pyclass(name = "_IntegratedPeakDensity", frozen)]
struct NativeDensity {
    inner: phasesmith_solution::IntegratedPeakDensity,
}
#[pymethods]
impl NativeDensity {
    #[new]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        shape: [usize; 3],
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        density: PyReadonlyArray1<'_, f64>,
        radius: f64,
    ) -> PyResult<Self> {
        let cell = cell(&cell_parameters)?;
        let indices = hkl_rows(&hkl)?;
        let amplitudes = observed.as_slice()?.to_vec();
        let rho = density.as_slice()?.to_vec();
        let inner = py
            .detach(move || {
                phasesmith_solution::IntegratedPeakDensity::new(
                    cell,
                    shape,
                    &indices,
                    &amplitudes,
                    &rho,
                    radius,
                )
            })
            .map_err(invalid)?;
        Ok(Self { inner })
    }
    fn evaluate(&self, point: [f64; 3]) -> PyResult<(f64, [f64; 3], f64)> {
        let value = self.inner.evaluate(point).map_err(invalid)?;
        Ok((value.value, value.gradient, value.radius_derivative))
    }
    #[getter]
    fn phased_reflections(&self) -> usize {
        self.inner.phased_reflections
    }
}
#[pyfunction(name = "_solution_ball_transform")]
fn ball(q: f64, radius: f64) -> PyResult<(f64, f64)> {
    phasesmith_solution::ball_transform(q, radius).map_err(invalid)
}
#[pyfunction(name = "_solution_density_labels")]
fn labels(mass: f64, scale: Option<f64>, numbers: Vec<u8>) -> PyResult<Vec<bool>> {
    phasesmith_solution::density_label_mask(mass, scale, &numbers).map_err(invalid)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePool>()?;
    module.add_class::<NativeDensity>()?;
    module.add_function(wrap_pyfunction!(ball, module)?)?;
    module.add_function(wrap_pyfunction!(labels, module)?)?;
    module.add_function(wrap_pyfunction!(shortlist, module)?)
}
