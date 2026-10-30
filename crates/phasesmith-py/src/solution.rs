//! Array copies isolate detached Rust calculations from Python buffer mutations.

use npy::{IntoPyArray, PyReadonlyArray1, ndarray::Array2};
use phasesmith_solution::{FlipOptions, Observation, SolveOptions};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{NativePreparedReflectionGenerator, crystallographic_cell, hkl_rows};

#[path = "solution_completion.rs"]
mod completion;
#[path = "solution_dual.rs"]
mod dual;
#[path = "solution_origin.rs"]
mod origin;
#[path = "solution_prediction.rs"]
mod prediction;
#[path = "solution_quality.rs"]
mod quality;
#[path = "solution_recycling.rs"]
mod recycling;
#[path = "solution_shared.rs"]
mod shared;
#[path = "solution_symmetry.rs"]
mod symmetry_adp;
#[path = "solution_tangent.rs"]
mod tangent;

type SharedSearchRequest = (
    Vec<String>,
    usize,
    usize,
    usize,
    bool,
    usize,
    usize,
    bool,
    f64,
    f64,
    usize,
    (
        bool,
        bool,
        bool,
        usize,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        (
            bool,
            bool,
            bool,
            bool,
            f64,
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            (
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
                (bool, bool),
            ),
        ),
    ),
);

fn invalid(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn cell(values: &[f64]) -> PyResult<phasesmith_engine::crystallography::UnitCell> {
    if values.len() != 6 {
        return Err(invalid("cell must contain six parameters"));
    }
    Ok(crystallographic_cell(
        values[0], values[1], values[2], values[3], values[4], values[5],
    ))
}

fn array2<T: npy::Element>(py: Python<'_>, rows: Vec<[T; 3]>) -> Bound<'_, npy::PyArray2<T>> {
    Array2::from_shape_vec((rows.len(), 3), rows.into_iter().flatten().collect())
        .expect("rectangular rows")
        .into_pyarray(py)
}

#[pyfunction(name = "_solve_structure")]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn solve_structure<'py>(
    py: Python<'py>,
    hkl: PyReadonlyArray1<'py, i64>,
    intensity: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    cell_parameters: Vec<f64>,
    group: PyRef<'py, NativePreparedReflectionGenerator>,
    composition: Vec<(String, usize)>,
    seed: u64,
    trials: usize,
    cycles: usize,
    delta_sigma: f64,
    weak_fraction: f64,
    cleanup_cycles: usize,
    polish_iterations: usize,
    assignment_rounds: usize,
    max_seconds: f64,
    r1_target: f64,
    grid_shape: Option<[usize; 3]>,
    cancellation: Option<PyRef<'py, NativeCancellation>>,
    refinement: String,
    symmetry_tolerance_angstrom: f64,
    symmetry_iterations: usize,
    phasing: String,
    completion: bool,
    shared_search: Option<SharedSearchRequest>,
) -> PyResult<Bound<'py, PyDict>> {
    let hkl = hkl_rows(&hkl)?;
    let intensity = intensity.as_slice()?.to_vec();
    let sigma = sigma.as_slice()?.to_vec();
    if hkl.len() != intensity.len() || hkl.len() != sigma.len() {
        return Err(invalid("HKL, intensity and sigma lengths differ"));
    }
    let observations: Vec<_> = hkl
        .into_iter()
        .zip(intensity)
        .zip(sigma)
        .map(|((hkl, intensity), sigma)| Observation {
            hkl,
            intensity,
            sigma,
        })
        .collect();
    let cell = cell(&cell_parameters)?;
    let group = group.space_group().clone();
    let refinement = match refinement.as_str() {
        "none" => None,
        "isotropic" => Some(phasesmith_solution::AdpMode::Isotropic),
        "anisotropic" => Some(phasesmith_solution::AdpMode::Anisotropic),
        _ => return Err(invalid("unknown symmetry refinement mode")),
    };
    let phasing = match phasing.as_str() {
        "charge_flip" => phasesmith_solution::PhasingMethod::ChargeFlip,
        "dual_space" => phasesmith_solution::PhasingMethod::DualSpace,
        "portfolio" => phasesmith_solution::PhasingMethod::Portfolio,
        _ => return Err(invalid("unknown phasing method")),
    };
    let options = SolveOptions {
        seed,
        trials,
        cycles,
        delta_sigma,
        weak_fraction,
        cleanup_cycles,
        polish_iterations,
        assignment_rounds,
        max_seconds,
        r1_target,
        grid_shape,
        cancellation: cancellation.map(|token| token.inner.clone()),
        refinement,
        symmetry_tolerance_angstrom,
        symmetry_iterations,
        phasing,
        completion,
    };
    let result = py
        .detach(move || {
            if let Some((
                elements,
                maximum_atoms,
                beam_width,
                rounds,
                staged_refinement,
                screening_iterations,
                refinement_shortlist,
                integrated_peak_density,
                peak_density_radius_angstrom,
                dual_peak_sigma,
                dual_peak_cap,
                (
                    dual_checkpoints,
                    cached_refinement,
                    symmetry_recycling,
                    recycling_cycles,
                    tangent_refinement,
                    tangent_in_recycling,
                    adaptive_counts,
                    wilson_normalization,
                    phase_quality_selection,
                    phase_first_selection,
                    calculated_shell_normalization,
                    (
                        phase_origin_selection,
                        chemical_assignment,
                        iterative_refinement,
                        missing_reflections,
                        resolution_extension,
                        adaptive_checkpoints,
                        independent_origin_sampling,
                        score_only_evaluation,
                        reusable_fourier_workspaces,
                        work_sliced_phasing,
                        isotropic_assignment_refinement,
                        (
                            filtered_peak_search,
                            batched_fft,
                            cached_normal_refinement,
                            real_fft,
                            fused_phase_workspace,
                            multiplicity_weighted_refinement,
                            ambiguous_assignments,
                            cached_assignment_refinement,
                            specialized_isotropic_refinement,
                            streamed_peak_search,
                            reuse_normal_matrices,
                            (short_phase_starts, blocked_normal_matrices),
                        ),
                    ),
                ),
            )) = shared_search
            {
                if !composition.is_empty() {
                    return Err(phasesmith_solution::SolutionError(
                        "shared search does not accept composition".into(),
                    ));
                }
                phasesmith_solution::solve_elements(
                    &observations,
                    cell,
                    &group,
                    &elements,
                    &options,
                    &phasesmith_solution::SharedSearchOptions {
                        maximum_atoms,
                        beam_width,
                        rounds,
                        staged_refinement,
                        screening_iterations,
                        refinement_shortlist,
                        integrated_peak_density,
                        peak_density_radius_angstrom,
                        dual_peak_sigma,
                        dual_peak_cap,
                        dual_checkpoints,
                        cached_refinement,
                        symmetry_recycling,
                        recycling_cycles,
                        tangent_refinement,
                        tangent_in_recycling,
                        adaptive_counts,
                        wilson_normalization,
                        phase_quality_selection,
                        phase_first_selection,
                        calculated_shell_normalization,
                        phase_origin_selection,
                        chemical_assignment,
                        iterative_refinement,
                        missing_reflections,
                        resolution_extension,
                        adaptive_checkpoints,
                        independent_origin_sampling,
                        score_only_evaluation,
                        reusable_fourier_workspaces,
                        work_sliced_phasing,
                        isotropic_assignment_refinement,
                        filtered_peak_search,
                        batched_fft,
                        real_fft,
                        fused_phase_workspace,
                        cached_normal_refinement,
                        multiplicity_weighted_refinement,
                        ambiguous_assignments,
                        cached_assignment_refinement,
                        specialized_isotropic_refinement,
                        streamed_peak_search,
                        reuse_normal_matrices,
                        short_phase_starts,
                        blocked_normal_matrices,
                    },
                )
            } else {
                phasesmith_solution::solve(&observations, cell, &group, &composition, &options)
            }
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    if let Some(audit) = &result.shared_search {
        out.set_item("shared_search", shared::summary(py, audit)?)?;
    } else {
        out.set_item("shared_search", py.None())?;
    }
    out.set_item("hkl", array2(py, result.data.hkl))?;
    out.set_item("intensity", result.data.intensity.into_pyarray(py))?;
    out.set_item("sigma", result.data.sigma.into_pyarray(py))?;
    out.set_item("amplitudes", result.data.amplitudes.into_pyarray(py))?;
    out.set_item("family_count", result.data.family_count)?;
    out.set_item("absent_observations", result.data.absent_observations)?;
    out.set_item("nonpositive_families", result.data.nonpositive_families)?;
    out.set_item("d_min", result.data.d_min)?;
    out.set_item("grid_shape", result.grid_shape)?;
    out.set_item("density", result.density.into_pyarray(py))?;
    out.set_item("termination", result.termination)?;
    out.set_item("target_met", result.target_met)?;
    out.set_item("histories", result.histories)?;
    out.set_item(
        "phasing_trials",
        dual::trial_records(py, result.phasing_trials)?,
    )?;
    out.set_item("rejected_trials", result.rejected_trials)?;
    let candidates = PyList::empty(py);
    for c in result.candidates {
        let item = PyDict::new(py);
        item.set_item("species", c.species)?;
        item.set_item("fractional_xyz", array2(py, c.fractional_xyz))?;
        item.set_item("u_iso", c.u_iso)?;
        item.set_item("amplitude_scale", c.amplitude_scale)?;
        item.set_item("r1", c.r1)?;
        item.set_item("objective", c.objective)?;
        item.set_item("trial", c.trial)?;
        item.set_item("polish_steps", c.polish_steps)?;
        item.set_item("element_swaps", c.element_swaps)?;
        item.set_item("refinement_diagnostic", c.refinement_diagnostic)?;
        if let Some(audit) = &c.completion {
            item.set_item("completion", completion::summary(py, audit)?)?;
        } else {
            item.set_item("completion", py.None())?;
        }
        if let Some(refined) = &c.refinement {
            item.set_item("refinement", symmetry_adp::summary(py, refined)?)?;
        } else {
            item.set_item("refinement", py.None())?;
        }
        candidates.append(item)?;
    }
    out.set_item("candidates", candidates)?;
    Ok(out)
}

type FlipArrays<'py> = (
    Bound<'py, npy::PyArray1<f64>>,
    Bound<'py, npy::PyArray1<f64>>,
    usize,
);

#[pyfunction(name = "_solution_charge_flip")]
#[allow(clippy::too_many_arguments)]
fn flip<'py>(
    py: Python<'py>,
    shape: [usize; 3],
    hkl: PyReadonlyArray1<'py, i64>,
    amplitudes: PyReadonlyArray1<'py, f64>,
    phases: PyReadonlyArray1<'py, f64>,
    support: PyReadonlyArray1<'py, u8>,
    cycles: usize,
    delta_sigma: f64,
    weak_fraction: f64,
    cleanup_cycles: usize,
) -> PyResult<FlipArrays<'py>> {
    let hkl = hkl_rows(&hkl)?;
    let amplitudes = amplitudes.as_slice()?.to_vec();
    let phases = phases.as_slice()?.to_vec();
    let support = support.as_slice()?.to_vec();
    let result = py
        .detach(move || {
            phasesmith_solution::charge_flip(
                shape,
                &hkl,
                &amplitudes,
                &phases,
                &support,
                &FlipOptions {
                    cycles,
                    delta_sigma,
                    weak_fraction,
                    cleanup_cycles,
                    cancellation: None,
                },
                None,
            )
        })
        .map_err(invalid)?;
    Ok((
        result.density.into_pyarray(py),
        result.history.into_pyarray(py),
        result.best_cycle,
    ))
}

#[pyfunction(name = "_solution_model")]
fn model<'py>(
    py: Python<'py>,
    cell_parameters: Vec<f64>,
    hkl: PyReadonlyArray1<'py, i64>,
    observed: PyReadonlyArray1<'py, f64>,
    species: Vec<String>,
    xyz: PyReadonlyArray1<'py, f64>,
    u: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let cell = cell(&cell_parameters)?;
    let hkl = hkl_rows(&hkl)?;
    let observed = observed.as_slice()?.to_vec();
    let flat = xyz.as_slice()?;
    if flat.len() != 3 * species.len() {
        return Err(invalid("coordinate shape differs from species"));
    }
    let xyz: Vec<_> = flat.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect();
    let result = py
        .detach(move || {
            phasesmith_solution::evaluate_model(cell, &hkl, &observed, &species, &xyz, u)
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("objective", result.objective)?;
    out.set_item("r1", result.r1)?;
    out.set_item("amplitude_scale", result.amplitude_scale)?;
    out.set_item("residual", result.residual.into_pyarray(py))?;
    out.set_item("jacobian", result.jacobian.into_pyarray(py))?;
    Ok(out)
}

#[pyclass(name = "_SolutionCancellation", frozen)]
struct NativeCancellation {
    inner: phasesmith_solution::Cancellation,
}
#[pymethods]
impl NativeCancellation {
    #[new]
    fn new() -> Self {
        Self {
            inner: phasesmith_solution::Cancellation::default(),
        }
    }
    fn cancel(&self) -> bool {
        self.inner.cancel()
    }
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    symmetry_adp::register(module)?;
    dual::register(module)?;
    completion::register(module)?;
    shared::register(module)?;
    quality::register(module)?;
    origin::register(module)?;
    prediction::register(module)?;
    recycling::register(module)?;
    tangent::register(module)?;
    module.add_class::<NativeCancellation>()?;
    module.add_function(wrap_pyfunction!(solve_structure, module)?)?;
    module.add_function(wrap_pyfunction!(flip, module)?)?;
    module.add_function(wrap_pyfunction!(model, module)?)?;
    Ok(())
}
