//! Owned array boundary for independent dual-space equation tests.
use super::{cell, invalid};
use crate::hkl_rows;
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::DualSpaceOptions;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

#[pyfunction(name = "_solution_dual_space")]
#[pyo3(signature=(shape,cell_parameters,hkl,amplitudes,phases,support,cycles,atom_count,mask_sigma,hybrid_m,omit_fraction,seed,patterson_rank,reusable_workspaces=false))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    shape: [usize; 3],
    cell_parameters: Vec<f64>,
    hkl: PyReadonlyArray1<'_, i64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    phases: PyReadonlyArray1<'_, f64>,
    support: PyReadonlyArray1<'_, u8>,
    cycles: usize,
    atom_count: usize,
    mask_sigma: f64,
    hybrid_m: f64,
    omit_fraction: f64,
    seed: u64,
    patterson_rank: Option<usize>,
    reusable_workspaces: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cell = cell(&cell_parameters)?;
    let hkl = hkl_rows(&hkl)?;
    let amplitudes = amplitudes.as_slice()?.to_vec();
    let phases = phases.as_slice()?.to_vec();
    let support = support.as_slice()?.to_vec();
    let options = DualSpaceOptions {
        cycles,
        peak_selection: phasesmith_solution::PeakSelection::Fixed(atom_count),
        mask_sigma,
        hybrid_m,
        omit_fraction,
        seed,
        patterson_rank,
        cancellation: None,
    };
    let state = py
        .detach(move || {
            phasesmith_solution::dual_space_with_execution(
                shape,
                cell,
                &hkl,
                &amplitudes,
                &phases,
                &support,
                &options,
                None,
                None,
                phasesmith_solution::PhasingExecution {
                    score_only_evaluation: false,
                    reusable_fourier_workspaces: reusable_workspaces,
                    filtered_peak_search: false,
                },
            )
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("density", state.trial.density.into_pyarray(py))?;
    out.set_item("history", state.trial.history.into_pyarray(py))?;
    out.set_item("best_cycle", state.trial.best_cycle)?;
    out.set_item("initialization", state.initialization)?;
    out.set_item("patterson_vector_grid", state.patterson_vector)?;
    Ok(out)
}

#[pyfunction(name = "_solution_atomic_projection")]
#[allow(clippy::too_many_arguments)]
fn projection<'py>(
    py: Python<'py>,
    density: PyReadonlyArray1<'_, f64>,
    shape: [usize; 3],
    cell_parameters: Vec<f64>,
    atom_count: usize,
    mask_sigma: f64,
    omit_fraction: f64,
    seed: u64,
    cycle: usize,
) -> PyResult<Bound<'py, npy::PyArray1<f64>>> {
    let density = density.as_slice()?.to_vec();
    let cell = cell(&cell_parameters)?;
    let options = DualSpaceOptions {
        cycles: 1,
        peak_selection: phasesmith_solution::PeakSelection::Fixed(atom_count),
        mask_sigma,
        hybrid_m: 3.,
        omit_fraction,
        seed,
        patterson_rank: None,
        cancellation: None,
    };
    let out = py
        .detach(move || {
            phasesmith_solution::atomic_projection(&density, shape, cell, &options, cycle)
        })
        .map_err(invalid)?;
    Ok(out.into_pyarray(py))
}

pub(super) fn adaptive_record<'py>(
    py: Python<'py>,
    h: &phasesmith_solution::AdaptivePeakHistory,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("sigma", h.sigma)?;
    d.set_item("cap", h.cap)?;
    d.set_item("selected", &h.selected)?;
    d.set_item("thresholds", &h.thresholds)?;
    d.set_item("omitted", &h.omitted)?;
    Ok(d)
}
#[pyfunction(name = "_solution_count_free_dual_space")]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (shape, cell_parameters, hkl, amplitudes, phases, support, cycles, peak_sigma, peak_cap, mask_sigma, hybrid_m, omit_fraction, seed, patterson_rank, checkpoint_accept_cycle=None, reusable_workspaces=false, filtered_peak_search=false))]
fn count_free<'py>(
    py: Python<'py>,
    shape: [usize; 3],
    cell_parameters: Vec<f64>,
    hkl: PyReadonlyArray1<'_, i64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    phases: PyReadonlyArray1<'_, f64>,
    support: PyReadonlyArray1<'_, u8>,
    cycles: usize,
    peak_sigma: f64,
    peak_cap: usize,
    mask_sigma: f64,
    hybrid_m: f64,
    omit_fraction: f64,
    seed: u64,
    patterson_rank: Option<usize>,
    checkpoint_accept_cycle: Option<usize>,
    reusable_workspaces: bool,
    filtered_peak_search: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cell = cell(&cell_parameters)?;
    let hkl = hkl_rows(&hkl)?;
    let amplitudes = amplitudes.as_slice()?.to_vec();
    let phases = phases.as_slice()?.to_vec();
    let support = support.as_slice()?.to_vec();
    let options = DualSpaceOptions {
        cycles,
        peak_selection: phasesmith_solution::PeakSelection::Adaptive {
            sigma: peak_sigma,
            cap: peak_cap,
        },
        mask_sigma,
        hybrid_m,
        omit_fraction,
        seed,
        patterson_rank,
        cancellation: None,
    };
    let state = py
        .detach(move || {
            if checkpoint_accept_cycle.is_none() {
                return phasesmith_solution::dual_space_with_execution(
                    shape,
                    cell,
                    &hkl,
                    &amplitudes,
                    &phases,
                    &support,
                    &options,
                    None,
                    None,
                    phasesmith_solution::PhasingExecution {
                        score_only_evaluation: false,
                        reusable_fourier_workspaces: reusable_workspaces,
                        filtered_peak_search,
                    },
                );
            }
            if filtered_peak_search || reusable_workspaces {
                return Err(phasesmith_solution::SolutionError(
                    "diagnostic checkpoint and execution modes must be tested separately".into(),
                ));
            }
            let mut handoff = |_: &[f64], done: usize, _: usize| {
                Ok(checkpoint_accept_cycle.is_some_and(|n| n > 0 && done >= n))
            };
            phasesmith_solution::dual_space_monitored(
                shape,
                cell,
                &hkl,
                &amplitudes,
                &phases,
                &support,
                &options,
                None,
                if checkpoint_accept_cycle.is_some() {
                    Some(&mut handoff)
                } else {
                    None
                },
            )
        })
        .map_err(invalid)?;
    let d = PyDict::new(py);
    d.set_item("density", state.trial.density.into_pyarray(py))?;
    d.set_item("history", state.trial.history.into_pyarray(py))?;
    d.set_item("best_cycle", state.trial.best_cycle)?;
    d.set_item("initialization", state.initialization)?;
    d.set_item("patterson_vector_grid", state.patterson_vector)?;
    if let Some(h) = state.adaptive_history {
        d.set_item("adaptive_peaks", adaptive_record(py, &h)?)?;
    }
    d.set_item("checkpoints", checkpoint_records(py, &state.checkpoints)?)?;
    Ok(d)
}

#[pyfunction(name = "_solution_filtered_peaks")]
#[pyo3(signature=(density,shape,sigma,cap,filtered=true))]
fn filtered_peaks<'py>(
    py: Python<'py>,
    density: PyReadonlyArray1<'_, f64>,
    shape: [usize; 3],
    sigma: f64,
    cap: usize,
    filtered: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let density = density.as_slice()?.to_vec();
    let (peaks, effective) = py
        .detach(|| {
            if filtered {
                phasesmith_solution::PreparedAdaptivePeaks::new(shape)?.select(&density, sigma, cap)
            } else {
                phasesmith_solution::adaptive_peaks(&density, shape, sigma, cap)
            }
        })
        .map_err(invalid)?;
    let out = PyDict::new(py);
    out.set_item("peaks", super::array2(py, peaks))?;
    out.set_item("effective_sigma", effective)?;
    Ok(out)
}
#[pyfunction(name = "_solution_adaptive_projection")]
#[allow(clippy::too_many_arguments)]
fn adaptive_projection<'py>(
    py: Python<'py>,
    density: PyReadonlyArray1<'_, f64>,
    shape: [usize; 3],
    cell_parameters: Vec<f64>,
    sigma: f64,
    cap: usize,
    mask_sigma: f64,
    omit_fraction: f64,
    seed: u64,
    cycle: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rho = density.as_slice()?.to_vec();
    let cell = cell(&cell_parameters)?;
    let options = DualSpaceOptions {
        cycles: 1,
        peak_selection: phasesmith_solution::PeakSelection::Adaptive { sigma, cap },
        mask_sigma,
        hybrid_m: 3.,
        omit_fraction,
        seed,
        patterson_rank: None,
        cancellation: None,
    };
    let (peaks, effective) =
        phasesmith_solution::adaptive_peaks(&rho, shape, sigma, cap).map_err(invalid)?;
    let projected = py
        .detach(move || phasesmith_solution::atomic_projection(&rho, shape, cell, &options, cycle))
        .map_err(invalid)?;
    let d = PyDict::new(py);
    d.set_item("density", projected.into_pyarray(py))?;
    d.set_item("peaks", super::array2(py, peaks))?;
    d.set_item("effective_sigma", effective)?;
    Ok(d)
}
fn checkpoint_records<'py>(
    py: Python<'py>,
    records: &[phasesmith_solution::DualSpaceCheckpoint],
) -> PyResult<Bound<'py, PyList>> {
    let items = PyList::empty(py);
    for c in records {
        let d = PyDict::new(py);
        d.set_item("completed_cycles", c.completed_cycles)?;
        d.set_item("best_cycle", c.best_cycle)?;
        d.set_item("score", c.score)?;
        d.set_item("accepted", c.accepted)?;
        items.append(d)?;
    }
    Ok(items)
}

pub(super) fn trial_records(
    py: Python<'_>,
    records: Vec<phasesmith_solution::PhasingTrial>,
) -> PyResult<Bound<'_, PyList>> {
    let trials = PyList::empty(py);
    for t in records {
        let item = PyDict::new(py);
        item.set_item("method", t.method)?;
        item.set_item("initialization", t.initialization)?;
        item.set_item("patterson_vector_grid", t.patterson_vector)?;
        item.set_item("seed", t.seed)?;
        item.set_item("completed_cycles", t.cycles)?;
        item.set_item("best_cycle", t.best_cycle)?;
        item.set_item("score_kind", t.score_kind)?;
        item.set_item("mask_sigma_angstrom", t.mask_sigma)?;
        item.set_item("checkpoints", checkpoint_records(py, &t.checkpoints)?)?;
        if let Some(h) = &t.adaptive_peaks {
            item.set_item("adaptive_peaks", adaptive_record(py, h)?)?;
            item.set_item("atomicity_mode", "adaptive")?;
        } else if t.method == "dual_space" {
            item.set_item("atomicity_mode", "fixed")?;
        }
        if t.method == "dual_space" {
            item.set_item("hybrid_m", 3.)?;
            item.set_item("sharpening_q", 0.5)?;
            item.set_item("omit_fraction", 0.1)?;
            item.set_item("omit_period", 3)?;
        }
        trials.append(item)?;
    }
    Ok(trials)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(filtered_peaks, module)?)?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    module.add_function(wrap_pyfunction!(count_free, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_projection, module)?)?;
    module.add_function(wrap_pyfunction!(projection, module)?)?;
    Ok(())
}
