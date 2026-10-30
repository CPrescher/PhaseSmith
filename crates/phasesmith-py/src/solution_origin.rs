//! Owned diagnostic boundary for measured-phase origin equations.
use super::invalid;
use crate::{NativePreparedReflectionGenerator, hkl_rows};
use npy::{IntoPyArray, PyReadonlyArray1};
use phasesmith_solution::{PhaseOriginMap, PreparedPhaseOrigin};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::time::{Duration, Instant};

pub(super) fn origin_records<'py>(
    py: Python<'py>,
    records: &[phasesmith_solution::PhaseOriginAudit],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for a in records {
        let d = PyDict::new(py);
        d.set_item("trial", a.trial)?;
        d.set_item("screening_families", a.screening_families)?;
        d.set_item("phase_families", a.phase_families)?;
        d.set_item("origin_candidates", a.origin_candidates)?;
        d.set_item("origin_sampling_shape", a.origin_sampling_shape)?;
        d.set_item("origin_harmonic_extent", a.origin_harmonic_extent)?;
        d.set_item("origin_seconds", a.origin_seconds)?;
        d.set_item("recycling_seconds", a.recycling_seconds)?;
        d.set_item("completion_seconds", a.completion_seconds)?;
        d.set_item("screening_seconds", a.screening_seconds)?;
        d.set_item("refinement_seconds", a.refinement_seconds)?;
        d.set_item("diagnostic", &a.diagnostic)?;
        d.set_item("assignment_parent_fits", a.assignment_parent_fits)?;
        d.set_item("assignment_neighbor_scores", a.assignment_neighbor_scores)?;
        d.set_item("assignment_neighbor_fits", a.assignment_neighbor_fits)?;
        d.set_item("assignment_retained", a.assignment_retained)?;
        d.set_item(
            "assignment_deferred_promotions",
            a.assignment_deferred_promotions,
        )?;
        d.set_item("assignment_families", a.assignment_families)?;
        d.set_item("assignment_scores", &a.assignment_scores)?;
        d.set_item("ambiguity_scores", a.ambiguity_scores)?;
        d.set_item("ambiguity_relaxations", a.ambiguity_relaxations)?;
        d.set_item("ambiguity_rounds", a.ambiguity_rounds)?;
        d.set_item("ambiguity_retained", a.ambiguity_retained)?;
        d.set_item("ambiguity_unresolved", a.ambiguity_unresolved)?;
        d.set_item(
            "ambiguity_metric_disagreements",
            a.ambiguity_metric_disagreements,
        )?;

        let settings = PyList::empty(py);
        for s in &a.settings {
            let r = PyDict::new(py);
            r.set_item("hand", s.hand)?;
            r.set_item("shift", s.shift)?;
            r.set_item("alpha", s.alpha)?;
            r.set_item("chord", s.chord)?;
            r.set_item("orbits", s.orbits)?;
            r.set_item("scales", &s.scales)?;
            r.set_item("candidates", s.candidates)?;
            r.set_item("evaluations", s.evaluations)?;
            r.set_item("changes", s.changes)?;
            r.set_item("screens", s.screens)?;
            r.set_item("promotions", s.promotions)?;
            r.set_item("diagnostic", &s.diagnostic)?;
            r.set_item("chemical_scales", &s.chemical_scales)?;
            r.set_item("reassignment_evaluations", s.reassignment_evaluations)?;
            r.set_item("reassignment_changes", s.reassignment_changes)?;
            r.set_item("reassignment_screens", s.reassignment_screens)?;
            settings.append(r)?;
        }
        d.set_item("settings", settings)?;
        out.append(d)?;
    }
    Ok(out)
}

#[pyclass(name = "_PeakCompletionPlan", frozen)]
struct NativeCompletion {
    inner: phasesmith_solution::PreparedPeakCompletion,
}
#[pymethods]
impl NativeCompletion {
    #[new]
    #[pyo3(signature=(cell_parameters,group,points,elements,hkl,observed,maximum_atoms=128))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        cell_parameters: Vec<f64>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        points: PyReadonlyArray1<'_, f64>,
        elements: Vec<String>,
        hkl: PyReadonlyArray1<'_, i64>,
        observed: PyReadonlyArray1<'_, f64>,
        maximum_atoms: usize,
    ) -> PyResult<Self> {
        let cell = super::cell(&cell_parameters)?;
        let group = group.space_group().clone();
        let raw = points.as_slice()?;
        if raw.len() % 3 != 0 {
            return Err(invalid("points must be flattened triples"));
        }
        let points: Vec<_> = raw.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let hkl = hkl_rows(&hkl)?;
        let observed = observed.as_slice()?.to_vec();
        py.detach(move || {
            phasesmith_solution::PreparedPeakCompletion::new(
                cell,
                &group,
                &points,
                &elements,
                &hkl,
                &observed,
                maximum_atoms,
            )
            .map(|inner| Self { inner })
        })
        .map_err(invalid)
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<i32>,
        u: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let labels = completion_labels(labels)?;
        let (objective, gradient) = py
            .detach(|| self.inner.evaluate(&labels, u))
            .map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("objective", objective)?;
        d.set_item("gradient", gradient)?;
        Ok(d)
    }
    #[pyo3(signature=(masses,scales,maximum=4,max_seconds=10.))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        masses: PyReadonlyArray1<'_, f64>,
        scales: PyReadonlyArray1<'_, f64>,
        maximum: usize,
        max_seconds: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !max_seconds.is_finite() || !(0.001..=3600.).contains(&max_seconds) {
            return Err(invalid("invalid completion time budget"));
        }
        let masses = masses.as_slice()?.to_vec();
        let scales = scales.as_slice()?.to_vec();
        let result = py
            .detach(|| {
                self.inner.search(
                    &masses,
                    &scales,
                    maximum,
                    Some(Instant::now() + Duration::from_secs_f64(max_seconds)),
                    None,
                )
            })
            .map_err(invalid)?;
        completion_result(py, result)
    }
    fn displaced(&self, site_u: Vec<f64>) -> PyResult<Self> {
        self.inner
            .clone()
            .with_site_displacements(&site_u)
            .map(|inner| Self { inner })
            .map_err(invalid)
    }
    #[pyo3(signature=(labels,site_u,maximum=4))]
    fn ambiguous<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<i32>,
        site_u: Vec<f64>,
        maximum: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let labels = completion_labels(labels)?;
        let result = py
            .detach(|| {
                self.inner.ambiguity_neighbors(
                    &labels,
                    &site_u,
                    maximum,
                    Some(Instant::now() + Duration::from_secs(10)),
                    None,
                )
            })
            .map_err(invalid)?;
        let out = PyDict::new(py);
        let candidates = PyList::empty(py);
        for c in result.candidates {
            let d = PyDict::new(py);
            d.set_item(
                "labels",
                c.labels
                    .iter()
                    .map(|e| e.map_or(-1, |v| i32::try_from(v).expect("bounded labels")))
                    .collect::<Vec<_>>(),
            )?;
            d.set_item("site_u", c.site_u)?;
            d.set_item("objective", c.objective)?;
            d.set_item("edits", c.edits)?;
            candidates.append(d)?;
        }
        out.set_item("candidates", candidates)?;
        out.set_item("evaluations", result.evaluations)?;
        Ok(out)
    }
    fn reassign<'py>(&self, py: Python<'py>, labels: Vec<i32>) -> PyResult<Bound<'py, PyDict>> {
        let labels = completion_labels(labels)?;
        let result = py
            .detach(|| {
                self.inner.reassign(
                    &labels,
                    Some(Instant::now() + Duration::from_secs(10)),
                    None,
                )
            })
            .map_err(invalid)?;
        completion_result(py, result)
    }
    #[pyo3(signature=(labels,maximum=4))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        labels: Vec<i32>,
        maximum: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let labels = completion_labels(labels)?;
        let result = py
            .detach(|| {
                self.inner.assignment_neighbors(
                    &labels,
                    maximum,
                    Some(Instant::now() + Duration::from_secs(10)),
                    None,
                )
            })
            .map_err(invalid)?;
        completion_result(py, result)
    }
}
fn completion_result(
    py: Python<'_>,
    result: phasesmith_solution::PeakCompletionResult,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    let candidates = PyList::empty(py);
    for c in result.candidates {
        let r = PyDict::new(py);
        let labels: Vec<_> = c
            .labels
            .into_iter()
            .map(|l| l.map_or(-1, |v| i32::try_from(v).unwrap_or(-1)))
            .collect();
        r.set_item("labels", labels)?;
        r.set_item("objective", c.objective)?;
        r.set_item("score", c.score)?;
        r.set_item("u_iso", c.u_iso)?;
        r.set_item("atoms", c.atoms)?;
        candidates.append(r)?;
    }
    d.set_item("candidates", candidates)?;
    d.set_item("evaluations", result.evaluations)?;
    d.set_item("changes", result.changes)?;
    Ok(d)
}

fn completion_labels(raw: Vec<i32>) -> PyResult<Vec<Option<usize>>> {
    raw.into_iter()
        .map(|l| {
            if l == -1 {
                Ok(None)
            } else {
                usize::try_from(l)
                    .map(Some)
                    .map_err(|_| invalid("invalid completion label"))
            }
        })
        .collect()
}

#[pyclass(name = "_PhaseOriginMap", frozen)]
struct NativeMap {
    inner: PhaseOriginMap,
}
#[pymethods]
impl NativeMap {
    fn evaluate<'py>(&self, py: Python<'py>, shift: [f64; 3]) -> PyResult<Bound<'py, PyDict>> {
        let e = self.inner.evaluate(shift).map_err(invalid)?;
        let d = PyDict::new(py);
        d.set_item("chord", e.chord)?;
        d.set_item("gradient", e.gradient)?;
        d.set_item("hessian", e.hessian)?;
        d.set_item("alpha", e.alpha)?;
        d.set_item("alpha_gradient", e.alpha_gradient)?;
        d.set_item("pairs", self.inner.pairs)?;
        Ok(d)
    }
}
#[pyclass(name = "_PhaseOriginPlan", frozen)]
struct NativePlan {
    inner: PreparedPhaseOrigin,
}
#[pymethods]
impl NativePlan {
    #[new]
    fn new(
        py: Python<'_>,
        group: PyRef<'_, NativePreparedReflectionGenerator>,
        hkl: PyReadonlyArray1<'_, i64>,
        amplitudes: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let group = group.space_group().clone();
        let hkl = hkl_rows(&hkl)?;
        let amplitudes = amplitudes.as_slice()?.to_vec();
        py.detach(move || {
            PreparedPhaseOrigin::new(&group, &hkl, &amplitudes).map(|inner| Self { inner })
        })
        .map_err(invalid)
    }
    fn map(
        &self,
        py: Python<'_>,
        density: PyReadonlyArray1<'_, f64>,
        shape: [usize; 3],
        hand: i32,
    ) -> PyResult<NativeMap> {
        let source = density.as_slice()?.to_vec();
        py.detach(|| {
            self.inner
                .map(&source, shape, hand)
                .map(|inner| NativeMap { inner })
        })
        .map_err(invalid)
    }
    #[pyo3(signature=(density,shape,maximum=4,max_seconds=10.,independent_sampling=false))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray1<'_, f64>,
        shape: [usize; 3],
        maximum: usize,
        max_seconds: f64,
        independent_sampling: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !max_seconds.is_finite() || !(0.001..=3600.).contains(&max_seconds) {
            return Err(invalid("invalid origin time budget"));
        }
        let source = density.as_slice()?.to_vec();
        let found = py
            .detach(|| {
                self.inner.search_sampled(
                    &source,
                    shape,
                    maximum,
                    Some(Instant::now() + Duration::from_secs_f64(max_seconds)),
                    None,
                    independent_sampling,
                )
            })
            .map_err(invalid)?;
        let d = PyDict::new(py);
        let origins = PyList::empty(py);
        for c in found {
            let row = PyDict::new(py);
            row.set_item("hand", c.hand)?;
            row.set_item("shift", c.shift)?;
            row.set_item("alpha", c.alpha)?;
            row.set_item("chord", c.chord)?;
            origins.append(row)?;
        }
        d.set_item("origins", origins)?;
        d.set_item("families", self.inner.selected_families)?;
        d.set_item("harmonic_extent", self.inner.sampling_extent())?;
        d.set_item(
            "sampling_shape",
            if independent_sampling {
                self.inner.sampling_shape()
            } else {
                shape.map(|n| n.clamp(16, 64))
            },
        )?;
        d.set_item("representatives", &self.inner.representatives)?;
        Ok(d)
    }
}
#[pyfunction(name = "_phase_checkpoint_select")]
#[pyo3(signature=(maps,cycles,quality,scores,ceilings,best_quality=None))]
fn checkpoint_select(
    py: Python<'_>,
    maps: Vec<Vec<f64>>,
    cycles: Vec<usize>,
    quality: Vec<f64>,
    scores: Vec<f64>,
    ceilings: Vec<bool>,
    best_quality: Option<Vec<f64>>,
) -> PyResult<Bound<'_, PyList>> {
    let n = maps.len();
    let best_quality = best_quality.unwrap_or_else(|| quality.clone());
    if [
        cycles.len(),
        quality.len(),
        scores.len(),
        ceilings.len(),
        best_quality.len(),
    ]
    .iter()
    .any(|&v| v != n)
    {
        return Err(invalid("mismatched checkpoint sequences"));
    }
    let result = py
        .detach(|| {
            let mut selector = phasesmith_solution::PhaseCheckpointSelector::default();
            (0..n)
                .map(|i| {
                    selector.consider(
                        &maps[i],
                        cycles[i],
                        quality[i],
                        scores[i],
                        best_quality[i],
                        ceilings[i],
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(invalid)?;
    let out = PyList::empty(py);
    for c in result {
        let row = PyDict::new(py);
        row.set_item("cycles", c.cycles)?;
        row.set_item("cfom", c.cfom)?;
        row.set_item("best_cfom", c.best_cfom)?;
        row.set_item("score", c.score)?;
        row.set_item("map_cosine", c.map_cosine)?;
        row.set_item("selected", c.selected)?;
        row.set_item("reason", c.reason)?;
        out.append(row)?;
    }
    Ok(out)
}
#[pyfunction(name = "_iterative_normal_step")]
#[pyo3(signature=(jacobian, residual, damping, cached=false))]
fn normal_step<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray1<'_, f64>,
    residual: PyReadonlyArray1<'_, f64>,
    damping: f64,
    cached: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let j = jacobian.as_slice()?.to_vec();
    let r = residual.as_slice()?.to_vec();
    let value = py
        .detach(|| {
            if cached {
                phasesmith_solution::cached_normal_step(&j, &r, damping, None, None)
            } else {
                phasesmith_solution::iterative_normal_step(&j, &r, damping, None, None).map(
                    |(step, iterations)| phasesmith_solution::CachedNormalStep {
                        step,
                        iterations,
                        used_cache: false,
                        fell_back: false,
                    },
                )
            }
        })
        .map_err(invalid)?;
    let phasesmith_solution::CachedNormalStep {
        step,
        iterations,
        used_cache,
        fell_back,
    } = value;
    let out = PyDict::new(py);
    out.set_item("step", step.into_pyarray(py))?;
    out.set_item("iterations", iterations)?;
    out.set_item("used_cache", used_cache)?;
    out.set_item("fell_back", fell_back)?;
    Ok(out)
}
#[pyfunction(name = "_chemical_density_scales")]
fn chemical_scales<'py>(
    py: Python<'py>,
    cell_parameters: Vec<f64>,
    group: PyRef<'_, NativePreparedReflectionGenerator>,
    points: PyReadonlyArray1<'_, f64>,
    masses: PyReadonlyArray1<'_, f64>,
    elements: Vec<String>,
) -> PyResult<Bound<'py, PyList>> {
    let cell = super::cell(&cell_parameters)?;
    let group = group.space_group().clone();
    let raw = points.as_slice()?;
    if raw.len() % 3 != 0 {
        return Err(invalid("chemical points must be flattened triples"));
    }
    let points: Vec<_> = raw.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let masses = masses.as_slice()?.to_vec();
    let proposals = py
        .detach(|| {
            phasesmith_solution::chemical_density_scales(cell, &group, &points, &masses, &elements)
        })
        .map_err(invalid)?;
    let out = PyList::empty(py);
    for p in proposals {
        let row = PyDict::new(py);
        row.set_item("kind", p.kind)?;
        row.set_item("scale", p.scale)?;
        row.set_item("support", p.support)?;
        out.append(row)?;
    }
    Ok(out)
}
#[pyfunction(name = "_phase_cycle_quota")]
#[pyo3(signature=(cycles,trials,trial,short=false))]
fn phase_quota(cycles: usize, trials: usize, trial: usize, short: bool) -> PyResult<usize> {
    phasesmith_solution::phase_cycle_quota_with_short_starts(cycles, trials, trial, short)
        .map_err(invalid)
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(phase_quota, module)?)?;
    module.add_function(wrap_pyfunction!(normal_step, module)?)?;
    module.add_function(wrap_pyfunction!(checkpoint_select, module)?)?;
    module.add_function(wrap_pyfunction!(chemical_scales, module)?)?;
    module.add_class::<NativePlan>()?;
    module.add_class::<NativeMap>()?;
    module.add_class::<NativeCompletion>()?;
    Ok(())
}
