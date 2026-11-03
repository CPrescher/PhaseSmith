//! One missing weak-species orbit, completed from a projected stronger framework.
use crate::solution::math3::{dot, mv, periodic_difference, sub};
use crate::solution::{
    AdpMode, Candidate, PreparedData, PreparedOrbitSearch, PreparedSymmetricModel, SolutionError,
    SolveOptions, error, extract_peaks, project_fixed_sets, recover_symmetry,
};
use phasesmith_crystallography::{
    CellGeometry, SpaceGroup, SymmetryOperation, UnitCell, xray_species_metadata,
};
use std::time::Instant;

/// Actual completion work and outcome, including unsuccessful attempts.
#[derive(Clone, Debug)]
pub struct CompletionSummary {
    /// Whether a completed candidate was retained, even if a later stop intervened.
    pub applied: bool,
    /// `applied`, `not_improved`, `failed`, `cancelled`, or `max_runtime`.
    pub status: &'static str,
    /// Explicit failure, unsupported-scope or stop reason.
    pub diagnostic: Option<String>,
    /// Distinct stronger species retained in the framework.
    pub framework_species: Vec<String>,
    /// Number of full-cell framework atoms.
    pub framework_atoms: usize,
    /// Weakest species proposed for replacement.
    pub added_species: Option<String>,
    /// Exact required number of symmetry images.
    pub added_atoms: usize,
    /// Difference-map and original-candidate proposal seeds inspected.
    pub seed_count: usize,
    /// Affine fixed-set projection attempts.
    pub projection_trials: usize,
    /// Distinct complete orbit proposals scored.
    pub orbit_trials: usize,
    /// Complete candidates passed to constrained refinement.
    pub refinements: usize,
    /// Distinct symmetry-compatible partial framework settings inspected.
    pub framework_hypotheses: usize,
    /// Zero-based framework setting chosen for a retained candidate.
    pub selected_framework: Option<usize>,
    /// Framework mismatch before projection, in angstrom.
    pub framework_maximum_error: Option<f64>,
    /// Atomic objective before this completion attempt.
    pub initial_objective: f64,
    /// Best complete polished objective when one was obtained.
    pub best_objective: Option<f64>,
}

#[derive(Clone)]
struct Proposal {
    point: [f64; 3],
    images: Vec<[f64; 3]>,
    objective: f64,
}

fn distance2(g: &CellGeometry, a: [f64; 3], b: [f64; 3]) -> Result<f64, SolutionError> {
    let d = periodic_difference(g, sub(a, b))?;
    Ok(dot(d, mv(g.direct_metric, d)))
}

fn separated(
    g: &CellGeometry,
    points: &[[f64; 3]],
    fixed: &[[f64; 3]],
) -> Result<bool, SolutionError> {
    for (i, &point) in points.iter().enumerate() {
        for &other in fixed.iter().chain(&points[..i]) {
            if distance2(g, point, other)? < 0.25 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Return true only if the candidate is symmetry-complete and materially improves
/// the old profiled amplitude objective. The original candidate is never mutated.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
pub fn complete_candidate(
    cell: UnitCell,
    group: &SpaceGroup,
    data: &PreparedData,
    shape: [usize; 3],
    candidate: &Candidate,
    mode: AdpMode,
    options: &SolveOptions,
    deadline: Instant,
) -> (Option<Candidate>, CompletionSummary) {
    let mut audit = CompletionSummary {
        applied: false,
        status: "failed",
        diagnostic: None,
        framework_species: Vec::new(),
        framework_atoms: 0,
        added_species: None,
        added_atoms: 0,
        seed_count: 0,
        projection_trials: 0,
        orbit_trials: 0,
        refinements: 0,
        framework_hypotheses: 0,
        selected_framework: None,
        framework_maximum_error: None,
        initial_objective: candidate.objective,
        best_objective: None,
    };
    let mut search = || -> Result<Option<Candidate>, SolutionError> {
        let weakest = candidate
            .species
            .iter()
            .min_by_key(|s| xray_species_metadata(s).map_or(u8::MAX, |m| m.atomic_number))
            .ok_or_else(|| error("empty completion candidate"))?
            .clone();
        let mut fixed_species = Vec::new();
        let mut fixed_xyz = Vec::new();
        let mut old_missing = Vec::new();
        for (element, &point) in candidate.species.iter().zip(&candidate.fractional_xyz) {
            if *element == weakest {
                old_missing.push(point);
            } else {
                fixed_species.push(element.clone());
                fixed_xyz.push(point);
            }
        }
        audit.added_species = Some(weakest.clone());
        audit.added_atoms = old_missing.len();
        audit.framework_atoms = fixed_xyz.len();
        audit.framework_species.clone_from(&fixed_species);
        audit.framework_species.sort();
        audit.framework_species.dedup();
        if fixed_xyz.is_empty()
            || old_missing.is_empty()
            || old_missing.len() > group.operations().len()
        {
            return Err(error(
                "completion requires a stronger framework and one missing-species orbit",
            ));
        }
        let hypotheses = crate::solution::recover_symmetry_hypotheses(
            cell,
            group,
            &fixed_species,
            &fixed_xyz,
            options.symmetry_tolerance_angstrom,
            Some(deadline),
            options.cancellation.as_ref(),
            8,
        )?;
        audit.framework_hypotheses = hypotheses.len();
        let original_fixed = fixed_xyz;
        let mut best: Option<Candidate> = None;
        for (hypothesis_index, partial) in hypotheses.into_iter().enumerate() {
            let mut fixed_xyz = original_fixed.clone();
            for site in &partial.sites {
                for image in &site.images {
                    fixed_xyz[image.source_index] =
                        image.operation.apply_fractional(site.fractional_xyz);
                }
            }
            let prepared = PreparedOrbitSearch::new(
                cell,
                group,
                &fixed_species,
                &fixed_xyz,
                &weakest,
                &data.hkl,
                &data.amplitudes,
                candidate.u_iso,
            )?;
            if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
                return Err(error("completion stopped"));
            }
            let map = prepared.difference_map(shape)?;
            let mut seeds = extract_peaks(&map.density, shape, 64)?;
            seeds.extend(old_missing.iter().map(|p| {
                std::array::from_fn(|i| {
                    (p[i] * f64::from(partial.hand) + partial.origin_shift[i]).rem_euclid(1.)
                })
            }));
            let mut proposals: Vec<Proposal> = Vec::new();
            'seeds: for seed in seeds {
                if proposals.len() >= 128 {
                    break;
                }
                audit.seed_count += 1;
                let initial = group
                    .expand_sites(&[seed], 1e-8)
                    .map_err(|e| error(e.to_string()))?;
                let mut states = vec![(
                    seed,
                    Vec::<SymmetryOperation>::new(),
                    initial.fractional_xyz.len(),
                )];
                let mut index = 0;
                while index < states.len() && index < 32 {
                    if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
                        return Err(error("completion stopped"));
                    }
                    let (point, constraints, count) = states[index].clone();
                    index += 1;
                    if count == old_missing.len() {
                        let images = group
                            .expand_sites(&[point], 1e-8)
                            .map_err(|e| error(e.to_string()))?
                            .fractional_xyz;
                        if !separated(&prepared.geometry, &images, &fixed_xyz)? {
                            continue;
                        }
                        let mut duplicate = false;
                        for previous in &proposals {
                            for &image in &previous.images {
                                if distance2(&prepared.geometry, point, image)? < 0.0004 {
                                    duplicate = true;
                                    break;
                                }
                            }
                            if duplicate {
                                break;
                            }
                        }
                        if !duplicate {
                            let evaluation = prepared.evaluate(point)?;
                            audit.orbit_trials += 1;
                            proposals.push(Proposal {
                                point,
                                images,
                                objective: evaluation.objective,
                            });
                            if proposals.len() >= 128 {
                                break 'seeds;
                            }
                        }
                        continue;
                    }
                    if count < old_missing.len() || constraints.len() >= 3 {
                        continue;
                    }
                    for &operation in group.operations() {
                        if states.len() >= 32 {
                            break;
                        }
                        if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
                            return Err(error("completion stopped"));
                        }
                        if constraints.contains(&operation) {
                            continue;
                        }
                        let mut combined = constraints.clone();
                        combined.push(operation);
                        audit.projection_trials += 1;
                        let Some(next) = project_fixed_sets(
                            &prepared.geometry,
                            seed,
                            &combined,
                            options.symmetry_tolerance_angstrom,
                        )?
                        else {
                            continue;
                        };
                        let expansion = group
                            .expand_sites(&[next], 1e-8)
                            .map_err(|e| error(e.to_string()))?;
                        let multiplicity = expansion.fractional_xyz.len();
                        if multiplicity >= count || multiplicity < old_missing.len() {
                            continue;
                        }
                        let mut duplicate = false;
                        for &(old, _, old_count) in &states {
                            if old_count == multiplicity
                                && distance2(&prepared.geometry, next, old)? < 1e-12
                            {
                                duplicate = true;
                                break;
                            }
                        }
                        if !duplicate {
                            states.push((next, combined, multiplicity));
                        }
                    }
                }
            }
            if proposals.is_empty() {
                continue;
            }
            proposals.sort_by(|a, b| {
                a.objective
                    .total_cmp(&b.objective)
                    .then_with(|| a.point[0].total_cmp(&b.point[0]))
                    .then_with(|| a.point[1].total_cmp(&b.point[1]))
                    .then_with(|| a.point[2].total_cmp(&b.point[2]))
            });
            for proposal in proposals.into_iter().take(8) {
                if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
                    break;
                }
                let mut points = fixed_xyz.clone();
                points.extend(&proposal.images);
                let mut species = fixed_species.clone();
                species.extend(std::iter::repeat_n(weakest.clone(), proposal.images.len()));
                let symmetry = recover_symmetry(
                    cell,
                    group,
                    &species,
                    &points,
                    1e-5,
                    Some(deadline),
                    options.cancellation.as_ref(),
                )?;
                let model = PreparedSymmetricModel::new_with_memory_budget(
                    symmetry,
                    &data.hkl,
                    &data.amplitudes,
                    mode,
                    true,
                    false,
                    options.dense_memory_budget,
                )?;
                if data.family_count <= model.parameter_count + 1 {
                    continue;
                }
                let initial = model.initial_parameters(candidate.u_iso.clamp(1e-5, 0.2))?;
                audit.refinements += 1;
                let fit = model.refine(
                    &initial,
                    options.symmetry_iterations,
                    Some(deadline),
                    options.cancellation.as_ref(),
                )?;
                let (mut summary, mut full) =
                    fit.summarize(mode, candidate.r1, candidate.objective);
                if !separated(&prepared.geometry, &full, &[])? {
                    continue;
                }
                let second_hand = summary.hand;
                summary.hand *= partial.hand;
                summary.origin_shift = std::array::from_fn(|i| {
                    (f64::from(second_hand) * partial.origin_shift[i] + summary.origin_shift[i])
                        .rem_euclid(1.)
                });
                for point in &mut full {
                    for (value, shift) in point.iter_mut().zip(partial.origin_shift) {
                        *value = ((*value - shift) * f64::from(partial.hand)).rem_euclid(1.);
                    }
                }
                let mut result = candidate.clone();
                result.species = species;
                result.fractional_xyz = full;
                result.r1 = fit.evaluation.r1;
                result.objective = fit.evaluation.objective;
                result.amplitude_scale = fit.evaluation.amplitude_scale;
                result.u_iso =
                    summary.full_u_equiv.iter().sum::<f64>() / summary.full_u_equiv.len() as f64;
                result.refinement = Some(summary);
                result.refinement_diagnostic = None;
                if best
                    .as_ref()
                    .is_none_or(|old| result.objective < old.objective)
                {
                    audit.selected_framework = Some(hypothesis_index);
                    audit.framework_maximum_error = Some(partial.maximum_matching_error);
                    best = Some(result);
                }
            }
        }
        audit.best_objective = best.as_ref().map(|c| c.objective);
        Ok(best.filter(|c| c.objective < candidate.objective * (1. - 1e-8)))
    };
    let result = search();
    let candidate = match result {
        Ok(Some(c)) => {
            audit.status = "applied";
            Some(c)
        }
        Ok(None) => {
            audit.status = "not_improved";
            audit.diagnostic =
                Some("complete orbit proposals did not improve the existing objective".to_owned());
            None
        }
        Err(e) => {
            audit.diagnostic = Some(e.to_string());
            None
        }
    };
    audit.applied = candidate.is_some();
    if crate::solution::stopped(Some(deadline), options.cancellation.as_ref()) {
        audit.status = if options
            .cancellation
            .as_ref()
            .is_some_and(crate::solution::Cancellation::is_cancelled)
        {
            "cancelled"
        } else {
            "max_runtime"
        };
    }
    (candidate, audit)
}
