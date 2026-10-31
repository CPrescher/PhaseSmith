//! Phase alignment, map completion and selective refinement, in that order.
use super::{
    FitPass, Instant, IntegratedPeakDensity, PreparedData, SharedSearchOptions,
    SharedSearchSummary, Solution, SolutionError, SolveOptions, SpaceGroup, State, UnitCell,
    complete, distance2, error, extract_peaks, project_fixed_sets, record, refine, separated,
    xray_species_metadata,
};
use crate::{PreparedPeakCompletion, PreparedPhaseOrigin};

#[path = "shared_assignment_repair.rs"]
mod joint_repair;
pub(super) use joint_repair::repair;
#[path = "isotropic_assignment.rs"]
mod assignment;

/// One tested phase-origin setting and its actual completion effort.
#[derive(Clone, Debug)]
pub struct PhaseSettingAudit {
    /// Source-to-supplied-setting hand.
    pub hand: i32,
    /// Source-to-supplied-setting origin shift.
    pub shift: [f64; 3],
    /// Wrapped phase-equivalence merit, random expectation one.
    pub alpha: f64,
    /// Smooth phase-equivalence loss.
    pub chord: f64,
    /// Unique physically separated peak orbits.
    pub orbits: usize,
    /// Proposed uncalibrated-density/electron-number scales.
    pub scales: Vec<f64>,
    /// Complete cheap count/label proposals.
    pub candidates: usize,
    /// Fixed-coordinate candidate scores, without nonlinear refinement.
    pub evaluations: usize,
    /// Accepted cheap categorical edits.
    pub changes: usize,
    /// Short isotropic screens actually attempted.
    pub screens: usize,
    /// Requested-mode refinements actually attempted.
    pub promotions: usize,
    /// Rejection or numerical/resource diagnostic.
    pub diagnostic: Option<String>,
    /// Motif kind, relative scale and supporting images.
    pub chemical_scales: Vec<(String, f64, usize)>,
    /// Cheap post-fit label scores.
    pub reassignment_evaluations: usize,
    /// Accepted cached label changes.
    pub reassignment_changes: usize,
    /// Post-fit label alternatives actually refined.
    pub reassignment_screens: usize,
}
/// Actual per-map work; all stage timers include unsuccessful attempts.
#[derive(Clone, Debug, Default)]
pub struct PhaseOriginAudit {
    /// P1 start index.
    pub trial: usize,
    /// Independent families used in the cheap amplitude screens.
    pub screening_families: usize,
    /// Independent families selected for phase alignment.
    pub phase_families: usize,
    /// Returned origin candidates before the alpha gate.
    pub origin_candidates: usize,
    /// Actual FFT seed grid for origin alignment.
    pub origin_sampling_shape: [usize; 3],
    /// Largest phase-equivalence harmonic per axis (Nyquist clipping is observable).
    pub origin_harmonic_extent: [usize; 3],
    /// Settings actually passed to density recycling/completion.
    pub settings: Vec<PhaseSettingAudit>,
    /// Time in phase-only origin search.
    pub origin_seconds: f64,
    /// Time in supplied-symmetry density recycling.
    pub recycling_seconds: f64,
    /// Time preparing/scoring cached peak proposals.
    pub completion_seconds: f64,
    /// Time in short isotropic fits.
    pub screening_seconds: f64,
    /// Time in requested-mode final fits.
    pub refinement_seconds: f64,
    /// Map-level rejection diagnostic.
    pub diagnostic: Option<String>,
    /// Longer isotropic parent fits attempted.
    pub assignment_parent_fits: usize,
    /// Fixed-geometry alternative-label scores.
    pub assignment_neighbor_scores: usize,
    /// Longer isotropic alternative fits attempted.
    pub assignment_neighbor_fits: usize,
    /// Comparable isotropic candidates retained for promotion.
    pub assignment_retained: usize,
    /// Original two-slot promotions avoided by selective promotion.
    pub assignment_deferred_promotions: usize,
    /// Independent observed families used for assignment relaxation.
    pub assignment_families: usize,
    /// Comparable independent-family isotropic ranks, ordered best first.
    pub assignment_scores: Vec<f64>,
    /// Actual bounded assignment-ambiguity diagnostic.
    pub ambiguity_scores: usize,
    /// Actual bounded assignment-ambiguity diagnostic.
    pub ambiguity_relaxations: usize,
    /// Actual bounded assignment-ambiguity diagnostic.
    pub ambiguity_rounds: usize,
    /// Actual bounded assignment-ambiguity diagnostic.
    pub ambiguity_retained: usize,
    /// Actual bounded assignment-ambiguity diagnostic.
    pub ambiguity_unresolved: bool,
    /// Final competitors with an SSE/R1 tradeoff; retained after the reference.
    pub ambiguity_metric_disagreements: usize,
}
pub(super) struct PhaseSearchContext {
    plan: PreparedPhaseOrigin,
    data: PreparedData,
    prediction: Option<crate::PreparedReflectionPrediction>,
}
impl PhaseSearchContext {
    pub(super) fn new(
        data: &PreparedData,
        group: &SpaceGroup,
        prediction: Option<crate::PreparedReflectionPrediction>,
    ) -> Result<Self, SolutionError> {
        let plan = PreparedPhaseOrigin::new(group, &data.hkl, &data.amplitudes)?;
        let reps = &plan.representatives;
        let count = reps.len().min(2048);
        let mut rows: Vec<_> = (0..count).map(|i| reps[i * reps.len() / count]).collect();
        rows.sort_unstable();
        let screening = PreparedData {
            hkl: rows.iter().map(|&i| data.hkl[i]).collect(),
            amplitudes: rows.iter().map(|&i| data.amplitudes[i]).collect(),
            intensity: rows.iter().map(|&i| data.intensity[i]).collect(),
            sigma: rows.iter().map(|&i| data.sigma[i]).collect(),
            family_count: rows.len(),
            absent_observations: data.absent_observations,
            nonpositive_families: rows.iter().filter(|&&i| data.intensity[i] <= 0.).count(),
            d_min: data.d_min,
        };
        Ok(Self {
            plan,
            data: screening,
            prediction,
        })
    }
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn peak_orbits(
    cell: UnitCell,
    group: &SpaceGroup,
    shape: [usize; 3],
    density: &[f64],
    data: &PreparedData,
    search: &SharedSearchOptions,
    options: &SolveOptions,
    deadline: Instant,
) -> Result<(Vec<[f64; 3]>, Vec<f64>), SolutionError> {
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let integral = IntegratedPeakDensity::new(
        cell,
        shape,
        &data.hkl,
        &data.amplitudes,
        density,
        search.peak_density_radius_angstrom,
    )?;
    let peaks = extract_peaks(density, shape, (4 * search.maximum_atoms).min(128))?;
    let mut result: Vec<([f64; 3], f64)> = Vec::new();
    let mut occupied = Vec::new();
    for mut point in peaks {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let near: Vec<_> = group
            .operations()
            .iter()
            .filter(|op| {
                distance2(&geometry, point, op.apply_fractional(point))
                    .is_ok_and(|d| d < options.symmetry_tolerance_angstrom.powi(2))
            })
            .copied()
            .collect();
        // Keep a rank-increasing basis of at most three affine constraints.
        let mut independent = Vec::new();
        let mut rows = Vec::new();
        let mut rank = 0;
        for &op in &near {
            let rotation = op.rotation();
            let next: Vec<_> = (0..3)
                .map(|a| std::array::from_fn(|b| f64::from(rotation[a][b]) - f64::from(a == b)))
                .collect();
            let mut combined = rows.clone();
            combined.extend(next);
            let new_rank =
                crate::math3::orthogonal_vectors(combined.iter().copied(), crate::math3::IDENTITY)
                    .len();
            if new_rank > rank {
                independent.push(op);
                rows = combined;
                rank = new_rank;
            }
            if rank == 3 {
                break;
            }
        }
        if let Some(projected) = project_fixed_sets(
            &geometry,
            point,
            &independent,
            options.symmetry_tolerance_angstrom,
        )? {
            if near.iter().all(|op| {
                distance2(&geometry, projected, op.apply_fractional(projected))
                    .is_ok_and(|d| d < 1e-12)
            }) {
                point = projected;
            }
        }
        let images = group
            .expand_sites(&[point], 1e-8)
            .map_err(|e| error(e.to_string()))?
            .fractional_xyz;
        if images.len() > search.maximum_atoms
            || images.len() + occupied.len() > 4 * search.maximum_atoms
            || !separated(&geometry, &images, &occupied)?
        {
            continue;
        }
        let mass = images
            .iter()
            .map(|&p| integral.evaluate(p).map(|v| v.value))
            .collect::<Result<Vec<_>, _>>()?;
        #[allow(clippy::cast_precision_loss)]
        let mass = mass.iter().sum::<f64>() / mass.len() as f64;
        if mass <= 0. {
            continue;
        }
        occupied.extend(images);
        result.push((point, mass));
        if result.len() >= 128 {
            break;
        }
    }
    result.sort_by(|a, b| b.1.total_cmp(&a.1));
    if let Some(first) = result.first() {
        let cutoff = first.1 * 0.02;
        result.retain(|(_, m)| *m >= cutoff);
    }
    Ok(result.into_iter().unzip())
}
fn density_scales(
    cell: UnitCell,
    group: &SpaceGroup,
    points: &[[f64; 3]],
    masses: &[f64],
    elements: &[String],
) -> Result<Vec<f64>, SolutionError> {
    let max_z = elements
        .iter()
        .filter_map(|e| xray_species_metadata(e).map(|m| m.atomic_number))
        .max()
        .unwrap_or(1);
    let heavy = masses[0] / f64::from(max_z);
    let mut scales = vec![heavy * 0.85, heavy, heavy * 1.15];
    if elements.iter().any(|e| e == "C") {
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let mut images = Vec::new();
        for (&point, &mass) in points.iter().zip(masses) {
            for p in group
                .expand_sites(&[point], 1e-8)
                .map_err(|e| error(e.to_string()))?
                .fractional_xyz
            {
                images.push((p, mass));
            }
            if images.len() > 256 {
                break;
            }
        }
        let mut carbon = Vec::new();
        for (i, (p, m)) in images.iter().enumerate() {
            for (q, n) in &images[..i] {
                if (m / n - 1.).abs() > 0.25 {
                    continue;
                }
                let d = distance2(&geometry, *p, *q)?;
                if (1.25_f64.powi(2)..=1.65_f64.powi(2)).contains(&d) {
                    carbon.push((m + n) / 12.);
                }
            }
        }
        if carbon.len() >= 3 {
            carbon.sort_by(f64::total_cmp);
            scales.push(carbon[carbon.len() / 2]);
        }
    }
    Ok(scales)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn search(
    context: &PhaseSearchContext,
    cell: UnitCell,
    group: &SpaceGroup,
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    elements: &[String],
    trial: usize,
    density: &[f64],
    deadline: Instant,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    tangent: Option<&crate::PreparedTangent>,
) -> Result<bool, SolutionError> {
    let mut trace = PhaseOriginAudit {
        trial,
        screening_families: context.data.family_count,
        phase_families: context.plan.selected_families,
        ..PhaseOriginAudit::default()
    };
    let start = Instant::now();
    trace.origin_sampling_shape = if search.independent_origin_sampling {
        context.plan.sampling_shape()
    } else {
        shape.map(|n| n.clamp(16, 64))
    };
    trace.origin_harmonic_extent = context.plan.sampling_extent();
    let origins = context.plan.search_sampled(
        density,
        shape,
        search.beam_width.min(4),
        Some(deadline),
        options.cancellation.as_ref(),
        search.independent_origin_sampling,
    );
    trace.origin_seconds = start.elapsed().as_secs_f64();
    let origins = match origins {
        Ok(c) => c,
        Err(e) => {
            trace.diagnostic = Some(e.to_string());
            audit.phase_origins.push(trace);
            return Ok(false);
        }
    };
    trace.origin_candidates = origins.len();
    let support = if let Some(p) = &context.prediction {
        p.support.clone()
    } else {
        crate::solver::reciprocal_support(shape, cell, group, &result.data)?
    };
    let mut screened = Vec::new();
    for origin in origins.into_iter().filter(|c| c.alpha <= 0.3).take(2) {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let mut setting = PhaseSettingAudit {
            hand: origin.hand,
            shift: origin.shift,
            alpha: origin.alpha,
            chord: origin.chord,
            orbits: 0,
            scales: Vec::new(),
            candidates: 0,
            evaluations: 0,
            changes: 0,
            screens: 0,
            promotions: 0,
            diagnostic: None,
            chemical_scales: Vec::new(),
            reassignment_evaluations: 0,
            reassignment_changes: 0,
            reassignment_screens: 0,
        };
        let start = Instant::now();
        let recycled = crate::symmetry_recycling::recycle_density_guided_prediction(
            shape,
            &result.data.hkl,
            &result.data.amplitudes,
            &support,
            group,
            density,
            origin.hand,
            origin.shift,
            search.recycling_cycles,
            Some(deadline),
            options.cancellation.as_ref(),
            tangent,
            options.seed.wrapping_add(trial as u64),
            context.prediction.as_ref(),
            search.reusable_fourier_workspaces,
            search.batched_fft,
        );
        trace.recycling_seconds += start.elapsed().as_secs_f64();
        let recycled = match recycled {
            Ok(c) => c,
            Err(e) => {
                setting.diagnostic = Some(e.to_string());
                trace.settings.push(setting);
                continue;
            }
        };
        audit.prediction_steps += recycled.prediction.len();
        audit.prediction_updates += recycled.prediction.iter().map(|s| s.updated).sum::<usize>();
        audit.prediction_clipped += recycled.prediction.iter().map(|s| s.clipped).sum::<usize>();
        audit.prediction_seconds += recycled.prediction_seconds;
        let start = Instant::now();
        let proposed = (|| {
            let (points, masses) = peak_orbits(
                cell,
                group,
                shape,
                &recycled.density,
                &result.data,
                search,
                options,
                deadline,
            )?;
            setting.orbits = points.len();
            if points.is_empty() {
                return Err(error("no separated positive peak orbits"));
            }
            setting.scales = density_scales(cell, group, &points, &masses, elements)?;
            if search.chemical_assignment {
                for proposal in
                    crate::chemical_density_scales(cell, group, &points, &masses, elements)?
                {
                    setting.chemical_scales.push((
                        proposal.kind.to_owned(),
                        proposal.scale,
                        proposal.support,
                    ));
                    for factor in [0.9, 1., 1.1] {
                        let scale = proposal.scale * factor;
                        if setting.scales.len() < 8
                            && setting.scales.iter().all(|v| (v / scale - 1.).abs() > 0.02)
                        {
                            setting.scales.push(scale);
                        }
                    }
                }
            }
            let cache = PreparedPeakCompletion::new(
                cell,
                group,
                &points,
                elements,
                &context.data.hkl,
                &context.data.amplitudes,
                search.maximum_atoms,
            )?;
            let found = cache.search(
                &masses,
                &setting.scales,
                search.beam_width.min(4),
                Some(deadline),
                options.cancellation.as_ref(),
            )?;
            setting.candidates = found.candidates.len();
            setting.evaluations = found.evaluations;
            setting.changes = found.changes;
            Ok((points, found.candidates))
        })();
        trace.completion_seconds += start.elapsed().as_secs_f64();
        let (points, candidates) = match proposed {
            Ok(c) => c,
            Err(e) => {
                setting.diagnostic = Some(e.to_string());
                trace.settings.push(setting);
                continue;
            }
        };
        let start = Instant::now();
        for candidate in candidates
            .into_iter()
            .filter(|c| c.objective <= 0.25)
            .take(3)
        {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            let sites: Vec<_> = candidate
                .labels
                .iter()
                .zip(&points)
                .filter_map(|(l, &p)| l.map(|e| (elements[e].clone(), p)))
                .collect();
            let model = crate::symmetry_model::model_in_setting(cell, group, &sites)?;
            setting.screens += 1;
            if let Ok(state) = refine(
                model,
                &context.data,
                options,
                deadline,
                (origin.hand, origin.shift),
                candidate.u_iso,
                trial * search.beam_width + trace.settings.len(),
                &[],
                "phase_completion",
                String::new(),
                audit,
                FitPass::Screen(search.screening_iterations.min(8)),
            ) {
                screened.push((state, trace.settings.len()));
            }
        }
        trace.screening_seconds += start.elapsed().as_secs_f64();
        trace.settings.push(setting);
    }
    screened.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
    if search.chemical_assignment {
        let start = Instant::now();
        for (state, setting_index) in screened.iter_mut().take(2) {
            for _ in 0..2 {
                if crate::stopped(Some(deadline), options.cancellation.as_ref())
                    || state.candidate.r1 > 0.3
                {
                    break;
                }
                let Some(seed) = state.promotion.as_ref() else {
                    break;
                };
                let points: Vec<_> = seed.model.sites.iter().map(|s| s.fractional_xyz).collect();
                let labels: Vec<_> = seed
                    .model
                    .sites
                    .iter()
                    .map(|s| elements.iter().position(|e| *e == s.species))
                    .collect();
                let cache = PreparedPeakCompletion::new(
                    cell,
                    group,
                    &points,
                    elements,
                    &context.data.hkl,
                    &context.data.amplitudes,
                    search.maximum_atoms,
                )?
                .with_site_displacements(&seed.site_u)?;
                let changed =
                    cache.reassign(&labels, Some(deadline), options.cancellation.as_ref())?;
                let setting = &mut trace.settings[*setting_index];
                setting.reassignment_evaluations += changed.evaluations;
                setting.reassignment_changes += changed.changes;
                let mut improved = None;
                for proposal in changed.candidates.into_iter().take(2) {
                    if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                        break;
                    }
                    let mut model = seed.model.clone();
                    for (site, label) in model.sites.iter_mut().zip(proposal.labels) {
                        site.species.clone_from(
                            &elements[label.ok_or_else(|| error("omitted reassignment orbit"))?],
                        );
                    }
                    setting.reassignment_screens += 1;
                    if let Ok(next) = refine(
                        model,
                        &context.data,
                        options,
                        deadline,
                        (1, [0.; 3]),
                        state.candidate.u_iso,
                        state.lineage,
                        &state.moves,
                        "chemical_relabel",
                        String::new(),
                        audit,
                        FitPass::ScreenSeed(&seed.site_u, search.screening_iterations.min(4)),
                    ) {
                        if next.score < state.score - 1e-8
                            && improved
                                .as_ref()
                                .is_none_or(|old: &State| next.score < old.score)
                        {
                            improved = Some(next);
                        }
                    }
                }
                if let Some(next) = improved {
                    *state = next;
                } else {
                    break;
                }
            }
        }
        trace.screening_seconds += start.elapsed().as_secs_f64();
        screened.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
    }
    if search.isotropic_assignment_refinement {
        let start = Instant::now();
        screened = assignment::relax(
            context,
            cell,
            group,
            options,
            search,
            elements,
            &context.data,
            deadline,
            screened,
            &mut trace,
            audit,
        )?;
        trace.screening_seconds += start.elapsed().as_secs_f64();
    }
    let start = Instant::now();
    if search.ambiguous_assignments {
        screened.sort_by(|a, b| {
            b.0.assignment_reference
                .cmp(&a.0.assignment_reference)
                .then(a.0.score.total_cmp(&b.0.score))
        });
    }
    let mut reference_fit = None::<(f64, f64)>;
    let best_isotropic = screened.first().map_or(f64::INFINITY, |s| s.0.score);
    let mut primary_fit = false;
    let mut primary_target = false;
    for (index, (state, setting)) in screened
        .into_iter()
        .filter(|(s, _)| s.candidate.r1 <= 0.3)
        .take(2)
        .enumerate()
    {
        if search.isotropic_assignment_refinement
            && trace.assignment_families > 0
            && index > 0
            && ((primary_target && !search.ambiguous_assignments)
                || (primary_fit && state.score > best_isotropic + 6.))
        {
            trace.assignment_deferred_promotions += 1;
            continue;
        }
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let seed = state.promotion.as_ref().expect("isotropic promotion seed");
        trace.settings[setting].promotions += 1;
        if let Ok(mut full) = refine(
            seed.model.clone(),
            &result.data,
            options,
            deadline,
            (1, [0.; 3]),
            state.candidate.u_iso,
            state.lineage,
            &[],
            "phase_completion",
            String::new(),
            audit,
            FitPass::Promote(&seed.site_u),
        ) {
            if search.ambiguous_assignments {
                if index == 0 {
                    reference_fit = Some((full.score, full.candidate.r1));
                } else if let Some((score, r1)) = reference_fit {
                    full.assignment_disfavoured =
                        !crate::peak_completion::ambiguity::replacement_supported(
                            full.score,
                            full.candidate.r1,
                            score,
                            r1,
                        );
                    trace.ambiguity_metric_disagreements +=
                        usize::from(full.assignment_disfavoured);
                }
            }
            primary_fit = true;
            primary_target = complete(&full, elements) && full.candidate.r1 <= options.r1_target;
            record(
                winners,
                &[full],
                elements,
                search.beam_width,
                trial,
                density,
                result,
            );
        }
    }
    trace.refinement_seconds = start.elapsed().as_secs_f64();
    if search.ambiguous_assignments {
        let current: Vec<_> = winners
            .iter()
            .filter(|s| s.candidate.trial == trial)
            .take(2)
            .collect();
        if let [a, b] = current.as_slice() {
            let counts = |s: &State| {
                let mut c = std::collections::BTreeMap::new();
                for e in &s.candidate.species {
                    *c.entry(e.clone()).or_insert(0usize) += 1;
                }
                c
            };
            trace.ambiguity_unresolved = (b.score - a.score).abs() <= 6. && counts(a) != counts(b);
        }
    }
    if trace.settings.is_empty() {
        trace.diagnostic = Some("no phase origin passed alpha <= 0.3".into());
    }
    audit.phase_origins.push(trace);
    let accepted = winners
        .first()
        .is_some_and(|s| complete(s, elements) && s.candidate.r1 <= options.r1_target);
    if accepted {
        result.termination = "r1_target";
    }
    Ok(accepted)
}
