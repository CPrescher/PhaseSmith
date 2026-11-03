//! Explicit bounded fast-first effort; escalation depends only on saved numerical candidates.
use super::{Assessment, DiscoveryReport, Error, Observation, Options, Result, SearchRun};
use phasesmith_single_crystal::solution as phasesmith_solution;
use std::time::{Duration, Instant};
/// Requested search effort; automatic effort shares one total deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchEffort {
    /// Coarse shared-map attempt only, including unsuccessful results.
    Fast,
    /// Try fast, then deepen failures except numerically equivalent group descriptions.
    Auto,
    /// Start directly with the configured full-resolution bounded search.
    Thorough,
}
impl SearchEffort {
    /// Stable diagnostic spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Auto => "auto",
            Self::Thorough => "thorough",
        }
    }
}
/// Explicit resource limits for the initial pass. No independent phasing restart runs here.
#[derive(Clone, Debug)]
pub struct FastSearchLimits {
    /// Independent starts, 1..=16.
    pub starts: usize,
    /// Per-start cycle ceiling, 32..=20000.
    pub cycles: usize,
    /// Inclusive initial minimum d in angstrom; final measurements remain unchanged.
    pub minimum_d_angstrom: f64,
    /// Maximum model settings, 1..=64.
    pub maximum_candidates: usize,
    /// Per-setting wall limit.
    pub candidate_seconds: f64,
    /// Complete fast-pass wall limit, within the original total budget.
    pub total_seconds: f64,
}
/// Automatic candidate acceptance and fast-resource policy. The configured outer
/// discovery controls own the full-resolution pass and total budget.
#[derive(Clone, Debug)]
pub struct EffortOptions {
    /// Fast, automatic escalation, or direct full-resolution effort.
    pub mode: SearchEffort,
    /// Maximum full common-data R1 for acceptance, in (0,1].
    pub maximum_common_r1: f64,
    /// Coarse first-pass controls, independent of final-data resolution.
    pub fast: FastSearchLimits,
}
impl EffortOptions {
    pub(super) fn check(&self, total: f64) -> Result<()> {
        let f = &self.fast;
        if !(1..=16).contains(&f.starts)
            || !(32..=20000).contains(&f.cycles)
            || !(1..=64).contains(&f.maximum_candidates)
            || !self.maximum_common_r1.is_finite()
            || self.maximum_common_r1 <= 0.
            || self.maximum_common_r1 > 1.
            || [f.minimum_d_angstrom, f.candidate_seconds, f.total_seconds]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
            || f.candidate_seconds > f.total_seconds
            || f.total_seconds > total
        {
            return Err(Error("invalid fast/auto/thorough effort controls".into()));
        }
        Ok(())
    }
}
/// Actual pass effort and its numerical acceptance; no reference grading enters this record.
#[derive(Clone, Debug)]
pub struct EffortPass {
    /// Executed pass, fast or thorough.
    pub mode: SearchEffort,
    /// Whether the saved candidate met every explicit acceptance check.
    pub accepted: bool,
    /// Acceptance, rejection, cancellation or error reason.
    pub reason: String,
    /// Actual pass seconds, including failed work.
    pub seconds: f64,
    /// Actual bounded-search termination.
    pub termination: String,
    /// Actual initial Fourier grid, if a map completed.
    pub phasing_shape: Option<[usize; 3]>,
    /// Actual completed cycles of every start.
    pub phase_cycles: Vec<usize>,
    /// Settings with actual model attempts.
    pub model_attempts: usize,
}
/// Evidence that further phasing would repeat numerically equivalent group descriptions.
/// This defers automatic work; it never makes an ambiguous candidate accepted.
#[derive(Clone, Debug)]
pub struct DeferredAmbiguity {
    /// Index of the retained setting in the full catalogue.
    pub selected_hypothesis: usize,
    /// Every other BIC-close setting; all have equivalent complete atomic models.
    pub equivalent_hypotheses: Vec<usize>,
    /// Maximum matched atomic separation under one origin/hand, in angstrom.
    pub maximum_position_difference_angstrom: f64,
    /// Maximum absolute difference on the identical normalized common residual set.
    pub maximum_common_residual_difference: f64,
}
/// Chronological effort plus the complete other pass when automatic escalation ran.
#[derive(Clone, Debug)]
pub struct EffortTrace {
    /// User-requested policy.
    pub requested: SearchEffort,
    /// Pass supplying the top-level retained model and setting.
    pub selected_pass: SearchEffort,
    /// Whether the retained candidate passed the explicit effort acceptance checks.
    pub accepted: bool,
    /// Actual chronological passes, including failures.
    pub passes: Vec<EffortPass>,
    /// Why finer/broader work was necessary; absent if it did not run.
    pub escalation_reason: Option<String>,
    /// Why this pass supplies the retained candidate; lower R1 alone cannot upgrade
    /// an existing approximation when neither pass meets acceptance.
    pub selection_reason: &'static str,
    /// Strict numerical equivalence causing automatic work to be deferred.
    /// Acceptance and the unresolved space-group alternatives remain unchanged.
    pub deferred_ambiguity: Option<DeferredAmbiguity>,
    /// Complete nonselected pass, including arrays, hypotheses and failed/unattempted settings.
    pub other_pass: Option<Box<DiscoveryReport>>,
}
fn acceptance(run: &SearchRun, limit: f64) -> &'static str {
    let Some((solution, _)) = &run.0 else {
        return "no_model";
    };
    if !solution.target_met {
        return "own_target_not_met";
    }
    let Some(index) = run.1.selected else {
        return "no_common_symmetry_model";
    };
    let a = &run.1.attempts[index];
    if a.candidate.is_none() || a.status != "candidate" {
        return "no_common_symmetry_model";
    }
    let report = &run.1;
    let hypothesis = &report.hypotheses[index];
    if let Some(prior) = &report.symmetry_prior {
        if !prior.admits(hypothesis) {
            return "supplied_symmetry_not_recovered";
        }
        if report.assessment.laue[report.assessment.processing_laue].status == "contradicted" {
            return "supplied_laue_contradicted";
        }
    } else if super::super::symmetry_prior::unresolved_laue(report, index) {
        return "unresolved_higher_laue";
    }
    if a.parameters
        .is_none_or(|p| p >= report.assessment.friedel_families)
    {
        return "insufficient_common_observations";
    }
    if report
        .alternatives
        .iter()
        .any(|&i| report.hypotheses[i].number != hypothesis.number)
    {
        return "ambiguous_space_group";
    }
    remaining_quality(run, index, limit)
}
// Kept separate so equivalence can never hide a second failed quality check.
fn remaining_quality(run: &SearchRun, index: usize, limit: f64) -> &'static str {
    let report = &run.1;
    let hypothesis = &report.hypotheses[index];
    let a = &report.attempts[index];
    if !super::is_centric(&hypothesis.group) {
        let baseline = report
            .hypotheses
            .iter()
            .zip(&report.attempts)
            .find(|(h, _)| super::inversion_only(&h.group));
        let Some((_, baseline)) = baseline.filter(|(_, a)| !a.origins.is_empty()) else {
            return "centrosymmetry_not_assessed";
        };
        if baseline.origins.iter().any(|o| o.alpha <= 0.3) {
            return "unresolved_centrosymmetry";
        }
    }
    if a.common_r1.is_none_or(|r| !r.is_finite() || r > limit) {
        return "common_r1_not_met";
    }
    if !a.origins.iter().any(|o| o.alpha <= 0.3) {
        return "phase_consistency_not_met";
    }
    "accepted"
}
// These are roundoff/equivalence bounds, not relaxed structure acceptance gates.
// No reference positions, published group or count information enters the decision.
fn deferred_ambiguity(
    run: &SearchRun,
    limit: f64,
    cell: phasesmith_crystallography::UnitCell,
    deadline: Instant,
    cancellation: &phasesmith_solution::Cancellation,
) -> Option<DeferredAmbiguity> {
    if acceptance(run, limit) != "ambiguous_space_group" {
        return None;
    }
    let index = run.1.selected?;
    if remaining_quality(run, index, limit) != "accepted" {
        return None;
    }
    let selected = &run.1.attempts[index];
    let a = selected.candidate.as_ref()?;
    let residuals = selected.common_residuals.as_ref()?;
    if residuals.is_empty() || residuals.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut equivalent = Vec::new();
    let mut different_group = false;
    let mut maximum_residual_difference = 0_f64;
    let mut maximum_position_difference = 0_f64;
    for &other in &run.1.alternatives {
        if cancellation.is_cancelled() || Instant::now() >= deadline {
            return None;
        }
        if other == index {
            continue;
        }
        let alternative = &run.1.attempts[other];
        if alternative.status != "candidate" || alternative.parameters != selected.parameters {
            return None;
        }
        let (Some(b), Some(other_residuals)) =
            (&alternative.candidate, &alternative.common_residuals)
        else {
            return None;
        };
        if residuals.len() != other_residuals.len()
            || other_residuals.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        let difference = residuals
            .iter()
            .zip(other_residuals)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        if difference > 128. * f64::EPSILON {
            return None;
        }
        // Full-cell, element-preserving bijection. The generic matcher may
        // conservatively fail; only roundoff-scale matches can defer work here.
        let distance = phasesmith_solution::model_agreement_distance(
            cell,
            &a.species,
            &a.fractional_xyz,
            &b.species,
            &b.fractional_xyz,
            Some(deadline),
            Some(cancellation),
        )
        .ok()??;
        if distance > 1e-8 {
            return None;
        }
        maximum_residual_difference = maximum_residual_difference.max(difference);
        maximum_position_difference = maximum_position_difference.max(distance);
        different_group |= run.1.hypotheses[other].number != run.1.hypotheses[index].number;
        equivalent.push(other);
    }
    different_group.then_some(DeferredAmbiguity {
        selected_hypothesis: index,
        equivalent_hypotheses: equivalent,
        maximum_position_difference_angstrom: maximum_position_difference,
        maximum_common_residual_difference: maximum_residual_difference,
    })
}

fn summary(run: &SearchRun, mode: SearchEffort, limit: f64) -> EffortPass {
    let reason = acceptance(run, limit);
    EffortPass {
        mode,
        accepted: reason == "accepted",
        reason: reason.into(),
        seconds: run.1.seconds,
        termination: run.1.termination.into(),
        phasing_shape: run.1.phase_map.as_ref().map(|m| m.phasing_shape),
        phase_cycles: run
            .1
            .phase_map
            .as_ref()
            .map_or_else(Vec::new, |m| m.histories.iter().map(Vec::len).collect()),
        model_attempts: run.1.attempts.iter().filter(|a| !a.runs.is_empty()).count(),
    }
}
fn pass_options(
    original: &Options,
    controls: &EffortOptions,
    fast: bool,
    remaining: f64,
) -> Options {
    let mut o = original.clone();
    let d = o.discovery.as_mut().expect("validated discovery");
    d.effort = None;
    d.total_seconds = remaining;
    if fast {
        d.allow_independent_restarts = false;
    }
    d.plausible_centrosymmetric_exit = fast;
    d.phase.phasing.first_pass_cycles = None;
    if fast {
        let f = &controls.fast;
        d.total_seconds = d.total_seconds.min(f.total_seconds);
        d.candidate_seconds = f.candidate_seconds;
        d.maximum_candidates = f.maximum_candidates;
        d.phase.starts = f.starts;
        d.phase.workers = d.phase.workers.min(f.starts);
        d.phase.cycles = f.cycles;
        d.phase.phasing.minimum_d_angstrom = Some(f.minimum_d_angstrom);
        d.phase.phasing.adaptive_quality = true;
        d.reuse_phase_map = true;
        o.search_trials = f.starts;
        o.search_cycles = f.cycles;
    } else {
        d.phase.phasing.minimum_d_angstrom = None;
    }
    d.candidate_seconds = d.candidate_seconds.min(d.total_seconds);
    d.phase.seconds = d.phase.seconds.min(d.total_seconds);
    d.phase.origin_seconds = d.phase.origin_seconds.min(d.phase.seconds);
    o
}
fn execute(
    observations: &[Observation],
    original: &Options,
    processing: &Options,
    assessment: Assessment,
    deadline: Instant,
) -> Result<SearchRun> {
    let mut processing = processing.clone();
    processing.discovery.clone_from(&original.discovery);
    super::search_pass(
        observations,
        original,
        &processing,
        assessment,
        Some(deadline),
    )
}
#[allow(clippy::too_many_arguments)] // Move the selected/other pass and chronological trace together.
fn finish(
    mut chosen: SearchRun,
    other: Option<DiscoveryReport>,
    controls: &EffortOptions,
    selected: SearchEffort,
    passes: Vec<EffortPass>,
    reason: Option<String>,
    started: Instant,
) -> SearchRun {
    chosen.1.seconds = started.elapsed().as_secs_f64();
    let selection_reason = if chosen.0.is_none() {
        "no_model_available"
    } else if selected == SearchEffort::Fast && passes.len() > 1 {
        "retained_first_approximation_no_accepted_upgrade"
    } else if passes.iter().any(|p| p.mode == selected && p.accepted) {
        "accepted_pass"
    } else {
        "only_available_approximation"
    };
    chosen.1.effort = Some(EffortTrace {
        requested: controls.mode,
        selected_pass: selected,
        accepted: passes
            .iter()
            .find(|p| p.mode == selected)
            .is_some_and(|p| p.accepted),
        passes,
        escalation_reason: reason,
        selection_reason,
        deferred_ambiguity: None,
        other_pass: other.map(Box::new),
    });
    chosen
}
#[allow(clippy::too_many_lines)] // Two bounded attempts with explicit failure retention and result selection.
pub(super) fn search(
    observations: &[Observation],
    original: &Options,
    processing: &Options,
    assessment: Assessment,
    controls: &EffortOptions,
) -> Result<SearchRun> {
    let started = Instant::now();
    let total = original
        .discovery
        .as_ref()
        .expect("validated discovery")
        .total_seconds;
    if controls.mode == SearchEffort::Thorough {
        let deep = execute(
            observations,
            &pass_options(original, controls, false, total),
            processing,
            assessment,
            started + Duration::from_secs_f64(total),
        )?;
        let pass = summary(&deep, SearchEffort::Thorough, controls.maximum_common_r1);
        return Ok(finish(
            deep,
            None,
            controls,
            SearchEffort::Thorough,
            vec![pass],
            None,
            started,
        ));
    }
    let fast = execute(
        observations,
        &pass_options(original, controls, true, total),
        processing,
        assessment.clone(),
        started + Duration::from_secs_f64(total.min(controls.fast.total_seconds)),
    )?;
    let fast_summary = summary(&fast, SearchEffort::Fast, controls.maximum_common_r1);
    let remaining = total - started.elapsed().as_secs_f64();
    if controls.mode == SearchEffort::Fast
        || fast_summary.accepted
        || original.cancellation.is_cancelled()
        || remaining <= 0.
    {
        return Ok(finish(
            fast,
            None,
            controls,
            SearchEffort::Fast,
            vec![fast_summary],
            None,
            started,
        ));
    }
    if let Some(equivalence) = deferred_ambiguity(
        &fast,
        controls.maximum_common_r1,
        original.priors.cell,
        started + Duration::from_secs_f64(total),
        &original.cancellation,
    ) {
        let mut result = finish(
            fast,
            None,
            controls,
            SearchEffort::Fast,
            vec![fast_summary],
            None,
            started,
        );
        let trace = result.1.effort.as_mut().expect("finished effort");
        trace.selection_reason = "deferred_equivalent_group_models";
        trace.deferred_ambiguity = Some(equivalence);
        return Ok(result);
    }
    let reason = fast_summary.reason.clone();
    let deep = match execute(
        observations,
        &pass_options(original, controls, false, remaining),
        processing,
        assessment,
        started + Duration::from_secs_f64(total),
    ) {
        Ok(deep) => deep,
        Err(error) => {
            let failed = EffortPass {
                mode: SearchEffort::Thorough,
                accepted: false,
                reason: error.to_string(),
                seconds: started.elapsed().as_secs_f64() - fast_summary.seconds,
                termination: "error".into(),
                phasing_shape: None,
                phase_cycles: Vec::new(),
                model_attempts: 0,
            };
            return Ok(finish(
                fast,
                None,
                controls,
                SearchEffort::Fast,
                vec![fast_summary, failed],
                Some(reason),
                started,
            ));
        }
    };
    let deep_summary = summary(&deep, SearchEffort::Thorough, controls.maximum_common_r1);
    let passes = vec![fast_summary, deep_summary.clone()];
    Ok(select_auto(
        fast,
        deep,
        controls,
        passes,
        Some(reason),
        started,
    ))
}
fn select_auto(
    fast: SearchRun,
    deep: SearchRun,
    controls: &EffortOptions,
    passes: Vec<EffortPass>,
    reason: Option<String>,
    started: Instant,
) -> SearchRun {
    if !passes.last().is_some_and(|p| p.accepted) && fast.0.is_some() {
        finish(
            fast,
            Some(deep.1),
            controls,
            SearchEffort::Fast,
            passes,
            reason,
            started,
        )
    } else {
        finish(
            deep,
            Some(fast.1),
            controls,
            SearchEffort::Thorough,
            passes,
            reason,
            started,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space_group_search::{self as sg, Hypothesis};
    use phasesmith_crystallography::UnitCell;
    fn h(number: i32) -> Hypothesis {
        let h = phasesmith_io::space_group_by_number(number).unwrap();
        Hypothesis {
            number,
            symbol: h.hm_symbol,
            hall: h.hall_symbol,
            group: h.space_group,
        }
    }
    fn run(number: i32) -> SearchRun {
        let cell = UnitCell {
            a_angstrom: 10.,
            b_angstrom: 10.,
            c_angstrom: 10.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let rows: Vec<_> = (1..=40)
            .map(|n| Observation {
                hkl: [2 * n, 2, 0],
                intensity: 100.,
                sigma: 2.,
            })
            .collect();
        let hypothesis = h(number);
        let assessment = sg::assess(
            &rows,
            cell,
            std::slice::from_ref(&hypothesis),
            &sg::AssessmentOptions::default(),
        )
        .unwrap();
        let prior = super::super::super::SymmetryPrior {
            laue_rotations: sg::laue_rotations(&hypothesis.group),
            centring_translations: hypothesis
                .group
                .operations()
                .iter()
                .filter(|o| o.rotation() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                .map(|o| o.translation())
                .collect(),
            provenance: "test intensity-symmetry prior".into(),
        };
        let candidate = phasesmith_solution::Candidate {
            species: vec!["C".into()],
            fractional_xyz: vec![[0.; 3]],
            u_iso: 0.01,
            amplitude_scale: 1.,
            r1: 0.01,
            objective: 0.001,
            trial: 0,
            polish_steps: 0,
            element_swaps: 0,
            refinement: None,
            refinement_diagnostic: None,
            completion: None,
        };
        let attempt = super::super::GroupAttempt {
            runs: vec![],
            phase_status: "consistent",
            origins: vec![phasesmith_solution::PhaseOriginCandidate {
                hand: 1,
                shift: [0.; 3],
                alpha: 0.1,
                chord: 0.1,
            }],
            phase_seconds: 0.,
            hypothesis: 0,
            status: "candidate",
            seconds: 0.,
            diagnostic: None,
            common_r1: Some(0.01),
            common_objective: Some(0.001),
            common_residuals: Some(vec![0.01; 40]),
            parameters: Some(4),
            bic: Some(-100.),
            candidate: Some(candidate.clone()),
        };
        let solution = phasesmith_solution::Solution {
            data: phasesmith_solution::prepare_data(&rows, cell, &hypothesis.group).unwrap(),
            grid_shape: [16; 3],
            candidates: vec![candidate],
            density: vec![],
            histories: vec![],
            phasing_trials: vec![],
            rejected_trials: vec![],
            termination: "r1_target",
            target_met: true,
            shared_search: None,
        };
        (
            Some((
                solution,
                phasesmith_solution::SharedSearchSummary::default(),
            )),
            DiscoveryReport {
                symmetry_prior: Some(prior),
                effort: None,
                phase_map: None,
                phase_diagnostic: None,
                hypotheses: vec![hypothesis],
                assessment,
                attempts: vec![attempt],
                selected: Some(0),
                alternatives: vec![0],
                termination: "catalogue_completed",
                seconds: 0.,
            },
        )
    }
    #[test]
    fn low_r1_cannot_override_supplied_or_unresolved_higher_symmetry_or_ambiguity() {
        let mut r = run(10);
        assert_eq!(acceptance(&r, 0.08), "accepted");
        let extra = h(139);
        r.1.hypotheses.push(extra);
        r.1.assessment = sg::assess(
            &r.1.assessment.observations,
            UnitCell {
                a_angstrom: 10.,
                b_angstrom: 10.,
                c_angstrom: 10.,
                alpha_deg: 90.,
                beta_deg: 90.,
                gamma_deg: 90.,
            },
            &r.1.hypotheses,
            &sg::AssessmentOptions::default(),
        )
        .unwrap();
        r.1.symmetry_prior = None;
        assert_eq!(acceptance(&r, 0.08), "unresolved_higher_laue");
        r.1.symmetry_prior = run(139).1.symmetry_prior;
        assert_eq!(acceptance(&r, 0.08), "supplied_symmetry_not_recovered");
        let mut r = run(139);
        r.1.hypotheses.push(h(141));
        r.1.alternatives.push(1);
        assert_eq!(acceptance(&r, 0.08), "ambiguous_space_group");
    }
    #[test]
    fn approximate_upgrade_requires_acceptance_and_keeps_both_passes() {
        let mut fast = run(139);
        let mut deep = run(141);
        fast.0.as_mut().unwrap().0.target_met = false;
        deep.0.as_mut().unwrap().0.target_met = false;
        deep.1.attempts[0].bic = Some(-1000.); // Lower score alone cannot upgrade.
        let controls = EffortOptions {
            mode: SearchEffort::Auto,
            maximum_common_r1: 0.08,
            fast: FastSearchLimits {
                starts: 4,
                cycles: 256,
                minimum_d_angstrom: 0.8,
                maximum_candidates: 4,
                candidate_seconds: 2.,
                total_seconds: 8.,
            },
        };
        let passes = vec![
            summary(&fast, SearchEffort::Fast, 0.08),
            summary(&deep, SearchEffort::Thorough, 0.08),
        ];
        let r = select_auto(
            fast.clone(),
            deep.clone(),
            &controls,
            passes,
            Some("own_target_not_met".into()),
            Instant::now(),
        );
        assert_eq!(r.1.hypotheses[0].number, 139);
        let trace = r.1.effort.unwrap();
        assert!(!trace.accepted);
        assert_eq!(
            trace.selection_reason,
            "retained_first_approximation_no_accepted_upgrade"
        );
        assert_eq!(trace.other_pass.unwrap().hypotheses[0].number, 141);
        deep.0.as_mut().unwrap().0.target_met = true;
        let passes = vec![
            summary(&fast, SearchEffort::Fast, 0.08),
            summary(&deep, SearchEffort::Thorough, 0.08),
        ];
        let r = select_auto(
            fast,
            deep,
            &controls,
            passes,
            Some("own_target_not_met".into()),
            Instant::now(),
        );
        assert_eq!(r.1.hypotheses[0].number, 141);
        assert!(r.1.effort.unwrap().accepted);
    }
    #[test]
    fn acentric_low_r1_does_not_resolve_a_consistent_inversion_phase_test() {
        let mut r = run(107);
        assert_eq!(acceptance(&r, 0.08), "centrosymmetry_not_assessed");
        r.1.hypotheses.push(h(2));
        let mut baseline = r.1.attempts[0].clone();
        baseline.status = "prior_excluded"; // Diagnostic inversion test is not an eligible model.
        baseline.candidate = None;
        r.1.attempts.push(baseline);
        assert_eq!(acceptance(&r, 0.08), "unresolved_centrosymmetry");
        r.1.attempts[1].origins[0].alpha = 0.8;
        assert_eq!(acceptance(&r, 0.08), "accepted");
    }
    fn ambiguous_run() -> SearchRun {
        let mut r = run(139);
        r.1.hypotheses.push(h(121));
        let mut alternative = r.1.attempts[0].clone();
        alternative.hypothesis = 1;
        r.1.attempts.push(alternative);
        r.1.alternatives.push(1);
        for a in &mut r.1.attempts {
            let c = a.candidate.as_mut().unwrap();
            c.species.push("O".into());
            c.fractional_xyz.push([0.1, 0.2, 0.3]);
        }
        r
    }
    fn deferral(r: &SearchRun) -> Option<DeferredAmbiguity> {
        deferred_ambiguity(
            r,
            0.08,
            UnitCell {
                a_angstrom: 10.,
                b_angstrom: 10.,
                c_angstrom: 10.,
                alpha_deg: 90.,
                beta_deg: 90.,
                gamma_deg: 90.,
            },
            Instant::now() + Duration::from_secs(5),
            &phasesmith_solution::Cancellation::default(),
        )
    }
    #[test]
    fn equivalent_group_deferral_preserves_rejection_and_requires_same_full_data_model() {
        let r = ambiguous_run();
        assert_eq!(acceptance(&r, 0.08), "ambiguous_space_group");
        assert!(deferral(&r).is_some());
        assert_eq!(acceptance(&r, 0.08), "ambiguous_space_group");
        let mut bad = r.clone();
        bad.1.hypotheses.push(h(119));
        let mut third = bad.1.attempts[1].clone();
        third.hypothesis = 2;
        third.common_residuals.as_mut().unwrap()[0] += 1e-10;
        bad.1.attempts.push(third);
        bad.1.alternatives.push(2);
        assert!(deferral(&bad).is_none()); // A single twin cannot hide a distinct alternative.
        let mut bad = r.clone();
        bad.1.attempts[1].common_residuals.as_mut().unwrap()[0] += 1e-10;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[1].candidate.as_mut().unwrap().fractional_xyz[1][0] += 1e-6;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[1].candidate.as_mut().unwrap().species[1] = "Si".into();
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[1].parameters = Some(5);
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[1].common_residuals = None;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[1].common_residuals.as_mut().unwrap()[0] = f64::NAN;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[0].common_r1 = Some(0.09);
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.attempts[0].origins[0].alpha = 0.8;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.0.as_mut().unwrap().0.target_met = false;
        assert!(deferral(&bad).is_none());
        let mut bad = r.clone();
        bad.1.hypotheses[0] = h(107);
        bad.1.symmetry_prior = run(107).1.symmetry_prior;
        assert!(deferral(&bad).is_none()); // Inversion baseline is still required.
        let mut bad = r;
        bad.1.attempts[0].parameters = Some(40);
        assert!(deferral(&bad).is_none());
    }
    #[test]
    fn effort_admission_rejects_invalid_acceptance_and_resource_limits() {
        let valid = EffortOptions {
            mode: SearchEffort::Auto,
            maximum_common_r1: 0.08,
            fast: FastSearchLimits {
                starts: 4,
                cycles: 256,
                minimum_d_angstrom: 0.8,
                maximum_candidates: 4,
                candidate_seconds: 2.,
                total_seconds: 8.,
            },
        };
        valid.check(180.).unwrap();
        for value in [f64::NAN, 0., -1., 1.01] {
            let mut p = valid.clone();
            p.maximum_common_r1 = value;
            assert!(p.check(180.).is_err());
        }
        let mut p = valid.clone();
        p.fast.cycles = 31;
        assert!(p.check(180.).is_err());
        let mut p = valid.clone();
        p.fast.starts = 17;
        assert!(p.check(180.).is_err());
        let mut p = valid.clone();
        p.fast.maximum_candidates = 0;
        assert!(p.check(180.).is_err());
        let mut p = valid.clone();
        p.fast.candidate_seconds = 9.;
        assert!(p.check(180.).is_err());
        let mut p = valid.clone();
        p.fast.minimum_d_angstrom = f64::INFINITY;
        assert!(p.check(180.).is_err());
        assert!(valid.check(4.).is_err());
    }
}
