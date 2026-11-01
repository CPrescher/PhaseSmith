//! Explicit bounded fast-first effort; escalation depends only on saved numerical candidates.
use super::{Assessment, DiscoveryReport, Error, Observation, Options, Result, SearchRun};
use std::time::{Duration, Instant};
/// Requested search effort; automatic effort shares one total deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchEffort {
    /// Coarse shared-map attempt only, including unsuccessful results.
    Fast,
    /// Try fast, then use the configured full-resolution search if acceptance fails.
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
    if a.common_r1.is_none_or(|r| !r.is_finite() || r > limit) {
        return "common_r1_not_met";
    }
    if !a.origins.iter().any(|o| o.alpha <= 0.3) {
        return "phase_consistency_not_met";
    }
    "accepted"
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
fn score(run: &SearchRun) -> f64 {
    run.1
        .selected
        .and_then(|i| run.1.attempts[i].bic)
        .unwrap_or(f64::INFINITY)
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
    chosen.1.effort = Some(EffortTrace {
        requested: controls.mode,
        selected_pass: selected,
        accepted: passes
            .iter()
            .find(|p| p.mode == selected)
            .is_some_and(|p| p.accepted),
        passes,
        escalation_reason: reason,
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
    if !deep_summary.accepted
        && fast.0.is_some()
        && (deep.0.is_none() || score(&fast) < score(&deep))
    {
        Ok(finish(
            fast,
            Some(deep.1),
            controls,
            SearchEffort::Fast,
            passes,
            Some(reason),
            started,
        ))
    } else {
        Ok(finish(
            deep,
            Some(fast.1),
            controls,
            SearchEffort::Thorough,
            passes,
            Some(reason),
            started,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
