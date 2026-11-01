//! Native measured-only unknown-symmetry coordination; no reference/count inputs.
use super::{Error, Measurements, Options, Reduced, Result, scaling};
use crate::space_group_search::{self as sg, Assessment, AssessmentOptions, Hypothesis};
use phasesmith_crystallography::UnitCell;
use phasesmith_solution::{Candidate, Observation, SharedSearchSummary, Solution};
use std::time::{Duration, Instant};
#[path = "unknown_effort.rs"]
mod effort;
pub use effort::{
    DeferredAmbiguity, EffortOptions, EffortPass, EffortTrace, FastSearchLimits, SearchEffort,
};
/// Shared P1 phase-map and bounded origin-screen controls.
#[derive(Clone, Debug)]
pub struct PhaseScreenOptions {
    /// Explicit optional coarse/adaptive initial phasing policy; default is full-data phasing.
    pub phasing: phasesmith_solution::PhaseOnlyPolicy,
    /// Bounded independent P1 workers, 1..=10.
    pub workers: usize,
    /// Independent phase starts, 1..=16.
    pub starts: usize,
    /// Maximum cycles per start, 1..=20000.
    pub cycles: usize,
    /// Maximum distinct settings phase-screened, 1..=530.
    pub maximum_settings: usize,
    /// Total phase-map/origin screen wall limit.
    pub seconds: f64,
    /// Per-setting origin-search limit.
    pub origin_seconds: f64,
    /// Maximum non-equivalent origins retained per setting, 1..=8.
    pub maximum_origins: usize,
    /// Reasonable phase alpha ceiling; random expectation one.
    pub maximum_alpha: f64,
}
/// Explicit catalogue and bounded discovery controls; exact symmetry is withheld.
#[derive(Clone, Debug)]
pub struct DiscoveryOptions {
    /// Optional supplied Laue/centring prior in the original index basis; no exact group.
    pub symmetry_prior: Option<super::SymmetryPrior>,
    /// Optional explicit fast/auto/thorough policy; absent preserves frozen legacy execution.
    pub effort: Option<EffortOptions>,
    /// Allow independent phasing after unsuccessful shared-map assignment.
    pub allow_independent_restarts: bool,
    /// Published light-element heuristic: stop after an accepted plausible centric model.
    /// Unsearched alternatives remain explicit and uniqueness is never inferred.
    pub plausible_centrosymmetric_exit: bool,
    /// Feed the shared P1 map directly into model assignment, with bounded fallback.
    pub reuse_phase_map: bool,
    /// Published phase-equivalence screening before full model searches.
    pub phase: PhaseScreenOptions,
    /// All settings admitted by the host's stated catalogue/index-basis policy.
    pub hypotheses: Vec<Hypothesis>,
    /// Measured-intensity prescreen controls.
    pub assessment: AssessmentOptions,
    /// Maximum complete element-search attempts, in 1..=64.
    pub maximum_candidates: usize,
    /// Per-candidate seconds, including unsuccessful search work.
    pub candidate_seconds: f64,
    /// Total shared search seconds, including phase/symmetry work.
    pub total_seconds: f64,
    /// Retained common-set BIC-like alternatives within this difference.
    pub ambiguity_delta: f64,
}
impl DiscoveryOptions {
    pub(super) fn check(&self) -> Result<()> {
        if let Some(prior) = &self.symmetry_prior {
            prior.processing_group()?;
        }
        if let Some(effort) = &self.effort {
            effort.check(self.total_seconds)?;
        }
        let p = &self.phase;
        if p.phasing
            .first_pass_cycles
            .is_some_and(|n| !p.phasing.adaptive_quality || n < 32 || n > p.cycles)
            || p.phasing
                .minimum_d_angstrom
                .is_some_and(|d| !d.is_finite() || d <= 0.)
            || !(1..=10).contains(&p.workers)
            || !(1..=16).contains(&p.starts)
            || !(1..=20000).contains(&p.cycles)
            || !(1..=530).contains(&p.maximum_settings)
            || !(1..=8).contains(&p.maximum_origins)
            || [p.seconds, p.origin_seconds, p.maximum_alpha]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
            || p.seconds > self.total_seconds
            || p.origin_seconds > p.seconds
            || p.maximum_alpha > 1.
        {
            return Err(Error("invalid phase screening controls".into()));
        }
        if self.hypotheses.is_empty()
            || self.hypotheses.len() > 530
            || !(1..=64).contains(&self.maximum_candidates)
            || [self.candidate_seconds, self.total_seconds]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0. || *v > 3600.)
            || self.candidate_seconds > self.total_seconds
            || !self.ambiguity_delta.is_finite()
            || self.ambiguity_delta < 0.
        {
            return Err(Error(
                "invalid space-group discovery catalogue/budgets".into(),
            ));
        }
        Ok(())
    }
}
/// Actual model-search work from either a reused phase map or an independent restart.
#[derive(Clone, Debug)]
pub struct GroupRun {
    /// `shared_p1_map` or `independent_phasing`.
    pub initialization: &'static str,
    /// Actual seconds including unsuccessful numerical work.
    pub seconds: f64,
    /// Numerical termination or error text.
    pub diagnostic: String,
    /// Number of returned complete model candidates before common-set evaluation.
    pub candidates: usize,
    /// Actual new phase starts; zero when reusing the shared map.
    pub phasing_starts: usize,
    /// Whether this run met its own group-filtered numerical R1 target.
    pub target_met: bool,
}
/// Every catalogue setting retains its actual search status; budgets never imply rejection.
#[derive(Clone, Debug)]
pub struct GroupAttempt {
    /// Every actual model-search initialization, including failed seeded/fallback work.
    pub runs: Vec<GroupRun>,
    /// Shared-map phase screening status, distinct from model-search success.
    pub phase_status: &'static str,
    /// Retained origin/hand alternatives and alpha/chord values.
    pub origins: Vec<phasesmith_solution::PhaseOriginCandidate>,
    /// Actual origin-screen wall seconds, including failures.
    pub phase_seconds: f64,
    /// Index in the declared setting catalogue.
    pub hypothesis: usize,
    /// `not_attempted`, `prior_excluded`, `laue_contradicted`, `candidate`, `no_model`, or `error`.
    pub status: &'static str,
    /// Wall seconds actually spent in this group's element search and assessment.
    pub seconds: f64,
    /// Error/termination diagnostic; no failed run is silently discarded.
    pub diagnostic: Option<String>,
    /// Comparable residual on every selected measured Friedel family, including forbidden rows.
    pub common_r1: Option<f64>,
    /// Comparable normalized amplitude objective on that common set.
    pub common_objective: Option<f64>,
    /// Common normalized residuals, retained for exact-data model-equivalence diagnostics.
    pub common_residuals: Option<Vec<f64>>,
    /// Model parameter count including amplitude scale.
    pub parameters: Option<usize>,
    /// BIC-like common-set model score; lower is preferred, not a probability.
    pub bic: Option<f64>,
    /// Complete retained candidate, including its density-frame/source transformation.
    pub candidate: Option<Candidate>,
}
/// Rank and bounded coverage, explicitly distinct from a proven space-group determination.
#[derive(Clone, Debug)]
pub struct DiscoveryReport {
    /// Explicit supplied intensity symmetry and translation lattice, if any.
    pub symmetry_prior: Option<super::SymmetryPrior>,
    /// Optional chronological effort and complete nonselected pass.
    pub effort: Option<EffortTrace>,
    /// Shared exact-group-free P1 map, phases and cycle histories.
    pub phase_map: Option<phasesmith_solution::PhaseOnlyResult>,
    /// Phase-map failure/limit diagnostic, retained independently of candidate searches.
    pub phase_diagnostic: Option<String>,
    /// Full setting metadata and exact operations, in stable input order.
    pub hypotheses: Vec<Hypothesis>,
    /// Measured-only intensity and absence evidence after scaling.
    pub assessment: Assessment,
    /// All settings including failures and untested hypotheses.
    pub attempts: Vec<GroupAttempt>,
    /// Index of the best common-set model, if one was found.
    pub selected: Option<usize>,
    /// Successful alternatives within the caller's BIC-like difference.
    pub alternatives: Vec<usize>,
    /// `plausible_centrosymmetric`, `candidate_budget`, `max_runtime`, `cancelled`, or `catalogue_completed`.
    pub termination: &'static str,
    /// Complete wall time including failed work.
    pub seconds: f64,
}
#[allow(clippy::cast_precision_loss)]
pub(super) fn aperture_assessment(
    m: &Measurements,
    geometry: &phasesmith_spots::reciprocal::Geometry,
    reduced: Option<&Reduced>,
    policy: &DiscoveryOptions,
    cell: UnitCell,
) -> Result<Assessment> {
    let physical = scaling::physical_factors(m, geometry)?;
    let observations: Vec<_> = m
        .hkl
        .iter()
        .zip(&m.aperture)
        .enumerate()
        .filter_map(|(i, (&h, a))| {
            // Coverage/contamination cuts only. There is no positive-intensity or SNR cut.
            if m.flags[i] != 0
                || !m.visible[i]
                || !a[0].is_finite()
                || !a[1].is_finite()
                || a[1] <= 0.
            {
                return None;
            }
            let factor = physical[i] * reduced.map_or(1., |r| r.empirical[i]);
            let intensity = a[0] * factor;
            let sigma =
                (a[1] * factor).hypot(reduced.map_or(0., |r| r.error_floor * intensity.max(0.)));
            Some(Observation {
                hkl: h,
                intensity,
                sigma,
            })
        })
        .collect();
    let mut assessment = sg::assess(&observations, cell, &policy.hypotheses, &policy.assessment)?;
    super::symmetry_prior::apply(&mut assessment, policy)?;
    Ok(assessment)
}
fn common_score(
    candidate: &Candidate,
    group: &phasesmith_crystallography::SpaceGroup,
    cell: UnitCell,
    data: &phasesmith_solution::PreparedData,
    memory: usize,
) -> Result<(f64, f64, usize, f64, Vec<f64>)> {
    use phasesmith_solution::{
        AdpMode, DenseMemoryBudget, PreparedSymmetricModel, recover_symmetry,
    };
    let model = recover_symmetry(
        cell,
        group,
        &candidate.species,
        &candidate.fractional_xyz,
        0.15,
        None,
        None,
    )?;
    let site_u: Vec<_> = model
        .sites
        .iter()
        .map(|site| {
            candidate.refinement.as_ref().map_or(candidate.u_iso, |r| {
                site.images
                    .iter()
                    .map(|image| r.full_u_equiv[image.source_index])
                    .sum::<f64>()
                    / site.images.len() as f64
            })
        })
        .collect();
    let p = PreparedSymmetricModel::new_with_memory_budget(
        model,
        &data.hkl,
        &data.amplitudes,
        AdpMode::Isotropic,
        true,
        false,
        DenseMemoryBudget::new(memory)?,
    )?;
    let evaluated = p.evaluate(&p.initial_site_parameters(&site_u)?)?;
    let parameters = p.parameter_count + 1;
    let n = data.family_count as f64;
    let bic = n * evaluated.objective.max(1e-300).ln() + parameters as f64 * n.ln();
    Ok((
        evaluated.r1,
        evaluated.objective,
        parameters,
        bic,
        evaluated.residual,
    ))
}
/// Complete bounded discovery outcome, retaining evidence even when no model is found.
pub type SearchRun = (Option<(Solution, SharedSearchSummary)>, DiscoveryReport);
pub(super) fn search(
    observations: &[Observation],
    original: &Options,
    processing: &Options,
    assessment: Assessment,
) -> Result<SearchRun> {
    if let Some(effort) = original.discovery.as_ref().and_then(|d| d.effort.as_ref()) {
        original.discovery.as_ref().expect("discovery").check()?;
        effort::search(observations, original, processing, assessment, effort)
    } else {
        search_pass(observations, original, processing, assessment, None)
    }
}
#[allow(clippy::too_many_lines)] // Ordered bounded state machine; numerical stages have separate owners.
pub(super) fn search_pass(
    observations: &[Observation],
    original: &Options,
    processing: &Options,
    assessment: Assessment,
    shared_deadline: Option<Instant>,
) -> Result<SearchRun> {
    let policy = original
        .discovery
        .as_ref()
        .ok_or_else(|| Error("discovery policy missing".into()))?;
    policy.check()?;
    let started = Instant::now();
    let own_deadline = started + Duration::from_secs_f64(policy.total_seconds);
    let deadline = shared_deadline.map_or(own_deadline, |d| d.min(own_deadline));
    let p1 = sg::rotational_group(&[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])?;
    let common = phasesmith_solution::prepare_data(observations, original.priors.cell, &p1)?;
    let mut attempts = initial_attempts(&assessment, policy);
    let (phase_map, phase_diagnostic) = screen_phases(
        observations,
        original,
        processing,
        &assessment,
        &mut attempts,
        deadline,
        shared_deadline.is_some() || policy.symmetry_prior.is_some(),
    )?;
    let mut ranking = assessment.ranking.clone();
    let priority = |s: &str| match s {
        "consistent" => 0,
        "uninformative" => 1,
        "not_attempted" | "budget_exhausted" => 2,
        _ => 3,
    };
    if !policy.plausible_centrosymmetric_exit {
        ranking.sort_by_key(|&i| priority(attempts[i].phase_status)); // stable within measured evidence order
    }
    let coefficients = if policy.plausible_centrosymmetric_exit {
        phase_map
            .as_ref()
            .map(|m| phasesmith_solution::PhaseOriginDensity::new(&m.density, m.shape))
            .transpose()?
    } else {
        None
    };
    let phase_deadline = (started + Duration::from_secs_f64(policy.phase.seconds)).min(deadline);
    let mut best: Option<(Solution, SharedSearchSummary)> = None;
    let mut best_score = f64::INFINITY;
    let mut selected = None;
    let mut count = 0;
    let mut termination = "catalogue_completed";
    for &index in &ranking {
        if original.cancellation.is_cancelled() {
            termination = "cancelled";
            break;
        }
        if Instant::now() >= deadline {
            termination = "max_runtime";
            break;
        }
        if count >= policy.maximum_candidates {
            termination = "candidate_budget";
            break;
        }
        if matches!(
            attempts[index].status,
            "laue_contradicted" | "prior_excluded"
        ) {
            continue;
        }
        if let Some(coefficients) = &coefficients {
            if attempts[index].phase_status == "not_attempted"
                && attempts
                    .iter()
                    .filter(|a| a.phase_status != "not_attempted")
                    .count()
                    < policy.phase.maximum_settings
                && Instant::now() < phase_deadline
            {
                screen_setting(
                    observations,
                    original,
                    &mut attempts[index],
                    &policy.hypotheses[index],
                    coefficients,
                    phase_deadline,
                );
            }
            if matches!(attempts[index].phase_status, "inconsistent" | "error") {
                continue;
            }
        }
        count += 1;
        let clock = Instant::now();
        let hypothesis = &policy.hypotheses[index];
        let mut options = processing.clone();
        options.priors.group = Some(hypothesis.group.clone());
        options.search_seconds = policy.candidate_seconds.min(
            deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64(),
        );
        eprintln!(
            "space-group search {count}/{}: {} [{}]",
            policy.maximum_candidates, hypothesis.symbol, hypothesis.hall
        );
        let (attempt, result) = attempt_model(
            observations,
            &options,
            &common,
            hypothesis,
            &attempts[index],
            clock,
            if policy.reuse_phase_map {
                phase_map.as_ref()
            } else {
                None
            },
        );
        attempts[index] = attempt;
        if let (Some(score), Some(result)) = (attempts[index].bic, result)
            && score < best_score
        {
            best_score = score;
            selected = Some(index);
            best = Some(result);
        }
        if policy.plausible_centrosymmetric_exit
            && selected == Some(index)
            && best.as_ref().is_some_and(|(s, _)| s.target_met)
            && plausible_centric(
                &original.priors.elements,
                &policy.hypotheses,
                &attempts,
                index,
                policy.phase.maximum_alpha,
            )
        {
            termination = "plausible_centrosymmetric";
            break;
        }
    }
    let alternatives = ranked_alternatives(&attempts, best_score, policy.ambiguity_delta);
    Ok((
        best,
        DiscoveryReport {
            symmetry_prior: policy.symmetry_prior.clone(),
            effort: None,
            phase_map,
            phase_diagnostic,
            hypotheses: policy.hypotheses.clone(),
            assessment,
            attempts,
            selected,
            alternatives,
            termination,
            seconds: started.elapsed().as_secs_f64(),
        },
    ))
}

fn is_centric(group: &phasesmith_crystallography::SpaceGroup) -> bool {
    group
        .operations()
        .iter()
        .any(|op| op.rotation() == [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
}
fn plausible_centric(
    elements: &[String],
    hypotheses: &[Hypothesis],
    attempts: &[GroupAttempt],
    index: usize,
    maximum_alpha: f64,
) -> bool {
    let alpha = maximum_alpha.min(0.3);
    if elements.is_empty()
        || elements.iter().any(|e| {
            phasesmith_crystallography::xray_species_metadata(e)
                .is_none_or(|m| m.atomic_number > 21)
        })
        || !is_centric(&hypotheses[index].group)
        || attempts[index].status != "candidate"
        || !attempts[index].origins.iter().any(|o| o.alpha < alpha)
    {
        return false;
    }
    hypotheses.iter().zip(attempts).any(|(h, a)| {
        h.group.operations().len() == 2
            && is_centric(&h.group)
            && h.group
                .operations()
                .iter()
                .all(|op| op.translation().iter().all(|v| v.numerator() == 0))
            && a.origins.iter().any(|o| o.alpha < alpha)
    })
}

fn screen_phases(
    observations: &[Observation],
    original: &Options,
    processing: &Options,
    assessment: &Assessment,
    attempts: &mut [GroupAttempt],
    deadline: Instant,
    explicit_inversion_test: bool,
) -> Result<(Option<phasesmith_solution::PhaseOnlyResult>, Option<String>)> {
    let policy = original
        .discovery
        .as_ref()
        .ok_or_else(|| Error("discovery policy missing".into()))?;
    let phase_deadline =
        (Instant::now() + Duration::from_secs_f64(policy.phase.seconds)).min(deadline);
    let phase_options = phasesmith_solution::SolveOptions {
        seed: original.seed,
        trials: policy.phase.starts,
        cycles: policy.phase.cycles,
        cancellation: Some(original.cancellation.clone()),
        max_seconds: phase_deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
            .max(1e-6),
        ..phasesmith_solution::SolveOptions::default()
    };
    let laue = processing
        .priors
        .group
        .as_ref()
        .ok_or_else(|| Error("provisional processing symmetry missing".into()))?;
    let (phase_map, phase_diagnostic) = match phasesmith_solution::phase_only_with_policy(
        observations,
        original.priors.cell,
        laue,
        &original.priors.elements,
        &phase_options,
        policy.phase.workers,
        &policy.phase.phasing,
    ) {
        Ok(map) => (Some(map), None),
        Err(e) => (None, Some(e.to_string())),
    };
    if let Some(map) = &phase_map {
        let coefficients = phasesmith_solution::PhaseOriginDensity::new(&map.density, map.shape)?;
        // Atomic inversion is not supplied by a Laue prior. Test it explicitly,
        // even when P-1 is ineligible as a model or ranks outside the bounded list.
        let baseline = if explicit_inversion_test {
            policy
                .hypotheses
                .iter()
                .position(|h| inversion_only(&h.group))
        } else {
            None
        };
        if let Some(index) = baseline
            && !original.cancellation.is_cancelled()
            && Instant::now() < phase_deadline
        {
            screen_setting(
                observations,
                original,
                &mut attempts[index],
                &policy.hypotheses[index],
                &coefficients,
                phase_deadline,
            );
        }
        for &index in assessment
            .ranking
            .iter()
            .filter(|&&i| {
                Some(i) != baseline
                    && attempts[i].status != "laue_contradicted"
                    && (attempts[i].status != "prior_excluded"
                        || inversion_only(&policy.hypotheses[i].group))
            })
            .take(
                policy
                    .phase
                    .maximum_settings
                    .saturating_sub(usize::from(baseline.is_some())),
            )
            .collect::<Vec<_>>()
        {
            if original.cancellation.is_cancelled() || Instant::now() >= phase_deadline {
                break;
            }
            if policy.plausible_centrosymmetric_exit
                && !inversion_only(&policy.hypotheses[index].group)
            {
                continue;
            }
            screen_setting(
                observations,
                original,
                &mut attempts[index],
                &policy.hypotheses[index],
                &coefficients,
                phase_deadline,
            );
        }
    }
    Ok((phase_map, phase_diagnostic))
}

fn inversion_only(group: &phasesmith_crystallography::SpaceGroup) -> bool {
    group.operations().len() == 2
        && is_centric(group)
        && group
            .operations()
            .iter()
            .all(|op| op.translation().iter().all(|v| v.numerator() == 0))
}
fn screen_setting(
    observations: &[Observation],
    original: &Options,
    a: &mut GroupAttempt,
    h: &Hypothesis,
    coefficients: &phasesmith_solution::PhaseOriginDensity,
    deadline: Instant,
) {
    let clock = Instant::now();
    let policy = &original
        .discovery
        .as_ref()
        .expect("checked discovery")
        .phase;
    let result = (|| -> Result<Vec<phasesmith_solution::PhaseOriginCandidate>> {
        let data = phasesmith_solution::prepare_data(observations, original.priors.cell, &h.group)?;
        let plan =
            phasesmith_solution::PreparedPhaseOrigin::new(&h.group, &data.hkl, &data.amplitudes)?;
        Ok(plan.search_cached(
            coefficients,
            policy.maximum_origins,
            Some((Instant::now() + Duration::from_secs_f64(policy.origin_seconds)).min(deadline)),
            Some(&original.cancellation),
            true,
        )?)
    })();
    match result {
        Ok(origins) => {
            a.phase_status = if h.group.rotations().len() == 1 {
                "uninformative"
            } else if origins.is_empty() {
                "budget_exhausted"
            } else if origins.iter().any(|o| o.alpha <= policy.maximum_alpha) {
                "consistent"
            } else {
                "inconsistent"
            };
            a.origins = origins;
        }
        Err(e) => {
            a.phase_status = "error";
            a.diagnostic = Some(e.to_string());
        }
    }
    a.phase_seconds = clock.elapsed().as_secs_f64();
}

fn attempt_model(
    observations: &[Observation],
    options: &Options,
    common: &phasesmith_solution::PreparedData,
    hypothesis: &Hypothesis,
    prior: &GroupAttempt,
    clock: Instant,
    map: Option<&phasesmith_solution::PhaseOnlyResult>,
) -> (GroupAttempt, Option<(Solution, SharedSearchSummary)>) {
    let mut a = prior.clone();
    let mut best = None;
    let end = clock + Duration::from_secs_f64(options.search_seconds);
    let started = Instant::now();
    let result = if let Some(map) = map {
        super::search_observations_seeded(observations, options, map)
    } else {
        super::search_observations(observations, options)
    };
    let target_met = result.as_ref().is_ok_and(|(s, _)| s.target_met);
    record_run(
        &mut a,
        &mut best,
        result,
        if map.is_some() {
            "shared_p1_map"
        } else {
            "independent_phasing"
        },
        started.elapsed().as_secs_f64(),
        hypothesis,
        (options, common),
    );
    let remaining = end.saturating_duration_since(Instant::now()).as_secs_f64();
    if map.is_some()
        && !target_met
        && remaining > 0.01
        && !options.cancellation.is_cancelled()
        && options
            .discovery
            .as_ref()
            .is_none_or(|d| d.allow_independent_restarts)
    {
        let mut fallback = options.clone();
        fallback.search_seconds = remaining;
        let started = Instant::now();
        let result = super::search_observations(observations, &fallback);
        record_run(
            &mut a,
            &mut best,
            result,
            "independent_phasing",
            started.elapsed().as_secs_f64(),
            hypothesis,
            (options, common),
        );
    }
    a.seconds = clock.elapsed().as_secs_f64();
    (a, best)
}
fn record_run(
    a: &mut GroupAttempt,
    best: &mut Option<(Solution, SharedSearchSummary)>,
    result: Result<(Solution, SharedSearchSummary)>,
    initialization: &'static str,
    seconds: f64,
    hypothesis: &Hypothesis,
    context: (&Options, &phasesmith_solution::PreparedData),
) {
    let (options, common) = context;
    let mut run = GroupRun {
        initialization,
        seconds,
        diagnostic: String::new(),
        candidates: 0,
        phasing_starts: 0,
        target_met: false,
    };
    match result {
        Ok((solution, audit)) => {
            run.diagnostic = solution.termination.into();
            run.candidates = solution.candidates.len();
            run.phasing_starts = audit.phasing_starts;
            run.target_met = solution.target_met;
            if let Some(candidate) = solution.candidates.first() {
                match common_score(
                    candidate,
                    &hypothesis.group,
                    options.priors.cell,
                    common,
                    options.dense_memory_bytes,
                ) {
                    Ok((r1, objective, parameters, bic, residuals)) => {
                        if a.bic.is_none_or(|old| bic < old) {
                            a.status = "candidate";
                            a.diagnostic = Some(solution.termination.into());
                            a.common_r1 = Some(r1);
                            a.common_objective = Some(objective);
                            a.common_residuals = Some(residuals);
                            a.parameters = Some(parameters);
                            a.bic = Some(bic);
                            a.candidate = Some(candidate.clone());
                            *best = Some((solution, audit));
                        }
                    }
                    Err(e) => {
                        run.diagnostic = format!("common-set model rejection: {e}");
                        if a.candidate.is_none() {
                            a.status = "error";
                            a.diagnostic = Some(e.to_string());
                        }
                    }
                }
            } else if a.candidate.is_none() {
                a.status = "no_model";
                a.diagnostic = Some(solution.termination.into());
            }
        }
        Err(e) => {
            run.diagnostic = e.to_string();
            if a.candidate.is_none() {
                a.status = "error";
                a.diagnostic = Some(e.to_string());
            }
        }
    }
    a.runs.push(run);
}

fn initial_attempts(assessment: &Assessment, policy: &DiscoveryOptions) -> Vec<GroupAttempt> {
    let attempts: Vec<_> = assessment
        .evidence
        .iter()
        .enumerate()
        .map(|(i, e)| GroupAttempt {
            hypothesis: i,
            runs: Vec::new(),
            phase_status: "not_attempted",
            origins: vec![],
            phase_seconds: 0.,
            status: if policy
                .symmetry_prior
                .as_ref()
                .is_some_and(|p| !p.admits(&policy.hypotheses[i]))
            {
                "prior_excluded"
            } else if assessment.laue[e.laue].status == "contradicted" {
                "laue_contradicted"
            } else {
                "not_attempted"
            },
            seconds: 0.,
            diagnostic: None,
            common_r1: None,
            common_objective: None,
            common_residuals: None,
            parameters: None,
            bic: None,
            candidate: None,
        })
        .collect();
    attempts
}

fn ranked_alternatives(attempts: &[GroupAttempt], best_score: f64, delta: f64) -> Vec<usize> {
    let mut alternatives: Vec<_> = attempts
        .iter()
        .enumerate()
        .filter(|(_, a)| a.bic.is_some_and(|b| b - best_score <= delta))
        .map(|(i, _)| i)
        .collect();
    alternatives.sort_by(|&a, &b| {
        attempts[a]
            .bic
            .unwrap()
            .total_cmp(&attempts[b].bic.unwrap())
            .then(a.cmp(&b))
    });
    alternatives
}

#[cfg(test)]
mod fast_exit_tests {
    use super::*;
    fn h(number: i32) -> Hypothesis {
        let h = phasesmith_io::space_group_by_number(number).unwrap();
        Hypothesis {
            number,
            symbol: h.hm_symbol,
            hall: h.hall_symbol,
            group: h.space_group,
        }
    }
    fn a(index: usize) -> GroupAttempt {
        GroupAttempt {
            runs: Vec::new(),
            phase_status: "consistent",
            origins: vec![phasesmith_solution::PhaseOriginCandidate {
                hand: 1,
                shift: [0.; 3],
                alpha: 0.1,
                chord: 0.1,
            }],
            phase_seconds: 0.,
            hypothesis: index,
            status: "candidate",
            seconds: 0.,
            diagnostic: None,
            common_r1: Some(0.05),
            common_objective: Some(0.002),
            common_residuals: None,
            parameters: Some(4),
            bic: Some(-100.),
            candidate: None,
        }
    }
    #[test]
    fn centric_shortcut_rejects_heavy_elements_acentric_models_and_bad_or_missing_phases() {
        let hs = vec![h(2), h(61), h(19)];
        let mut attempts = vec![a(0), a(1), a(2)];
        let light = vec!["Mg".into(), "Si".into(), "O".into()];
        assert!(plausible_centric(&light, &hs, &attempts, 1, 0.3));
        assert!(!plausible_centric(&["Fe".into()], &hs, &attempts, 1, 0.3));
        assert!(!plausible_centric(&[], &hs, &attempts, 1, 0.3));
        assert!(!plausible_centric(
            &["unknown".into()],
            &hs,
            &attempts,
            1,
            0.3
        ));
        assert!(!plausible_centric(&light, &hs, &attempts, 2, 0.3));
        attempts[0].origins.clear();
        assert!(!plausible_centric(&light, &hs, &attempts, 1, 0.3));
        attempts[0] = a(0);
        attempts[0].origins[0].alpha = 0.3;
        assert!(!plausible_centric(&light, &hs, &attempts, 1, 0.3));
        attempts[0] = a(0);
        attempts[1].origins[0].alpha = 0.3;
        assert!(!plausible_centric(&light, &hs, &attempts, 1, 0.3));
        attempts[1] = a(1);
        attempts[1].status = "error";
        assert!(!plausible_centric(&light, &hs, &attempts, 1, 0.3));
    }
}
