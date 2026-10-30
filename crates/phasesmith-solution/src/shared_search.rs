//! One count-independent phase map and a bounded beam of editable symmetry orbits.
use crate::math3::{dot, mv, periodic_difference, sub};
use crate::{
    AdpMode, Candidate, FlipOptions, IntegratedPeakDensity, Observation, PhasingMethod,
    PhasingTrial, PreparedData, PreparedElementPool, PreparedSymmetricModel, Solution,
    SolutionError, SolveOptions, SymmetryModel, charge_flip, error, extract_peaks, prepare_data,
    project_fixed_sets, recover_symmetry, recover_symmetry_hypotheses,
};
use phasesmith_crystallography::{
    CellGeometry, PreparedXrayScattering, SpaceGroup, SymmetryOperation, UnitCell,
    xray_species_metadata,
};
use std::time::{Duration, Instant};
#[path = "shared_phase_path.rs"]
mod phase_path;
pub use phase_path::{PhaseOriginAudit, PhaseSettingAudit};

/// Explicit bounds for shared site/element editing, never a composition.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug)]
pub struct SharedSearchOptions {
    /// Maximum full-cell atoms, in [1,128].
    pub maximum_atoms: usize,
    /// Maximum live states and initial settings, in [1,8].
    pub beam_width: usize,
    /// Maximum edit rounds per phasing start, in [1,16].
    pub rounds: usize,
    /// Screen edits isotropically before requested anisotropic fits.
    pub staged_refinement: bool,
    /// Maximum isotropic screening iterations, in [1,40].
    pub screening_iterations: usize,
    /// Minimum number shortlisted per parent, in [1,16], plus ties/rescues.
    pub refinement_shortlist: usize,
    /// Guide element edits using relative integrated measured-amplitude density.
    pub integrated_peak_density: bool,
    /// Cartesian integration radius, in [1e-4,2] angstrom.
    pub peak_density_radius_angstrom: f64,
    /// Count-free atomicity map RMS threshold, in [0.1,10].
    pub dual_peak_sigma: f64,
    /// Count-free atomicity resource cap, in [1,128].
    pub dual_peak_cap: usize,
    /// Test window maps with complete shared refinement before the cycle ceiling.
    pub dual_checkpoints: bool,
    /// Cache fixed reflection/image geometry in constrained refinements.
    pub cached_refinement: bool,
    /// Recycle each supplied-symmetry setting before weak-peak assignment.
    pub symmetry_recycling: bool,
    /// Positivity/phase-projection cycles, in [1,50].
    pub recycling_cycles: usize,
    /// Add measured-only tangent phases, annealing and negative-quartet checks.
    pub tangent_refinement: bool,
    /// When both options are enabled, refine phases within supplied-symmetry recycling.
    pub tangent_in_recycling: bool,
    /// Explore nested orbit counts and bound early-map refinement effort.
    pub adaptive_counts: bool,
    /// Fit resolution-dependent intensity normalization without composition counts.
    pub wilson_normalization: bool,
    /// Rank Fourier maps by CC-Rweak before count assignment.
    pub phase_quality_selection: bool,
    /// Retain the earlier phase-first schedule instead of checkpoint alternatives.
    pub phase_first_selection: bool,
    /// Independently normalize calculated amplitudes in resolution shells for Rweak.
    pub calculated_shell_normalization: bool,
    /// Align origins from measured phases and screen complete peak-orbit models first.
    pub phase_origin_selection: bool,
    /// Hand off stable quality-ready maps before the phase cycle ceiling.
    pub adaptive_checkpoints: bool,
    /// Evaluate scalar phase merits without unused gradients.
    pub score_only_evaluation: bool,
    /// Reuse dual-space Fourier/map/mask buffers across cycles.
    pub reusable_fourier_workspaces: bool,
    /// Hand early phase starts to model validation at a deterministic half-work quota.
    pub work_sliced_phasing: bool,
    /// Select origin FFT sampling from phase-equivalence frequencies.
    pub independent_origin_sampling: bool,
    /// Add motif density scales and post-screen coupled element assignments.
    pub chemical_assignment: bool,
    /// Solve large damped refinement systems with an iterative normal operator.
    pub iterative_refinement: bool,
    /// Regularize allowed unmeasured coefficients using the observed Wilson envelope.
    pub missing_reflections: bool,
    /// Reciprocal-radius extension multiplier, in [1,1.5].
    pub resolution_extension: f64,
}
impl Default for SharedSearchOptions {
    fn default() -> Self {
        Self {
            maximum_atoms: 128,
            beam_width: 4,
            rounds: 8,
            staged_refinement: true,
            screening_iterations: 8,
            refinement_shortlist: 3,
            integrated_peak_density: true,
            peak_density_radius_angstrom: 0.7,
            dual_peak_sigma: 2.,
            dual_peak_cap: 128,
            dual_checkpoints: true,
            cached_refinement: true,
            symmetry_recycling: false,
            recycling_cycles: 10,
            tangent_refinement: false,
            tangent_in_recycling: true,
            adaptive_counts: true,
            wilson_normalization: false,
            phase_quality_selection: false,
            phase_first_selection: true,
            calculated_shell_normalization: true,
            phase_origin_selection: false,
            adaptive_checkpoints: false,
            score_only_evaluation: false,
            reusable_fourier_workspaces: false,
            work_sliced_phasing: false,
            independent_origin_sampling: false,
            chemical_assignment: false,
            iterative_refinement: false,
            missing_reflections: false,
            resolution_extension: 1.,
        }
    }
}
/// One accepted edit along the selected state's lineage.
#[derive(Clone, Debug)]
pub struct SharedMove {
    /// `initial`, `birth`, `delete`, `relabel`, or `recycle_batch`.
    pub kind: &'static str,
    /// Element assigned/deleted; empty for initialization.
    pub element: String,
    /// Resulting full-cell atom count.
    pub atoms: usize,
    /// Resulting unweighted amplitude R1.
    pub r1: f64,
    /// Descriptive penalized fit score.
    pub score: f64,
}
/// One provisional strong-atom density calibration, with no source composition.
#[derive(Clone, Debug)]
pub struct DensityCalibration {
    /// Actual phasing start index.
    pub trial: usize,
    /// Retained setting lineage.
    pub lineage: usize,
    /// Global edit-round counter.
    pub round: usize,
    /// Relative mass per neutral atomic number, or no usable evidence.
    pub scale: Option<f64>,
    /// Strong asymmetric orbits with positive mass.
    pub anchors: usize,
    /// Density-ambiguity cap on the final-fit shortlist for this parent.
    pub shortlist_limit: usize,
}
/// One actual symmetry-setting recycling attempt.
#[derive(Clone, Debug)]
pub struct RecyclingAudit {
    /// Phase-start index.
    pub trial: usize,
    /// Strong-framework lineage.
    pub lineage: usize,
    /// Source-map hand to supplied symmetry setting.
    pub hand: i32,
    /// Source-map origin shift to supplied symmetry setting.
    pub origin_shift: [f64; 3],
    /// Actual completed positivity cycles.
    pub cycles: usize,
    /// Allowed unmeasured predicted coefficients, including F000.
    pub predicted_coefficients: usize,
    /// Extracted positive map maxima.
    pub peaks: usize,
    /// Whole weak orbits proposed together.
    pub batch_orbits: usize,
    /// Alternative-label scores in joint proposals and coordinate-fixed sweeps.
    pub label_scores: usize,
    /// Accepted coordinate-fixed label changes.
    pub label_changes: usize,
    /// Batch candidate survived a complete requested-mode fit.
    pub batch_refined: bool,
    /// Actual optional phase updates in this supplied symmetry setting.
    pub tangent: Option<crate::TangentAudit>,
}
struct RecyclingContext {
    shape: [usize; 3],
    peaks: Vec<[f64; 3]>,
    integral: IntegratedPeakDensity,
    frame: (i32, [f64; 3]),
    audit: usize,
}
// Converts recycled-frame coordinates to a current candidate's symmetry frame.
fn recycling_frame(parent: &State, context: &RecyclingContext) -> (i32, [f64; 3]) {
    let current = parent.candidate.refinement.as_ref().expect("refined state");
    let hand = current.hand * context.frame.0;
    (
        hand,
        std::array::from_fn(|i| {
            (current.origin_shift[i] - f64::from(hand) * context.frame.1[i]).rem_euclid(1.)
        }),
    )
}

/// Actual shared work, including unsuccessful edits and budget-limited states.
// These are independent configuration/outcome facts in a diagnostic record.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, Default)]
pub struct SharedSearchSummary {
    /// Required element types, without counts.
    pub elements: Vec<String>,
    /// Actual count-independent phase starts.
    pub phasing_starts: usize,
    /// Whether early dual-space structure handoffs were requested.
    pub dual_checkpoints: bool,
    /// Cache fixed reflection/image geometry in constrained refinements.
    pub cached_refinement: bool,
    /// Whether supplied-symmetry map recycling was enabled.
    pub symmetry_recycling: bool,
    /// Requested cycle budget per setting.
    pub recycling_cycles: usize,
    /// Actual setting attempts and batch proposals.
    pub recycling: Vec<RecyclingAudit>,
    /// Whether optional reciprocal phase refinement was requested.
    pub tangent_refinement: bool,
    /// Requested placement policy; active only with tangent and symmetry recycling.
    pub tangent_in_recycling: bool,
    /// Whether joint count-prefix exploration is active.
    pub adaptive_counts: bool,
    /// Requested empirical Wilson normalization.
    pub wilson_normalization: bool,
    /// Requested measured-only phase-quality ranking.
    pub phase_quality_selection: bool,
    /// Requested legacy phase-first schedule (also used without checkpoints).
    pub phase_first_selection: bool,
    /// Independently normalize calculated amplitudes in resolution shells for Rweak.
    pub calculated_shell_normalization: bool,
    /// Requested phase-origin/cheap-completion path.
    pub phase_origin_selection: bool,
    /// Hand off stable quality-ready maps before the phase cycle ceiling.
    pub adaptive_checkpoints: bool,
    /// Evaluate scalar phase merits without unused gradients.
    pub score_only_evaluation: bool,
    /// Reuse dual-space Fourier/map/mask buffers across cycles.
    pub reusable_fourier_workspaces: bool,
    /// Hand early phase starts to model validation at a deterministic half-work quota.
    pub work_sliced_phasing: bool,
    /// Select origin FFT sampling from phase-equivalence frequencies.
    pub independent_origin_sampling: bool,
    /// Add motif density scales and post-screen coupled element assignments.
    pub chemical_assignment: bool,
    /// Solve large damped refinement systems with an iterative normal operator.
    pub iterative_refinement: bool,
    /// Requested observed-envelope gap treatment.
    pub missing_reflections: bool,
    /// Requested reciprocal-radius extension.
    pub resolution_extension: f64,
    /// Actual phase compute slices and their stopping reasons.
    pub phase_slices: Vec<crate::PhaseSliceAudit>,
    /// Actual scalar-only phase scores evaluated.
    pub score_only_evaluations: usize,
    /// Actual P1 projections evaluated with reusable full-grid buffers.
    pub fourier_workspace_cycles: usize,
    /// Per-start phase checkpoint decisions (start index, decision).
    pub phase_checkpoints: Vec<(usize, crate::PhaseCheckpoint)>,
    /// Actual reciprocal support provenance, absent for the original path.
    pub prediction: Option<crate::PredictionAudit>,
    /// Actual prediction calls in phasing and supplied-symmetry recycling.
    pub prediction_steps: usize,
    /// Actual nonzero predicted pair updates, summed over calls.
    pub prediction_updates: usize,
    /// Updates reaching the normalized magnitude cap.
    pub prediction_clipped: usize,
    /// Time in prediction updates (also included in parent stage timers).
    pub prediction_seconds: f64,
    /// Actual origin/setting/completion work, including failed maps.
    pub phase_origins: Vec<PhaseOriginAudit>,
    /// Successful normalization, or None when not requested/failed.
    pub normalization: Option<crate::WilsonAudit>,
    /// Explicit normalization failure; fallback sharpening is labelled.
    pub normalization_diagnostic: Option<String>,
    /// Actual per-start CC/Rweak values and deferred handoffs.
    pub phase_quality: Vec<crate::PhaseQualityTrace>,
    /// Actual phase-start order sent to count refinement after CFOM ranking.
    pub phase_model_trials: Vec<usize>,
    /// Isotropic count prefixes actually screened.
    pub count_prefix_screenings: usize,
    /// Phase maps handed off under a local refinement budget.
    pub bounded_map_handoffs: usize,
    /// Actual measured-only phase plans and completed work per start.
    pub tangent_trials: Vec<crate::TangentAudit>,
    /// Recovered initial framework settings.
    pub framework_hypotheses: usize,
    /// Number of strongest-map peaks in the selected initial closed prefix.
    pub initial_prefix: usize,
    /// Actual completed/started edit rounds.
    pub rounds: usize,
    /// Distinct candidate orbits scored.
    pub orbit_proposals: usize,
    /// Serial edit label scores; recycling batch scores are recorded per setting.
    pub element_scores: usize,
    /// Complete constrained refinements attempted.
    pub refinements: usize,
    /// Actual short isotropic screening fits.
    pub screening_refinements: usize,
    /// Requested final-mode fits, including initial frameworks and promotions.
    pub final_refinements: usize,
    /// Screened edits sent to full refinement.
    pub promoted_candidates: usize,
    /// Valid screened edits omitted from the full-fit shortlist.
    pub screened_out: usize,
    /// Whether integrated-density assignment guidance was enabled.
    pub integrated_peak_density: bool,
    /// Integration sphere radius in angstrom.
    pub peak_density_radius_angstrom: f64,
    /// Actual measured-amplitude integrated maps prepared.
    pub density_map_builds: usize,
    /// Atomic-center integrals evaluated, including orbit averages.
    pub density_evaluations: usize,
    /// Raw birth label scores outside the density prior.
    pub density_label_rejections: usize,
    /// Rejected birth labels rescued by missing-type or best-fit rules.
    pub density_label_rescues: usize,
    /// Net birth screening fits saved relative to two per type.
    pub density_birth_skips: usize,
    /// Relabel screening fits omitted by the density guide.
    pub density_relabel_skips: usize,
    /// Strong-atom scale evidence for each processed parent.
    pub density_calibrations: Vec<DensityCalibration>,
    /// Whether staged scheduling is active for this requested ADP mode.
    pub staged_refinement: bool,
    /// Declared maximum screening steps.
    pub screening_iterations: usize,
    /// Declared base shortlist size.
    pub refinement_shortlist: usize,
    /// Whole-orbit deletion trials.
    pub deletion_trials: usize,
    /// Whole-orbit relabel trials.
    pub relabel_trials: usize,
    /// Maximum full-cell count requested.
    pub maximum_atoms: usize,
    /// Maximum retained states requested.
    pub beam_width: usize,
    /// Whether every required element occurs in the best retained candidate.
    pub required_elements_present: bool,
    /// Selected lineage edit history.
    pub accepted_moves: Vec<SharedMove>,
}
#[derive(Clone)]
struct PromotionSeed {
    model: SymmetryModel,
    site_u: Vec<f64>,
}
#[derive(Clone, Copy)]
enum FitPass<'a> {
    Full,
    Screen(usize),
    ScreenSeed(&'a [f64], usize),
    Promote(&'a [f64]),
}
#[derive(Clone)]
struct State {
    candidate: Candidate,
    aligned: Vec<[f64; 3]>,
    score: f64,
    lineage: usize,
    moves: Vec<SharedMove>,
    initial_prefix: usize,
    promotion: Option<PromotionSeed>,
    recycling_validated: bool,
}

fn orbit_mass(
    map: &IntegratedPeakDensity,
    group: &SpaceGroup,
    point: [f64; 3],
    frame: (i32, [f64; 3]),
    audit: &mut SharedSearchSummary,
) -> Result<f64, SolutionError> {
    let images = group
        .expand_sites(&[point], 1e-8)
        .map_err(|e| error(e.to_string()))?
        .fractional_xyz;
    image_mass(map, &images, frame, audit)
}
fn image_mass(
    map: &IntegratedPeakDensity,
    images: &[[f64; 3]],
    frame: (i32, [f64; 3]),
    audit: &mut SharedSearchSummary,
) -> Result<f64, SolutionError> {
    let mut sum = 0.;
    for p in images {
        let original =
            std::array::from_fn(|i| ((p[i] - frame.1[i]) * f64::from(frame.0)).rem_euclid(1.));
        sum += map.evaluate(original)?.value;
        audit.density_evaluations += 1;
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(sum / images.len() as f64)
}

fn density_scale(
    map: &IntegratedPeakDensity,
    group: &SpaceGroup,
    parent: &State,
    frame: (i32, [f64; 3]),
    audit: &mut SharedSearchSummary,
) -> Result<Option<f64>, SolutionError> {
    let model = parent
        .candidate
        .refinement
        .as_ref()
        .expect("refined parent");
    let strongest = model
        .species
        .iter()
        .filter_map(|e| xray_species_metadata(e).map(|m| m.atomic_number))
        .max()
        .unwrap_or(0);
    let mut ratios = Vec::new();
    for (element, &point) in model.species.iter().zip(&model.fractional_xyz) {
        let z = xray_species_metadata(element).map_or(0, |m| m.atomic_number);
        if f64::from(z) < 0.75 * f64::from(strongest) {
            continue;
        }
        let mass = orbit_mass(map, group, point, frame, audit)?;
        if mass > 0. {
            ratios.push(mass / f64::from(z));
        }
    }
    ratios.sort_by(f64::total_cmp);
    let scale = if ratios.is_empty() || ratios[ratios.len() - 1] > 2. * ratios[0] {
        None
    } else {
        let n = ratios.len();
        Some(if n % 2 == 0 {
            ratios[n / 2 - 1] / 2. + ratios[n / 2] / 2.
        } else {
            ratios[n / 2]
        })
    };
    audit.density_calibrations.push(DensityCalibration {
        trial: parent.candidate.trial,
        lineage: parent.lineage,
        round: audit.rounds,
        scale,
        anchors: ratios.len(),
        shortlist_limit: audit.refinement_shortlist,
    });
    Ok(scale)
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
    for (i, &p) in points.iter().enumerate() {
        for &q in fixed.iter().chain(&points[..i]) {
            if distance2(g, p, q)? < 0.25 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
#[allow(clippy::cast_precision_loss)]
fn rank(j: f64, n: usize, p: usize) -> f64 {
    n as f64 * j.max(1e-300).ln() + p as f64 * (n as f64).ln()
}
// Nested data-proposed orbit prefixes, including the maximal admissible prefix.
fn count_prefixes(available: usize) -> Vec<usize> {
    let mut out: Vec<_> = [1, 2, 4, 8, 16, 32]
        .into_iter()
        .filter(|&n| n <= available)
        .collect();
    if available > 0 && out.last() != Some(&available) {
        out.push(available);
    }
    out
}
fn complete(state: &State, elements: &[String]) -> bool {
    elements.iter().all(|e| state.candidate.species.contains(e))
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn refine(
    model: SymmetryModel,
    data: &PreparedData,
    options: &SolveOptions,
    deadline: Instant,
    frame: (i32, [f64; 3]),
    u: f64,
    lineage: usize,
    old_moves: &[SharedMove],
    kind: &'static str,
    element: String,
    audit: &mut SharedSearchSummary,
    pass: FitPass<'_>,
) -> Result<State, SolutionError> {
    let requested_mode = options
        .refinement
        .ok_or_else(|| error("shared search requires constrained refinement"))?;
    let (mode, iterations) = match pass {
        FitPass::Screen(n) | FitPass::ScreenSeed(_, n) => (AdpMode::Isotropic, n),
        _ => (requested_mode, options.symmetry_iterations),
    };
    let mut species = vec![String::new(); model.full_count];
    for site in &model.sites {
        for image in &site.images {
            species[image.source_index].clone_from(&site.species);
        }
    }
    let prepared = PreparedSymmetricModel::new_with_geometry_cache(
        model,
        &data.hkl,
        &data.amplitudes,
        mode,
        audit.cached_refinement,
    )?;
    if data.family_count <= prepared.parameter_count + 1 {
        return Err(error("shared model is underdetermined"));
    }
    let initial = match pass {
        FitPass::Promote(site_u) | FitPass::ScreenSeed(site_u, _) => {
            prepared.initial_site_parameters(site_u)?
        }
        _ => prepared.initial_parameters(u.clamp(1e-5, 0.2))?,
    };
    let baseline = prepared.evaluate(&initial)?;
    audit.refinements += 1;
    if matches!(pass, FitPass::Screen(_) | FitPass::ScreenSeed(_, _)) {
        audit.screening_refinements += 1;
    } else {
        audit.final_refinements += 1;
    }
    let fit = prepared.refine_with_solver(
        &initial,
        iterations,
        Some(deadline),
        options.cancellation.as_ref(),
        audit.iterative_refinement,
    )?;
    let (mut summary, mut full) = fit.summarize(mode, baseline.r1, baseline.objective);
    let aligned: Vec<_> = full
        .iter()
        .map(|p| {
            std::array::from_fn(|i| {
                (f64::from(summary.hand) * p[i] + summary.origin_shift[i]).rem_euclid(1.)
            })
        })
        .collect();
    if !separated(&fit.model.symmetry.geometry, &aligned, &[])? {
        return Err(error("refined shared model has overlapping atoms"));
    }
    summary.origin_shift = std::array::from_fn(|i| {
        (f64::from(summary.hand) * frame.1[i] + summary.origin_shift[i]).rem_euclid(1.)
    });
    summary.hand *= frame.0;
    for p in &mut full {
        for (value, shift) in p.iter_mut().zip(frame.1) {
            *value = ((*value - shift) * f64::from(frame.0)).rem_euclid(1.);
        }
    }
    let promotion = matches!(pass, FitPass::Screen(_) | FitPass::ScreenSeed(_, _))
        .then(|| promotion_seed(&fit, &summary));
    let score = rank(
        fit.evaluation.objective,
        data.family_count,
        summary.parameter_count + 1,
    );
    let atoms = full.len();
    let candidate = Candidate {
        species,
        fractional_xyz: full,
        u_iso: summary.full_u_equiv.iter().sum::<f64>() / atoms as f64,
        amplitude_scale: fit.evaluation.amplitude_scale,
        r1: fit.evaluation.r1,
        objective: fit.evaluation.objective,
        trial: lineage / audit.beam_width.max(1),
        polish_steps: fit.accepted_steps,
        element_swaps: 0,
        refinement: Some(summary),
        refinement_diagnostic: None,
        completion: None,
    };
    let mut moves = old_moves.to_vec();
    moves.push(SharedMove {
        kind,
        element,
        atoms,
        r1: candidate.r1,
        score,
    });
    Ok(State {
        candidate,
        aligned,
        score,
        lineage,
        moves,
        initial_prefix: audit.initial_prefix,
        promotion,
        recycling_validated: kind != "recycle_batch"
            && !old_moves.iter().any(|m| m.kind == "recycle_batch"),
    })
}

fn promotion_seed(
    fit: &crate::SymmetricRefinement,
    summary: &crate::RefinementSummary,
) -> PromotionSeed {
    let mut model = fit.model.symmetry.clone();
    for (site, &point) in model.sites.iter_mut().zip(&fit.evaluation.fractional_xyz) {
        site.fractional_xyz = point;
    }
    model.hand = summary.hand;
    model.origin_shift = summary.origin_shift;
    PromotionSeed {
        model,
        site_u: summary.u_equiv.iter().map(|u| u.clamp(1e-5, 0.2)).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn edited(
    cell: UnitCell,
    group: &SpaceGroup,
    data: &PreparedData,
    parent: &State,
    sites: &[(String, [f64; 3])],
    kind: &'static str,
    element: String,
    options: &SolveOptions,
    deadline: Instant,
    audit: &mut SharedSearchSummary,
    pass: FitPass<'_>,
) -> Result<State, SolutionError> {
    let mut species = Vec::new();
    let mut points = Vec::new();
    for (e, p) in sites {
        let images = group
            .expand_sites(&[*p], 1e-8)
            .map_err(|e| error(e.to_string()))?;
        species.extend(std::iter::repeat_n(e.clone(), images.fractional_xyz.len()));
        points.extend(images.fractional_xyz);
    }
    if points.len() > audit.maximum_atoms || points.is_empty() {
        return Err(error("shared atom bound exceeded"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    if !separated(&geometry, &points, &[])? {
        return Err(error("shared edit overlaps atoms"));
    }
    let model = recover_symmetry(
        cell,
        group,
        &species,
        &points,
        1e-5,
        Some(deadline),
        options.cancellation.as_ref(),
    )?;
    let frame = parent.candidate.refinement.as_ref().expect("refined state");
    refine(
        model,
        data,
        options,
        deadline,
        (frame.hand, frame.origin_shift),
        parent.candidate.u_iso,
        parent.lineage,
        &parent.moves,
        kind,
        element,
        audit,
        pass,
    )
}

// All multiplicities are discovered from candidate geometry, never requested counts.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn proposals(
    group: &SpaceGroup,
    g: &CellGeometry,
    seeds: &[[f64; 3]],
    fixed: &[[f64; 3]],
    maximum_atoms: usize,
    tolerance: f64,
    options: &SolveOptions,
    deadline: Instant,
) -> Result<Vec<[f64; 3]>, SolutionError> {
    let mut result: Vec<([f64; 3], Vec<[f64; 3]>)> = Vec::new();
    'seeds: for &seed in seeds {
        let count = group
            .expand_sites(&[seed], 1e-8)
            .map_err(|e| error(e.to_string()))?
            .fractional_xyz
            .len();
        let mut states = vec![(seed, Vec::<SymmetryOperation>::new(), count)];
        let mut index = 0;
        while index < states.len() && index < 32 {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break 'seeds;
            }
            let (point, constraints, count) = states[index].clone();
            index += 1;
            let images = group
                .expand_sites(&[point], 1e-8)
                .map_err(|e| error(e.to_string()))?
                .fractional_xyz;
            if count + fixed.len() <= maximum_atoms && separated(g, &images, fixed)? {
                let mut duplicate = false;
                for (_, previous) in &result {
                    if previous.len() == count {
                        for &q in previous {
                            if distance2(g, point, q)? < 0.0004 {
                                duplicate = true;
                                break;
                            }
                        }
                    }
                    if duplicate {
                        break;
                    }
                }
                if !duplicate {
                    result.push((point, images));
                    if result.len() >= 64 {
                        break 'seeds;
                    }
                }
            }
            if constraints.len() >= 3 {
                continue;
            }
            for &operation in group.operations() {
                if states.len() >= 32 {
                    break;
                }
                if constraints.contains(&operation) {
                    continue;
                }
                let mut combined = constraints.clone();
                combined.push(operation);
                let Some(next) = project_fixed_sets(g, seed, &combined, tolerance)? else {
                    continue;
                };
                let multiplicity = group
                    .expand_sites(&[next], 1e-8)
                    .map_err(|e| error(e.to_string()))?
                    .fractional_xyz
                    .len();
                if multiplicity >= count {
                    continue;
                }
                let mut duplicate = false;
                for &(old, _, n) in &states {
                    if n == multiplicity && distance2(g, next, old)? < 1e-12 {
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
    Ok(result.into_iter().map(|(p, _)| p).collect())
}

fn sites(state: &State) -> Vec<(String, [f64; 3])> {
    let s = state.candidate.refinement.as_ref().expect("refined state");
    s.species
        .iter()
        .cloned()
        .zip(s.fractional_xyz.iter().copied())
        .collect()
}
fn same(a: &State, b: &State) -> bool {
    a.lineage == b.lineage
        && (a.score - b.score).abs() < 1e-7
        && sites(a).len() == sites(b).len()
        && sites(a).iter().zip(sites(b)).all(|((e, p), (f, q))| {
            *e == f
                && p.iter().zip(q).all(|(x, y)| {
                    let d = (x - y).rem_euclid(1.);
                    d.min(1. - d) < 1e-6
                })
        })
}
fn ordering(a: &State, b: &State, elements: &[String]) -> std::cmp::Ordering {
    complete(b, elements)
        .cmp(&complete(a, elements))
        .then(a.score.total_cmp(&b.score))
}
fn record(
    winners: &mut Vec<State>,
    beam: &[State],
    elements: &[String],
    width: usize,
    trial: usize,
    density: &[f64],
    result: &mut Solution,
) {
    winners.extend(beam.iter().cloned());
    winners.sort_by(|a, b| ordering(a, b, elements));
    let mut kept = Vec::new();
    for state in winners.drain(..) {
        if !kept.iter().any(|a| same(a, &state)) {
            kept.push(state);
            if kept.len() == width {
                break;
            }
        }
    }
    *winners = kept;
    if winners.first().is_some_and(|s| s.lineage / width == trial) {
        result.density = density.to_vec();
    }
}

fn retain(mut candidates: Vec<State>, width: usize) -> Vec<State> {
    candidates.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.lineage.cmp(&b.lineage)));
    let mut chosen: Vec<State> = Vec::new();
    for c in &candidates {
        if !chosen.iter().any(|a| a.lineage == c.lineage) {
            chosen.push(c.clone());
            if chosen.len() == width {
                return chosen;
            }
        }
    }
    for c in candidates {
        if !chosen.iter().any(|a| same(a, &c)) {
            chosen.push(c);
            if chosen.len() == width {
                break;
            }
        }
    }
    chosen
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn recycled_batch(
    cell: UnitCell,
    group: &SpaceGroup,
    data: &PreparedData,
    parent: &State,
    context: &RecyclingContext,
    elements: &[String],
    search: &SharedSearchOptions,
    options: &SolveOptions,
    deadline: Instant,
    audit: &mut SharedSearchSummary,
) -> Result<Option<State>, SolutionError> {
    let frame = recycling_frame(parent, context);
    let Some(scale) = density_scale(&context.integral, group, parent, frame, audit)? else {
        return Ok(None);
    };
    let pool = PreparedElementPool::new(
        cell,
        group,
        &parent.candidate.species,
        &parent.aligned,
        elements,
        &data.hkl,
        &data.amplitudes,
        parent.candidate.u_iso.clamp(0., 0.2),
    )?;
    let mut seeds: Vec<_> = context
        .peaks
        .iter()
        .take(if search.adaptive_counts { 128 } else { 24 })
        .map(|p| std::array::from_fn(|i| (f64::from(frame.0) * p[i] + frame.1[i]).rem_euclid(1.)))
        .collect();
    if search.adaptive_counts {
        let residual = pool.parent.difference_map(context.shape)?;
        seeds.extend(extract_peaks(
            &residual.density,
            context.shape,
            search.maximum_atoms.max(32),
        )?);
    }
    let points = proposals(
        group,
        &cell.geometry().map_err(|e| error(e.to_string()))?,
        &seeds,
        &parent.aligned,
        search.maximum_atoms,
        options.symmetry_tolerance_angstrom,
        options,
        deadline,
    )?;
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let numbers: Vec<_> = elements
        .iter()
        .map(|e| xray_species_metadata(e).map_or(0, |m| m.atomic_number))
        .collect();
    let min_z = f64::from(*numbers.iter().min().expect("elements"));
    let max_z = f64::from(*numbers.iter().max().expect("elements"));
    let mut candidates = Vec::new();
    for point in points {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let values = pool.evaluate(point)?;
        audit.recycling[context.audit].label_scores += values.len();
        let mass = image_mass(&context.integral, &values[0].fractional_xyz, frame, audit)?;
        let estimate = mass / scale;
        let mass_plausible =
            estimate.is_finite() && estimate >= 0.6 * min_z && estimate <= 1.6 * max_z;
        if !mass_plausible && !search.adaptive_counts {
            continue;
        }
        let label = if mass_plausible {
            (0..elements.len())
                .min_by(|&i, &j| {
                    (estimate / f64::from(numbers[i]))
                        .ln()
                        .abs()
                        .total_cmp(&(estimate / f64::from(numbers[j])).ln().abs())
                })
                .expect("elements")
        } else {
            (0..elements.len())
                .min_by(|&i, &j| values[i].objective.total_cmp(&values[j].objective))
                .expect("labels")
        };
        candidates.push((
            values[label].objective,
            label,
            point,
            values[0].fractional_xyz.clone(),
        ));
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut selected = sites(parent);
    let mut occupied = parent.aligned.clone();
    let mut added = 0;
    for (_, label, point, images) in candidates {
        if added >= if search.adaptive_counts { 2 } else { 8 }
            || images.len() + occupied.len() > search.maximum_atoms
        {
            continue;
        }
        if !separated(&geometry, &images, &occupied)? {
            continue;
        }
        selected.push((elements[label].clone(), point));
        occupied.extend(images);
        added += 1;
    }
    audit.recycling[context.audit].batch_orbits = added;
    if added == 0 {
        return Ok(None);
    }
    // Coordinate-fixed, shared-phase label sweeps; no count enumeration.
    for _ in 0..4 {
        let mut changed = false;
        for index in sites(parent).len()..selected.len() {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            let mut fixed_species = Vec::new();
            let mut fixed_points = Vec::new();
            for (j, (label, point)) in selected.iter().enumerate() {
                if j == index {
                    continue;
                }
                let images = group
                    .expand_sites(&[*point], 1e-8)
                    .map_err(|e| error(e.to_string()))?;
                fixed_species.extend(std::iter::repeat_n(
                    label.clone(),
                    images.fractional_xyz.len(),
                ));
                fixed_points.extend(images.fractional_xyz);
            }
            let pool = PreparedElementPool::new(
                cell,
                group,
                &fixed_species,
                &fixed_points,
                elements,
                &data.hkl,
                &data.amplitudes,
                parent.candidate.u_iso.clamp(0., 0.2),
            )?;
            let scored = pool.evaluate(selected[index].1)?;
            audit.recycling[context.audit].label_scores += scored.len();
            let current = elements
                .iter()
                .position(|e| *e == selected[index].0)
                .expect("allowed label");
            let best = (0..elements.len())
                .min_by(|&i, &j| scored[i].objective.total_cmp(&scored[j].objective))
                .expect("labels");
            if scored[best].objective < scored[current].objective - 1e-10 {
                selected[index].0.clone_from(&elements[best]);
                audit.recycling[context.audit].label_changes += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if !audit.staged_refinement {
        return match edited(
            cell,
            group,
            data,
            parent,
            &selected,
            "recycle_batch",
            String::new(),
            options,
            deadline,
            audit,
            FitPass::Full,
        ) {
            Ok(state) if state.score < parent.score => {
                audit.recycling[context.audit].batch_refined = true;
                Ok(Some(state))
            }
            _ => Ok(None),
        };
    }
    // Compete nested whole-orbit counts under the same isotropic score before
    // paying for a single requested-mode fit. No expected N enters this path.
    let base = sites(parent).len();
    let prefixes = if search.adaptive_counts {
        count_prefixes(selected.len() - base)
    } else {
        vec![selected.len() - base]
    };
    let mut screens = Vec::new();
    for count in prefixes {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        if let Ok(state) = edited(
            cell,
            group,
            data,
            parent,
            &selected[..base + count],
            "recycle_batch",
            String::new(),
            options,
            deadline,
            audit,
            FitPass::Screen(search.screening_iterations),
        ) {
            audit.count_prefix_screenings += usize::from(search.adaptive_counts);
            screens.push(state);
        }
    }
    let Some(screen) = screens
        .into_iter()
        .min_by(|a, b| a.score.total_cmp(&b.score))
    else {
        return Ok(None);
    };
    if screen.score >= parent.score {
        audit.screened_out += 1;
        return Ok(None);
    }
    let seed = screen.promotion.as_ref().expect("screened seed");
    audit.promoted_candidates += 1;
    let full = refine(
        seed.model.clone(),
        data,
        options,
        deadline,
        (1, [0.; 3]),
        screen.candidate.u_iso,
        parent.lineage,
        &parent.moves,
        "recycle_batch",
        String::new(),
        audit,
        FitPass::Promote(&seed.site_u),
    );
    match full {
        Ok(state) if state.score < parent.score => {
            audit.recycling[context.audit].batch_refined = true;
            Ok(Some(state))
        }
        _ => Ok(None),
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn search_phase_map(
    cell: UnitCell,
    group: &SpaceGroup,
    options: &SolveOptions,
    search: &SharedSearchOptions,
    shape: [usize; 3],
    minimum: usize,
    elements: &[String],
    trial: usize,
    density: &[f64],
    mut deadline: Instant,
    global_deadline: Instant,
    winners: &mut Vec<State>,
    result: &mut Solution,
    audit: &mut SharedSearchSummary,
    tangent: Option<&crate::PreparedTangent>,
    phase_context: Option<&phase_path::PhaseSearchContext>,
) -> Result<bool, SolutionError> {
    if let Some(context) = phase_context {
        return phase_path::search(
            context, cell, group, options, search, shape, elements, trial, density, deadline,
            winners, result, audit, tangent,
        );
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let integrated_map = if search.integrated_peak_density {
        audit.density_map_builds += 1;
        Some(IntegratedPeakDensity::new(
            cell,
            shape,
            &result.data.hkl,
            &result.data.amplitudes,
            density,
            search.peak_density_radius_angstrom,
        )?)
    } else {
        None
    };
    let peaks = extract_peaks(density, shape, 32.max(minimum))?;
    if winners.is_empty() {
        result.density = density.to_vec();
    }
    if peaks.len() < minimum {
        result
            .rejected_trials
            .push((trial, "too few initial peaks".into()));
        return Ok(false);
    }
    // A density-height prefix is a proposal heuristic, never supplied atom counts.
    let top = peaks
        .iter()
        .map(|p| {
            let index = std::array::from_fn(|i| (p[i] * shape[i] as f64).round() as i32);
            density[crate::fourier::grid_index(index, shape)]
        })
        .collect::<Vec<_>>();
    let desired = top
        .iter()
        .take_while(|v| **v >= 0.75 * top[0])
        .count()
        .max(minimum);
    let mut prefixes = (minimum..=peaks.len().min(search.maximum_atoms)).collect::<Vec<_>>();
    prefixes.sort_by_key(|n| (n.abs_diff(desired), *n));
    let mut frameworks = Vec::new();
    for prefix in prefixes {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let initial_species = vec![elements[0].clone(); prefix];
        if let Ok(found) = recover_symmetry_hypotheses(
            cell,
            group,
            &initial_species,
            &peaks[..prefix],
            options.symmetry_tolerance_angstrom,
            Some(deadline),
            options.cancellation.as_ref(),
            search.beam_width,
        ) {
            frameworks = found;
            audit.initial_prefix = prefix;
            break;
        }
    }
    if frameworks.is_empty() {
        result.rejected_trials.push((
            trial,
            "no initial density prefix matches supplied symmetry".into(),
        ));
        return Ok(false);
    }
    audit.framework_hypotheses += frameworks.len();
    let mut beam = Vec::new();
    for (lineage, model) in frameworks.into_iter().enumerate() {
        if let Ok(state) = refine(
            model,
            &result.data,
            options,
            deadline,
            (1, [0.; 3]),
            0.01,
            trial * search.beam_width + lineage,
            &[],
            "initial",
            elements[0].clone(),
            audit,
            FitPass::Full,
        ) {
            beam.push(state);
        }
    }
    let mut recycling: Vec<Option<RecyclingContext>> =
        (0..search.beam_width).map(|_| None).collect();
    if search.symmetry_recycling && !crate::stopped(Some(deadline), options.cancellation.as_ref()) {
        let support = crate::solver::reciprocal_support(shape, cell, group, &result.data)?;
        for parent in &beam {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            let frame = parent.candidate.refinement.as_ref().expect("refined state");
            let attempt = crate::recycle_density_guided(
                shape,
                &result.data.hkl,
                &result.data.amplitudes,
                &support,
                group,
                density,
                frame.hand,
                frame.origin_shift,
                search.recycling_cycles,
                Some(deadline),
                options.cancellation.as_ref(),
                tangent,
                options.seed.wrapping_add(trial as u64),
            );
            if let Err(error) = &attempt {
                result.rejected_trials.push((
                    trial,
                    format!("symmetry recycling lineage {}: {error}", parent.lineage),
                ));
            }
            if let Ok(recycled) = attempt {
                let mut peaks = extract_peaks(
                    &recycled.density,
                    shape,
                    if search.adaptive_counts {
                        search.maximum_atoms.max(32)
                    } else {
                        32
                    },
                )?;
                if let Some(first) = peaks.first() {
                    let idx = std::array::from_fn(|i| (first[i] * shape[i] as f64).round() as i32);
                    let threshold = 0.1 * recycled.density[crate::fourier::grid_index(idx, shape)];
                    peaks.retain(|point| {
                        let idx =
                            std::array::from_fn(|i| (point[i] * shape[i] as f64).round() as i32);
                        recycled.density[crate::fourier::grid_index(idx, shape)] >= threshold
                    });
                }
                let integral = IntegratedPeakDensity::new(
                    cell,
                    shape,
                    &result.data.hkl,
                    &result.data.amplitudes,
                    &recycled.density,
                    search.peak_density_radius_angstrom,
                )?;
                audit.density_map_builds += 1;
                let index = audit.recycling.len();
                audit.recycling.push(RecyclingAudit {
                    trial,
                    lineage: parent.lineage,
                    hand: frame.hand,
                    origin_shift: frame.origin_shift,
                    cycles: recycled.cycles,
                    predicted_coefficients: recycled.predicted_coefficients,
                    peaks: peaks.len(),
                    batch_orbits: 0,
                    label_scores: 0,
                    label_changes: 0,
                    batch_refined: false,
                    tangent: recycled.tangent.map(|mut a| {
                        a.trial = trial;
                        a
                    }),
                });
                recycling[parent.lineage % search.beam_width] = Some(RecyclingContext {
                    shape,
                    peaks,
                    integral,
                    frame: (frame.hand, frame.origin_shift),
                    audit: index,
                });
            }
        }
    }
    record(
        winners,
        &beam,
        elements,
        search.beam_width,
        trial,
        density,
        result,
    );
    if winners.first().is_some_and(|s| {
        complete(s, elements) && s.recycling_validated && s.candidate.r1 <= options.r1_target
    }) {
        result.termination = "r1_target";
        return Ok(true);
    }
    if search.symmetry_recycling && search.integrated_peak_density {
        let initial_search = SharedSearchOptions {
            adaptive_counts: false,
            ..search.clone()
        };
        let mut expanded = beam.clone();
        for parent in &beam {
            if let Some(context) = &recycling[parent.lineage % search.beam_width] {
                if let Some(batch) = recycled_batch(
                    cell,
                    group,
                    &result.data,
                    parent,
                    context,
                    elements,
                    &initial_search,
                    options,
                    deadline,
                    audit,
                )? {
                    expanded.push(batch);
                }
            }
        }
        beam = retain(expanded, search.beam_width);
    }
    // A promising atomic model earns the remaining budget; poor early maps
    // return to Fourier recycling instead of monopolizing count editing.
    if search.adaptive_counts && beam.iter().any(|s| s.candidate.r1 < 0.3) {
        deadline = global_deadline;
    }
    for _ in 0..search.rounds {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            result.termination = crate::solver::stop_reason(options);
            break;
        }
        audit.rounds += 1;
        let previous_preferred = beam
            .iter()
            .min_by(|a, b| ordering(a, b, elements))
            .map(|s| (s.score, complete(s, elements), s.lineage));
        let mut next = beam.clone();
        for parent in &beam {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            let base_sites = sites(parent);
            let context = recycling[parent.lineage % search.beam_width].as_ref();
            if search.adaptive_counts
                && search.integrated_peak_density
                && let Some(context) = context
                && let Some(batch) = recycled_batch(
                    cell,
                    group,
                    &result.data,
                    parent,
                    context,
                    elements,
                    search,
                    options,
                    deadline,
                    audit,
                )?
            {
                next.push(batch);
                continue;
            }
            let active_map = if search.integrated_peak_density {
                context.map(|c| &c.integral).or(integrated_map.as_ref())
            } else {
                None
            };
            let frame_ref = parent.candidate.refinement.as_ref().expect("refined state");
            let density_frame = context.map_or((frame_ref.hand, frame_ref.origin_shift), |c| {
                recycling_frame(parent, c)
            });
            let calibration = if let Some(map) = active_map {
                density_scale(map, group, parent, density_frame, audit)?
            } else {
                None
            };
            let numbers: Vec<_> = elements
                .iter()
                .map(|e| xray_species_metadata(e).map_or(0, |m| m.atomic_number))
                .collect();
            let mut site_masks = Vec::new();
            for (_, point) in &base_sites {
                let mask = if let Some(map) = active_map {
                    crate::density_label_mask(
                        orbit_mass(map, group, *point, density_frame, audit)?,
                        calibration,
                        &numbers,
                    )?
                } else {
                    vec![true; elements.len()]
                };
                site_masks.push(mask);
            }
            let mut screened = Vec::new();
            let pass = if audit.staged_refinement {
                FitPass::Screen(search.screening_iterations)
            } else {
                FitPass::Full
            };
            let pool = PreparedElementPool::new(
                cell,
                group,
                &parent.candidate.species,
                &parent.aligned,
                elements,
                &result.data.hkl,
                &result.data.amplitudes,
                parent.candidate.u_iso.clamp(0., 0.2),
            )?;
            let map = pool.parent.difference_map(shape)?;
            let mut seeds = extract_peaks(&map.density, shape, 24)?;
            let frame = parent.candidate.refinement.as_ref().expect("refined state");
            if let Some(context) = context {
                seeds.extend(context.peaks.iter().take(16).map(|p| {
                    std::array::from_fn(|i| {
                        (f64::from(density_frame.0) * p[i] + density_frame.1[i]).rem_euclid(1.)
                    })
                }));
            } else {
                seeds.extend(peaks.iter().take(8).map(|p| {
                    std::array::from_fn(|i| {
                        (f64::from(frame.hand) * p[i] + frame.origin_shift[i]).rem_euclid(1.)
                    })
                }));
            }
            let proposals = proposals(
                group,
                &geometry,
                &seeds,
                &parent.aligned,
                search.maximum_atoms,
                options.symmetry_tolerance_angstrom,
                options,
                deadline,
            )?;
            let mut scored = vec![Vec::new(); elements.len()];
            let mut raw_best: Option<(f64, usize)> = None;
            let mut plausible_label_count = elements.len();
            for point in proposals {
                if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                    break;
                }
                let evaluations = pool.evaluate(point)?;
                let mask = if let Some(map) = active_map {
                    crate::density_label_mask(
                        image_mass(map, &evaluations[0].fractional_xyz, density_frame, audit)?,
                        calibration,
                        &numbers,
                    )?
                } else {
                    vec![true; elements.len()]
                };
                audit.orbit_proposals += 1;
                audit.element_scores += elements.len();
                for (index, evaluation) in evaluations.into_iter().enumerate() {
                    if raw_best.is_none_or(|(best, _)| evaluation.objective < best) {
                        raw_best = Some((evaluation.objective, index));
                        plausible_label_count = mask.iter().filter(|&&allowed| allowed).count();
                    }
                    if !mask[index] {
                        audit.density_label_rejections += 1;
                    }
                    scored[index].push((evaluation.objective, point, mask[index]));
                }
            }
            for (index, mut proposals) in scored.into_iter().enumerate() {
                proposals.sort_by(|a, b| a.0.total_cmp(&b.0));
                let baseline_count = proposals.len().min(2);
                // Prune the original amplitude shortlist without adding new,
                // lower-ranked topology proposals solely from the density prior.
                let mut selected: Vec<_> = proposals
                    .iter()
                    .take(2)
                    .enumerate()
                    .filter(|(_, p)| p.2)
                    .map(|(i, _)| i)
                    .collect();
                let missing = !parent.candidate.species.contains(&elements[index]);
                if (missing || raw_best.is_some_and(|(_, e)| e == index))
                    && !proposals.is_empty()
                    && !selected.contains(&0)
                {
                    selected.push(0);
                    if !proposals[0].2 {
                        audit.density_label_rescues += 1;
                    }
                }
                audit.density_birth_skips += baseline_count - selected.len();
                selected.sort_unstable();
                for selected_index in selected {
                    let (_, point, _) = proposals[selected_index];
                    if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                        break;
                    }
                    let mut added = base_sites.clone();
                    added.push((elements[index].clone(), point));
                    if let Ok(s) = edited(
                        cell,
                        group,
                        &result.data,
                        parent,
                        &added,
                        "birth",
                        elements[index].clone(),
                        options,
                        deadline,
                        audit,
                        pass,
                    ) {
                        if audit.staged_refinement {
                            screened.push(s);
                        } else if s.score < parent.score - 1e-7 {
                            next.push(s);
                        }
                    }
                }
            }
            for index in 0..base_sites.len() {
                if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                    break;
                }
                if base_sites.len() > 1 {
                    let mut deleted = base_sites.clone();
                    let (e, _) = deleted.remove(index);
                    audit.deletion_trials += 1;
                    if let Ok(s) = edited(
                        cell,
                        group,
                        &result.data,
                        parent,
                        &deleted,
                        "delete",
                        e,
                        options,
                        deadline,
                        audit,
                        pass,
                    ) {
                        if audit.staged_refinement {
                            screened.push(s);
                        } else if s.score < parent.score - 1e-7 {
                            next.push(s);
                        }
                    }
                }
                for (label_index, element) in elements.iter().enumerate() {
                    if *element == base_sites[index].0 {
                        continue;
                    }
                    if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                        break;
                    }
                    if !site_masks[index][label_index] && parent.candidate.species.contains(element)
                    {
                        audit.density_relabel_skips += 1;
                        continue;
                    }
                    let mut relabeled = base_sites.clone();
                    relabeled[index].0.clone_from(element);
                    audit.relabel_trials += 1;
                    if let Ok(s) = edited(
                        cell,
                        group,
                        &result.data,
                        parent,
                        &relabeled,
                        "relabel",
                        element.clone(),
                        options,
                        deadline,
                        audit,
                        pass,
                    ) {
                        if audit.staged_refinement {
                            screened.push(s);
                        } else if s.score < parent.score - 1e-7 {
                            next.push(s);
                        }
                    }
                }
            }
            if audit.staged_refinement {
                let shortlist_limit = search
                    .refinement_shortlist
                    .min(plausible_label_count.max(1));
                if let Some(c) = audit
                    .density_calibrations
                    .last_mut()
                    .filter(|_| search.integrated_peak_density)
                {
                    c.shortlist_limit = shortlist_limit;
                }
                let screening_ranks: Vec<_> = screened.iter().map(|s| s.score).collect();
                let births: Vec<_> = screened
                    .iter()
                    .map(|s| {
                        let m = s.moves.last().expect("screened edit");
                        (m.kind == "birth").then(|| m.element.clone())
                    })
                    .collect();
                let missing: Vec<_> = elements
                    .iter()
                    .filter(|e| !parent.candidate.species.contains(e))
                    .cloned()
                    .collect();
                let selected = crate::screening_shortlist(
                    &screening_ranks,
                    &births,
                    &missing,
                    shortlist_limit,
                )?;
                audit.screened_out += screened.len() - selected.len();
                for index in selected {
                    if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                        break;
                    }
                    let candidate = &screened[index];
                    let seed = candidate.promotion.as_ref().expect("screened seed");
                    let edit = candidate.moves.last().expect("screened edit");
                    audit.promoted_candidates += 1;
                    if let Ok(full) = refine(
                        seed.model.clone(),
                        &result.data,
                        options,
                        deadline,
                        (1, [0.; 3]),
                        candidate.candidate.u_iso,
                        candidate.lineage,
                        &candidate.moves[..candidate.moves.len() - 1],
                        edit.kind,
                        edit.element.clone(),
                        audit,
                        FitPass::Promote(&seed.site_u),
                    ) {
                        if full.score < parent.score - 1e-7 {
                            next.push(full);
                        }
                    }
                }
            }
        }
        let previous = beam.clone();
        beam = retain(next, search.beam_width);
        if search.adaptive_counts && beam.iter().any(|s| s.candidate.r1 < 0.3) {
            deadline = global_deadline;
        }
        let progressed = beam.iter().any(|s| {
            s.score
                < previous
                    .iter()
                    .filter(|p| p.lineage == s.lineage)
                    .map(|p| p.score)
                    .fold(f64::INFINITY, f64::min)
                    - 1e-7
        });
        record(
            winners,
            &beam,
            elements,
            search.beam_width,
            trial,
            density,
            result,
        );
        if !crate::stopped(Some(deadline), options.cancellation.as_ref())
            && previous_preferred.is_some_and(|(score, covered, lineage)| {
                covered
                    && winners
                        .first()
                        .is_some_and(|best| best.lineage == lineage && best.score >= score - 1e-7)
            })
        {
            if let Some(best) = winners.first_mut() {
                best.recycling_validated = true;
            }
        }
        if winners.first().is_some_and(|s| {
            complete(s, elements) && s.recycling_validated && s.candidate.r1 <= options.r1_target
        }) {
            result.termination = "r1_target";
            result.density = density.to_vec();
            return Ok(true);
        }
        if !progressed {
            result.termination = "stagnated";
            break;
        }
    }
    Ok(false)
}

/// Infer counts and labels by shared-map orbit editing, without composition enumeration.
/// # Errors
/// Rejects invalid options, required neutral elements, data, geometry or budgets.
/// # Panics
/// Only if an internal refined-state invariant is violated (not from user data).
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn solve_elements(
    observations: &[Observation],
    cell: UnitCell,
    group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    search: &SharedSearchOptions,
) -> Result<Solution, SolutionError> {
    crate::solver::validate_options(options)?;
    crate::ball_transform(0., search.peak_density_radius_angstrom)?;
    if !(1..=128).contains(&search.maximum_atoms)
        || !(1..=8).contains(&search.beam_width)
        || !(1..=16).contains(&search.rounds)
        || !(1..=40).contains(&search.screening_iterations)
        || !(1..=16).contains(&search.refinement_shortlist)
        || !search.dual_peak_sigma.is_finite()
        || !(0.1..=10.).contains(&search.dual_peak_sigma)
        || !(1..=128).contains(&search.dual_peak_cap)
        || !(1..=50).contains(&search.recycling_cycles)
        || (search.tangent_refinement && options.phasing != PhasingMethod::DualSpace)
        || (search.phase_quality_selection && options.phasing != PhasingMethod::DualSpace)
        || (search.chemical_assignment && !search.phase_origin_selection)
        || (search.independent_origin_sampling && !search.phase_origin_selection)
        || (search.score_only_evaluation && !search.phase_quality_selection)
        || (search.reusable_fourier_workspaces && options.phasing != PhasingMethod::DualSpace)
        || (search.work_sliced_phasing
            && (!search.phase_origin_selection
                || !search.phase_quality_selection
                || !search.phase_first_selection
                || search.adaptive_checkpoints))
        || (search.adaptive_checkpoints
            && (!search.phase_origin_selection
                || !search.phase_quality_selection
                || !search.dual_checkpoints
                || !search.phase_first_selection))
        || !search.resolution_extension.is_finite()
        || !(1. ..=1.5).contains(&search.resolution_extension)
        || ((search.missing_reflections || search.resolution_extension > 1.)
            && (!search.phase_origin_selection
                || !search.wilson_normalization
                || options.phasing != PhasingMethod::DualSpace))
        || options.refinement.is_none()
    {
        return Err(error(
            "invalid shared search bounds; requires constrained refinement",
        ));
    }
    let mut elements = elements.to_vec();
    if elements.is_empty()
        || elements.len() > 8
        || elements
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != elements.len()
    {
        return Err(error(
            "shared elements must be 1..8 distinct neutral non-H symbols",
        ));
    }
    for e in &elements {
        let m = xray_species_metadata(e).ok_or_else(|| error("unknown shared element"))?;
        if m.atomic_number <= 1
            || !(1..=2).contains(&e.len())
            || !e.as_bytes()[0].is_ascii_uppercase()
            || e.as_bytes().get(1).is_some_and(|c| !c.is_ascii_lowercase())
        {
            return Err(error("shared elements must be neutral non-H symbols"));
        }
    }
    elements.sort_by_key(|e| {
        std::cmp::Reverse(xray_species_metadata(e).map_or(0, |m| m.atomic_number))
    });
    let deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(options.max_seconds))
        .ok_or_else(|| error("invalid shared deadline"))?;
    let data = prepare_data(observations, cell, group)?;
    if (5 * search.maximum_atoms + 7) * data.hkl.len() > 2_000_000 {
        return Err(error("shared dense refinement memory bound exceeded"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let lengths = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom];
    if lengths.iter().any(|l| l / data.d_min >= 64.) {
        return Err(error("shared resolution sphere exceeds FFT bound"));
    }
    let prediction_enabled = search.missing_reflections || search.resolution_extension > 1.;
    let grid_factor = if search.resolution_extension > 1. {
        lengths
            .iter()
            .map(|l| (64. - 1e-6) * data.d_min / l)
            .fold(search.resolution_extension, f64::min)
    } else {
        1.
    };
    let shape = options.grid_shape.unwrap_or_else(|| {
        lengths.map(|l| {
            let n = (2 * (l * grid_factor / data.d_min).ceil() as usize + 2)
                .max(8)
                .next_power_of_two();
            if search.resolution_extension > 1. {
                n.min(128)
            } else {
                n
            }
        })
    });
    crate::fourier::grid_len(shape)?;
    if (0..3).any(|i| shape[i] as f64 <= 2. * lengths[i] / data.d_min) {
        return Err(error(
            "shared FFT does not contain measured resolution sphere",
        ));
    }
    let support = crate::solver::reciprocal_support(shape, cell, group, &data)?;
    let s: Vec<_> = data
        .hkl
        .iter()
        .map(|h| {
            let q = mv(geometry.reciprocal_basis, h.map(f64::from));
            dot(q, q).sqrt() / 2.
        })
        .collect();
    let factors = PreparedXrayScattering::new(&elements)
        .and_then(|p| p.evaluate(&s))
        .map_err(|e| error(e.to_string()))?
        .real;
    let mut sharpened: Vec<_> = data
        .amplitudes
        .iter()
        .zip(factors.chunks_exact(elements.len()))
        .map(|(&a, f)| a / f.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let normalization_result = if search.wilson_normalization || search.phase_quality_selection {
        Some(crate::normalize_wilson(
            cell,
            group,
            &data.hkl,
            &data.intensity,
            &elements,
        ))
    } else {
        None
    };
    let (normalization, normalization_diagnostic) = match normalization_result {
        Some(Ok(norm)) => (Some(norm), None),
        Some(Err(e)) => (None, Some(e.to_string())),
        None => (None, None),
    };
    if search.wilson_normalization
        && let Some(norm) = &normalization
    {
        sharpened.clone_from(&norm.e_values);
    }
    let prediction_f = if prediction_enabled {
        Some(crate::PreparedReflectionPrediction::new(
            shape,
            cell,
            group,
            &data,
            normalization
                .as_ref()
                .ok_or_else(|| error("prediction normalization failed"))?,
            &elements,
            &data.amplitudes,
            1.,
            search.missing_reflections,
            search.resolution_extension,
        )?)
    } else {
        None
    };
    let prediction_g = if prediction_enabled {
        let mixed: Vec<_> = sharpened
            .iter()
            .zip(&data.amplitudes)
            .map(|(&e, &f)| e.sqrt() * f.sqrt())
            .collect();
        Some(crate::PreparedReflectionPrediction::new(
            shape,
            cell,
            group,
            &data,
            normalization
                .as_ref()
                .ok_or_else(|| error("prediction normalization failed"))?,
            &elements,
            &mixed,
            0.5,
            search.missing_reflections,
            search.resolution_extension,
        )?)
    } else {
        None
    };
    let support = prediction_g.as_ref().map_or(support, |p| p.support.clone());
    let quality_plan = if search.phase_quality_selection {
        normalization
            .as_ref()
            .map(|norm| {
                crate::PreparedPhaseQuality::new(
                    cell,
                    group,
                    &data,
                    observations,
                    &elements,
                    norm,
                    search.wilson_normalization,
                    search.calculated_shell_normalization,
                )
            })
            .transpose()?
    } else {
        None
    };
    let phase_first =
        quality_plan.is_some() && (search.phase_first_selection || !search.dual_checkpoints);
    let mut minimum = group.operations().len();
    for x in [0., 0.125, 0.25, 1. / 3., 0.137] {
        for y in [0., 0.125, 0.25, 1. / 3., 0.271] {
            for z in [0., 0.125, 0.25, 1. / 3., 0.137] {
                minimum = minimum.min(
                    group
                        .expand_sites(&[[x, y, z]], 1e-8)
                        .map_err(|e| error(e.to_string()))?
                        .fractional_xyz
                        .len(),
                );
            }
        }
    }
    if minimum > search.maximum_atoms {
        return Err(error("no sampled initial orbit fits atom bound"));
    }
    let mut result = Solution {
        data,
        grid_shape: shape,
        candidates: Vec::new(),
        density: Vec::new(),
        histories: Vec::new(),
        phasing_trials: Vec::new(),
        rejected_trials: Vec::new(),
        termination: "search_budget",
        target_met: false,
        shared_search: None,
    };
    let mut audit = SharedSearchSummary {
        elements: elements.clone(),
        dual_checkpoints: search.dual_checkpoints,
        cached_refinement: search.cached_refinement,
        symmetry_recycling: search.symmetry_recycling,
        recycling_cycles: search.recycling_cycles,
        tangent_refinement: search.tangent_refinement,
        tangent_in_recycling: search.tangent_in_recycling,
        adaptive_counts: search.adaptive_counts,
        wilson_normalization: search.wilson_normalization,
        phase_quality_selection: search.phase_quality_selection,
        phase_first_selection: search.phase_first_selection,
        calculated_shell_normalization: search.calculated_shell_normalization,
        phase_origin_selection: search.phase_origin_selection,
        adaptive_checkpoints: search.adaptive_checkpoints,
        score_only_evaluation: search.score_only_evaluation,
        reusable_fourier_workspaces: search.reusable_fourier_workspaces,
        work_sliced_phasing: search.work_sliced_phasing,
        independent_origin_sampling: search.independent_origin_sampling,
        chemical_assignment: search.chemical_assignment,
        iterative_refinement: search.iterative_refinement,
        missing_reflections: search.missing_reflections,
        resolution_extension: search.resolution_extension,
        prediction: prediction_f.as_ref().map(|p| p.audit.clone()),
        normalization: normalization.as_ref().map(|n| n.audit.clone()),
        normalization_diagnostic,
        staged_refinement: search.staged_refinement
            && options.refinement == Some(AdpMode::Anisotropic),
        screening_iterations: search.screening_iterations,
        refinement_shortlist: search.refinement_shortlist,
        integrated_peak_density: search.integrated_peak_density,
        peak_density_radius_angstrom: search.peak_density_radius_angstrom,
        maximum_atoms: search.maximum_atoms,
        beam_width: search.beam_width,
        ..SharedSearchSummary::default()
    };
    let tangent_plan = if search.tangent_refinement
        && !crate::stopped(Some(deadline), options.cancellation.as_ref())
    {
        match crate::PreparedTangent::new(
            cell,
            group,
            &result.data.hkl,
            &result.data.intensity,
            &result.data.sigma,
            Some(deadline),
            options.cancellation.as_ref(),
        ) {
            Ok(plan) => Some(plan),
            Err(_) if crate::stopped(Some(deadline), options.cancellation.as_ref()) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let recycle_tangent = if (search.symmetry_recycling || search.phase_origin_selection)
        && search.tangent_in_recycling
    {
        tangent_plan.as_ref()
    } else {
        None
    };
    let dual_tangent = if recycle_tangent.is_some() {
        None
    } else {
        tangent_plan.as_ref()
    };
    let phase_context = if search.phase_origin_selection {
        Some(phase_path::PhaseSearchContext::new(
            &result.data,
            group,
            prediction_f,
        )?)
    } else {
        None
    };
    let mut winners: Vec<State> = Vec::new();
    let mut phase_pool: Vec<(f64, usize, Vec<f64>)> = Vec::new();
    for trial in 0..options.trials {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            result.termination = crate::solver::stop_reason(options);
            break;
        }
        let seed = options.seed.wrapping_add(trial as u64);
        let use_dual = options.phasing == PhasingMethod::DualSpace
            || (options.phasing == PhasingMethod::Portfolio && trial % 2 == 1);
        let phases = crate::solver::initial_phases(&result.data.hkl, seed);
        let (flip, initialization, vector, adaptive, checkpoints) = if use_dual {
            let dual_index = if options.phasing == PhasingMethod::Portfolio {
                trial / 2
            } else {
                trial
            };
            let mixed: Vec<_> = sharpened
                .iter()
                .zip(&result.data.amplitudes)
                .map(|(&e, &f)| e.sqrt() * f.sqrt())
                .collect();
            // The callback owns downstream state; phasing uses independent owned input arrays.
            let phasing_hkl = result.data.hkl.clone();
            let mask_sigma = 0.35 * result.data.d_min;
            let mut chosen_cycle = None;
            let mut handoff = |density: &[f64], cycles: usize, selected_cycle: usize| {
                // Keep candidates and their density frame together across failed checkpoints.
                let old_winners = std::mem::take(&mut winners);
                let old_density = std::mem::take(&mut result.density);
                let map_deadline = if search.adaptive_counts {
                    audit.bounded_map_handoffs += 1;
                    deadline.min(
                        Instant::now() + Duration::from_secs_f64(if cycles < 64 { 2. } else { 4. }),
                    )
                } else {
                    deadline
                };
                search_phase_map(
                    cell,
                    group,
                    options,
                    search,
                    shape,
                    minimum,
                    &elements,
                    trial,
                    density,
                    map_deadline,
                    deadline,
                    &mut winners,
                    &mut result,
                    &mut audit,
                    recycle_tangent,
                    phase_context.as_ref(),
                )?;
                if old_winners.first().is_some_and(|old| {
                    winners
                        .first()
                        .is_none_or(|new| ordering(old, new, &elements).is_le())
                }) {
                    winners = old_winners;
                    result.density = old_density;
                } else if !winners.is_empty() {
                    chosen_cycle = Some(selected_cycle);
                }
                let accepted = winners.first().is_some_and(|state| {
                    complete(state, &elements)
                        && state.recycling_validated
                        && state.candidate.r1 <= options.r1_target
                });
                if !accepted && result.termination == "r1_target" {
                    result.termination = "search_budget";
                }
                Ok(accepted)
            };
            // Quality selection first explores cheap independent phase starts,
            // retaining complete candidate maps before any atom-count refinement.
            let phase_deadline = if phase_first {
                deadline.min(
                    Instant::now()
                        + Duration::from_secs_f64(
                            (0.30 * options.max_seconds / options.trials as f64).max(0.01),
                        ),
                )
            } else {
                deadline
            };
            let cycle_quota = if search.work_sliced_phasing && phase_first {
                crate::phase_cycle_quota(options.cycles, options.trials, trial)?
            } else {
                options.cycles
            };
            let phase_started = Instant::now();
            let mut state = crate::dual_space::dual_space_guided_prediction(
                shape,
                cell,
                &phasing_hkl,
                &mixed,
                &phases,
                &support,
                &crate::DualSpaceOptions {
                    cycles: cycle_quota,
                    peak_selection: crate::PeakSelection::Adaptive {
                        sigma: search.dual_peak_sigma,
                        cap: search.dual_peak_cap,
                    },
                    mask_sigma,
                    hybrid_m: 3.,
                    omit_fraction: 0.1,
                    seed,
                    patterson_rank: if dual_index % 4 == 3 {
                        None
                    } else {
                        Some(dual_index)
                    },
                    cancellation: options.cancellation.clone(),
                },
                Some(phase_deadline),
                if search.dual_checkpoints && !phase_first {
                    Some(&mut handoff)
                } else {
                    None
                },
                dual_tangent,
                quality_plan.as_ref(),
                prediction_g.as_ref(),
                search.adaptive_checkpoints,
                search.score_only_evaluation,
                search.reusable_fourier_workspaces,
            )?;
            audit.phase_slices.push(crate::PhaseSliceAudit {
                trial,
                cycle_quota,
                completed_cycles: state.trial.history.len(),
                seconds: phase_started.elapsed().as_secs_f64(),
                stop_reason: if state.trial.cancelled {
                    "cancelled"
                } else if state.trial.timed_out {
                    "phase_time_limit"
                } else if state.trial.history.len() >= cycle_quota {
                    "cycle_limit"
                } else if state
                    .quality
                    .as_ref()
                    .and_then(|q| q.cfom.get(state.trial.best_cycle))
                    .is_some_and(|q| *q >= 0.9)
                {
                    "quality_target"
                } else {
                    "degenerate"
                },
            });
            audit.score_only_evaluations += state.score_only_evaluations;
            audit.fourier_workspace_cycles += state.fourier_workspace_cycles;
            audit
                .phase_checkpoints
                .extend(state.phase_checkpoints.drain(..).map(|c| (trial, c)));
            audit.prediction_steps += state.prediction.len();
            audit.prediction_updates += state.prediction.iter().map(|s| s.updated).sum::<usize>();
            audit.prediction_clipped += state.prediction.iter().map(|s| s.clipped).sum::<usize>();
            audit.prediction_seconds += state.prediction_seconds;
            if let Some(mut quality) = state.quality.take() {
                if phase_first
                    && !search.phase_origin_selection
                    && let Some(&score) = quality.cfom.get(state.trial.best_cycle)
                {
                    phase_pool.push((score, trial, state.trial.density.clone()));
                    phase_pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    phase_pool.truncate(search.beam_width);
                }
                quality.trial = trial;
                audit.phase_quality.push(quality);
            }
            if let Some(mut tangent) = state.tangent.take() {
                tangent.trial = trial;
                audit.tangent_trials.push(tangent);
            }
            if let Some(selected_cycle) = chosen_cycle {
                state.trial.best_cycle = selected_cycle;
                state.trial.density.clone_from(&result.density);
            } else if winners.is_empty() {
                result.density.clone_from(&state.trial.density);
            }
            (
                state.trial,
                state.initialization,
                state.patterson_vector,
                state.adaptive_history,
                state.checkpoints,
            )
        } else {
            (
                charge_flip(
                    shape,
                    &result.data.hkl,
                    &sharpened,
                    &phases,
                    &support,
                    &FlipOptions {
                        cycles: options.cycles,
                        delta_sigma: options.delta_sigma,
                        weak_fraction: options.weak_fraction,
                        cleanup_cycles: options.cleanup_cycles,
                        cancellation: options.cancellation.clone(),
                    },
                    Some(deadline),
                )?,
                "random_element_sharpening",
                None,
                None,
                Vec::new(),
            )
        };
        audit.phasing_starts += 1;
        result.phasing_trials.push(PhasingTrial {
            method: if use_dual {
                "dual_space"
            } else {
                "charge_flip"
            },
            initialization,
            patterson_vector: vector,
            seed,
            cycles: flip.history.len(),
            best_cycle: flip.best_cycle,
            score_kind: if use_dual {
                "profiled_amplitude_sse"
            } else {
                "normalized_l1"
            },
            mask_sigma: if use_dual {
                Some(0.35 * result.data.d_min)
            } else {
                None
            },
            adaptive_peaks: adaptive,
            checkpoints,
        });
        result.histories.push(flip.history);
        // A promising P1 score earns cheap phase-origin/completion validation.
        // If the resulting complete model fails, continue independent starts.
        if phase_first && search.phase_origin_selection {
            audit.phase_model_trials.push(trial);
            let local = deadline.min(Instant::now() + Duration::from_secs(4));
            search_phase_map(
                cell,
                group,
                options,
                search,
                shape,
                minimum,
                &elements,
                trial,
                &flip.density,
                local,
                deadline,
                &mut winners,
                &mut result,
                &mut audit,
                recycle_tangent,
                phase_context.as_ref(),
            )?;
        }
        if (!use_dual || !search.dual_checkpoints) && quality_plan.is_none() {
            search_phase_map(
                cell,
                group,
                options,
                search,
                shape,
                minimum,
                &elements,
                trial,
                &flip.density,
                deadline,
                deadline,
                &mut winners,
                &mut result,
                &mut audit,
                recycle_tangent,
                phase_context.as_ref(),
            )?;
        }
        if phase_first
            && !search.phase_origin_selection
            && phase_pool.first().is_some_and(|p| p.0 >= 0.9)
        {
            break;
        }
        if winners.first().is_some_and(|state| {
            complete(state, &elements)
                && state.recycling_validated
                && state.candidate.r1 <= options.r1_target
        }) {
            result.termination = "r1_target";
            break;
        }
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            result.termination = crate::solver::stop_reason(options);
            break;
        }
    }
    if phase_first && !search.phase_origin_selection {
        phase_pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, trial, density) in phase_pool {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            audit.phase_model_trials.push(trial);
            let local = deadline.min(Instant::now() + Duration::from_secs(4));
            search_phase_map(
                cell,
                group,
                options,
                search,
                shape,
                minimum,
                &elements,
                trial,
                &density,
                local,
                deadline,
                &mut winners,
                &mut result,
                &mut audit,
                recycle_tangent,
                phase_context.as_ref(),
            )?;
            if winners.first().is_some_and(|s| {
                complete(s, &elements)
                    && s.recycling_validated
                    && s.candidate.r1 <= options.r1_target
            }) {
                result.termination = "r1_target";
                break;
            }
        }
    }
    winners.sort_by(|a, b| ordering(a, b, &elements));
    if let Some(best) = winners.first() {
        audit.initial_prefix = best.initial_prefix;
        audit.required_elements_present = complete(best, &elements);
        audit.accepted_moves.clone_from(&best.moves);
        result.target_met = audit.required_elements_present
            && best.recycling_validated
            && best.candidate.r1 <= options.r1_target;
    }
    result.candidates = winners
        .into_iter()
        .take(search.beam_width)
        .map(|s| s.candidate)
        .collect();
    if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
        result.termination = crate::solver::stop_reason(options);
    }
    if result.density.is_empty() {
        result.density = vec![0.; crate::fourier::grid_len(shape)?];
    }
    result.shared_search = Some(audit);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::evaluate_model;

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn relabel_and_delete_recover_a_known_generated_model() {
        let cell = UnitCell {
            a_angstrom: 6.,
            b_angstrom: 7.,
            c_angstrom: 8.,
            alpha_deg: 90.,
            beta_deg: 100.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let species = ["Fe", "Si", "O"].map(str::to_owned);
        let points = [
            [0.137, 0.259, 0.382],
            [0.421, 0.148, 0.723],
            [0.752, 0.19, 0.158],
        ];
        let mut hkl = Vec::new();
        for h in -5..=5 {
            for k in -5..=5 {
                for l in -5..=5 {
                    if h * h + k * k + l * l <= 25 && [h, k, l] != [0; 3] {
                        hkl.push([h, k, l]);
                    }
                }
            }
        }
        // This test exercises edit orchestration; independent Fourier/FD equations
        // are checked in Python, rather than treating this fixture as an oracle.
        let unit = vec![1.; hkl.len()];
        let calc = evaluate_model(cell, &hkl, &unit, &species, &points, 0.012).unwrap();
        let amplitudes: Vec<_> = calc
            .residual
            .iter()
            .map(|r| (r * (hkl.len() as f64).sqrt() + 1.) / calc.amplitude_scale)
            .collect();
        let observations: Vec<_> = hkl
            .iter()
            .zip(amplitudes)
            .map(|(&h, a)| Observation {
                hkl: h,
                intensity: a * a,
                sigma: 1.,
            })
            .collect();
        let data = prepare_data(&observations, cell, &group).unwrap();
        let options = SolveOptions {
            symmetry_iterations: 120,
            ..SolveOptions::default()
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut audit = SharedSearchSummary {
            maximum_atoms: 32,
            ..SharedSearchSummary::default()
        };
        let wrong = ["Fe", "Fe", "O"].map(str::to_owned);
        let model = recover_symmetry(cell, &group, &wrong, &points, 1e-5, None, None).unwrap();
        let state = refine(
            model,
            &data,
            &options,
            deadline,
            (1, [0.; 3]),
            0.012,
            0,
            &[],
            "initial",
            String::new(),
            &mut audit,
            FitPass::Full,
        )
        .unwrap();
        let mut changed = sites(&state);
        changed[1].0 = "Si".into();
        let fixed = edited(
            cell,
            &group,
            &data,
            &state,
            &changed,
            "relabel",
            "Si".into(),
            &options,
            deadline,
            &mut audit,
            FitPass::Full,
        )
        .unwrap();
        assert!(fixed.candidate.objective < state.candidate.objective);
        assert!(fixed.candidate.r1 < 1e-7);
        assert_eq!(fixed.moves.last().unwrap().kind, "relabel");
        let mut extra = sites(&fixed);
        extra.push(("O".into(), [0.61, 0.44, 0.91]));
        let over = edited(
            cell,
            &group,
            &data,
            &fixed,
            &extra,
            "birth",
            "O".into(),
            &options,
            deadline,
            &mut audit,
            FitPass::Full,
        )
        .unwrap();
        let mut deleted = sites(&over);
        deleted.pop();
        let pruned = edited(
            cell,
            &group,
            &data,
            &over,
            &deleted,
            "delete",
            "O".into(),
            &options,
            deadline,
            &mut audit,
            FitPass::Full,
        )
        .unwrap();
        assert!(pruned.candidate.objective < over.candidate.objective);
        assert_eq!(pruned.candidate.species.len(), 3);
        assert!(pruned.candidate.r1 < 1e-6);
        assert_eq!(pruned.moves.last().unwrap().kind, "delete");
    }
}
