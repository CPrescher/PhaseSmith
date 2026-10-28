//! Budgeted multi-start phasing and full-cell P1 candidate construction.

use std::time::{Duration, Instant};

use phasesmith_crystallography::{
    PreparedXrayScattering, SpaceGroup, UnitCell, xray_species_metadata,
};

use crate::fourier::{grid_index, grid_len};
use crate::{
    FlipOptions, Observation, PreparedData, SolutionError, charge_flip, error, extract_peaks,
    polish_model, prepare_data,
};

/// Explicit resource budgets and experimental numerical controls.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    /// Reproducible `SplitMix64` starting seed.
    pub seed: u64,
    /// Maximum independent phase starts, in [1,128].
    pub trials: usize,
    /// Phasing cycles per trial, in [1,20000].
    pub cycles: usize,
    /// Base threshold; successive starts cycle through 1,1.1,1.2 times this value.
    pub delta_sigma: f64,
    /// Weak-reflection perturbation fraction, in [0,0.5].
    pub weak_fraction: f64,
    /// Final low-density-elimination cycles, in [0,100].
    pub cleanup_cycles: usize,
    /// Coordinate/U polishing iteration budget, in [0,200].
    pub polish_iterations: usize,
    /// Greedy unlike-element swap/refinement rounds, in [0,8]; counts stay fixed.
    pub assignment_rounds: usize,
    /// Wall-clock budget for the complete native solve, in (0,3600] seconds.
    pub max_seconds: f64,
    /// Numerical candidate target; never interpreted as proof of correctness.
    pub r1_target: f64,
    /// Optional explicit FFT grid; auto selects power-of-two axes otherwise.
    pub grid_shape: Option<[usize; 3]>,
    /// Optional shared cooperative stop token; separate from reproducible numerics.
    pub cancellation: Option<crate::Cancellation>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            trials: 8,
            cycles: 800,
            delta_sigma: 1.0,
            weak_fraction: 0.2,
            cleanup_cycles: 10,
            polish_iterations: 50,
            assignment_rounds: 4,
            max_seconds: 60.0,
            r1_target: 0.1,
            grid_shape: None,
            cancellation: None,
        }
    }
}

/// One full-cell P1 atomic hypothesis, with provisional element assignments.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// One species for every full-cell atom.
    pub species: Vec<String>,
    /// Fractional coordinates in [0,1), in an arbitrary origin and hand.
    pub fractional_xyz: Vec<[f64; 3]>,
    /// Common isotropic displacement parameter in square angstrom.
    pub u_iso: f64,
    /// Profiled amplitude scale.
    pub amplitude_scale: f64,
    /// Unweighted amplitude R1 on the expanded input set.
    pub r1: f64,
    /// Normalized amplitude least-squares objective.
    pub objective: f64,
    /// Zero-based start that produced the hypothesis.
    pub trial: usize,
    /// Number of accepted coordinate-polishing steps.
    pub polish_steps: usize,
    /// Number of accepted composition-preserving element swaps.
    pub element_swaps: usize,
}

/// A complete bounded result; the best model may still be incorrect.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Symmetry-merged input and its preparation diagnostics.
    pub data: PreparedData,
    /// Actual FFT shape.
    pub grid_shape: [usize; 3],
    /// Candidate hypotheses, sorted by amplitude objective then trial index.
    pub candidates: Vec<Candidate>,
    /// Map associated with the best atomic candidate (or the last diagnostic map).
    pub density: Vec<f64>,
    /// Phasing misfit histories for all attempted trials.
    pub histories: Vec<Vec<f64>>,
    /// Trials with too few maxima or a numerically degenerate candidate.
    pub rejected_trials: Vec<(usize, String)>,
    /// `r1_target`, `trial_budget`, or `max_runtime`; not a correctness certificate.
    pub termination: &'static str,
    /// Whether an atomic candidate meets the requested R1 target.
    pub target_met: bool,
}

/// Solve from corrected integrated intensities without using reference coordinates.
///
/// `composition` must enumerate every non-hydrogen atom in the full conventional
/// cell. Input symmetry expands intensity constraints only; output remains P1.
///
/// # Errors
/// Rejects invalid controls, cell/symmetry, composition, data, or resource sizes.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn solve(
    observations: &[Observation],
    cell: UnitCell,
    group: &SpaceGroup,
    composition: &[(String, usize)],
    options: &SolveOptions,
) -> Result<Solution, SolutionError> {
    if !(1..=128).contains(&options.trials)
        || !(1..=20_000).contains(&options.cycles)
        || options.cleanup_cycles > 100
        || options.polish_iterations > 200
        || options.assignment_rounds > 8
        || !options.delta_sigma.is_finite()
        || !(0.1..=2.5).contains(&options.delta_sigma)
        || !options.weak_fraction.is_finite()
        || !(0.0..=0.5).contains(&options.weak_fraction)
        || !options.max_seconds.is_finite()
        || !(0.0..=3600.0).contains(&options.max_seconds)
        || options.max_seconds == 0.0
        || !options.r1_target.is_finite()
        || !(0.0..=1.0).contains(&options.r1_target)
    {
        return Err(error("invalid structure-solution options or budgets"));
    }
    let started = Instant::now();
    let deadline = started
        .checked_add(Duration::from_secs_f64(options.max_seconds))
        .ok_or_else(|| error("invalid deadline"))?;
    let mut species = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for (name, count) in composition {
        let metadata = xray_species_metadata(name)
            .ok_or_else(|| error(format!("unknown X-ray species {name}")))?;
        if metadata.atomic_number <= 1
            || !(1..=2).contains(&name.len())
            || !name.as_bytes()[0].is_ascii_uppercase()
            || name
                .as_bytes()
                .get(1)
                .is_some_and(|c| !c.is_ascii_lowercase())
            || *count == 0
            || *count > 128
            || !names.insert(name)
            || species.len() + count > 128
        {
            return Err(error(
                "composition must give distinct non-hydrogen species and positive full-cell counts totaling at most 128",
            ));
        }
        species.extend(std::iter::repeat_n(
            (metadata.atomic_number, name.clone()),
            *count,
        ));
    }
    if species.is_empty() {
        return Err(error("nonempty full-cell composition required"));
    }
    species.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let species: Vec<String> = species.into_iter().map(|(_, name)| name).collect();
    let data = prepare_data(observations, cell, group)?;
    if (5 * species.len() + 7) * data.hkl.len() > 2_000_000 {
        return Err(error("expanded data exceed dense polishing memory budget"));
    }
    if [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom]
        .iter()
        .any(|length| {
            let extent = length / data.d_min;
            !extent.is_finite() || extent >= 64.0
        })
    {
        return Err(error(
            "resolution sphere exceeds the maximum supported FFT grid",
        ));
    }
    let shape = options.grid_shape.unwrap_or_else(|| {
        [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom].map(|length| {
            let required = 2 * (length / data.d_min).ceil() as usize + 2;
            required
                .max(8)
                .checked_next_power_of_two()
                .unwrap_or(usize::MAX)
        })
    });
    let n = grid_len(shape)?;
    for axis in 0..3 {
        let extent = [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom][axis] / data.d_min;
        if shape[axis] as f64 <= 2.0 * extent {
            return Err(error(
                "FFT grid does not strictly contain the full measured resolution sphere",
            ));
        }
    }
    let support = reciprocal_support(shape, cell, group, &data)?;
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let s: Vec<_> = data
        .hkl
        .iter()
        .map(|&h| geometry.d_spacing_and_derivatives(h).map(|(d, _)| 0.5 / d))
        .collect::<Result<_, _>>()
        .map_err(|e| error(e.to_string()))?;
    let scattering = PreparedXrayScattering::new(&species)
        .and_then(|p| p.evaluate(&s))
        .map_err(|e| error(e.to_string()))?;
    let sharpened: Vec<_> = data
        .amplitudes
        .iter()
        .zip(scattering.real.chunks_exact(species.len()))
        .map(|(&a, row)| a / row.iter().map(|f| f * f).sum::<f64>().sqrt())
        .collect();
    let mut result = Solution {
        data,
        grid_shape: shape,
        candidates: Vec::new(),
        density: vec![0.; n],
        histories: Vec::new(),
        rejected_trials: Vec::new(),
        termination: "trial_budget",
        target_met: false,
    };
    let mut best_objective = f64::INFINITY;
    for trial in 0..options.trials {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            result.termination = stop_reason(options);
            break;
        }
        let phases = initial_phases(&result.data.hkl, options.seed.wrapping_add(trial as u64));
        let flip = charge_flip(
            shape,
            &result.data.hkl,
            &sharpened,
            &phases,
            &support,
            &FlipOptions {
                cycles: options.cycles,
                delta_sigma: options.delta_sigma * (1.0 + 0.1 * (trial % 3) as f64),
                weak_fraction: options.weak_fraction,
                cleanup_cycles: options.cleanup_cycles,
                cancellation: options.cancellation.clone(),
            },
            Some(deadline),
        )?;
        result.histories.push(flip.history);
        let peaks = extract_peaks(&flip.density, shape, species.len())?;
        if peaks.len() == species.len() {
            match polish_model(
                cell,
                &result.data.hkl,
                &result.data.amplitudes,
                &species,
                &peaks,
                0.01,
                options.polish_iterations,
                Some(deadline),
                options.cancellation.as_ref(),
            ) {
                Ok(polished) => {
                    let (assigned, polished, swaps) =
                        assign_elements(cell, &result.data, &species, polished, options, deadline)?;
                    let fit = polished.evaluation;
                    if fit.objective < best_objective {
                        best_objective = fit.objective;
                        result.density = flip.density;
                    }
                    result.candidates.push(Candidate {
                        species: assigned,
                        fractional_xyz: polished.fractional_xyz,
                        u_iso: polished.u_iso,
                        amplitude_scale: fit.amplitude_scale,
                        r1: fit.r1,
                        objective: fit.objective,
                        trial,
                        polish_steps: polished.accepted_steps,
                        element_swaps: swaps,
                    });
                }
                Err(e) => result.rejected_trials.push((trial, e.to_string())),
            }
        } else {
            result
                .rejected_trials
                .push((trial, "too few positive density maxima".into()));
            if result.candidates.is_empty() {
                result.density = flip.density;
            }
        }
        if flip.timed_out
            || flip.cancelled
            || crate::stopped(Some(deadline), options.cancellation.as_ref())
        {
            result.termination = stop_reason(options);
            break;
        }
        if result.candidates.iter().any(|c| c.r1 <= options.r1_target) {
            result.target_met = true;
            result.termination = "r1_target";
            break;
        }
    }
    result.candidates.sort_by(|a, b| {
        a.objective
            .total_cmp(&b.objective)
            .then(a.trial.cmp(&b.trial))
    });
    // Timeout takes precedence as a stop reason, but never hides a retained fit.
    result.target_met = result.candidates.iter().any(|c| c.r1 <= options.r1_target);
    Ok(result)
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // Validated grid axes <=128.
fn reciprocal_support(
    shape: [usize; 3],
    cell: UnitCell,
    group: &SpaceGroup,
    data: &PreparedData,
) -> Result<Vec<u8>, SolutionError> {
    let n = grid_len(shape)?;
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let qmax2 = data.d_min.recip().powi(2) * (1.0 + 1e-12);
    let mut support = vec![0; n];
    support[0] = 1;
    for (index, value) in support.iter_mut().enumerate().skip(1) {
        let p = [
            index / (shape[1] * shape[2]),
            (index / shape[2]) % shape[1],
            index % shape[2],
        ];
        let h = std::array::from_fn(|axis| {
            if p[axis] < shape[axis] / 2 {
                p[axis] as i32
            } else {
                p[axis] as i32 - shape[axis] as i32
            }
        });
        let (d, _) = geometry
            .d_spacing_and_derivatives(h)
            .map_err(|e| error(e.to_string()))?;
        if d.recip().powi(2) <= qmax2
            && !group
                .is_systematically_absent(h)
                .map_err(|e| error(e.to_string()))?
        {
            *value = 1;
        }
    }
    for &h in &data.hkl {
        support[grid_index(h, shape)] = 2;
    }
    Ok(support)
}

fn assign_elements(
    cell: UnitCell,
    data: &PreparedData,
    initial: &[String],
    mut model: crate::PolishedModel,
    options: &SolveOptions,
    deadline: Instant,
) -> Result<(Vec<String>, crate::PolishedModel, usize), SolutionError> {
    let mut species = initial.to_vec();
    let mut accepted = 0;
    for _ in 0..options.assignment_rounds {
        if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
            break;
        }
        let mut pairs = Vec::new();
        for left in 0..species.len() {
            for right in left + 1..species.len() {
                if species[left] == species[right] {
                    continue;
                }
                let a = xray_species_metadata(&species[left]).map_or(0, |m| m.atomic_number);
                let b = xray_species_metadata(&species[right]).map_or(0, |m| m.atomic_number);
                pairs.push((a.abs_diff(b), left, right));
            }
        }
        pairs.sort_unstable();
        let mut best_pair = None;
        let mut best_score = model.evaluation.objective;
        for (_, left, right) in pairs.into_iter().take(256) {
            if crate::stopped(Some(deadline), options.cancellation.as_ref()) {
                break;
            }
            species.swap(left, right);
            let trial = crate::evaluate_model(
                cell,
                &data.hkl,
                &data.amplitudes,
                &species,
                &model.fractional_xyz,
                model.u_iso,
            )?;
            species.swap(left, right);
            if trial.objective < best_score * (1.0 - 1e-8) {
                best_score = trial.objective;
                best_pair = Some((left, right));
            }
        }
        let Some((left, right)) = best_pair else {
            break;
        };
        species.swap(left, right);
        let mut next = polish_model(
            cell,
            &data.hkl,
            &data.amplitudes,
            &species,
            &model.fractional_xyz,
            model.u_iso,
            options.polish_iterations,
            Some(deadline),
            options.cancellation.as_ref(),
        )?;
        next.accepted_steps += model.accepted_steps;
        model = next;
        accepted += 1;
    }
    Ok((species, model, accepted))
}

#[allow(clippy::cast_precision_loss)]
fn initial_phases(hkl: &[[i32; 3]], seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut phases = vec![0.; hkl.len()];
    for (i, &h) in hkl.iter().enumerate() {
        let negative = h.map(|v| -v);
        if h < negative {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let phase = std::f64::consts::TAU * ((z >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0;
            phases[i] = phase;
            let j = hkl
                .binary_search(&negative)
                .expect("prepared data include Friedel mates");
            phases[j] = -phase;
        }
    }
    phases
}

fn stop_reason(options: &SolveOptions) -> &'static str {
    if options
        .cancellation
        .as_ref()
        .is_some_and(crate::Cancellation::is_cancelled)
    {
        "cancelled"
    } else {
        "max_runtime"
    }
}
