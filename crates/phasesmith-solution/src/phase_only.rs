//! Reusable P1 phasing from a supplied rotational intensity hypothesis, without an atomic model.
use crate::{
    DualSpaceOptions, Observation, PeakSelection, PhasingExecution, SolutionError, SolveOptions,
    error, normalize_wilson, prepare_data,
};
use phasesmith_crystallography::{SpaceGroup, UnitCell};
use std::time::{Duration, Instant};
/// Explicit optional coarse/adaptive P1 policy. Default preserves the original trajectory.
#[derive(Clone, Debug, Default)]
pub struct PhaseOnlyPolicy {
    /// Inclusive initial phasing cutoff d >= minimum; final observations stay unchanged.
    pub minimum_d_angstrom: Option<f64>,
    /// Use the existing measured correlation/weak-score merit and checkpoint handoff.
    pub adaptive_quality: bool,
}
/// A shared P1 map and actual starts, independent of exact space-group translations.
#[derive(Clone, Debug)]
pub struct PhaseOnlyResult {
    /// Supplied cell identity; exact reuse is checked before candidate assignment.
    pub cell: UnitCell,
    /// Supplied element-type identity, without atom counts.
    pub elements: Vec<String>,
    /// Original measured rows, retained for exact source-data reuse checks.
    pub observations: Vec<Observation>,
    /// P1 density in an arbitrary origin and hand.
    pub density: Vec<f64>,
    /// Actual FFT dimensions.
    pub shape: [usize; 3],
    /// Actual initial Fourier grid, before optional interpolation to the final data grid.
    pub phasing_shape: [usize; 3],
    /// Actual signed measured input rows used by initial phasing.
    pub phasing_observations: usize,
    /// Actual optional per-start measured-quality histories.
    pub quality: Vec<Option<crate::PhaseQualityTrace>>,
    /// Actual checkpoint decisions, separate from atomic-model acceptance.
    pub checkpoints: Vec<Vec<crate::PhaseCheckpoint>>,
    /// Score histories for every attempted start.
    pub histories: Vec<Vec<f64>>,
    /// Selected start, by common mixed-amplitude misfit; no atomic-reference ranking.
    pub selected_start: usize,
    /// Actual phase-only seconds, including unsuccessful starts.
    pub seconds: f64,
}
/// Create one reusable P1 phase map before testing exact space-group translations.
/// The rotational group must have zero translations; no systematic absence is imposed.
/// This reuses the existing Wilson, Patterson, atomicity and Fourier kernels unchanged.
/// # Errors
/// Invalid cell/data/control, nonzero symmetry translations, oversized grid or no completed map.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn phase_only(
    observations: &[Observation],
    cell: UnitCell,
    rotational_group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
) -> Result<PhaseOnlyResult, SolutionError> {
    phase_only_with_workers(observations, cell, rotational_group, elements, options, 1)
}
/// Run bounded independent P1 starts with private buffers and ordered result selection.
/// Worker counts lie in 1..=10; no nested solution workers are used. Arithmetic
/// and start identities match `phase_only` when time/cancellation limits do not intervene.
/// # Errors
/// Invalid workers, ordinary phasing errors, thread admission or a worker panic.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn phase_only_with_workers(
    observations: &[Observation],
    cell: UnitCell,
    rotational_group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    workers: usize,
) -> Result<PhaseOnlyResult, SolutionError> {
    phase_only_with_policy(
        observations,
        cell,
        rotational_group,
        elements,
        options,
        workers,
        &PhaseOnlyPolicy::default(),
    )
}
/// Phase on an optional coarse subset and interpolate the selected map for full-data fitting.
/// The original signed rows remain the exact map identity. Quality uses existing kernels.
/// # Errors
/// Invalid controls/cutoff/data, no retained reflections or no completed map.
#[allow(clippy::too_many_arguments)]
pub fn phase_only_with_policy(
    observations: &[Observation],
    cell: UnitCell,
    rotational_group: &SpaceGroup,
    elements: &[String],
    options: &SolveOptions,
    workers: usize,
    policy: &PhaseOnlyPolicy,
) -> Result<PhaseOnlyResult, SolutionError> {
    if policy
        .minimum_d_angstrom
        .is_some_and(|d| !d.is_finite() || d <= 0.)
    {
        return Err(error("phase-only minimum d must be finite and positive"));
    }
    if !(1..=10).contains(&workers) {
        return Err(error("phase-only workers must lie in 1..=10"));
    }
    crate::solver::validate_options(options)?;
    if elements.is_empty()
        || elements.len() > 8
        || rotational_group
            .operations()
            .iter()
            .any(|op| op.translation().iter().any(|t| t.numerator() != 0))
    {
        return Err(error(
            "phase-only input needs element types and a translation-free intensity group",
        ));
    }
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(options.max_seconds);
    let full_data = prepare_data(observations, cell, rotational_group)?;
    let full_shape = phase_shape(cell, full_data.d_min)?;
    let rows = phasing_subset(observations, cell, policy.minimum_d_angstrom)?;
    let data = if policy.minimum_d_angstrom.is_some() {
        prepare_data(&rows, cell, rotational_group)?
    } else {
        full_data
    };
    let shape = phase_shape(cell, data.d_min)?;
    let support = crate::solver::reciprocal_support(shape, cell, rotational_group, &data)?;
    let norm = normalize_wilson(cell, rotational_group, &data.hkl, &data.intensity, elements)?;
    let mixed: Vec<_> = norm
        .e_values
        .iter()
        .zip(&data.amplitudes)
        .map(|(&e, &f)| e.sqrt() * f.sqrt())
        .collect();
    let quality = if policy.adaptive_quality {
        Some(crate::PreparedPhaseQuality::new(
            cell,
            rotational_group,
            &data,
            &rows,
            elements,
            &norm,
            true,
            true,
        )?)
    } else {
        None
    };
    let input = PhaseInput {
        cell,
        shape,
        hkl: &data.hkl,
        mixed: &mixed,
        support: &support,
        d_min: data.d_min,
        options,
        deadline,
        quality: quality.as_ref(),
    };
    let selection = select_starts(&input, workers)?;
    Ok(PhaseOnlyResult {
        cell,
        elements: elements.to_vec(),
        observations: observations.to_vec(),
        density: crate::interpolate_phase_density(
            &selection
                .density
                .ok_or_else(|| error("no phase-only start completed within budget"))?,
            shape,
            full_shape,
        )?,
        shape: full_shape,
        phasing_shape: shape,
        phasing_observations: rows.len(),
        quality: selection.quality,
        checkpoints: selection.checkpoints,
        histories: selection.histories,
        selected_start: selection.start,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Finite extent is explicitly bounded below 64.
fn phase_shape(cell: UnitCell, d_min: f64) -> Result<[usize; 3], SolutionError> {
    let mut shape = [0; 3];
    for (a, length) in [cell.a_angstrom, cell.b_angstrom, cell.c_angstrom]
        .iter()
        .enumerate()
    {
        let extent = length / d_min;
        if !extent.is_finite() || extent >= 64. {
            return Err(error(
                "phase-only resolution sphere exceeds 128-grid support",
            ));
        }
        shape[a] = (2 * extent.ceil() as usize + 2).max(8).next_power_of_two();
    }
    Ok(shape)
}
fn phasing_subset(
    rows: &[Observation],
    cell: UnitCell,
    cutoff: Option<f64>,
) -> Result<Vec<Observation>, SolutionError> {
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    Ok(rows
        .iter()
        .filter(|r| {
            cutoff.is_none_or(|d| {
                let q = crate::math3::mv(geometry.reciprocal_basis, r.hkl.map(f64::from));
                crate::math3::dot(q, q) <= 1. / (d * d)
            })
        })
        .cloned()
        .collect())
}

struct PhaseInput<'a> {
    cell: UnitCell,
    shape: [usize; 3],
    hkl: &'a [[i32; 3]],
    mixed: &'a [f64],
    support: &'a [u8],
    d_min: f64,
    options: &'a SolveOptions,
    deadline: Instant,
    quality: Option<&'a crate::PreparedPhaseQuality>,
}
impl PhaseInput<'_> {
    #[allow(clippy::cast_precision_loss)]
    fn trial(&self, start: usize) -> Result<crate::DualSpaceResult, SolutionError> {
        let seed = self.options.seed.wrapping_add(start as u64);
        crate::dual_space::dual_space_guided_prediction(
            self.shape,
            self.cell,
            self.hkl,
            self.mixed,
            &crate::solver::initial_phases(self.hkl, seed),
            self.support,
            &DualSpaceOptions {
                cycles: self.options.cycles,
                peak_selection: PeakSelection::Adaptive {
                    sigma: 2.,
                    cap: 128,
                },
                mask_sigma: 0.35 * self.d_min,
                hybrid_m: 3.,
                omit_fraction: 0.1,
                seed,
                patterson_rank: if start % 4 == 3 { None } else { Some(start) },
                cancellation: self.options.cancellation.clone(),
            },
            Some(self.deadline),
            None,
            None,
            self.quality,
            None,
            self.quality.is_some(),
            PhasingExecution {
                score_only_evaluation: self.quality.is_some(),
                reusable_fourier_workspaces: true,
                filtered_peak_search: true,
                streamed_peak_search: true,
                batched_fft: true,
                real_fft: true,
                fused_phase_workspace: true,
                ..PhasingExecution::default()
            },
        )
    }
}
struct PhaseSelection {
    quality: Vec<Option<crate::PhaseQualityTrace>>,
    checkpoints: Vec<Vec<crate::PhaseCheckpoint>>,
    histories: Vec<Vec<f64>>,
    density: Option<Vec<f64>>,
    start: usize,
}
fn select_starts(input: &PhaseInput<'_>, workers: usize) -> Result<PhaseSelection, SolutionError> {
    let mut selected = PhaseSelection {
        quality: Vec::new(),
        checkpoints: Vec::new(),
        histories: Vec::new(),
        density: None,
        start: 0,
    };
    let mut best = f64::INFINITY;
    for first in (0..input.options.trials).step_by(workers) {
        if crate::stopped(Some(input.deadline), input.options.cancellation.as_ref()) {
            break;
        }
        let last = (first + workers).min(input.options.trials);
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut failure = None;
            for start in first..last {
                match std::thread::Builder::new().spawn_scoped(scope, move || {
                    crate::with_solution_workers(1, || input.trial(start)).map(|(result, _)| result)
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(e) => {
                        failure = Some(error(format!("cannot admit phase worker: {e}")));
                        break;
                    }
                }
            }
            let mut results = Vec::new();
            for handle in handles {
                match handle.join() {
                    Ok(Ok(value)) => results.push(value),
                    Ok(Err(e)) => {
                        if failure.is_none() {
                            failure = Some(e);
                        }
                    }
                    Err(_) => {
                        if failure.is_none() {
                            failure = Some(error("phase worker panicked"));
                        }
                    }
                }
            }
            if let Some(e) = failure {
                Err(e)
            } else {
                Ok(results)
            }
        })?;
        for (offset, result) in results.into_iter().enumerate() {
            let score = result
                .trial
                .history
                .get(result.trial.best_cycle)
                .copied()
                .unwrap_or(f64::INFINITY);
            let score = result.quality.as_ref().map_or(score, |q| {
                q.cfom
                    .get(result.trial.best_cycle)
                    .map_or(f64::INFINITY, |v| -v)
            });
            if score < best {
                best = score;
                selected.start = first + offset;
                selected.density = Some(result.trial.density);
            }
            selected.histories.push(result.trial.history);
            selected.quality.push(result.quality);
            selected.checkpoints.push(result.phase_checkpoints);
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasesmith_crystallography::{Rational, SymmetryOperation};
    #[test]
    fn phase_map_has_no_atomic_or_exact_translation_prior_and_keeps_stop_contracts() {
        let (cell, group, rows, options) = fixture();
        let a = phase_only(&rows, cell, &group, &["Si".into()], &options).unwrap();
        assert_eq!(a.histories.len(), 1);
        assert_eq!(a.histories[0].len(), 8);
        assert_eq!(a.density.len(), a.shape.iter().product());
        assert!(a.density.iter().all(|x| x.is_finite()));
        let translation = SymmetryOperation::new(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [
                Rational::new(1, 2).unwrap(),
                Rational::zero(),
                Rational::zero(),
            ],
        )
        .unwrap();
        let centred = SpaceGroup::new(vec![SymmetryOperation::identity(), translation]).unwrap();
        assert!(phase_only(&rows, cell, &centred, &["Si".into()], &options).is_err());
        let token = crate::Cancellation::default();
        assert!(token.cancel());
        assert!(
            phase_only(
                &rows,
                cell,
                &group,
                &["Si".into()],
                &SolveOptions {
                    cancellation: Some(token),
                    ..options
                }
            )
            .is_err()
        );
    }
    fn fixture() -> (UnitCell, SpaceGroup, Vec<Observation>, SolveOptions) {
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 6.,
            c_angstrom: 7.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let group = SpaceGroup::new(vec![SymmetryOperation::identity()]).unwrap();
        let mut rows = Vec::new();
        for h in -2i32..=2 {
            for k in -2i32..=2 {
                for l in -2i32..=2 {
                    if [h, k, l] != [0; 3] {
                        rows.push(Observation {
                            hkl: [h, k, l],
                            intensity: f64::from(1 + h * h + 2 * k * k + 3 * l * l),
                            sigma: 1.,
                        });
                    }
                }
            }
        }
        let options = SolveOptions {
            trials: 1,
            cycles: 8,
            max_seconds: 5.,
            ..SolveOptions::default()
        };
        (cell, group, rows, options)
    }
    #[test]
    fn shared_map_reuse_skips_phasing_and_rejects_other_source_data() {
        let (cell, group, rows, options) = fixture();
        let a = phase_only(&rows, cell, &group, &["Si".into()], &options).unwrap();
        let seeded_options = SolveOptions {
            max_seconds: 1.,
            refinement: Some(crate::AdpMode::Isotropic),
            symmetry_iterations: 0,
            ..options.clone()
        };
        let search = crate::SharedSearchOptions {
            maximum_atoms: 4,
            beam_width: 1,
            rounds: 1,
            phase_origin_selection: true,
            ..crate::SharedSearchOptions::default()
        };
        let seeded = crate::solve_elements_from_phase_map(
            &rows,
            cell,
            &group,
            &["Si".into()],
            &seeded_options,
            &search,
            &a,
        )
        .unwrap();
        assert_eq!(seeded.shared_search.unwrap().phasing_starts, 0);
        assert_eq!(seeded.grid_shape, a.shape);
        assert_eq!(seeded.phasing_trials[0].method, "shared_p1");
        let mut changed = rows.clone();
        changed[0].sigma = 2.;
        assert!(
            crate::solve_elements_from_phase_map(
                &changed,
                cell,
                &group,
                &["Si".into()],
                &seeded_options,
                &search,
                &a
            )
            .is_err()
        );
        let mut other_cell = cell;
        other_cell.a_angstrom += 0.001;
        assert!(
            crate::solve_elements_from_phase_map(
                &rows,
                other_cell,
                &group,
                &["Si".into()],
                &seeded_options,
                &search,
                &a
            )
            .is_err()
        );
        assert!(
            crate::solve_elements_from_phase_map(
                &rows,
                cell,
                &group,
                &["Fe".into()],
                &seeded_options,
                &search,
                &a
            )
            .is_err()
        );
    }
    #[test]
    fn ordered_parallel_starts_preserve_every_phase_value() {
        let (cell, group, rows, mut options) = fixture();
        options.trials = 4;
        let serial = phase_only(&rows, cell, &group, &["Si".into()], &options).unwrap();
        let parallel =
            phase_only_with_workers(&rows, cell, &group, &["Si".into()], &options, 4).unwrap();
        assert_eq!(serial.selected_start, parallel.selected_start);
        assert_eq!(
            serial
                .histories
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            parallel
                .histories
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            serial
                .density
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            parallel
                .density
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(phase_only_with_workers(&rows, cell, &group, &["Si".into()], &options, 0).is_err());
        assert!(
            phase_only_with_workers(&rows, cell, &group, &["Si".into()], &options, 11).is_err()
        );
    }
}
