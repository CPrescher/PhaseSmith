//! Patterson superposition, hybrid difference Fourier and random-omit atomicity.

use crate::fourier::{
    Transform, grid_index, grid_len, local_maxima, prepare_fourier, refine_peak_indices,
};
use crate::{FlipResult, SolutionError, error, extract_peaks};
use phasesmith_crystallography::{CellGeometry, UnitCell};
use rustfft::num_complex::Complex;
use std::time::Instant;

/// Atomicity selection, explicitly distinguishing supplied counts from data thresholds.
#[derive(Clone, Debug)]
pub enum PeakSelection {
    /// Supplied complete-cell count; legacy warm-up doubles it for 50 cycles.
    Fixed(usize),
    /// Positive maxima above a map RMS threshold, bounded without a count prior.
    Adaptive {
        /// Positive density threshold as a multiple of map RMS.
        sigma: f64,
        /// Resource bound, independent of composition.
        cap: usize,
    },
}
/// Actual adaptive selection aligned to completed Fourier cycles.
#[derive(Clone, Debug)]
pub struct AdaptivePeakHistory {
    /// Requested density RMS multiplier.
    pub sigma: f64,
    /// Resource cap on simultaneous mask peaks.
    pub cap: usize,
    /// Actual pre-omission selected peak counts.
    pub selected: Vec<usize>,
    /// Actual effective RMS multipliers after overflow adjustment.
    pub thresholds: Vec<f64>,
    /// Actual omitted counts, zero outside omission cycles.
    pub omitted: Vec<usize>,
}
/// Explicit controls for the independently implemented published dual-space mechanisms.
#[derive(Clone, Debug)]
pub struct DualSpaceOptions {
    /// Maximum cycles, in [1,20000].
    pub cycles: usize,
    /// Fixed-count or data-adaptive atomicity selection.
    pub peak_selection: PeakSelection,
    /// Cartesian Gaussian mask sigma in angstrom, strictly positive.
    pub mask_sigma: f64,
    /// Hybrid difference multiplier, in [1,4].
    pub hybrid_m: f64,
    /// Fraction of mask peaks omitted every third cycle, in [0,0.3].
    pub omit_fraction: f64,
    /// Stateless omission seed.
    pub seed: u64,
    /// Ranked Patterson vector, or None to use the supplied random phases.
    pub patterson_rank: Option<usize>,
    /// Cooperative stop token.
    pub cancellation: Option<crate::Cancellation>,
}

/// One attempted handoff of a Fourier window to full shared structure search.
#[derive(Clone, Debug)]
pub struct DualSpaceCheckpoint {
    /// Completed Fourier cycles, including this window.
    pub completed_cycles: usize,
    /// Zero-based cycle of this window's best non-omitted atomic map.
    pub best_cycle: usize,
    /// Profiled mixed-amplitude SSE of that map.
    pub score: f64,
    /// True only if downstream complete-model refinement meets its target.
    pub accepted: bool,
}

/// Fourier trial and actual initialization (fallbacks are never hidden).
#[derive(Clone, Debug)]
pub struct DualSpaceResult {
    /// Atomic map, score history and stop flags.
    pub trial: FlipResult,
    /// `patterson`, `random`, or `random_no_patterson_vector`.
    pub initialization: &'static str,
    /// Integer Patterson shift modulo grid shape, if used.
    pub patterson_vector: Option<[i32; 3]>,
    /// Present only for genuinely count-free adaptive selection.
    pub adaptive_history: Option<AdaptivePeakHistory>,
    /// Actual structure-search handoffs, empty for unmonitored runs.
    pub checkpoints: Vec<DualSpaceCheckpoint>,
    /// Actual optional tangent/annealing work, absent for the legacy path.
    pub tangent: Option<crate::TangentAudit>,
    /// Actual CC/Rweak ranking trace, absent on the legacy trajectory.
    pub quality: Option<crate::PhaseQualityTrace>,
    /// Actual optional prediction steps, not observed scores.
    pub prediction: Vec<crate::PredictionStep>,
    /// Time in prediction updates.
    pub prediction_seconds: f64,
    /// Actual optional phase checkpoint decisions.
    pub phase_checkpoints: Vec<crate::PhaseCheckpoint>,
    /// Actual scalar-only merit evaluations.
    pub score_only_evaluations: usize,
    /// Actual projections using reused full-grid buffers.
    pub fourier_workspace_cycles: usize,
    /// Actual cycles using exact threshold-filtered peak selection.
    pub filtered_peak_cycles: usize,
    /// Actual batched 3D transforms in this P1 phase call.
    pub batched_phase_transforms: usize,
    /// Actual packed-real transforms in P1 cycles, excluding initialization.
    pub real_phase_transforms: usize,
}

pub(crate) fn hash64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn omitted(count: usize, cycle: usize, options: &DualSpaceOptions) -> Vec<bool> {
    let mut mask = vec![false; count];
    if cycle % 3 != 2 || count < 2 || options.omit_fraction == 0. {
        return mask;
    }
    let number = ((count as f64 * options.omit_fraction).floor() as usize)
        .max(1)
        .min(count - 1);
    let mut ranks: Vec<_> = (0..count).collect();
    ranks.sort_by_key(|&i| {
        (
            hash64(options.seed ^ hash64(cycle as u64) ^ hash64(i as u64)),
            i,
        )
    });
    for i in ranks.into_iter().take(number) {
        mask[i] = true;
    }
    mask
}

fn mask_bounds(shape: [usize; 3], g: &CellGeometry, sigma: f64) -> Result<[i32; 3], SolutionError> {
    let mut bounds = [0; 3];
    for i in 0..3 {
        let fractional = 3. * sigma * g.reciprocal_metric[i][i].sqrt();
        let steps = fractional * f64::from(u16::try_from(shape[i]).unwrap_or(0));
        if !steps.is_finite() || fractional >= 0.5 || steps > 16. {
            return Err(error(
                "Gaussian mask exceeds half-axis or 16-step support budget",
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            bounds[i] = steps.ceil() as i32 + 1;
        }
    }
    Ok(bounds)
}

/// Finite Cartesian Gaussian atomicity projection, with deterministic random omission.
///
/// # Errors
/// Rejects malformed arrays, controls, cell geometry or an excessive mask support.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments
)]
pub fn atomic_projection(
    density: &[f64],
    shape: [usize; 3],
    cell: UnitCell,
    options: &DualSpaceOptions,
    cycle: usize,
) -> Result<Vec<f64>, SolutionError> {
    Ok(project_atomicity(density, shape, cell, options, cycle)?.density)
}
struct Projection {
    density: Vec<f64>,
    selected: usize,
    sigma: f64,
    omitted: usize,
}
/// Data-selected positions and effective threshold for independent diagnostics.
/// # Errors
/// Rejects invalid controls, maps, grid dimensions or a zero RMS multiplier.
#[allow(clippy::cast_precision_loss)]
pub fn adaptive_peaks(
    density: &[f64],
    shape: [usize; 3],
    sigma: f64,
    cap: usize,
) -> Result<(Vec<[f64; 3]>, f64), SolutionError> {
    if !sigma.is_finite() || !(0.1..=10.).contains(&sigma) || !(1..=128).contains(&cap) {
        return Err(error("invalid adaptive peak threshold or cap"));
    }
    let maxima = local_maxima(density, shape)?;
    let maximum = density.iter().map(|v| v.abs()).fold(0., f64::max);
    if maximum == 0. {
        return Ok((Vec::new(), sigma));
    }
    let rms = maximum
        * (density.iter().map(|v| (v / maximum).powi(2)).sum::<f64>() / density.len() as f64)
            .sqrt();
    let mut effective = sigma;
    loop {
        let chosen: Vec<_> = maxima
            .iter()
            .copied()
            .take_while(|&i| density[i] >= effective * rms)
            .collect();
        if chosen.len() <= cap {
            return Ok((refine_peak_indices(&chosen, density, shape), effective));
        }
        effective *= 1.5;
    }
}
struct AtomicWorkspace {
    geometry: CellGeometry,
    bounds: [i32; 3],
    radius2: f64,
    normalization: f64,
    mask: Vec<f64>,
    projected: Vec<f64>,
    peaks: Option<crate::PreparedAdaptivePeaks>,
}
impl AtomicWorkspace {
    fn new(
        shape: [usize; 3],
        cell: UnitCell,
        options: &DualSpaceOptions,
        filtered_peaks: bool,
    ) -> Result<Self, SolutionError> {
        validate_options(options)?;
        let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
        let bounds = mask_bounds(shape, &geometry, options.mask_sigma)?;
        let n = grid_len(shape)?;
        Ok(Self {
            geometry,
            bounds,
            radius2: (3. * options.mask_sigma).powi(2) * (1. + 1e-12),
            normalization: (std::f64::consts::TAU).powf(-1.5) / options.mask_sigma.powi(3),
            mask: vec![0.; n],
            projected: vec![0.; n],
            peaks: if filtered_peaks {
                Some(crate::PreparedAdaptivePeaks::new(shape)?)
            } else {
                None
            },
        })
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn update(
        &mut self,
        density: &[f64],
        shape: [usize; 3],
        options: &DualSpaceOptions,
        cycle: usize,
    ) -> Result<(usize, f64, usize), SolutionError> {
        let (peaks, effective) = match options.peak_selection {
            PeakSelection::Fixed(count) => (
                extract_peaks(
                    density,
                    shape,
                    if cycle < 50 {
                        (2 * count).min(128)
                    } else {
                        count
                    },
                )?,
                0.,
            ),
            PeakSelection::Adaptive { sigma, cap } => {
                if let Some(plan) = &mut self.peaks {
                    plan.select(density, sigma, cap)?
                } else {
                    adaptive_peaks(density, shape, sigma, cap)?
                }
            }
        };
        let skip = omitted(peaks.len(), cycle, options);
        let omitted_count = skip.iter().filter(|&&v| v).count();
        self.mask.fill(0.);
        let radius2 = self.radius2;
        let normalization = self.normalization;
        for (&point, &skip) in peaks.iter().zip(&skip) {
            if skip {
                continue;
            }
            let centre = std::array::from_fn::<_, 3, _>(|i| point[i] * shape[i] as f64);
            let origin = centre.map(|v| v.round() as i32);
            for dx in -self.bounds[0]..=self.bounds[0] {
                for dy in -self.bounds[1]..=self.bounds[1] {
                    for dz in -self.bounds[2]..=self.bounds[2] {
                        let index = [origin[0] + dx, origin[1] + dy, origin[2] + dz];
                        let delta = std::array::from_fn(|i| {
                            (f64::from(index[i]) - centre[i]) / shape[i] as f64
                        });
                        let squared = crate::math3::dot(
                            delta,
                            crate::math3::mv(self.geometry.direct_metric, delta),
                        );
                        if squared <= radius2 {
                            self.mask[grid_index(index, shape)] +=
                                normalization * (-0.5 * squared / options.mask_sigma.powi(2)).exp();
                        }
                    }
                }
            }
        }

        for ((out, &v), &m) in self.projected.iter_mut().zip(density).zip(&self.mask) {
            *out = v.max(0.) * m;
        }
        Ok((peaks.len(), effective, omitted_count))
    }
}
// Retain the original allocating kernel as the execution-control ablation.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn project_atomicity(
    density: &[f64],
    shape: [usize; 3],
    cell: UnitCell,
    options: &DualSpaceOptions,
    cycle: usize,
) -> Result<Projection, SolutionError> {
    validate_options(options)?;
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    let bounds = mask_bounds(shape, &geometry, options.mask_sigma)?;
    let (peaks, effective) = match options.peak_selection {
        PeakSelection::Fixed(count) => (
            extract_peaks(
                density,
                shape,
                if cycle < 50 {
                    (2 * count).min(128)
                } else {
                    count
                },
            )?,
            0.,
        ),
        PeakSelection::Adaptive { sigma, cap } => adaptive_peaks(density, shape, sigma, cap)?,
    };
    let skip = omitted(peaks.len(), cycle, options);
    let omitted_count = skip.iter().filter(|&&v| v).count();
    let mut mask = vec![0.; density.len()];
    let radius2 = (3. * options.mask_sigma).powi(2) * (1. + 1e-12);
    let normalization = (std::f64::consts::TAU).powf(-1.5) / options.mask_sigma.powi(3);
    for (&point, &skip) in peaks.iter().zip(&skip) {
        if skip {
            continue;
        }
        let centre = std::array::from_fn::<_, 3, _>(|i| point[i] * shape[i] as f64);
        let origin = centre.map(|v| v.round() as i32);
        for dx in -bounds[0]..=bounds[0] {
            for dy in -bounds[1]..=bounds[1] {
                for dz in -bounds[2]..=bounds[2] {
                    let index = [origin[0] + dx, origin[1] + dy, origin[2] + dz];
                    let delta = std::array::from_fn(|i| {
                        (f64::from(index[i]) - centre[i]) / shape[i] as f64
                    });
                    let squared =
                        crate::math3::dot(delta, crate::math3::mv(geometry.direct_metric, delta));
                    if squared <= radius2 {
                        mask[grid_index(index, shape)] +=
                            normalization * (-0.5 * squared / options.mask_sigma.powi(2)).exp();
                    }
                }
            }
        }
    }
    Ok(Projection {
        density: density
            .iter()
            .zip(mask)
            .map(|(&v, m)| v.max(0.) * m)
            .collect(),
        selected: peaks.len(),
        sigma: effective,
        omitted: omitted_count,
    })
}

fn validate_options(o: &DualSpaceOptions) -> Result<(), SolutionError> {
    if !(1..=20_000).contains(&o.cycles)
        || match o.peak_selection {
            PeakSelection::Fixed(n) => !(1..=128).contains(&n),
            PeakSelection::Adaptive { sigma, cap } => {
                !sigma.is_finite() || !(0.1..=10.).contains(&sigma) || !(1..=128).contains(&cap)
            }
        }
        || !o.mask_sigma.is_finite()
        || !(1e-4..=5.).contains(&o.mask_sigma)
        || !o.hybrid_m.is_finite()
        || !(1.0..=4.0).contains(&o.hybrid_m)
        || !o.omit_fraction.is_finite()
        || !(0.0..=0.3).contains(&o.omit_fraction)
        || o.patterson_rank.is_some_and(|v| v > 127)
    {
        return Err(error("invalid dual-space controls"));
    }
    Ok(())
}

struct PattersonStart {
    density: Vec<f64>,
    vector: [i32; 3],
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn patterson_start(
    shape: [usize; 3],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    rank: usize,
    fft: &mut Transform,
) -> Result<Option<PattersonStart>, SolutionError> {
    let n = grid_len(shape)?;
    let mut coefficients = vec![Complex::default(); n];
    for (&h, &a) in hkl.iter().zip(amplitudes) {
        coefficients[grid_index(h, shape)] = Complex::new(a * a, 0.);
    }
    fft.apply(&mut coefficients, false);
    let map: Vec<_> = coefficients.iter().map(|v| v.re / n as f64).collect();
    let peaks = extract_peaks(&map, shape, 128)?;
    let mut vectors = Vec::new();
    for peak in peaks {
        let vector = std::array::from_fn::<_, 3, _>(|i| (peak[i] * shape[i] as f64).round() as i32);
        let negative = vector.map(|v| -v);
        if grid_index(vector, shape) == 0 || grid_index(vector, shape) > grid_index(negative, shape)
        {
            continue;
        }
        let centring = hkl.iter().all(|h| {
            let phase: f64 = (0..3)
                .map(|i| f64::from(h[i]) * f64::from(vector[i]) / shape[i] as f64)
                .sum();
            (phase - phase.round()).abs() < 1e-9
        });
        if !centring {
            vectors.push(vector);
        }
    }
    if vectors.is_empty() {
        return Ok(None);
    }
    let vector = vectors[rank % vectors.len()];
    let initial = (0..n)
        .map(|i| {
            let p = [
                i / (shape[1] * shape[2]),
                (i / shape[2]) % shape[1],
                i % shape[2],
            ];
            let shifted = std::array::from_fn(|j| i32::try_from(p[j]).unwrap_or(0) - vector[j]);
            map[i].min(map[grid_index(shifted, shape)]).max(0.)
        })
        .collect();
    Ok(Some(PattersonStart {
        density: initial,
        vector,
    }))
}

/// Run published hybrid difference-Fourier recycling with a Gaussian atomicity mask.
///
/// # Errors
/// Rejects the shared Fourier boundary or bounded dual-space controls. Degenerate
/// iterates return an unsuccessful finite trial, never a fabricated success.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
pub fn dual_space(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
) -> Result<DualSpaceResult, SolutionError> {
    dual_space_monitored(
        shape, cell, hkl, amplitudes, phases, support, options, deadline, None,
    )
}

/// Dual-space iteration with a separately prepared predicted-reflection domain.
/// # Errors
/// Rejects incompatible prediction support, Fourier arrays or iteration controls.
#[allow(clippy::too_many_arguments)]
pub fn dual_space_with_prediction(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
    prediction: &crate::PreparedReflectionPrediction,
) -> Result<DualSpaceResult, SolutionError> {
    if prediction.audit.shape != shape
        || prediction.support != support
        || prediction.audit.measured_rows != hkl.len()
    {
        return Err(error("prediction plan does not match Fourier support"));
    }
    dual_space_guided_prediction(
        shape,
        cell,
        hkl,
        amplitudes,
        phases,
        support,
        options,
        deadline,
        None,
        None,
        None,
        Some(prediction),
        false,
        PhasingExecution::default(),
    )
}

/// Execution-only optimizations, independent of the numerical phasing options.
#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)] // Independent execution ablation controls.
pub struct PhasingExecution {
    /// Cache scalar merits; no gradient computation in scoring-only callers.
    pub score_only_evaluation: bool,
    /// Reuse full-grid buffers and process contiguous FFT lines in place.
    pub reusable_fourier_workspaces: bool,
    /// Filter impossible adaptive peaks before cached periodic neighbor comparisons.
    pub filtered_peak_search: bool,
    /// Batch contiguous FFT lines through blocked small-slab staging.
    pub batched_fft: bool,
    /// Packed real-density transforms for the two P1 cycle Fourier operations.
    pub real_fft: bool,
}

/// Diagnostic/benchmark entry point with explicit arithmetic-preserving execution.
/// # Errors
/// Rejects invalid phasing arrays and score-only requests without a quality plan.
#[allow(clippy::too_many_arguments)]
pub fn dual_space_with_execution(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
    quality: Option<&crate::PreparedPhaseQuality>,
    execution: PhasingExecution,
) -> Result<DualSpaceResult, SolutionError> {
    if execution.score_only_evaluation && quality.is_none() {
        return Err(error("score-only execution requires a quality plan"));
    }
    dual_space_guided_prediction(
        shape, cell, hkl, amplitudes, phases, support, options, deadline, None, None, quality,
        None, false, execution,
    )
}

/// A read-only window-map handoff; true means the caller's full target was met.
pub type CheckpointCallback<'a> =
    dyn FnMut(&[f64], usize, usize) -> Result<bool, SolutionError> + 'a;

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
/// Recycle Fourier iterates with optional complete-model handoffs at 4,16,64,... cycles.
/// Rejected handoffs resume the same trajectory; the cycle ceiling remains global.
/// # Errors
/// Propagates input validation, Fourier errors and errors from the caller's handoff.
pub fn dual_space_monitored(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
    checkpoint: Option<&mut CheckpointCallback<'_>>,
) -> Result<DualSpaceResult, SolutionError> {
    dual_space_guided(
        shape, cell, hkl, amplitudes, phases, support, options, deadline, checkpoint, None, None,
    )
}

/// Existing dual-space trajectory with optional measured-only phase relations.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
pub(crate) fn dual_space_guided(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
    checkpoint: Option<&mut CheckpointCallback<'_>>,
    tangent: Option<&crate::PreparedTangent>,
    quality: Option<&crate::PreparedPhaseQuality>,
) -> Result<DualSpaceResult, SolutionError> {
    dual_space_guided_prediction(
        shape,
        cell,
        hkl,
        amplitudes,
        phases,
        support,
        options,
        deadline,
        checkpoint,
        tangent,
        quality,
        None,
        false,
        PhasingExecution::default(),
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
pub(crate) fn dual_space_guided_prediction(
    shape: [usize; 3],
    cell: UnitCell,
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &DualSpaceOptions,
    deadline: Option<Instant>,
    mut checkpoint: Option<&mut CheckpointCallback<'_>>,
    tangent: Option<&crate::PreparedTangent>,
    quality: Option<&crate::PreparedPhaseQuality>,
    prediction: Option<&crate::PreparedReflectionPrediction>,
    adaptive_checkpoints: bool,
    execution: PhasingExecution,
) -> Result<DualSpaceResult, SolutionError> {
    let PhasingExecution {
        score_only_evaluation,
        reusable_fourier_workspaces,
        filtered_peak_search,
        batched_fft,
        real_fft,
    } = execution;
    validate_options(options)?;
    if (filtered_peak_search || batched_fft) && !reusable_fourier_workspaces {
        return Err(error(
            "filtered peak search and batched FFT require reusable Fourier workspaces",
        ));
    }
    if real_fft && !batched_fft {
        return Err(error("real FFT requires batched FFTs"));
    }
    let geometry = cell.geometry().map_err(|e| error(e.to_string()))?;
    mask_bounds(shape, &geometry, options.mask_sigma)?;
    let (indices, mut f) = prepare_fourier(shape, hkl, amplitudes, phases, support)?;
    if let Some(plan) = tangent {
        if !plan.matches_hkl(hkl) {
            return Err(error("tangent plan does not match measured reflections"));
        }
    }
    let n = grid_len(shape)?;
    let maximum = amplitudes.iter().copied().fold(0.0_f64, f64::max);
    let rms = (amplitudes
        .iter()
        .map(|a| (a / maximum).powi(2))
        .sum::<f64>()
        / amplitudes.len() as f64)
        .sqrt();
    let target: Vec<_> = amplitudes.iter().map(|a| (a / maximum) / rms).collect();
    for (&i, &a) in indices.iter().zip(&target) {
        f[i] = Complex::from_polar(a, f[i].arg());
    }
    let mut fft = if real_fft {
        Transform::real(shape)
    } else if batched_fft {
        Transform::batched(shape)
    } else if reusable_fourier_workspaces {
        Transform::reusable(shape)
    } else {
        Transform::new(shape)
    };
    let start = if let Some(rank) = options.patterson_rank {
        patterson_start(shape, hkl, &target, rank, &mut fft)?
    } else {
        None
    };
    let (mut density, initialization, vector) = if let Some(start) = start {
        (start.density, "patterson", Some(start.vector))
    } else {
        fft.apply(&mut f, false);
        (
            f.iter().map(|v| v.re / n as f64).collect(),
            if options.patterson_rank.is_some() {
                "random_no_patterson_vector"
            } else {
                "random"
            },
            None,
        )
    };
    let mut tangent_audit = tangent.map(crate::PreparedTangent::audit);
    let mut quality_trace = quality.map(crate::PreparedPhaseQuality::trace);
    let mut scalar_score = if score_only_evaluation {
        quality.map(crate::PreparedPhaseScore::new).transpose()?
    } else {
        None
    };
    let mut score_only_evaluations = 0;
    let mut fourier_workspace_cycles = 0;
    let quality_indices: Vec<_> = quality.map_or_else(Vec::new, |plan| {
        plan.hkl.iter().map(|&h| grid_index(h, shape)).collect()
    });
    let mut prediction_trace = Vec::new();
    let mut prediction_seconds = 0.;
    let mut history = Vec::with_capacity(options.cycles);
    let mut adaptive_history = match options.peak_selection {
        PeakSelection::Adaptive { sigma, cap } => Some(AdaptivePeakHistory {
            sigma,
            cap,
            selected: Vec::new(),
            thresholds: Vec::new(),
            omitted: Vec::new(),
        }),
        PeakSelection::Fixed(_) => None,
    };
    let mut window_best = Vec::new();
    let mut window_score = f64::INFINITY;
    let mut window_rank = f64::INFINITY;
    let mut window_cycle = 0;
    let mut amplitude_best = Vec::new();
    let mut amplitude_score = f64::INFINITY;
    let mut amplitude_cycle = 0;
    let mut phase_selector = crate::PhaseCheckpointSelector::default();
    let mut phase_checkpoints = Vec::new();
    let mut next_phase_checkpoint = 16;
    let mut next_checkpoint = 4;
    let mut checkpoints = Vec::new();
    let mut best = density.clone();
    let mut best_score = f64::INFINITY;
    let mut best_rank = f64::INFINITY;
    let mut best_cycle = 0;
    let target_norm: f64 = target.iter().map(|a| a * a).sum();
    let mut workspace = if reusable_fourier_workspaces {
        Some(AtomicWorkspace::new(
            shape,
            cell,
            options,
            filtered_peak_search,
        )?)
    } else {
        None
    };
    let mut recycled_coefficients = if reusable_fourier_workspaces {
        f
    } else {
        Vec::new()
    };
    let mut quality_magnitudes = Vec::new();
    let mut tangent_phases = Vec::new();
    let mut tangent_magnitudes = Vec::new();
    for cycle in 0..options.cycles {
        if crate::stopped(deadline, options.cancellation.as_ref()) {
            break;
        }
        let projection = if let Some(w) = &mut workspace {
            fourier_workspace_cycles += 1;
            let stats = w.update(&density, shape, options, cycle)?;
            Projection {
                density: std::mem::take(&mut w.projected),
                selected: stats.0,
                sigma: stats.1,
                omitted: stats.2,
            }
        } else {
            project_atomicity(&density, shape, cell, options, cycle)?
        };
        let mut projected = projection.density;
        if crate::stopped(deadline, options.cancellation.as_ref()) {
            break;
        }
        let mut calculated: Vec<_> = if reusable_fourier_workspaces {
            let mut values = std::mem::take(&mut recycled_coefficients);
            for (out, &v) in values.iter_mut().zip(&projected) {
                *out = Complex::new(v, 0.);
            }
            values
        } else {
            projected.iter().map(|&v| Complex::new(v, 0.)).collect()
        };
        fft.density_to_fourier(&mut calculated);
        if let Some(audit) = &mut adaptive_history {
            audit.selected.push(projection.selected);
            audit.thresholds.push(projection.sigma);
            audit.omitted.push(projection.omitted);
        }
        let norm: f64 = indices.iter().map(|&i| calculated[i].norm_sqr()).sum();
        if !norm.is_finite() || norm <= 1e-250 {
            history.push(1.);
            density = projected;
            break;
        }
        let scale = indices
            .iter()
            .zip(&target)
            .map(|(&i, &a)| calculated[i].norm() * a)
            .sum::<f64>()
            / norm;
        let score = indices
            .iter()
            .zip(&target)
            .map(|(&i, &a)| (scale * calculated[i].norm() - a).powi(2))
            .sum::<f64>()
            / target_norm;
        history.push(score);
        let quality_score = if let Some(plan) = quality {
            let mut evaluate = |magnitudes: &[f64]| {
                if let Some(scorer) = &mut scalar_score {
                    score_only_evaluations += 1;
                    scorer.evaluate(magnitudes)
                } else {
                    plan.evaluate(magnitudes)
                        .map(crate::PhaseQualityValue::from)
                }
            };
            Some(if reusable_fourier_workspaces {
                quality_magnitudes.clear();
                quality_magnitudes.extend(quality_indices.iter().map(|&i| calculated[i].norm()));
                evaluate(&quality_magnitudes)?
            } else {
                let magnitudes: Vec<_> = quality_indices
                    .iter()
                    .map(|&i| calculated[i].norm())
                    .collect();
                evaluate(&magnitudes)?
            })
        } else {
            None
        };

        if let (Some(q), Some(trace)) = (&quality_score, &mut quality_trace) {
            trace.correlation.push(q.correlation);
            trace.r_weak.push(q.r_weak);
            trace.cfom.push(q.cfom);
        }
        let ranking = quality_score.as_ref().map_or(score, |q| -q.cfom);
        let eligible = options.omit_fraction == 0. || cycle % 3 != 2;
        if checkpoint.is_some() && quality.is_some() && eligible && score < amplitude_score {
            amplitude_score = score;
            amplitude_best.clone_from(&projected);
            amplitude_cycle = cycle;
        }
        if checkpoint.is_some() && eligible && ranking < window_rank {
            window_rank = ranking;
            window_score = score;
            window_best.clone_from(&projected);
            window_cycle = cycle;
        }
        let phase_sample = if adaptive_checkpoints
            && (cycle + 1 == next_phase_checkpoint
                || cycle + 1 == options.cycles
                || (cycle + 1 >= 32
                    && best_rank.min(if eligible { ranking } else { f64::INFINITY }) <= -0.9))
        {
            Some(projected.clone())
        } else {
            None
        };
        if eligible
            && (cycle >= 50 || options.cycles <= 50 || quality.is_some())
            && ranking < best_rank
        {
            best_rank = ranking;
            best_score = score;
            if reusable_fourier_workspaces {
                std::mem::swap(&mut best, &mut projected);
            } else {
                best = std::mem::take(&mut projected);
            }
            best_cycle = cycle;
        }
        if let Some(sample) = phase_sample
            && best_score.is_finite()
        {
            let decision = phase_selector.consider(
                &sample,
                cycle + 1,
                -ranking,
                score,
                -best_rank,
                cycle + 1 == options.cycles,
            )?;
            let selected = decision.selected;
            phase_checkpoints.push(decision);
            next_phase_checkpoint *= 4;
            if selected {
                break;
            }
        }
        if !adaptive_checkpoints
            && quality.is_some()
            && checkpoint.is_none()
            && cycle + 1 >= 32
            && best_rank <= -0.9
        {
            break;
        }
        if let Some(accept) = &mut checkpoint {
            if (cycle + 1 == next_checkpoint || cycle + 1 == options.cycles)
                && window_score.is_finite()
            {
                // Preserve the amplitude candidate: P1 quality alone is not a
                // reliable proxy for recovery after supplied-symmetry recycling.
                // Only spend additional work on a distinct quality candidate if
                // the first model was not accepted and budget remains.
                if quality.is_some() {
                    let accepted = accept(&amplitude_best, cycle + 1, amplitude_cycle)?;
                    checkpoints.push(DualSpaceCheckpoint {
                        completed_cycles: cycle + 1,
                        best_cycle: amplitude_cycle,
                        score: amplitude_score,
                        accepted,
                    });
                    if accepted {
                        best = amplitude_best;
                        best_cycle = amplitude_cycle;
                        best_score = amplitude_score;
                        break;
                    }
                }
                if quality.is_none()
                    || (window_cycle != amplitude_cycle
                        && !crate::stopped(deadline, options.cancellation.as_ref()))
                {
                    if let Some(trace) = &mut quality_trace {
                        trace.alternative_handoffs += 1;
                    }
                    let accepted = accept(&window_best, cycle + 1, window_cycle)?;
                    checkpoints.push(DualSpaceCheckpoint {
                        completed_cycles: cycle + 1,
                        best_cycle: window_cycle,
                        score: window_score,
                        accepted,
                    });
                    if accepted {
                        best = window_best;
                        best_cycle = window_cycle;
                        best_score = window_score;
                        break;
                    }
                }
                window_score = f64::INFINITY;
                window_rank = f64::INFINITY;
                amplitude_score = f64::INFINITY;
                next_checkpoint *= 4;
            }
        }
        if let (Some(plan), Some(audit)) = (tangent, &mut tangent_audit) {
            let update = if reusable_fourier_workspaces {
                tangent_phases.clear();
                tangent_magnitudes.clear();
                tangent_phases.extend(
                    plan.hkl
                        .iter()
                        .map(|&h| calculated[grid_index(h, shape)].arg()),
                );
                tangent_magnitudes.extend(
                    plan.hkl
                        .iter()
                        .map(|&h| calculated[grid_index(h, shape)].norm()),
                );
                plan.step(&tangent_phases, &tangent_magnitudes, cycle, options.seed)?
            } else {
                let independent: Vec<_> = plan
                    .hkl
                    .iter()
                    .map(|&h| calculated[grid_index(h, shape)].arg())
                    .collect();
                let magnitudes: Vec<_> = plan
                    .hkl
                    .iter()
                    .map(|&h| calculated[grid_index(h, shape)].norm())
                    .collect();
                plan.step(&independent, &magnitudes, cycle, options.seed)?
            };
            audit.steps += 1;
            audit.annealed_steps += usize::from(update.temperature > 0.);
            audit.proposals += update.proposals;
            audit.changed += update.changed;
            audit.declined_steps += usize::from(update.blend == 0.);
            audit.quartet_before.push(update.quartet_before);
            audit.quartet_after.push(update.quartet_after);
            for (&index, term) in indices.iter().zip(&plan.full_terms) {
                calculated[index] = Complex::from_polar(
                    calculated[index].norm(),
                    f64::from(term.sign) * update.phases[term.index],
                );
            }
        }
        for (value, &code) in calculated.iter_mut().zip(support) {
            *value = if code == 0 {
                Complex::default()
            } else {
                *value * scale
            };
        }
        if let Some(plan) = prediction {
            let start = Instant::now();
            prediction_trace.push(plan.apply(&mut calculated, 1., cycle)?);
            prediction_seconds += start.elapsed().as_secs_f64();
        }
        for (&i, &a) in indices.iter().zip(&target) {
            let magnitude = calculated[i].norm();
            let phase = if magnitude > 0. {
                calculated[i] / magnitude
            } else {
                Complex::new(1., 0.)
            };
            calculated[i] = phase * (options.hybrid_m * a - (options.hybrid_m - 1.) * magnitude);
        }
        fft.fourier_to_density(&mut calculated);
        if reusable_fourier_workspaces {
            for (out, v) in density.iter_mut().zip(&calculated) {
                *out = v.re / n as f64;
            }
            recycled_coefficients = calculated;
            workspace.as_mut().expect("workspace enabled").projected = projected;
        } else {
            density = calculated.iter().map(|v| v.re / n as f64).collect();
        }
        if density.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite dual-space iterate"));
        }
    }
    if !best_score.is_finite() {
        best = density;
    }
    Ok(DualSpaceResult {
        trial: FlipResult {
            density: best,
            history,
            best_cycle,
            timed_out: deadline.is_some_and(|end| Instant::now() >= end),
            cancelled: options
                .cancellation
                .as_ref()
                .is_some_and(crate::Cancellation::is_cancelled),
        },
        initialization,
        patterson_vector: vector,
        adaptive_history,
        checkpoints,
        tangent: tangent_audit,
        quality: quality_trace,
        prediction: prediction_trace,
        prediction_seconds,
        phase_checkpoints,
        score_only_evaluations,
        fourier_workspace_cycles,
        batched_phase_transforms: fft.batched_calls(),
        real_phase_transforms: fft.real_calls(),
        filtered_peak_cycles: if filtered_peak_search
            && matches!(options.peak_selection, PeakSelection::Adaptive { .. })
        {
            fourier_workspace_cycles
        } else {
            0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::too_many_lines)]
    fn execution_modes_preserve_all_cycles_maps_and_scores_bitwise() {
        let shape = [16, 24, 8];
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 6.,
            c_angstrom: 7.,
            alpha_deg: 90.,
            beta_deg: 98.,
            gamma_deg: 90.,
        };
        let hkl: Vec<_> = (-2..=2)
            .flat_map(|h| {
                (-2..=2).flat_map(move |k| {
                    (-2..=2).filter_map(move |l| ([h, k, l] != [0; 3]).then_some([h, k, l]))
                })
            })
            .collect();
        let amplitude: Vec<_> = hkl
            .iter()
            .map(|h| {
                (2. + (0.7 * f64::from(h[0]) + 1.1 * f64::from(h[1]) + 0.3 * f64::from(h[2])).cos())
                    .sqrt()
            })
            .collect();
        let phases = crate::solver::initial_phases(&hkl, 19);
        let mut support = vec![0; 16 * 24 * 8];
        support[0] = 1;
        for &h in &hkl {
            support[grid_index(h, shape)] = 2;
        }
        let p = crate::PreparedPhaseQuality {
            hkl: hkl.clone(),
            target: amplitude.clone(),
            e_weights: vec![1.; hkl.len()],
            weak: (0..hkl.len()).map(|i| i % 10 == 0).collect(),
            allowed: hkl.len(),
            calculated_shell: Vec::new(),
        };
        for selection in [
            PeakSelection::Fixed(4),
            PeakSelection::Adaptive {
                sigma: 1.5,
                cap: 32,
            },
        ] {
            for rank in [None, Some(0)] {
                let options = DualSpaceOptions {
                    cycles: 64,
                    peak_selection: selection.clone(),
                    mask_sigma: 0.25,
                    hybrid_m: 3.,
                    omit_fraction: 0.1,
                    seed: 19,
                    patterson_rank: rank,
                    cancellation: None,
                };
                let old = dual_space_with_execution(
                    shape,
                    cell,
                    &hkl,
                    &amplitude,
                    &phases,
                    &support,
                    &options,
                    None,
                    Some(&p),
                    PhasingExecution::default(),
                )
                .unwrap();
                for execution in [
                    PhasingExecution {
                        score_only_evaluation: true,
                        reusable_fourier_workspaces: false,
                        filtered_peak_search: false,
                        batched_fft: false,
                        real_fft: false,
                    },
                    PhasingExecution {
                        score_only_evaluation: false,
                        reusable_fourier_workspaces: true,
                        filtered_peak_search: false,
                        batched_fft: false,
                        real_fft: false,
                    },
                    PhasingExecution {
                        score_only_evaluation: true,
                        reusable_fourier_workspaces: true,
                        filtered_peak_search: false,
                        batched_fft: false,
                        real_fft: false,
                    },
                    PhasingExecution {
                        score_only_evaluation: true,
                        reusable_fourier_workspaces: true,
                        filtered_peak_search: true,
                        batched_fft: false,
                        real_fft: false,
                    },
                    PhasingExecution {
                        score_only_evaluation: true,
                        reusable_fourier_workspaces: true,
                        filtered_peak_search: true,
                        batched_fft: true,
                        real_fft: false,
                    },
                ] {
                    let new = dual_space_with_execution(
                        shape,
                        cell,
                        &hkl,
                        &amplitude,
                        &phases,
                        &support,
                        &options,
                        None,
                        Some(&p),
                        execution,
                    )
                    .unwrap();
                    assert_eq!(old.trial.history, new.trial.history);
                    assert_eq!(old.trial.best_cycle, new.trial.best_cycle);
                    assert_eq!(
                        old.trial
                            .density
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        new.trial
                            .density
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        old.quality.as_ref().unwrap().cfom,
                        new.quality.as_ref().unwrap().cfom
                    );
                    assert_eq!(
                        new.score_only_evaluations,
                        if execution.score_only_evaluation {
                            new.trial.history.len()
                        } else {
                            0
                        }
                    );
                    assert_eq!(
                        new.fourier_workspace_cycles,
                        if execution.reusable_fourier_workspaces {
                            new.trial.history.len()
                        } else {
                            0
                        }
                    );
                }
            }
        }
    }
    #[test]
    #[allow(clippy::too_many_lines)]
    fn quality_checkpoints_preserve_early_amplitude_maps_and_offer_alternatives() {
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 5.,
            c_angstrom: 5.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let shape = [8; 3];
        let hkl: Vec<_> = (-2..=2)
            .flat_map(|h| {
                (-2..=2).flat_map(move |k| {
                    (-2..=2).filter_map(move |l| ([h, k, l] != [0; 3]).then_some([h, k, l]))
                })
            })
            .collect();
        let amplitudes: Vec<_> = hkl
            .iter()
            .map(|h| {
                (2. + (0.7 * f64::from(h[0]) + 1.1 * f64::from(h[1]) + 0.3 * f64::from(h[2])).cos())
                    .sqrt()
            })
            .collect();
        let phases = crate::solver::initial_phases(&hkl, 0);
        let mut support = vec![0; 512];
        support[0] = 1;
        for &h in &hkl {
            support[grid_index(h, shape)] = 2;
        }
        let options = DualSpaceOptions {
            cycles: 16,
            peak_selection: PeakSelection::Fixed(4),
            mask_sigma: 0.2,
            hybrid_m: 3.,
            omit_fraction: 0.1,
            seed: 0,
            patterson_rank: None,
            cancellation: None,
        };
        let plan = crate::PreparedPhaseQuality {
            hkl: hkl.clone(),
            target: amplitudes.clone(),
            e_weights: vec![1.; hkl.len()],
            weak: (0..hkl.len()).map(|i| i % 10 == 0).collect(),
            allowed: hkl.len(),
            calculated_shell: Vec::new(),
        };
        let mut legacy_maps = Vec::new();
        let mut legacy_callback = |map: &[f64], cycles, selected| {
            legacy_maps.push((cycles, selected, map.to_vec()));
            Ok(false)
        };
        let legacy = dual_space_guided(
            shape,
            cell,
            &hkl,
            &amplitudes,
            &phases,
            &support,
            &options,
            None,
            Some(&mut legacy_callback),
            None,
            None,
        )
        .unwrap();
        let mut quality_maps = Vec::new();
        let mut quality_callback = |map: &[f64], cycles, selected| {
            quality_maps.push((cycles, selected, map.to_vec()));
            Ok(false)
        };
        let quality = dual_space_guided(
            shape,
            cell,
            &hkl,
            &amplitudes,
            &phases,
            &support,
            &options,
            None,
            Some(&mut quality_callback),
            None,
            Some(&plan),
        )
        .unwrap();
        assert_eq!(legacy.trial.history, quality.trial.history);
        assert_eq!(legacy_maps[0].0, 4);
        for primary in &legacy_maps {
            assert_eq!(
                quality_maps.iter().find(|m| m.0 == primary.0).unwrap(),
                primary
            );
        }
        let trace = quality.quality.unwrap();
        assert_eq!(trace.deferred_handoffs, 0);
        assert_eq!(
            quality_maps.len(),
            legacy_maps.len() + trace.alternative_handoffs
        );
        for group in quality
            .checkpoints
            .chunk_by(|a, b| a.completed_cycles == b.completed_cycles)
        {
            assert!(group.len() <= 2);
            if group.len() == 2 {
                assert_ne!(group[0].best_cycle, group[1].best_cycle);
                assert!(trace.cfom[group[1].best_cycle] >= trace.cfom[group[0].best_cycle]);
            }
        }
    }
    #[test]
    fn centring_fallback_and_cooperative_stops_are_explicit() {
        let cell = UnitCell {
            a_angstrom: 5.,
            b_angstrom: 5.,
            c_angstrom: 5.,
            alpha_deg: 90.,
            beta_deg: 90.,
            gamma_deg: 90.,
        };
        let shape = [8; 3];
        let hkl = [[-2, 0, 0], [2, 0, 0]];
        let mut support = vec![0; 512];
        support[0] = 1;
        for h in hkl {
            support[grid_index(h, shape)] = 2;
        }
        let mut options = DualSpaceOptions {
            cycles: 2,
            peak_selection: PeakSelection::Fixed(1),
            mask_sigma: 0.2,
            hybrid_m: 3.,
            omit_fraction: 0.1,
            seed: 0,
            patterson_rank: Some(0),
            cancellation: None,
        };
        let deadline = Some(Instant::now());
        let stopped = dual_space(
            shape,
            cell,
            &hkl,
            &[1., 1.],
            &[0., 0.],
            &support,
            &options,
            deadline,
        )
        .unwrap();
        assert!(stopped.trial.timed_out);
        assert_eq!(stopped.trial.history, [] as [f64; 0]);
        assert_eq!(stopped.initialization, "random_no_patterson_vector");
        let token = crate::Cancellation::default();
        assert!(token.cancel());
        options.cancellation = Some(token);
        let stopped = dual_space(
            shape,
            cell,
            &hkl,
            &[1., 1.],
            &[0., 0.],
            &support,
            &options,
            None,
        )
        .unwrap();
        assert!(stopped.trial.cancelled);
        assert!(!stopped.trial.timed_out);
        assert_eq!(stopped.trial.history, [] as [f64; 0]);
    }
}
