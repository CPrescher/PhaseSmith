//! Published charge-flipping cycle, with deterministic periodic peak extraction.

use std::sync::Arc;
use std::time::Instant;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{SolutionError, error};

#[path = "fft_layout.rs"]
mod layout;
#[path = "real_fourier.rs"]
mod real;

/// Charge-flipping controls; delta is fixed within a trial.
#[derive(Clone, Debug)]
pub struct FlipOptions {
    /// Maximum number of phasing cycles.
    pub cycles: usize,
    /// Threshold in units of the initial real-map standard deviation.
    pub delta_sigma: f64,
    /// Fraction of weakest measured amplitudes receiving opposite pi/2 shifts.
    pub weak_fraction: f64,
    /// Number of final low-density-elimination cycles.
    pub cleanup_cycles: usize,
    /// Optional shared cooperative stop token.
    pub cancellation: Option<crate::Cancellation>,
}

/// A finite phasing trial, including unsuccessful trials.
#[derive(Clone, Debug)]
pub struct FlipResult {
    /// Cleaned map in arbitrary units, flattened in C xyz order.
    pub density: Vec<f64>,
    /// Normalized reciprocal-space misfit for every completed phasing cycle.
    pub history: Vec<f64>,
    /// Zero-based cycle selected by the density figure of merit.
    pub best_cycle: usize,
    /// True if the supplied wall-clock deadline interrupted this trial.
    pub timed_out: bool,
    /// Whether the shared cancellation token was requested.
    pub cancelled: bool,
}

pub(crate) struct Transform {
    shape: [usize; 3],
    direct_contiguous: bool,
    forward: [Arc<dyn Fft<f64>>; 3],
    inverse: [Arc<dyn Fft<f64>>; 3],
    line: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    batch: Vec<Complex<f64>>,
    batched_calls: usize,
    real: Option<real::RealTransform>,
    real_calls: usize,
}

impl Transform {
    pub(crate) fn new(shape: [usize; 3]) -> Self {
        let mut planner = FftPlanner::new();
        let forward = shape.map(|n| planner.plan_fft_forward(n));
        let inverse = shape.map(|n| planner.plan_fft_inverse(n));
        let scratch_len = forward
            .iter()
            .chain(&inverse)
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Self {
            shape,
            direct_contiguous: false,
            forward,
            inverse,
            line: vec![Complex::default(); *shape.iter().max().unwrap()],
            scratch: vec![Complex::default(); scratch_len],
            batch: Vec::new(),
            batched_calls: 0,
            real: None,
            real_calls: 0,
        }
    }

    pub(crate) fn reusable(shape: [usize; 3]) -> Self {
        let mut plan = Self::new(shape);
        plan.direct_contiguous = true;
        plan
    }

    pub(crate) fn batched(shape: [usize; 3]) -> Self {
        let mut plan = Self::reusable(shape);
        plan.batch = vec![Complex::default(); layout::BATCH_LINES * shape.iter().max().unwrap()];
        plan
    }

    pub(crate) fn real(shape: [usize; 3]) -> Self {
        let mut plan = Self::batched(shape);
        plan.real = Some(real::RealTransform::new(
            shape,
            &plan.forward,
            &plan.inverse,
        ));
        plan
    }
    pub(crate) fn real_calls(&self) -> usize {
        self.real_calls
    }
    pub(crate) fn density_to_fourier(&mut self, values: &mut [Complex<f64>]) {
        if let Some(plan) = &mut self.real {
            self.real_calls += 1;
            plan.density_to_spectrum(values);
        } else {
            self.apply(values, true);
        }
    }
    pub(crate) fn fourier_to_density(&mut self, values: &mut [Complex<f64>]) {
        if let Some(plan) = &mut self.real {
            self.real_calls += 1;
            plan.spectrum_to_density(values);
        } else {
            self.apply(values, false);
        }
    }
    pub(crate) fn density_slice_to_fourier(
        &mut self,
        density: &[f64],
        values: &mut [Complex<f64>],
    ) {
        let plan = self
            .real
            .as_mut()
            .expect("real FFT required for direct transfers");
        self.real_calls += 1;
        plan.density_slice_to_spectrum(density, values);
    }
    pub(crate) fn fourier_to_density_slice(
        &mut self,
        values: &[Complex<f64>],
        density: &mut [f64],
    ) {
        let plan = self
            .real
            .as_mut()
            .expect("real FFT required for direct transfers");
        self.real_calls += 1;
        plan.spectrum_to_density_slice(values, density);
    }
    pub(crate) fn batched_calls(&self) -> usize {
        self.batched_calls
    }

    pub(crate) fn apply(&mut self, values: &mut [Complex<f64>], inverse: bool) {
        let batched = !self.batch.is_empty();
        self.batched_calls += usize::from(batched);
        for axis in (0..3).rev() {
            let length = self.shape[axis];
            let stride: usize = self.shape[axis + 1..].iter().product();
            let block = length * stride;
            let plan = if inverse {
                &self.inverse[axis]
            } else {
                &self.forward[axis]
            };
            if batched {
                if stride == 1 {
                    plan.process_with_scratch(values, &mut self.scratch);
                } else {
                    for outer in 0..values.len() / block {
                        for inner in (0..stride).step_by(layout::BATCH_LINES) {
                            let lanes = layout::BATCH_LINES.min(stride - inner);
                            let offset = outer * block + inner;
                            let batch = &mut self.batch[..lanes * length];
                            layout::pack(values, batch, offset, stride, length, lanes);
                            plan.process_with_scratch(batch, &mut self.scratch);
                            layout::unpack(batch, values, offset, stride, length, lanes);
                        }
                    }
                }
                continue;
            }
            if self.direct_contiguous && stride == 1 {
                for line in values.chunks_exact_mut(length) {
                    plan.process_with_scratch(line, &mut self.scratch);
                }
                continue;
            }
            for outer in 0..values.len() / block {
                for inner in 0..stride {
                    let offset = outer * block + inner;
                    for i in 0..length {
                        self.line[i] = values[offset + i * stride];
                    }
                    plan.process_with_scratch(&mut self.line[..length], &mut self.scratch);
                    for i in 0..length {
                        values[offset + i * stride] = self.line[i];
                    }
                }
            }
        }
    }
}

/// Unnormalized complex 3D transform in C-order XYZ layout, for independent checks.
/// `inverse=true` uses the positive-exponent transform; a round trip multiplies by N.
/// # Errors
/// Rejects invalid Fourier grid bounds, length mismatch or nonfinite complex inputs.
pub fn transform_3d(
    shape: [usize; 3],
    values: &[Complex<f64>],
    inverse: bool,
    batched: bool,
) -> Result<Vec<Complex<f64>>, SolutionError> {
    if values.len() != grid_len(shape)?
        || values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(error("invalid complex transform input"));
    }
    let mut plan = if batched {
        Transform::batched(shape)
    } else {
        Transform::reusable(shape)
    };
    let mut output = values.to_vec();
    plan.apply(&mut output, inverse);
    Ok(output)
}

/// Positive DFT of a real density, or real part of a negative complex DFT.
/// Both directions remain unnormalized; the latter accepts arbitrary complex input.
/// # Errors
/// Rejects the original grid/array bounds and non-real density input.
pub fn real_density_transform(
    shape: [usize; 3],
    values: &[Complex<f64>],
    to_density: bool,
) -> Result<Vec<Complex<f64>>, SolutionError> {
    if values.len() != grid_len(shape)?
        || values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite() || (!to_density && v.im != 0.))
    {
        return Err(error("invalid real-density transform input"));
    }
    let mut fft = Transform::real(shape);
    let mut out = values.to_vec();
    if to_density {
        fft.fourier_to_density(&mut out);
    } else {
        fft.density_to_fourier(&mut out);
    }
    Ok(out)
}

pub(crate) fn grid_len(shape: [usize; 3]) -> Result<usize, SolutionError> {
    if shape.iter().any(|&n| !(8..=128).contains(&n) || n % 2 != 0) {
        return Err(error("grid axes must be even and lie in [8,128]"));
    }
    let count: usize = shape.iter().product();
    if count > 2_097_152 {
        return Err(error("grid exceeds 2097152 voxels"));
    }
    Ok(count)
}

pub(crate) fn grid_index(hkl: [i32; 3], shape: [usize; 3]) -> usize {
    let h = std::array::from_fn::<_, 3, _>(|i| {
        #[allow(clippy::cast_possible_wrap)]
        let n = i32::try_from(shape[i]).expect("validated grid axis <=128");
        usize::try_from(hkl[i].rem_euclid(n)).expect("nonnegative grid residue")
    });
    (h[0] * shape[1] + h[1]) * shape[2] + h[2]
}

/// Validate the shared Fourier boundary and construct owned Hermitian coefficients.
#[allow(clippy::too_many_lines, clippy::naive_bytecount)]
pub(crate) fn prepare_fourier(
    shape: [usize; 3],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
) -> Result<(Vec<usize>, Vec<Complex<f64>>), SolutionError> {
    let n = grid_len(shape)?;
    if hkl.is_empty()
        || hkl.len() != amplitudes.len()
        || hkl.len() != phases.len()
        || hkl.len() > 100_000
        || support.len() != n
        || support[0] != 1
        || support.iter().any(|&v| v > 2)
        || hkl.windows(2).any(|w| w[0] >= w[1])
        || amplitudes
            .iter()
            .any(|a| !a.is_finite() || *a < 0.0 || *a > 1e100)
        || phases.iter().any(|p| !p.is_finite())
    {
        return Err(error("invalid Fourier arrays or support"));
    }
    let sum_a: f64 = amplitudes.iter().sum();
    if sum_a <= 0.0 || !sum_a.is_finite() {
        return Err(error("positive finite amplitude sum required"));
    }
    let mut f = vec![Complex::default(); n];
    let mut indices = Vec::with_capacity(hkl.len());
    for ((&h, &a), &phase) in hkl.iter().zip(amplitudes).zip(phases) {
        if h == [0; 3]
            || (0..3).any(|i| {
                i64::from(h[i]).abs() * 2 >= i64::from(u16::try_from(shape[i]).unwrap_or(0))
            })
        {
            return Err(error(
                "measured HKL aliases the origin or a Nyquist/grid boundary",
            ));
        }
        let idx = grid_index(h, shape);
        if support[idx] != 2 {
            return Err(error("every measured HKL must have support code 2"));
        }
        indices.push(idx);
        f[idx] = Complex::from_polar(a, phase);
    }
    if support.iter().filter(|&&v| v == 2).count() != hkl.len() {
        return Err(error(
            "measured support must match the supplied HKLs exactly",
        ));
    }
    for (index, &code) in support.iter().enumerate() {
        let p = [
            index / (shape[1] * shape[2]),
            (index / shape[2]) % shape[1],
            index % shape[2],
        ];
        let negative = p.map(|v| i32::try_from(v).map_or(0, |v| -v));
        if support[grid_index(negative, shape)] != code {
            return Err(error("reciprocal support must preserve Friedel symmetry"));
        }
    }
    for (i, &h) in hkl.iter().enumerate() {
        let negative = h.map(|x| -x);
        let j = hkl
            .binary_search(&negative)
            .map_err(|_| error("missing Friedel mate"))?;
        if (f[indices[i]] - f[indices[j]].conj()).norm() > 1e-10 * (1.0 + amplitudes[i]) {
            return Err(error(
                "starting coefficients must satisfy Friedel/Hermitian symmetry",
            ));
        }
    }
    Ok((indices, f))
}

/// Run charge flipping with explicit initial phases and a reciprocal support mask.
///
/// `support` is 0 for fixed zeros (outside sphere or known absences), 1 for free
/// missing coefficients, and 2 for measured coefficients. 000 must be free.
/// Input must contain both Friedel mates; their amplitudes and phases must agree.
///
/// # Errors
/// Rejects aliased/duplicate/unsorted HKLs, non-Hermitian inputs, invalid mask,
/// zero amplitudes everywhere, nonfinite values, or controls beyond fixed bounds.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::naive_bytecount // Dependency-free bounded mask validation.
)]
pub fn charge_flip(
    shape: [usize; 3],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    phases: &[f64],
    support: &[u8],
    options: &FlipOptions,
    deadline: Option<Instant>,
) -> Result<FlipResult, SolutionError> {
    let n = grid_len(shape)?;
    if !(1..=20_000).contains(&options.cycles)
        || !(0.0..=0.5).contains(&options.weak_fraction)
        || !options.weak_fraction.is_finite()
        || !options.delta_sigma.is_finite()
        || !(0.1..=3.0).contains(&options.delta_sigma)
        || options.cleanup_cycles > 100
    {
        return Err(error("invalid charge-flipping controls"));
    }
    let (indices, mut f) = prepare_fourier(shape, hkl, amplitudes, phases, support)?;
    let sum_a: f64 = amplitudes.iter().sum();
    // Ties all receive the same weak treatment, preserving Friedel symmetry.
    let mut ordered = amplitudes.to_vec();
    ordered.sort_by(f64::total_cmp);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let weak_count = (options.weak_fraction * hkl.len() as f64).floor() as usize;
    let weak_limit = if weak_count == 0 {
        -1.0
    } else {
        ordered[weak_count - 1]
    };
    let weak: Vec<_> = amplitudes.iter().map(|&a| a <= weak_limit).collect();
    let mut fft = Transform::new(shape);
    fft.apply(&mut f, false);
    let mut rho: Vec<_> = f.iter().map(|v| v.re / n as f64).collect();
    let delta = options.delta_sigma * (rho.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let mut best = rho.clone();
    let mut best_score = f64::INFINITY;
    let mut best_cycle = 0;
    let mut history = Vec::with_capacity(options.cycles);
    let mut work = vec![Complex::default(); n];
    let mut timed_out = false;
    for cycle in 0..options.cycles {
        if crate::stopped(deadline, options.cancellation.as_ref()) {
            timed_out = true;
            break;
        }
        for (g, &r) in work.iter_mut().zip(&rho) {
            *g = Complex::new(if r < delta { -r } else { r }, 0.0);
        }
        fft.apply(&mut work, true);
        let total_g: f64 = indices.iter().map(|&i| work[i].norm()).sum();
        let score: f64 = if total_g > 0.0 {
            indices
                .iter()
                .zip(amplitudes)
                .map(|(&i, &a)| (a / sum_a - work[i].norm() / total_g).abs())
                .sum()
        } else {
            2.0
        };
        history.push(score);
        // Exclude the initial transient when the budget permits it.
        if (cycle >= 20 || options.cycles <= 20) && score < best_score {
            best_score = score;
            best.clone_from(&rho);
            best_cycle = cycle;
        }
        constrain(&mut work, support, &indices, hkl, amplitudes, &weak, true);
        fft.apply(&mut work, false);
        for (r, g) in rho.iter_mut().zip(&work) {
            *r = g.re / n as f64;
        }
        if rho.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite charge-flipping iterate"));
        }
    }
    if !best_score.is_finite() {
        best = rho;
    }
    for _ in 0..options.cleanup_cycles {
        if crate::stopped(deadline, options.cancellation.as_ref()) {
            timed_out = true;
            break;
        }
        for (g, &r) in work.iter_mut().zip(&best) {
            *g = Complex::new(if r < delta { 0.0 } else { r }, 0.0);
        }
        fft.apply(&mut work, true);
        constrain(&mut work, support, &indices, hkl, amplitudes, &weak, false);
        fft.apply(&mut work, false);
        for (r, g) in best.iter_mut().zip(&work) {
            *r = g.re / n as f64;
        }
    }
    Ok(FlipResult {
        density: best,
        history,
        best_cycle,
        timed_out: timed_out && deadline.is_some_and(|end| Instant::now() >= end),
        cancelled: options
            .cancellation
            .as_ref()
            .is_some_and(crate::Cancellation::is_cancelled),
    })
}

fn constrain(
    work: &mut [Complex<f64>],
    support: &[u8],
    indices: &[usize],
    hkl: &[[i32; 3]],
    amplitudes: &[f64],
    weak: &[bool],
    perturb: bool,
) {
    for (value, &code) in work.iter_mut().zip(support) {
        if code == 0 {
            *value = Complex::default();
        }
    }
    for (row, &idx) in indices.iter().enumerate() {
        let value = &mut work[idx];
        if perturb && weak[row] {
            let sign = if hkl[row] > hkl[row].map(|x| -x) {
                1.0
            } else {
                -1.0
            };
            *value *= Complex::new(0.0, sign);
        } else {
            let magnitude = value.norm();
            *value = if magnitude > 0.0 {
                *value * (amplitudes[row] / magnitude)
            } else {
                Complex::new(amplitudes[row], 0.0)
            };
        }
    }
}

/// Select periodic strict/local maxima, with index-stable plateau ties.
///
/// # Errors
/// Rejects malformed/nonfinite maps or atom budgets outside [1,128]. Returns
/// fewer than `count` points when the map has too few positive local maxima.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
pub fn extract_peaks(
    density: &[f64],
    shape: [usize; 3],
    count: usize,
) -> Result<Vec<[f64; 3]>, SolutionError> {
    if density.len() != grid_len(shape)?
        || density.iter().any(|v| !v.is_finite())
        || !(1..=128).contains(&count)
    {
        return Err(error("invalid density map or atom count"));
    }
    let mut maxima = local_maxima(density, shape)?;
    maxima.truncate(count);
    Ok(refine_peak_indices(&maxima, density, shape))
}

#[allow(clippy::cast_possible_wrap, clippy::float_cmp)]
pub(crate) fn local_maxima(
    density: &[f64],
    shape: [usize; 3],
) -> Result<Vec<usize>, SolutionError> {
    if density.len() != grid_len(shape)? || density.iter().any(|v| !v.is_finite()) {
        return Err(error("invalid density map"));
    }
    let mut maxima = Vec::new();
    for i in 0..density.len() {
        if density[i] <= 0.0 {
            continue;
        }
        let p = [
            i / (shape[1] * shape[2]),
            (i / shape[2]) % shape[1],
            i % shape[2],
        ];
        let h = p.map(|x| i32::try_from(x).expect("validated grid axis <= 128"));
        let mut is_maximum = true;
        'neighbours: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if [dx, dy, dz] == [0; 3] {
                        continue;
                    }
                    let j = grid_index([h[0] + dx, h[1] + dy, h[2] + dz], shape);
                    if density[j] > density[i] || (density[j] == density[i] && j < i) {
                        is_maximum = false;
                        break 'neighbours;
                    }
                }
            }
        }
        if is_maximum {
            maxima.push(i);
        }
    }
    maxima.sort_by(|&i, &j| density[j].total_cmp(&density[i]).then(i.cmp(&j)));
    Ok(maxima)
}
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
pub(crate) fn refine_peak_indices(
    maxima: &[usize],
    density: &[f64],
    shape: [usize; 3],
) -> Vec<[f64; 3]> {
    maxima
        .iter()
        .copied()
        .map(|i| {
            let p = [
                i / (shape[1] * shape[2]),
                (i / shape[2]) % shape[1],
                i % shape[2],
            ];
            std::array::from_fn(|axis| {
                let mut left = p.map(|x| x as i32);
                left[axis] -= 1;
                let mut right = p.map(|x| x as i32);
                right[axis] += 1;
                let a = density[grid_index(left, shape)];
                let b = density[i];
                let c = density[grid_index(right, shape)];
                let denominator = a - 2.0 * b + c;
                let offset = if denominator < 0.0 {
                    (0.5 * (a - c) / denominator).clamp(-0.5, 0.5)
                } else {
                    0.0
                };
                ((p[axis] as f64 + offset) / shape[axis] as f64).rem_euclid(1.0)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn batched_fft_layout_preserves_every_complex_value_bitwise() {
        for shape in [[8; 3], [16, 24, 8], [32, 16, 64], [8, 128, 8]] {
            let n = grid_len(shape).unwrap();
            let values: Vec<_> = (0..n)
                .map(|i| Complex::new((i as f64 * 0.137).sin(), (i as f64 * 0.371).cos()))
                .collect();
            for inverse in [false, true] {
                let old = transform_3d(shape, &values, inverse, false).unwrap();
                let new = transform_3d(shape, &values, inverse, true).unwrap();
                assert_eq!(
                    old.iter()
                        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                        .collect::<Vec<_>>(),
                    new.iter()
                        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                        .collect::<Vec<_>>(),
                    "shape={shape:?},inverse={inverse}"
                );
            }
        }
    }
    #[test]
    fn periodic_peak_and_plateau_ties() {
        let mut map = vec![0.0; 512];
        map[0] = 4.;
        map[7] = 1.;
        map[1] = 3.;
        let points = extract_peaks(&map, [8; 3], 1).unwrap();
        assert!((points[0][2] - 0.03125).abs() < 1e-15);
        assert_eq!(extract_peaks(&vec![1.; 512], [8; 3], 10).unwrap().len(), 1);
        assert!(extract_peaks(&vec![f64::NAN; 512], [8; 3], 1).is_err());
    }
}
