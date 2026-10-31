//! Exact adaptive peak selection with a threshold filter and cached periodic offsets.
use crate::fourier::{grid_len, refine_peak_indices};
use crate::{SolutionError, error};

/// Reusable index geometry; storage scales with axis lengths, not voxel neighbors.
pub struct PreparedAdaptivePeaks {
    shape: [usize; 3],
    offsets: [Vec<[usize; 3]>; 3],
    maxima: Vec<usize>,
    streamed: bool,
    exact_rms_fallbacks: usize,
    streamed_calls: usize,
}
impl PreparedAdaptivePeaks {
    /// Prepare the existing even-axis `[8,128]`, at-most-2097152-voxel domain.
    /// # Errors
    /// Rejects the same grid boundaries as the original Fourier routines.
    pub fn new(shape: [usize; 3]) -> Result<Self, SolutionError> {
        grid_len(shape)?;
        let offsets = std::array::from_fn(|axis| {
            let n = shape[axis];
            let stride: usize = shape[axis + 1..].iter().product();
            (0..n)
                .map(|p| {
                    [
                        if p == 0 { n - 1 } else { p - 1 },
                        p,
                        if p + 1 == n { 0 } else { p + 1 },
                    ]
                    .map(|v| v * stride)
                })
                .collect()
        });
        Ok(Self {
            shape,
            offsets,
            maxima: Vec::new(),
            streamed: false,
            exact_rms_fallbacks: 0,
            streamed_calls: 0,
        })
    }

    /// Stream threshold candidates before loading periodic neighbor geometry.
    #[must_use]
    pub fn with_streamed_scan(mut self, enabled: bool) -> Self {
        self.streamed = enabled;
        self
    }

    /// Actual valid adaptive selections using streamed statistics.
    #[must_use]
    pub fn streamed_calls(&self) -> usize {
        self.streamed_calls
    }

    /// Actual guarded calls that recomputed the original scalar RMS.
    #[must_use]
    pub fn exact_rms_fallbacks(&self) -> usize {
        self.exact_rms_fallbacks
    }

    #[allow(clippy::float_cmp)] // The existing tie rule is exact equality.
    fn scan_streamed(&mut self, density: &[f64], cutoff: f64) {
        let nz = self.shape[2];
        for x in &self.offsets[0] {
            for y in &self.offsets[1] {
                let base = x[1] + y[1];
                for (column, &value) in density[base..base + nz].iter().enumerate() {
                    if value <= 0. || value < cutoff {
                        continue;
                    }
                    let i = base + column;
                    let z = self.offsets[2][column];
                    // Most shoulders can be rejected using contiguous central-row neighbors.
                    let loses = |j: usize| density[j] > value || (density[j] == value && j < i);
                    if loses(base + z[0]) || loses(base + z[2]) {
                        continue;
                    }
                    let mut is_maximum = true;
                    'neighbors: for (a, &plane) in x.iter().enumerate() {
                        for (b, &row) in y.iter().enumerate() {
                            if a == 1 && b == 1 {
                                continue;
                            }
                            for &offset in &z {
                                if loses(plane + row + offset) {
                                    is_maximum = false;
                                    break 'neighbors;
                                }
                            }
                        }
                    }
                    if is_maximum {
                        self.maxima.push(i);
                    }
                }
            }
        }
    }

    /// Select exactly the original positive maxima, threshold and refined coordinates.
    /// Voxels below the initial RMS threshold cannot survive any later raised threshold.
    /// Neighbor comparisons still include all 26 neighbors and the original index tie rule.
    /// # Errors
    /// Rejects malformed/nonfinite density or sigma/cap outside the original bounds.
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    pub fn select(
        &mut self,
        density: &[f64],
        sigma: f64,
        cap: usize,
    ) -> Result<(Vec<[f64; 3]>, f64), SolutionError> {
        if density.len() != self.shape.iter().product::<usize>()
            || density.iter().any(|v| !v.is_finite())
            || !sigma.is_finite()
            || !(0.1..=10.).contains(&sigma)
            || !(1..=128).contains(&cap)
        {
            return Err(error("invalid filtered peak map or controls"));
        }
        self.streamed_calls += usize::from(self.streamed);
        self.maxima.clear();
        // Keep the original arithmetic and reduction order, including extreme scales.
        let maximum = density.iter().map(|v| v.abs()).fold(0., f64::max);
        if maximum == 0. {
            return Ok((Vec::new(), sigma));
        }
        let mut exact = !self.streamed;
        let mut rms = if exact {
            scalar_rms(density, maximum)
        } else {
            lane_rms(density, maximum)
        };
        let cutoff = if exact {
            sigma * rms
        } else if let Some(bounds) = rms_bounds(rms, sigma, density.len()) {
            bounds[0]
        } else {
            rms = scalar_rms(density, maximum);
            exact = true;
            self.exact_rms_fallbacks += 1;
            sigma * rms
        };
        if self.streamed {
            self.scan_streamed(density, cutoff);
        } else {
            for x in &self.offsets[0] {
                for y in &self.offsets[1] {
                    for z in &self.offsets[2] {
                        let i = x[1] + y[1] + z[1];
                        let value = density[i];
                        if value <= 0. || value < cutoff {
                            continue;
                        }
                        let mut is_maximum = true;
                        'neighbors: for (a, &plane) in x.iter().enumerate() {
                            for (b, &row) in y.iter().enumerate() {
                                for (c, &column) in z.iter().enumerate() {
                                    if [a, b, c] == [1; 3] {
                                        continue;
                                    }
                                    let j = plane + row + column;
                                    if density[j] > value || (density[j] == value && j < i) {
                                        is_maximum = false;
                                        break 'neighbors;
                                    }
                                }
                            }
                        }
                        if is_maximum {
                            self.maxima.push(i);
                        }
                    }
                }
            }
        }
        self.maxima
            .sort_by(|&i, &j| density[j].total_cmp(&density[i]).then(i.cmp(&j)));
        let mut effective = sigma;
        loop {
            if !exact {
                let uncertain = rms_bounds(rms, effective, density.len()).is_none_or(|[lo, hi]| {
                    self.maxima
                        .iter()
                        .any(|&i| density[i] >= lo && density[i] <= hi)
                });
                if uncertain {
                    rms = scalar_rms(density, maximum);
                    exact = true;
                    self.exact_rms_fallbacks += 1;
                }
            }
            let count = self
                .maxima
                .iter()
                .take_while(|&&i| density[i] >= effective * rms)
                .count();
            if count <= cap {
                return Ok((
                    refine_peak_indices(&self.maxima[..count], density, self.shape),
                    effective,
                ));
            }
            effective *= 1.5;
        }
    }
}

#[allow(clippy::cast_precision_loss)] // grid_len bounds N by 2^21.
fn scalar_rms(density: &[f64], maximum: f64) -> f64 {
    maximum
        * (density.iter().map(|v| (v / maximum).powi(2)).sum::<f64>() / density.len() as f64).sqrt()
}
#[allow(clippy::cast_precision_loss)]
fn lane_rms(density: &[f64], maximum: f64) -> f64 {
    let mut sums = [0.; 8];
    let mut chunks = density.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for i in 0..8 {
            let value = chunk[i] / maximum;
            sums[i] += value * value;
        }
    }
    let mut sum = sums.into_iter().sum::<f64>();
    for &value in chunks.remainder() {
        sum += (value / maximum).powi(2);
    }
    maximum * (sum / density.len() as f64).sqrt()
}
// Nonnegative, identical terms <=1, with at least one term exactly 1. The
// original N-term and eight-lane sums, division, sqrt and final multiplications
// fit well inside this conservative 32*N*epsilon envelope (N<=2^21).
// Outward endpoints and scalar fallback cover subnormal/overflow thresholds.
#[allow(clippy::cast_precision_loss)]
fn rms_bounds(rms: f64, sigma: f64, count: usize) -> Option<[f64; 2]> {
    let cutoff = sigma * rms;
    let radius = (32. * count as f64 + 128.) * f64::EPSILON;
    let lower = cutoff * (1. - radius);
    let upper = cutoff * (1. + radius);
    if !cutoff.is_normal() || !lower.is_normal() || !upper.is_finite() {
        return None;
    }
    // Positive IEEE-754 values have monotone bit patterns. Step outward by one
    // ULP, retaining the project's Rust 1.85 minimum (before next_up/next_down).
    let lo = f64::from_bits(lower.to_bits() - 1);
    let hi = f64::from_bits(upper.to_bits() + 1);
    (lo.is_normal() && hi.is_finite()).then_some([lo, hi])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn periodic_ties_threshold_raising_and_scales_are_bitwise_identical() {
        for shape in [[8; 3], [16, 24, 8], [32, 16, 32]] {
            let n = grid_len(shape).unwrap();
            let mut plan = PreparedAdaptivePeaks::new(shape)
                .unwrap()
                .with_streamed_scan(true);
            for scale in [1e-250, 1., 1e250] {
                #[allow(clippy::cast_precision_loss)]
                let density: Vec<_> = (0..n)
                    .map(|i| {
                        if i % 23 == 0 {
                            scale
                        } else {
                            scale * (i as f64 * 0.73).sin()
                        }
                    })
                    .collect();
                for sigma in [0.1, 1.5, 10.] {
                    for cap in [1, 8, 128] {
                        let old = crate::adaptive_peaks(&density, shape, sigma, cap).unwrap();
                        let new = plan.select(&density, sigma, cap).unwrap();
                        assert_eq!(old.1.to_bits(), new.1.to_bits());
                        assert_eq!(
                            old.0
                                .iter()
                                .flatten()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            new.0
                                .iter()
                                .flatten()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn conservative_intervals_enclose_original_order_at_all_grid_scales() {
        for shape in [[8; 3], [10, 14, 18], [32, 16, 64], [128; 3]] {
            let n = grid_len(shape).unwrap();
            for scale in [1e-300, 1., 1e300] {
                let density: Vec<_> = (0..n).map(|i| scale * (i as f64 * 0.713).sin()).collect();
                let maximum = density.iter().map(|v| v.abs()).fold(0., f64::max);
                let a = scalar_rms(&density, maximum);
                let b = lane_rms(&density, maximum);
                for sigma in [0.1, 1., 10., 1000.] {
                    if let Some([lo, hi]) = rms_bounds(b, sigma, n) {
                        assert!(lo <= sigma * a && sigma * a <= hi);
                    }
                }
            }
        }
    }
    #[test]
    fn tied_and_subnormal_maps_exercise_scalar_fallback() {
        for scale in [f64::from_bits(1), 1e-310, 1., 1e308] {
            for sigma in [1. - f64::EPSILON, 1., 1. + f64::EPSILON, 10.] {
                let density = vec![scale; 512];
                let mut plan = PreparedAdaptivePeaks::new([8; 3])
                    .unwrap()
                    .with_streamed_scan(true);
                let a = plan.select(&density, sigma, 1).unwrap();
                let b = crate::adaptive_peaks(&density, [8; 3], sigma, 1).unwrap();
                assert_eq!(a, b);
                assert_eq!(plan.streamed_calls(), 1);
                if sigma < 2. || scale < f64::MIN_POSITIVE || scale > 1e307 {
                    assert_eq!(plan.exact_rms_fallbacks(), 1);
                }
            }
        }
    }
}
