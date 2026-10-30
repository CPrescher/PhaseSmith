//! Exact adaptive peak selection with a threshold filter and cached periodic offsets.
use crate::fourier::{grid_len, refine_peak_indices};
use crate::{SolutionError, error};

/// Reusable index geometry; storage scales with axis lengths, not voxel neighbors.
pub struct PreparedAdaptivePeaks {
    shape: [usize; 3],
    offsets: [Vec<[usize; 3]>; 3],
    maxima: Vec<usize>,
}
impl PreparedAdaptivePeaks {
    /// Prepare the existing even-axis [8,128], at-most-2097152-voxel domain.
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
        })
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
        self.maxima.clear();
        // Keep the original arithmetic and reduction order, including extreme scales.
        let maximum = density.iter().map(|v| v.abs()).fold(0., f64::max);
        if maximum == 0. {
            return Ok((Vec::new(), sigma));
        }
        let rms = maximum
            * (density.iter().map(|v| (v / maximum).powi(2)).sum::<f64>() / density.len() as f64)
                .sqrt();
        let cutoff = sigma * rms;
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
        self.maxima
            .sort_by(|&i, &j| density[j].total_cmp(&density[i]).then(i.cmp(&j)));
        let mut effective = sigma;
        loop {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn periodic_ties_threshold_raising_and_scales_are_bitwise_identical() {
        for shape in [[8; 3], [16, 24, 8], [32, 16, 32]] {
            let n = grid_len(shape).unwrap();
            let mut plan = PreparedAdaptivePeaks::new(shape).unwrap();
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
}
