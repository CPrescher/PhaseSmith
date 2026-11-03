//! Exact axis-factorized Gaussian mask geometry; no changed support or reduction order.
use crate::solution::math3::dot;
use phasesmith_crystallography::CellGeometry;

#[derive(Clone, Copy, Default)]
struct AxisSample {
    delta: f64,
    index: usize,
    products: [f64; 3],
}

pub(super) struct PreparedMask {
    shape: [usize; 3],
    metric: [[f64; 3]; 3],
    bounds: [i32; 3],
    radius2: f64,
    normalization: f64,
    sigma2: f64,
    axes: [[AxisSample; 33]; 3],
}
impl PreparedMask {
    pub(super) fn new(
        shape: [usize; 3],
        geometry: &CellGeometry,
        bounds: [i32; 3],
        sigma: f64,
    ) -> Self {
        Self {
            shape,
            metric: geometry.direct_metric,
            bounds,
            radius2: (3. * sigma).powi(2) * (1. + 1e-12),
            normalization: std::f64::consts::TAU.powf(-1.5) / sigma.powi(3),
            sigma2: sigma.powi(2),
            axes: [[AxisSample::default(); 33]; 3],
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(super) fn add_peak(&mut self, mask: &mut [f64], point: [f64; 3]) {
        let centre = std::array::from_fn::<_, 3, _>(|i| point[i] * self.shape[i] as f64);
        let origin = centre.map(|v| v.round() as i32);
        // Original support budget is at most 16 steps on either side of an axis.
        for axis in 0..3 {
            let stride: usize = self.shape[axis + 1..].iter().product();
            let axis_length = i32::try_from(self.shape[axis]).expect("validated axis at most 128");
            for (sample, d) in self.axes[axis]
                .iter_mut()
                .zip(-self.bounds[axis]..=self.bounds[axis])
            {
                let coordinate = origin[axis] + d;
                sample.delta = (f64::from(coordinate) - centre[axis]) / self.shape[axis] as f64;
                sample.index = coordinate.rem_euclid(axis_length) as usize * stride;
                sample.products = self.metric.map(|row| row[axis] * sample.delta);
            }
        }
        let lengths = self.bounds.map(|b| (2 * b + 1) as usize);
        for x in &self.axes[0][..lengths[0]] {
            for y in &self.axes[1][..lengths[1]] {
                // Match Iterator::sum's identity and its original first-two-term order.
                let prefix = std::array::from_fn::<_, 3, _>(|i| {
                    [x.products[i], y.products[i]].into_iter().sum::<f64>()
                });
                for z in &self.axes[2][..lengths[2]] {
                    let vector = std::array::from_fn(|i| prefix[i] + z.products[i]);
                    let squared = dot([x.delta, y.delta, z.delta], vector);
                    if squared <= self.radius2 {
                        mask[x.index + y.index + z.index] +=
                            self.normalization * (-0.5 * squared / self.sigma2).exp();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::fourier::grid_index;
    use crate::solution::math3::mv;
    use phasesmith_crystallography::UnitCell;
    #[test]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn cached_masks_match_direct_at_periodic_and_support_boundaries_bitwise() {
        for cell in [
            UnitCell {
                a_angstrom: 6.,
                b_angstrom: 7.,
                c_angstrom: 8.,
                alpha_deg: 90.,
                beta_deg: 90.,
                gamma_deg: 90.,
            },
            UnitCell {
                a_angstrom: 6.,
                b_angstrom: 7.,
                c_angstrom: 8.,
                alpha_deg: 78.,
                beta_deg: 96.,
                gamma_deg: 107.,
            },
        ] {
            let geometry = cell.geometry().unwrap();
            for shape in [[8; 3], [10, 14, 18], [32, 16, 64]] {
                let sigma = 0.25;
                let bounds =
                    crate::solution::dual_space::mask_bounds(shape, &geometry, sigma).unwrap();
                let mut plan = PreparedMask::new(shape, &geometry, bounds, sigma);
                assert!(size_of::<PreparedMask>() <= 4608);
                let mut cached = vec![0.; shape.iter().product()];
                let mut direct = cached.clone();
                for point in [
                    [0.; 3],
                    [0.999_999, 0.03125, 0.5],
                    [0.1234, 0.4567, 0.8912],
                    [0.5; 3],
                ] {
                    plan.add_peak(&mut cached, point);
                    let centre = std::array::from_fn::<_, 3, _>(|i| point[i] * shape[i] as f64);
                    let origin = centre.map(|v| v.round() as i32);
                    for dx in -bounds[0]..=bounds[0] {
                        for dy in -bounds[1]..=bounds[1] {
                            for dz in -bounds[2]..=bounds[2] {
                                let index = [origin[0] + dx, origin[1] + dy, origin[2] + dz];
                                let delta = std::array::from_fn(|i| {
                                    (f64::from(index[i]) - centre[i]) / shape[i] as f64
                                });
                                let q2 = dot(delta, mv(geometry.direct_metric, delta));
                                if q2 <= plan.radius2 {
                                    direct[grid_index(index, shape)] +=
                                        plan.normalization * (-0.5 * q2 / sigma.powi(2)).exp();
                                }
                            }
                        }
                    }
                }
                assert_eq!(
                    cached.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
