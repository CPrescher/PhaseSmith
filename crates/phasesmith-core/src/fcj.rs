//! Finger-Cox-Jephcoat axial-divergence convolution.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::profile::SupportRange;
use crate::tch::{TchError, TchShape, TchWidths};

const DEGREE_TO_RADIAN: f64 = std::f64::consts::PI / 180.0;
const RADIAN_TO_DEGREE: f64 = 180.0 / std::f64::consts::PI;
pub(crate) const QUADRATURE_ORDER: usize = 48;
const SMALL_SPAN_QUADRATURE_ORDER: usize = 8;
// The convergence study in docs/fcj-profile.md bounds every value and direct
// derivative against an independent 256-point integral at this ratio.
const SMALL_SPAN_RATIO_LIMIT: f64 = 0.2;

const SMALL_SPAN_QUADRATURE_NODES: [f64; SMALL_SPAN_QUADRATURE_ORDER] = [
    1.985_507_175_123_191_2e-2,
    1.016_667_612_931_866_4e-1,
    2.372_337_950_418_355e-1,
    4.082_826_787_521_750_5e-1,
    5.917_173_212_478_25e-1,
    7.627_662_049_581_645e-1,
    8.983_332_387_068_134e-1,
    9.801_449_282_487_681e-1,
];

const SMALL_SPAN_QUADRATURE_WEIGHTS: [f64; SMALL_SPAN_QUADRATURE_ORDER] = [
    5.061_426_814_518_826e-2,
    1.111_905_172_266_872_3e-1,
    1.568_533_229_389_435_8e-1,
    1.813_418_916_891_809e-1,
    1.813_418_916_891_809e-1,
    1.568_533_229_389_435_8e-1,
    1.111_905_172_266_872_3e-1,
    5.061_426_814_518_826e-2,
];

// Gauss-Legendre nodes and weights transformed from [-1, 1] to [0, 1].
pub(crate) const QUADRATURE_NODES: [f64; QUADRATURE_ORDER] = [
    6.144_963_737_869_658e-4,
    3.234_913_866_824_618e-3,
    7.937_708_138_586_574e-3,
    1.470_420_372_687_636_4e-2,
    2.350_614_841_978_46e-2,
    3.430_665_464_672_283_4e-2,
    4.706_043_164_221_518_4e-2,
    6.171_398_986_287_607e-2,
    7.820_586_918_780_326e-2,
    9.646_689_798_527_869e-2,
    1.164_204_837_421_298_2e-1,
    1.379_829_345_380_926_8e-1,
    1.610_638_101_836_680_5e-1,
    1.855_663_016_117_432e-1,
    2.113_876_369_580_136_6e-1,
    2.384_195_126_388_835e-1,
    2.665_485_476_245_208e-1,
    2.956_567_590_046_416e-1,
    3.256_220_568_539_196_5e-1,
    3.563_187_563_222_722e-1,
    3.876_181_048_026_554_6e-1,
    4.193_888_219_655_541_6e-1,
    4.514_976_503_952_687e-1,
    4.838_099_145_185_653e-1,
    5.161_900_854_814_346e-1,
    5.485_023_496_047_313e-1,
    5.806_111_780_344_458e-1,
    6.123_818_951_973_445e-1,
    6.436_812_436_777_277e-1,
    6.743_779_431_460_803e-1,
    7.043_432_409_953_584e-1,
    7.334_514_523_754_792e-1,
    7.615_804_873_611_165e-1,
    7.886_123_630_419_863e-1,
    8.144_336_983_882_567e-1,
    8.389_361_898_163_319e-1,
    8.620_170_654_619_073e-1,
    8.835_795_162_578_701e-1,
    9.035_331_020_147_213e-1,
    9.217_941_308_121_967e-1,
    9.382_860_101_371_24e-1,
    9.529_395_683_577_848e-1,
    9.656_933_453_532_772e-1,
    9.764_938_515_802_154e-1,
    9.852_957_962_731_237e-1,
    9.920_622_918_614_135e-1,
    9.967_650_861_331_754e-1,
    9.993_855_036_262_13e-1,
];

pub(crate) const QUADRATURE_WEIGHTS: [f64; QUADRATURE_ORDER] = [
    1.576_673_026_154_921e-3,
    3.663_776_950_637_925_2e-3,
    5.738_617_289_617_35e-3,
    7.789_657_861_471_74e-3,
    9.808_080_228_678_052e-3,
    1.178_538_041_966_200_5e-2,
    1.371_325_485_417_852_6e-2,
    1.558_361_391_639_905_8e-2,
    1.738_861_128_238_521e-2,
    1.912_067_553_291_523_6e-2,
    2.077_254_147_173_226_6e-2,
    2.233_728_042_834_712_3e-2,
    2.380_832_924_624_513_5e-2,
    2.517_951_777_692_711e-2,
    2.644_509_474_259_671_2e-2,
    2.759_975_184_999_202e-2,
    2.863_864_605_020_144e-2,
    2.955_741_984_919_768e-2,
    3.035_221_958_294_678e-2,
    3.101_971_157_994_621e-2,
    3.155_709_614_312_688e-2,
    3.196_211_929_232_394e-2,
    3.223_308_221_797_490_5e-2,
    3.236_884_840_634_181e-2,
    3.236_884_840_634_181e-2,
    3.223_308_221_797_490_5e-2,
    3.196_211_929_232_394e-2,
    3.155_709_614_312_688e-2,
    3.101_971_157_994_621e-2,
    3.035_221_958_294_678e-2,
    2.955_741_984_919_768e-2,
    2.863_864_605_020_144e-2,
    2.759_975_184_999_202e-2,
    2.644_509_474_259_671_2e-2,
    2.517_951_777_692_711e-2,
    2.380_832_924_624_513_5e-2,
    2.233_728_042_834_712_3e-2,
    2.077_254_147_173_226_6e-2,
    1.912_067_553_291_523_6e-2,
    1.738_861_128_238_521e-2,
    1.558_361_391_639_905_8e-2,
    1.371_325_485_417_852_6e-2,
    1.178_538_041_966_200_5e-2,
    9.808_080_228_678_052e-3,
    7.789_657_861_471_74e-3,
    5.738_617_289_617_35e-3,
    3.663_776_950_637_925_2e-3,
    1.576_673_026_154_921e-3,
];

/// Dimensionless FCJ axial geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FcjGeometry {
    /// Sample axial half-height divided by diffractometer radius.
    pub sample_over_radius: f64,
    /// Receiving-slit axial half-height divided by diffractometer radius.
    pub detector_over_radius: f64,
}

/// One FCJ-convolved TCH profile value and direct-input derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FcjProfilePoint {
    /// Unit-area profile value before finite-support truncation.
    pub value: f64,
    /// Derivative with respect to the ideal Bragg position in degrees.
    pub d_position: f64,
    /// Derivative with respect to Gaussian component FWHM in degrees.
    pub d_gaussian_fwhm: f64,
    /// Derivative with respect to Lorentzian component FWHM in degrees.
    pub d_lorentzian_fwhm: f64,
    /// Derivative with respect to `sample_over_radius`.
    pub d_sample_over_radius: f64,
    /// Derivative with respect to `detector_over_radius`.
    pub d_detector_over_radius: f64,
}

/// FCJ profile domain errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FcjError {
    /// Ideal Bragg position is non-finite or outside `(0, 180)` degrees.
    InvalidPosition,
    /// An axial geometry ratio is negative or non-finite.
    InvalidGeometry,
    /// The axial extent crosses the valid angular domain.
    GeometryOutsideAngularDomain,
    /// Component-width transformation failed.
    InvalidWidths {
        /// TCH component-width failure.
        reason: TchError,
    },
    /// The transformed quadrature normalization is invalid.
    InvalidNormalization,
}

impl Display for FcjError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPosition => {
                write!(
                    formatter,
                    "FCJ position must be finite and within (0, 180) degrees"
                )
            }
            Self::InvalidGeometry => {
                write!(
                    formatter,
                    "FCJ axial ratios must be non-negative and finite"
                )
            }
            Self::GeometryOutsideAngularDomain => {
                write!(
                    formatter,
                    "FCJ axial geometry extends outside the angular domain"
                )
            }
            Self::InvalidWidths { reason } => write!(formatter, "invalid TCH widths: {reason}"),
            Self::InvalidNormalization => {
                write!(formatter, "FCJ normalization is not positive and finite")
            }
        }
    }
}

impl Error for FcjError {}

#[derive(Clone, Copy, Debug, Default)]
struct PreparedNode {
    apparent_position_deg: f64,
    weighted_geometry: f64,
    d_weighted_geometry_d_position: f64,
    d_weighted_geometry_d_major: f64,
    d_weighted_geometry_d_minor: f64,
    d_apparent_d_position: f64,
    d_apparent_d_major: f64,
    d_apparent_d_minor: f64,
}

/// Precomputed FCJ geometry and TCH shape for repeated sample evaluation.
#[derive(Clone, Debug)]
pub struct FcjProfile {
    shape: TchShape,
    geometry: FcjGeometry,
    position_deg: f64,
    nodes: Box<[PreparedNode]>,
    normalization: f64,
    d_normalization_d_position: f64,
    d_normalization_d_major: f64,
    d_normalization_d_minor: f64,
    apparent_limit_deg: f64,
}

impl FcjProfile {
    /// Prepare one FCJ-convolved TCH profile.
    ///
    /// # Errors
    ///
    /// Returns [`FcjError`] for invalid position, geometry, or component widths.
    pub fn new(
        position_deg: f64,
        widths: TchWidths,
        geometry: FcjGeometry,
    ) -> Result<Self, FcjError> {
        validate_position(position_deg)?;
        validate_geometry(geometry)?;
        let shape = TchShape::from_component_fwhm(widths)
            .map_err(|reason| FcjError::InvalidWidths { reason })?;
        let maximum_height = geometry.sample_over_radius + geometry.detector_over_radius;
        let position_rad = position_deg * DEGREE_TO_RADIAN;
        let limit_argument = position_rad.cos() * (1.0 + maximum_height * maximum_height).sqrt();
        if !(-1.0..=1.0).contains(&limit_argument) {
            return Err(FcjError::GeometryOutsideAngularDomain);
        }
        let apparent_limit_deg = if maximum_height == 0.0 {
            position_deg
        } else {
            limit_argument.acos() * RADIAN_TO_DEGREE
        };
        if maximum_height == 0.0 {
            let nodes = Box::new([PreparedNode {
                apparent_position_deg: position_deg,
                weighted_geometry: 1.0,
                d_apparent_d_position: 1.0,
                ..PreparedNode::default()
            }]);
            return Ok(Self {
                shape,
                geometry,
                position_deg,
                nodes,
                normalization: 1.0,
                d_normalization_d_position: 0.0,
                d_normalization_d_major: 0.0,
                d_normalization_d_minor: 0.0,
                apparent_limit_deg,
            });
        }

        let major = geometry
            .sample_over_radius
            .max(geometry.detector_over_radius);
        let minor = geometry
            .sample_over_radius
            .min(geometry.detector_over_radius);
        let difference = major - minor;
        let (quadrature_nodes, quadrature_weights) =
            quadrature_rule((apparent_limit_deg - position_deg).abs(), shape.total_fwhm);
        let piece_count = if difference == 0.0 { 1 } else { 2 };
        let mut nodes = Vec::with_capacity(piece_count * quadrature_nodes.len());
        let mut normalization = 0.0;
        let mut d_normalization_d_position = 0.0;
        let mut d_normalization_d_major = 0.0;
        let mut d_normalization_d_minor = 0.0;
        for (&t, &weight) in quadrature_nodes.iter().zip(quadrature_weights) {
            // Equal sample/detector heights have no flat overlap interval.
            // Its separate major/minor derivatives are equal and opposite, so
            // they also cancel in the symmetry-averaged public derivatives.
            if difference != 0.0 {
                let flat = prepare_node(
                    position_rad,
                    difference * t,
                    difference * weight,
                    weight,
                    -weight,
                    t,
                    -t,
                );
                normalization += flat.weighted_geometry;
                d_normalization_d_position += flat.d_weighted_geometry_d_position;
                d_normalization_d_major += flat.d_weighted_geometry_d_major;
                d_normalization_d_minor += flat.d_weighted_geometry_d_minor;
                nodes.push(flat);
            }
            let slope_weight = 2.0 * minor * weight * (1.0 - t);
            let slope = prepare_node(
                position_rad,
                difference + 2.0 * minor * t,
                slope_weight,
                0.0,
                2.0 * weight * (1.0 - t),
                1.0,
                -1.0 + 2.0 * t,
            );
            normalization += slope.weighted_geometry;
            d_normalization_d_position += slope.d_weighted_geometry_d_position;
            d_normalization_d_major += slope.d_weighted_geometry_d_major;
            d_normalization_d_minor += slope.d_weighted_geometry_d_minor;
            nodes.push(slope);
        }
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(FcjError::InvalidNormalization);
        }
        Ok(Self {
            shape,
            geometry,
            position_deg,
            nodes: nodes.into_boxed_slice(),
            normalization,
            d_normalization_d_position,
            d_normalization_d_major,
            d_normalization_d_minor,
            apparent_limit_deg,
        })
    }

    /// Evaluate the full FCJ-convolved profile at one sample coordinate.
    #[must_use]
    pub fn evaluate(&self, x_deg: f64) -> FcjProfilePoint {
        self.evaluate_with_radius(x_deg, f64::INFINITY)
    }

    #[must_use]
    pub(crate) fn evaluate_supported(
        &self,
        x_deg: f64,
        support_radius_deg: f64,
    ) -> FcjProfilePoint {
        self.evaluate_with_radius(x_deg, support_radius_deg)
    }

    #[must_use]
    pub(crate) fn support_range(&self, support_radius_deg: f64) -> SupportRange {
        SupportRange {
            left: self.apparent_limit_deg.min(self.position_deg) - support_radius_deg,
            right: self.apparent_limit_deg.max(self.position_deg) + support_radius_deg,
        }
    }

    fn evaluate_with_radius(&self, x_deg: f64, support_radius_deg: f64) -> FcjProfilePoint {
        let mut numerator = 0.0;
        let mut numerator_position = 0.0;
        let mut numerator_gaussian = 0.0;
        let mut numerator_lorentzian = 0.0;
        let mut numerator_major = 0.0;
        let mut numerator_minor = 0.0;
        for node in &self.nodes {
            let delta = x_deg - node.apparent_position_deg;
            if delta.abs() > support_radius_deg {
                continue;
            }
            let point = self.shape.evaluate(delta);
            numerator += node.weighted_geometry * point.value;
            numerator_position += node.d_weighted_geometry_d_position * point.value
                - node.weighted_geometry * point.d_delta * node.d_apparent_d_position;
            numerator_gaussian += node.weighted_geometry * point.d_gaussian_fwhm;
            numerator_lorentzian += node.weighted_geometry * point.d_lorentzian_fwhm;
            numerator_major += node.d_weighted_geometry_d_major * point.value
                - node.weighted_geometry * point.d_delta * node.d_apparent_d_major;
            numerator_minor += node.d_weighted_geometry_d_minor * point.value
                - node.weighted_geometry * point.d_delta * node.d_apparent_d_minor;
        }
        let value = numerator / self.normalization;
        let d_position =
            (numerator_position - value * self.d_normalization_d_position) / self.normalization;
        let d_gaussian_fwhm = numerator_gaussian / self.normalization;
        let d_lorentzian_fwhm = numerator_lorentzian / self.normalization;
        let d_major = (numerator_major - value * self.d_normalization_d_major) / self.normalization;
        let d_minor = (numerator_minor - value * self.d_normalization_d_minor) / self.normalization;
        let (d_sample_over_radius, d_detector_over_radius) =
            if self.geometry.sample_over_radius > self.geometry.detector_over_radius {
                (d_major, d_minor)
            } else if self.geometry.detector_over_radius > self.geometry.sample_over_radius {
                (d_minor, d_major)
            } else {
                let equal = 0.5 * (d_major + d_minor);
                (equal, equal)
            };
        FcjProfilePoint {
            value,
            d_position,
            d_gaussian_fwhm,
            d_lorentzian_fwhm,
            d_sample_over_radius,
            d_detector_over_radius,
        }
    }
}

fn quadrature_rule(axial_span_deg: f64, profile_fwhm_deg: f64) -> (&'static [f64], &'static [f64]) {
    if axial_span_deg / profile_fwhm_deg <= SMALL_SPAN_RATIO_LIMIT {
        (&SMALL_SPAN_QUADRATURE_NODES, &SMALL_SPAN_QUADRATURE_WEIGHTS)
    } else {
        (&QUADRATURE_NODES, &QUADRATURE_WEIGHTS)
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_node(
    position_rad: f64,
    height: f64,
    coefficient: f64,
    d_coefficient_d_major: f64,
    d_coefficient_d_minor: f64,
    d_height_d_major: f64,
    d_height_d_minor: f64,
) -> PreparedNode {
    let square_root = (1.0 + height * height).sqrt();
    let apparent_rad = (position_rad.cos() * square_root).acos();
    let sine_apparent = apparent_rad.sin();
    let d_apparent_d_height = -position_rad.cos() * height / (square_root * sine_apparent);
    let d_apparent_d_position = position_rad.sin() * square_root / sine_apparent;
    let geometry = (square_root * sine_apparent).recip();
    let cotangent_apparent = apparent_rad.cos() / sine_apparent;
    let d_geometry_d_height =
        geometry * (-height / (1.0 + height * height) - cotangent_apparent * d_apparent_d_height);
    let d_geometry_d_position =
        geometry * -cotangent_apparent * d_apparent_d_position * DEGREE_TO_RADIAN;
    PreparedNode {
        apparent_position_deg: apparent_rad * RADIAN_TO_DEGREE,
        weighted_geometry: coefficient * geometry,
        d_weighted_geometry_d_position: coefficient * d_geometry_d_position,
        d_weighted_geometry_d_major: d_coefficient_d_major * geometry
            + coefficient * d_geometry_d_height * d_height_d_major,
        d_weighted_geometry_d_minor: d_coefficient_d_minor * geometry
            + coefficient * d_geometry_d_height * d_height_d_minor,
        d_apparent_d_position,
        d_apparent_d_major: d_apparent_d_height * RADIAN_TO_DEGREE * d_height_d_major,
        d_apparent_d_minor: d_apparent_d_height * RADIAN_TO_DEGREE * d_height_d_minor,
    }
}

fn validate_position(position_deg: f64) -> Result<(), FcjError> {
    if !position_deg.is_finite() || position_deg <= 0.0 || position_deg >= 180.0 {
        return Err(FcjError::InvalidPosition);
    }
    Ok(())
}

fn validate_geometry(geometry: FcjGeometry) -> Result<(), FcjError> {
    if !geometry.sample_over_radius.is_finite()
        || !geometry.detector_over_radius.is_finite()
        || geometry.sample_over_radius < 0.0
        || geometry.detector_over_radius < 0.0
    {
        return Err(FcjError::InvalidGeometry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> FcjProfile {
        FcjProfile::new(
            12.0,
            TchWidths {
                gaussian_fwhm: 0.018,
                lorentzian_fwhm: 0.006,
            },
            FcjGeometry {
                sample_over_radius: 0.013,
                detector_over_radius: 0.009,
            },
        )
        .expect("valid FCJ profile")
    }

    fn assert_relative_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }

    #[test]
    fn zero_geometry_recovers_symmetric_tch_exactly() {
        let position = 42.0;
        let widths = TchWidths {
            gaussian_fwhm: 0.04,
            lorentzian_fwhm: 0.01,
        };
        let fcj = FcjProfile::new(
            position,
            widths,
            FcjGeometry {
                sample_over_radius: 0.0,
                detector_over_radius: 0.0,
            },
        )
        .expect("zero geometry");
        assert_eq!(fcj.nodes.len(), 1);
        let x = position + 0.017;
        let expected = TchShape::from_component_fwhm(widths)
            .expect("shape")
            .evaluate(x - position);
        let actual = fcj.evaluate(x);
        assert_relative_close(actual.value, expected.value, 0.0);
        assert_relative_close(actual.d_position, -expected.d_delta, 0.0);
        assert_relative_close(actual.d_gaussian_fwhm, expected.d_gaussian_fwhm, 0.0);
        assert_relative_close(actual.d_lorentzian_fwhm, expected.d_lorentzian_fwhm, 0.0);
        assert_relative_close(actual.d_sample_over_radius, 0.0, 0.0);
        assert_relative_close(actual.d_detector_over_radius, 0.0, 0.0);
    }

    #[test]
    fn all_direct_derivatives_match_centered_differences() {
        let x = 11.987;
        let baseline = profile().evaluate(x);
        let parameters = [12.0, 0.018, 0.006, 0.013, 0.009];
        let steps = [1e-6, 1e-7, 1e-7, 1e-7, 1e-7];
        let analytical = [
            baseline.d_position,
            baseline.d_gaussian_fwhm,
            baseline.d_lorentzian_fwhm,
            baseline.d_sample_over_radius,
            baseline.d_detector_over_radius,
        ];
        for parameter in 0..parameters.len() {
            let mut plus = parameters;
            let mut minus = parameters;
            plus[parameter] += steps[parameter];
            minus[parameter] -= steps[parameter];
            let evaluate = |values: [f64; 5]| {
                FcjProfile::new(
                    values[0],
                    TchWidths {
                        gaussian_fwhm: values[1],
                        lorentzian_fwhm: values[2],
                    },
                    FcjGeometry {
                        sample_over_radius: values[3],
                        detector_over_radius: values[4],
                    },
                )
                .expect("perturbed profile")
                .evaluate(x)
                .value
            };
            let finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * steps[parameter]);
            assert_relative_close(analytical[parameter], finite_difference, 2e-6);
        }
    }

    #[test]
    fn support_union_reverses_above_ninety_degrees() {
        let low = profile();
        let high = FcjProfile::new(
            138.0,
            TchWidths {
                gaussian_fwhm: 0.05,
                lorentzian_fwhm: 0.02,
            },
            FcjGeometry {
                sample_over_radius: 0.014,
                detector_over_radius: 0.014,
            },
        )
        .expect("high-angle profile");
        let low_support = low.support_range(0.2);
        let high_support = high.support_range(0.2);
        assert!(low_support.left < 12.0 - 0.2);
        assert_relative_close(low_support.right, 12.0 + 0.2, 1e-10);
        assert_relative_close(high_support.left, 138.0 - 0.2, 1e-10);
        assert!(high_support.right > 138.0 + 0.2);
    }

    #[test]
    fn quadrature_order_tracks_axial_span_relative_to_peak_width() {
        let broad_small_span = FcjProfile::new(
            70.0,
            TchWidths {
                gaussian_fwhm: 0.035,
                lorentzian_fwhm: 0.012,
            },
            FcjGeometry {
                sample_over_radius: 0.016,
                detector_over_radius: 0.009,
            },
        )
        .expect("small-span profile");
        assert_eq!(
            broad_small_span.nodes.len(),
            2 * SMALL_SPAN_QUADRATURE_ORDER
        );

        let narrow_large_span = profile();
        assert_eq!(narrow_large_span.nodes.len(), 2 * QUADRATURE_ORDER);
    }

    #[test]
    fn equal_heights_need_only_the_sloping_overlap_piece() {
        let equal = FcjProfile::new(
            70.0,
            TchWidths {
                gaussian_fwhm: 0.035,
                lorentzian_fwhm: 0.012,
            },
            FcjGeometry {
                sample_over_radius: 0.012,
                detector_over_radius: 0.012,
            },
        )
        .expect("equal-height profile");
        assert_eq!(equal.nodes.len(), SMALL_SPAN_QUADRATURE_ORDER);
    }
}
