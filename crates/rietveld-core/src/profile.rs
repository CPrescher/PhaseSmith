//! Symmetric pseudo-Voigt profile and fused peak accumulation.

use std::error::Error;
use std::fmt::{Display, Formatter};

const FOUR_LN_2: f64 = 4.0 * std::f64::consts::LN_2;
const GAUSSIAN_NORMALIZATION: f64 = 0.939_437_278_699_651_3; // sqrt(4 ln(2) / pi)

/// Parameters for one symmetric pseudo-Voigt peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// Peak center in the same coordinate system as the sampling grid.
    pub position: f64,
    /// Integrated intensity before applying the finite support window.
    pub intensity: f64,
    /// Full width at half maximum; must be positive.
    pub fwhm: f64,
    /// Lorentzian mixing fraction in the inclusive range `[0, 1]`.
    pub eta: f64,
}

/// A profile value and its analytical first derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProfilePoint {
    /// Unit-area profile value.
    pub value: f64,
    /// Derivative with respect to `delta = x - position`.
    pub d_delta: f64,
    /// Derivative with respect to FWHM.
    pub d_fwhm: f64,
    /// Derivative with respect to the Lorentzian fraction.
    pub d_eta: f64,
}

/// Result from fused multi-peak accumulation.
#[derive(Clone, Debug, PartialEq)]
pub struct Accumulation {
    /// Summed calculated profile, with one value per grid sample.
    pub y: Vec<f64>,
    /// Row-major Jacobian with shape `(peak, parameter, sample)`.
    ///
    /// The parameter order is intensity, position, FWHM, and eta.
    pub jacobian: Vec<f64>,
    /// Number of peaks represented by `jacobian`.
    pub peak_count: usize,
    /// Number of samples represented by `y` and `jacobian`.
    pub sample_count: usize,
}

impl Accumulation {
    /// Return the flat offset for a peak, parameter, and sample.
    #[must_use]
    pub const fn jacobian_index(
        &self,
        peak_index: usize,
        parameter_index: usize,
        sample_index: usize,
    ) -> usize {
        (peak_index * 4 + parameter_index) * self.sample_count + sample_index
    }
}

/// Input validation errors for profile evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// The sampling grid contains NaN or infinity.
    NonFiniteGrid {
        /// Index of the invalid grid value.
        index: usize,
    },
    /// The sampling grid is not strictly increasing.
    UnsortedGrid {
        /// Index of the first value not greater than its predecessor.
        index: usize,
    },
    /// A peak parameter contains NaN or infinity.
    NonFinitePeak {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// A peak FWHM is zero or negative.
    NonPositiveFwhm {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// A peak eta is outside `[0, 1]`.
    InvalidEta {
        /// Index of the invalid peak.
        peak: usize,
    },
    /// The requested support is zero, negative, NaN, or infinity.
    InvalidSupport,
    /// A Jacobian allocation would overflow `usize`.
    AllocationOverflow,
}

impl Display for ProfileError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteGrid { index } => {
                write!(formatter, "grid value at index {index} is not finite")
            }
            Self::UnsortedGrid { index } => write!(
                formatter,
                "grid must be strictly increasing (violation at index {index})"
            ),
            Self::NonFinitePeak { peak } => {
                write!(formatter, "peak {peak} contains a non-finite parameter")
            }
            Self::NonPositiveFwhm { peak } => {
                write!(formatter, "peak {peak} has a non-positive FWHM")
            }
            Self::InvalidEta { peak } => {
                write!(formatter, "peak {peak} has eta outside [0, 1]")
            }
            Self::InvalidSupport => write!(formatter, "support_fwhm must be positive and finite"),
            Self::AllocationOverflow => write!(formatter, "requested Jacobian is too large"),
        }
    }
}

impl Error for ProfileError {}

/// Evaluate a unit-area symmetric pseudo-Voigt profile and derivatives.
///
/// `delta` is `x - position`, `fwhm` is the common full width at half maximum,
/// and `eta` is the Lorentzian fraction. Callers should validate `fwhm` and
/// `eta` at their API boundary; this low-level scalar function is branch-free.
#[must_use]
pub fn symmetric_pseudo_voigt(delta: f64, fwhm: f64, eta: f64) -> ProfilePoint {
    let inverse_fwhm = fwhm.recip();
    let z = delta * inverse_fwhm;
    let z_squared = z * z;

    let gaussian = GAUSSIAN_NORMALIZATION * inverse_fwhm * (-FOUR_LN_2 * z_squared).exp();
    let lorentzian_denominator = 1.0 + 4.0 * z_squared;
    let lorentzian = 2.0 * inverse_fwhm / (std::f64::consts::PI * lorentzian_denominator);

    let d_gaussian_delta = gaussian * (-2.0 * FOUR_LN_2 * delta * inverse_fwhm.powi(2));
    let d_lorentzian_delta =
        lorentzian * (-8.0 * delta * inverse_fwhm.powi(2) / lorentzian_denominator);

    let d_gaussian_fwhm = gaussian * inverse_fwhm * (-1.0 + 2.0 * FOUR_LN_2 * z_squared);
    let d_lorentzian_fwhm =
        lorentzian * inverse_fwhm * (-1.0 + 8.0 * z_squared / lorentzian_denominator);

    let gaussian_weight = 1.0 - eta;
    ProfilePoint {
        value: eta * lorentzian + gaussian_weight * gaussian,
        d_delta: eta * d_lorentzian_delta + gaussian_weight * d_gaussian_delta,
        d_fwhm: eta * d_lorentzian_fwhm + gaussian_weight * d_gaussian_fwhm,
        d_eta: lorentzian - gaussian,
    }
}

/// Accumulate peaks and all analytical derivatives on a sorted grid.
///
/// Each peak is evaluated only at samples satisfying
/// `abs(x - position) <= support_fwhm * fwhm`. The Jacobian parameter order is
/// intensity, position, FWHM, and eta. The support's active sample set is held
/// fixed when differentiating.
///
/// # Errors
///
/// Returns [`ProfileError`] when the grid is not finite and strictly sorted, a
/// peak parameter is outside its domain, support is invalid, or the requested
/// Jacobian size overflows the platform allocation index.
pub fn accumulate_peaks(
    x: &[f64],
    peaks: &[Peak],
    support_fwhm: f64,
) -> Result<Accumulation, ProfileError> {
    validate_inputs(x, peaks, support_fwhm)?;

    let sample_count = x.len();
    let jacobian_length = peaks
        .len()
        .checked_mul(4)
        .and_then(|count| count.checked_mul(sample_count))
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut result = Accumulation {
        y: vec![0.0; sample_count],
        jacobian: vec![0.0; jacobian_length],
        peak_count: peaks.len(),
        sample_count,
    };

    for (peak_index, peak) in peaks.iter().copied().enumerate() {
        let radius = support_fwhm * peak.fwhm;
        let lower = x.partition_point(|value| *value < peak.position - radius);
        let upper = x.partition_point(|value| *value <= peak.position + radius);
        let parameter_base = peak_index * 4 * sample_count;

        for (relative_index, coordinate) in x[lower..upper].iter().copied().enumerate() {
            let sample_index = lower + relative_index;
            let profile = symmetric_pseudo_voigt(coordinate - peak.position, peak.fwhm, peak.eta);
            result.y[sample_index] += peak.intensity * profile.value;
            result.jacobian[parameter_base + sample_index] = profile.value;
            result.jacobian[parameter_base + sample_count + sample_index] =
                -peak.intensity * profile.d_delta;
            result.jacobian[parameter_base + 2 * sample_count + sample_index] =
                peak.intensity * profile.d_fwhm;
            result.jacobian[parameter_base + 3 * sample_count + sample_index] =
                peak.intensity * profile.d_eta;
        }
    }

    Ok(result)
}

fn validate_inputs(x: &[f64], peaks: &[Peak], support_fwhm: f64) -> Result<(), ProfileError> {
    if !support_fwhm.is_finite() || support_fwhm <= 0.0 {
        return Err(ProfileError::InvalidSupport);
    }

    for (index, value) in x.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ProfileError::NonFiniteGrid { index });
        }
        if index > 0 && value <= x[index - 1] {
            return Err(ProfileError::UnsortedGrid { index });
        }
    }

    for (peak_index, peak) in peaks.iter().enumerate() {
        if !peak.position.is_finite()
            || !peak.intensity.is_finite()
            || !peak.fwhm.is_finite()
            || !peak.eta.is_finite()
        {
            return Err(ProfileError::NonFinitePeak { peak: peak_index });
        }
        if peak.fwhm <= 0.0 {
            return Err(ProfileError::NonPositiveFwhm { peak: peak_index });
        }
        if !(0.0..=1.0).contains(&peak.eta) {
            return Err(ProfileError::InvalidEta { peak: peak_index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }

    #[test]
    fn components_have_expected_height_and_half_maximum() {
        for eta in [0.0, 0.25, 1.0] {
            let center = symmetric_pseudo_voigt(0.0, 2.0, eta).value;
            let half_maximum = symmetric_pseudo_voigt(1.0, 2.0, eta).value;
            assert_close(half_maximum, center / 2.0, 1e-15);
        }
    }

    #[test]
    fn scalar_derivatives_match_centered_differences() {
        let delta = 0.37;
        let fwhm = 0.82;
        let eta = 0.41;
        let h = 1e-6;
        let analytic = symmetric_pseudo_voigt(delta, fwhm, eta);

        let d_delta = (symmetric_pseudo_voigt(delta + h, fwhm, eta).value
            - symmetric_pseudo_voigt(delta - h, fwhm, eta).value)
            / (2.0 * h);
        let d_fwhm = (symmetric_pseudo_voigt(delta, fwhm + h, eta).value
            - symmetric_pseudo_voigt(delta, fwhm - h, eta).value)
            / (2.0 * h);
        let d_eta = (symmetric_pseudo_voigt(delta, fwhm, eta + h).value
            - symmetric_pseudo_voigt(delta, fwhm, eta - h).value)
            / (2.0 * h);

        assert_close(analytic.d_delta, d_delta, 2e-10);
        assert_close(analytic.d_fwhm, d_fwhm, 2e-10);
        assert_close(analytic.d_eta, d_eta, 2e-10);
    }

    #[test]
    fn accumulation_is_exactly_support_limited() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let peaks = [Peak {
            position: 0.0,
            intensity: 3.0,
            fwhm: 1.0,
            eta: 0.5,
        }];
        let result = accumulate_peaks(&x, &peaks, 1.0).expect("valid profile");
        assert_close(result.y[0], 0.0, 0.0);
        assert!(result.y[1] > 0.0);
        assert!(result.y[2] > 0.0);
        assert!(result.y[3] > 0.0);
        assert_close(result.y[4], 0.0, 0.0);
        assert_close(result.jacobian[result.jacobian_index(0, 0, 0)], 0.0, 0.0);
    }

    #[test]
    fn overlapping_peaks_sum_without_overwriting_jacobian_rows() {
        let x = [-0.25, 0.0, 0.25];
        let peaks = [
            Peak {
                position: 0.0,
                intensity: 2.0,
                fwhm: 1.0,
                eta: 0.2,
            },
            Peak {
                position: 0.1,
                intensity: 4.0,
                fwhm: 0.7,
                eta: 0.8,
            },
        ];
        let result = accumulate_peaks(&x, &peaks, 10.0).expect("valid profile");
        for (sample, coordinate) in x.iter().copied().enumerate() {
            let first = symmetric_pseudo_voigt(coordinate, 1.0, 0.2);
            let second = symmetric_pseudo_voigt(coordinate - 0.1, 0.7, 0.8);
            assert_close(
                result.y[sample],
                2.0 * first.value + 4.0 * second.value,
                1e-14,
            );
            assert_close(
                result.jacobian[result.jacobian_index(0, 0, sample)],
                first.value,
                1e-14,
            );
            assert_close(
                result.jacobian[result.jacobian_index(1, 0, sample)],
                second.value,
                1e-14,
            );
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let peak = Peak {
            position: 0.0,
            intensity: 1.0,
            fwhm: 1.0,
            eta: 0.5,
        };
        assert!(matches!(
            accumulate_peaks(&[0.0, 0.0], &[peak], 5.0),
            Err(ProfileError::UnsortedGrid { index: 1 })
        ));
        assert!(matches!(
            accumulate_peaks(&[0.0], &[Peak { fwhm: 0.0, ..peak }], 5.0),
            Err(ProfileError::NonPositiveFwhm { peak: 0 })
        ));
        assert!(matches!(
            accumulate_peaks(&[0.0], &[Peak { eta: 1.1, ..peak }], 5.0),
            Err(ProfileError::InvalidEta { peak: 0 })
        ));
    }
}
