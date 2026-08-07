//! Thompson-Cox-Hastings component-width transform and symmetric profile.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::profile::{
    Accumulation, GridView, PatternDerivatives, ProfileError, SupportJacobian, SupportPolicy,
    symmetric_pseudo_voigt, zeroed_f64_vec,
};

const WIDTH_COEFFICIENT_1: f64 = 2.692_69;
const WIDTH_COEFFICIENT_2: f64 = 2.428_43;
const WIDTH_COEFFICIENT_3: f64 = 4.471_63;
const WIDTH_COEFFICIENT_4: f64 = 0.078_42;
const ETA_COEFFICIENT_1: f64 = 1.366_03;
const ETA_COEFFICIENT_2: f64 = 0.477_19;
const ETA_COEFFICIENT_3: f64 = 0.111_16;

/// Gaussian and Lorentzian component FWHMs in one common coordinate unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TchWidths {
    /// Gaussian full width at half maximum.
    pub gaussian_fwhm: f64,
    /// Lorentzian full width at half maximum.
    pub lorentzian_fwhm: f64,
}

/// Transformed TCH pseudo-Voigt shape and analytical width derivatives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TchShape {
    /// Common pseudo-Voigt full width at half maximum.
    pub total_fwhm: f64,
    /// Lorentzian mixing fraction.
    pub eta: f64,
    /// Derivative of total FWHM with respect to Gaussian component FWHM.
    pub d_total_fwhm_d_gaussian_fwhm: f64,
    /// Derivative of total FWHM with respect to Lorentzian component FWHM.
    pub d_total_fwhm_d_lorentzian_fwhm: f64,
    /// Derivative of eta with respect to Gaussian component FWHM.
    pub d_eta_d_gaussian_fwhm: f64,
    /// Derivative of eta with respect to Lorentzian component FWHM.
    pub d_eta_d_lorentzian_fwhm: f64,
}

/// One TCH profile value and derivatives with respect to its direct inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TchProfilePoint {
    /// Unit-area profile value.
    pub value: f64,
    /// Derivative with respect to `delta = x - position`.
    pub d_delta: f64,
    /// Derivative with respect to Gaussian component FWHM.
    pub d_gaussian_fwhm: f64,
    /// Derivative with respect to Lorentzian component FWHM.
    pub d_lorentzian_fwhm: f64,
}

/// Component-width domain errors for the TCH transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TchError {
    /// Gaussian FWHM is NaN or infinite.
    NonFiniteGaussianFwhm,
    /// Lorentzian FWHM is NaN or infinite.
    NonFiniteLorentzianFwhm,
    /// Gaussian FWHM is negative.
    NegativeGaussianFwhm,
    /// Lorentzian FWHM is negative.
    NegativeLorentzianFwhm,
    /// Both component widths are zero.
    ZeroComponentWidths,
    /// The transformed total width exceeds finite floating-point range.
    NonFiniteTransform,
}

impl Display for TchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteGaussianFwhm => {
                write!(formatter, "Gaussian FWHM must be finite")
            }
            Self::NonFiniteLorentzianFwhm => {
                write!(formatter, "Lorentzian FWHM must be finite")
            }
            Self::NegativeGaussianFwhm => {
                write!(formatter, "Gaussian FWHM must be non-negative")
            }
            Self::NegativeLorentzianFwhm => {
                write!(formatter, "Lorentzian FWHM must be non-negative")
            }
            Self::ZeroComponentWidths => {
                write!(formatter, "at least one component FWHM must be positive")
            }
            Self::NonFiniteTransform => {
                write!(formatter, "transformed TCH width is outside finite range")
            }
        }
    }
}

impl Error for TchError {}

impl TchShape {
    /// Transform Gaussian and Lorentzian component FWHMs into `(H, eta)`.
    ///
    /// # Errors
    ///
    /// Returns [`TchError`] for non-finite or negative widths, or when both
    /// widths are zero.
    pub fn from_component_fwhm(widths: TchWidths) -> Result<Self, TchError> {
        validate_widths(widths)?;
        let width_scale = widths.gaussian_fwhm.max(widths.lorentzian_fwhm);
        let gaussian = widths.gaussian_fwhm / width_scale;
        let lorentzian = widths.lorentzian_fwhm / width_scale;
        let gaussian_2 = gaussian * gaussian;
        let gaussian_3 = gaussian_2 * gaussian;
        let gaussian_4 = gaussian_3 * gaussian;
        let lorentzian_2 = lorentzian * lorentzian;
        let lorentzian_3 = lorentzian_2 * lorentzian;
        let lorentzian_4 = lorentzian_3 * lorentzian;
        let width_polynomial = gaussian_4 * gaussian
            + WIDTH_COEFFICIENT_1 * gaussian_4 * lorentzian
            + WIDTH_COEFFICIENT_2 * gaussian_3 * lorentzian_2
            + WIDTH_COEFFICIENT_3 * gaussian_2 * lorentzian_3
            + WIDTH_COEFFICIENT_4 * gaussian * lorentzian_4
            + lorentzian_4 * lorentzian;
        let normalized_total_fwhm = width_polynomial.powf(0.2);
        let total_fwhm = width_scale * normalized_total_fwhm;
        if !total_fwhm.is_finite() {
            return Err(TchError::NonFiniteTransform);
        }
        let normalized_total_fwhm_4 = normalized_total_fwhm.powi(4);

        let d_polynomial_d_gaussian = 5.0 * gaussian_4
            + 4.0 * WIDTH_COEFFICIENT_1 * gaussian_3 * lorentzian
            + 3.0 * WIDTH_COEFFICIENT_2 * gaussian_2 * lorentzian_2
            + 2.0 * WIDTH_COEFFICIENT_3 * gaussian * lorentzian_3
            + WIDTH_COEFFICIENT_4 * lorentzian_4;
        let d_polynomial_d_lorentzian = WIDTH_COEFFICIENT_1 * gaussian_4
            + 2.0 * WIDTH_COEFFICIENT_2 * gaussian_3 * lorentzian
            + 3.0 * WIDTH_COEFFICIENT_3 * gaussian_2 * lorentzian_2
            + 4.0 * WIDTH_COEFFICIENT_4 * gaussian * lorentzian_3
            + 5.0 * lorentzian_4;
        let derivative_scale = (5.0 * normalized_total_fwhm_4).recip();
        let d_total_fwhm_d_gaussian_fwhm = d_polynomial_d_gaussian * derivative_scale;
        let d_total_fwhm_d_lorentzian_fwhm = d_polynomial_d_lorentzian * derivative_scale;

        let ratio = lorentzian / normalized_total_fwhm;
        let ratio_2 = ratio * ratio;
        let eta = ETA_COEFFICIENT_1 * ratio - ETA_COEFFICIENT_2 * ratio_2
            + ETA_COEFFICIENT_3 * ratio_2 * ratio;
        let d_eta_d_ratio =
            ETA_COEFFICIENT_1 - 2.0 * ETA_COEFFICIENT_2 * ratio + 3.0 * ETA_COEFFICIENT_3 * ratio_2;
        let d_ratio_d_gaussian = -ratio * d_total_fwhm_d_gaussian_fwhm / total_fwhm;
        let d_ratio_d_lorentzian = (1.0 - ratio * d_total_fwhm_d_lorentzian_fwhm) / total_fwhm;

        Ok(Self {
            total_fwhm,
            eta,
            d_total_fwhm_d_gaussian_fwhm,
            d_total_fwhm_d_lorentzian_fwhm,
            d_eta_d_gaussian_fwhm: d_eta_d_ratio * d_ratio_d_gaussian,
            d_eta_d_lorentzian_fwhm: d_eta_d_ratio * d_ratio_d_lorentzian,
        })
    }

    /// Evaluate a profile point using this precomputed width transform.
    #[must_use]
    pub fn evaluate(self, delta: f64) -> TchProfilePoint {
        tch_pseudo_voigt_from_shape(delta, self)
    }
}

/// Evaluate the TCH pseudo-Voigt profile and component-width derivatives.
///
/// # Errors
///
/// Returns [`TchError`] if either component width is invalid.
pub fn tch_pseudo_voigt(delta: f64, widths: TchWidths) -> Result<TchProfilePoint, TchError> {
    let shape = TchShape::from_component_fwhm(widths)?;
    Ok(shape.evaluate(delta))
}

fn tch_pseudo_voigt_from_shape(delta: f64, shape: TchShape) -> TchProfilePoint {
    let primitive = symmetric_pseudo_voigt(delta, shape.total_fwhm, shape.eta);
    TchProfilePoint {
        value: primitive.value,
        d_delta: primitive.d_delta,
        d_gaussian_fwhm: primitive.d_fwhm * shape.d_total_fwhm_d_gaussian_fwhm
            + primitive.d_eta * shape.d_eta_d_gaussian_fwhm,
        d_lorentzian_fwhm: primitive.d_fwhm * shape.d_total_fwhm_d_lorentzian_fwhm
            + primitive.d_eta * shape.d_eta_d_lorentzian_fwhm,
    }
}

/// Validated borrowed structure-of-arrays TCH peak batch.
#[derive(Clone, Copy, Debug)]
pub struct TchPeakBatchView<'a> {
    positions: &'a [f64],
    intensities: &'a [f64],
    gaussian_fwhms: &'a [f64],
    lorentzian_fwhms: &'a [f64],
}

impl<'a> TchPeakBatchView<'a> {
    /// Validate and borrow equal-length TCH peak arrays.
    ///
    /// # Errors
    ///
    /// Returns [`ProfileError`] if lengths differ, positions or intensities are
    /// non-finite, or component widths are invalid.
    pub fn new(
        positions: &'a [f64],
        intensities: &'a [f64],
        gaussian_fwhms: &'a [f64],
        lorentzian_fwhms: &'a [f64],
    ) -> Result<Self, ProfileError> {
        let peak_count = positions.len();
        if intensities.len() != peak_count
            || gaussian_fwhms.len() != peak_count
            || lorentzian_fwhms.len() != peak_count
        {
            return Err(ProfileError::TchPeakLengthMismatch);
        }
        for peak in 0..peak_count {
            if !positions[peak].is_finite() || !intensities[peak].is_finite() {
                return Err(ProfileError::NonFinitePeak { peak });
            }
            validate_widths(TchWidths {
                gaussian_fwhm: gaussian_fwhms[peak],
                lorentzian_fwhm: lorentzian_fwhms[peak],
            })
            .map_err(|reason| ProfileError::InvalidTchPeak { peak, reason })?;
        }
        Ok(Self {
            positions,
            intensities,
            gaussian_fwhms,
            lorentzian_fwhms,
        })
    }

    /// Number of peaks in the batch.
    #[must_use]
    pub const fn len(self) -> usize {
        self.positions.len()
    }

    /// Whether the batch contains no peaks.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.positions.is_empty()
    }

    fn widths(self, peak: usize) -> TchWidths {
        TchWidths {
            gaussian_fwhm: self.gaussian_fwhms[peak],
            lorentzian_fwhm: self.lorentzian_fwhms[peak],
        }
    }
}

/// Accumulate TCH peaks and direct-input derivatives in one support-limited pass.
///
/// Local derivative order is intensity, position, Gaussian FWHM, and
/// Lorentzian FWHM.
///
/// # Errors
///
/// Returns [`ProfileError`] if support is invalid or allocation fails.
pub fn accumulate_tch_batch(
    grid: GridView<'_>,
    peaks: TchPeakBatchView<'_>,
    support: SupportPolicy,
) -> Result<Accumulation, ProfileError> {
    support.validate()?;
    let x = grid.as_slice();
    let peak_count = peaks.len();
    let mut shapes = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    shapes
        .try_reserve_exact(peak_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    starts
        .try_reserve_exact(peak_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets
        .try_reserve_exact(
            peak_count
                .checked_add(1)
                .ok_or(ProfileError::AllocationOverflow)?,
        )
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets.push(0);

    for peak in 0..peak_count {
        let shape = TchShape::from_component_fwhm(peaks.widths(peak))
            .map_err(|reason| ProfileError::InvalidTchPeak { peak, reason })?;
        let range = support.range(peaks.positions[peak], shape.total_fwhm);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        let next_offset = offsets[peak]
            .checked_add(upper - lower)
            .ok_or(ProfileError::AllocationOverflow)?;
        shapes.push(shape);
        starts.push(lower);
        offsets.push(next_offset);
    }

    let active_sample_count = offsets.last().copied().unwrap_or(0);
    let value_count = active_sample_count
        .checked_mul(4)
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut values = zeroed_f64_vec(value_count)?;
    for peak in 0..peak_count {
        let start = starts[peak];
        let active_begin = offsets[peak];
        let active_end = offsets[peak + 1];
        for active_index in active_begin..active_end {
            let sample = start + active_index - active_begin;
            let point = shapes[peak].evaluate(x[sample] - peaks.positions[peak]);
            let intensity = peaks.intensities[peak];
            y[sample] += intensity * point.value;
            let value_base = active_index * 4;
            values[value_base] = point.value;
            values[value_base + 1] = -intensity * point.d_delta;
            values[value_base + 2] = intensity * point.d_gaussian_fwhm;
            values[value_base + 3] = intensity * point.d_lorentzian_fwhm;
        }
    }

    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values,
                parameter_count: 4,
            },
            global: None,
        },
        sample_count: x.len(),
    })
}

fn validate_widths(widths: TchWidths) -> Result<(), TchError> {
    if !widths.gaussian_fwhm.is_finite() {
        return Err(TchError::NonFiniteGaussianFwhm);
    }
    if !widths.lorentzian_fwhm.is_finite() {
        return Err(TchError::NonFiniteLorentzianFwhm);
    }
    if widths.gaussian_fwhm < 0.0 {
        return Err(TchError::NegativeGaussianFwhm);
    }
    if widths.lorentzian_fwhm < 0.0 {
        return Err(TchError::NegativeLorentzianFwhm);
    }
    if widths.gaussian_fwhm == 0.0 && widths.lorentzian_fwhm == 0.0 {
        return Err(TchError::ZeroComponentWidths);
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
    fn pure_component_limits_are_exact() {
        let gaussian = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: 0.2,
            lorentzian_fwhm: 0.0,
        })
        .expect("Gaussian limit");
        assert_close(gaussian.total_fwhm, 0.2, 1e-16);
        assert_close(gaussian.eta, 0.0, 0.0);

        let lorentzian = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: 0.0,
            lorentzian_fwhm: 0.3,
        })
        .expect("Lorentzian limit");
        assert_close(lorentzian.total_fwhm, 0.3, 1e-16);
        assert_close(lorentzian.eta, 1.0, 2e-16);
    }

    #[test]
    fn transform_derivatives_match_centered_differences() {
        let widths = TchWidths {
            gaussian_fwhm: 0.071,
            lorentzian_fwhm: 0.023,
        };
        let shape = TchShape::from_component_fwhm(widths).expect("shape");
        let step = 1e-7;
        let gaussian_plus = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: widths.gaussian_fwhm + step,
            ..widths
        })
        .expect("plus");
        let gaussian_minus = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: widths.gaussian_fwhm - step,
            ..widths
        })
        .expect("minus");
        let lorentzian_plus = TchShape::from_component_fwhm(TchWidths {
            lorentzian_fwhm: widths.lorentzian_fwhm + step,
            ..widths
        })
        .expect("plus");
        let lorentzian_minus = TchShape::from_component_fwhm(TchWidths {
            lorentzian_fwhm: widths.lorentzian_fwhm - step,
            ..widths
        })
        .expect("minus");
        assert_close(
            shape.d_total_fwhm_d_gaussian_fwhm,
            (gaussian_plus.total_fwhm - gaussian_minus.total_fwhm) / (2.0 * step),
            2e-10,
        );
        assert_close(
            shape.d_eta_d_gaussian_fwhm,
            (gaussian_plus.eta - gaussian_minus.eta) / (2.0 * step),
            2e-9,
        );
        assert_close(
            shape.d_total_fwhm_d_lorentzian_fwhm,
            (lorentzian_plus.total_fwhm - lorentzian_minus.total_fwhm) / (2.0 * step),
            2e-10,
        );
        assert_close(
            shape.d_eta_d_lorentzian_fwhm,
            (lorentzian_plus.eta - lorentzian_minus.eta) / (2.0 * step),
            2e-9,
        );
    }

    #[test]
    fn invalid_component_widths_are_rejected() {
        assert_eq!(
            TchShape::from_component_fwhm(TchWidths {
                gaussian_fwhm: 0.0,
                lorentzian_fwhm: 0.0,
            }),
            Err(TchError::ZeroComponentWidths)
        );
        assert_eq!(
            TchShape::from_component_fwhm(TchWidths {
                gaussian_fwhm: -0.1,
                lorentzian_fwhm: 0.2,
            }),
            Err(TchError::NegativeGaussianFwhm)
        );
    }

    #[test]
    fn normalized_polynomial_handles_extreme_finite_scales() {
        let unit = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: 0.7,
            lorentzian_fwhm: 0.3,
        })
        .expect("unit-scale shape");
        let tiny_scale = 1e-250;
        let tiny = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: 0.7 * tiny_scale,
            lorentzian_fwhm: 0.3 * tiny_scale,
        })
        .expect("tiny shape");
        assert_close(tiny.total_fwhm / tiny_scale, unit.total_fwhm, 5e-16);
        assert_close(tiny.eta, unit.eta, 5e-16);
        assert_close(
            tiny.d_total_fwhm_d_gaussian_fwhm,
            unit.d_total_fwhm_d_gaussian_fwhm,
            5e-16,
        );
        assert_eq!(
            TchShape::from_component_fwhm(TchWidths {
                gaussian_fwhm: f64::MAX,
                lorentzian_fwhm: f64::MAX,
            }),
            Err(TchError::NonFiniteTransform)
        );
    }
}
