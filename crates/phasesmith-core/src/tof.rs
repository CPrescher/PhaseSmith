//! Neutron time-of-flight calibration, profile parameters, and accumulation.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::fcj::{QUADRATURE_NODES, QUADRATURE_ORDER, QUADRATURE_WEIGHTS};
use crate::profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, ProfileError, SupportJacobian,
    SupportRange, zeroed_f64_vec,
};
use crate::tch::{TchError, TchShape, TchWidths};

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;
/// Public order: zero, difC, difA, difB, alpha, beta0, beta1, betaq,
/// sigma0, sigma1, sigma2, sigmaq, X, Y, Z.
pub const TOF_GLOBAL_PARAMETER_COUNT: usize = 15;
const LOCAL_PARAMETER_COUNT: usize = 2;
const TOF_QUADRATURE_PANELS: usize = 8;
const TOF_QUADRATURE_PANELS_F64: f64 = 8.0;
const TOF_SUPPORT_QUADRATURE_PANELS: usize = 4;
const TOF_SUPPORT_QUADRATURE_PANELS_F64: f64 = 4.0;
const TOF_QUADRATURE_COUNT: usize = TOF_QUADRATURE_PANELS * QUADRATURE_ORDER;

/// TOF calibration and d-dependent profile coefficients in public units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TofInstrument {
    /// Additive time zero in microseconds.
    pub zero_us: f64,
    /// Linear calibration coefficient in microseconds per ångström.
    pub difc_us_per_angstrom: f64,
    /// Quadratic calibration coefficient in microseconds per ångström squared.
    pub difa_us_per_angstrom2: f64,
    /// Reciprocal calibration coefficient in microsecond ångströms.
    pub difb_us_angstrom: f64,
    /// Exponential-rise numerator; `alpha = alpha_coefficient / d`.
    pub alpha_coefficient: f64,
    /// Constant exponential-decay rate term in inverse microseconds.
    pub beta0_per_us: f64,
    /// `d^-4` exponential-decay coefficient.
    pub beta1_angstrom4_per_us: f64,
    /// `d^-2` exponential-decay coefficient.
    pub betaq_angstrom2_per_us: f64,
    /// Constant Gaussian variance term in microseconds squared.
    pub sigma0_us2: f64,
    /// `d^2` Gaussian variance coefficient.
    pub sigma1_us2_per_angstrom2: f64,
    /// `d^4` Gaussian variance coefficient.
    pub sigma2_us2_per_angstrom4: f64,
    /// Linear-d Gaussian variance coefficient.
    pub sigmaq_us2_per_angstrom: f64,
    /// Linear-d Lorentzian FWHM coefficient.
    pub x_us_per_angstrom: f64,
    /// Quadratic-d Lorentzian FWHM coefficient.
    pub y_us_per_angstrom2: f64,
    /// Constant Lorentzian FWHM in microseconds.
    pub z_us: f64,
}

impl TofInstrument {
    /// Validate finite coefficients and a positive linear calibration scale.
    ///
    /// # Errors
    ///
    /// Returns [`TofError`] for non-finite coefficients or non-positive
    /// linear calibration.
    pub fn validate(self) -> Result<(), TofError> {
        let values = [
            self.zero_us,
            self.difc_us_per_angstrom,
            self.difa_us_per_angstrom2,
            self.difb_us_angstrom,
            self.alpha_coefficient,
            self.beta0_per_us,
            self.beta1_angstrom4_per_us,
            self.betaq_angstrom2_per_us,
            self.sigma0_us2,
            self.sigma1_us2_per_angstrom2,
            self.sigma2_us2_per_angstrom4,
            self.sigmaq_us2_per_angstrom,
            self.x_us_per_angstrom,
            self.y_us_per_angstrom2,
            self.z_us,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TofError::NonFiniteInstrumentParameter);
        }
        if self.difc_us_per_angstrom <= 0.0 {
            return Err(TofError::NonPositiveDifc);
        }
        Ok(())
    }
}

/// Derived TOF position, widths, exponential rates, and analytical chains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TofProfileParameters {
    /// Calibrated position in microseconds.
    pub position_us: f64,
    /// Leading-edge exponential rate in inverse microseconds.
    pub alpha_per_us: f64,
    /// Trailing-edge exponential rate in inverse microseconds.
    pub beta_per_us: f64,
    /// Gaussian variance in microseconds squared.
    pub gaussian_variance_us2: f64,
    /// Gaussian component FWHM in microseconds.
    pub gaussian_fwhm_us: f64,
    /// Lorentzian component FWHM in microseconds.
    pub lorentzian_fwhm_us: f64,
    /// TCH transform of the component widths.
    pub tch: TchShape,
    /// Direct-input derivatives with respect to d-spacing.
    pub d_position_d_d: f64,
    /// Alpha-rate derivative with respect to d-spacing.
    pub d_alpha_d_d: f64,
    /// Beta-rate derivative with respect to d-spacing.
    pub d_beta_d_d: f64,
    /// Gaussian-FWHM derivative with respect to d-spacing.
    pub d_gaussian_fwhm_d_d: f64,
    /// Lorentzian-FWHM derivative with respect to d-spacing.
    pub d_lorentzian_fwhm_d_d: f64,
    /// Parameter-major chains in [`TOF_GLOBAL_PARAMETER_COUNT`] order.
    pub d_position_d_instrument: [f64; TOF_GLOBAL_PARAMETER_COUNT],
    /// Alpha-rate chains in global instrument order.
    pub d_alpha_d_instrument: [f64; TOF_GLOBAL_PARAMETER_COUNT],
    /// Beta-rate chains in global instrument order.
    pub d_beta_d_instrument: [f64; TOF_GLOBAL_PARAMETER_COUNT],
    /// Gaussian-FWHM chains in global instrument order.
    pub d_gaussian_fwhm_d_instrument: [f64; TOF_GLOBAL_PARAMETER_COUNT],
    /// Lorentzian-FWHM chains in global instrument order.
    pub d_lorentzian_fwhm_d_instrument: [f64; TOF_GLOBAL_PARAMETER_COUNT],
}

impl TofProfileParameters {
    /// Derive all profile quantities for one positive d-spacing.
    ///
    /// # Errors
    ///
    /// Returns [`TofError`] for invalid inputs or nonphysical derived profile
    /// parameters.
    pub fn from_instrument(d: f64, instrument: TofInstrument) -> Result<Self, TofError> {
        instrument.validate()?;
        if !d.is_finite() || d <= 0.0 {
            return Err(TofError::InvalidDSpacing);
        }
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d2 * d2;
        let inverse_d = d.recip();
        let inverse_d2 = inverse_d * inverse_d;
        let inverse_d3 = inverse_d2 * inverse_d;
        let inverse_d4 = inverse_d2 * inverse_d2;
        let inverse_d5 = inverse_d4 * inverse_d;
        let position_us = instrument.zero_us
            + instrument.difc_us_per_angstrom * d
            + instrument.difa_us_per_angstrom2 * d2
            + instrument.difb_us_angstrom * inverse_d;
        let alpha_per_us = instrument.alpha_coefficient * inverse_d;
        let beta_per_us = instrument.beta0_per_us
            + instrument.beta1_angstrom4_per_us * inverse_d4
            + instrument.betaq_angstrom2_per_us * inverse_d2;
        let gaussian_variance_us2 = instrument.sigma0_us2
            + instrument.sigma1_us2_per_angstrom2 * d2
            + instrument.sigma2_us2_per_angstrom4 * d4
            + instrument.sigmaq_us2_per_angstrom * d;
        let lorentzian_fwhm_us =
            instrument.z_us + instrument.x_us_per_angstrom * d + instrument.y_us_per_angstrom2 * d2;
        if !position_us.is_finite() {
            return Err(TofError::InvalidPosition);
        }
        if !alpha_per_us.is_finite() || alpha_per_us <= 0.0 {
            return Err(TofError::NonPositiveAlpha);
        }
        if !beta_per_us.is_finite() || beta_per_us <= 0.0 {
            return Err(TofError::NonPositiveBeta);
        }
        if !gaussian_variance_us2.is_finite() || gaussian_variance_us2 <= 0.0 {
            return Err(TofError::NonPositiveGaussianVariance);
        }
        if !lorentzian_fwhm_us.is_finite() || lorentzian_fwhm_us < 0.0 {
            return Err(TofError::NegativeLorentzianFwhm);
        }
        let sigma = gaussian_variance_us2.sqrt();
        let gaussian_fwhm_us = GAUSSIAN_FWHM_PER_SIGMA * sigma;
        let tch = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: gaussian_fwhm_us,
            lorentzian_fwhm: lorentzian_fwhm_us,
        })
        .map_err(|reason| TofError::InvalidTch { reason })?;
        let d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * sigma);
        let d_variance_d_d = 2.0 * instrument.sigma1_us2_per_angstrom2 * d
            + 4.0 * instrument.sigma2_us2_per_angstrom4 * d3
            + instrument.sigmaq_us2_per_angstrom;

        let mut d_position_d_instrument = [0.0; TOF_GLOBAL_PARAMETER_COUNT];
        d_position_d_instrument[..4].copy_from_slice(&[1.0, d, d2, inverse_d]);
        let mut d_alpha_d_instrument = [0.0; TOF_GLOBAL_PARAMETER_COUNT];
        d_alpha_d_instrument[4] = inverse_d;
        let mut d_beta_d_instrument = [0.0; TOF_GLOBAL_PARAMETER_COUNT];
        d_beta_d_instrument[5..8].copy_from_slice(&[1.0, inverse_d4, inverse_d2]);
        let mut d_gaussian_fwhm_d_instrument = [0.0; TOF_GLOBAL_PARAMETER_COUNT];
        d_gaussian_fwhm_d_instrument[8..12].copy_from_slice(&[
            d_gaussian_d_variance,
            d_gaussian_d_variance * d2,
            d_gaussian_d_variance * d4,
            d_gaussian_d_variance * d,
        ]);
        let mut d_lorentzian_fwhm_d_instrument = [0.0; TOF_GLOBAL_PARAMETER_COUNT];
        d_lorentzian_fwhm_d_instrument[12..15].copy_from_slice(&[d, d2, 1.0]);
        Ok(Self {
            position_us,
            alpha_per_us,
            beta_per_us,
            gaussian_variance_us2,
            gaussian_fwhm_us,
            lorentzian_fwhm_us,
            tch,
            d_position_d_d: instrument.difc_us_per_angstrom
                + 2.0 * instrument.difa_us_per_angstrom2 * d
                - instrument.difb_us_angstrom * inverse_d2,
            d_alpha_d_d: -instrument.alpha_coefficient * inverse_d2,
            d_beta_d_d: -4.0 * instrument.beta1_angstrom4_per_us * inverse_d5
                - 2.0 * instrument.betaq_angstrom2_per_us * inverse_d3,
            d_gaussian_fwhm_d_d: d_gaussian_d_variance * d_variance_d_d,
            d_lorentzian_fwhm_d_d: instrument.x_us_per_angstrom
                + 2.0 * instrument.y_us_per_angstrom2 * d,
            d_position_d_instrument,
            d_alpha_d_instrument,
            d_beta_d_instrument,
            d_gaussian_fwhm_d_instrument,
            d_lorentzian_fwhm_d_instrument,
        })
    }
}

/// One asymmetric TOF profile value and direct-input derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TofProfilePoint {
    /// Unit-area profile density in inverse microseconds.
    pub value: f64,
    /// Derivative with respect to ideal position in microseconds.
    pub d_position: f64,
    /// Derivative with respect to the leading-edge alpha rate.
    pub d_alpha: f64,
    /// Derivative with respect to the trailing-edge beta rate.
    pub d_beta: f64,
    /// Derivative with respect to Gaussian component FWHM.
    pub d_gaussian_fwhm: f64,
    /// Derivative with respect to Lorentzian component FWHM.
    pub d_lorentzian_fwhm: f64,
}

/// Prepared truncated double-exponential convolution of a TCH profile.
#[derive(Clone, Debug)]
pub struct TofProfile {
    shape: TchShape,
    alpha: f64,
    beta: f64,
    quadrature: Arc<TofQuadrature>,
}

#[derive(Debug)]
struct TofQuadrature {
    tail_log: f64,
    nodes: [f64; TOF_QUADRATURE_COUNT],
    weights: [f64; TOF_QUADRATURE_COUNT],
}

impl TofQuadrature {
    fn new(tail_log: f64) -> Result<Self, TofError> {
        if !tail_log.is_finite() || tail_log <= 0.0 {
            return Err(TofError::InvalidTailLog);
        }
        let mut nodes = [0.0; TOF_QUADRATURE_COUNT];
        let mut weights = [0.0; TOF_QUADRATURE_COUNT];
        let mut normalization = 0.0;
        let panel_scale = TOF_QUADRATURE_PANELS_F64.recip();
        let mut panel_offset = 0.0;
        for panel in 0..TOF_QUADRATURE_PANELS {
            for quadrature in 0..QUADRATURE_ORDER {
                let index = panel * QUADRATURE_ORDER + quadrature;
                let unit_node = panel_offset + panel_scale * QUADRATURE_NODES[quadrature];
                nodes[index] = tail_log * unit_node;
                weights[index] =
                    tail_log * panel_scale * QUADRATURE_WEIGHTS[quadrature] * (-nodes[index]).exp();
                normalization += weights[index];
            }
            panel_offset += panel_scale;
        }
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(TofError::InvalidQuadrature);
        }
        for weight in &mut weights {
            *weight /= normalization;
        }
        Ok(Self {
            tail_log,
            nodes,
            weights,
        })
    }
}

impl TofProfile {
    /// Prepare a unit-area profile. Both exponential tails are truncated at
    /// `exp(-tail_log)` and renormalized before convolution.
    ///
    /// # Errors
    ///
    /// Returns [`TofError`] for invalid rates, widths, tail cutoff, or
    /// quadrature normalization.
    pub fn new(
        alpha_per_us: f64,
        beta_per_us: f64,
        widths: TchWidths,
        tail_log: f64,
    ) -> Result<Self, TofError> {
        Self::validate_rates(alpha_per_us, beta_per_us)?;
        let quadrature = Arc::new(TofQuadrature::new(tail_log)?);
        Self::from_validated_rates(alpha_per_us, beta_per_us, widths, quadrature)
    }

    fn validate_rates(alpha_per_us: f64, beta_per_us: f64) -> Result<(), TofError> {
        if !alpha_per_us.is_finite() || alpha_per_us <= 0.0 {
            return Err(TofError::NonPositiveAlpha);
        }
        if !beta_per_us.is_finite() || beta_per_us <= 0.0 {
            return Err(TofError::NonPositiveBeta);
        }
        Ok(())
    }

    fn from_validated_rates(
        alpha_per_us: f64,
        beta_per_us: f64,
        widths: TchWidths,
        quadrature: Arc<TofQuadrature>,
    ) -> Result<Self, TofError> {
        let shape = TchShape::from_component_fwhm(widths)
            .map_err(|reason| TofError::InvalidTch { reason })?;
        Ok(Self {
            shape,
            alpha: alpha_per_us,
            beta: beta_per_us,
            quadrature,
        })
    }

    /// Evaluate without finite TCH support truncation.
    #[must_use]
    pub fn evaluate(&self, x_minus_position_us: f64) -> TofProfilePoint {
        self.evaluate_with_radius(x_minus_position_us, f64::INFINITY)
    }

    fn evaluate_with_radius(&self, delta: f64, base_radius: f64) -> TofProfilePoint {
        if base_radius.is_finite() {
            return self.evaluate_supported(delta, base_radius);
        }
        let sum = self.alpha + self.beta;
        let left_fraction = self.beta / sum;
        let right_fraction = self.alpha / sum;
        let d_left_d_alpha = -self.beta / (sum * sum);
        let d_left_d_beta = self.alpha / (sum * sum);
        let mut left = TofProfilePoint::default();
        let mut right = TofProfilePoint::default();
        let mut left_alpha_shift = 0.0;
        let mut right_beta_shift = 0.0;
        for index in 0..TOF_QUADRATURE_COUNT {
            let node = self.quadrature.nodes[index];
            let weight = self.quadrature.weights[index];
            let left_delta = delta + node / self.alpha;
            if left_delta.abs() <= base_radius {
                let point = self.shape.evaluate(left_delta);
                left.value += weight * point.value;
                left.d_position += weight * point.d_delta;
                left.d_gaussian_fwhm += weight * point.d_gaussian_fwhm;
                left.d_lorentzian_fwhm += weight * point.d_lorentzian_fwhm;
                left_alpha_shift += weight * point.d_delta * (-node / self.alpha.powi(2));
            }
            let right_delta = delta - node / self.beta;
            if right_delta.abs() <= base_radius {
                let point = self.shape.evaluate(right_delta);
                right.value += weight * point.value;
                right.d_position += weight * point.d_delta;
                right.d_gaussian_fwhm += weight * point.d_gaussian_fwhm;
                right.d_lorentzian_fwhm += weight * point.d_lorentzian_fwhm;
                right_beta_shift += weight * point.d_delta * (node / self.beta.powi(2));
            }
        }
        TofProfilePoint {
            value: left_fraction * left.value + right_fraction * right.value,
            d_position: -(left_fraction * left.d_position + right_fraction * right.d_position),
            d_alpha: d_left_d_alpha * left.value + left_fraction * left_alpha_shift
                - d_left_d_alpha * right.value,
            d_beta: d_left_d_beta * left.value + right_fraction * right_beta_shift
                - d_left_d_beta * right.value,
            d_gaussian_fwhm: left_fraction * left.d_gaussian_fwhm
                + right_fraction * right.d_gaussian_fwhm,
            d_lorentzian_fwhm: left_fraction * left.d_lorentzian_fwhm
                + right_fraction * right.d_lorentzian_fwhm,
        }
    }

    fn evaluate_supported(&self, delta: f64, base_radius: f64) -> TofProfilePoint {
        let sum = self.alpha + self.beta;
        let left_fraction = self.beta / sum;
        let right_fraction = self.alpha / sum;
        let d_left_d_alpha = -self.beta / (sum * sum);
        let d_left_d_beta = self.alpha / (sum * sum);
        let tail_log = self.quadrature.tail_log;
        let normalization = 1.0 - (-tail_log).exp();
        let left_low = (self.alpha * (-base_radius - delta)).clamp(0.0, tail_log);
        let left_high = (self.alpha * (base_radius - delta)).clamp(0.0, tail_log);
        let right_low = (self.beta * (delta - base_radius)).clamp(0.0, tail_log);
        let right_high = (self.beta * (delta + base_radius)).clamp(0.0, tail_log);
        let mut left = TofProfilePoint::default();
        let mut right = TofProfilePoint::default();
        let mut left_alpha_shift = 0.0;
        let mut right_beta_shift = 0.0;

        if left_low < left_high {
            let panel_width = (left_high - left_low) / TOF_SUPPORT_QUADRATURE_PANELS_F64;
            let mut panel_left = left_low;
            for _ in 0..TOF_SUPPORT_QUADRATURE_PANELS {
                for quadrature in 0..QUADRATURE_ORDER {
                    let node = panel_left + panel_width * QUADRATURE_NODES[quadrature];
                    let weight = panel_width * QUADRATURE_WEIGHTS[quadrature] * (-node).exp()
                        / normalization;
                    let point = self.shape.evaluate(delta + node / self.alpha);
                    left.value += weight * point.value;
                    left.d_position += weight * point.d_delta;
                    left.d_gaussian_fwhm += weight * point.d_gaussian_fwhm;
                    left.d_lorentzian_fwhm += weight * point.d_lorentzian_fwhm;
                    left_alpha_shift += weight * point.d_delta * (-node / self.alpha.powi(2));
                }
                panel_left += panel_width;
            }
        }
        if right_low < right_high {
            let panel_width = (right_high - right_low) / TOF_SUPPORT_QUADRATURE_PANELS_F64;
            let mut panel_left = right_low;
            for _ in 0..TOF_SUPPORT_QUADRATURE_PANELS {
                for quadrature in 0..QUADRATURE_ORDER {
                    let node = panel_left + panel_width * QUADRATURE_NODES[quadrature];
                    let weight = panel_width * QUADRATURE_WEIGHTS[quadrature] * (-node).exp()
                        / normalization;
                    let point = self.shape.evaluate(delta - node / self.beta);
                    right.value += weight * point.value;
                    right.d_position += weight * point.d_delta;
                    right.d_gaussian_fwhm += weight * point.d_gaussian_fwhm;
                    right.d_lorentzian_fwhm += weight * point.d_lorentzian_fwhm;
                    right_beta_shift += weight * point.d_delta * (node / self.beta.powi(2));
                }
                panel_left += panel_width;
            }
        }
        TofProfilePoint {
            value: left_fraction * left.value + right_fraction * right.value,
            d_position: -(left_fraction * left.d_position + right_fraction * right.d_position),
            d_alpha: d_left_d_alpha * left.value + left_fraction * left_alpha_shift
                - d_left_d_alpha * right.value,
            d_beta: d_left_d_beta * left.value + right_fraction * right_beta_shift
                - d_left_d_beta * right.value,
            d_gaussian_fwhm: left_fraction * left.d_gaussian_fwhm
                + right_fraction * right.d_gaussian_fwhm,
            d_lorentzian_fwhm: left_fraction * left.d_lorentzian_fwhm
                + right_fraction * right.d_lorentzian_fwhm,
        }
    }

    fn support_range(&self, position: f64, base_radius: f64) -> SupportRange {
        SupportRange {
            left: position - base_radius - self.quadrature.tail_log / self.alpha,
            right: position + base_radius + self.quadrature.tail_log / self.beta,
        }
    }
}

/// TOF domain or accumulation error.
#[derive(Clone, Debug, PartialEq)]
pub enum TofError {
    /// At least one instrument coefficient is non-finite.
    NonFiniteInstrumentParameter,
    /// The linear calibration coefficient is not positive.
    NonPositiveDifc,
    /// A reflection d-spacing is not positive and finite.
    InvalidDSpacing,
    /// Calibration produced a non-finite position.
    InvalidPosition,
    /// The derived leading-edge exponential rate is invalid.
    NonPositiveAlpha,
    /// The derived trailing-edge exponential rate is invalid.
    NonPositiveBeta,
    /// The derived Gaussian variance is invalid.
    NonPositiveGaussianVariance,
    /// The derived Lorentzian width is invalid.
    NegativeLorentzianFwhm,
    /// The requested exponential tail cutoff is invalid.
    InvalidTailLog,
    /// Quadrature normalization failed.
    InvalidQuadrature,
    /// Component widths could not be transformed into a TCH shape.
    InvalidTch {
        /// Underlying component-width transformation failure.
        reason: TchError,
    },
    /// Reflection d-spacing and intensity arrays have different lengths.
    LengthMismatch,
    /// One reflection intensity is non-finite.
    NonFiniteIntensity {
        /// Index of the invalid reflection.
        reflection: usize,
    },
    /// Allocation-size arithmetic overflowed or allocation failed.
    AllocationOverflow,
    /// Grid or support validation failed.
    Profile {
        /// Underlying grid or support failure.
        reason: ProfileError,
    },
}

impl Display for TofError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInstrumentParameter => {
                write!(formatter, "TOF coefficients must be finite")
            }
            Self::NonPositiveDifc => write!(formatter, "difC must be positive"),
            Self::InvalidDSpacing => write!(formatter, "d-spacing must be positive and finite"),
            Self::InvalidPosition => write!(formatter, "derived TOF position must be finite"),
            Self::NonPositiveAlpha => write!(formatter, "TOF alpha must be positive and finite"),
            Self::NonPositiveBeta => write!(formatter, "TOF beta must be positive and finite"),
            Self::NonPositiveGaussianVariance => write!(
                formatter,
                "derived TOF Gaussian variance must be positive and finite"
            ),
            Self::NegativeLorentzianFwhm => write!(
                formatter,
                "derived TOF Lorentzian FWHM must be non-negative and finite"
            ),
            Self::InvalidTailLog => write!(formatter, "tail_log must be positive and finite"),
            Self::InvalidQuadrature => write!(formatter, "TOF quadrature normalization is invalid"),
            Self::InvalidTch { reason } => write!(formatter, "invalid TOF TCH widths: {reason}"),
            Self::LengthMismatch => {
                write!(formatter, "TOF reflection arrays must have equal length")
            }
            Self::NonFiniteIntensity { reflection } => write!(
                formatter,
                "reflection {reflection} intensity must be finite"
            ),
            Self::AllocationOverflow => write!(formatter, "TOF allocation size overflow"),
            Self::Profile { reason } => Display::fmt(reason, formatter),
        }
    }
}

impl Error for TofError {}

impl From<ProfileError> for TofError {
    fn from(reason: ProfileError) -> Self {
        Self::Profile { reason }
    }
}

struct PreparedReflection {
    parameters: TofProfileParameters,
    profile: TofProfile,
}

/// Fused finite-support TOF accumulation with local intensity/d-spacing rows.
///
/// # Errors
///
/// Returns [`TofError`] for invalid grids, reflection arrays, coefficients,
/// derived profiles, supports, or checked allocation failures.
#[allow(clippy::too_many_lines)]
pub fn accumulate_tof_batch(
    grid: GridView<'_>,
    d_spacings: &[f64],
    intensities: &[f64],
    instrument: TofInstrument,
    support_fwhm: f64,
    tail_log: f64,
) -> Result<Accumulation, TofError> {
    if d_spacings.len() != intensities.len() {
        return Err(TofError::LengthMismatch);
    }
    if !support_fwhm.is_finite() || support_fwhm <= 0.0 {
        return Err(TofError::Profile {
            reason: ProfileError::InvalidSupport,
        });
    }
    instrument.validate()?;
    let x = grid.as_slice();
    let count = d_spacings.len();
    let mut prepared = Vec::new();
    let mut starts = Vec::new();
    let mut offsets = Vec::new();
    prepared
        .try_reserve_exact(count)
        .map_err(|_| TofError::AllocationOverflow)?;
    starts
        .try_reserve_exact(count)
        .map_err(|_| TofError::AllocationOverflow)?;
    let offset_count = count.checked_add(1).ok_or(TofError::AllocationOverflow)?;
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| TofError::AllocationOverflow)?;
    offsets.push(0usize);
    let mut shared_quadrature = None;
    for reflection in 0..count {
        if !intensities[reflection].is_finite() {
            return Err(TofError::NonFiniteIntensity { reflection });
        }
        let parameters = TofProfileParameters::from_instrument(d_spacings[reflection], instrument)?;
        TofProfile::validate_rates(parameters.alpha_per_us, parameters.beta_per_us)?;
        let quadrature = if let Some(quadrature) = &shared_quadrature {
            Arc::clone(quadrature)
        } else {
            let quadrature = Arc::new(TofQuadrature::new(tail_log)?);
            shared_quadrature = Some(Arc::clone(&quadrature));
            quadrature
        };
        let profile = TofProfile::from_validated_rates(
            parameters.alpha_per_us,
            parameters.beta_per_us,
            TchWidths {
                gaussian_fwhm: parameters.gaussian_fwhm_us,
                lorentzian_fwhm: parameters.lorentzian_fwhm_us,
            },
            quadrature,
        )?;
        let base_radius = support_fwhm * parameters.tch.total_fwhm;
        let range = profile.support_range(parameters.position_us, base_radius);
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        offsets.push(
            offsets[reflection]
                .checked_add(upper - lower)
                .ok_or(TofError::AllocationOverflow)?,
        );
        starts.push(lower);
        prepared.push(PreparedReflection {
            parameters,
            profile,
        });
    }
    let active = offsets.last().copied().unwrap_or(0);
    let mut y = zeroed_f64_vec(x.len())?;
    let mut local = zeroed_f64_vec(
        active
            .checked_mul(LOCAL_PARAMETER_COUNT)
            .ok_or(TofError::AllocationOverflow)?,
    )?;
    let mut global = zeroed_f64_vec(
        TOF_GLOBAL_PARAMETER_COUNT
            .checked_mul(x.len())
            .ok_or(TofError::AllocationOverflow)?,
    )?;
    for reflection in 0..count {
        let item = &prepared[reflection];
        let intensity = intensities[reflection];
        let base_radius = support_fwhm * item.parameters.tch.total_fwhm;
        let begin = offsets[reflection];
        let end = offsets[reflection + 1];
        for active_index in begin..end {
            let sample = starts[reflection] + active_index - begin;
            let point = item
                .profile
                .evaluate_with_radius(x[sample] - item.parameters.position_us, base_radius);
            y[sample] += intensity * point.value;
            local[active_index * LOCAL_PARAMETER_COUNT] = point.value;
            local[active_index * LOCAL_PARAMETER_COUNT + 1] = intensity
                * (point.d_position * item.parameters.d_position_d_d
                    + point.d_alpha * item.parameters.d_alpha_d_d
                    + point.d_beta * item.parameters.d_beta_d_d
                    + point.d_gaussian_fwhm * item.parameters.d_gaussian_fwhm_d_d
                    + point.d_lorentzian_fwhm * item.parameters.d_lorentzian_fwhm_d_d);
            for parameter in 0..TOF_GLOBAL_PARAMETER_COUNT {
                let derivative = point.d_position
                    * item.parameters.d_position_d_instrument[parameter]
                    + point.d_alpha * item.parameters.d_alpha_d_instrument[parameter]
                    + point.d_beta * item.parameters.d_beta_d_instrument[parameter]
                    + point.d_gaussian_fwhm
                        * item.parameters.d_gaussian_fwhm_d_instrument[parameter]
                    + point.d_lorentzian_fwhm
                        * item.parameters.d_lorentzian_fwhm_d_instrument[parameter];
                global[parameter * x.len() + sample] += intensity * derivative;
            }
        }
    }
    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values: local,
                parameter_count: LOCAL_PARAMETER_COUNT,
            },
            global: Some(DenseJacobian {
                values: global,
                parameter_count: TOF_GLOBAL_PARAMETER_COUNT,
                sample_count: x.len(),
            }),
        },
        sample_count: x.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrument() -> TofInstrument {
        TofInstrument {
            zero_us: -0.773_346_536_757,
            difc_us_per_angstrom: 5_084.827_630_65,
            difa_us_per_angstrom2: -2.630_417_748_6,
            difb_us_angstrom: 0.0,
            alpha_coefficient: 5.0,
            beta0_per_us: 0.033_276_398_966_5,
            beta1_angstrom4_per_us: 0.000_964_057_827_372,
            betaq_angstrom2_per_us: 0.0,
            sigma0_us2: 0.0,
            sigma1_us2_per_angstrom2: 15.140_286_726_8,
            sigma2_us2_per_angstrom4: 0.0,
            sigmaq_us2_per_angstrom: 0.0,
            x_us_per_angstrom: 0.0,
            y_us_per_angstrom2: 0.0,
            z_us: 0.0,
        }
    }

    #[test]
    fn parameter_chains_match_centered_differences() {
        let d = 1.7;
        let step = 1.0e-6;
        let actual = TofProfileParameters::from_instrument(d, instrument()).expect("parameters");
        let plus = TofProfileParameters::from_instrument(d + step, instrument()).expect("plus");
        let minus = TofProfileParameters::from_instrument(d - step, instrument()).expect("minus");
        let finite = |high: f64, low: f64| (high - low) / (2.0 * step);
        assert!((actual.d_position_d_d - finite(plus.position_us, minus.position_us)).abs() < 1e-6);
        assert!((actual.d_alpha_d_d - finite(plus.alpha_per_us, minus.alpha_per_us)).abs() < 1e-9);
        assert!((actual.d_beta_d_d - finite(plus.beta_per_us, minus.beta_per_us)).abs() < 1e-9);
        assert!(
            (actual.d_gaussian_fwhm_d_d - finite(plus.gaussian_fwhm_us, minus.gaussian_fwhm_us))
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn profiles_can_share_quadrature_storage() {
        let quadrature = Arc::new(TofQuadrature::new(20.0).expect("quadrature"));
        let widths = TchWidths {
            gaussian_fwhm: 22.0,
            lorentzian_fwhm: 4.0,
        };
        let first = TofProfile::from_validated_rates(0.08, 0.03, widths, Arc::clone(&quadrature))
            .expect("first profile");
        let second = TofProfile::from_validated_rates(0.09, 0.04, widths, Arc::clone(&quadrature))
            .expect("second profile");

        assert!(Arc::ptr_eq(&first.quadrature, &second.quadrature));
        assert!(std::mem::size_of::<TofProfile>() < 128);
    }

    #[test]
    fn direct_profile_derivatives_match_centered_differences() {
        let alpha = 0.08;
        let beta = 0.03;
        let gaussian = 22.0;
        let lorentzian = 4.0;
        let delta = 5.0;
        let tail = 20.0;
        let point = TofProfile::new(
            alpha,
            beta,
            TchWidths {
                gaussian_fwhm: gaussian,
                lorentzian_fwhm: lorentzian,
            },
            tail,
        )
        .expect("profile")
        .evaluate(delta);
        let step = 1.0e-6;
        let value = |a, b, g, l, x| {
            TofProfile::new(
                a,
                b,
                TchWidths {
                    gaussian_fwhm: g,
                    lorentzian_fwhm: l,
                },
                tail,
            )
            .expect("profile")
            .evaluate(x)
            .value
        };
        let fd = |plus, minus| (plus - minus) / (2.0 * step);
        assert!(
            (point.d_position
                - fd(
                    value(alpha, beta, gaussian, lorentzian, delta - step),
                    value(alpha, beta, gaussian, lorentzian, delta + step)
                ))
            .abs()
                < 1e-8
        );
        assert!(
            (point.d_alpha
                - fd(
                    value(alpha + step, beta, gaussian, lorentzian, delta),
                    value(alpha - step, beta, gaussian, lorentzian, delta)
                ))
            .abs()
                < 1e-7
        );
        assert!(
            (point.d_beta
                - fd(
                    value(alpha, beta + step, gaussian, lorentzian, delta),
                    value(alpha, beta - step, gaussian, lorentzian, delta)
                ))
            .abs()
                < 1e-7
        );
        assert!(
            (point.d_gaussian_fwhm
                - fd(
                    value(alpha, beta, gaussian + step, lorentzian, delta),
                    value(alpha, beta, gaussian - step, lorentzian, delta)
                ))
            .abs()
                < 1e-8
        );
        assert!(
            (point.d_lorentzian_fwhm
                - fd(
                    value(alpha, beta, gaussian, lorentzian + step, delta),
                    value(alpha, beta, gaussian, lorentzian - step, delta)
                ))
            .abs()
                < 1e-8
        );
    }
}
