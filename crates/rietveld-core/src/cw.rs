//! Constant-wavelength U/V/W/X/Y profile broadening.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, ProfileError, SupportJacobian,
    SupportPolicy, zeroed_f64_vec,
};
use crate::tch::{TchShape, TchWidths};

const GAUSSIAN_FWHM_PER_SIGMA: f64 = 2.354_820_045_030_949_3;
const DEGREE_HALF_ANGLE_TO_RADIAN: f64 = std::f64::consts::PI / 360.0;
const LOCAL_PARAMETER_COUNT: usize = 2;
const GLOBAL_PARAMETER_COUNT: usize = 5;

/// Shared constant-wavelength instrument profile parameters in public units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantWavelengthInstrument {
    /// Radiation wavelength in ångströms.
    pub wavelength_angstrom: f64,
    /// Coefficient of `tan(theta)^2` in Gaussian variance, degrees squared.
    pub u_deg2: f64,
    /// Coefficient of `tan(theta)` in Gaussian variance, degrees squared.
    pub v_deg2: f64,
    /// Constant Gaussian variance, degrees squared.
    pub w_deg2: f64,
    /// Coefficient of `sec(theta)` in Lorentzian FWHM, degrees.
    pub x_deg: f64,
    /// Coefficient of `tan(theta)` in Lorentzian FWHM, degrees.
    pub y_deg: f64,
}

impl ConstantWavelengthInstrument {
    /// Validate the wavelength and finite profile coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`CwError`] if the wavelength is not positive and finite or a
    /// profile coefficient is non-finite.
    pub fn validate(self) -> Result<(), CwError> {
        if !self.wavelength_angstrom.is_finite() || self.wavelength_angstrom <= 0.0 {
            return Err(CwError::InvalidWavelength);
        }
        for (name, value) in [
            ("U", self.u_deg2),
            ("V", self.v_deg2),
            ("W", self.w_deg2),
            ("X", self.x_deg),
            ("Y", self.y_deg),
        ] {
            if !value.is_finite() {
                return Err(CwError::NonFiniteInstrumentParameter { name });
            }
        }
        Ok(())
    }
}

/// Derived component widths, TCH shape, and width derivatives for one reflection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CwProfileParameters {
    /// Gaussian variance in degrees squared.
    pub gaussian_variance_deg2: f64,
    /// Gaussian component FWHM in degrees.
    pub gaussian_fwhm_deg: f64,
    /// Lorentzian component FWHM in degrees.
    pub lorentzian_fwhm_deg: f64,
    /// Transformed TCH total width, eta, and derivatives.
    pub tch: TchShape,
    /// Gaussian-FWHM derivatives in `(U, V, W, X, Y)` order.
    pub d_gaussian_fwhm_d_instrument: [f64; GLOBAL_PARAMETER_COUNT],
    /// Lorentzian-FWHM derivatives in `(U, V, W, X, Y)` order.
    pub d_lorentzian_fwhm_d_instrument: [f64; GLOBAL_PARAMETER_COUNT],
    /// Gaussian-FWHM derivative with respect to reflection `two_theta` in degrees.
    pub d_gaussian_fwhm_d_two_theta: f64,
    /// Lorentzian-FWHM derivative with respect to reflection `two_theta` in degrees.
    pub d_lorentzian_fwhm_d_two_theta: f64,
}

impl CwProfileParameters {
    /// Derive profile parameters for one reflection position.
    ///
    /// # Errors
    ///
    /// Returns [`CwError`] if the instrument, angle, or a derived component
    /// width is outside its domain.
    pub fn from_instrument(
        two_theta_deg: f64,
        instrument: ConstantWavelengthInstrument,
    ) -> Result<Self, CwError> {
        instrument.validate()?;
        Self::from_validated_instrument(two_theta_deg, instrument)
    }

    fn from_validated_instrument(
        two_theta_deg: f64,
        instrument: ConstantWavelengthInstrument,
    ) -> Result<Self, CwError> {
        validate_two_theta(two_theta_deg)?;
        let theta = two_theta_deg * DEGREE_HALF_ANGLE_TO_RADIAN;
        let tangent = theta.tan();
        let secant = theta.cos().recip();
        let tangent_2 = tangent * tangent;
        let gaussian_variance_deg2 =
            instrument.u_deg2 * tangent_2 + instrument.v_deg2 * tangent + instrument.w_deg2;
        if !gaussian_variance_deg2.is_finite() || gaussian_variance_deg2 <= 0.0 {
            return Err(CwError::NonPositiveGaussianVariance);
        }
        let gaussian_sigma = gaussian_variance_deg2.sqrt();
        let gaussian_fwhm_deg = GAUSSIAN_FWHM_PER_SIGMA * gaussian_sigma;
        let lorentzian_fwhm_deg = instrument.x_deg * secant + instrument.y_deg * tangent;
        if !lorentzian_fwhm_deg.is_finite() || lorentzian_fwhm_deg < 0.0 {
            return Err(CwError::NegativeLorentzianFwhm);
        }
        let tch = TchShape::from_component_fwhm(TchWidths {
            gaussian_fwhm: gaussian_fwhm_deg,
            lorentzian_fwhm: lorentzian_fwhm_deg,
        })
        .map_err(|_| CwError::InvalidTchTransform)?;

        let d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * gaussian_sigma);
        let d_gaussian_fwhm_d_instrument = [
            d_gaussian_d_variance * tangent_2,
            d_gaussian_d_variance * tangent,
            d_gaussian_d_variance,
            0.0,
            0.0,
        ];
        let d_lorentzian_fwhm_d_instrument = [0.0, 0.0, 0.0, secant, tangent];
        let d_tangent_d_two_theta = DEGREE_HALF_ANGLE_TO_RADIAN * secant * secant;
        let d_secant_d_two_theta = DEGREE_HALF_ANGLE_TO_RADIAN * secant * tangent;
        let d_variance_d_two_theta =
            (2.0 * instrument.u_deg2 * tangent + instrument.v_deg2) * d_tangent_d_two_theta;
        let d_gaussian_fwhm_d_two_theta = d_gaussian_d_variance * d_variance_d_two_theta;
        let d_lorentzian_fwhm_d_two_theta =
            instrument.x_deg * d_secant_d_two_theta + instrument.y_deg * d_tangent_d_two_theta;

        Ok(Self {
            gaussian_variance_deg2,
            gaussian_fwhm_deg,
            lorentzian_fwhm_deg,
            tch,
            d_gaussian_fwhm_d_instrument,
            d_lorentzian_fwhm_d_instrument,
            d_gaussian_fwhm_d_two_theta,
            d_lorentzian_fwhm_d_two_theta,
        })
    }
}

/// Constant-wavelength profile domain errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CwError {
    /// Wavelength is not positive and finite.
    InvalidWavelength,
    /// One instrument coefficient is non-finite.
    NonFiniteInstrumentParameter {
        /// Conventional coefficient name.
        name: &'static str,
    },
    /// Reflection position is non-finite or outside `(0, 180)` degrees.
    InvalidTwoTheta,
    /// Derived Gaussian variance is not positive and finite.
    NonPositiveGaussianVariance,
    /// Derived Lorentzian FWHM is negative or non-finite.
    NegativeLorentzianFwhm,
    /// The downstream TCH transform could not represent the widths.
    InvalidTchTransform,
}

impl Display for CwError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWavelength => write!(formatter, "wavelength must be positive and finite"),
            Self::NonFiniteInstrumentParameter { name } => {
                write!(formatter, "instrument parameter {name} must be finite")
            }
            Self::InvalidTwoTheta => {
                write!(
                    formatter,
                    "two_theta must be finite and within (0, 180) degrees"
                )
            }
            Self::NonPositiveGaussianVariance => {
                write!(
                    formatter,
                    "derived Gaussian variance must be positive and finite"
                )
            }
            Self::NegativeLorentzianFwhm => {
                write!(
                    formatter,
                    "derived Lorentzian FWHM must be non-negative and finite"
                )
            }
            Self::InvalidTchTransform => write!(formatter, "derived widths fail the TCH transform"),
        }
    }
}

impl Error for CwError {}

/// Errors while validating or accumulating a constant-wavelength batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CwBatchError {
    /// Reflection position and intensity arrays have different lengths.
    ReflectionLengthMismatch,
    /// An integrated intensity is non-finite.
    NonFiniteIntensity {
        /// Index of the invalid reflection.
        reflection: usize,
    },
    /// The shared instrument model is invalid.
    InvalidInstrument {
        /// Instrument validation failure.
        reason: CwError,
    },
    /// A reflection angle or its derived widths are invalid.
    InvalidReflection {
        /// Index of the invalid reflection.
        reflection: usize,
        /// Reflection-specific validation failure.
        reason: CwError,
    },
    /// Generic grid, support, or allocation failure from the accumulator.
    Accumulation {
        /// Underlying generic profile error.
        reason: ProfileError,
    },
}

impl Display for CwBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReflectionLengthMismatch => write!(
                formatter,
                "two_theta positions and integrated intensities must have equal length"
            ),
            Self::NonFiniteIntensity { reflection } => {
                write!(
                    formatter,
                    "reflection {reflection} intensity must be finite"
                )
            }
            Self::InvalidInstrument { reason } => {
                write!(
                    formatter,
                    "invalid constant-wavelength instrument: {reason}"
                )
            }
            Self::InvalidReflection { reflection, reason } => write!(
                formatter,
                "constant-wavelength reflection {reflection} is invalid: {reason}"
            ),
            Self::Accumulation { reason } => Display::fmt(reason, formatter),
        }
    }
}

impl Error for CwBatchError {}

impl From<ProfileError> for CwBatchError {
    fn from(reason: ProfileError) -> Self {
        Self::Accumulation { reason }
    }
}

/// Validated borrowed constant-wavelength reflection arrays.
#[derive(Clone, Copy, Debug)]
pub struct CwReflectionBatchView<'a> {
    two_theta_deg: &'a [f64],
    intensities: &'a [f64],
}

impl<'a> CwReflectionBatchView<'a> {
    /// Validate and borrow reflection positions and integrated intensities.
    ///
    /// # Errors
    ///
    /// Returns [`CwBatchError`] if array lengths differ, a position is outside
    /// `(0, 180)` degrees, or an intensity is non-finite.
    pub fn new(two_theta_deg: &'a [f64], intensities: &'a [f64]) -> Result<Self, CwBatchError> {
        if two_theta_deg.len() != intensities.len() {
            return Err(CwBatchError::ReflectionLengthMismatch);
        }
        for reflection in 0..two_theta_deg.len() {
            validate_two_theta(two_theta_deg[reflection])
                .map_err(|reason| CwBatchError::InvalidReflection { reflection, reason })?;
            if !intensities[reflection].is_finite() {
                return Err(CwBatchError::NonFiniteIntensity { reflection });
            }
        }
        Ok(Self {
            two_theta_deg,
            intensities,
        })
    }

    /// Number of reflections.
    #[must_use]
    pub const fn len(self) -> usize {
        self.two_theta_deg.len()
    }

    /// Whether there are no reflections.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.two_theta_deg.is_empty()
    }
}

/// Accumulate a CW reflection batch with sparse local and dense global derivatives.
///
/// Local derivative order is intensity and position. Dense global derivative
/// order is U, V, W, X, and Y.
///
/// # Errors
///
/// Returns [`CwBatchError`] if the instrument produces an invalid width for a
/// reflection, support is invalid, or allocation fails.
pub fn accumulate_cw_batch(
    grid: GridView<'_>,
    reflections: CwReflectionBatchView<'_>,
    instrument: ConstantWavelengthInstrument,
    support: SupportPolicy,
) -> Result<Accumulation, CwBatchError> {
    support.validate()?;
    instrument
        .validate()
        .map_err(|reason| CwBatchError::InvalidInstrument { reason })?;
    let x = grid.as_slice();
    let reflection_count = reflections.len();
    let mut parameters = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    parameters
        .try_reserve_exact(reflection_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    starts
        .try_reserve_exact(reflection_count)
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets
        .try_reserve_exact(
            reflection_count
                .checked_add(1)
                .ok_or(ProfileError::AllocationOverflow)?,
        )
        .map_err(|_| ProfileError::AllocationOverflow)?;
    offsets.push(0);

    for reflection in 0..reflection_count {
        let profile = CwProfileParameters::from_validated_instrument(
            reflections.two_theta_deg[reflection],
            instrument,
        )
        .map_err(|reason| CwBatchError::InvalidReflection { reflection, reason })?;
        let range = support.range(
            reflections.two_theta_deg[reflection],
            profile.tch.total_fwhm,
        );
        let lower = x.partition_point(|value| *value < range.left);
        let upper = x.partition_point(|value| *value <= range.right);
        let next_offset = offsets[reflection]
            .checked_add(upper - lower)
            .ok_or(ProfileError::AllocationOverflow)?;
        parameters.push(profile);
        starts.push(lower);
        offsets.push(next_offset);
    }

    let active_sample_count = offsets.last().copied().unwrap_or(0);
    let local_value_count = active_sample_count
        .checked_mul(LOCAL_PARAMETER_COUNT)
        .ok_or(ProfileError::AllocationOverflow)?;
    let global_value_count = GLOBAL_PARAMETER_COUNT
        .checked_mul(x.len())
        .ok_or(ProfileError::AllocationOverflow)?;
    let mut y = zeroed_f64_vec(x.len())?;
    let mut local_values = zeroed_f64_vec(local_value_count)?;
    let mut global_values = zeroed_f64_vec(global_value_count)?;

    for reflection in 0..reflection_count {
        let start = starts[reflection];
        let active_begin = offsets[reflection];
        let active_end = offsets[reflection + 1];
        let profile = parameters[reflection];
        let intensity = reflections.intensities[reflection];
        for active_index in active_begin..active_end {
            let sample = start + active_index - active_begin;
            let point = profile
                .tch
                .evaluate(x[sample] - reflections.two_theta_deg[reflection]);
            y[sample] += intensity * point.value;
            let local_base = active_index * LOCAL_PARAMETER_COUNT;
            local_values[local_base] = point.value;
            local_values[local_base + 1] = intensity
                * (-point.d_delta
                    + point.d_gaussian_fwhm * profile.d_gaussian_fwhm_d_two_theta
                    + point.d_lorentzian_fwhm * profile.d_lorentzian_fwhm_d_two_theta);
            for parameter in 0..GLOBAL_PARAMETER_COUNT {
                let derivative = point.d_gaussian_fwhm
                    * profile.d_gaussian_fwhm_d_instrument[parameter]
                    + point.d_lorentzian_fwhm * profile.d_lorentzian_fwhm_d_instrument[parameter];
                global_values[parameter * x.len() + sample] += intensity * derivative;
            }
        }
    }

    Ok(Accumulation {
        y,
        derivatives: PatternDerivatives {
            local: SupportJacobian {
                starts,
                offsets,
                values: local_values,
                parameter_count: LOCAL_PARAMETER_COUNT,
            },
            global: Some(DenseJacobian {
                values: global_values,
                parameter_count: GLOBAL_PARAMETER_COUNT,
                sample_count: x.len(),
            }),
        },
        sample_count: x.len(),
    })
}

fn validate_two_theta(two_theta_deg: f64) -> Result<(), CwError> {
    if !two_theta_deg.is_finite() || !(0.0..180.0).contains(&two_theta_deg) || two_theta_deg == 0.0
    {
        return Err(CwError::InvalidTwoTheta);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrument() -> ConstantWavelengthInstrument {
        ConstantWavelengthInstrument {
            wavelength_angstrom: 1.5406,
            u_deg2: 2e-4,
            v_deg2: -1e-4,
            w_deg2: 1e-4,
            x_deg: 1e-3,
            y_deg: 2e-3,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.1e}"
        );
    }

    fn assert_relative_close(actual: f64, expected: f64, relative_tolerance: f64) {
        let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
        assert_close(actual, expected, relative_tolerance * scale);
    }

    #[test]
    fn gsas_unit_converted_width_formula_is_exact() {
        let position = 19.712_609_2;
        let profile = CwProfileParameters::from_instrument(position, instrument()).expect("valid");
        assert_close(profile.gaussian_variance_deg2 * 1e4, 0.886_630_51, 5e-9);
        let theta = position * DEGREE_HALF_ANGLE_TO_RADIAN;
        let expected_lorentzian = 1e-3 / theta.cos() + 2e-3 * theta.tan();
        assert_close(profile.lorentzian_fwhm_deg, expected_lorentzian, 1e-18);
    }

    #[test]
    fn derived_width_derivatives_match_centered_differences() {
        let position = 63.2;
        let baseline = CwProfileParameters::from_instrument(position, instrument()).expect("valid");
        let instrument_step = 1e-8;
        for parameter in 0..GLOBAL_PARAMETER_COUNT {
            let mut plus = instrument();
            let mut minus = instrument();
            let plus_parameter = match parameter {
                0 => &mut plus.u_deg2,
                1 => &mut plus.v_deg2,
                2 => &mut plus.w_deg2,
                3 => &mut plus.x_deg,
                _ => &mut plus.y_deg,
            };
            *plus_parameter += instrument_step;
            let minus_parameter = match parameter {
                0 => &mut minus.u_deg2,
                1 => &mut minus.v_deg2,
                2 => &mut minus.w_deg2,
                3 => &mut minus.x_deg,
                _ => &mut minus.y_deg,
            };
            *minus_parameter -= instrument_step;
            let plus_profile = CwProfileParameters::from_instrument(position, plus).expect("plus");
            let minus_profile =
                CwProfileParameters::from_instrument(position, minus).expect("minus");
            assert_relative_close(
                baseline.d_gaussian_fwhm_d_instrument[parameter],
                (plus_profile.gaussian_fwhm_deg - minus_profile.gaussian_fwhm_deg)
                    / (2.0 * instrument_step),
                2e-8,
            );
            assert_close(
                baseline.d_lorentzian_fwhm_d_instrument[parameter],
                (plus_profile.lorentzian_fwhm_deg - minus_profile.lorentzian_fwhm_deg)
                    / (2.0 * instrument_step),
                2e-10,
            );
        }
        let position_step = 1e-5;
        let plus = CwProfileParameters::from_instrument(position + position_step, instrument())
            .expect("+");
        let minus = CwProfileParameters::from_instrument(position - position_step, instrument())
            .expect("-");
        assert_close(
            baseline.d_gaussian_fwhm_d_two_theta,
            (plus.gaussian_fwhm_deg - minus.gaussian_fwhm_deg) / (2.0 * position_step),
            2e-10,
        );
        assert_close(
            baseline.d_lorentzian_fwhm_d_two_theta,
            (plus.lorentzian_fwhm_deg - minus.lorentzian_fwhm_deg) / (2.0 * position_step),
            2e-11,
        );
    }

    #[test]
    fn invalid_derived_widths_are_errors() {
        assert_eq!(
            CwProfileParameters::from_instrument(
                30.0,
                ConstantWavelengthInstrument {
                    w_deg2: -1.0,
                    ..instrument()
                },
            ),
            Err(CwError::NonPositiveGaussianVariance)
        );
        assert_eq!(
            CwProfileParameters::from_instrument(
                30.0,
                ConstantWavelengthInstrument {
                    x_deg: -1.0,
                    y_deg: 0.0,
                    ..instrument()
                },
            ),
            Err(CwError::NegativeLorentzianFwhm)
        );
    }
}
