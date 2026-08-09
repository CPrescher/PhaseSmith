//! Typed integrated-reflection intensity corrections.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// One correction value and its reciprocal-metric derivative per reflection.
#[derive(Clone, Debug, PartialEq)]
pub struct IntegratedIntensityCorrection {
    /// Multiplicative integrated-intensity correction `C_h`.
    pub values: Vec<f64>,
    /// Analytical derivative `d C_h / d(q²)`.
    pub d_values_d_q_squared: Vec<f64>,
    /// Analytical derivative `d C_h / d(lambda)` at fixed `q²`.
    pub d_values_d_wavelength: Vec<f64>,
}

/// Explicit integrated-intensity geometry model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntegratedIntensityCorrectionModel {
    /// Raw multiplicity-weighted structural intensity, `C_h = 1`.
    Neutral,
    /// Monochromatic unpolarized symmetric Bragg--Brentano integrated LP.
    BraggBrentanoUnpolarizedLp {
        /// Monochromatic wavelength in ångströms.
        wavelength_angstrom: f64,
    },
    /// Monochromatic polarized symmetric Bragg--Brentano integrated LP.
    BraggBrentanoPolarizedLp {
        /// Monochromatic wavelength in ångströms.
        wavelength_angstrom: f64,
        /// Fraction in the constant polarization term, constrained to `[0, 1]`.
        polarization: f64,
    },
    /// Monochromatic constant-wavelength neutron powder Lorentz factor.
    ConstantWavelengthNeutronLorentz {
        /// Monochromatic wavelength in ångströms.
        wavelength_angstrom: f64,
    },
}

/// Invalid correction model or reflection geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegratedIntensityCorrectionError {
    /// Wavelength is not positive and finite.
    InvalidWavelength,
    /// Polarization is not finite or lies outside `[0, 1]`.
    InvalidPolarization,
    /// A reciprocal squared length is not positive and finite.
    InvalidQSquared,
    /// A reflection does not satisfy `0 < 2theta < 180°` for the wavelength.
    ReflectionOutsideAngularDomain,
}

impl Display for IntegratedIntensityCorrectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidWavelength => "correction wavelength must be positive and finite",
            Self::InvalidPolarization => {
                "Bragg-Brentano polarization must be finite and within [0, 1]"
            }
            Self::InvalidQSquared => "correction q_squared must be positive and finite",
            Self::ReflectionOutsideAngularDomain => {
                "Bragg-Brentano LP requires reflections strictly within 0 < 2theta < 180 degrees"
            }
        })
    }
}

impl Error for IntegratedIntensityCorrectionError {}

impl IntegratedIntensityCorrectionModel {
    /// Return the same correction family evaluated at another wavelength.
    #[must_use]
    pub const fn with_wavelength(self, wavelength_angstrom: f64) -> Self {
        match self {
            Self::Neutral => Self::Neutral,
            Self::BraggBrentanoUnpolarizedLp { .. } => Self::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            },
            Self::BraggBrentanoPolarizedLp { polarization, .. } => Self::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                polarization,
            },
            Self::ConstantWavelengthNeutronLorentz { .. } => {
                Self::ConstantWavelengthNeutronLorentz {
                    wavelength_angstrom,
                }
            }
        }
    }

    /// Evaluate values and `q²` derivatives together for one reflection batch.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratedIntensityCorrectionError`] for invalid wavelength,
    /// reciprocal lengths, or inaccessible Bragg angles.
    pub fn evaluate(
        self,
        q_squared_inverse_angstrom2: &[f64],
    ) -> Result<IntegratedIntensityCorrection, IntegratedIntensityCorrectionError> {
        if q_squared_inverse_angstrom2
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(IntegratedIntensityCorrectionError::InvalidQSquared);
        }
        match self {
            Self::Neutral => Ok(IntegratedIntensityCorrection {
                values: vec![1.0; q_squared_inverse_angstrom2.len()],
                d_values_d_q_squared: vec![0.0; q_squared_inverse_angstrom2.len()],
                d_values_d_wavelength: vec![0.0; q_squared_inverse_angstrom2.len()],
            }),
            Self::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            }
            | Self::BraggBrentanoPolarizedLp {
                wavelength_angstrom,
                ..
            }
            | Self::ConstantWavelengthNeutronLorentz {
                wavelength_angstrom,
            } => {
                if !wavelength_angstrom.is_finite() || wavelength_angstrom <= 0.0 {
                    return Err(IntegratedIntensityCorrectionError::InvalidWavelength);
                }
                let (polarization, factor) = match self {
                    Self::BraggBrentanoUnpolarizedLp { .. } => (0.5, 1.0),
                    Self::BraggBrentanoPolarizedLp { polarization, .. } => (polarization, 1.0),
                    Self::ConstantWavelengthNeutronLorentz { .. } => (1.0, 0.5),
                    Self::Neutral => unreachable!(),
                };
                if !polarization.is_finite() || !(0.0..=1.0).contains(&polarization) {
                    return Err(IntegratedIntensityCorrectionError::InvalidPolarization);
                }
                let mut values = Vec::with_capacity(q_squared_inverse_angstrom2.len());
                let mut derivatives = Vec::with_capacity(q_squared_inverse_angstrom2.len());
                let mut wavelength_derivatives =
                    Vec::with_capacity(q_squared_inverse_angstrom2.len());
                for &q_squared in q_squared_inverse_angstrom2 {
                    let (value, derivative, wavelength_derivative) =
                        bragg_brentano_lp(q_squared, wavelength_angstrom, polarization)?;
                    values.push(factor * value);
                    derivatives.push(factor * derivative);
                    wavelength_derivatives.push(factor * wavelength_derivative);
                }
                Ok(IntegratedIntensityCorrection {
                    values,
                    d_values_d_q_squared: derivatives,
                    d_values_d_wavelength: wavelength_derivatives,
                })
            }
        }
    }
}

fn bragg_brentano_lp(
    q_squared: f64,
    wavelength: f64,
    polarization: f64,
) -> Result<(f64, f64, f64), IntegratedIntensityCorrectionError> {
    let root_q = q_squared.sqrt();
    let sin_theta = 0.5 * wavelength * root_q;
    if !(0.0..1.0).contains(&sin_theta) {
        return Err(IntegratedIntensityCorrectionError::ReflectionOutsideAngularDomain);
    }
    let theta = sin_theta.asin();
    let cos_theta = theta.cos();
    let two_theta = 2.0 * theta;
    let (sin_two_theta, cos_two_theta) = two_theta.sin_cos();
    let numerator = polarization + (1.0 - polarization) * cos_two_theta * cos_two_theta;
    let value = numerator / (sin_theta * sin_theta * cos_theta);
    let d_log_d_two_theta = -2.0 * (1.0 - polarization) * sin_two_theta * cos_two_theta / numerator
        - 1.0 / theta.tan()
        + 0.5 * theta.tan();
    let d_two_theta_d_q_squared = wavelength / (2.0 * root_q * cos_theta);
    let d_two_theta_d_wavelength = root_q / cos_theta;
    Ok((
        value,
        value * d_log_d_two_theta * d_two_theta_d_q_squared,
        value * d_log_d_two_theta * d_two_theta_d_wavelength,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_model_is_exact_for_empty_and_nonempty_batches() {
        let empty = IntegratedIntensityCorrectionModel::Neutral
            .evaluate(&[])
            .expect("empty neutral");
        assert!(empty.values.is_empty());
        let values = IntegratedIntensityCorrectionModel::Neutral
            .evaluate(&[0.01, 0.1, 1.0])
            .expect("neutral");
        assert_eq!(values.values, vec![1.0; 3]);
        assert_eq!(values.d_values_d_q_squared, vec![0.0; 3]);
        assert_eq!(values.d_values_d_wavelength, vec![0.0; 3]);
    }

    #[test]
    fn bragg_brentano_values_and_derivatives_match_closed_form_differences() {
        let wavelength = 1.5406;
        let model = IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom: wavelength,
        };
        let q_squared = [0.02, 0.11, 0.37, 0.91];
        let actual = model.evaluate(&q_squared).expect("LP values");
        for (index, value) in q_squared.into_iter().enumerate() {
            let theta = (0.5 * wavelength * value.sqrt()).asin();
            let expected =
                (1.0 + (2.0 * theta).cos().powi(2)) / (2.0 * theta.sin().powi(2) * theta.cos());
            assert!((actual.values[index] - expected).abs() < 2.0e-14 * expected);
            let step = 1.0e-6 * value;
            let plus = model.evaluate(&[value + step]).expect("plus").values[0];
            let minus = model.evaluate(&[value - step]).expect("minus").values[0];
            let finite_difference = (plus - minus) / (2.0 * step);
            assert!(
                (actual.d_values_d_q_squared[index] - finite_difference).abs()
                    < 2.0e-8 * finite_difference.abs().max(1.0)
            );
            let wavelength_step = 1.0e-6 * wavelength;
            let plus = IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom: wavelength + wavelength_step,
            }
            .evaluate(&[value])
            .expect("wavelength plus")
            .values[0];
            let minus = IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom: wavelength - wavelength_step,
            }
            .evaluate(&[value])
            .expect("wavelength minus")
            .values[0];
            let finite_difference = (plus - minus) / (2.0 * wavelength_step);
            assert!(
                (actual.d_values_d_wavelength[index] - finite_difference).abs()
                    < 2.0e-8 * finite_difference.abs().max(1.0)
            );
        }
    }

    #[test]
    fn polarized_lp_matches_closed_form_derivatives_and_unpolarized_limit() {
        let wavelength = 1.54051;
        let polarization = 0.7;
        let q_squared = [0.03, 0.19, 0.62];
        let model = IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom: wavelength,
            polarization,
        };
        let actual = model.evaluate(&q_squared).expect("polarized LP");
        for (index, value) in q_squared.into_iter().enumerate() {
            let theta = (0.5 * wavelength * value.sqrt()).asin();
            let expected = (polarization + (1.0 - polarization) * (2.0 * theta).cos().powi(2))
                / (theta.sin().powi(2) * theta.cos());
            assert!((actual.values[index] - expected).abs() < 2.0e-14 * expected);
            let q_step = value * 1.0e-6;
            let q_finite = (model.evaluate(&[value + q_step]).expect("q plus").values[0]
                - model.evaluate(&[value - q_step]).expect("q minus").values[0])
                / (2.0 * q_step);
            assert!(
                (actual.d_values_d_q_squared[index] - q_finite).abs()
                    < 2.0e-8 * q_finite.abs().max(1.0)
            );
            let wavelength_step = wavelength * 1.0e-6;
            let evaluate_at = |selected| {
                IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                    wavelength_angstrom: selected,
                    polarization,
                }
                .evaluate(&[value])
                .expect("wavelength finite difference")
                .values[0]
            };
            let wavelength_finite = (evaluate_at(wavelength + wavelength_step)
                - evaluate_at(wavelength - wavelength_step))
                / (2.0 * wavelength_step);
            assert!(
                (actual.d_values_d_wavelength[index] - wavelength_finite).abs()
                    < 2.0e-8 * wavelength_finite.abs().max(1.0)
            );
        }
        let unpolarized = IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
            wavelength_angstrom: wavelength,
        }
        .evaluate(&q_squared)
        .expect("unpolarized");
        let half = IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
            wavelength_angstrom: wavelength,
            polarization: 0.5,
        }
        .evaluate(&q_squared)
        .expect("half polarized");
        assert_eq!(half, unpolarized);
    }

    #[test]
    fn neutron_lorentz_matches_constant_wavelength_powder_equation() {
        let wavelength = 1.909;
        let q_squared = [0.03, 0.19, 0.62];
        let model = IntegratedIntensityCorrectionModel::ConstantWavelengthNeutronLorentz {
            wavelength_angstrom: wavelength,
        };
        let actual = model.evaluate(&q_squared).expect("neutron Lorentz values");
        for (index, value) in q_squared.into_iter().enumerate() {
            let theta = (0.5 * wavelength * value.sqrt()).asin();
            let expected = 1.0 / (theta.sin() * (2.0 * theta).sin());
            assert!((actual.values[index] - expected).abs() < 2.0e-14 * expected);
            let step = 1.0e-6 * value;
            let finite_difference = (model.evaluate(&[value + step]).unwrap().values[0]
                - model.evaluate(&[value - step]).unwrap().values[0])
                / (2.0 * step);
            assert!(
                (actual.d_values_d_q_squared[index] - finite_difference).abs()
                    < 2.0e-8 * finite_difference.abs().max(1.0)
            );
        }
    }

    #[test]
    fn invalid_domains_are_explicit() {
        assert_eq!(
            IntegratedIntensityCorrectionModel::Neutral.evaluate(&[0.0]),
            Err(IntegratedIntensityCorrectionError::InvalidQSquared)
        );
        assert_eq!(
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom: 0.0,
            }
            .evaluate(&[1.0]),
            Err(IntegratedIntensityCorrectionError::InvalidWavelength)
        );
        assert_eq!(
            IntegratedIntensityCorrectionModel::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom: 2.0,
            }
            .evaluate(&[1.0]),
            Err(IntegratedIntensityCorrectionError::ReflectionOutsideAngularDomain)
        );
        assert_eq!(
            IntegratedIntensityCorrectionModel::BraggBrentanoPolarizedLp {
                wavelength_angstrom: 1.0,
                polarization: 1.1,
            }
            .evaluate(&[1.0]),
            Err(IntegratedIntensityCorrectionError::InvalidPolarization)
        );
    }
}
