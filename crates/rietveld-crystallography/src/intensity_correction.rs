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
}

/// Invalid correction model or reflection geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegratedIntensityCorrectionError {
    /// Wavelength is not positive and finite.
    InvalidWavelength,
    /// A reciprocal squared length is not positive and finite.
    InvalidQSquared,
    /// A reflection does not satisfy `0 < 2theta < 180°` for the wavelength.
    ReflectionOutsideAngularDomain,
}

impl Display for IntegratedIntensityCorrectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidWavelength => "correction wavelength must be positive and finite",
            Self::InvalidQSquared => "correction q_squared must be positive and finite",
            Self::ReflectionOutsideAngularDomain => {
                "Bragg-Brentano LP requires reflections strictly within 0 < 2theta < 180 degrees"
            }
        })
    }
}

impl Error for IntegratedIntensityCorrectionError {}

impl IntegratedIntensityCorrectionModel {
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
            }),
            Self::BraggBrentanoUnpolarizedLp {
                wavelength_angstrom,
            } => {
                if !wavelength_angstrom.is_finite() || wavelength_angstrom <= 0.0 {
                    return Err(IntegratedIntensityCorrectionError::InvalidWavelength);
                }
                let mut values = Vec::with_capacity(q_squared_inverse_angstrom2.len());
                let mut derivatives = Vec::with_capacity(q_squared_inverse_angstrom2.len());
                for &q_squared in q_squared_inverse_angstrom2 {
                    let (value, derivative) = bragg_brentano_lp(q_squared, wavelength_angstrom)?;
                    values.push(value);
                    derivatives.push(derivative);
                }
                Ok(IntegratedIntensityCorrection {
                    values,
                    d_values_d_q_squared: derivatives,
                })
            }
        }
    }
}

fn bragg_brentano_lp(
    q_squared: f64,
    wavelength: f64,
) -> Result<(f64, f64), IntegratedIntensityCorrectionError> {
    let root_q = q_squared.sqrt();
    let sin_theta = 0.5 * wavelength * root_q;
    if !(0.0..1.0).contains(&sin_theta) {
        return Err(IntegratedIntensityCorrectionError::ReflectionOutsideAngularDomain);
    }
    let theta = sin_theta.asin();
    let cos_theta = theta.cos();
    let two_theta = 2.0 * theta;
    let (sin_two_theta, cos_two_theta) = two_theta.sin_cos();
    let numerator = 1.0 + cos_two_theta * cos_two_theta;
    let value = numerator / (2.0 * sin_theta * sin_theta * cos_theta);
    let d_log_d_two_theta =
        -2.0 * sin_two_theta * cos_two_theta / numerator - 1.0 / theta.tan() + 0.5 * theta.tan();
    let d_two_theta_d_q_squared = wavelength / (2.0 * root_q * cos_theta);
    Ok((value, value * d_log_d_two_theta * d_two_theta_d_q_squared))
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
    }
}
