//! Explicit approximations for CW structural profiles; defaults preserve legacy arithmetic.

use crate::{ProfileError, SupportPolicy};

/// Reproducible, opt-in numerical controls independent of physical geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProfileAccuracy {
    /// Use lower-order Gauss--Legendre rules for very small axial spans.
    pub fast_fcj: bool,
    /// Maximum continuous discarded area of each normalized node profile.
    /// Overrides FWHM support when present; valid range is [1e-8, 0.1].
    pub tail_area_tolerance: Option<f64>,
}

impl ProfileAccuracy {
    /// Check the supported tail-area budget.
    /// # Errors
    /// Returns [`ProfileError::InvalidSupport`] for an invalid budget.
    pub fn validate(self) -> Result<(), ProfileError> {
        if self
            .tail_area_tolerance
            .is_some_and(|value| !value.is_finite() || !(1e-8..=0.1).contains(&value))
        {
            return Err(ProfileError::InvalidSupport);
        }
        Ok(())
    }

    /// Half-width in FWHM units whose conservative tail bound meets the budget.
    /// The Gaussian bound uses erfc(z) <= exp(-z²); the Lorentzian tail is exact.
    #[must_use]
    pub(crate) fn tail_multiple(tolerance: f64, eta: f64) -> f64 {
        let mut lower = 0.0;
        let mut upper = 1.0 / (std::f64::consts::PI * tolerance) + 3.0;
        for _ in 0..48 {
            let middle = lower + (upper - lower) * 0.5;
            let tail = eta * (2.0 / std::f64::consts::PI) * (0.5 / middle).atan()
                + (1.0 - eta) * (-4.0 * std::f64::consts::LN_2 * middle * middle).exp();
            if tail <= tolerance {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        upper
    }

    pub(crate) fn radius(self, fwhm: f64, eta: f64, support: SupportPolicy) -> f64 {
        self.tail_area_tolerance.map_or_else(
            || support.radius(fwhm),
            |tolerance| Self::tail_multiple(tolerance, eta) * fwhm,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_policy_validates_and_bounds_pure_components() {
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY, 0.100_01, 1e-9] {
            assert!(
                ProfileAccuracy {
                    fast_fcj: true,
                    tail_area_tolerance: Some(bad)
                }
                .validate()
                .is_err()
            );
        }
        for budget in [1e-8, 0.001, 0.01, 0.1] {
            let lorentz = ProfileAccuracy::tail_multiple(budget, 1.0);
            assert!(2.0 / std::f64::consts::PI * (0.5 / lorentz).atan() <= budget);
            let gaussian = ProfileAccuracy::tail_multiple(budget, 0.0);
            assert!((-4.0 * std::f64::consts::LN_2 * gaussian * gaussian).exp() <= budget);
        }
    }
}
