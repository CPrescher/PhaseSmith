//! Wavelength-component models for constant-wavelength radiation.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors in a wavelength-component array model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavelengthComponentsError {
    /// Wavelength and relative-intensity arrays differ in length.
    LengthMismatch,
    /// At least one component is required.
    Empty,
    /// A wavelength is not positive and finite.
    InvalidWavelength {
        /// Index of the invalid component.
        component: usize,
    },
    /// A relative integrated intensity is negative or non-finite.
    InvalidRelativeIntensity {
        /// Index of the invalid component.
        component: usize,
    },
    /// The reference component must have positive relative intensity.
    NonPositiveReferenceIntensity,
}

impl Display for WavelengthComponentsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch => write!(
                formatter,
                "wavelength and relative-intensity arrays must have equal length"
            ),
            Self::Empty => write!(formatter, "at least one wavelength component is required"),
            Self::InvalidWavelength { component } => write!(
                formatter,
                "wavelength component {component} must be positive and finite"
            ),
            Self::InvalidRelativeIntensity { component } => write!(
                formatter,
                "relative intensity component {component} must be non-negative and finite"
            ),
            Self::NonPositiveReferenceIntensity => write!(
                formatter,
                "reference component relative intensity must be positive"
            ),
        }
    }
}

impl Error for WavelengthComponentsError {}

/// Validated borrowed wavelength and relative integrated-intensity arrays.
///
/// Component zero is the reference. Secondary wavelength and intensity
/// parameters are represented as ratios to component zero. Relative
/// intensities are normalized to unit sum during accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavelengthComponentsView<'a> {
    wavelengths_angstrom: &'a [f64],
    relative_intensities: &'a [f64],
}

impl<'a> WavelengthComponentsView<'a> {
    /// Validate and borrow a wavelength-component model.
    ///
    /// # Errors
    ///
    /// Returns [`WavelengthComponentsError`] for unequal/empty arrays,
    /// invalid wavelengths or intensities, or a zero reference intensity.
    pub fn new(
        wavelengths_angstrom: &'a [f64],
        relative_intensities: &'a [f64],
    ) -> Result<Self, WavelengthComponentsError> {
        if wavelengths_angstrom.len() != relative_intensities.len() {
            return Err(WavelengthComponentsError::LengthMismatch);
        }
        if wavelengths_angstrom.is_empty() {
            return Err(WavelengthComponentsError::Empty);
        }
        for (component, wavelength) in wavelengths_angstrom.iter().copied().enumerate() {
            if !wavelength.is_finite() || wavelength <= 0.0 {
                return Err(WavelengthComponentsError::InvalidWavelength { component });
            }
        }
        for (component, intensity) in relative_intensities.iter().copied().enumerate() {
            if !intensity.is_finite() || intensity < 0.0 {
                return Err(WavelengthComponentsError::InvalidRelativeIntensity { component });
            }
        }
        if relative_intensities[0] <= 0.0 {
            return Err(WavelengthComponentsError::NonPositiveReferenceIntensity);
        }
        Ok(Self {
            wavelengths_angstrom,
            relative_intensities,
        })
    }

    /// Number of wavelength components.
    #[must_use]
    pub const fn len(self) -> usize {
        self.wavelengths_angstrom.len()
    }

    /// Whether the model contains no components.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.wavelengths_angstrom.is_empty()
    }

    pub(crate) const fn wavelength(self, component: usize) -> f64 {
        self.wavelengths_angstrom[component]
    }

    pub(crate) const fn relative_intensity(self, component: usize) -> f64 {
        self.relative_intensities[component]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_component_arrays_and_reference_intensity() {
        let wavelengths = [1.540_56, 1.544_39];
        let intensities = [1.0, 0.5];
        let model = WavelengthComponentsView::new(&wavelengths, &intensities).expect("valid");
        assert_eq!(model.len(), 2);
        assert_eq!(model.wavelength(1).to_bits(), 1.544_39_f64.to_bits());
        assert_eq!(model.relative_intensity(1).to_bits(), 0.5_f64.to_bits());
        assert_eq!(
            WavelengthComponentsView::new(&wavelengths, &[0.0, 1.0]),
            Err(WavelengthComponentsError::NonPositiveReferenceIntensity)
        );
    }
}
