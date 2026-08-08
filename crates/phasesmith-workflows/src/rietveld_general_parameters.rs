//! Complete native Rietveld parameter identities and accepted-state installation.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::ConstantWavelengthInstrument;

use crate::{
    BackgroundError, DifferentiableBackground, LatticeBounds, ParameterBounds, ParameterError,
    ParameterKey, ParameterSet, ParameterSpec, RietveldError, RietveldInput,
    RietveldParameterError, RietveldStructuralLayout, RietveldStructuralSelection,
};

/// Supported built-in monochromatic instrument and position parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RietveldInstrumentParameter {
    /// Gaussian Caglioti U in square degrees.
    UDeg2,
    /// Gaussian Caglioti V in square degrees.
    VDeg2,
    /// Gaussian Caglioti W in square degrees.
    WDeg2,
    /// Lorentzian X in degrees.
    XDeg,
    /// Lorentzian Y in degrees.
    YDeg,
    /// Monochromatic wavelength in ångströms.
    WavelengthAngstrom,
    /// Constant additive two-theta shift in degrees.
    ZeroShiftDeg,
    /// Bragg--Brentano sample displacement in millimetres.
    SampleDisplacementMm,
    /// Debye--Scherrer X displacement in micrometres.
    DisplaceXMicrometre,
    /// Debye--Scherrer Y displacement in micrometres.
    DisplaceYMicrometre,
}

impl RietveldInstrumentParameter {
    /// Return the stable scripting/wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UDeg2 => "u_deg2",
            Self::VDeg2 => "v_deg2",
            Self::WDeg2 => "w_deg2",
            Self::XDeg => "x_deg",
            Self::YDeg => "y_deg",
            Self::WavelengthAngstrom => "wavelength_angstrom",
            Self::ZeroShiftDeg => "zero_shift_deg",
            Self::SampleDisplacementMm => "sample_displacement_mm",
            Self::DisplaceXMicrometre => "displace_x_micrometre",
            Self::DisplaceYMicrometre => "displace_y_micrometre",
        }
    }
}

/// Complete selected built-in parameter families for one native Rietveld run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RietveldParameterSelection {
    /// Structural phase/site/lattice selection.
    pub structural: RietveldStructuralSelection,
    /// Ordered unique instrument/position parameters.
    pub instrument: Vec<RietveldInstrumentParameter>,
    /// Refine all coefficients of the attached analytical background.
    pub background: bool,
    /// Refine built-in per-phase sample-physics parameters.
    pub sample_physics: bool,
}

impl RietveldParameterSelection {
    /// Validate ordered unique instrument selections.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError::DuplicateInstrumentParameter`]
    /// when one stable parameter is selected twice.
    pub fn new(
        structural: RietveldStructuralSelection,
        instrument: Vec<RietveldInstrumentParameter>,
        background: bool,
        sample_physics: bool,
    ) -> Result<Self, RietveldGeneralParameterError> {
        let result = Self {
            structural,
            instrument,
            background,
            sample_physics,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<(), RietveldGeneralParameterError> {
        if self
            .instrument
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != self.instrument.len()
        {
            return Err(RietveldGeneralParameterError::DuplicateInstrumentParameter);
        }
        if self.sample_physics {
            return Err(RietveldGeneralParameterError::SamplePhysicsNotConfigured);
        }
        Ok(())
    }
}

/// Ordered complete physical parameters and state transforms.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldParameterLayout {
    parameters: ParameterSet,
    structural: RietveldStructuralLayout,
    structural_indices: Vec<usize>,
    instrument: Vec<(RietveldInstrumentParameter, usize)>,
    background_indices: Vec<usize>,
    background_id: Option<String>,
}

impl RietveldParameterLayout {
    /// Build complete stable parameters in instrument, background, structural order.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError`] for incompatible selections,
    /// domains, bounds, or attached state.
    pub fn new(
        input: &RietveldInput,
        selection: &RietveldParameterSelection,
        lattice_bounds: &[Option<LatticeBounds>],
    ) -> Result<Self, RietveldGeneralParameterError> {
        input.validate()?;
        selection.validate()?;
        let structural =
            RietveldStructuralLayout::new(&input.phases, selection.structural, lattice_bounds)?;
        let mut specs = Vec::new();
        let mut instrument = Vec::new();
        for selected in &selection.instrument {
            let value = instrument_value(input, *selected)?;
            let (unit, bounds, floor) = instrument_metadata(*selected)?;
            let index = specs.len();
            specs.push(ParameterSpec::new(
                ParameterKey::new("instrument", "cw", selected.as_str())?,
                value,
                unit,
                bounds,
                value.abs().max(floor),
                true,
            )?);
            instrument.push((*selected, index));
        }
        let mut background_indices = Vec::new();
        let mut background_id = None;
        if selection.background {
            let background = input
                .background
                .as_ref()
                .ok_or(RietveldGeneralParameterError::MissingBackground)?;
            let names = background.parameter_names();
            let coefficients = background.coefficients();
            let bounds = background.parameter_bounds();
            background_id = Some(background.background_id().to_owned());
            for ((name, value), bounds) in names.into_iter().zip(coefficients).zip(bounds) {
                let index = specs.len();
                specs.push(ParameterSpec::new(
                    ParameterKey::new("background", background.background_id(), name)?,
                    value,
                    "intensity",
                    bounds,
                    value.abs().max(1.0),
                    true,
                )?);
                background_indices.push(index);
            }
        }
        let structural_indices = structural
            .parameters()
            .specs()
            .iter()
            .map(|spec| {
                let index = specs.len();
                specs.push(spec.clone());
                index
            })
            .collect();
        Ok(Self {
            parameters: ParameterSet::new(specs)?,
            structural,
            structural_indices,
            instrument,
            background_indices,
            background_id,
        })
    }

    /// Borrow all physical parameters in stable packing order.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Borrow the structural sub-layout.
    #[must_use]
    pub const fn structural_layout(&self) -> &RietveldStructuralLayout {
        &self.structural
    }

    /// Extract one structural tangent from a complete physical tangent.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError::ValueLengthMismatch`] for a
    /// wrong complete direction length.
    pub fn structural_direction(
        &self,
        direction: &[f64],
    ) -> Result<Vec<f64>, RietveldGeneralParameterError> {
        self.validate_length(direction)?;
        Ok(self
            .structural_indices
            .iter()
            .map(|index| direction[*index])
            .collect())
    }

    /// Combine a structural reverse product with zeroed non-structural rows.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError::StructuralGradientLengthMismatch`]
    /// for a stale structural product.
    pub fn expand_structural_gradient(
        &self,
        structural: &[f64],
    ) -> Result<Vec<f64>, RietveldGeneralParameterError> {
        if structural.len() != self.structural_indices.len() {
            return Err(RietveldGeneralParameterError::StructuralGradientLengthMismatch);
        }
        let mut result = vec![0.0; self.parameters.specs().len()];
        for (value, index) in structural.iter().zip(&self.structural_indices) {
            result[*index] = *value;
        }
        Ok(result)
    }

    pub(crate) fn instrument_indices(&self) -> &[(RietveldInstrumentParameter, usize)] {
        &self.instrument
    }

    pub(crate) fn background_indices(&self) -> &[usize] {
        &self.background_indices
    }

    /// Install complete bounded physical values into a cloned request.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError`] for wrong, non-finite,
    /// out-of-bounds, or domain-invalid values.
    pub fn apply_values(
        &self,
        input: &RietveldInput,
        values: &[f64],
    ) -> Result<RietveldInput, RietveldGeneralParameterError> {
        self.validate_length(values)?;
        for (spec, value) in self.parameters.specs().iter().zip(values) {
            if !value.is_finite() || !spec.bounds().contains(*value) {
                return Err(RietveldGeneralParameterError::Parameter(
                    ParameterError::ValueOutsideBounds {
                        key: spec.key().clone(),
                        value: *value,
                    },
                ));
            }
        }
        let structural_values = self
            .structural_indices
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>();
        let mut updated = input.clone();
        updated.phases = self
            .structural
            .apply_values(&input.phases, &structural_values)?;
        let mut wavelength = None;
        for (parameter, index) in &self.instrument {
            install_instrument_value(&mut updated, *parameter, values[*index])?;
            if *parameter == RietveldInstrumentParameter::WavelengthAngstrom {
                wavelength = Some(values[*index]);
            }
        }
        if let Some(wavelength) = wavelength {
            updated.phases = updated
                .phases
                .iter()
                .map(|phase| phase.with_wavelength(wavelength))
                .collect::<Result<Vec<_>, _>>()?;
        }
        if !self.background_indices.is_empty() {
            let background = updated
                .background
                .as_ref()
                .ok_or(RietveldGeneralParameterError::MissingBackground)?;
            if Some(background.background_id()) != self.background_id.as_deref() {
                return Err(RietveldGeneralParameterError::BackgroundIdentityMismatch);
            }
            let coefficients = self
                .background_indices
                .iter()
                .map(|index| values[*index])
                .collect::<Vec<_>>();
            updated.background = Some(background.replace_coefficients(&coefficients)?);
        }
        updated.validate()?;
        Ok(updated)
    }

    fn validate_length(&self, values: &[f64]) -> Result<(), RietveldGeneralParameterError> {
        if values.len() != self.parameters.specs().len() {
            return Err(RietveldGeneralParameterError::ValueLengthMismatch);
        }
        Ok(())
    }
}

fn instrument_value(
    input: &RietveldInput,
    parameter: RietveldInstrumentParameter,
) -> Result<f64, RietveldGeneralParameterError> {
    Ok(match parameter {
        RietveldInstrumentParameter::UDeg2 => input.instrument.u_deg2,
        RietveldInstrumentParameter::VDeg2 => input.instrument.v_deg2,
        RietveldInstrumentParameter::WDeg2 => input.instrument.w_deg2,
        RietveldInstrumentParameter::XDeg => input.instrument.x_deg,
        RietveldInstrumentParameter::YDeg => input.instrument.y_deg,
        RietveldInstrumentParameter::WavelengthAngstrom => input.instrument.wavelength_angstrom,
        RietveldInstrumentParameter::ZeroShiftDeg => input.position_correction.zero_shift_deg,
        RietveldInstrumentParameter::SampleDisplacementMm => input
            .position_correction
            .bragg_brentano_mm
            .map(|value| value.0)
            .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?,
        RietveldInstrumentParameter::DisplaceXMicrometre => input
            .position_correction
            .debye_scherrer_micrometre
            .map(|value| value.0)
            .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?,
        RietveldInstrumentParameter::DisplaceYMicrometre => input
            .position_correction
            .debye_scherrer_micrometre
            .map(|value| value.1)
            .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?,
    })
}

fn instrument_metadata(
    parameter: RietveldInstrumentParameter,
) -> Result<(&'static str, ParameterBounds, f64), ParameterError> {
    Ok(match parameter {
        RietveldInstrumentParameter::WavelengthAngstrom => (
            "angstrom",
            ParameterBounds::new(f64::MIN_POSITIVE, f64::INFINITY)?,
            0.1,
        ),
        RietveldInstrumentParameter::SampleDisplacementMm => {
            ("millimetre", ParameterBounds::default(), 1.0e-2)
        }
        RietveldInstrumentParameter::DisplaceXMicrometre
        | RietveldInstrumentParameter::DisplaceYMicrometre => {
            ("micrometre", ParameterBounds::default(), 1.0e3)
        }
        RietveldInstrumentParameter::UDeg2
        | RietveldInstrumentParameter::VDeg2
        | RietveldInstrumentParameter::WDeg2 => ("degree^2", ParameterBounds::default(), 1.0e-4),
        RietveldInstrumentParameter::XDeg | RietveldInstrumentParameter::YDeg => {
            ("degree", ParameterBounds::default(), 1.0e-3)
        }
        RietveldInstrumentParameter::ZeroShiftDeg => ("degree", ParameterBounds::default(), 5.0e-2),
    })
}

fn install_instrument_value(
    input: &mut RietveldInput,
    parameter: RietveldInstrumentParameter,
    value: f64,
) -> Result<(), RietveldGeneralParameterError> {
    let ConstantWavelengthInstrument {
        wavelength_angstrom,
        u_deg2,
        v_deg2,
        w_deg2,
        x_deg,
        y_deg,
    } = &mut input.instrument;
    match parameter {
        RietveldInstrumentParameter::UDeg2 => *u_deg2 = value,
        RietveldInstrumentParameter::VDeg2 => *v_deg2 = value,
        RietveldInstrumentParameter::WDeg2 => *w_deg2 = value,
        RietveldInstrumentParameter::XDeg => *x_deg = value,
        RietveldInstrumentParameter::YDeg => *y_deg = value,
        RietveldInstrumentParameter::WavelengthAngstrom => *wavelength_angstrom = value,
        RietveldInstrumentParameter::ZeroShiftDeg => {
            input.position_correction.zero_shift_deg = value;
        }
        RietveldInstrumentParameter::SampleDisplacementMm => {
            let (_, radius) = input
                .position_correction
                .bragg_brentano_mm
                .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?;
            input.position_correction.bragg_brentano_mm = Some((value, radius));
        }
        RietveldInstrumentParameter::DisplaceXMicrometre => {
            let (_, y, radius) = input
                .position_correction
                .debye_scherrer_micrometre
                .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?;
            input.position_correction.debye_scherrer_micrometre = Some((value, y, radius));
        }
        RietveldInstrumentParameter::DisplaceYMicrometre => {
            let (x, _, radius) = input
                .position_correction
                .debye_scherrer_micrometre
                .ok_or(RietveldGeneralParameterError::InstrumentGeometryMismatch)?;
            input.position_correction.debye_scherrer_micrometre = Some((x, value, radius));
        }
    }
    Ok(())
}

/// Invalid complete native Rietveld parameter state.
#[derive(Debug)]
pub enum RietveldGeneralParameterError {
    /// One instrument parameter was selected more than once.
    DuplicateInstrumentParameter,
    /// Sample-physics selection requires native model records not yet attached.
    SamplePhysicsNotConfigured,
    /// Background selection requires an attached analytical model.
    MissingBackground,
    /// A selected position parameter does not match the configured geometry.
    InstrumentGeometryMismatch,
    /// Accepted background identity changed under the layout.
    BackgroundIdentityMismatch,
    /// Complete value/direction length is wrong.
    ValueLengthMismatch,
    /// Structural reverse-product length is wrong.
    StructuralGradientLengthMismatch,
    /// Stable scalar parameter state is invalid.
    Parameter(ParameterError),
    /// Structural parameter state is invalid.
    Structural(RietveldParameterError),
    /// Request state is invalid.
    Rietveld(RietveldError),
    /// Background replacement failed.
    Background(BackgroundError),
}

impl Display for RietveldGeneralParameterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateInstrumentParameter => {
                formatter.write_str("Rietveld instrument selections must be unique")
            }
            Self::SamplePhysicsNotConfigured => {
                formatter.write_str("native sample-physics parameter records are not configured")
            }
            Self::MissingBackground => {
                formatter.write_str("background refinement requires an analytical background")
            }
            Self::InstrumentGeometryMismatch => formatter
                .write_str("selected Rietveld position parameter does not match the geometry"),
            Self::BackgroundIdentityMismatch => {
                formatter.write_str("Rietveld background identity changed under the layout")
            }
            Self::ValueLengthMismatch => {
                formatter.write_str("complete Rietveld value/direction length is wrong")
            }
            Self::StructuralGradientLengthMismatch => {
                formatter.write_str("structural Rietveld gradient length is wrong")
            }
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RietveldGeneralParameterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::Structural(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Background(error) => Some(error),
            Self::DuplicateInstrumentParameter
            | Self::SamplePhysicsNotConfigured
            | Self::MissingBackground
            | Self::InstrumentGeometryMismatch
            | Self::BackgroundIdentityMismatch
            | Self::ValueLengthMismatch
            | Self::StructuralGradientLengthMismatch => None,
        }
    }
}

impl From<ParameterError> for RietveldGeneralParameterError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<RietveldParameterError> for RietveldGeneralParameterError {
    fn from(value: RietveldParameterError) -> Self {
        Self::Structural(value)
    }
}
impl From<RietveldError> for RietveldGeneralParameterError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}
impl From<BackgroundError> for RietveldGeneralParameterError {
    fn from(value: BackgroundError) -> Self {
        Self::Background(value)
    }
}
