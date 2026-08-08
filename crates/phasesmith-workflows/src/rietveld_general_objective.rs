//! Complete matrix-free Rietveld objective with small explicit global columns.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    DifferentiableBackground, PreparedRietveldObjective, RietveldCalculation,
    RietveldCalculationOptions, RietveldError, RietveldGeneralParameterError, RietveldInput,
    RietveldInstrumentParameter, RietveldObjectiveError, RietveldParameterLayout,
    calculate_rietveld_pattern,
};

/// Reusable complete physical objective for one accepted native state.
pub struct PreparedGeneralRietveldObjective {
    input: RietveldInput,
    options: RietveldCalculationOptions,
    layout: RietveldParameterLayout,
    structural: PreparedRietveldObjective,
    calculation: RietveldCalculation,
    explicit_columns: Vec<(usize, Vec<f64>)>,
}

impl PreparedGeneralRietveldObjective {
    /// Prepare structural products and explicit global/background columns.
    ///
    /// The potentially large structural Jacobian remains matrix-free. Only the
    /// selected experiment/background columns are materialized.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for invalid state or derivative
    /// layout mismatches.
    pub fn new(
        input: RietveldInput,
        options: RietveldCalculationOptions,
        layout: RietveldParameterLayout,
    ) -> Result<Self, RietveldGeneralObjectiveError> {
        let structural = PreparedRietveldObjective::new(
            input.clone(),
            options.clone(),
            layout.structural_layout().clone(),
        )?;
        let calculation = calculate_rietveld_pattern(&input, &options)?;
        let sample_count = input.pattern.sample_count();
        let mut explicit_columns = Vec::new();
        for (parameter, index) in layout.instrument_indices() {
            let row = instrument_global_row(&input, *parameter)?;
            let mut column = vec![0.0; sample_count];
            for phase in &calculation.phases {
                let global = phase
                    .result
                    .accumulation
                    .derivatives
                    .global
                    .as_ref()
                    .ok_or(RietveldGeneralObjectiveError::MissingGlobalDerivatives)?;
                if row >= global.parameter_count || global.sample_count != sample_count {
                    return Err(RietveldGeneralObjectiveError::GlobalDerivativeShape);
                }
                let values = global
                    .values
                    .get(row * sample_count..(row + 1) * sample_count)
                    .ok_or(RietveldGeneralObjectiveError::GlobalDerivativeShape)?;
                for (target, value) in column.iter_mut().zip(values) {
                    *target += value;
                }
            }
            explicit_columns.push((*index, column));
        }
        if !layout.background_indices().is_empty() {
            let background = input
                .background
                .as_ref()
                .ok_or(RietveldGeneralParameterError::MissingBackground)?;
            let basis = background.basis(&input.pattern.x_deg)?;
            if basis.columns != layout.background_indices().len() || basis.rows != sample_count {
                return Err(RietveldGeneralObjectiveError::BackgroundDerivativeShape);
            }
            for (column_index, parameter_index) in layout.background_indices().iter().enumerate() {
                explicit_columns.push((
                    *parameter_index,
                    basis
                        .column(column_index)
                        .ok_or(RietveldGeneralObjectiveError::BackgroundDerivativeShape)?,
                ));
            }
        }
        Ok(Self {
            input,
            options,
            layout,
            structural,
            calculation,
            explicit_columns,
        })
    }

    /// Borrow the complete stable physical layout.
    #[must_use]
    pub const fn layout(&self) -> &RietveldParameterLayout {
        &self.layout
    }

    /// Borrow the accepted-state calculation used for global columns.
    #[must_use]
    pub const fn calculation(&self) -> &RietveldCalculation {
        &self.calculation
    }

    /// Calculate profile values and one complete physical directional derivative.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for a wrong direction or
    /// numerical product failure.
    pub fn jvp(
        &self,
        direction: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), RietveldGeneralObjectiveError> {
        if direction.len() != self.layout.parameters().specs().len() {
            return Err(RietveldGeneralParameterError::ValueLengthMismatch.into());
        }
        let structural_direction = self.layout.structural_direction(direction)?;
        let (profile, mut derivative) = self.structural.jvp(&structural_direction)?;
        for (parameter, column) in &self.explicit_columns {
            let coefficient = direction[*parameter];
            for (target, value) in derivative.iter_mut().zip(column) {
                *target += coefficient * value;
            }
        }
        Ok((profile, derivative))
    }

    /// Apply the complete physical Jacobian transpose.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for sample shape or engine
    /// product failures.
    pub fn vjp(&self, sample_weights: &[f64]) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if sample_weights.len() != self.input.pattern.sample_count() {
            return Err(RietveldGeneralObjectiveError::SampleLengthMismatch);
        }
        let structural = self.structural.vjp(sample_weights)?;
        let mut result = self.layout.expand_structural_gradient(&structural)?;
        for (parameter, column) in &self.explicit_columns {
            result[*parameter] += column
                .iter()
                .zip(sample_weights)
                .map(|(left, right)| left * right)
                .sum::<f64>();
        }
        Ok(result)
    }

    /// Apply `J^T W J + damping I` in complete physical coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for invalid damping or product
    /// state.
    pub fn normal_product(
        &self,
        direction: &[f64],
        damping: f64,
    ) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(RietveldGeneralObjectiveError::InvalidDamping);
        }
        let (_, derivative) = self.jvp(direction)?;
        let weighted = self.weight_samples(&derivative);
        let mut result = self.vjp(&weighted)?;
        for (value, direction) in result.iter_mut().zip(direction) {
            *value += damping * direction;
        }
        Ok(result)
    }

    /// Calculate the complete accepted-state gradient.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for residual or product state.
    pub fn gradient(&self) -> Result<(Vec<f64>, Vec<f64>), RietveldGeneralObjectiveError> {
        let observed = self
            .input
            .pattern
            .observed_y
            .as_ref()
            .ok_or(RietveldError::MissingObservations)?;
        let residual = self
            .calculation
            .y
            .iter()
            .zip(observed)
            .map(|(calculated, observed)| calculated - observed)
            .collect::<Vec<_>>();
        Ok((
            self.calculation.y.clone(),
            self.vjp(&self.weight_samples(&residual))?,
        ))
    }

    fn weight_samples(&self, values: &[f64]) -> Vec<f64> {
        let mask = self.input.pattern.mask.as_deref();
        let uncertainty = self
            .options
            .use_uncertainty
            .then_some(self.input.pattern.uncertainty.as_deref())
            .flatten();
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if mask.is_some_and(|mask| !mask[index]) {
                    0.0
                } else if let Some(sigma) = uncertainty {
                    value / (sigma[index] * sigma[index])
                } else {
                    *value
                }
            })
            .collect()
    }
}

fn instrument_global_row(
    input: &RietveldInput,
    parameter: RietveldInstrumentParameter,
) -> Result<usize, RietveldGeneralObjectiveError> {
    Ok(match parameter {
        RietveldInstrumentParameter::UDeg2 => 0,
        RietveldInstrumentParameter::VDeg2 => 1,
        RietveldInstrumentParameter::WDeg2 => 2,
        RietveldInstrumentParameter::XDeg => 3,
        RietveldInstrumentParameter::YDeg => 4,
        RietveldInstrumentParameter::WavelengthAngstrom => 5,
        RietveldInstrumentParameter::ZeroShiftDeg => 6,
        RietveldInstrumentParameter::SampleDisplacementMm => {
            if input.position_correction.bragg_brentano_mm.is_none() {
                return Err(RietveldGeneralParameterError::InstrumentGeometryMismatch.into());
            }
            7
        }
        RietveldInstrumentParameter::DisplaceXMicrometre => {
            if input
                .position_correction
                .debye_scherrer_micrometre
                .is_none()
            {
                return Err(RietveldGeneralParameterError::InstrumentGeometryMismatch.into());
            }
            7
        }
        RietveldInstrumentParameter::DisplaceYMicrometre => {
            if input
                .position_correction
                .debye_scherrer_micrometre
                .is_none()
            {
                return Err(RietveldGeneralParameterError::InstrumentGeometryMismatch.into());
            }
            8
        }
    })
}

/// Invalid complete native Rietveld objective state.
#[derive(Debug)]
pub enum RietveldGeneralObjectiveError {
    /// Complete parameter state is invalid.
    Parameter(RietveldGeneralParameterError),
    /// Structural objective state is invalid.
    Structural(RietveldObjectiveError),
    /// Request/calculation state is invalid.
    Rietveld(RietveldError),
    /// Engine did not expose required global derivatives.
    MissingGlobalDerivatives,
    /// An engine global derivative matrix has an unexpected shape.
    GlobalDerivativeShape,
    /// A background basis has an unexpected shape.
    BackgroundDerivativeShape,
    /// Sample reverse-product length is wrong.
    SampleLengthMismatch,
    /// Damping must be finite and non-negative.
    InvalidDamping,
}

impl Display for RietveldGeneralObjectiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::MissingGlobalDerivatives => {
                formatter.write_str("native engine omitted Rietveld global derivatives")
            }
            Self::GlobalDerivativeShape => {
                formatter.write_str("native Rietveld global derivative shape is invalid")
            }
            Self::BackgroundDerivativeShape => {
                formatter.write_str("native Rietveld background derivative shape is invalid")
            }
            Self::SampleLengthMismatch => {
                formatter.write_str("native Rietveld sample weight length is wrong")
            }
            Self::InvalidDamping => {
                formatter.write_str("native Rietveld damping must be finite and non-negative")
            }
        }
    }
}

impl Error for RietveldGeneralObjectiveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::Structural(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::MissingGlobalDerivatives
            | Self::GlobalDerivativeShape
            | Self::BackgroundDerivativeShape
            | Self::SampleLengthMismatch
            | Self::InvalidDamping => None,
        }
    }
}

impl From<RietveldGeneralParameterError> for RietveldGeneralObjectiveError {
    fn from(value: RietveldGeneralParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<RietveldObjectiveError> for RietveldGeneralObjectiveError {
    fn from(value: RietveldObjectiveError) -> Self {
        Self::Structural(value)
    }
}
impl From<RietveldError> for RietveldGeneralObjectiveError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}
impl From<crate::BackgroundError> for RietveldGeneralObjectiveError {
    fn from(value: crate::BackgroundError) -> Self {
        Self::Parameter(RietveldGeneralParameterError::Background(value))
    }
}
