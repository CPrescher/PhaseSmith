//! Complete matrix-free Rietveld objective with small explicit global columns.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    ConstraintDerivativeMatrix, DifferentiableBackground, LatticeError, LatticeParameterization,
    PreparedRietveldObjective, RietveldCalculation, RietveldCalculationOptions, RietveldError,
    RietveldGeneralParameterError, RietveldInput, RietveldInstrumentParameter,
    RietveldObjectiveError, RietveldParameterLayout, calculate_rietveld_pattern,
};

/// Default upper bound for reusable native structural Jacobians.
///
/// Ten million `f64` elements require at most about 80 MB before the smaller
/// complete-parameter columns are composed. Larger requests retain the
/// matrix-free objective.
pub const DEFAULT_MAX_LINEARIZATION_ELEMENTS: usize = 10_000_000;

/// Complete weighted Jacobian in scaled free-parameter coordinates.
///
/// Rows are free parameters and columns are pattern samples. Masked samples
/// are zero and included samples are divided by their uncertainty when the
/// objective uses uncertainty weighting. This matches the dense scripting
/// optimizer contract, so its JVP and VJP need no further weighting.
pub struct PreparedGeneralFreeLinearization {
    calculation: RietveldCalculation,
    weighted_jacobian: Vec<f64>,
    sample_scale: Vec<f64>,
    parameter_count: usize,
}

impl PreparedGeneralFreeLinearization {
    /// Borrow the calculation produced by the same fused native pass.
    #[must_use]
    pub const fn calculation(&self) -> &RietveldCalculation {
        &self.calculation
    }

    /// Return the scaled free-parameter count.
    #[must_use]
    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Apply the weighted free Jacobian.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralParameterError::ValueLengthMismatch`] for a
    /// direction with the wrong free dimension.
    pub fn jvp(&self, direction: &[f64]) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if direction.len() != self.parameter_count {
            return Err(RietveldGeneralParameterError::ValueLengthMismatch.into());
        }
        Ok(dense_forward_product(
            &self.weighted_jacobian,
            direction,
            self.sample_scale.len(),
        ))
    }

    /// Apply the weighted free Jacobian transpose.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError::SampleLengthMismatch`] for a
    /// vector with the wrong sample dimension.
    pub fn vjp(&self, samples: &[f64]) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if samples.len() != self.sample_scale.len() {
            return Err(RietveldGeneralObjectiveError::SampleLengthMismatch);
        }
        Ok(dense_reverse_product(
            &self.weighted_jacobian,
            samples,
            self.parameter_count,
        ))
    }

    /// Return `J_w^T J_w direction + damping direction` in scaled free coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for invalid damping or shape.
    pub fn normal_product(
        &self,
        direction: &[f64],
        damping: f64,
    ) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(RietveldGeneralObjectiveError::InvalidDamping);
        }
        let product = self.jvp(direction)?;
        let mut result = self.vjp(&product)?;
        for (value, direction) in result.iter_mut().zip(direction) {
            *value += damping * direction;
        }
        Ok(result)
    }

    /// Calculate the scaled free gradient for the stored calculation.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError::SampleLengthMismatch`] for an
    /// observed vector with the wrong sample dimension.
    pub fn gradient(&self, observed: &[f64]) -> Result<Vec<f64>, RietveldGeneralObjectiveError> {
        if observed.len() != self.sample_scale.len() {
            return Err(RietveldGeneralObjectiveError::SampleLengthMismatch);
        }
        let weighted_residual = self
            .calculation
            .y
            .iter()
            .zip(observed)
            .zip(&self.sample_scale)
            .map(|((calculated, observed), scale)| (calculated - observed) * scale)
            .collect::<Vec<_>>();
        self.vjp(&weighted_residual)
    }
}

/// Reusable complete physical objective for one accepted native state.
pub struct PreparedGeneralRietveldObjective {
    input: RietveldInput,
    options: RietveldCalculationOptions,
    layout: RietveldParameterLayout,
    structural: PreparedRietveldObjective,
    calculation: RietveldCalculation,
    dense_structural_jacobian: Option<Vec<f64>>,
    explicit_columns: Vec<(usize, Vec<f64>)>,
}

impl PreparedGeneralRietveldObjective {
    /// Prepare structural products and explicit global/background columns.
    ///
    /// The structural Jacobian is materialized when it fits under the default
    /// memory ceiling. Larger objectives retain matrix-free structural
    /// products; selected experiment/background columns remain explicit.
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
        Self::new_with_max_linearization_elements(
            input,
            options,
            layout,
            DEFAULT_MAX_LINEARIZATION_ELEMENTS,
        )
    }

    /// Prepare a complete objective under an explicit dense-memory ceiling.
    ///
    /// A zero ceiling forces matrix-free products. Requests whose native
    /// structural Jacobian fits within the ceiling materialize it once and
    /// reuse it for every gradient and normal product at this state.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for invalid state, derivative
    /// layout mismatches, or failed structural products.
    pub fn new_with_max_linearization_elements(
        input: RietveldInput,
        options: RietveldCalculationOptions,
        layout: RietveldParameterLayout,
        max_linearization_elements: usize,
    ) -> Result<Self, RietveldGeneralObjectiveError> {
        let structural = PreparedRietveldObjective::new(
            input.clone(),
            options.clone(),
            layout.structural_layout().clone(),
        )?;
        let dense_enabled = max_linearization_elements > 0
            && structural.dense_element_count()? <= max_linearization_elements;
        let (calculation, dense_structural_jacobian) = if dense_enabled {
            let linearization = structural.linearize()?;
            (linearization.calculation, Some(linearization.jacobian))
        } else {
            (calculate_rietveld_pattern(&input, &options)?, None)
        };
        let mut explicit_columns = instrument_columns(&input, &calculation, &layout)?;
        append_sample_physics_columns(&input, &calculation, &layout, &mut explicit_columns)?;
        append_background_columns(&input, &layout, &mut explicit_columns)?;
        Ok(Self {
            input,
            options,
            layout,
            structural,
            calculation,
            dense_structural_jacobian,
            explicit_columns,
        })
    }

    /// Return whether this state reuses a bounded dense structural Jacobian.
    #[must_use]
    pub const fn uses_dense_linearization(&self) -> bool {
        self.dense_structural_jacobian.is_some()
    }

    /// Return expensive model products consumed while preparing the gradient.
    #[must_use]
    pub const fn preparation_evaluation_count(&self) -> usize {
        if self.uses_dense_linearization() {
            1
        } else {
            2
        }
    }

    /// Return expensive model products consumed by one normal-product call.
    #[must_use]
    pub const fn normal_product_evaluation_count(&self) -> usize {
        if self.uses_dense_linearization() {
            0
        } else {
            2
        }
    }

    /// Return expensive model products consumed by one forward-product call.
    #[must_use]
    pub const fn jvp_evaluation_count(&self) -> usize {
        if self.uses_dense_linearization() {
            0
        } else {
            1
        }
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

    /// Project the cached dense objective into weighted scaled-free rows.
    ///
    /// Returns `None` for the bounded matrix-free fallback. The derivative
    /// matrix maps scaled free coordinates to the complete physical layout.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldGeneralObjectiveError`] for incompatible dimensions
    /// or failed dense products.
    pub fn free_linearization(
        &self,
        derivative: &ConstraintDerivativeMatrix,
    ) -> Result<Option<PreparedGeneralFreeLinearization>, RietveldGeneralObjectiveError> {
        if !self.uses_dense_linearization() {
            return Ok(None);
        }
        if derivative.rows != self.layout.parameters().specs().len() {
            return Err(RietveldGeneralParameterError::ValueLengthMismatch.into());
        }
        let sample_count = self.input.pattern.sample_count();
        let sample_scale = self.sample_scale();
        let element_count = derivative
            .columns
            .checked_mul(sample_count)
            .ok_or(RietveldObjectiveError::AllocationOverflow)?;
        let mut weighted_jacobian = Vec::with_capacity(element_count);
        for free in 0..derivative.columns {
            let physical = derivative
                .values
                .chunks_exact(derivative.columns)
                .map(|row| row[free])
                .collect::<Vec<_>>();
            let (_, mut row) = self.jvp(&physical)?;
            for (value, scale) in row.iter_mut().zip(&sample_scale) {
                *value *= scale;
            }
            weighted_jacobian.extend(row);
        }
        Ok(Some(PreparedGeneralFreeLinearization {
            calculation: self.calculation.clone(),
            weighted_jacobian,
            sample_scale,
            parameter_count: derivative.columns,
        }))
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
        let (profile, mut derivative) = if let Some(jacobian) = &self.dense_structural_jacobian {
            (
                self.calculation.profile_y.clone(),
                dense_forward_product(
                    jacobian,
                    &structural_direction,
                    self.input.pattern.sample_count(),
                ),
            )
        } else {
            self.structural.jvp(&structural_direction)?
        };
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
        let structural = if let Some(jacobian) = &self.dense_structural_jacobian {
            dense_reverse_product(
                jacobian,
                sample_weights,
                self.layout.structural_layout().parameters().specs().len(),
            )
        } else {
            self.structural.vjp(sample_weights)?
        };
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

    fn sample_scale(&self) -> Vec<f64> {
        let mask = self.input.pattern.mask.as_deref();
        let uncertainty = self
            .options
            .use_uncertainty
            .then_some(self.input.pattern.uncertainty.as_deref())
            .flatten();
        (0..self.input.pattern.sample_count())
            .map(|index| {
                if mask.is_some_and(|mask| !mask[index]) {
                    0.0
                } else {
                    uncertainty.map_or(1.0, |sigma| sigma[index].recip())
                }
            })
            .collect()
    }
}

fn dense_forward_product(jacobian: &[f64], direction: &[f64], sample_count: usize) -> Vec<f64> {
    let mut result = vec![0.0; sample_count];
    for (coefficient, row) in direction.iter().zip(jacobian.chunks_exact(sample_count)) {
        for (target, value) in result.iter_mut().zip(row) {
            *target += coefficient * value;
        }
    }
    result
}

fn dense_reverse_product(
    jacobian: &[f64],
    sample_weights: &[f64],
    parameter_count: usize,
) -> Vec<f64> {
    jacobian
        .chunks_exact(sample_weights.len())
        .take(parameter_count)
        .map(|row| {
            row.iter()
                .zip(sample_weights)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn instrument_columns(
    input: &RietveldInput,
    calculation: &RietveldCalculation,
    layout: &RietveldParameterLayout,
) -> Result<Vec<(usize, Vec<f64>)>, RietveldGeneralObjectiveError> {
    let sample_count = input.pattern.sample_count();
    let mut result = Vec::new();
    for (parameter, index) in layout.instrument_indices() {
        let row = instrument_global_row(input, *parameter)?;
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
        result.push((*index, column));
    }
    Ok(result)
}

fn append_sample_physics_columns(
    input: &RietveldInput,
    calculation: &RietveldCalculation,
    layout: &RietveldParameterLayout,
    explicit_columns: &mut Vec<(usize, Vec<f64>)>,
) -> Result<(), RietveldGeneralObjectiveError> {
    let mut terms = layout
        .sample_physics_indices()
        .map(|(phase, name, parameter)| (phase, name.to_owned(), parameter, 1.0))
        .collect::<Vec<_>>();
    for (phase_index, phase) in input.phases.iter().enumerate() {
        let (_, names) =
            phase.resolved_sample_physics(input.instrument, input.position_correction)?;
        if !names
            .iter()
            .any(|name| name.starts_with("march_dollase.cell."))
        {
            continue;
        }
        let parameterization = LatticeParameterization::new(
            phase.definition().space_group.clone(),
            phase.definition().cell,
        )?;
        let lattice_values = parameterization.values_from_cell(phase.definition().cell)?;
        let jacobian = parameterization.cell_jacobian(&lattice_values)?;
        let columns = lattice_values.len();
        for (parameter_index, spec) in layout.parameters().specs().iter().enumerate() {
            if spec.key().module() != "lattice"
                || spec.key().owner_id() != phase.phase_id().as_str()
            {
                continue;
            }
            let column = parameterization
                .parameter_names()
                .iter()
                .position(|name| name == spec.key().name())
                .ok_or(RietveldGeneralObjectiveError::GlobalDerivativeShape)?;
            for (cell_row, name) in [
                "a_angstrom",
                "b_angstrom",
                "c_angstrom",
                "alpha_deg",
                "beta_deg",
                "gamma_deg",
            ]
            .iter()
            .enumerate()
            {
                let coefficient = jacobian[cell_row * columns + column];
                if coefficient != 0.0 {
                    terms.push((
                        phase_index,
                        format!("march_dollase.cell.{name}"),
                        parameter_index,
                        coefficient,
                    ));
                }
            }
        }
    }
    for (phase_index, name, parameter_index, coefficient) in terms {
        append_sample_physics_column(
            input,
            calculation,
            explicit_columns,
            phase_index,
            &name,
            parameter_index,
            coefficient,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_sample_physics_column(
    input: &RietveldInput,
    calculation: &RietveldCalculation,
    explicit_columns: &mut Vec<(usize, Vec<f64>)>,
    phase_index: usize,
    name: &str,
    parameter_index: usize,
    coefficient: f64,
) -> Result<(), RietveldGeneralObjectiveError> {
    let phase = input
        .phases
        .get(phase_index)
        .ok_or(RietveldGeneralObjectiveError::GlobalDerivativeShape)?;
    let (_, names) = phase.resolved_sample_physics(input.instrument, input.position_correction)?;
    let provider_row = names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or(RietveldGeneralObjectiveError::GlobalDerivativeShape)?;
    let sample_count = input.pattern.sample_count();
    let row = sample_physics_global_start(input) + provider_row;
    let global = calculation.phases[phase_index]
        .result
        .accumulation
        .derivatives
        .global
        .as_ref()
        .ok_or(RietveldGeneralObjectiveError::MissingGlobalDerivatives)?;
    let values = global
        .values
        .get(row * sample_count..(row + 1) * sample_count)
        .ok_or(RietveldGeneralObjectiveError::GlobalDerivativeShape)?;
    if let Some((_, column)) = explicit_columns
        .iter_mut()
        .find(|(index, _)| *index == parameter_index)
    {
        for (target, value) in column.iter_mut().zip(values) {
            *target += coefficient * value;
        }
    } else {
        explicit_columns.push((
            parameter_index,
            values.iter().map(|value| coefficient * value).collect(),
        ));
    }
    Ok(())
}

fn append_background_columns(
    input: &RietveldInput,
    layout: &RietveldParameterLayout,
    explicit_columns: &mut Vec<(usize, Vec<f64>)>,
) -> Result<(), RietveldGeneralObjectiveError> {
    if layout.background_indices().is_empty() {
        return Ok(());
    }
    let background = input
        .background
        .as_ref()
        .ok_or(RietveldGeneralParameterError::MissingBackground)?;
    let basis = background.basis(&input.pattern.x_deg)?;
    if basis.columns != layout.background_indices().len()
        || basis.rows != input.pattern.sample_count()
    {
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
    Ok(())
}

fn sample_physics_global_start(input: &RietveldInput) -> usize {
    7 - usize::from(input.fixed_spectrum.is_some())
        + usize::from(input.position_correction.bragg_brentano_mm.is_some())
        + 2 * usize::from(
            input
                .position_correction
                .debye_scherrer_micrometre
                .is_some(),
        )
        + 2 * usize::from(input.axial_geometry.is_some())
}

fn instrument_global_row(
    input: &RietveldInput,
    parameter: RietveldInstrumentParameter,
) -> Result<usize, RietveldGeneralObjectiveError> {
    let monochromatic_row = match parameter {
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
    };
    if input.fixed_spectrum.is_some() {
        if parameter == RietveldInstrumentParameter::WavelengthAngstrom {
            return Err(RietveldGeneralParameterError::SpectrumWavelengthRefinement.into());
        }
        Ok(monochromatic_row - usize::from(monochromatic_row > 5))
    } else {
        Ok(monochromatic_row)
    }
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
    /// Lattice-to-sample derivative mapping failed.
    Lattice(LatticeError),
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
            Self::Lattice(error) => Display::fmt(error, formatter),
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
            Self::Lattice(error) => Some(error),
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
impl From<LatticeError> for RietveldGeneralObjectiveError {
    fn from(value: LatticeError) -> Self {
        Self::Lattice(value)
    }
}
impl From<crate::BackgroundError> for RietveldGeneralObjectiveError {
    fn from(value: crate::BackgroundError) -> Self {
        Self::Parameter(RietveldGeneralParameterError::Background(value))
    }
}
