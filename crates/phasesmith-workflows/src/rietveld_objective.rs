//! Reusable matrix-free structural Rietveld objective products.

use phasesmith_engine::StructuralPreparationCache;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use phasesmith_core::SupportPolicy;
use phasesmith_engine::{
    PreparedStructuralModelInputView, PreparedStructuralMultiphase, StructuralMultiphaseError,
};

use crate::rietveld::{
    assemble_rietveld_calculation, prepare_phase_model, resolve_phase_contributions,
};
use crate::{
    DifferentiableBackground, RietveldCalculation, RietveldCalculationOptions, RietveldError,
    RietveldInput, RietveldParameterError, RietveldStructuralLayout,
};

/// One reusable parameter-major physical structural Jacobian and its values.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedRietveldLinearization {
    /// Complete calculated pattern at the linearized state.
    pub calculation: RietveldCalculation,
    /// Structural Jacobian in physical-parameter-major order.
    pub jacobian: Vec<f64>,
    /// Number of physical structural parameter rows.
    pub parameter_count: usize,
}

/// Reusable structural values and derivative products for one accepted state.
pub struct PreparedRietveldObjective {
    input: RietveldInput,
    options: RietveldCalculationOptions,
    layout: RietveldStructuralLayout,
    prepared: PreparedStructuralMultiphase,
    cache: Arc<StructuralPreparationCache>,
}

impl PreparedRietveldObjective {
    /// Validate and prepare one structural-only matrix-free objective.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for invalid inputs or preparation.
    pub fn new(
        input: RietveldInput,
        options: RietveldCalculationOptions,
        layout: RietveldStructuralLayout,
    ) -> Result<Self, RietveldObjectiveError> {
        Self::new_cached(input, options, layout, Arc::default())
    }

    pub(crate) fn new_cached(
        input: RietveldInput,
        options: RietveldCalculationOptions,
        layout: RietveldStructuralLayout,
        cache: Arc<StructuralPreparationCache>,
    ) -> Result<Self, RietveldObjectiveError> {
        input.validate()?;
        options.validate()?;
        let models = input
            .phases
            .iter()
            .map(|phase| {
                prepare_phase_model(phase, input.fixed_spectrum.as_ref(), &options.execution)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prepared = PreparedStructuralMultiphase::new(models, options.execution.clone())?;
        layout.validate_phases(&input.phases)?;
        Ok(Self {
            input,
            options,
            layout,
            prepared,
            cache,
        })
    }

    /// Borrow the stable physical structural layout.
    #[must_use]
    pub const fn layout(&self) -> &RietveldStructuralLayout {
        &self.layout
    }

    /// Return the native dense element count required by [`Self::linearize`].
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError::AllocationOverflow`] when the product
    /// is not representable as `usize`.
    pub fn dense_element_count(&self) -> Result<usize, RietveldObjectiveError> {
        self.prepared
            .structural_parameter_counts()
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .and_then(|rows| rows.checked_mul(self.input.pattern.sample_count()))
            .ok_or(RietveldObjectiveError::AllocationOverflow)
    }

    /// Materialize one reusable physical structural Jacobian and calculation.
    ///
    /// Values and native analytical derivatives are produced in the same
    /// fused engine pass. The layout then projects native rows into stable
    /// physical parameter order.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for engine, shape, projection, or
    /// calculation failures.
    pub fn linearize(&self) -> Result<PreparedRietveldLinearization, RietveldObjectiveError> {
        let selected = self.layout.native_active_rows();
        let products = self.with_inputs(|inputs| {
            self.prepared
                .linearize_selected(inputs, &selected, Some(&self.cache))
        })?;
        let sample_count = self.input.pattern.sample_count();
        let jacobians = products
            .iter()
            .map(|product| (product.d_y.as_slice(), product.parameter_count))
            .collect::<Vec<_>>();
        let jacobian = self
            .layout
            .project_native_jacobians(&jacobians, sample_count)?;
        let calculation = assemble_rietveld_calculation(
            &self.input,
            &self.options,
            products.into_iter().map(|product| product.result).collect(),
        )?;
        Ok(PreparedRietveldLinearization {
            calculation,
            jacobian,
            parameter_count: self.layout.parameters().specs().len(),
        })
    }

    /// Calculate the structural profile and its physical directional derivative.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for direction or engine failures.
    pub fn jvp(&self, direction: &[f64]) -> Result<(Vec<f64>, Vec<f64>), RietveldObjectiveError> {
        let tangents = self.layout.native_tangents(direction)?;
        let tangent_views = tangents.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let products = self.with_inputs(|inputs| self.prepared.jvp(inputs, &tangent_views))?;
        let sample_count = self.input.pattern.sample_count();
        let mut profile = vec![0.0; sample_count];
        let mut derivative = vec![0.0; sample_count];
        for product in products {
            for sample in 0..sample_count {
                profile[sample] += product.result.accumulation.y[sample];
                derivative[sample] += product.d_y[sample];
            }
        }
        Ok((profile, derivative))
    }

    /// Apply the physical structural pattern Jacobian transpose.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for weight or engine failures.
    pub fn vjp(&self, sample_weights: &[f64]) -> Result<Vec<f64>, RietveldObjectiveError> {
        let products = self.with_inputs(|inputs| self.prepared.vjp(inputs, sample_weights))?;
        let gradients = products
            .iter()
            .map(|item| item.gradient.as_slice())
            .collect::<Vec<_>>();
        Ok(self.layout.project_native_gradients(&gradients)?)
    }

    /// Apply `J^T W J + damping I` without a dense sample Jacobian.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for invalid damping or product state.
    pub fn normal_product(
        &self,
        direction: &[f64],
        damping: f64,
    ) -> Result<Vec<f64>, RietveldObjectiveError> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(RietveldObjectiveError::InvalidDamping);
        }
        let (_, derivative) = self.jvp(direction)?;
        let weights = self.weight_samples(&derivative);
        let mut product = self.vjp(&weights)?;
        for (value, direction) in product.iter_mut().zip(direction) {
            *value += damping * direction;
        }
        Ok(product)
    }

    /// Calculate `J^T W (Ycalc - Yobs)` and the complete calculated pattern.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldObjectiveError`] for calculation or reverse-product
    /// failures.
    pub fn gradient(&self) -> Result<(Vec<f64>, Vec<f64>), RietveldObjectiveError> {
        let zero = vec![0.0; self.layout.parameters().specs().len()];
        let (profile, _) = self.jvp(&zero)?;
        let mut background = self.input.pattern.background_y.clone();
        if let Some(model) = &self.input.background {
            for (target, value) in background.iter_mut().zip(
                model
                    .calculate(&self.input.pattern.x_deg)
                    .map_err(RietveldError::Background)?,
            ) {
                *target += value;
            }
        }
        let calculated = profile
            .iter()
            .zip(background)
            .map(|(profile, background)| profile + background)
            .collect::<Vec<_>>();
        if calculated.iter().any(|value| !value.is_finite()) {
            return Err(RietveldError::NonFiniteCalculation.into());
        }
        let observed = self
            .input
            .pattern
            .observed_y
            .as_deref()
            .ok_or(RietveldError::MissingObservations)?;
        let residual = calculated
            .iter()
            .zip(observed)
            .map(|(calculated, observed)| calculated - observed)
            .collect::<Vec<_>>();
        Ok((calculated, self.vjp(&self.weight_samples(&residual))?))
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

    fn with_inputs<R>(
        &self,
        operation: impl FnOnce(
            &[PreparedStructuralModelInputView<'_>],
        ) -> Result<R, StructuralMultiphaseError>,
    ) -> Result<R, RietveldObjectiveError> {
        let owned = self
            .input
            .phases
            .iter()
            .map(|phase| resolve_phase_contributions(phase, &self.input))
            .collect::<Result<Vec<_>, _>>()?;
        let views = owned
            .iter()
            .map(|phase| phase.iter().map(|item| item.as_view()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let inputs = views
            .iter()
            .map(|contributions| PreparedStructuralModelInputView {
                x_deg: &self.input.pattern.x_deg,
                instrument: self.input.instrument,
                axial_geometry: self.input.axial_geometry,
                position_correction: self.input.position_correction,
                contributions,
                support: SupportPolicy::FwhmMultiple(self.options.support_fwhm),
                profile_accuracy: self.options.profile_accuracy,
                calculate_axial_derivatives: true,
            })
            .collect::<Vec<_>>();
        operation(&inputs).map_err(Into::into)
    }
}

/// Invalid prepared Rietveld objective state.
#[derive(Debug)]
pub enum RietveldObjectiveError {
    /// Owned Rietveld state is invalid.
    Rietveld(RietveldError),
    /// Parameter transform state is invalid.
    Parameter(RietveldParameterError),
    /// Structural engine product failed.
    Structural(StructuralMultiphaseError),
    /// Damping must be finite and non-negative.
    InvalidDamping,
    /// A requested reusable derivative allocation exceeds addressable memory.
    AllocationOverflow,
}

impl Display for RietveldObjectiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::InvalidDamping => {
                formatter.write_str("Rietveld damping must be finite and non-negative")
            }
            Self::AllocationOverflow => {
                formatter.write_str("Rietveld dense linearization allocation overflow")
            }
        }
    }
}

impl Error for RietveldObjectiveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rietveld(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Structural(error) => Some(error),
            Self::InvalidDamping | Self::AllocationOverflow => None,
        }
    }
}
impl From<RietveldError> for RietveldObjectiveError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}
impl From<RietveldParameterError> for RietveldObjectiveError {
    fn from(value: RietveldParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<StructuralMultiphaseError> for RietveldObjectiveError {
    fn from(value: StructuralMultiphaseError) -> Self {
        Self::Structural(value)
    }
}
