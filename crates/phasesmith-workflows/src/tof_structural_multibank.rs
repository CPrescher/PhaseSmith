//! Guarded multi-bank structural neutron TOF calculation and objective products.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_core::{
    TOF_GLOBAL_PARAMETER_COUNT, TofBankGeometry, TofError, TofInstrument, TofInstrumentParameter,
};
use phasesmith_crystallography::{IntegratedIntensityCorrectionModel, P1ParameterLayout};
use phasesmith_engine::{
    BuiltInScatteringModel, StructuralTofError, StructuralTofInputView, StructuralTofResult,
    calculate_structural_tof_pattern_jvp_with_context,
    calculate_structural_tof_pattern_vjp_with_context,
    calculate_structural_tof_pattern_with_context,
};
use phasesmith_execution::ExecutionPolicy;
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};

use crate::{
    LatticeBounds, ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec,
    ResidualError, ResidualEvaluation, ResidualOptions, RietveldError, RietveldParameterError,
    RietveldPhase, RietveldStructuralLayout, RietveldStructuralSelection, TofChebyshevBackground,
    TofInstrumentParameterBound, TofLeBailError, evaluate_tof_residuals,
};

/// One observed bank and every bank-local structural-TOF quantity.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofBank {
    /// Stable bank identity used to namespace all local parameters.
    pub bank_id: RecordId,
    /// Microsecond observations, uncertainties, mask, and fixed background.
    pub pattern: TofPatternRecord,
    /// Bank-local TOF calibration and profile coefficients.
    pub instrument: TofInstrument,
    /// Fixed bank scattering angle.
    pub geometry: TofBankGeometry,
    /// Explicit neutral or fixed-angle neutron TOF correction.
    pub correction_model: IntegratedIntensityCorrectionModel,
    /// Bank-local structural intensity scale.
    pub scale: f64,
    /// Closed physical scale bounds.
    pub scale_bounds: ParameterBounds,
    /// Whether the bank scale participates in objective parameter products.
    pub refine_scale: bool,
    /// Optional additive refinable background on top of the fixed pattern background.
    pub background: Option<TofChebyshevBackground>,
    /// Whether every coefficient of `background` participates in products.
    pub refine_background: bool,
    /// Selected bank-local instrument coefficients and their physical bounds.
    pub instrument_bounds: Vec<TofInstrumentParameterBound>,
}

/// Shared structural phase plus one or more explicit TOF banks.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankInput {
    /// One fixed-topology neutron structural phase; its scale/correction are neutral placeholders.
    pub phase: RietveldPhase,
    /// Shared structural parameter families, with `phase_scale` required false.
    pub structural_selection: RietveldStructuralSelection,
    /// Required setting-aware bounds when lattice refinement is selected.
    pub lattice_bounds: Option<LatticeBounds>,
    /// Ordered bank-local observations and models.
    pub banks: Vec<StructuralTofBank>,
    /// Inclusive finite profile support radius in total-FWHM units.
    pub support_fwhm: f64,
    /// Exponential quadrature truncation exponent.
    pub tail_log: f64,
    /// Use supplied one-sigma uncertainties in the summed objective.
    pub use_uncertainty: bool,
    /// Bounded execution policy shared by all bank calculations.
    pub execution: ExecutionPolicy,
}

impl StructuralTofMultiBankInput {
    /// Validate the complete shared/local structural TOF contract.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for invalid physical state,
    /// identity, parameter selection, or observation arrays.
    pub fn validate(&self) -> Result<(), StructuralTofMultiBankError> {
        self.phase.validate()?;
        let definition = self.phase.definition();
        if definition.scattering_model != BuiltInScatteringModel::NeutronNuclear
            || !definition.scattering_real_offset.is_empty()
            || !definition.scattering_imag_offset.is_empty()
        {
            return Err(StructuralTofMultiBankError::InvalidPhaseContract(
                "structural TOF requires built-in neutron scattering without X-ray offsets",
            ));
        }
        if definition.correction_model != IntegratedIntensityCorrectionModel::Neutral
            || definition.scale.to_bits() != 1.0_f64.to_bits()
        {
            return Err(StructuralTofMultiBankError::InvalidPhaseContract(
                "the shared phase must use neutral correction and unit placeholder scale",
            ));
        }
        if self.phase.sample_physics().is_some()
            || self.phase.reflection_domain().is_some()
            || self.phase.contributions()
                != &phasesmith_core::OwnedCwContributions::neutral(definition.hkl.len())
        {
            return Err(StructuralTofMultiBankError::InvalidPhaseContract(
                "CW sample physics and dynamic topology are not part of structural TOF",
            ));
        }
        if self.structural_selection.phase_scale {
            return Err(StructuralTofMultiBankError::InvalidPhaseContract(
                "structural TOF phase scale is bank-local and cannot be shared",
            ));
        }
        if !self.support_fwhm.is_finite()
            || self.support_fwhm <= 0.0
            || !self.tail_log.is_finite()
            || self.tail_log <= 0.0
        {
            return Err(StructuralTofMultiBankError::InvalidSupport);
        }
        if self.banks.is_empty() {
            return Err(StructuralTofMultiBankError::TooFewBanks);
        }
        if self
            .banks
            .iter()
            .map(|bank| &bank.bank_id)
            .collect::<BTreeSet<_>>()
            .len()
            != self.banks.len()
        {
            return Err(StructuralTofMultiBankError::DuplicateBankId);
        }
        for bank in &self.banks {
            validate_bank(bank)?;
        }
        RietveldStructuralLayout::new(
            std::slice::from_ref(&self.phase),
            self.structural_selection,
            std::slice::from_ref(&self.lattice_bounds),
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct BankParameterMapping {
    bank_id: RecordId,
    pattern: TofPatternRecord,
    geometry: TofBankGeometry,
    correction_model: IntegratedIntensityCorrectionModel,
    scale_bounds: ParameterBounds,
    scale: Option<usize>,
    instrument_bounds: Vec<TofInstrumentParameterBound>,
    instrument: Vec<(TofInstrumentParameter, usize)>,
    background_contract: Option<(RecordId, [f64; 2], usize)>,
    background: Vec<usize>,
}

/// Stable shared/local physical parameter packing for structural TOF banks.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankLayout {
    parameters: ParameterSet,
    structural: RietveldStructuralLayout,
    structural_count: usize,
    structural_selection: RietveldStructuralSelection,
    lattice_bounds: Option<LatticeBounds>,
    reflection_ids: Vec<String>,
    support_fwhm: f64,
    tail_log: f64,
    use_uncertainty: bool,
    execution: ExecutionPolicy,
    banks: Vec<BankParameterMapping>,
}

impl StructuralTofMultiBankLayout {
    /// Build the symmetry-aware shared and namespaced bank-local parameter set.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for an invalid request or scalar.
    pub fn new(input: &StructuralTofMultiBankInput) -> Result<Self, StructuralTofMultiBankError> {
        input.validate()?;
        let structural = RietveldStructuralLayout::new(
            std::slice::from_ref(&input.phase),
            input.structural_selection,
            std::slice::from_ref(&input.lattice_bounds),
        )?;
        let mut specs = structural.parameters().specs().to_vec();
        let structural_count = specs.len();
        let mut banks = Vec::with_capacity(input.banks.len());
        for bank in &input.banks {
            let owner = bank.bank_id.as_str();
            let scale = if bank.refine_scale {
                let index = specs.len();
                specs.push(ParameterSpec::new(
                    ParameterKey::new("tof_scale", owner, "scale")?,
                    bank.scale,
                    "relative",
                    bank.scale_bounds,
                    bank.scale.abs().max(1.0),
                    true,
                )?);
                Some(index)
            } else {
                None
            };
            let instrument_values = bank.instrument.values();
            let mut instrument = Vec::with_capacity(bank.instrument_bounds.len());
            for bound in &bank.instrument_bounds {
                let index = specs.len();
                let value = instrument_values[bound.parameter.index()];
                let half_span = 0.5 * (bound.upper - bound.lower);
                specs.push(ParameterSpec::new(
                    ParameterKey::new("tof_instrument", owner, bound.parameter.name())?,
                    value,
                    instrument_unit(bound.parameter),
                    ParameterBounds::new(bound.lower, bound.upper)?,
                    value.abs().max(half_span).max(f64::EPSILON.sqrt()),
                    true,
                )?);
                instrument.push((bound.parameter, index));
            }
            let mut background = Vec::new();
            if bank.refine_background {
                let model = bank.background.as_ref().ok_or(
                    StructuralTofMultiBankError::InvalidBankContract(
                        "refine_background requires a background model",
                    ),
                )?;
                for (order, value) in model.coefficients().iter().copied().enumerate() {
                    let index = specs.len();
                    specs.push(ParameterSpec::new(
                        ParameterKey::new("tof_background", owner, format!("coefficient_{order}"))?,
                        value,
                        "intensity",
                        ParameterBounds::default(),
                        value.abs().max(1.0),
                        true,
                    )?);
                    background.push(index);
                }
            }
            banks.push(BankParameterMapping {
                bank_id: bank.bank_id.clone(),
                pattern: bank.pattern.clone(),
                geometry: bank.geometry,
                correction_model: bank.correction_model,
                scale_bounds: bank.scale_bounds,
                scale,
                instrument_bounds: bank.instrument_bounds.clone(),
                instrument,
                background_contract: bank.background.as_ref().map(|model| {
                    (
                        model.background_id().clone(),
                        model.domain_us(),
                        model.coefficients().len(),
                    )
                }),
                background,
            });
        }
        Ok(Self {
            parameters: ParameterSet::new(specs)?,
            structural,
            structural_count,
            structural_selection: input.structural_selection,
            lattice_bounds: input.lattice_bounds.clone(),
            reflection_ids: input.phase.reflection_ids().to_vec(),
            support_fwhm: input.support_fwhm,
            tail_log: input.tail_log,
            use_uncertainty: input.use_uncertainty,
            execution: input.execution.clone(),
            banks,
        })
    }

    /// Borrow stable physical parameters in solver order.
    #[must_use]
    pub const fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }

    /// Install one absolute physical state into cloned shared/local records.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for a stale contract, wrong
    /// value count, violated bound, or invalid resulting physical model.
    pub fn apply_values(
        &self,
        input: &StructuralTofMultiBankInput,
        values: &[f64],
    ) -> Result<StructuralTofMultiBankInput, StructuralTofMultiBankError> {
        let current = self
            .parameters
            .specs()
            .iter()
            .map(ParameterSpec::value)
            .collect::<Vec<_>>();
        self.apply_value_change(input, &current, values)
    }

    /// Install a change between two absolute solver states.
    ///
    /// Symmetry-constrained site coordinates are local tangent coordinates,
    /// so their difference is applied to the current structure. Every other
    /// selected value is installed absolutely.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for a stale contract, wrong
    /// value count, violated bound, or invalid resulting physical model.
    pub fn apply_value_change(
        &self,
        input: &StructuralTofMultiBankInput,
        current_values: &[f64],
        values: &[f64],
    ) -> Result<StructuralTofMultiBankInput, StructuralTofMultiBankError> {
        self.validate_contract(input)?;
        if current_values.len() != self.parameters.specs().len()
            || values.len() != self.parameters.specs().len()
            || current_values.iter().any(|value| !value.is_finite())
        {
            return Err(StructuralTofMultiBankError::ParameterLengthMismatch);
        }
        for (spec, value) in self.parameters.specs().iter().zip(values) {
            if !value.is_finite() || !spec.bounds().contains(*value) {
                return Err(StructuralTofMultiBankError::Parameter(
                    ParameterError::ValueOutsideBounds {
                        key: spec.key().clone(),
                        value: *value,
                    },
                ));
            }
        }
        let phase = self
            .structural
            .apply_value_change(
                std::slice::from_ref(&input.phase),
                &current_values[..self.structural_count],
                &values[..self.structural_count],
            )?
            .into_iter()
            .next()
            .ok_or(StructuralTofMultiBankError::InternalInvariant)?;
        let mut result = input.clone();
        result.phase = phase;
        for ((bank, mapping), original) in
            result.banks.iter_mut().zip(&self.banks).zip(&input.banks)
        {
            if let Some(index) = mapping.scale {
                bank.scale = values[index];
            }
            let mut instrument_values = original.instrument.values();
            for &(parameter, index) in &mapping.instrument {
                instrument_values[parameter.index()] = values[index];
            }
            bank.instrument = TofInstrument::from_values(instrument_values)?;
            if !mapping.background.is_empty() {
                let coefficients = mapping
                    .background
                    .iter()
                    .map(|index| values[*index])
                    .collect();
                bank.background = Some(
                    original
                        .background
                        .as_ref()
                        .ok_or(StructuralTofMultiBankError::InternalInvariant)?
                        .with_coefficients(coefficients)?,
                );
            }
        }
        result.validate()?;
        Ok(result)
    }

    fn validate_contract(
        &self,
        input: &StructuralTofMultiBankInput,
    ) -> Result<(), StructuralTofMultiBankError> {
        input.validate()?;
        self.structural
            .validate_phases(std::slice::from_ref(&input.phase))?;
        if input.structural_selection != self.structural_selection
            || input.lattice_bounds != self.lattice_bounds
            || input.phase.reflection_ids() != self.reflection_ids
            || input.support_fwhm.to_bits() != self.support_fwhm.to_bits()
            || input.tail_log.to_bits() != self.tail_log.to_bits()
            || input.use_uncertainty != self.use_uncertainty
            || input.execution != self.execution
            || input.banks.len() != self.banks.len()
        {
            return Err(StructuralTofMultiBankError::BankContractMismatch);
        }
        for (bank, mapping) in input.banks.iter().zip(&self.banks) {
            let background_contract = bank.background.as_ref().map(|model| {
                (
                    model.background_id().clone(),
                    model.domain_us(),
                    model.coefficients().len(),
                )
            });
            if bank.bank_id != mapping.bank_id
                || bank.pattern != mapping.pattern
                || bank.geometry != mapping.geometry
                || bank.correction_model != mapping.correction_model
                || bank.scale_bounds != mapping.scale_bounds
                || bank.refine_scale != mapping.scale.is_some()
                || bank.instrument_bounds != mapping.instrument_bounds
                || bank.refine_background == mapping.background.is_empty()
                || background_contract != mapping.background_contract
            {
                return Err(StructuralTofMultiBankError::BankContractMismatch);
            }
        }
        Ok(())
    }

    fn native_tangent(
        &self,
        bank_index: usize,
        direction: &[f64],
        native_count: usize,
    ) -> Result<Vec<f64>, StructuralTofMultiBankError> {
        if direction.len() != self.parameters.specs().len() {
            return Err(StructuralTofMultiBankError::ParameterLengthMismatch);
        }
        let mut tangent = self
            .structural
            .native_tangents(&direction[..self.structural_count])?
            .into_iter()
            .next()
            .ok_or(StructuralTofMultiBankError::InternalInvariant)?;
        if tangent.len() != native_count {
            return Err(StructuralTofMultiBankError::InternalInvariant);
        }
        if let Some(index) = self.banks[bank_index].scale {
            tangent[native_count - 1] = direction[index];
        }
        Ok(tangent)
    }

    fn scatter_native_gradient(
        &self,
        bank_index: usize,
        native: &[f64],
        output: &mut [f64],
    ) -> Result<(), StructuralTofMultiBankError> {
        let shared = self.structural.project_native_gradients(&[native])?;
        for (target, value) in output[..self.structural_count].iter_mut().zip(shared) {
            *target += value;
        }
        if let Some(index) = self.banks[bank_index].scale {
            output[index] += native
                .last()
                .copied()
                .ok_or(StructuralTofMultiBankError::InternalInvariant)?;
        }
        Ok(())
    }
}

/// One bank's display-ready structural TOF calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofBankCalculation {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Complete calculated values including fixed and refinable background.
    pub y: Vec<f64>,
    /// Structural finite-support profile contribution.
    pub profile_y: Vec<f64>,
    /// Fixed plus refinable background contribution.
    pub background_y: Vec<f64>,
    /// Structural reflection and fused profile intermediates.
    pub structural: StructuralTofResult,
    /// Per-bank residual metrics under the objective weighting contract.
    pub metrics: ResidualEvaluation,
}

/// Ordered multi-bank structural TOF calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankCalculation {
    /// Bank calculations in request order.
    pub banks: Vec<StructuralTofBankCalculation>,
    /// Half the summed weighted squared residual over all banks.
    pub objective: f64,
}

/// One bank's profile and joint-parameter directional derivative.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankProduct {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Complete calculated values at the prepared state.
    pub y: Vec<f64>,
    /// Directional derivative of calculated values.
    pub derivative: Vec<f64>,
}

/// Complete objective values and gradient in joint physical order.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTofMultiBankGradient {
    /// Accepted-state calculations and residual metrics.
    pub calculation: StructuralTofMultiBankCalculation,
    /// Gradient of half the summed weighted residual square.
    pub gradient: Vec<f64>,
}

/// Prepared matrix-free structural TOF sum over all banks.
pub struct PreparedStructuralTofMultiBankObjective {
    input: StructuralTofMultiBankInput,
    layout: StructuralTofMultiBankLayout,
    background_bases: Vec<Option<crate::TofChebyshevBasis>>,
}

impl PreparedStructuralTofMultiBankObjective {
    /// Prepare stable parameter mappings and bank-local background bases.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for invalid request state.
    pub fn new(input: StructuralTofMultiBankInput) -> Result<Self, StructuralTofMultiBankError> {
        let layout = StructuralTofMultiBankLayout::new(&input)?;
        let background_bases = input
            .banks
            .iter()
            .map(|bank| {
                bank.background
                    .as_ref()
                    .map(|model| model.basis(&bank.pattern.tof_us))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input,
            layout,
            background_bases,
        })
    }

    /// Borrow the complete prepared request.
    #[must_use]
    pub const fn input(&self) -> &StructuralTofMultiBankInput {
        &self.input
    }

    /// Borrow the stable shared/local parameter layout.
    #[must_use]
    pub const fn layout(&self) -> &StructuralTofMultiBankLayout {
        &self.layout
    }

    /// Calculate all banks and the summed objective.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for a numerical or residual failure.
    pub fn calculate(
        &self,
    ) -> Result<StructuralTofMultiBankCalculation, StructuralTofMultiBankError> {
        let mut banks = Vec::with_capacity(self.input.banks.len());
        let mut objective = 0.0;
        for (index, bank) in self.input.banks.iter().enumerate() {
            let structural = calculate_bank(&self.input, bank)?;
            let profile_y = structural.accumulation.y.clone();
            let background_y = background_values(bank, self.background_bases[index].as_ref())?;
            let y = profile_y
                .iter()
                .zip(&background_y)
                .map(|(profile, background)| profile + background)
                .collect::<Vec<_>>();
            let metrics = evaluate_tof_residuals(
                &bank.pattern,
                &y,
                ResidualOptions {
                    use_uncertainty: self.input.use_uncertainty,
                    parameter_count: self.layout.parameters.specs().len(),
                },
            )?;
            objective += 0.5 * metrics.chi_square;
            banks.push(StructuralTofBankCalculation {
                bank_id: bank.bank_id.clone(),
                y,
                profile_y,
                background_y,
                structural,
                metrics,
            });
        }
        Ok(StructuralTofMultiBankCalculation { banks, objective })
    }

    /// Apply every bank Jacobian to one joint physical direction.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for an invalid direction or product.
    pub fn jvp(
        &self,
        direction: &[f64],
    ) -> Result<Vec<StructuralTofMultiBankProduct>, StructuralTofMultiBankError> {
        let native_count = P1ParameterLayout {
            site_count: self.input.phase.definition().fractional_xyz.len(),
        }
        .parameter_count();
        let mut products = Vec::with_capacity(self.input.banks.len());
        for (bank_index, bank) in self.input.banks.iter().enumerate() {
            let tangent = self
                .layout
                .native_tangent(bank_index, direction, native_count)?;
            let species = species(self.input.phase.definition());
            let view = bank_view(&self.input, bank, &species);
            let forward = calculate_structural_tof_pattern_jvp_with_context(
                self.input.phase.definition().cell,
                &self.input.phase.definition().space_group,
                &view,
                &tangent,
                self.input.execution.context(),
            )?;
            let mut derivative = forward.d_y;
            add_instrument_jvp(
                &mut derivative,
                &forward.result,
                &self.layout.banks[bank_index],
                direction,
            )?;
            add_background_jvp(
                &mut derivative,
                self.background_bases[bank_index].as_ref(),
                &self.layout.banks[bank_index],
                direction,
            )?;
            let background = background_values(bank, self.background_bases[bank_index].as_ref())?;
            let y = forward
                .result
                .accumulation
                .y
                .iter()
                .zip(background)
                .map(|(profile, background)| profile + background)
                .collect();
            products.push(StructuralTofMultiBankProduct {
                bank_id: bank.bank_id.clone(),
                y,
                derivative,
            });
        }
        Ok(products)
    }

    /// Apply the transpose of all bank Jacobians and sum shared rows.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for bank/sample shape or product failures.
    pub fn vjp(&self, weights: &[Vec<f64>]) -> Result<Vec<f64>, StructuralTofMultiBankError> {
        if weights.len() != self.input.banks.len() {
            return Err(StructuralTofMultiBankError::BankWeightCountMismatch);
        }
        let mut result = vec![0.0; self.layout.parameters.specs().len()];
        for (bank_index, (bank, weights)) in self.input.banks.iter().zip(weights).enumerate() {
            let species = species(self.input.phase.definition());
            let view = bank_view(&self.input, bank, &species);
            let reverse = calculate_structural_tof_pattern_vjp_with_context(
                self.input.phase.definition().cell,
                &self.input.phase.definition().space_group,
                &view,
                weights,
                self.input.execution.context(),
            )?;
            self.layout
                .scatter_native_gradient(bank_index, &reverse.gradient, &mut result)?;
            add_instrument_vjp(
                &mut result,
                &reverse.result,
                &self.layout.banks[bank_index],
                weights,
            )?;
            add_background_vjp(
                &mut result,
                self.background_bases[bank_index].as_ref(),
                &self.layout.banks[bank_index],
                weights,
            )?;
        }
        Ok(result)
    }

    /// Evaluate the accepted objective and its physical gradient.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for residual or reverse-product state.
    pub fn gradient(&self) -> Result<StructuralTofMultiBankGradient, StructuralTofMultiBankError> {
        let calculation = self.calculate()?;
        let weights = self
            .input
            .banks
            .iter()
            .zip(&calculation.banks)
            .map(|(bank, calculation)| {
                objective_weights(bank, &calculation.metrics, self.input.use_uncertainty)
            })
            .collect::<Vec<_>>();
        let gradient = self.vjp(&weights)?;
        Ok(StructuralTofMultiBankGradient {
            calculation,
            gradient,
        })
    }

    /// Apply `J^T W J + damping I` in joint physical coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`StructuralTofMultiBankError`] for invalid damping or products.
    pub fn normal_product(
        &self,
        direction: &[f64],
        damping: f64,
    ) -> Result<Vec<f64>, StructuralTofMultiBankError> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(StructuralTofMultiBankError::InvalidDamping);
        }
        let products = self.jvp(direction)?;
        let weights = self
            .input
            .banks
            .iter()
            .zip(products)
            .map(|(bank, product)| {
                weighted_direction(bank, product.derivative, self.input.use_uncertainty)
            })
            .collect::<Vec<_>>();
        let mut result = self.vjp(&weights)?;
        for (value, direction) in result.iter_mut().zip(direction) {
            *value += damping * direction;
        }
        Ok(result)
    }
}

fn validate_bank(bank: &StructuralTofBank) -> Result<(), StructuralTofMultiBankError> {
    bank.pattern.validate()?;
    if bank.pattern.observed_y.is_none() {
        return Err(StructuralTofMultiBankError::MissingObservations);
    }
    bank.instrument.validate()?;
    bank.geometry.validate()?;
    if !bank.scale.is_finite() || bank.scale < 0.0 || !bank.scale_bounds.contains(bank.scale) {
        return Err(StructuralTofMultiBankError::InvalidBankContract(
            "bank scale must be finite, non-negative, and inside its bounds",
        ));
    }
    match bank.correction_model {
        IntegratedIntensityCorrectionModel::Neutral => {}
        IntegratedIntensityCorrectionModel::TimeOfFlightNeutronLorentz { two_theta_deg }
            if two_theta_deg.to_bits() == bank.geometry.two_theta_deg.to_bits() => {}
        _ => {
            return Err(StructuralTofMultiBankError::InvalidBankContract(
                "bank correction must be neutral or match the bank angle exactly",
            ));
        }
    }
    let mut selected = BTreeSet::new();
    let values = bank.instrument.values();
    for bound in &bank.instrument_bounds {
        if !bound.lower.is_finite()
            || !bound.upper.is_finite()
            || bound.lower >= bound.upper
            || !selected.insert(bound.parameter)
            || !(bound.lower..=bound.upper).contains(&values[bound.parameter.index()])
        {
            return Err(StructuralTofMultiBankError::InvalidBankContract(
                "instrument selections require unique finite bounds containing the current value",
            ));
        }
    }
    if let Some(background) = &bank.background {
        background.validate()?;
        background.basis(&bank.pattern.tof_us)?;
    } else if bank.refine_background {
        return Err(StructuralTofMultiBankError::InvalidBankContract(
            "refine_background requires a background model",
        ));
    }
    Ok(())
}

fn species(definition: &phasesmith_engine::StructuralPhaseDefinition) -> Vec<&str> {
    definition
        .scattering_species
        .iter()
        .map(String::as_str)
        .collect()
}

fn bank_view<'a>(
    input: &'a StructuralTofMultiBankInput,
    bank: &'a StructuralTofBank,
    species: &'a [&'a str],
) -> StructuralTofInputView<'a> {
    let definition = input.phase.definition();
    StructuralTofInputView {
        tof_us: &bank.pattern.tof_us,
        hkl: &definition.hkl,
        multiplicity: &definition.multiplicity,
        fractional_xyz: &definition.fractional_xyz,
        occupancy: &definition.occupancy,
        u_iso_angstrom2: &definition.u_iso_angstrom2,
        anisotropic_mask: &definition.anisotropic_mask,
        u_aniso_cif_angstrom2: &definition.u_aniso_cif_angstrom2,
        scattering_species: species,
        scale: bank.scale,
        coordinate_tolerance: definition.coordinate_tolerance,
        correction_model: bank.correction_model,
        bank_geometry: bank.geometry,
        instrument: bank.instrument,
        support_fwhm: input.support_fwhm,
        tail_log: input.tail_log,
    }
}

fn calculate_bank(
    input: &StructuralTofMultiBankInput,
    bank: &StructuralTofBank,
) -> Result<StructuralTofResult, StructuralTofMultiBankError> {
    let species = species(input.phase.definition());
    Ok(calculate_structural_tof_pattern_with_context(
        input.phase.definition().cell,
        &input.phase.definition().space_group,
        &bank_view(input, bank, &species),
        input.execution.context(),
    )?)
}

fn background_values(
    bank: &StructuralTofBank,
    basis: Option<&crate::TofChebyshevBasis>,
) -> Result<Vec<f64>, StructuralTofMultiBankError> {
    let mut values = if let (Some(model), Some(basis)) = (&bank.background, basis) {
        model.calculate_from_basis(basis)?
    } else {
        vec![0.0; bank.pattern.sample_count()]
    };
    for (value, fixed) in values.iter_mut().zip(&bank.pattern.background_y) {
        *value += fixed;
    }
    Ok(values)
}

fn add_instrument_jvp(
    derivative: &mut [f64],
    result: &StructuralTofResult,
    mapping: &BankParameterMapping,
    direction: &[f64],
) -> Result<(), StructuralTofMultiBankError> {
    let global = result
        .accumulation
        .derivatives
        .global
        .as_ref()
        .ok_or(StructuralTofMultiBankError::InternalInvariant)?;
    if global.parameter_count != TOF_GLOBAL_PARAMETER_COUNT
        || global.values.len() != TOF_GLOBAL_PARAMETER_COUNT * derivative.len()
    {
        return Err(StructuralTofMultiBankError::InternalInvariant);
    }
    for &(parameter, index) in &mapping.instrument {
        let row = &global.values
            [parameter.index() * derivative.len()..(parameter.index() + 1) * derivative.len()];
        for (target, value) in derivative.iter_mut().zip(row) {
            *target += direction[index] * value;
        }
    }
    Ok(())
}

fn add_instrument_vjp(
    output: &mut [f64],
    result: &StructuralTofResult,
    mapping: &BankParameterMapping,
    weights: &[f64],
) -> Result<(), StructuralTofMultiBankError> {
    let global = result
        .accumulation
        .derivatives
        .global
        .as_ref()
        .ok_or(StructuralTofMultiBankError::InternalInvariant)?;
    if global.values.len() != TOF_GLOBAL_PARAMETER_COUNT * weights.len() {
        return Err(StructuralTofMultiBankError::InternalInvariant);
    }
    for &(parameter, index) in &mapping.instrument {
        let row = &global.values
            [parameter.index() * weights.len()..(parameter.index() + 1) * weights.len()];
        output[index] += row.iter().zip(weights).map(|(a, b)| a * b).sum::<f64>();
    }
    Ok(())
}

fn add_background_jvp(
    derivative: &mut [f64],
    basis: Option<&crate::TofChebyshevBasis>,
    mapping: &BankParameterMapping,
    direction: &[f64],
) -> Result<(), StructuralTofMultiBankError> {
    if mapping.background.is_empty() {
        return Ok(());
    }
    let basis = basis.ok_or(StructuralTofMultiBankError::InternalInvariant)?;
    if basis.rows != derivative.len() || basis.columns != mapping.background.len() {
        return Err(StructuralTofMultiBankError::InternalInvariant);
    }
    for (sample, row) in basis.values.chunks_exact(basis.columns).enumerate() {
        derivative[sample] += row
            .iter()
            .zip(&mapping.background)
            .map(|(value, index)| value * direction[*index])
            .sum::<f64>();
    }
    Ok(())
}

fn add_background_vjp(
    output: &mut [f64],
    basis: Option<&crate::TofChebyshevBasis>,
    mapping: &BankParameterMapping,
    weights: &[f64],
) -> Result<(), StructuralTofMultiBankError> {
    if mapping.background.is_empty() {
        return Ok(());
    }
    let basis = basis.ok_or(StructuralTofMultiBankError::InternalInvariant)?;
    if basis.rows != weights.len() || basis.columns != mapping.background.len() {
        return Err(StructuralTofMultiBankError::InternalInvariant);
    }
    for (sample, row) in basis.values.chunks_exact(basis.columns).enumerate() {
        for (value, index) in row.iter().zip(&mapping.background) {
            output[*index] += weights[sample] * value;
        }
    }
    Ok(())
}

fn objective_weights(
    bank: &StructuralTofBank,
    metrics: &ResidualEvaluation,
    use_uncertainty: bool,
) -> Vec<f64> {
    (0..bank.pattern.sample_count())
        .map(|sample| {
            if !metrics.included[sample] {
                0.0
            } else if use_uncertainty {
                let sigma = bank.pattern.uncertainty.as_ref().map_or(1.0, |v| v[sample]);
                metrics.residual[sample] / (sigma * sigma)
            } else {
                metrics.residual[sample]
            }
        })
        .collect()
}

fn weighted_direction(
    bank: &StructuralTofBank,
    direction: Vec<f64>,
    use_uncertainty: bool,
) -> Vec<f64> {
    direction
        .into_iter()
        .enumerate()
        .map(|(sample, value)| {
            if bank.pattern.mask.as_ref().is_some_and(|mask| !mask[sample]) {
                0.0
            } else if use_uncertainty {
                let sigma = bank.pattern.uncertainty.as_ref().map_or(1.0, |v| v[sample]);
                value / (sigma * sigma)
            } else {
                value
            }
        })
        .collect()
}

const fn instrument_unit(parameter: TofInstrumentParameter) -> &'static str {
    match parameter {
        TofInstrumentParameter::Zero | TofInstrumentParameter::Z => "microsecond",
        TofInstrumentParameter::Difc | TofInstrumentParameter::X => "microsecond/angstrom",
        TofInstrumentParameter::Difa | TofInstrumentParameter::Y => "microsecond/angstrom^2",
        TofInstrumentParameter::Difb => "microsecond*angstrom",
        TofInstrumentParameter::Alpha => "microsecond^-1*angstrom",
        TofInstrumentParameter::Beta0 => "microsecond^-1",
        TofInstrumentParameter::Beta1 => "angstrom^4/microsecond",
        TofInstrumentParameter::Betaq => "angstrom^2/microsecond",
        TofInstrumentParameter::Sigma0 => "microsecond^2",
        TofInstrumentParameter::Sigma1 => "microsecond^2/angstrom^2",
        TofInstrumentParameter::Sigma2 => "microsecond^2/angstrom^4",
        TofInstrumentParameter::Sigmaq => "microsecond^2/angstrom",
    }
}

/// Invalid structural multi-bank TOF request or objective product.
#[derive(Debug)]
pub enum StructuralTofMultiBankError {
    /// At least one bank is required for a structural TOF objective.
    TooFewBanks,
    /// Stable bank IDs must be unique.
    DuplicateBankId,
    /// Observed intensities are required in every bank.
    MissingObservations,
    /// Shared structural phase uses an unsupported TOF contract.
    InvalidPhaseContract(&'static str),
    /// A bank-local model or selection is invalid.
    InvalidBankContract(&'static str),
    /// Support or tail controls are nonphysical.
    InvalidSupport,
    /// Prepared bank identities or selected families changed.
    BankContractMismatch,
    /// Physical value or direction length differs from the layout.
    ParameterLengthMismatch,
    /// Reverse products do not contain one vector per bank.
    BankWeightCountMismatch,
    /// Damping must be finite and non-negative.
    InvalidDamping,
    /// A supposedly validated internal shape was inconsistent.
    InternalInvariant,
    /// TOF pattern domain validation failed.
    Pattern(DomainError),
    /// Shared phase validation failed.
    Rietveld(RietveldError),
    /// Symmetry-aware structural parameter mapping failed.
    StructuralParameters(RietveldParameterError),
    /// Stable scalar parameter validation failed.
    Parameter(ParameterError),
    /// Background evaluation failed.
    Background(TofLeBailError),
    /// Structural TOF engine evaluation failed.
    Structural(StructuralTofError),
    /// TOF instrument or bank geometry failed.
    Tof(TofError),
    /// Residual evaluation failed.
    Residual(ResidualError),
}

impl Display for StructuralTofMultiBankError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewBanks => formatter.write_str("structural TOF requires at least one bank"),
            Self::DuplicateBankId => formatter.write_str("structural TOF bank IDs must be unique"),
            Self::MissingObservations => {
                formatter.write_str("every structural TOF bank requires observations")
            }
            Self::InvalidPhaseContract(reason) | Self::InvalidBankContract(reason) => {
                formatter.write_str(reason)
            }
            Self::InvalidSupport => {
                formatter.write_str("structural TOF support controls must be finite and positive")
            }
            Self::BankContractMismatch => formatter
                .write_str("structural TOF bank contract changed under the prepared layout"),
            Self::ParameterLengthMismatch => {
                formatter.write_str("structural TOF parameter value/direction length is wrong")
            }
            Self::BankWeightCountMismatch => formatter
                .write_str("structural TOF reverse products require one weight vector per bank"),
            Self::InvalidDamping => {
                formatter.write_str("structural TOF damping must be finite and non-negative")
            }
            Self::InternalInvariant => {
                formatter.write_str("structural TOF internal shape invariant failed")
            }
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::StructuralParameters(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Background(error) => Display::fmt(error, formatter),
            Self::Structural(error) => Display::fmt(error, formatter),
            Self::Tof(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for StructuralTofMultiBankError {}

impl From<DomainError> for StructuralTofMultiBankError {
    fn from(value: DomainError) -> Self {
        Self::Pattern(value)
    }
}
impl From<RietveldError> for StructuralTofMultiBankError {
    fn from(value: RietveldError) -> Self {
        Self::Rietveld(value)
    }
}
impl From<RietveldParameterError> for StructuralTofMultiBankError {
    fn from(value: RietveldParameterError) -> Self {
        Self::StructuralParameters(value)
    }
}
impl From<ParameterError> for StructuralTofMultiBankError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}
impl From<TofLeBailError> for StructuralTofMultiBankError {
    fn from(value: TofLeBailError) -> Self {
        Self::Background(value)
    }
}
impl From<StructuralTofError> for StructuralTofMultiBankError {
    fn from(value: StructuralTofError) -> Self {
        Self::Structural(value)
    }
}
impl From<TofError> for StructuralTofMultiBankError {
    fn from(value: TofError) -> Self {
        Self::Tof(value)
    }
}
impl From<ResidualError> for StructuralTofMultiBankError {
    fn from(value: ResidualError) -> Self {
        Self::Residual(value)
    }
}
