//! Native fixed-reflection Le Bail integrated-intensity extraction.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::DMatrix;
use phasesmith_core::{
    Accumulation, ConstantWavelengthInstrument, CwContributionsError, GridView,
    OwnedCwContributionArrays, OwnedCwContributions, ProfileError, SupportPolicy,
    accumulate_cw_contributions_batch_with_context,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_model::{DomainError, PatternRecord};

use crate::{
    DiagnosticValue, RefinementEventKind, RefinementLimits, RefinementRuntime, ResidualError,
    ResidualEvaluation, ResidualOptions, RuntimeError, TerminationReason, evaluate_residuals,
};

/// One fixed reflection phase whose integrated intensities are extracted.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailPhase {
    phase_id: String,
    name: String,
    reflection_ids: Vec<String>,
    hkl: Vec<[i32; 3]>,
    d_spacing_angstrom: Vec<f64>,
    two_theta_deg: Vec<f64>,
    integrated_intensity: Vec<f64>,
    scale: f64,
    preserve_unobserved: Vec<bool>,
}

impl LeBailPhase {
    /// Validate and own one ordered fixed-reflection phase.
    ///
    /// `preserve_unobserved` marks generated reflections outside the currently
    /// visible interval. Their checkpoint intensity is retained when their
    /// finite profile support has no included samples. Pass an empty vector for
    /// ordinary fixed reflection lists.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError::InvalidPhase`] for invalid identity, shape, or
    /// numerical state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        phase_id: impl Into<String>,
        name: impl Into<String>,
        reflection_ids: Vec<String>,
        hkl: Vec<[i32; 3]>,
        d_spacing_angstrom: Vec<f64>,
        two_theta_deg: Vec<f64>,
        integrated_intensity: Vec<f64>,
        scale: f64,
        preserve_unobserved: Vec<bool>,
    ) -> Result<Self, LeBailError> {
        let phase = Self {
            phase_id: phase_id.into(),
            name: name.into(),
            reflection_ids,
            hkl,
            d_spacing_angstrom,
            two_theta_deg,
            integrated_intensity,
            scale,
            preserve_unobserved,
        };
        phase.validate()?;
        Ok(phase)
    }

    fn validate(&self) -> Result<(), LeBailError> {
        validate_stable_label("phase_id", &self.phase_id)?;
        if self.name.trim().is_empty() {
            return Err(invalid_phase("phase name must be non-empty"));
        }
        let count = self.reflection_ids.len();
        if count == 0 {
            return Err(invalid_phase("at least one reflection is required"));
        }
        if self.hkl.len() != count
            || self.d_spacing_angstrom.len() != count
            || self.two_theta_deg.len() != count
            || self.integrated_intensity.len() != count
            || (!self.preserve_unobserved.is_empty() && self.preserve_unobserved.len() != count)
        {
            return Err(invalid_phase("reflection arrays must have equal lengths"));
        }
        let mut identities = std::collections::BTreeSet::new();
        for reflection_id in &self.reflection_ids {
            validate_stable_label("reflection_id", reflection_id)?;
            if !identities.insert(reflection_id) {
                return Err(invalid_phase(
                    "reflection IDs must be unique within a phase",
                ));
            }
        }
        if self
            .d_spacing_angstrom
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid_phase("d-spacings must be positive and finite"));
        }
        if self
            .two_theta_deg
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 180.0)
        {
            return Err(invalid_phase(
                "reflection positions must lie strictly inside (0, 180) degrees",
            ));
        }
        if self
            .integrated_intensity
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_phase(
                "integrated intensities must be non-negative and finite",
            ));
        }
        if !self.scale.is_finite() || self.scale < 0.0 {
            return Err(invalid_phase("phase scale must be non-negative and finite"));
        }
        Ok(())
    }

    /// Borrow the stable phase ID.
    #[must_use]
    pub fn phase_id(&self) -> &str {
        &self.phase_id
    }

    /// Borrow the display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow reflection IDs in calculation order.
    #[must_use]
    pub fn reflection_ids(&self) -> &[String] {
        &self.reflection_ids
    }

    /// Borrow Miller indices in reflection order.
    #[must_use]
    pub fn hkl(&self) -> &[[i32; 3]] {
        &self.hkl
    }

    /// Borrow d-spacings in ångströms.
    #[must_use]
    pub fn d_spacing_angstrom(&self) -> &[f64] {
        &self.d_spacing_angstrom
    }

    /// Borrow fixed reflection positions in degrees `2theta`.
    #[must_use]
    pub fn two_theta_deg(&self) -> &[f64] {
        &self.two_theta_deg
    }

    /// Borrow current integrated intensities.
    #[must_use]
    pub fn integrated_intensity(&self) -> &[f64] {
        &self.integrated_intensity
    }

    /// Return the phase scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Borrow the preserve-if-unobserved mask.
    #[must_use]
    pub fn preserve_unobserved(&self) -> &[bool] {
        &self.preserve_unobserved
    }

    fn replace_intensities(&self, values: &[f64]) -> Result<Self, LeBailError> {
        if values.len() != self.integrated_intensity.len()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid_phase(
                "replacement intensities must match and remain non-negative",
            ));
        }
        let mut phase = self.clone();
        phase.integrated_intensity.copy_from_slice(values);
        Ok(phase)
    }
}

/// Observations, instrument, and ordered fixed-reflection phases.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailInput {
    /// Observed pattern and fixed supplied background.
    pub pattern: PatternRecord,
    /// Constant-wavelength U/V/W/X/Y profile.
    pub instrument: ConstantWavelengthInstrument,
    /// Ordered non-empty phase list.
    pub phases: Vec<LeBailPhase>,
}

impl LeBailInput {
    /// Validate a complete fixed-reflection Le Bail request.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] for invalid observations, instrument, phase
    /// state, or repeated phase IDs.
    pub fn new(
        pattern: PatternRecord,
        instrument: ConstantWavelengthInstrument,
        phases: Vec<LeBailPhase>,
    ) -> Result<Self, LeBailError> {
        pattern.validate().map_err(LeBailError::Pattern)?;
        if pattern.observed_y.is_none() {
            return Err(LeBailError::MissingObservations);
        }
        instrument
            .validate()
            .map_err(|error| LeBailError::Profile {
                message: error.to_string(),
            })?;
        if phases.is_empty() {
            return Err(invalid_phase("at least one phase is required"));
        }
        let mut phase_ids = std::collections::BTreeSet::new();
        for phase in &phases {
            phase.validate()?;
            if !phase_ids.insert(phase.phase_id()) {
                return Err(invalid_phase("phase IDs must be unique"));
            }
        }
        Ok(Self {
            pattern,
            instrument,
            phases,
        })
    }
}

/// Deterministic controls for fixed-reflection extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailOptions {
    /// Maximum accepted iterations.
    pub max_iterations: usize,
    /// Minimum accepted iterations before convergence.
    pub min_iterations: usize,
    /// Maximum relative integrated-intensity change for convergence.
    pub intensity_tolerance: f64,
    /// Absolute Rwp change for convergence.
    pub rwp_tolerance: f64,
    /// Multiplicative redistribution damping in `(0, 1]`.
    pub redistribution_damping: f64,
    /// Minimum calculated profile accepted in the observed/calculated ratio.
    pub minimum_calculated: f64,
    /// Positive starting and relative-change denominator floor.
    pub initial_intensity_floor: f64,
    /// Whether supplied one-sigma uncertainty is used.
    pub use_uncertainty: bool,
    /// Correlation threshold used by optional unresolved-group diagnostics.
    pub unresolved_correlation: f64,
    /// Whether coincident reflection rank diagnostics are calculated.
    pub diagnose_rank_deficiency: bool,
    /// Finite profile support in FWHM units.
    pub support_fwhm: f64,
    /// Persistent bounded native worker policy.
    pub execution: ExecutionPolicy,
}

impl LeBailOptions {
    /// Validate all convergence and execution controls.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError::InvalidOptions`] for an invalid control.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_iterations: usize,
        min_iterations: usize,
        intensity_tolerance: f64,
        rwp_tolerance: f64,
        redistribution_damping: f64,
        minimum_calculated: f64,
        initial_intensity_floor: f64,
        use_uncertainty: bool,
        unresolved_correlation: f64,
        diagnose_rank_deficiency: bool,
        support_fwhm: f64,
        execution: ExecutionPolicy,
    ) -> Result<Self, LeBailError> {
        let options = Self {
            max_iterations,
            min_iterations,
            intensity_tolerance,
            rwp_tolerance,
            redistribution_damping,
            minimum_calculated,
            initial_intensity_floor,
            use_uncertainty,
            unresolved_correlation,
            diagnose_rank_deficiency,
            support_fwhm,
            execution,
        };
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), LeBailError> {
        if self.max_iterations == 0
            || self.min_iterations == 0
            || self.min_iterations > self.max_iterations
        {
            return Err(invalid_options(
                "iteration counts must be positive and minimum must not exceed maximum",
            ));
        }
        for (name, value) in [
            ("intensity_tolerance", self.intensity_tolerance),
            ("rwp_tolerance", self.rwp_tolerance),
            ("minimum_calculated", self.minimum_calculated),
            ("initial_intensity_floor", self.initial_intensity_floor),
            ("support_fwhm", self.support_fwhm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(LeBailError::InvalidOptions {
                    message: format!("{name} must be positive and finite"),
                });
            }
        }
        if !self.redistribution_damping.is_finite()
            || self.redistribution_damping <= 0.0
            || self.redistribution_damping > 1.0
        {
            return Err(invalid_options("redistribution_damping must lie in (0, 1]"));
        }
        if !self.unresolved_correlation.is_finite()
            || !(0.0..=1.0).contains(&self.unresolved_correlation)
        {
            return Err(invalid_options("unresolved_correlation must lie in [0, 1]"));
        }
        Ok(())
    }

    /// Construct the scripting-compatible defaults with an explicit policy.
    ///
    /// # Errors
    ///
    /// Returns [`LeBailError`] if the controls cannot be constructed.
    pub fn scripting_defaults(execution: ExecutionPolicy) -> Result<Self, LeBailError> {
        Self::new(
            50,
            2,
            1.0e-6,
            1.0e-8,
            1.0,
            1.0e-15,
            1.0e-12,
            true,
            1.0 - 1.0e-10,
            false,
            20.0,
            execution,
        )
    }
}

/// One display-ready phase curve.
#[derive(Clone, Debug, PartialEq)]
pub struct PhasePatternComponent {
    /// Stable phase ID.
    pub phase_id: String,
    /// Sample-aligned phase contribution.
    pub y: Vec<f64>,
}

/// Native fixed-phase pattern result used by extraction and adapters.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailCalculation {
    /// Profile plus fixed supplied background.
    pub y: Vec<f64>,
    /// Sum of all phase profiles.
    pub profile_y: Vec<f64>,
    /// Fixed supplied background.
    pub background_y: Vec<f64>,
    /// Sparse local and dense global profile derivatives.
    pub accumulation: Accumulation,
    /// `(phase_id, reflection_id)` in local-Jacobian order.
    pub reflection_keys: Vec<(String, String)>,
    /// Prefix sum of phase reflection counts.
    pub phase_offsets: Vec<usize>,
    /// One diagnostic curve per phase.
    pub phase_components: Vec<PhasePatternComponent>,
}

/// One non-negative multiplicative redistribution result.
#[derive(Clone, Debug, PartialEq)]
pub struct IntensityExtractionResult {
    /// New integrated intensities in reflection order.
    pub intensities: Vec<f64>,
    /// Largest floored relative intensity change.
    pub maximum_relative_change: f64,
    /// Reflection keys without included finite support.
    pub unobserved_reflections: Vec<(String, String)>,
}

/// One immutable accepted fixed-reflection iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailIterationRecord {
    /// One-based attempted iteration.
    pub iteration: usize,
    /// Unweighted profile residual.
    pub rp: f64,
    /// Weighted profile residual.
    pub rwp: f64,
    /// Weighted residual sum of squares.
    pub chi_square: f64,
    /// Chi-square per positive residual degree of freedom.
    pub reduced_chi_square: f64,
    /// Largest relative integrated-intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Iteration warnings in deterministic order.
    pub warnings: Vec<String>,
}

/// Final stable reflection identity and intensity.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflectionIntensity {
    /// Stable phase ID.
    pub phase_id: String,
    /// Stable reflection ID.
    pub reflection_id: String,
    /// Non-negative integrated intensity.
    pub integrated_intensity: f64,
}

/// Numerically coincident profile columns and their matrix rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoincidentReflectionGroup {
    /// Stable reflection keys.
    pub reflection_keys: Vec<(String, String)>,
    /// Numerical rank of the joined support matrix.
    pub rank: usize,
}

/// Complete immutable continuation state for the fixed-reflection workflow.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailCheckpoint {
    /// Number of accepted iterations.
    pub completed_iterations: usize,
    /// Current phase records and integrated intensities.
    pub phases: Vec<LeBailPhase>,
    /// Flattened current integrated intensities.
    pub intensities: Vec<f64>,
    /// Rwp from the last non-converged accepted iteration.
    pub previous_rwp: f64,
    /// Complete accepted deterministic history.
    pub history: Vec<LeBailIterationRecord>,
}

impl LeBailCheckpoint {
    fn validate(&self) -> Result<(), LeBailError> {
        if self.completed_iterations != self.history.len() {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint iteration count must equal its history length".to_owned(),
            });
        }
        if self.previous_rwp.is_nan() || self.previous_rwp == f64::NEG_INFINITY {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint previous_rwp must be finite or positive infinity".to_owned(),
            });
        }
        for phase in &self.phases {
            phase.validate()?;
        }
        let expected = self.phases.iter().map(reflection_count).sum::<usize>();
        if self.intensities.len() != expected
            || self
                .intensities
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint intensities must match its phases".to_owned(),
            });
        }
        if flatten_intensities(&self.phases) != self.intensities {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint phase and flattened intensities disagree".to_owned(),
            });
        }
        if self
            .history
            .iter()
            .enumerate()
            .any(|(index, record)| record.iteration != index + 1)
        {
            return Err(LeBailError::InvalidCheckpoint {
                message: "checkpoint history iterations must be contiguous and one-based"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Complete native fixed-reflection Le Bail result.
#[derive(Clone, Debug, PartialEq)]
pub struct LeBailResult {
    /// Final calculated pattern and sparse derivative storage.
    pub calculation: LeBailCalculation,
    /// Final phases.
    pub phases: Vec<LeBailPhase>,
    /// Final labeled integrated intensities.
    pub intensities: Vec<ReflectionIntensity>,
    /// Final residual arrays and metrics.
    pub metrics: ResidualEvaluation,
    /// Accepted deterministic iteration history.
    pub history: Vec<LeBailIterationRecord>,
    /// Stable termination category.
    pub termination_reason: TerminationReason,
    /// Optional unresolved reflection diagnostics.
    pub rank_deficient_groups: Vec<CoincidentReflectionGroup>,
    /// Complete restart state.
    pub checkpoint: LeBailCheckpoint,
}

/// Calculate all fixed phases through one native fused accumulation.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid pattern/profile state or allocation.
pub fn calculate_lebail_pattern(
    pattern: &PatternRecord,
    instrument: ConstantWavelengthInstrument,
    phases: &[LeBailPhase],
    support_fwhm: f64,
    execution: &ExecutionPolicy,
) -> Result<LeBailCalculation, LeBailError> {
    pattern.validate().map_err(LeBailError::Pattern)?;
    if phases.is_empty() {
        return Err(invalid_phase("at least one phase is required"));
    }
    if !support_fwhm.is_finite() || support_fwhm <= 0.0 {
        return Err(invalid_options("support_fwhm must be positive and finite"));
    }
    let reflection_count = phases.iter().map(reflection_count).sum::<usize>();
    let mut positions = Vec::with_capacity(reflection_count);
    let mut intensities = Vec::with_capacity(reflection_count);
    let mut multipliers = Vec::with_capacity(reflection_count);
    let mut reflection_keys = Vec::with_capacity(reflection_count);
    let mut phase_offsets = Vec::with_capacity(phases.len() + 1);
    let phase_derivative_count = phases
        .len()
        .checked_mul(reflection_count)
        .ok_or(LeBailError::SizeOverflow)?;
    let mut derivative_multipliers = vec![0.0; phase_derivative_count];
    phase_offsets.push(0);
    for (phase_index, phase) in phases.iter().enumerate() {
        phase.validate()?;
        let begin = positions.len();
        positions.extend_from_slice(&phase.two_theta_deg);
        intensities.extend_from_slice(&phase.integrated_intensity);
        multipliers.extend(std::iter::repeat_n(phase.scale, reflection_count_of(phase)));
        reflection_keys.extend(
            phase
                .reflection_ids
                .iter()
                .map(|reflection_id| (phase.phase_id.clone(), reflection_id.clone())),
        );
        let end = positions.len();
        derivative_multipliers
            [phase_index * reflection_count + begin..phase_index * reflection_count + end]
            .fill(1.0);
        phase_offsets.push(end);
    }
    let contributions = OwnedCwContributions::new(
        reflection_count,
        phases.len(),
        OwnedCwContributionArrays {
            gaussian_variance_deg2: vec![0.0; reflection_count],
            lorentzian_fwhm_deg: vec![0.0; reflection_count],
            intensity_multiplier: multipliers,
            d_gaussian_variance_d_position: vec![0.0; reflection_count],
            d_lorentzian_fwhm_d_position: vec![0.0; reflection_count],
            d_intensity_multiplier_d_position: vec![0.0; reflection_count],
            d_gaussian_variance_d_parameters: vec![0.0; phase_derivative_count],
            d_lorentzian_fwhm_d_parameters: vec![0.0; phase_derivative_count],
            d_intensity_multiplier_d_parameters: derivative_multipliers,
        },
    )
    .map_err(LeBailError::Calculation)?;
    let grid = GridView::new(&pattern.x_deg).map_err(LeBailError::Grid)?;
    let accumulation = accumulate_cw_contributions_batch_with_context(
        grid,
        &positions,
        &intensities,
        instrument,
        contributions.as_view(),
        SupportPolicy::FwhmMultiple(support_fwhm),
        execution.context(),
    )
    .map_err(LeBailError::Calculation)?;
    let profile_y = accumulation.y.clone();
    let y = profile_y
        .iter()
        .zip(&pattern.background_y)
        .map(|(profile, background)| profile + background)
        .collect::<Vec<_>>();
    let mut phase_components = Vec::with_capacity(phases.len());
    for (phase_index, phase) in phases.iter().enumerate() {
        let mut phase_y = vec![0.0; pattern.sample_count()];
        let first = phase_offsets[phase_index];
        let last = phase_offsets[phase_index + 1];
        for (reflection, intensity) in intensities.iter().enumerate().take(last).skip(first) {
            let begin = accumulation.derivatives.local.offsets[reflection];
            let end = accumulation.derivatives.local.offsets[reflection + 1];
            let start = accumulation.derivatives.local.starts[reflection];
            for active in begin..end {
                let sample = start + active - begin;
                phase_y[sample] += intensity
                    * accumulation.derivatives.local.values
                        [active * accumulation.derivatives.local.parameter_count];
            }
        }
        phase_components.push(PhasePatternComponent {
            phase_id: phase.phase_id.clone(),
            y: phase_y,
        });
    }
    Ok(LeBailCalculation {
        y,
        profile_y,
        background_y: pattern.background_y.clone(),
        accumulation,
        reflection_keys,
        phase_offsets,
        phase_components,
    })
}

/// Return deterministic positive starting intensities in phase order.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input or option state.
pub fn initialize_lebail_intensities(
    input: &LeBailInput,
    options: &LeBailOptions,
) -> Result<Vec<f64>, LeBailError> {
    options.validate()?;
    let values = flatten_intensities(&input.phases);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid_phase(
            "starting intensities must be non-negative and finite",
        ));
    }
    if values.iter().any(|value| *value > 0.0) {
        return Ok(values
            .into_iter()
            .map(|value| value.max(options.initial_intensity_floor))
            .collect());
    }
    let observed = input
        .pattern
        .observed_y
        .as_deref()
        .ok_or(LeBailError::MissingObservations)?;
    let weights = bin_integration_weights(&input.pattern.x_deg);
    let area = observed
        .iter()
        .zip(&input.pattern.background_y)
        .zip(weights)
        .map(|((observed, background), width)| (observed - background).max(0.0) * width)
        .sum::<f64>();
    let starting = (area / count_as_f64(values.len().max(1))).max(options.initial_intensity_floor);
    Ok(vec![starting; values.len()])
}

/// Perform one non-negative multiplicative redistribution step.
///
/// # Errors
///
/// Returns [`LeBailError`] for shape, observation, or finite-state failures.
pub fn extract_lebail_intensities(
    pattern: &PatternRecord,
    calculation: &LeBailCalculation,
    current: &[f64],
    options: &LeBailOptions,
    preserve_unobserved: &[bool],
) -> Result<IntensityExtractionResult, LeBailError> {
    options.validate()?;
    let observed = pattern
        .observed_y
        .as_deref()
        .ok_or(LeBailError::MissingObservations)?;
    let reflection_count = calculation.accumulation.derivatives.local.peak_count();
    if current.len() != reflection_count
        || current
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(LeBailError::IntensityShapeMismatch);
    }
    if preserve_unobserved.len() != reflection_count {
        return Err(LeBailError::PreserveMaskLengthMismatch);
    }
    let included = pattern
        .mask
        .clone()
        .unwrap_or_else(|| vec![true; pattern.sample_count()]);
    let ratio = observed
        .iter()
        .zip(&pattern.background_y)
        .zip(&calculation.profile_y)
        .zip(&included)
        .map(|(((observed, background), calculated), included)| {
            if *included && *calculated > options.minimum_calculated {
                (observed - background).max(0.0) / calculated
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut weights = bin_integration_weights(&pattern.x_deg);
    if options.use_uncertainty
        && let Some(uncertainty) = &pattern.uncertainty
    {
        for (weight, uncertainty) in weights.iter_mut().zip(uncertainty) {
            *weight /= uncertainty * uncertainty;
        }
    }
    for (weight, included) in weights.iter_mut().zip(&included) {
        if !included {
            *weight = 0.0;
        }
    }
    let local = &calculation.accumulation.derivatives.local;
    let mut updated = vec![0.0; reflection_count];
    let mut unobserved_reflections = Vec::new();
    for reflection in 0..reflection_count {
        let begin = local.offsets[reflection];
        let end = local.offsets[reflection + 1];
        let start = local.starts[reflection];
        let mut denominator = 0.0;
        let mut numerator = 0.0;
        for active in begin..end {
            let sample = start + active - begin;
            let profile = local.values[active * local.parameter_count];
            let weighted_profile = weights[sample] * profile;
            denominator += weighted_profile;
            numerator += weighted_profile * ratio[sample];
        }
        if denominator <= 0.0 {
            unobserved_reflections.push(calculation.reflection_keys[reflection].clone());
            if preserve_unobserved[reflection] {
                updated[reflection] = current[reflection];
            }
            continue;
        }
        let raw = (current[reflection] * numerator / denominator).max(0.0);
        updated[reflection] =
            current[reflection] + options.redistribution_damping * (raw - current[reflection]);
    }
    let maximum_relative_change = updated
        .iter()
        .zip(current)
        .map(|(updated, current)| {
            (updated - current).abs() / current.abs().max(options.initial_intensity_floor)
        })
        .fold(0.0_f64, f64::max);
    Ok(IntensityExtractionResult {
        intensities: updated,
        maximum_relative_change,
        unobserved_reflections,
    })
}

/// Run fixed-reflection Le Bail extraction with a workflow-owned runtime.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid state, runtime construction, profile
/// evaluation, metrics, or checkpoint delivery.
pub fn refine_lebail(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let max_evaluations = options
        .max_iterations
        .checked_add(1)
        .ok_or(LeBailError::SizeOverflow)?;
    let limits = RefinementLimits::new(options.max_iterations, max_evaluations, None, 1)
        .map_err(LeBailError::Runtime)?;
    let mut runtime = RefinementRuntime::new(limits, None).map_err(LeBailError::Runtime)?;
    refine_lebail_with_runtime(input, options, checkpoint, &mut runtime)
}

/// Advance exactly one accepted Le Bail iteration for custom orchestration.
///
/// Pass the returned checkpoint to the next call. A cancellation-aware host
/// that needs to stop before the iteration should use
/// [`refine_lebail_with_runtime`] with a one-iteration runtime budget.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input, options, checkpoint, or numerical
/// evaluation state.
pub fn iterate_lebail_once(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let completed = checkpoint.map_or(0, |value| value.completed_iterations);
    let iteration = completed.checked_add(1).ok_or(LeBailError::SizeOverflow)?;
    let mut selected = options.clone();
    selected.min_iterations = iteration;
    selected.max_iterations = iteration;
    selected.validate()?;
    refine_lebail(input, &selected, checkpoint)
}

/// Run fixed-reflection extraction with host-owned cancellation/events/checkpoints.
///
/// The runtime should be fresh for a new run. A continuation restores its
/// accepted counter from the supplied checkpoint before numerical work begins.
///
/// # Errors
///
/// Returns [`LeBailError`] for invalid input/checkpoint state, non-normal
/// runtime failures, or numerical evaluation failures.
pub fn refine_lebail_with_runtime(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    options.validate()?;
    let mut state = restore_state(input, options, checkpoint)?;
    if let Some(checkpoint) = checkpoint {
        runtime
            .resume_accepted(checkpoint.completed_iterations)
            .map_err(LeBailError::Runtime)?;
    }
    runtime
        .emit(
            RefinementEventKind::Start,
            "lebail",
            "Le Bail extraction started",
            Vec::new(),
        )
        .map_err(LeBailError::Runtime)?;
    state.calculation = Some(calculate_lebail_pattern(
        &input.pattern,
        input.instrument,
        &state.phases,
        options.support_fwhm,
        &options.execution,
    )?);
    let termination = run_lebail_iterations(input, options, &mut state, runtime)?;
    finish_result(
        input,
        options,
        state.phases,
        &state.intensities,
        state.history,
        state.previous_rwp,
        state.calculation.ok_or(LeBailError::InternalInvariant)?,
        termination,
        runtime,
    )
}

fn run_lebail_iterations(
    input: &LeBailInput,
    options: &LeBailOptions,
    state: &mut RestoredLeBailState,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<TerminationReason, LeBailError> {
    if let Err(error) = runtime.begin_evaluation() {
        return stop_reason_or_error(error);
    }
    for iteration in state.first_iteration..=options.max_iterations {
        if let Err(error) = runtime.begin_iteration(iteration) {
            return stop_reason_or_error(error);
        }
        if let Err(error) = runtime.begin_evaluation() {
            return stop_reason_or_error(error);
        }
        let candidate = evaluate_lebail_iteration(input, options, state)?;
        state.history.push(LeBailIterationRecord {
            iteration,
            rp: candidate.metrics.rp,
            rwp: candidate.metrics.rwp,
            chi_square: candidate.metrics.chi_square,
            reduced_chi_square: candidate.metrics.reduced_chi_square,
            maximum_relative_intensity_change: candidate.extraction.maximum_relative_change,
            warnings: candidate.warnings,
        });
        state.phases = candidate.phases;
        state.intensities = candidate.extraction.intensities;
        state.calculation = Some(candidate.calculation);
        accept_lebail_iteration(runtime, state, &candidate.metrics)?;
        if iteration >= options.min_iterations
            && candidate.extraction.maximum_relative_change < options.intensity_tolerance
            && (state.previous_rwp - candidate.metrics.rwp).abs() < options.rwp_tolerance
        {
            return Ok(TerminationReason::Converged);
        }
        state.previous_rwp = candidate.metrics.rwp;
    }
    Ok(TerminationReason::MaxIterations)
}

struct EvaluatedLeBailIteration {
    extraction: IntensityExtractionResult,
    phases: Vec<LeBailPhase>,
    calculation: LeBailCalculation,
    metrics: ResidualEvaluation,
    warnings: Vec<String>,
}

fn evaluate_lebail_iteration(
    input: &LeBailInput,
    options: &LeBailOptions,
    state: &RestoredLeBailState,
) -> Result<EvaluatedLeBailIteration, LeBailError> {
    let extraction = extract_lebail_intensities(
        &input.pattern,
        state.calculation()?,
        &state.intensities,
        options,
        &flatten_preserve_mask(&state.phases),
    )?;
    let phases = replace_flat_intensities(&state.phases, &extraction.intensities)?;
    let calculation = calculate_lebail_pattern(
        &input.pattern,
        input.instrument,
        &phases,
        options.support_fwhm,
        &options.execution,
    )?;
    let metrics = evaluate_residuals(
        &input.pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: 0,
        },
    )
    .map_err(LeBailError::Residual)?;
    let warnings = if extraction.unobserved_reflections.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} reflections have no included support",
            extraction.unobserved_reflections.len()
        )]
    };
    Ok(EvaluatedLeBailIteration {
        extraction,
        phases,
        calculation,
        metrics,
        warnings,
    })
}

fn accept_lebail_iteration(
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
    state: &RestoredLeBailState,
    metrics: &ResidualEvaluation,
) -> Result<(), LeBailError> {
    let checkpoint = LeBailCheckpoint {
        completed_iterations: state.history.len(),
        phases: state.phases.clone(),
        intensities: state.intensities.clone(),
        previous_rwp: metrics.rwp,
        history: state.history.clone(),
    };
    runtime
        .accept_step(Some(&checkpoint))
        .map_err(LeBailError::Runtime)?;
    runtime
        .emit(
            RefinementEventKind::Iteration,
            "lebail_iteration",
            "Le Bail iteration accepted",
            vec![
                ("rwp".to_owned(), DiagnosticValue::Float(metrics.rwp)),
                (
                    "maximum_relative_intensity_change".to_owned(),
                    DiagnosticValue::Float(
                        state
                            .history
                            .last()
                            .ok_or(LeBailError::InternalInvariant)?
                            .maximum_relative_intensity_change,
                    ),
                ),
            ],
        )
        .map_err(LeBailError::Runtime)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_result(
    input: &LeBailInput,
    options: &LeBailOptions,
    phases: Vec<LeBailPhase>,
    intensities: &[f64],
    history: Vec<LeBailIterationRecord>,
    previous_rwp: f64,
    calculation: LeBailCalculation,
    termination: TerminationReason,
    runtime: &mut RefinementRuntime<LeBailCheckpoint>,
) -> Result<LeBailResult, LeBailError> {
    let metrics = evaluate_residuals(
        &input.pattern,
        &calculation.y,
        ResidualOptions {
            use_uncertainty: options.use_uncertainty,
            parameter_count: 0,
        },
    )
    .map_err(LeBailError::Residual)?;
    let checkpoint = LeBailCheckpoint {
        completed_iterations: history.len(),
        phases: phases.clone(),
        intensities: intensities.to_owned(),
        previous_rwp: if termination == TerminationReason::Cancelled {
            previous_rwp
        } else {
            metrics.rwp
        },
        history: history.clone(),
    };
    checkpoint.validate()?;
    let labeled = calculation
        .reflection_keys
        .iter()
        .zip(intensities)
        .map(
            |((phase_id, reflection_id), intensity)| ReflectionIntensity {
                phase_id: phase_id.clone(),
                reflection_id: reflection_id.clone(),
                integrated_intensity: *intensity,
            },
        )
        .collect();
    let rank_deficient_groups = if options.diagnose_rank_deficiency {
        rank_deficient_groups(&calculation, options.unresolved_correlation)
    } else {
        Vec::new()
    };
    runtime
        .emit(
            RefinementEventKind::Termination,
            "lebail",
            "Le Bail extraction terminated",
            vec![(
                "termination_reason".to_owned(),
                DiagnosticValue::String(termination.as_str().to_owned()),
            )],
        )
        .map_err(LeBailError::Runtime)?;
    Ok(LeBailResult {
        calculation,
        phases,
        intensities: labeled,
        metrics,
        history,
        termination_reason: termination,
        rank_deficient_groups,
        checkpoint,
    })
}

struct RestoredLeBailState {
    phases: Vec<LeBailPhase>,
    intensities: Vec<f64>,
    history: Vec<LeBailIterationRecord>,
    previous_rwp: f64,
    first_iteration: usize,
    calculation: Option<LeBailCalculation>,
}

impl RestoredLeBailState {
    fn calculation(&self) -> Result<&LeBailCalculation, LeBailError> {
        self.calculation
            .as_ref()
            .ok_or(LeBailError::InternalInvariant)
    }
}

fn restore_state(
    input: &LeBailInput,
    options: &LeBailOptions,
    checkpoint: Option<&LeBailCheckpoint>,
) -> Result<RestoredLeBailState, LeBailError> {
    let Some(checkpoint) = checkpoint else {
        let intensities = initialize_lebail_intensities(input, options)?;
        let phases = replace_flat_intensities(&input.phases, &intensities)?;
        return Ok(RestoredLeBailState {
            phases,
            intensities,
            history: Vec::new(),
            previous_rwp: f64::INFINITY,
            first_iteration: 1,
            calculation: None,
        });
    };
    checkpoint.validate()?;
    if checkpoint.completed_iterations >= options.max_iterations {
        return Err(LeBailError::InvalidCheckpoint {
            message: "checkpoint already reached the configured maximum iteration".to_owned(),
        });
    }
    let input_identity = phase_identity(&input.phases);
    let checkpoint_identity = phase_identity(&checkpoint.phases);
    if input_identity != checkpoint_identity {
        return Err(LeBailError::InvalidCheckpoint {
            message: "checkpoint phase/reflection identities do not match the input".to_owned(),
        });
    }
    Ok(RestoredLeBailState {
        phases: checkpoint.phases.clone(),
        intensities: checkpoint.intensities.clone(),
        history: checkpoint.history.clone(),
        previous_rwp: checkpoint.previous_rwp,
        first_iteration: checkpoint.completed_iterations + 1,
        calculation: None,
    })
}

fn rank_deficient_groups(
    calculation: &LeBailCalculation,
    threshold: f64,
) -> Vec<CoincidentReflectionGroup> {
    let local = &calculation.accumulation.derivatives.local;
    let count = local.peak_count();
    let mut parents = (0..count).collect::<Vec<_>>();
    let norms = (0..count)
        .map(|reflection| {
            let begin = local.offsets[reflection];
            let end = local.offsets[reflection + 1];
            (begin..end)
                .map(|active| {
                    let value = local.values[active * local.parameter_count];
                    value * value
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    for left in 0..count {
        let left_begin = local.offsets[left];
        let left_end = local.offsets[left + 1];
        let left_start = local.starts[left];
        let left_stop = left_start + left_end - left_begin;
        for right in left + 1..count {
            let right_begin = local.offsets[right];
            let right_end = local.offsets[right + 1];
            let right_start = local.starts[right];
            let right_stop = right_start + right_end - right_begin;
            let start = left_start.max(right_start);
            let stop = left_stop.min(right_stop);
            if start >= stop || norms[left] == 0.0 || norms[right] == 0.0 {
                continue;
            }
            let correlation = (start..stop)
                .map(|sample| {
                    let left_active = left_begin + sample - left_start;
                    let right_active = right_begin + sample - right_start;
                    local.values[left_active * local.parameter_count]
                        * local.values[right_active * local.parameter_count]
                })
                .sum::<f64>()
                / (norms[left] * norms[right]);
            if correlation >= threshold {
                union(&mut parents, left, right);
            }
        }
    }
    let mut grouped = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for reflection in 0..count {
        let root = root(&mut parents, reflection);
        grouped.entry(root).or_default().push(reflection);
    }
    grouped
        .into_values()
        .filter(|indices| indices.len() > 1)
        .map(|indices| {
            let first = indices
                .iter()
                .map(|index| local.starts[*index])
                .min()
                .unwrap_or(0);
            let last = indices
                .iter()
                .map(|index| {
                    local.starts[*index] + local.offsets[*index + 1] - local.offsets[*index]
                })
                .max()
                .unwrap_or(first);
            let mut matrix = DMatrix::zeros(last - first, indices.len());
            for (column, reflection) in indices.iter().enumerate() {
                let begin = local.offsets[*reflection];
                let end = local.offsets[*reflection + 1];
                let start = local.starts[*reflection] - first;
                for active in begin..end {
                    matrix[(start + active - begin, column)] =
                        local.values[active * local.parameter_count];
                }
            }
            let singular_values = matrix.svd(false, false).singular_values;
            let maximum = singular_values.iter().copied().fold(0.0_f64, f64::max);
            let tolerance =
                count_as_f64((last - first).max(indices.len())) * f64::EPSILON * maximum;
            let rank = singular_values
                .iter()
                .filter(|value| **value > tolerance)
                .count();
            CoincidentReflectionGroup {
                reflection_keys: indices
                    .iter()
                    .map(|index| calculation.reflection_keys[*index].clone())
                    .collect(),
                rank,
            }
        })
        .collect()
}

fn root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left_root = root(parents, left);
    let right_root = root(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

fn bin_integration_weights(x: &[f64]) -> Vec<f64> {
    match x.len() {
        0 => Vec::new(),
        1 => vec![1.0],
        count => {
            let mut widths = vec![0.0; count];
            widths[0] = 0.5 * (x[1] - x[0]);
            widths[count - 1] = 0.5 * (x[count - 1] - x[count - 2]);
            for index in 1..count - 1 {
                widths[index] = 0.5 * (x[index + 1] - x[index - 1]);
            }
            widths
        }
    }
}

fn replace_flat_intensities(
    phases: &[LeBailPhase],
    intensities: &[f64],
) -> Result<Vec<LeBailPhase>, LeBailError> {
    let expected = phases.iter().map(reflection_count).sum::<usize>();
    if intensities.len() != expected {
        return Err(LeBailError::IntensityShapeMismatch);
    }
    let mut offset = 0;
    phases
        .iter()
        .map(|phase| {
            let end = offset + reflection_count(phase);
            let updated = phase.replace_intensities(&intensities[offset..end]);
            offset = end;
            updated
        })
        .collect()
}

fn flatten_intensities(phases: &[LeBailPhase]) -> Vec<f64> {
    phases
        .iter()
        .flat_map(|phase| phase.integrated_intensity.iter().copied())
        .collect()
}

fn flatten_preserve_mask(phases: &[LeBailPhase]) -> Vec<bool> {
    phases
        .iter()
        .flat_map(|phase| {
            if phase.preserve_unobserved.is_empty() {
                vec![false; reflection_count(phase)]
            } else {
                phase.preserve_unobserved.clone()
            }
        })
        .collect()
}

fn phase_identity(phases: &[LeBailPhase]) -> Vec<(&str, Vec<&str>)> {
    phases
        .iter()
        .map(|phase| {
            (
                phase.phase_id(),
                phase.reflection_ids().iter().map(String::as_str).collect(),
            )
        })
        .collect()
}

fn reflection_count(phase: &LeBailPhase) -> usize {
    phase.reflection_ids.len()
}

fn reflection_count_of(phase: &LeBailPhase) -> usize {
    reflection_count(phase)
}

fn validate_stable_label(name: &'static str, value: &str) -> Result<(), LeBailError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(LeBailError::InvalidPhase {
            message: format!(
                "{name} must be non-empty, trimmed, and contain no control characters"
            ),
        });
    }
    Ok(())
}

fn normal_stop_reason(error: &RuntimeError) -> Option<TerminationReason> {
    match error {
        RuntimeError::Stopped(stop) => Some(stop.reason),
        _ => None,
    }
}

fn stop_reason_or_error(error: RuntimeError) -> Result<TerminationReason, LeBailError> {
    normal_stop_reason(&error).ok_or(LeBailError::Runtime(error))
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    value as f64
}

fn invalid_phase(message: &str) -> LeBailError {
    LeBailError::InvalidPhase {
        message: message.to_owned(),
    }
}

fn invalid_options(message: &str) -> LeBailError {
    LeBailError::InvalidOptions {
        message: message.to_owned(),
    }
}

/// Invalid native fixed-reflection Le Bail state or operation.
#[derive(Debug)]
pub enum LeBailError {
    /// Pattern domain state is invalid.
    Pattern(DomainError),
    /// Observations are required.
    MissingObservations,
    /// One phase or reflection record is invalid.
    InvalidPhase {
        /// Stable diagnostic message.
        message: String,
    },
    /// One option is invalid.
    InvalidOptions {
        /// Stable diagnostic message.
        message: String,
    },
    /// A checkpoint cannot continue this request.
    InvalidCheckpoint {
        /// Stable diagnostic message.
        message: String,
    },
    /// Current intensities do not match the calculated reflection order.
    IntensityShapeMismatch,
    /// Preserve-if-unobserved mask does not match the reflection count.
    PreserveMaskLengthMismatch,
    /// Pattern grid validation failed.
    Grid(ProfileError),
    /// Native CW accumulation failed.
    Calculation(CwContributionsError),
    /// Residual evaluation failed.
    Residual(ResidualError),
    /// Bounded runtime or host callback failed.
    Runtime(RuntimeError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
    /// A workflow-owned allocation/budget count overflowed.
    SizeOverflow,
    /// A lower-level profile/instrument validation failed.
    Profile {
        /// Stable diagnostic message.
        message: String,
    },
    /// Private workflow state became inconsistent.
    InternalInvariant,
}

impl Display for LeBailError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("observed_y is required for Le Bail extraction")
            }
            Self::InvalidPhase { message }
            | Self::InvalidOptions { message }
            | Self::InvalidCheckpoint { message }
            | Self::Profile { message } => formatter.write_str(message),
            Self::IntensityShapeMismatch => {
                formatter.write_str("current intensities must match the reflection count")
            }
            Self::PreserveMaskLengthMismatch => {
                formatter.write_str("preserve_unobserved must match the reflection count")
            }
            Self::Grid(error) => Display::fmt(error, formatter),
            Self::Calculation(error) => Display::fmt(error, formatter),
            Self::Residual(error) => Display::fmt(error, formatter),
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
            Self::SizeOverflow => formatter.write_str("Le Bail workflow size or budget overflowed"),
            Self::InternalInvariant => {
                formatter.write_str("internal Le Bail workflow state is inconsistent")
            }
        }
    }
}

impl Error for LeBailError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::Grid(error) => Some(error),
            Self::Calculation(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::MissingObservations
            | Self::InvalidPhase { .. }
            | Self::InvalidOptions { .. }
            | Self::InvalidCheckpoint { .. }
            | Self::IntensityShapeMismatch
            | Self::PreserveMaskLengthMismatch
            | Self::SizeOverflow
            | Self::Profile { .. }
            | Self::InternalInvariant => None,
        }
    }
}
