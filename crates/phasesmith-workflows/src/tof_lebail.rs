//! Native fixed-instrument time-of-flight Le Bail extraction.
//!
//! TOF coordinates remain in microseconds and reflections remain parameterized
//! by d-spacing. Profile values, intensity/d-spacing derivatives, and all 15
//! shared instrument derivatives are produced by the fused core accumulation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{
    Accumulation, GridView, TofError, TofInstrument, accumulate_tof_batch_with_context,
};
use phasesmith_execution::{ExecutionPolicy, ExecutionPolicyError};
use phasesmith_model::{DomainError, RecordId, TofPatternRecord};

use crate::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_tof_residuals};

/// One fixed-topology phase in a TOF Le Bail extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailPhase {
    phase_id: RecordId,
    name: String,
    reflection_ids: Vec<String>,
    hkl: Vec<[i32; 3]>,
    d_spacing_angstrom: Vec<f64>,
    integrated_intensity: Vec<f64>,
    scale: f64,
}

impl TofLeBailPhase {
    /// Validate and own one fixed TOF reflection batch.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for empty, mismatched, duplicated, or
    /// nonphysical phase data.
    pub fn new(
        phase_id: RecordId,
        name: impl Into<String>,
        reflection_ids: Vec<String>,
        hkl: Vec<[i32; 3]>,
        d_spacing_angstrom: Vec<f64>,
        integrated_intensity: Vec<f64>,
        scale: f64,
    ) -> Result<Self, TofLeBailError> {
        let result = Self {
            phase_id,
            name: name.into(),
            reflection_ids,
            hkl,
            d_spacing_angstrom,
            integrated_intensity,
            scale,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate caller-mutated or adapter-decoded phase state.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] when a phase invariant is violated.
    pub fn validate(&self) -> Result<(), TofLeBailError> {
        let count = self.reflection_ids.len();
        if self.name.trim().is_empty() || count == 0 {
            return Err(TofLeBailError::InvalidPhase(
                "TOF Le Bail phases require a name and at least one reflection",
            ));
        }
        if self.hkl.len() != count
            || self.d_spacing_angstrom.len() != count
            || self.integrated_intensity.len() != count
        {
            return Err(TofLeBailError::InvalidPhase(
                "TOF phase reflection arrays must have equal length",
            ));
        }
        if self.reflection_ids.iter().any(String::is_empty)
            || self
                .reflection_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != count
        {
            return Err(TofLeBailError::InvalidPhase(
                "TOF reflection IDs must be non-empty and unique within a phase",
            ));
        }
        if self
            .d_spacing_angstrom
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .integrated_intensity
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(TofLeBailError::InvalidPhase(
                "TOF d-spacings and scale must be positive; intensities must be nonnegative",
            ));
        }
        Ok(())
    }

    /// Stable phase identifier.
    #[must_use]
    pub const fn phase_id(&self) -> &RecordId {
        &self.phase_id
    }

    /// Human-readable phase name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stable reflection identifiers.
    #[must_use]
    pub fn reflection_ids(&self) -> &[String] {
        &self.reflection_ids
    }

    /// Miller indices in reflection order.
    #[must_use]
    pub fn hkl(&self) -> &[[i32; 3]] {
        &self.hkl
    }

    /// Durable local reflection coordinates in ångströms.
    #[must_use]
    pub fn d_spacing_angstrom(&self) -> &[f64] {
        &self.d_spacing_angstrom
    }

    /// Current nonnegative integrated intensities.
    #[must_use]
    pub fn integrated_intensity(&self) -> &[f64] {
        &self.integrated_intensity
    }

    /// Phase scale applied before accumulation.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    fn with_intensities(&self, values: &[f64]) -> Result<Self, TofLeBailError> {
        if values.len() != self.integrated_intensity.len() {
            return Err(TofLeBailError::IntensityLengthMismatch);
        }
        let mut result = self.clone();
        result.integrated_intensity.copy_from_slice(values);
        result.validate()?;
        Ok(result)
    }
}

/// Refinable Chebyshev series on one explicit TOF interval.
///
/// For `u = 2 (tof - lower) / (upper - lower) - 1`, the background is
/// `sum_k coefficient[k] T_k(u)`. Coefficient derivatives are the corresponding
/// Chebyshev basis values and are independent of the coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct TofChebyshevBackground {
    background_id: RecordId,
    coefficients: Vec<f64>,
    domain_us: [f64; 2],
}

impl TofChebyshevBackground {
    /// Construct a non-empty finite Chebyshev series on an increasing interval.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for invalid identity, coefficients, or domain.
    pub fn new(
        background_id: RecordId,
        coefficients: Vec<f64>,
        domain_us: [f64; 2],
    ) -> Result<Self, TofLeBailError> {
        let result = Self {
            background_id,
            coefficients,
            domain_us,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate the model and its explicit microsecond domain.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for empty/non-finite coefficients or an invalid domain.
    pub fn validate(&self) -> Result<(), TofLeBailError> {
        if self.coefficients.is_empty()
            || self.coefficients.iter().any(|value| !value.is_finite())
            || self.domain_us.iter().any(|value| !value.is_finite())
            || self.domain_us[0] >= self.domain_us[1]
        {
            return Err(TofLeBailError::InvalidBackground);
        }
        Ok(())
    }

    /// Stable background identifier.
    #[must_use]
    pub const fn background_id(&self) -> &RecordId {
        &self.background_id
    }

    /// Current coefficients in increasing Chebyshev order.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Explicit closed TOF domain in microseconds.
    #[must_use]
    pub const fn domain_us(&self) -> [f64; 2] {
        self.domain_us
    }

    /// Evaluate the series on a sorted finite microsecond grid.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] if the grid is invalid or outside the domain.
    pub fn calculate(&self, tof_us: &[f64]) -> Result<Vec<f64>, TofLeBailError> {
        let basis = self.basis(tof_us)?;
        self.calculate_from_basis(&basis)
    }

    fn calculate_from_basis(&self, basis: &TofChebyshevBasis) -> Result<Vec<f64>, TofLeBailError> {
        let expected = basis
            .rows
            .checked_mul(basis.columns)
            .ok_or(TofLeBailError::AllocationOverflow)?;
        if basis.columns != self.coefficients.len() || basis.values.len() != expected {
            return Err(TofLeBailError::BackgroundBasisShape);
        }
        Ok(basis
            .values
            .chunks_exact(basis.columns)
            .map(|row| {
                row.iter()
                    .zip(&self.coefficients)
                    .map(|(basis, coefficient)| basis * coefficient)
                    .sum()
            })
            .collect())
    }

    fn basis(&self, tof_us: &[f64]) -> Result<TofChebyshevBasis, TofLeBailError> {
        self.validate()?;
        if tof_us.is_empty()
            || tof_us.iter().any(|value| !value.is_finite())
            || tof_us.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(TofLeBailError::InvalidBackgroundGrid);
        }
        let lower = self.domain_us[0];
        let upper = self.domain_us[1];
        let tolerance = 64.0 * f64::EPSILON * lower.abs().max(upper.abs()).max(1.0);
        if tof_us
            .iter()
            .any(|value| *value < lower - tolerance || *value > upper + tolerance)
        {
            return Err(TofLeBailError::BackgroundGridOutsideDomain);
        }
        let columns = self.coefficients.len();
        let count = tof_us
            .len()
            .checked_mul(columns)
            .ok_or(TofLeBailError::AllocationOverflow)?;
        let mut values = vec![0.0; count];
        for (tof, row) in tof_us.iter().zip(values.chunks_exact_mut(columns)) {
            let normalized = 2.0 * (tof - lower) / (upper - lower) - 1.0;
            row[0] = 1.0;
            if columns > 1 {
                row[1] = normalized;
            }
            for order in 2..columns {
                row[order] = 2.0 * normalized * row[order - 1] - row[order - 2];
            }
        }
        Ok(TofChebyshevBasis {
            rows: tof_us.len(),
            columns,
            values,
        })
    }

    fn with_coefficients(&self, coefficients: Vec<f64>) -> Result<Self, TofLeBailError> {
        if coefficients.len() != self.coefficients.len() {
            return Err(TofLeBailError::BackgroundCoefficientLengthMismatch);
        }
        Self::new(self.background_id.clone(), coefficients, self.domain_us)
    }
}

/// Sample-major analytical Chebyshev coefficient derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct TofChebyshevBasis {
    /// Number of TOF samples.
    pub rows: usize,
    /// Number of coefficients.
    pub columns: usize,
    /// Sample-major basis values.
    pub values: Vec<f64>,
}

/// Observations, fixed TOF instrument, and ordered phases.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailInput {
    /// Explicit microsecond-domain observed pattern.
    pub pattern: TofPatternRecord,
    /// Fixed 15-coefficient TOF profile model.
    pub instrument: TofInstrument,
    /// Ordered non-empty phase list.
    pub phases: Vec<TofLeBailPhase>,
    /// Optional refinable Chebyshev background; otherwise the pattern background is fixed.
    pub background: Option<TofChebyshevBackground>,
}

impl TofLeBailInput {
    /// Validate a complete TOF extraction request.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for invalid pattern, instrument, phase, or
    /// identity state.
    pub fn new(
        pattern: TofPatternRecord,
        instrument: TofInstrument,
        phases: Vec<TofLeBailPhase>,
    ) -> Result<Self, TofLeBailError> {
        let result = Self {
            pattern,
            instrument,
            phases,
            background: None,
        };
        result.validate()?;
        Ok(result)
    }

    /// Attach a refinable Chebyshev residual on top of the fixed pattern background.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] if the model or its grid/domain relationship is invalid.
    pub fn with_refinable_background(
        mut self,
        background: TofChebyshevBackground,
    ) -> Result<Self, TofLeBailError> {
        self.background = Some(background);
        self.validate()?;
        Ok(self)
    }

    /// Revalidate all trust-boundary state.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] when any request invariant is violated.
    pub fn validate(&self) -> Result<(), TofLeBailError> {
        self.pattern.validate().map_err(TofLeBailError::Pattern)?;
        if self.pattern.observed_y.is_none() {
            return Err(TofLeBailError::MissingObservations);
        }
        self.instrument
            .validate()
            .map_err(TofLeBailError::Profile)?;
        if self.phases.is_empty() {
            return Err(TofLeBailError::InvalidPhase(
                "TOF Le Bail input requires at least one phase",
            ));
        }
        let mut phase_ids = std::collections::BTreeSet::new();
        for phase in &self.phases {
            phase.validate()?;
            if !phase_ids.insert(phase.phase_id()) {
                return Err(TofLeBailError::InvalidPhase(
                    "TOF Le Bail phase IDs must be unique",
                ));
            }
        }
        if let Some(background) = &self.background {
            background.validate()?;
            background.basis(&self.pattern.tof_us)?;
        }
        Ok(())
    }
}

/// Deterministic fixed-instrument TOF extraction controls.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailOptions {
    /// Number of nonnegative redistribution cycles.
    pub cycles: usize,
    /// Multiplicative update damping in `(0, 1]`.
    pub redistribution_damping: f64,
    /// Positive floor used only to initialize an all-zero reflection list.
    pub initial_intensity_floor: f64,
    /// Minimum calculated profile accepted in an observed/calculated ratio.
    pub minimum_calculated: f64,
    /// Symmetric TCH support radius in total-FWHM units.
    pub support_fwhm: f64,
    /// Exponential truncation exponent; each tail ends at `exp(-tail_log)`.
    pub tail_log: f64,
    /// Use supplied one-sigma uncertainties in background fitting and metrics.
    pub use_uncertainty: bool,
    /// Use supplied one-sigma uncertainties in multiplicative redistribution.
    ///
    /// This defaults to `use_uncertainty` but can be disabled independently
    /// for GSAS-compatible unweighted Le Bail partitioning while retaining
    /// uncertainty-weighted background fitting and residual metrics.
    pub redistribution_use_uncertainty: bool,
    /// Bounded execution policy passed to the fused core kernel.
    pub execution: ExecutionPolicy,
}

impl TofLeBailOptions {
    /// Construct validated extraction controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for invalid numerical controls.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycles: usize,
        redistribution_damping: f64,
        initial_intensity_floor: f64,
        minimum_calculated: f64,
        support_fwhm: f64,
        tail_log: f64,
        use_uncertainty: bool,
        execution: ExecutionPolicy,
    ) -> Result<Self, TofLeBailError> {
        let result = Self {
            cycles,
            redistribution_damping,
            initial_intensity_floor,
            minimum_calculated,
            support_fwhm,
            tail_log,
            use_uncertainty,
            redistribution_use_uncertainty: use_uncertainty,
            execution,
        };
        result.validate()?;
        Ok(result)
    }

    /// Select uncertainty weighting independently for intensity redistribution.
    #[must_use]
    pub fn with_redistribution_uncertainty(mut self, enabled: bool) -> Self {
        self.redistribution_use_uncertainty = enabled;
        self
    }

    /// Construct practical deterministic defaults.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] if the defaults cannot be validated.
    pub fn scripting_defaults(execution: ExecutionPolicy) -> Result<Self, TofLeBailError> {
        Self::new(50, 1.0, 1.0e-12, 1.0e-15, 20.0, 20.0, true, execution)
    }

    /// Revalidate numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofLeBailError`] for zero, non-finite, or out-of-range values.
    pub fn validate(&self) -> Result<(), TofLeBailError> {
        if self.cycles == 0
            || !self.redistribution_damping.is_finite()
            || self.redistribution_damping <= 0.0
            || self.redistribution_damping > 1.0
            || !self.initial_intensity_floor.is_finite()
            || self.initial_intensity_floor <= 0.0
            || !self.minimum_calculated.is_finite()
            || self.minimum_calculated <= 0.0
            || !self.support_fwhm.is_finite()
            || self.support_fwhm <= 0.0
            || !self.tail_log.is_finite()
            || self.tail_log <= 0.0
        {
            return Err(TofLeBailError::InvalidOptions);
        }
        Ok(())
    }
}

/// Display-ready TOF profile plus the fused derivative product.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailCalculation {
    /// Profile plus fixed or refinable background.
    pub y: Vec<f64>,
    /// Sum of all reflection profiles.
    pub profile_y: Vec<f64>,
    /// Evaluated fixed or refinable background.
    pub background_y: Vec<f64>,
    /// Sample-major analytical coefficient derivatives for a refinable background.
    pub background_basis: Option<TofChebyshevBasis>,
    /// Fused intensity/d-spacing and 15-row instrument derivatives.
    pub accumulation: Accumulation,
    /// `(phase_id, reflection_id)` in local derivative order.
    pub reflection_keys: Vec<(String, String)>,
    /// Prefix sum of phase reflection counts.
    pub phase_offsets: Vec<usize>,
}

/// One accepted extraction cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailIterationRecord {
    /// One-based cycle index.
    pub iteration: usize,
    /// Residual metrics after redistribution.
    pub metrics: ResidualEvaluation,
    /// Largest floored relative intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Largest absolute Chebyshev coefficient change, or zero for a fixed background.
    pub maximum_absolute_background_change: f64,
}

/// Stable final reflection intensity.
#[derive(Clone, Debug, PartialEq)]
pub struct TofReflectionIntensity {
    /// Stable phase ID.
    pub phase_id: String,
    /// Stable reflection ID.
    pub reflection_id: String,
    /// Nonnegative extracted integrated intensity.
    pub integrated_intensity: f64,
}

/// Complete fixed-instrument TOF extraction result.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailResult {
    /// Final calculated pattern and analytical derivatives.
    pub calculation: TofLeBailCalculation,
    /// Final residual metrics.
    pub metrics: ResidualEvaluation,
    /// Final phase state.
    pub phases: Vec<TofLeBailPhase>,
    /// Final refinable background state, if requested.
    pub background: Option<TofChebyshevBackground>,
    /// Flattened stable reflection intensities.
    pub intensities: Vec<TofReflectionIntensity>,
    /// Complete deterministic cycle history.
    pub history: Vec<TofLeBailIterationRecord>,
}

/// Calculate one TOF pattern and all direct profile derivatives in one pass.
///
/// # Errors
///
/// Returns [`TofLeBailError`] for invalid input, profile evaluation, or
/// allocation failure.
pub fn calculate_tof_lebail_pattern(
    input: &TofLeBailInput,
    options: &TofLeBailOptions,
) -> Result<TofLeBailCalculation, TofLeBailError> {
    input.validate()?;
    options.validate()?;
    let reflection_count = input
        .phases
        .iter()
        .try_fold(0_usize, |count, phase| {
            count.checked_add(phase.reflection_ids.len())
        })
        .ok_or(TofLeBailError::AllocationOverflow)?;
    let mut d_spacing = Vec::with_capacity(reflection_count);
    let mut intensity = Vec::with_capacity(reflection_count);
    let mut reflection_keys = Vec::with_capacity(reflection_count);
    let mut phase_offsets = Vec::with_capacity(input.phases.len() + 1);
    phase_offsets.push(0);
    for phase in &input.phases {
        d_spacing.extend_from_slice(&phase.d_spacing_angstrom);
        intensity.extend(
            phase
                .integrated_intensity
                .iter()
                .map(|value| phase.scale * value),
        );
        reflection_keys.extend(
            phase.reflection_ids.iter().map(|reflection_id| {
                (phase.phase_id.as_str().to_owned(), reflection_id.to_owned())
            }),
        );
        phase_offsets.push(d_spacing.len());
    }
    let accumulation = accumulate_tof_batch_with_context(
        GridView::new(&input.pattern.tof_us).map_err(TofError::from)?,
        &d_spacing,
        &intensity,
        input.instrument,
        options.support_fwhm,
        options.tail_log,
        options.execution.context(),
    )?;
    let profile_y = accumulation.y.clone();
    let (mut background_y, background_basis) = if let Some(background) = &input.background {
        let basis = background.basis(&input.pattern.tof_us)?;
        let values = background.calculate_from_basis(&basis)?;
        (values, Some(basis))
    } else {
        (vec![0.0; input.pattern.sample_count()], None)
    };
    for (residual, fixed) in background_y.iter_mut().zip(&input.pattern.background_y) {
        *residual += fixed;
    }
    let y = profile_y
        .iter()
        .zip(&background_y)
        .map(|(profile, background)| profile + background)
        .collect();
    Ok(TofLeBailCalculation {
        y,
        profile_y,
        background_y,
        background_basis,
        accumulation,
        reflection_keys,
        phase_offsets,
    })
}

/// Run deterministic nonnegative TOF Le Bail redistribution.
///
/// # Errors
///
/// Returns [`TofLeBailError`] for invalid state, profile evaluation, or
/// residual calculation failure.
pub fn refine_tof_lebail(
    input: &TofLeBailInput,
    options: &TofLeBailOptions,
) -> Result<TofLeBailResult, TofLeBailError> {
    input.validate()?;
    options.validate()?;
    let mut phases = initialize_intensities(input, options)?;
    let mut background = input.background.clone();
    let mut history = Vec::with_capacity(options.cycles);
    for iteration in 1..=options.cycles {
        let current_input = state_input(input, phases.clone(), background.clone())?;
        let calculation = calculate_tof_lebail_pattern(&current_input, options)?;
        let current = flatten_intensities(&phases);
        let updated = redistribute(&input.pattern, &calculation, &current, options)?;
        let maximum_relative_intensity_change = updated
            .iter()
            .zip(&current)
            .map(|(updated, current)| {
                (updated - current).abs() / current.abs().max(options.initial_intensity_floor)
            })
            .fold(0.0_f64, f64::max);
        phases = install_intensities(&phases, &updated)?;
        let intensity_input = state_input(input, phases.clone(), background.clone())?;
        let intensity_calculation = calculate_tof_lebail_pattern(&intensity_input, options)?;
        let previous_background = background.clone();
        background = refine_background(
            &input.pattern,
            &intensity_calculation.profile_y,
            background.as_ref(),
            options,
        )?;
        let maximum_absolute_background_change =
            maximum_background_change(previous_background.as_ref(), background.as_ref());
        let accepted_input = state_input(input, phases.clone(), background.clone())?;
        let accepted = calculate_tof_lebail_pattern(&accepted_input, options)?;
        let background_parameter_count = background
            .as_ref()
            .map_or(0, |background| background.coefficients.len());
        let metrics = evaluate_tof_residuals(
            &input.pattern,
            &accepted.y,
            ResidualOptions {
                use_uncertainty: options.use_uncertainty,
                parameter_count: updated.len() + background_parameter_count,
            },
        )?;
        history.push(TofLeBailIterationRecord {
            iteration,
            metrics,
            maximum_relative_intensity_change,
            maximum_absolute_background_change,
        });
    }
    let final_input = state_input(input, phases.clone(), background.clone())?;
    let calculation = calculate_tof_lebail_pattern(&final_input, options)?;
    let metrics = history
        .last()
        .ok_or(TofLeBailError::InvalidOptions)?
        .metrics
        .clone();
    let intensities = phases
        .iter()
        .flat_map(|phase| {
            phase
                .reflection_ids
                .iter()
                .zip(&phase.integrated_intensity)
                .map(
                    |(reflection_id, integrated_intensity)| TofReflectionIntensity {
                        phase_id: phase.phase_id.as_str().to_owned(),
                        reflection_id: reflection_id.clone(),
                        integrated_intensity: *integrated_intensity,
                    },
                )
        })
        .collect();
    Ok(TofLeBailResult {
        calculation,
        metrics,
        phases,
        background,
        intensities,
        history,
    })
}

fn initialize_intensities(
    input: &TofLeBailInput,
    options: &TofLeBailOptions,
) -> Result<Vec<TofLeBailPhase>, TofLeBailError> {
    let current = flatten_intensities(&input.phases);
    if current.iter().any(|value| *value > 0.0) {
        let values = current
            .iter()
            .map(|value| value.max(options.initial_intensity_floor))
            .collect::<Vec<_>>();
        return install_intensities(&input.phases, &values);
    }
    let observed = input
        .pattern
        .observed_y
        .as_deref()
        .ok_or(TofLeBailError::MissingObservations)?;
    let mut background = input.pattern.background_y.clone();
    if let Some(residual) = &input.background {
        for (fixed, value) in background
            .iter_mut()
            .zip(residual.calculate(&input.pattern.tof_us)?)
        {
            *fixed += value;
        }
    }
    let widths = bin_integration_weights(&input.pattern.tof_us);
    let area = observed
        .iter()
        .zip(&background)
        .zip(widths)
        .map(|((observed, background), width)| (observed - background).max(0.0) * width)
        .sum::<f64>();
    let count = current.len().max(1);
    #[allow(clippy::cast_precision_loss)]
    let starting = (area / count as f64).max(options.initial_intensity_floor);
    install_intensities(&input.phases, &vec![starting; current.len()])
}

fn redistribute(
    pattern: &TofPatternRecord,
    calculation: &TofLeBailCalculation,
    current: &[f64],
    options: &TofLeBailOptions,
) -> Result<Vec<f64>, TofLeBailError> {
    let observed = pattern
        .observed_y
        .as_deref()
        .ok_or(TofLeBailError::MissingObservations)?;
    let included = pattern
        .mask
        .clone()
        .unwrap_or_else(|| vec![true; pattern.sample_count()]);
    let ratio = observed
        .iter()
        .zip(&calculation.background_y)
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
    let mut weights = bin_integration_weights(&pattern.tof_us);
    if options.redistribution_use_uncertainty
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
    if local.peak_count() != current.len() {
        return Err(TofLeBailError::IntensityLengthMismatch);
    }
    let mut updated = vec![0.0; current.len()];
    for reflection in 0..current.len() {
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
        let raw = if denominator > 0.0 {
            (current[reflection] * numerator / denominator).max(0.0)
        } else {
            current[reflection]
        };
        updated[reflection] =
            current[reflection] + options.redistribution_damping * (raw - current[reflection]);
    }
    Ok(updated)
}

fn state_input(
    original: &TofLeBailInput,
    phases: Vec<TofLeBailPhase>,
    background: Option<TofChebyshevBackground>,
) -> Result<TofLeBailInput, TofLeBailError> {
    let mut result = TofLeBailInput::new(original.pattern.clone(), original.instrument, phases)?;
    result.background = background;
    result.validate()?;
    Ok(result)
}

fn refine_background(
    pattern: &TofPatternRecord,
    profile_y: &[f64],
    background: Option<&TofChebyshevBackground>,
    options: &TofLeBailOptions,
) -> Result<Option<TofChebyshevBackground>, TofLeBailError> {
    let Some(background) = background else {
        return Ok(None);
    };
    let observed = pattern
        .observed_y
        .as_deref()
        .ok_or(TofLeBailError::MissingObservations)?;
    let basis = background.basis(&pattern.tof_us)?;
    if profile_y.len() != pattern.sample_count() || basis.rows != pattern.sample_count() {
        return Err(TofLeBailError::BackgroundBasisShape);
    }
    let included_count = (0..pattern.sample_count())
        .filter(|index| pattern.mask.as_ref().is_none_or(|mask| mask[*index]))
        .count();
    if included_count < basis.columns {
        return Err(TofLeBailError::InsufficientBackgroundObservations);
    }
    let count = included_count
        .checked_mul(basis.columns)
        .ok_or(TofLeBailError::AllocationOverflow)?;
    let mut design = Vec::with_capacity(count);
    let mut target = Vec::with_capacity(included_count);
    for sample in 0..pattern.sample_count() {
        if pattern.mask.as_ref().is_some_and(|mask| !mask[sample]) {
            continue;
        }
        let sigma = if options.use_uncertainty {
            pattern
                .uncertainty
                .as_ref()
                .map_or(1.0, |values| values[sample])
        } else {
            1.0
        };
        let row = basis
            .values
            .get(sample * basis.columns..(sample + 1) * basis.columns)
            .ok_or(TofLeBailError::BackgroundBasisShape)?;
        design.extend(row.iter().map(|value| value / sigma));
        target.push((observed[sample] - pattern.background_y[sample] - profile_y[sample]) / sigma);
    }
    let matrix = DMatrix::from_row_slice(included_count, basis.columns, &design);
    let target = DVector::from_vec(target);
    let coefficients = matrix
        .svd(true, true)
        .solve(&target, 1.0e-12)
        .map_err(|_| TofLeBailError::BackgroundLinearSolve)?;
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(TofLeBailError::BackgroundLinearSolve);
    }
    background
        .with_coefficients(coefficients.as_slice().to_vec())
        .map(Some)
}

fn maximum_background_change(
    previous: Option<&TofChebyshevBackground>,
    updated: Option<&TofChebyshevBackground>,
) -> f64 {
    previous.zip(updated).map_or(0.0, |(previous, updated)| {
        previous
            .coefficients
            .iter()
            .zip(&updated.coefficients)
            .map(|(previous, updated)| (updated - previous).abs())
            .fold(0.0_f64, f64::max)
    })
}

fn flatten_intensities(phases: &[TofLeBailPhase]) -> Vec<f64> {
    phases
        .iter()
        .flat_map(|phase| phase.integrated_intensity.iter().copied())
        .collect()
}

fn install_intensities(
    phases: &[TofLeBailPhase],
    values: &[f64],
) -> Result<Vec<TofLeBailPhase>, TofLeBailError> {
    if phases
        .iter()
        .map(|phase| phase.reflection_ids.len())
        .sum::<usize>()
        != values.len()
    {
        return Err(TofLeBailError::IntensityLengthMismatch);
    }
    let mut offset = 0;
    phases
        .iter()
        .map(|phase| {
            let end = offset + phase.reflection_ids.len();
            let result = phase.with_intensities(&values[offset..end]);
            offset = end;
            result
        })
        .collect()
}

fn bin_integration_weights(x: &[f64]) -> Vec<f64> {
    match x.len() {
        0 => Vec::new(),
        1 => vec![1.0],
        count => (0..count)
            .map(|index| {
                if index == 0 {
                    0.5 * (x[1] - x[0])
                } else if index + 1 == count {
                    0.5 * (x[count - 1] - x[count - 2])
                } else {
                    0.5 * (x[index + 1] - x[index - 1])
                }
            })
            .collect(),
    }
}

/// Invalid TOF Le Bail request or numerical state.
#[derive(Debug)]
pub enum TofLeBailError {
    /// TOF pattern validation failed.
    Pattern(DomainError),
    /// The observed intensity array is absent.
    MissingObservations,
    /// Instrument/profile evaluation failed.
    Profile(TofError),
    /// A phase invariant failed.
    InvalidPhase(&'static str),
    /// Numerical options are invalid.
    InvalidOptions,
    /// A Chebyshev background has invalid coefficients or domain.
    InvalidBackground,
    /// A background grid is empty, non-finite, or unsorted.
    InvalidBackgroundGrid,
    /// The TOF grid extends outside the background domain.
    BackgroundGridOutsideDomain,
    /// Replacement background coefficients have the wrong length.
    BackgroundCoefficientLengthMismatch,
    /// A background derivative basis has an inconsistent shape.
    BackgroundBasisShape,
    /// Too few included observations remain to determine every coefficient.
    InsufficientBackgroundObservations,
    /// The weighted Chebyshev least-squares solve failed.
    BackgroundLinearSolve,
    /// Flattened reflection intensities have the wrong length.
    IntensityLengthMismatch,
    /// Checked allocation arithmetic overflowed.
    AllocationOverflow,
    /// Residual evaluation failed.
    Residual(ResidualError),
    /// Execution policy construction failed.
    Execution(ExecutionPolicyError),
}

impl Display for TofLeBailError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern(error) => Display::fmt(error, formatter),
            Self::MissingObservations => {
                formatter.write_str("observed_y is required for TOF Le Bail extraction")
            }
            Self::Profile(error) => Display::fmt(error, formatter),
            Self::InvalidPhase(message) => formatter.write_str(message),
            Self::InvalidOptions => formatter.write_str("invalid TOF Le Bail options"),
            Self::InvalidBackground => {
                formatter.write_str("invalid TOF Chebyshev background coefficients or domain")
            }
            Self::InvalidBackgroundGrid => {
                formatter.write_str("TOF Chebyshev background grid must be finite and increasing")
            }
            Self::BackgroundGridOutsideDomain => {
                formatter.write_str("TOF grid extends outside the Chebyshev background domain")
            }
            Self::BackgroundCoefficientLengthMismatch => {
                formatter.write_str("TOF Chebyshev background coefficient length mismatch")
            }
            Self::BackgroundBasisShape => {
                formatter.write_str("TOF Chebyshev background basis shape mismatch")
            }
            Self::InsufficientBackgroundObservations => formatter.write_str(
                "TOF Chebyshev refinement has fewer included observations than coefficients",
            ),
            Self::BackgroundLinearSolve => {
                formatter.write_str("TOF Chebyshev weighted linear solve failed")
            }
            Self::IntensityLengthMismatch => {
                formatter.write_str("TOF reflection intensity length mismatch")
            }
            Self::AllocationOverflow => formatter.write_str("TOF Le Bail allocation overflow"),
            Self::Residual(error) => Display::fmt(error, formatter),
            Self::Execution(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for TofLeBailError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pattern(error) => Some(error),
            Self::Profile(error) => Some(error),
            Self::Residual(error) => Some(error),
            Self::Execution(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TofError> for TofLeBailError {
    fn from(value: TofError) -> Self {
        Self::Profile(value)
    }
}

impl From<ResidualError> for TofLeBailError {
    fn from(value: ResidualError) -> Self {
        Self::Residual(value)
    }
}

impl From<ExecutionPolicyError> for TofLeBailError {
    fn from(value: ExecutionPolicyError) -> Self {
        Self::Execution(value)
    }
}
