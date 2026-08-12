//! Atomic multi-bank fixed-cell TOF Le Bail extraction.
//!
//! Every bank owns its grid, instrument, observations, background, phase
//! scales, and extracted intensities. Stable phase/reflection identities,
//! Miller indices, and d-spacings form the shared fixed-cell contract.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use phasesmith_model::RecordId;

use crate::tof_lebail::{
    flatten_intensities, initialize_intensities, install_intensities, maximum_background_change,
    normal_tof_stop, redistribute, refine_background, state_input,
};
use crate::{
    DiagnosticValue, RefinementEventKind, RefinementLimits, RefinementRuntime, ResidualEvaluation,
    ResidualOptions, RuntimeError, TerminationReason, TofChebyshevBackground, TofLeBailCalculation,
    TofLeBailError, TofLeBailInput, TofLeBailOptions, TofLeBailPhase, TofReflectionIntensity,
    calculate_tof_lebail_pattern, evaluate_tof_residuals,
};

/// One detector bank in an atomic multi-bank extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLeBailBank {
    /// Stable bank identity used by application and persistence layers.
    pub bank_id: RecordId,
    /// Bank-local observations, instrument, phase scales/intensities, and background.
    pub input: TofLeBailInput,
}

/// Two or more TOF banks sharing fixed phase/reflection geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankInput {
    /// Banks in deterministic caller-owned order.
    pub banks: Vec<TofLeBailBank>,
}

impl TofMultiBankInput {
    /// Validate bank identities, local inputs, and shared fixed-cell topology.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankError`] for fewer than two banks, duplicate IDs,
    /// invalid local state, or inconsistent phase/reflection geometry.
    pub fn validate(&self) -> Result<(), TofMultiBankError> {
        if self.banks.len() < 2 {
            return Err(TofMultiBankError::TooFewBanks);
        }
        let mut bank_ids = BTreeSet::new();
        for bank in &self.banks {
            bank.input.validate()?;
            if !bank_ids.insert(bank.bank_id.clone()) {
                return Err(TofMultiBankError::DuplicateBankId {
                    bank_id: bank.bank_id.clone(),
                });
            }
        }
        let shared = &self.banks[0].input.phases;
        for bank in self.banks.iter().skip(1) {
            validate_shared_topology(shared, &bank.input.phases, &bank.bank_id)?;
        }
        Ok(())
    }
}

/// Aggregate residual metrics over every included sample in all banks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TofMultiBankMetrics {
    /// Included observations summed over all banks.
    pub included_samples: usize,
    /// Sum of absolute residuals divided by summed absolute observations.
    pub rp: f64,
    /// Square root of joint chi-square divided by weighted observed square sum.
    pub rwp: f64,
    /// Sum of selected squared weighted residuals over every bank.
    pub chi_square: f64,
    /// Joint chi-square divided by included samples minus fitted local parameters.
    pub reduced_chi_square: f64,
}

/// One bank's accepted display state.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankResultBank {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Final calculation and fused derivative product.
    pub calculation: TofLeBailCalculation,
    /// Final bank-local residual arrays and metrics.
    pub metrics: ResidualEvaluation,
    /// Final bank-local phase scales and extracted intensities.
    pub phases: Vec<TofLeBailPhase>,
    /// Final bank-local refinable background.
    pub background: Option<TofChebyshevBackground>,
    /// Flattened stable reflection intensities for this bank.
    pub intensities: Vec<TofReflectionIntensity>,
}

/// One bank's last accepted state inside a joint checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankCheckpointBank {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Accepted local phase/intensity state.
    pub phases: Vec<TofLeBailPhase>,
    /// Accepted local background state.
    pub background: Option<TofChebyshevBackground>,
}

/// One atomically accepted multi-bank redistribution cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankIterationRecord {
    /// One-based accepted cycle index.
    pub iteration: usize,
    /// Bank-local residual records in input order.
    pub bank_metrics: Vec<ResidualEvaluation>,
    /// Aggregate metrics over every bank.
    pub metrics: TofMultiBankMetrics,
    /// Largest relative intensity change over every bank/reflection.
    pub maximum_relative_intensity_change: f64,
    /// Largest absolute background-coefficient change over every bank.
    pub maximum_absolute_background_change: f64,
}

/// Complete immutable continuation state for atomic multi-bank extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankCheckpoint {
    /// Number of atomically accepted cycles.
    pub completed_iterations: usize,
    /// Accepted bank-local states in immutable request order.
    pub banks: Vec<TofMultiBankCheckpointBank>,
    /// Complete deterministic joint history.
    pub history: Vec<TofMultiBankIterationRecord>,
}

impl TofMultiBankCheckpoint {
    /// Revalidate this continuation against the exact multi-bank contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankError::InvalidCheckpoint`] for stale identities,
    /// topology, background contracts, or non-contiguous history.
    pub fn validate_for(
        &self,
        input: &TofMultiBankInput,
        options: &TofLeBailOptions,
    ) -> Result<(), TofMultiBankError> {
        input.validate()?;
        options.validate()?;
        if self.completed_iterations != self.history.len()
            || self.completed_iterations > options.cycles
            || self.banks.len() != input.banks.len()
        {
            return Err(TofMultiBankError::InvalidCheckpoint(
                "multi-bank checkpoint counts differ from the request",
            ));
        }
        for (saved, original) in self.banks.iter().zip(&input.banks) {
            if saved.bank_id != original.bank_id {
                return Err(TofMultiBankError::InvalidCheckpoint(
                    "multi-bank checkpoint bank order or identity changed",
                ));
            }
            validate_checkpoint_bank(saved, original)?;
        }
        if self.history.iter().enumerate().any(|(index, record)| {
            record.iteration != index + 1 || record.bank_metrics.len() != input.banks.len()
        }) {
            return Err(TofMultiBankError::InvalidCheckpoint(
                "multi-bank checkpoint history is not contiguous and bank-aligned",
            ));
        }
        Ok(())
    }
}

/// Complete result from one atomic multi-bank fixed-cell extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankResult {
    /// Final bank-local states and calculations in request order.
    pub banks: Vec<TofMultiBankResultBank>,
    /// Final aggregate residual metrics.
    pub metrics: TofMultiBankMetrics,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankIterationRecord>,
    /// Stable bounded-runtime termination category.
    pub termination_reason: TerminationReason,
    /// Complete last accepted atomic state.
    pub checkpoint: TofMultiBankCheckpoint,
}

/// Run atomic fixed-cell TOF Le Bail extraction over two or more banks.
///
/// # Errors
///
/// Returns [`TofMultiBankError`] for an invalid shared/local contract or a
/// numerical/runtime failure.
pub fn refine_tof_multibank(
    input: &TofMultiBankInput,
    options: &TofLeBailOptions,
) -> Result<TofMultiBankResult, TofMultiBankError> {
    input.validate()?;
    options.validate()?;
    let max_evaluations = options
        .cycles
        .checked_mul(3)
        .ok_or(TofLeBailError::AllocationOverflow)?;
    let limits = RefinementLimits::new(options.cycles, max_evaluations, None, 1)?;
    let mut runtime = RefinementRuntime::new(limits, None)?;
    refine_tof_multibank_with_runtime(input, options, None, &mut runtime)
}

/// Run atomic multi-bank extraction with host runtime and optional continuation.
///
/// A model evaluation covers all banks. Candidate state is accepted only after
/// every bank completes the cycle, so cancellation or failure cannot expose a
/// partially advanced bank set.
///
/// # Errors
///
/// Returns [`TofMultiBankError`] for contract, calculation, residual, or
/// non-normal runtime failures.
#[allow(clippy::too_many_lines)]
pub fn refine_tof_multibank_with_runtime(
    input: &TofMultiBankInput,
    options: &TofLeBailOptions,
    checkpoint: Option<&TofMultiBankCheckpoint>,
    runtime: &mut RefinementRuntime<TofMultiBankCheckpoint>,
) -> Result<TofMultiBankResult, TofMultiBankError> {
    input.validate()?;
    options.validate()?;
    let restored = restore_state(input, options, checkpoint)?;
    let mut states = restored.states;
    let mut history = restored.history;
    if let Some(checkpoint) = checkpoint {
        runtime.resume_accepted(checkpoint.completed_iterations)?;
    }
    runtime.emit(
        RefinementEventKind::Start,
        "tof_multibank",
        "multi-bank TOF Le Bail extraction started",
        vec![(
            "bank_count".to_owned(),
            DiagnosticValue::Unsigned(input.banks.len() as u64),
        )],
    )?;
    let mut accepted_calculations = None;
    let mut termination = TerminationReason::MaxIterations;
    for iteration in restored.first_iteration..=options.cycles {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_tof_stop(error)?;
            break;
        }
        if let Err(error) = runtime.begin_evaluation() {
            termination = normal_tof_stop(error)?;
            break;
        }
        let current_calculations = calculate_states(input, &states, options)?;
        let mut candidate_states = Vec::with_capacity(states.len());
        let mut maximum_relative_intensity_change = 0.0_f64;
        for ((bank, state), calculation) in
            input.banks.iter().zip(&states).zip(&current_calculations)
        {
            let current = flatten_intensities(&state.phases);
            let updated = redistribute(&bank.input.pattern, calculation, &current, options)?;
            maximum_relative_intensity_change = maximum_relative_intensity_change.max(
                updated
                    .iter()
                    .zip(&current)
                    .map(|(updated, current)| {
                        (updated - current).abs()
                            / current.abs().max(options.initial_intensity_floor)
                    })
                    .fold(0.0_f64, f64::max),
            );
            candidate_states.push(AcceptedBankState {
                phases: install_intensities(&state.phases, &updated)?,
                background: state.background.clone(),
            });
        }
        if let Err(error) = runtime.begin_evaluation() {
            termination = normal_tof_stop(error)?;
            break;
        }
        let intensity_calculations = calculate_states(input, &candidate_states, options)?;
        let mut maximum_absolute_background_change = 0.0_f64;
        for (((bank, previous), candidate), calculation) in input
            .banks
            .iter()
            .zip(&states)
            .zip(&mut candidate_states)
            .zip(&intensity_calculations)
        {
            let background = refine_background(
                &bank.input.pattern,
                &calculation.profile_y,
                candidate.background.as_ref(),
                options,
            )?;
            maximum_absolute_background_change = maximum_absolute_background_change.max(
                maximum_background_change(previous.background.as_ref(), background.as_ref()),
            );
            candidate.background = background;
        }
        if let Err(error) = runtime.begin_evaluation() {
            termination = normal_tof_stop(error)?;
            break;
        }
        let calculations = calculate_states(input, &candidate_states, options)?;
        let bank_metrics = evaluate_bank_metrics(input, &candidate_states, &calculations, options)?;
        let parameter_count = fitted_parameter_count(&candidate_states)?;
        let metrics = aggregate_metrics(input, &calculations, options, parameter_count)?;
        states = candidate_states;
        history.push(TofMultiBankIterationRecord {
            iteration,
            bank_metrics: bank_metrics.clone(),
            metrics,
            maximum_relative_intensity_change,
            maximum_absolute_background_change,
        });
        accepted_calculations = Some(calculations);
        let accepted_checkpoint = checkpoint_from_states(input, &states, &history);
        runtime.accept_step(Some(&accepted_checkpoint))?;
        runtime.emit(
            RefinementEventKind::Iteration,
            "tof_multibank_iteration",
            "multi-bank TOF Le Bail cycle accepted",
            vec![
                ("rwp".to_owned(), DiagnosticValue::Float(metrics.rwp)),
                (
                    "maximum_relative_intensity_change".to_owned(),
                    DiagnosticValue::Float(maximum_relative_intensity_change),
                ),
                (
                    "maximum_absolute_background_change".to_owned(),
                    DiagnosticValue::Float(maximum_absolute_background_change),
                ),
            ],
        )?;
    }
    let calculations = match accepted_calculations {
        Some(calculations) => calculations,
        None => calculate_states(input, &states, options)?,
    };
    let bank_metrics = evaluate_bank_metrics(input, &states, &calculations, options)?;
    let parameter_count = fitted_parameter_count(&states)?;
    let metrics = aggregate_metrics(input, &calculations, options, parameter_count)?;
    let checkpoint = checkpoint_from_states(input, &states, &history);
    checkpoint.validate_for(input, options)?;
    let banks = input
        .banks
        .iter()
        .zip(states)
        .zip(calculations)
        .zip(bank_metrics)
        .map(
            |(((bank, state), calculation), metrics)| TofMultiBankResultBank {
                bank_id: bank.bank_id.clone(),
                calculation,
                metrics,
                intensities: reflection_intensities(&state.phases),
                phases: state.phases,
                background: state.background,
            },
        )
        .collect();
    runtime.emit(
        RefinementEventKind::Termination,
        "tof_multibank",
        "multi-bank TOF Le Bail extraction terminated",
        vec![(
            "termination_reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(TofMultiBankResult {
        banks,
        metrics,
        history,
        termination_reason: termination,
        checkpoint,
    })
}

#[derive(Clone)]
struct AcceptedBankState {
    phases: Vec<TofLeBailPhase>,
    background: Option<TofChebyshevBackground>,
}

struct RestoredState {
    states: Vec<AcceptedBankState>,
    history: Vec<TofMultiBankIterationRecord>,
    first_iteration: usize,
}

fn restore_state(
    input: &TofMultiBankInput,
    options: &TofLeBailOptions,
    checkpoint: Option<&TofMultiBankCheckpoint>,
) -> Result<RestoredState, TofMultiBankError> {
    let Some(checkpoint) = checkpoint else {
        let states = input
            .banks
            .iter()
            .map(|bank| {
                Ok(AcceptedBankState {
                    phases: initialize_intensities(&bank.input, options)?,
                    background: bank.input.background.clone(),
                })
            })
            .collect::<Result<Vec<_>, TofMultiBankError>>()?;
        return Ok(RestoredState {
            states,
            history: Vec::new(),
            first_iteration: 1,
        });
    };
    checkpoint.validate_for(input, options)?;
    Ok(RestoredState {
        states: checkpoint
            .banks
            .iter()
            .map(|bank| AcceptedBankState {
                phases: bank.phases.clone(),
                background: bank.background.clone(),
            })
            .collect(),
        history: checkpoint.history.clone(),
        first_iteration: checkpoint
            .completed_iterations
            .checked_add(1)
            .ok_or(TofLeBailError::AllocationOverflow)?,
    })
}

fn calculate_states(
    input: &TofMultiBankInput,
    states: &[AcceptedBankState],
    options: &TofLeBailOptions,
) -> Result<Vec<TofLeBailCalculation>, TofMultiBankError> {
    input
        .banks
        .iter()
        .zip(states)
        .map(|(bank, state)| {
            let live = state_input(&bank.input, state.phases.clone(), state.background.clone())?;
            Ok(calculate_tof_lebail_pattern(&live, options)?)
        })
        .collect()
}

fn evaluate_bank_metrics(
    input: &TofMultiBankInput,
    states: &[AcceptedBankState],
    calculations: &[TofLeBailCalculation],
    options: &TofLeBailOptions,
) -> Result<Vec<ResidualEvaluation>, TofMultiBankError> {
    input
        .banks
        .iter()
        .zip(states)
        .zip(calculations)
        .map(|((bank, state), calculation)| {
            Ok(evaluate_tof_residuals(
                &bank.input.pattern,
                &calculation.y,
                ResidualOptions {
                    use_uncertainty: options.use_uncertainty,
                    parameter_count: flatten_intensities(&state.phases).len()
                        + state
                            .background
                            .as_ref()
                            .map_or(0, |background| background.coefficients().len()),
                },
            )
            .map_err(TofLeBailError::from)?)
        })
        .collect()
}

fn fitted_parameter_count(states: &[AcceptedBankState]) -> Result<usize, TofMultiBankError> {
    states.iter().try_fold(0_usize, |total, state| {
        total
            .checked_add(flatten_intensities(&state.phases).len())
            .and_then(|value| {
                value.checked_add(
                    state
                        .background
                        .as_ref()
                        .map_or(0, |background| background.coefficients().len()),
                )
            })
            .ok_or_else(|| TofLeBailError::AllocationOverflow.into())
    })
}

fn aggregate_metrics(
    input: &TofMultiBankInput,
    calculations: &[TofLeBailCalculation],
    options: &TofLeBailOptions,
    parameter_count: usize,
) -> Result<TofMultiBankMetrics, TofMultiBankError> {
    let mut included_samples = 0_usize;
    let mut absolute_residual_sum = 0.0;
    let mut absolute_observed_sum = 0.0;
    let mut weighted_observed_square_sum = 0.0;
    let mut chi_square = 0.0;
    for (bank, calculation) in input.banks.iter().zip(calculations) {
        let pattern = &bank.input.pattern;
        let observed = pattern
            .observed_y
            .as_ref()
            .ok_or(TofLeBailError::MissingObservations)?;
        let uncertainty = options
            .use_uncertainty
            .then_some(pattern.uncertainty.as_deref())
            .flatten();
        for sample in 0..pattern.sample_count() {
            if pattern.mask.as_ref().is_some_and(|mask| !mask[sample]) {
                continue;
            }
            included_samples = included_samples
                .checked_add(1)
                .ok_or(TofLeBailError::AllocationOverflow)?;
            let residual = calculation.y[sample] - observed[sample];
            absolute_residual_sum += residual.abs();
            absolute_observed_sum += observed[sample].abs();
            let (weighted_residual, weighted_observed) = uncertainty.map_or_else(
                || (residual, observed[sample]),
                |sigma| (residual / sigma[sample], observed[sample] / sigma[sample]),
            );
            chi_square += weighted_residual * weighted_residual;
            weighted_observed_square_sum += weighted_observed * weighted_observed;
        }
    }
    let degrees_of_freedom = included_samples.checked_sub(parameter_count);
    Ok(TofMultiBankMetrics {
        included_samples,
        rp: if absolute_observed_sum == 0.0 {
            f64::INFINITY
        } else {
            absolute_residual_sum / absolute_observed_sum
        },
        rwp: if weighted_observed_square_sum == 0.0 {
            f64::INFINITY
        } else {
            (chi_square / weighted_observed_square_sum).sqrt()
        },
        chi_square,
        reduced_chi_square: match degrees_of_freedom {
            Some(degrees) if degrees > 0 => chi_square / count_as_f64(degrees),
            _ => f64::INFINITY,
        },
    })
}

fn checkpoint_from_states(
    input: &TofMultiBankInput,
    states: &[AcceptedBankState],
    history: &[TofMultiBankIterationRecord],
) -> TofMultiBankCheckpoint {
    TofMultiBankCheckpoint {
        completed_iterations: history.len(),
        banks: input
            .banks
            .iter()
            .zip(states)
            .map(|(bank, state)| TofMultiBankCheckpointBank {
                bank_id: bank.bank_id.clone(),
                phases: state.phases.clone(),
                background: state.background.clone(),
            })
            .collect(),
        history: history.to_vec(),
    }
}

fn validate_shared_topology(
    shared: &[TofLeBailPhase],
    local: &[TofLeBailPhase],
    bank_id: &RecordId,
) -> Result<(), TofMultiBankError> {
    if shared.len() != local.len()
        || shared.iter().zip(local).any(|(shared, local)| {
            shared.phase_id() != local.phase_id()
                || shared.name() != local.name()
                || shared.reflection_ids() != local.reflection_ids()
                || shared.hkl() != local.hkl()
                || shared.d_spacing_angstrom() != local.d_spacing_angstrom()
        })
    {
        return Err(TofMultiBankError::SharedTopologyMismatch {
            bank_id: bank_id.clone(),
        });
    }
    Ok(())
}

fn validate_checkpoint_bank(
    saved: &TofMultiBankCheckpointBank,
    original: &TofLeBailBank,
) -> Result<(), TofMultiBankError> {
    validate_shared_topology(&original.input.phases, &saved.phases, &saved.bank_id)?;
    if saved
        .phases
        .iter()
        .zip(&original.input.phases)
        .any(|(saved, original)| saved.scale().to_bits() != original.scale().to_bits())
    {
        return Err(TofMultiBankError::InvalidCheckpoint(
            "multi-bank checkpoint phase scale changed",
        ));
    }
    match (&saved.background, &original.input.background) {
        (None, None) => Ok(()),
        (Some(saved), Some(original))
            if saved.background_id() == original.background_id()
                && saved
                    .domain_us()
                    .iter()
                    .zip(original.domain_us())
                    .all(|(saved, original)| saved.to_bits() == original.to_bits())
                && saved.coefficients().len() == original.coefficients().len() =>
        {
            Ok(())
        }
        _ => Err(TofMultiBankError::InvalidCheckpoint(
            "multi-bank checkpoint background contract changed",
        )),
    }
}

fn reflection_intensities(phases: &[TofLeBailPhase]) -> Vec<TofReflectionIntensity> {
    phases
        .iter()
        .flat_map(|phase| {
            phase
                .reflection_ids()
                .iter()
                .zip(phase.integrated_intensity())
                .map(
                    |(reflection_id, integrated_intensity)| TofReflectionIntensity {
                        phase_id: phase.phase_id().as_str().to_owned(),
                        reflection_id: reflection_id.clone(),
                        integrated_intensity: *integrated_intensity,
                    },
                )
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(value: usize) -> f64 {
    value as f64
}

/// Invalid multi-bank TOF request, shared contract, or numerical state.
#[derive(Debug)]
pub enum TofMultiBankError {
    /// At least two banks are required.
    TooFewBanks,
    /// A stable bank identity is repeated.
    DuplicateBankId {
        /// Repeated bank identity.
        bank_id: RecordId,
    },
    /// One bank disagrees with shared phase/reflection geometry.
    SharedTopologyMismatch {
        /// Inconsistent bank identity.
        bank_id: RecordId,
    },
    /// One bank's single-pattern workflow failed.
    LeBail(TofLeBailError),
    /// A continuation state is stale or malformed.
    InvalidCheckpoint(&'static str),
    /// Runtime control or checkpoint delivery failed.
    Runtime(RuntimeError),
}

impl Display for TofMultiBankError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewBanks => formatter.write_str("multi-bank TOF requires at least two banks"),
            Self::DuplicateBankId { bank_id } => write!(formatter, "duplicate TOF bank {bank_id}"),
            Self::SharedTopologyMismatch { bank_id } => write!(
                formatter,
                "TOF bank {bank_id} differs from shared phase/reflection geometry"
            ),
            Self::LeBail(error) => Display::fmt(error, formatter),
            Self::InvalidCheckpoint(message) => formatter.write_str(message),
            Self::Runtime(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for TofMultiBankError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LeBail(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TofLeBailError> for TofMultiBankError {
    fn from(value: TofLeBailError) -> Self {
        Self::LeBail(value)
    }
}

impl From<RuntimeError> for TofMultiBankError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
