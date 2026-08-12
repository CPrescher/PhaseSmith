//! Bounded shared-cell refinement over atomic multi-bank TOF Le Bail cycles.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};
use phasesmith_crystallography::UnitCell;
use phasesmith_model::RecordId;

use crate::tof_lebail::{initialize_intensities, normal_tof_stop};
use crate::tof_multibank::{
    AcceptedBankState, MultiBankCycleCandidate, MultiBankCycleOutcome, aggregate_metrics,
    calculate_states, evaluate_bank_metrics, fitted_parameter_count, prepare_multibank_cycle,
    reflection_intensities,
};
use crate::{
    DiagnosticValue, LatticeBounds, LatticeError, LatticeParameterization, RefinementEventKind,
    RefinementLimits, RefinementRuntime, ResidualEvaluation, RuntimeError, TerminationReason,
    TofLeBailError, TofLeBailOptions, TofMultiBankCheckpointBank, TofMultiBankError,
    TofMultiBankInput, TofMultiBankMetrics, TofMultiBankResultBank, tof_lattice_geometry,
};

/// One phase whose symmetry-independent cell is shared across every detector bank.
#[derive(Clone, Debug, PartialEq)]
pub struct TofSharedLatticePhase {
    phase_id: RecordId,
    parameterization: LatticeParameterization,
    bounds: LatticeBounds,
    initial_cell: UnitCell,
}

impl TofSharedLatticePhase {
    /// Construct one bounded shared-cell phase.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankLatticeError`] when the initial cell or bounds do
    /// not match the setting-aware parameterization.
    pub fn new(
        phase_id: RecordId,
        parameterization: LatticeParameterization,
        bounds: LatticeBounds,
        initial_cell: UnitCell,
    ) -> Result<Self, TofMultiBankLatticeError> {
        let result = Self {
            phase_id,
            parameterization,
            bounds,
            initial_cell,
        };
        result.validate_cell(initial_cell)?;
        Ok(result)
    }

    /// Stable phase identity shared with every bank-local phase record.
    #[must_use]
    pub const fn phase_id(&self) -> &RecordId {
        &self.phase_id
    }

    /// Setting-aware independent-cell mapping.
    #[must_use]
    pub const fn parameterization(&self) -> &LatticeParameterization {
        &self.parameterization
    }

    /// Closed physical bounds in independent-parameter order.
    #[must_use]
    pub const fn bounds(&self) -> &LatticeBounds {
        &self.bounds
    }

    /// Initial shared unit cell.
    #[must_use]
    pub const fn initial_cell(&self) -> UnitCell {
        self.initial_cell
    }

    fn validate_cell(&self, cell: UnitCell) -> Result<Vec<f64>, TofMultiBankLatticeError> {
        if self.bounds.parameter_names() != self.parameterization.parameter_names() {
            return Err(TofMultiBankLatticeError::InvalidModel(
                "TOF lattice bounds use a different independent-parameter order",
            ));
        }
        let values = self.parameterization.values_from_cell(cell)?;
        if values
            .iter()
            .zip(self.bounds.lower().iter().zip(self.bounds.upper()))
            .any(|(value, (lower, upper))| value < lower || value > upper)
        {
            return Err(TofMultiBankLatticeError::InvalidModel(
                "TOF lattice cell lies outside its declared bounds",
            ));
        }
        for corner in self.bounds.corner_values() {
            self.parameterization.to_cell(&corner)?;
        }
        Ok(values)
    }
}

/// Multi-bank observations plus one or more shared phase-cell models.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankLatticeInput {
    /// Atomic bank-local TOF request.
    pub multibank: TofMultiBankInput,
    /// Shared cells in deterministic parameter packing order.
    pub lattice_phases: Vec<TofSharedLatticePhase>,
}

impl TofMultiBankLatticeInput {
    /// Validate bank-local state, selected phase identities, and bounded cells.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankLatticeError`] for an invalid multi-bank request,
    /// empty/duplicate lattice selection, or a phase absent from the banks.
    pub fn validate(&self) -> Result<(), TofMultiBankLatticeError> {
        self.multibank.validate()?;
        if self.lattice_phases.is_empty() {
            return Err(TofMultiBankLatticeError::InvalidModel(
                "shared TOF lattice refinement requires at least one phase",
            ));
        }
        let mut phase_ids = BTreeSet::new();
        for lattice in &self.lattice_phases {
            if !phase_ids.insert(lattice.phase_id.clone()) {
                return Err(TofMultiBankLatticeError::InvalidModel(
                    "shared TOF lattice phase IDs must be unique",
                ));
            }
            lattice.validate_cell(lattice.initial_cell)?;
            for bank in &self.multibank.banks {
                let phase = bank
                    .input
                    .phases
                    .iter()
                    .find(|phase| phase.phase_id() == lattice.phase_id())
                    .ok_or(TofMultiBankLatticeError::InvalidModel(
                        "shared TOF lattice phase is absent from the bank topology",
                    ))?;
                let geometry = tof_lattice_geometry(
                    lattice.parameterization(),
                    lattice.initial_cell(),
                    phase.hkl(),
                    bank.input.instrument,
                )?;
                if geometry.d_spacing_angstrom != phase.d_spacing_angstrom() {
                    return Err(TofMultiBankLatticeError::InvalidModel(
                        "shared TOF lattice initial cell disagrees with bank d-spacings",
                    ));
                }
            }
        }
        Ok(())
    }

    fn parameter_count(&self) -> Result<usize, TofMultiBankLatticeError> {
        self.lattice_phases
            .iter()
            .try_fold(0_usize, |count, phase| {
                count
                    .checked_add(phase.parameterization.parameter_names().len())
                    .ok_or(TofMultiBankLatticeError::AllocationOverflow)
            })
    }
}

/// Controls for alternating local Le Bail updates and one shared lattice step.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankLatticeOptions {
    /// Existing TOF redistribution, profile-support, weighting, and execution controls.
    pub lebail: TofLeBailOptions,
    /// Non-negative diagonal regularization in scaled lattice coordinates.
    pub lattice_damping: f64,
    /// Maximum absolute lattice step in scaled coordinates.
    pub max_scaled_lattice_step: f64,
    /// Number of objective backtracking halvings after the initial trial.
    pub max_lattice_backtracks: usize,
}

impl TofMultiBankLatticeOptions {
    /// Construct validated shared-lattice solver controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankLatticeError`] for invalid Le Bail or lattice controls.
    pub fn new(
        lebail: TofLeBailOptions,
        lattice_damping: f64,
        max_scaled_lattice_step: f64,
        max_lattice_backtracks: usize,
    ) -> Result<Self, TofMultiBankLatticeError> {
        let result = Self {
            lebail,
            lattice_damping,
            max_scaled_lattice_step,
            max_lattice_backtracks,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate all local and shared numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankLatticeError`] for non-finite or out-of-range controls.
    pub fn validate(&self) -> Result<(), TofMultiBankLatticeError> {
        self.lebail.validate()?;
        if !self.lattice_damping.is_finite()
            || self.lattice_damping < 0.0
            || !self.max_scaled_lattice_step.is_finite()
            || self.max_scaled_lattice_step <= 0.0
        {
            return Err(TofMultiBankLatticeError::InvalidOptions);
        }
        Ok(())
    }
}

/// One accepted shared-cell value.
#[derive(Clone, Debug, PartialEq)]
pub struct TofSharedLatticeState {
    /// Stable phase identity.
    pub phase_id: RecordId,
    /// Accepted symmetry-compatible unit cell.
    pub cell: UnitCell,
}

/// One accepted physical lattice-parameter change.
#[derive(Clone, Debug, PartialEq)]
pub struct TofLatticeParameterChange {
    /// Stable phase identity.
    pub phase_id: RecordId,
    /// Setting-aware independent parameter name.
    pub parameter_name: String,
    /// Physical value before the step.
    pub before: f64,
    /// Physical value after the step.
    pub after: f64,
    /// Change divided by the parameter scale.
    pub scaled_change: f64,
}

/// One atomically accepted local-extraction plus shared-lattice cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankLatticeIterationRecord {
    /// One-based accepted cycle index.
    pub iteration: usize,
    /// Bank-local residual records in input order.
    pub bank_metrics: Vec<ResidualEvaluation>,
    /// Aggregate residual metrics with shared lattice parameters counted once.
    pub metrics: TofMultiBankMetrics,
    /// Largest relative bank-local intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Largest absolute bank-local background-coefficient change.
    pub maximum_absolute_background_change: f64,
    /// Norm of the accepted shared lattice step in scaled coordinates.
    pub scaled_lattice_step_norm: f64,
    /// Physical shared-cell changes in deterministic phase/parameter order.
    pub lattice_parameter_changes: Vec<TofLatticeParameterChange>,
}

/// Complete last-accepted state for exact shared-lattice continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankLatticeCheckpoint {
    /// Number of atomically accepted cycles.
    pub completed_iterations: usize,
    /// Accepted bank-local intensity/background state.
    pub banks: Vec<TofMultiBankCheckpointBank>,
    /// Accepted shared cells in model order.
    pub lattice_phases: Vec<TofSharedLatticeState>,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankLatticeIterationRecord>,
}

impl TofMultiBankLatticeCheckpoint {
    /// Revalidate this continuation against the exact local/shared contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankLatticeError`] for stale identities, topology,
    /// bounds, backgrounds, cell geometry, or history.
    pub fn validate_for(
        &self,
        input: &TofMultiBankLatticeInput,
        options: &TofMultiBankLatticeOptions,
    ) -> Result<(), TofMultiBankLatticeError> {
        input.validate()?;
        options.validate()?;
        if self.completed_iterations != self.history.len()
            || self.completed_iterations > options.lebail.cycles
            || self.banks.len() != input.multibank.banks.len()
            || self.lattice_phases.len() != input.lattice_phases.len()
        {
            return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                "shared TOF lattice checkpoint counts differ from the request",
            ));
        }
        for (saved, model) in self.lattice_phases.iter().zip(&input.lattice_phases) {
            if &saved.phase_id != model.phase_id() {
                return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                    "shared TOF lattice checkpoint phase order changed",
                ));
            }
            model.validate_cell(saved.cell)?;
        }
        for (saved, original) in self.banks.iter().zip(&input.multibank.banks) {
            validate_checkpoint_bank(saved, original, input, &self.lattice_phases)?;
        }
        if self.history.iter().enumerate().any(|(index, record)| {
            record.iteration != index + 1
                || record.bank_metrics.len() != input.multibank.banks.len()
        }) {
            return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                "shared TOF lattice checkpoint history is not contiguous and bank-aligned",
            ));
        }
        Ok(())
    }
}

/// Complete result from shared analytical multi-bank TOF lattice refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankLatticeResult {
    /// Final bank-local display states and fused derivative calculations.
    pub banks: Vec<TofMultiBankResultBank>,
    /// Final shared phase cells in model order.
    pub lattice_phases: Vec<TofSharedLatticeState>,
    /// Final aggregate residual metrics.
    pub metrics: TofMultiBankMetrics,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankLatticeIterationRecord>,
    /// Stable bounded-runtime termination category.
    pub termination_reason: TerminationReason,
    /// Complete last accepted state.
    pub checkpoint: TofMultiBankLatticeCheckpoint,
}

/// Refine shared phase cells against the summed objective over all TOF banks.
///
/// # Errors
///
/// Returns [`TofMultiBankLatticeError`] for invalid contracts, geometry,
/// calculation, linear solve, or runtime state.
pub fn refine_tof_multibank_lattice(
    input: &TofMultiBankLatticeInput,
    options: &TofMultiBankLatticeOptions,
) -> Result<TofMultiBankLatticeResult, TofMultiBankLatticeError> {
    input.validate()?;
    options.validate()?;
    let evaluations_per_cycle = options
        .max_lattice_backtracks
        .checked_add(4)
        .ok_or(TofMultiBankLatticeError::AllocationOverflow)?;
    let max_evaluations = options
        .lebail
        .cycles
        .checked_mul(evaluations_per_cycle)
        .ok_or(TofMultiBankLatticeError::AllocationOverflow)?;
    let limits = RefinementLimits::new(options.lebail.cycles, max_evaluations, None, 1)?;
    let mut runtime = RefinementRuntime::new(limits, None)?;
    refine_tof_multibank_lattice_with_runtime(input, options, None, &mut runtime)
}

/// Refine shared cells with host cancellation/events and optional continuation.
///
/// Candidate intensities, backgrounds, and cells are accepted only after the
/// full joint cycle finishes; a stop during backtracking discards that entire
/// cycle.
///
/// # Errors
///
/// Returns [`TofMultiBankLatticeError`] for invalid contracts, geometry,
/// calculation, linear solve, or non-normal runtime failures.
#[allow(clippy::too_many_lines)]
pub fn refine_tof_multibank_lattice_with_runtime(
    input: &TofMultiBankLatticeInput,
    options: &TofMultiBankLatticeOptions,
    checkpoint: Option<&TofMultiBankLatticeCheckpoint>,
    runtime: &mut RefinementRuntime<TofMultiBankLatticeCheckpoint>,
) -> Result<TofMultiBankLatticeResult, TofMultiBankLatticeError> {
    input.validate()?;
    options.validate()?;
    let parameter_count = input.parameter_count()?;
    let restored = restore_state(input, options, checkpoint)?;
    let mut states = restored.states;
    let mut cells = restored.cells;
    let mut history = restored.history;
    if let Some(checkpoint) = checkpoint {
        runtime.resume_accepted(checkpoint.completed_iterations)?;
    }
    runtime.emit(
        RefinementEventKind::Start,
        "tof_multibank_lattice",
        "shared-cell multi-bank TOF Le Bail refinement started",
        vec![
            (
                "bank_count".to_owned(),
                DiagnosticValue::Unsigned(input.multibank.banks.len() as u64),
            ),
            (
                "lattice_parameter_count".to_owned(),
                DiagnosticValue::Unsigned(parameter_count as u64),
            ),
        ],
    )?;
    let mut accepted_calculations = None;
    let mut termination = TerminationReason::MaxIterations;
    for iteration in restored.first_iteration..=options.lebail.cycles {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_tof_stop(error)?;
            break;
        }
        let candidate = match prepare_multibank_cycle(
            &input.multibank,
            &states,
            &options.lebail,
            parameter_count,
            runtime,
        )? {
            MultiBankCycleOutcome::Candidate(candidate) => candidate,
            MultiBankCycleOutcome::Stopped(reason) => {
                termination = reason;
                break;
            }
        };
        let lattice = match lattice_update(input, options, &cells, candidate, runtime)? {
            LatticeUpdateOutcome::Candidate(candidate) => candidate,
            LatticeUpdateOutcome::Stopped(reason) => {
                termination = reason;
                break;
            }
        };
        states = lattice.cycle.states;
        cells = lattice.cells;
        accepted_calculations = Some(lattice.cycle.calculations.clone());
        history.push(TofMultiBankLatticeIterationRecord {
            iteration,
            bank_metrics: lattice.cycle.bank_metrics.clone(),
            metrics: lattice.cycle.metrics,
            maximum_relative_intensity_change: lattice.cycle.maximum_relative_intensity_change,
            maximum_absolute_background_change: lattice.cycle.maximum_absolute_background_change,
            scaled_lattice_step_norm: lattice.step_norm,
            lattice_parameter_changes: lattice.changes,
        });
        let accepted = checkpoint_from_state(input, &states, &cells, &history);
        runtime.accept_step(Some(&accepted))?;
        runtime.emit(
            RefinementEventKind::Iteration,
            "tof_multibank_lattice_iteration",
            "shared-cell multi-bank TOF cycle accepted",
            vec![
                (
                    "rwp".to_owned(),
                    DiagnosticValue::Float(lattice.cycle.metrics.rwp),
                ),
                (
                    "scaled_lattice_step_norm".to_owned(),
                    DiagnosticValue::Float(lattice.step_norm),
                ),
            ],
        )?;
    }
    let calculations = match accepted_calculations {
        Some(calculations) => calculations,
        None => calculate_states(&input.multibank, &states, &options.lebail)?,
    };
    let bank_metrics =
        evaluate_bank_metrics(&input.multibank, &states, &calculations, &options.lebail)?;
    let fitted = fitted_parameter_count(&states)?
        .checked_add(parameter_count)
        .ok_or(TofMultiBankLatticeError::AllocationOverflow)?;
    let metrics = aggregate_metrics(&input.multibank, &calculations, &options.lebail, fitted)?;
    let checkpoint = checkpoint_from_state(input, &states, &cells, &history);
    checkpoint.validate_for(input, options)?;
    let banks = input
        .multibank
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
    let lattice_phases = lattice_states(input, &cells);
    runtime.emit(
        RefinementEventKind::Termination,
        "tof_multibank_lattice",
        "shared-cell multi-bank TOF refinement terminated",
        vec![(
            "termination_reason".to_owned(),
            DiagnosticValue::String(termination.as_str().to_owned()),
        )],
    )?;
    Ok(TofMultiBankLatticeResult {
        banks,
        lattice_phases,
        metrics,
        history,
        termination_reason: termination,
        checkpoint,
    })
}

struct RestoredState {
    states: Vec<AcceptedBankState>,
    cells: Vec<UnitCell>,
    history: Vec<TofMultiBankLatticeIterationRecord>,
    first_iteration: usize,
}

fn restore_state(
    input: &TofMultiBankLatticeInput,
    options: &TofMultiBankLatticeOptions,
    checkpoint: Option<&TofMultiBankLatticeCheckpoint>,
) -> Result<RestoredState, TofMultiBankLatticeError> {
    let Some(checkpoint) = checkpoint else {
        let states = input
            .multibank
            .banks
            .iter()
            .map(|bank| {
                Ok(AcceptedBankState {
                    phases: initialize_intensities(&bank.input, &options.lebail)?,
                    background: bank.input.background.clone(),
                })
            })
            .collect::<Result<Vec<_>, TofMultiBankLatticeError>>()?;
        let cells = input
            .lattice_phases
            .iter()
            .map(TofSharedLatticePhase::initial_cell)
            .collect::<Vec<_>>();
        return Ok(RestoredState {
            states: apply_cells(input, &states, &cells)?,
            cells,
            history: Vec::new(),
            first_iteration: 1,
        });
    };
    checkpoint.validate_for(input, options)?;
    let cells = checkpoint
        .lattice_phases
        .iter()
        .map(|phase| phase.cell)
        .collect();
    Ok(RestoredState {
        states: checkpoint
            .banks
            .iter()
            .map(|bank| AcceptedBankState {
                phases: bank.phases.clone(),
                background: bank.background.clone(),
            })
            .collect(),
        cells,
        history: checkpoint.history.clone(),
        first_iteration: checkpoint
            .completed_iterations
            .checked_add(1)
            .ok_or(TofMultiBankLatticeError::AllocationOverflow)?,
    })
}

struct LatticeUpdate {
    cycle: MultiBankCycleCandidate,
    cells: Vec<UnitCell>,
    step_norm: f64,
    changes: Vec<TofLatticeParameterChange>,
}

enum LatticeUpdateOutcome {
    Candidate(LatticeUpdate),
    Stopped(TerminationReason),
}

#[allow(clippy::too_many_lines)]
fn lattice_update<C>(
    input: &TofMultiBankLatticeInput,
    options: &TofMultiBankLatticeOptions,
    cells: &[UnitCell],
    mut cycle: MultiBankCycleCandidate,
    runtime: &mut RefinementRuntime<C>,
) -> Result<LatticeUpdateOutcome, TofMultiBankLatticeError> {
    let packed = pack_lattice(input, cells)?;
    let (jacobian, residual) = lattice_system(input, cells, &cycle, &packed.scales, options)?;
    let normal = jacobian.transpose() * &jacobian;
    let rhs = jacobian.transpose() * &residual;
    let mut regularized = normal;
    for parameter in 0..regularized.nrows() {
        regularized[(parameter, parameter)] += options.lattice_damping;
    }
    let mut step = if let Some(solution) = regularized.lu().solve(&rhs) {
        solution
    } else {
        jacobian
            .svd(true, true)
            .solve(&residual, f64::EPSILON)
            .map_err(|_| TofMultiBankLatticeError::LinearSolve)?
    };
    if step.iter().any(|value| !value.is_finite()) {
        return Err(TofMultiBankLatticeError::LinearSolve);
    }
    for parameter in 0..step.len() {
        let current = packed.values[parameter] / packed.scales[parameter];
        let lower = (packed.lower[parameter] / packed.scales[parameter] - current)
            .max(-options.max_scaled_lattice_step);
        let upper = (packed.upper[parameter] / packed.scales[parameter] - current)
            .min(options.max_scaled_lattice_step);
        step[parameter] = step[parameter].clamp(lower, upper);
    }
    if step.norm() == 0.0 {
        return Ok(LatticeUpdateOutcome::Candidate(LatticeUpdate {
            cycle,
            cells: cells.to_vec(),
            step_norm: 0.0,
            changes: Vec::new(),
        }));
    }
    let parameter_count = input.parameter_count()?;
    let local_parameter_count = fitted_parameter_count(&cycle.states)?;
    let total_parameter_count = local_parameter_count
        .checked_add(parameter_count)
        .ok_or(TofMultiBankLatticeError::AllocationOverflow)?;
    let mut factor = 1.0;
    for _ in 0..=options.max_lattice_backtracks {
        let trial_values = packed
            .values
            .iter()
            .zip(&packed.scales)
            .zip(step.iter())
            .map(|((value, scale), step)| value + factor * scale * step)
            .collect::<Vec<_>>();
        let trial_cells = match unpack_lattice(input, &trial_values) {
            Ok(cells) => cells,
            Err(error) if recoverable_trial_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error),
        };
        let trial_states = match apply_cells(input, &cycle.states, &trial_cells) {
            Ok(states) => states,
            Err(error) if recoverable_trial_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error),
        };
        if let Err(error) = runtime.begin_evaluation() {
            return Ok(LatticeUpdateOutcome::Stopped(normal_tof_stop(error)?));
        }
        let trial_calculations =
            match calculate_states(&input.multibank, &trial_states, &options.lebail) {
                Ok(calculations) => calculations,
                Err(error) if recoverable_multibank_trial_error(&error) => {
                    factor *= 0.5;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
        let trial_metrics = aggregate_metrics(
            &input.multibank,
            &trial_calculations,
            &options.lebail,
            total_parameter_count,
        )?;
        if trial_metrics.chi_square < cycle.metrics.chi_square {
            let trial_bank_metrics = evaluate_bank_metrics(
                &input.multibank,
                &trial_states,
                &trial_calculations,
                &options.lebail,
            )?;
            let changes = lattice_changes(input, &packed.values, &trial_values, &packed.scales)?;
            cycle.states = trial_states;
            cycle.calculations = trial_calculations;
            cycle.bank_metrics = trial_bank_metrics;
            cycle.metrics = trial_metrics;
            return Ok(LatticeUpdateOutcome::Candidate(LatticeUpdate {
                cycle,
                cells: trial_cells,
                step_norm: factor * step.norm(),
                changes,
            }));
        }
        factor *= 0.5;
    }
    Ok(LatticeUpdateOutcome::Candidate(LatticeUpdate {
        cycle,
        cells: cells.to_vec(),
        step_norm: 0.0,
        changes: Vec::new(),
    }))
}

fn recoverable_trial_error(error: &TofMultiBankLatticeError) -> bool {
    matches!(
        error,
        TofMultiBankLatticeError::Lattice(_)
            | TofMultiBankLatticeError::InvalidModel(_)
            | TofMultiBankLatticeError::MultiBank(TofMultiBankError::LeBail(
                TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_)
            ))
    )
}

fn recoverable_multibank_trial_error(error: &TofMultiBankError) -> bool {
    matches!(
        error,
        TofMultiBankError::LeBail(TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_))
    )
}

struct PackedLattice {
    values: Vec<f64>,
    scales: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn pack_lattice(
    input: &TofMultiBankLatticeInput,
    cells: &[UnitCell],
) -> Result<PackedLattice, TofMultiBankLatticeError> {
    if cells.len() != input.lattice_phases.len() {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    let mut values = Vec::new();
    let mut scales = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for (phase, cell) in input.lattice_phases.iter().zip(cells) {
        let phase_values = phase.validate_cell(*cell)?;
        scales.extend(phase_values.iter().map(|value| value.abs().max(1.0)));
        values.extend(phase_values);
        lower.extend_from_slice(phase.bounds.lower());
        upper.extend_from_slice(phase.bounds.upper());
    }
    Ok(PackedLattice {
        values,
        scales,
        lower,
        upper,
    })
}

fn unpack_lattice(
    input: &TofMultiBankLatticeInput,
    values: &[f64],
) -> Result<Vec<UnitCell>, TofMultiBankLatticeError> {
    if values.len() != input.parameter_count()? {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    let mut offset = 0;
    input
        .lattice_phases
        .iter()
        .map(|phase| {
            let end = offset + phase.parameterization.parameter_names().len();
            let cell = phase.parameterization.to_cell(&values[offset..end])?;
            phase.validate_cell(cell)?;
            offset = end;
            Ok(cell)
        })
        .collect()
}

fn apply_cells(
    input: &TofMultiBankLatticeInput,
    states: &[AcceptedBankState],
    cells: &[UnitCell],
) -> Result<Vec<AcceptedBankState>, TofMultiBankLatticeError> {
    if states.len() != input.multibank.banks.len() || cells.len() != input.lattice_phases.len() {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    input
        .multibank
        .banks
        .iter()
        .zip(states)
        .map(|(bank, state)| {
            let mut phases = state.phases.clone();
            for (model, cell) in input.lattice_phases.iter().zip(cells) {
                let index = phases
                    .iter()
                    .position(|phase| phase.phase_id() == model.phase_id())
                    .ok_or(TofMultiBankLatticeError::InternalInvariant)?;
                let geometry = tof_lattice_geometry(
                    model.parameterization(),
                    *cell,
                    phases[index].hkl(),
                    bank.input.instrument,
                )?;
                phases[index] = phases[index].with_d_spacings(&geometry.d_spacing_angstrom)?;
            }
            Ok(AcceptedBankState {
                phases,
                background: state.background.clone(),
            })
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn lattice_system(
    input: &TofMultiBankLatticeInput,
    cells: &[UnitCell],
    cycle: &MultiBankCycleCandidate,
    scales: &[f64],
    options: &TofMultiBankLatticeOptions,
) -> Result<(DMatrix<f64>, DVector<f64>), TofMultiBankLatticeError> {
    let columns = input.parameter_count()?;
    if columns != scales.len() {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    let rows = input
        .multibank
        .banks
        .iter()
        .try_fold(0_usize, |count, bank| {
            let included = bank.input.pattern.mask.as_ref().map_or_else(
                || bank.input.pattern.sample_count(),
                |mask| mask.iter().filter(|value| **value).count(),
            );
            count
                .checked_add(included)
                .ok_or(TofMultiBankLatticeError::AllocationOverflow)
        })?;
    if rows < columns {
        return Err(TofMultiBankLatticeError::LinearSolve);
    }
    let mut selected_jacobian = DMatrix::zeros(rows, columns);
    let mut selected_residual = DVector::zeros(rows);
    let mut row_offset = 0;
    for ((bank, state), calculation) in input
        .multibank
        .banks
        .iter()
        .zip(&cycle.states)
        .zip(&cycle.calculations)
    {
        let samples = bank.input.pattern.sample_count();
        let mut physical = DMatrix::<f64>::zeros(samples, columns);
        let local = &calculation.accumulation.derivatives.local;
        if local.parameter_count != 2 {
            return Err(TofMultiBankLatticeError::InternalInvariant);
        }
        let mut parameter_offset = 0;
        for (model, cell) in input.lattice_phases.iter().zip(cells) {
            let phase_index = state
                .phases
                .iter()
                .position(|phase| phase.phase_id() == model.phase_id())
                .ok_or(TofMultiBankLatticeError::InternalInvariant)?;
            let phase = &state.phases[phase_index];
            let geometry = tof_lattice_geometry(
                model.parameterization(),
                *cell,
                phase.hkl(),
                bank.input.instrument,
            )?;
            let phase_columns = geometry.parameter_names.len();
            let reflection_offset = calculation.phase_offsets[phase_index];
            for phase_reflection in 0..phase.reflection_ids().len() {
                let reflection = reflection_offset + phase_reflection;
                let begin = local.offsets[reflection];
                let end = local.offsets[reflection + 1];
                let start = local.starts[reflection];
                for parameter in 0..phase_columns {
                    let chain = geometry.d_d_spacing_d_parameters
                        [phase_reflection * phase_columns + parameter]
                        * scales[parameter_offset + parameter];
                    for active in begin..end {
                        physical[(start + active - begin, parameter_offset + parameter)] +=
                            local.values[active * local.parameter_count + 1] * chain;
                    }
                }
            }
            parameter_offset += phase_columns;
        }
        if parameter_offset != columns {
            return Err(TofMultiBankLatticeError::InternalInvariant);
        }
        let observed = bank
            .input
            .pattern
            .observed_y
            .as_ref()
            .ok_or(TofLeBailError::MissingObservations)?;
        for sample in 0..samples {
            if bank
                .input
                .pattern
                .mask
                .as_ref()
                .is_some_and(|mask| !mask[sample])
            {
                continue;
            }
            let weight = if options.lebail.use_uncertainty {
                bank.input
                    .pattern
                    .uncertainty
                    .as_ref()
                    .map_or(1.0, |sigma| sigma[sample].recip())
            } else {
                1.0
            };
            selected_residual[row_offset] = (observed[sample] - calculation.y[sample]) * weight;
            for parameter in 0..columns {
                selected_jacobian[(row_offset, parameter)] = physical[(sample, parameter)] * weight;
            }
            row_offset += 1;
        }
    }
    if row_offset != rows {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    Ok((selected_jacobian, selected_residual))
}

fn lattice_changes(
    input: &TofMultiBankLatticeInput,
    before: &[f64],
    after: &[f64],
    scales: &[f64],
) -> Result<Vec<TofLatticeParameterChange>, TofMultiBankLatticeError> {
    if before.len() != after.len() || before.len() != scales.len() {
        return Err(TofMultiBankLatticeError::InternalInvariant);
    }
    let mut changes = Vec::new();
    let mut offset = 0;
    for phase in &input.lattice_phases {
        for name in phase.parameterization.parameter_names() {
            if before[offset].to_bits() != after[offset].to_bits() {
                changes.push(TofLatticeParameterChange {
                    phase_id: phase.phase_id.clone(),
                    parameter_name: name.clone(),
                    before: before[offset],
                    after: after[offset],
                    scaled_change: (after[offset] - before[offset]) / scales[offset],
                });
            }
            offset += 1;
        }
    }
    Ok(changes)
}

fn checkpoint_from_state(
    input: &TofMultiBankLatticeInput,
    states: &[AcceptedBankState],
    cells: &[UnitCell],
    history: &[TofMultiBankLatticeIterationRecord],
) -> TofMultiBankLatticeCheckpoint {
    TofMultiBankLatticeCheckpoint {
        completed_iterations: history.len(),
        banks: input
            .multibank
            .banks
            .iter()
            .zip(states)
            .map(|(bank, state)| TofMultiBankCheckpointBank {
                bank_id: bank.bank_id.clone(),
                phases: state.phases.clone(),
                background: state.background.clone(),
            })
            .collect(),
        lattice_phases: lattice_states(input, cells),
        history: history.to_vec(),
    }
}

fn lattice_states(
    input: &TofMultiBankLatticeInput,
    cells: &[UnitCell],
) -> Vec<TofSharedLatticeState> {
    input
        .lattice_phases
        .iter()
        .zip(cells)
        .map(|(phase, cell)| TofSharedLatticeState {
            phase_id: phase.phase_id.clone(),
            cell: *cell,
        })
        .collect()
}

fn validate_checkpoint_bank(
    saved: &TofMultiBankCheckpointBank,
    original: &crate::TofLeBailBank,
    input: &TofMultiBankLatticeInput,
    cells: &[TofSharedLatticeState],
) -> Result<(), TofMultiBankLatticeError> {
    if saved.bank_id != original.bank_id || saved.phases.len() != original.input.phases.len() {
        return Err(TofMultiBankLatticeError::InvalidCheckpoint(
            "shared TOF lattice checkpoint bank identity or phase count changed",
        ));
    }
    for (saved_phase, original_phase) in saved.phases.iter().zip(&original.input.phases) {
        saved_phase.validate()?;
        if saved_phase.phase_id() != original_phase.phase_id()
            || saved_phase.name() != original_phase.name()
            || saved_phase.reflection_ids() != original_phase.reflection_ids()
            || saved_phase.hkl() != original_phase.hkl()
            || saved_phase.scale().to_bits() != original_phase.scale().to_bits()
        {
            return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                "shared TOF lattice checkpoint bank topology changed",
            ));
        }
        if let Some((model, cell)) = input
            .lattice_phases
            .iter()
            .zip(cells)
            .find(|(model, _)| model.phase_id() == saved_phase.phase_id())
        {
            let expected = tof_lattice_geometry(
                model.parameterization(),
                cell.cell,
                saved_phase.hkl(),
                original.input.instrument,
            )?;
            if saved_phase.d_spacing_angstrom() != expected.d_spacing_angstrom {
                return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                    "shared TOF lattice checkpoint d-spacings disagree with its cell",
                ));
            }
        } else if saved_phase.d_spacing_angstrom() != original_phase.d_spacing_angstrom() {
            return Err(TofMultiBankLatticeError::InvalidCheckpoint(
                "fixed TOF phase d-spacings changed in a lattice checkpoint",
            ));
        }
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
        _ => Err(TofMultiBankLatticeError::InvalidCheckpoint(
            "shared TOF lattice checkpoint background contract changed",
        )),
    }
}

/// Invalid shared-lattice TOF request or numerical state.
#[derive(Debug)]
pub enum TofMultiBankLatticeError {
    /// Bank-local or fixed-cell multi-bank contract failed.
    MultiBank(TofMultiBankError),
    /// Setting-aware lattice geometry failed.
    Lattice(LatticeError),
    /// Shared phase/cell selection is inconsistent.
    InvalidModel(&'static str),
    /// Solver controls are invalid.
    InvalidOptions,
    /// Weighted lattice normal equations could not be solved finitely.
    LinearSolve,
    /// A continuation state disagrees with the immutable request contract.
    InvalidCheckpoint(&'static str),
    /// Checked allocation arithmetic overflowed.
    AllocationOverflow,
    /// Bounded runtime or event/checkpoint delivery failed.
    Runtime(RuntimeError),
    /// Internal array alignment failed after validated construction.
    InternalInvariant,
}

impl Display for TofMultiBankLatticeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultiBank(error) => Display::fmt(error, formatter),
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::InvalidModel(message) | Self::InvalidCheckpoint(message) => {
                formatter.write_str(message)
            }
            Self::InvalidOptions => formatter.write_str("invalid shared TOF lattice options"),
            Self::LinearSolve => formatter.write_str("shared TOF lattice linear solve failed"),
            Self::AllocationOverflow => {
                formatter.write_str("shared TOF lattice allocation overflow")
            }
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::InternalInvariant => {
                formatter.write_str("shared TOF lattice internal array invariant failed")
            }
        }
    }
}

impl Error for TofMultiBankLatticeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MultiBank(error) => Some(error),
            Self::Lattice(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TofMultiBankError> for TofMultiBankLatticeError {
    fn from(value: TofMultiBankError) -> Self {
        Self::MultiBank(value)
    }
}

impl From<TofLeBailError> for TofMultiBankLatticeError {
    fn from(value: TofLeBailError) -> Self {
        Self::MultiBank(TofMultiBankError::LeBail(value))
    }
}

impl From<LatticeError> for TofMultiBankLatticeError {
    fn from(value: LatticeError) -> Self {
        Self::Lattice(value)
    }
}

impl From<RuntimeError> for TofMultiBankLatticeError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
