//! Joint shared-cell and bank-local instrument refinement for multi-bank TOF Le Bail.

use std::error::Error;
use std::fmt::{Display, Formatter};

use nalgebra::{DMatrix, DVector};
use phasesmith_core::{TofInstrument, TofInstrumentParameter, TofProfileParameters};
use phasesmith_crystallography::UnitCell;
use phasesmith_model::RecordId;

use crate::tof_lebail::{initialize_intensities, normal_tof_stop};
use crate::tof_multibank::{
    AcceptedBankState, MultiBankCycleCandidate, MultiBankCycleOutcome, aggregate_metrics,
    calculate_states, evaluate_bank_metrics, fitted_parameter_count, prepare_multibank_cycle,
    reflection_intensities,
};
use crate::tof_multibank_instrument::{
    PackedInstruments, instrument_system, live_input, pack_instruments, unpack_instruments,
};
use crate::tof_multibank_lattice::{
    PackedLattice, apply_cells, lattice_changes, lattice_system, pack_lattice, unpack_lattice,
};
use crate::{
    DiagnosticValue, RefinementEventKind, RefinementLimits, RefinementRuntime, ResidualEvaluation,
    RuntimeError, TerminationReason, TofBankInstrumentModel, TofBankInstrumentState,
    TofChebyshevBackground, TofInstrumentParameterChange, TofLatticeParameterChange,
    TofLeBailError, TofLeBailOptions, TofLeBailPhase, TofMultiBankError,
    TofMultiBankInstrumentError, TofMultiBankInstrumentInput, TofMultiBankLatticeError,
    TofMultiBankLatticeInput, TofMultiBankMetrics, TofMultiBankResultBank, TofSharedLatticeState,
    tof_lattice_geometry,
};

/// Shared-cell model plus selected bank-local instrument coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryInput {
    /// Validated shared-cell multi-bank model.
    pub lattice: TofMultiBankLatticeInput,
    /// Selected bank-local coefficients in deterministic packing order.
    pub instrument_models: Vec<TofBankInstrumentModel>,
}

impl TofMultiBankGeometryInput {
    /// Validate both numerical models against the exact same bank request.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankGeometryError`] for invalid lattice, instrument,
    /// bank, or bound contracts.
    pub fn validate(&self) -> Result<(), TofMultiBankGeometryError> {
        self.lattice.validate()?;
        self.instrument_input().validate()?;
        Ok(())
    }

    fn instrument_input(&self) -> TofMultiBankInstrumentInput {
        TofMultiBankInstrumentInput {
            multibank: self.lattice.multibank.clone(),
            instrument_models: self.instrument_models.clone(),
        }
    }

    fn parameter_count(&self) -> Result<usize, TofMultiBankGeometryError> {
        self.lattice
            .parameter_count()?
            .checked_add(self.instrument_input().parameter_count()?)
            .ok_or(TofMultiBankGeometryError::AllocationOverflow)
    }
}

/// Controls for one genuinely joint lattice/instrument step.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryOptions {
    /// Existing TOF redistribution, support, weighting, and execution controls.
    pub lebail: TofLeBailOptions,
    /// Non-negative diagonal regularization in scaled geometry coordinates.
    pub geometry_damping: f64,
    /// Maximum absolute geometry step in scaled coordinates.
    pub max_scaled_geometry_step: f64,
    /// Number of objective backtracking halvings after the initial trial.
    pub max_geometry_backtracks: usize,
    /// Absolute weighted-column correlation reported as unresolved.
    pub unresolved_correlation: f64,
}

impl TofMultiBankGeometryOptions {
    /// Construct validated joint geometry controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankGeometryError::InvalidOptions`] for invalid fields.
    pub fn new(
        lebail: TofLeBailOptions,
        geometry_damping: f64,
        max_scaled_geometry_step: f64,
        max_geometry_backtracks: usize,
        unresolved_correlation: f64,
    ) -> Result<Self, TofMultiBankGeometryError> {
        let result = Self {
            lebail,
            geometry_damping,
            max_scaled_geometry_step,
            max_geometry_backtracks,
            unresolved_correlation,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate numerical controls.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankGeometryError::InvalidOptions`] for invalid fields.
    pub fn validate(&self) -> Result<(), TofMultiBankGeometryError> {
        self.lebail.validate()?;
        if !self.geometry_damping.is_finite()
            || self.geometry_damping < 0.0
            || !self.max_scaled_geometry_step.is_finite()
            || self.max_scaled_geometry_step <= 0.0
            || !self.unresolved_correlation.is_finite()
            || !(0.0..=1.0).contains(&self.unresolved_correlation)
        {
            return Err(TofMultiBankGeometryError::InvalidOptions);
        }
        Ok(())
    }
}

/// Stable identity of one joint physical geometry column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TofGeometryParameterKey {
    /// Shared symmetry-independent cell variable.
    Lattice {
        /// Stable phase identity.
        phase_id: RecordId,
        /// Setting-aware parameter name.
        parameter_name: String,
    },
    /// Bank-local calibration/profile coefficient.
    Instrument {
        /// Stable bank identity.
        bank_id: RecordId,
        /// Selected coefficient.
        parameter: TofInstrumentParameter,
    },
}

/// One unresolved lattice/instrument weighted-column pair.
#[derive(Clone, Debug, PartialEq)]
pub struct TofGeometryCorrelation {
    /// First physical parameter.
    pub left: TofGeometryParameterKey,
    /// Second physical parameter.
    pub right: TofGeometryParameterKey,
    /// Normalized weighted-column dot product.
    pub correlation: f64,
}

/// Identifiability diagnostics for the final joint geometry system.
#[derive(Clone, Debug, PartialEq)]
pub struct TofGeometryDiagnostics {
    /// Total shared-cell plus local-instrument parameter count.
    pub parameter_count: usize,
    /// Numerical rank of the weighted joint Jacobian.
    pub jacobian_rank: usize,
    /// Largest finite absolute correlation between nonzero columns.
    pub maximum_absolute_correlation: Option<f64>,
    /// Named pairs at or above the configured unresolved threshold.
    pub unresolved_correlations: Vec<TofGeometryCorrelation>,
}

/// One atomically accepted extraction plus joint geometry cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryIterationRecord {
    /// One-based accepted cycle index.
    pub iteration: usize,
    /// Bank-local residual records in input order.
    pub bank_metrics: Vec<ResidualEvaluation>,
    /// Aggregate metrics with all geometry parameters counted once.
    pub metrics: TofMultiBankMetrics,
    /// Largest relative intensity change.
    pub maximum_relative_intensity_change: f64,
    /// Largest absolute background-coefficient change.
    pub maximum_absolute_background_change: f64,
    /// Norm of the accepted joint geometry step in scaled coordinates.
    pub scaled_geometry_step_norm: f64,
    /// Accepted physical shared-cell changes.
    pub lattice_parameter_changes: Vec<TofLatticeParameterChange>,
    /// Accepted physical bank-local instrument changes.
    pub instrument_parameter_changes: Vec<TofInstrumentParameterChange>,
}

/// One bank's complete accepted joint-geometry checkpoint state.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryCheckpointBank {
    /// Stable bank identity.
    pub bank_id: RecordId,
    /// Accepted bank-local instrument.
    pub instrument: TofInstrument,
    /// Accepted phase geometry and extracted intensities.
    pub phases: Vec<TofLeBailPhase>,
    /// Accepted refinable background, when configured.
    pub background: Option<TofChebyshevBackground>,
}

/// Complete last-accepted joint geometry state.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryCheckpoint {
    /// Number of atomically accepted cycles.
    pub completed_iterations: usize,
    /// Accepted bank-local state in input order.
    pub banks: Vec<TofMultiBankGeometryCheckpointBank>,
    /// Accepted shared cells in model order.
    pub lattice_phases: Vec<TofSharedLatticeState>,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankGeometryIterationRecord>,
}

/// Complete joint shared-cell plus local-instrument result.
#[derive(Clone, Debug, PartialEq)]
pub struct TofMultiBankGeometryResult {
    /// Final bank-local display states and fused calculations.
    pub banks: Vec<TofMultiBankResultBank>,
    /// Accepted shared phase cells.
    pub lattice_phases: Vec<TofSharedLatticeState>,
    /// Accepted instruments in bank input order.
    pub instruments: Vec<TofBankInstrumentState>,
    /// Final aggregate residual metrics.
    pub metrics: TofMultiBankMetrics,
    /// Final joint rank/correlation diagnostics.
    pub diagnostics: TofGeometryDiagnostics,
    /// Complete deterministic accepted history.
    pub history: Vec<TofMultiBankGeometryIterationRecord>,
    /// Stable bounded-runtime termination category.
    pub termination_reason: TerminationReason,
    /// Complete last accepted state.
    pub checkpoint: TofMultiBankGeometryCheckpoint,
}

/// Refine shared cells and selected bank-local instruments in one system.
///
/// # Errors
///
/// Returns [`TofMultiBankGeometryError`] for invalid contracts, geometry,
/// calculation, linear solve, or runtime state.
pub fn refine_tof_multibank_geometry(
    input: &TofMultiBankGeometryInput,
    options: &TofMultiBankGeometryOptions,
) -> Result<TofMultiBankGeometryResult, TofMultiBankGeometryError> {
    input.validate()?;
    options.validate()?;
    let evaluations_per_cycle = options
        .max_geometry_backtracks
        .checked_add(4)
        .ok_or(TofMultiBankGeometryError::AllocationOverflow)?;
    let max_evaluations = options
        .lebail
        .cycles
        .checked_mul(evaluations_per_cycle)
        .ok_or(TofMultiBankGeometryError::AllocationOverflow)?;
    let limits = RefinementLimits::new(options.lebail.cycles, max_evaluations, None, 1)?;
    let mut runtime = RefinementRuntime::new(limits, None)?;
    refine_tof_multibank_geometry_with_runtime(input, options, None, &mut runtime)
}

/// Refine joint geometry with host cancellation/events and continuation.
///
/// Candidate local extraction, cells, and instruments publish only after the
/// complete joint trial succeeds. Stops discard the entire in-progress cycle.
///
/// # Errors
///
/// Returns [`TofMultiBankGeometryError`] for invalid contracts, geometry,
/// calculation, linear solve, or non-normal runtime failures.
#[allow(clippy::too_many_lines)]
pub fn refine_tof_multibank_geometry_with_runtime(
    input: &TofMultiBankGeometryInput,
    options: &TofMultiBankGeometryOptions,
    checkpoint: Option<&TofMultiBankGeometryCheckpoint>,
    runtime: &mut RefinementRuntime<TofMultiBankGeometryCheckpoint>,
) -> Result<TofMultiBankGeometryResult, TofMultiBankGeometryError> {
    input.validate()?;
    options.validate()?;
    let parameter_count = input.parameter_count()?;
    let restored = restore_state(input, options, checkpoint)?;
    let mut states = restored.states;
    let mut cells = restored.cells;
    let mut instruments = restored.instruments;
    let mut history = restored.history;
    if let Some(checkpoint) = checkpoint {
        runtime.resume_accepted(checkpoint.completed_iterations)?;
    }
    runtime.emit(
        RefinementEventKind::Start,
        "tof_multibank_geometry",
        "joint shared-cell and bank-local TOF instrument refinement started",
        vec![(
            "geometry_parameter_count".to_owned(),
            DiagnosticValue::Unsigned(parameter_count as u64),
        )],
    )?;
    let mut accepted_calculations = None;
    let mut termination = TerminationReason::MaxIterations;
    for iteration in restored.first_iteration..=options.lebail.cycles {
        if let Err(error) = runtime.begin_iteration(iteration) {
            termination = normal_tof_stop(error)?;
            break;
        }
        let instrument_input = input.instrument_input();
        let live = live_input(&instrument_input, &instruments)?;
        let candidate = match prepare_multibank_cycle(
            &live,
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
        let updated =
            match geometry_update(input, options, &cells, &instruments, candidate, runtime)? {
                GeometryUpdateOutcome::Candidate(candidate) => candidate,
                GeometryUpdateOutcome::Stopped(reason) => {
                    termination = reason;
                    break;
                }
            };
        states = updated.cycle.states;
        cells = updated.cells;
        instruments = updated.instruments;
        accepted_calculations = Some(updated.cycle.calculations.clone());
        history.push(TofMultiBankGeometryIterationRecord {
            iteration,
            bank_metrics: updated.cycle.bank_metrics.clone(),
            metrics: updated.cycle.metrics,
            maximum_relative_intensity_change: updated.cycle.maximum_relative_intensity_change,
            maximum_absolute_background_change: updated.cycle.maximum_absolute_background_change,
            scaled_geometry_step_norm: updated.step_norm,
            lattice_parameter_changes: updated.lattice_changes,
            instrument_parameter_changes: updated.instrument_changes,
        });
        let accepted = checkpoint_from_state(input, &states, &cells, &instruments, &history);
        runtime.accept_step(Some(&accepted))?;
        runtime.emit(
            RefinementEventKind::Iteration,
            "tof_multibank_geometry_iteration",
            "joint multi-bank TOF geometry cycle accepted",
            vec![
                (
                    "rwp".to_owned(),
                    DiagnosticValue::Float(updated.cycle.metrics.rwp),
                ),
                (
                    "scaled_geometry_step_norm".to_owned(),
                    DiagnosticValue::Float(updated.step_norm),
                ),
            ],
        )?;
    }
    let instrument_input = input.instrument_input();
    let live = live_input(&instrument_input, &instruments)?;
    let calculations = match accepted_calculations {
        Some(calculations) => calculations,
        None => calculate_states(&live, &states, &options.lebail)?,
    };
    let bank_metrics = evaluate_bank_metrics(&live, &states, &calculations, &options.lebail)?;
    let fitted = fitted_parameter_count(&states)?
        .checked_add(parameter_count)
        .ok_or(TofMultiBankGeometryError::AllocationOverflow)?;
    let metrics = aggregate_metrics(&live, &calculations, &options.lebail, fitted)?;
    let packed = pack_geometry(input, &cells, &instruments)?;
    let final_cycle = cycle_view(
        states.clone(),
        calculations.clone(),
        bank_metrics.clone(),
        metrics,
    );
    let (jacobian, _) = geometry_system(input, &live, &cells, &final_cycle, &packed, options)?;
    let diagnostics = geometry_diagnostics(input, &jacobian, options.unresolved_correlation)?;
    let checkpoint = checkpoint_from_state(input, &states, &cells, &instruments, &history);
    checkpoint.validate_for(input, options)?;
    let banks = live
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
    let instruments = instrument_states(input, &instruments);
    runtime.emit(
        RefinementEventKind::Termination,
        "tof_multibank_geometry",
        "joint multi-bank TOF geometry refinement terminated",
        vec![
            (
                "termination_reason".to_owned(),
                DiagnosticValue::String(termination.as_str().to_owned()),
            ),
            (
                "jacobian_rank".to_owned(),
                DiagnosticValue::Unsigned(diagnostics.jacobian_rank as u64),
            ),
        ],
    )?;
    Ok(TofMultiBankGeometryResult {
        banks,
        lattice_phases,
        instruments,
        metrics,
        diagnostics,
        history,
        termination_reason: termination,
        checkpoint,
    })
}

fn cycle_view(
    states: Vec<AcceptedBankState>,
    calculations: Vec<crate::TofLeBailCalculation>,
    bank_metrics: Vec<ResidualEvaluation>,
    metrics: TofMultiBankMetrics,
) -> MultiBankCycleCandidate {
    MultiBankCycleCandidate {
        states,
        calculations,
        bank_metrics,
        metrics,
        maximum_relative_intensity_change: 0.0,
        maximum_absolute_background_change: 0.0,
    }
}

struct RestoredState {
    states: Vec<AcceptedBankState>,
    cells: Vec<UnitCell>,
    instruments: Vec<TofInstrument>,
    history: Vec<TofMultiBankGeometryIterationRecord>,
    first_iteration: usize,
}

fn restore_state(
    input: &TofMultiBankGeometryInput,
    options: &TofMultiBankGeometryOptions,
    checkpoint: Option<&TofMultiBankGeometryCheckpoint>,
) -> Result<RestoredState, TofMultiBankGeometryError> {
    let Some(checkpoint) = checkpoint else {
        let states = input
            .lattice
            .multibank
            .banks
            .iter()
            .map(|bank| {
                Ok(AcceptedBankState {
                    phases: initialize_intensities(&bank.input, &options.lebail)?,
                    background: bank.input.background.clone(),
                })
            })
            .collect::<Result<Vec<_>, TofMultiBankGeometryError>>()?;
        let cells = input
            .lattice
            .lattice_phases
            .iter()
            .map(crate::TofSharedLatticePhase::initial_cell)
            .collect::<Vec<_>>();
        return Ok(RestoredState {
            states: apply_cells(&input.lattice, &states, &cells)?,
            cells,
            instruments: input
                .lattice
                .multibank
                .banks
                .iter()
                .map(|bank| bank.input.instrument)
                .collect(),
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
        cells: checkpoint
            .lattice_phases
            .iter()
            .map(|phase| phase.cell)
            .collect(),
        instruments: checkpoint
            .banks
            .iter()
            .map(|bank| bank.instrument)
            .collect(),
        history: checkpoint.history.clone(),
        first_iteration: checkpoint
            .completed_iterations
            .checked_add(1)
            .ok_or(TofMultiBankGeometryError::AllocationOverflow)?,
    })
}

struct PackedGeometry {
    lattice: PackedLattice,
    instrument: PackedInstruments,
    values: Vec<f64>,
    scales: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn pack_geometry(
    input: &TofMultiBankGeometryInput,
    cells: &[UnitCell],
    instruments: &[TofInstrument],
) -> Result<PackedGeometry, TofMultiBankGeometryError> {
    let lattice = pack_lattice(&input.lattice, cells)?;
    let instrument_input = input.instrument_input();
    let instrument = pack_instruments(&instrument_input, instruments)?;
    let mut values = lattice.values.clone();
    values.extend_from_slice(&instrument.values);
    let mut scales = lattice.scales.clone();
    scales.extend_from_slice(&instrument.scales);
    let mut lower = lattice.lower.clone();
    lower.extend_from_slice(&instrument.lower);
    let mut upper = lattice.upper.clone();
    upper.extend_from_slice(&instrument.upper);
    if values.len() != input.parameter_count()? {
        return Err(TofMultiBankGeometryError::InternalInvariant);
    }
    Ok(PackedGeometry {
        lattice,
        instrument,
        values,
        scales,
        lower,
        upper,
    })
}

struct GeometryUpdate {
    cycle: MultiBankCycleCandidate,
    cells: Vec<UnitCell>,
    instruments: Vec<TofInstrument>,
    step_norm: f64,
    lattice_changes: Vec<TofLatticeParameterChange>,
    instrument_changes: Vec<TofInstrumentParameterChange>,
}

enum GeometryUpdateOutcome {
    Candidate(Box<GeometryUpdate>),
    Stopped(TerminationReason),
}

#[allow(clippy::too_many_lines)]
fn geometry_update<C>(
    input: &TofMultiBankGeometryInput,
    options: &TofMultiBankGeometryOptions,
    cells: &[UnitCell],
    instruments: &[TofInstrument],
    mut cycle: MultiBankCycleCandidate,
    runtime: &mut RefinementRuntime<C>,
) -> Result<GeometryUpdateOutcome, TofMultiBankGeometryError> {
    let packed = pack_geometry(input, cells, instruments)?;
    let instrument_input = input.instrument_input();
    let live = live_input(&instrument_input, instruments)?;
    let (jacobian, residual) = geometry_system(input, &live, cells, &cycle, &packed, options)?;
    let normal = jacobian.transpose() * &jacobian;
    let rhs = jacobian.transpose() * &residual;
    let mut regularized = normal;
    for parameter in 0..regularized.nrows() {
        regularized[(parameter, parameter)] += options.geometry_damping;
    }
    let mut step = if let Some(solution) = regularized.lu().solve(&rhs) {
        solution
    } else {
        jacobian
            .svd(true, true)
            .solve(&residual, f64::EPSILON)
            .map_err(|_| TofMultiBankGeometryError::LinearSolve)?
    };
    if step.iter().any(|value| !value.is_finite()) {
        return Err(TofMultiBankGeometryError::LinearSolve);
    }
    for parameter in 0..step.len() {
        let current = packed.values[parameter] / packed.scales[parameter];
        let lower = (packed.lower[parameter] / packed.scales[parameter] - current)
            .max(-options.max_scaled_geometry_step);
        let upper = (packed.upper[parameter] / packed.scales[parameter] - current)
            .min(options.max_scaled_geometry_step);
        step[parameter] = step[parameter].clamp(lower, upper);
    }
    if step.norm() == 0.0 {
        return Ok(GeometryUpdateOutcome::Candidate(Box::new(GeometryUpdate {
            cycle,
            cells: cells.to_vec(),
            instruments: instruments.to_vec(),
            step_norm: 0.0,
            lattice_changes: Vec::new(),
            instrument_changes: Vec::new(),
        })));
    }
    let parameter_count = input.parameter_count()?;
    let local_parameter_count = fitted_parameter_count(&cycle.states)?;
    let total_parameter_count = local_parameter_count
        .checked_add(parameter_count)
        .ok_or(TofMultiBankGeometryError::AllocationOverflow)?;
    let lattice_count = packed.lattice.values.len();
    let mut factor = 1.0;
    for _ in 0..=options.max_geometry_backtracks {
        let trial_values = packed
            .values
            .iter()
            .zip(&packed.scales)
            .zip(step.iter())
            .map(|((value, scale), step)| value + factor * scale * step)
            .collect::<Vec<_>>();
        let trial_cells = match unpack_lattice(&input.lattice, &trial_values[..lattice_count]) {
            Ok(cells) => cells,
            Err(error) if recoverable_lattice_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let trial_instruments = match unpack_instruments(
            &instrument_input,
            instruments,
            &trial_values[lattice_count..],
        ) {
            Ok(instruments) => instruments,
            Err(error) if recoverable_instrument_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let trial_live = match live_input(&instrument_input, &trial_instruments) {
            Ok(input) => input,
            Err(error) if recoverable_instrument_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let trial_lattice = TofMultiBankLatticeInput {
            multibank: trial_live.clone(),
            lattice_phases: input.lattice.lattice_phases.clone(),
        };
        let trial_states = match apply_cells(&trial_lattice, &cycle.states, &trial_cells) {
            Ok(states) => states,
            Err(error) if recoverable_lattice_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        if let Err(error) = runtime.begin_evaluation() {
            return Ok(GeometryUpdateOutcome::Stopped(normal_tof_stop(error)?));
        }
        let trial_calculations = match calculate_states(&trial_live, &trial_states, &options.lebail)
        {
            Ok(calculations) => calculations,
            Err(error) if recoverable_multibank_error(&error) => {
                factor *= 0.5;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let trial_metrics = aggregate_metrics(
            &trial_live,
            &trial_calculations,
            &options.lebail,
            total_parameter_count,
        )?;
        if trial_metrics.chi_square < cycle.metrics.chi_square {
            let trial_bank_metrics = evaluate_bank_metrics(
                &trial_live,
                &trial_states,
                &trial_calculations,
                &options.lebail,
            )?;
            let lattice_parameter_changes = lattice_changes(
                &input.lattice,
                &packed.lattice.values,
                &trial_values[..lattice_count],
                &packed.lattice.scales,
            )?;
            let instrument_parameter_changes = geometry_instrument_changes(
                input,
                &packed.instrument,
                &trial_values[lattice_count..],
            )?;
            cycle.states = trial_states;
            cycle.calculations = trial_calculations;
            cycle.bank_metrics = trial_bank_metrics;
            cycle.metrics = trial_metrics;
            return Ok(GeometryUpdateOutcome::Candidate(Box::new(GeometryUpdate {
                cycle,
                cells: trial_cells,
                instruments: trial_instruments,
                step_norm: factor * step.norm(),
                lattice_changes: lattice_parameter_changes,
                instrument_changes: instrument_parameter_changes,
            })));
        }
        factor *= 0.5;
    }
    Ok(GeometryUpdateOutcome::Candidate(Box::new(GeometryUpdate {
        cycle,
        cells: cells.to_vec(),
        instruments: instruments.to_vec(),
        step_norm: 0.0,
        lattice_changes: Vec::new(),
        instrument_changes: Vec::new(),
    })))
}

fn geometry_system(
    input: &TofMultiBankGeometryInput,
    live: &crate::TofMultiBankInput,
    cells: &[UnitCell],
    cycle: &MultiBankCycleCandidate,
    packed: &PackedGeometry,
    options: &TofMultiBankGeometryOptions,
) -> Result<(DMatrix<f64>, DVector<f64>), TofMultiBankGeometryError> {
    let live_lattice = TofMultiBankLatticeInput {
        multibank: live.clone(),
        lattice_phases: input.lattice.lattice_phases.clone(),
    };
    let (lattice, lattice_residual) = lattice_system(
        &live_lattice,
        cells,
        cycle,
        &packed.lattice.scales,
        &options.lebail,
    )?;
    let instrument_input = TofMultiBankInstrumentInput {
        multibank: live.clone(),
        instrument_models: input.instrument_models.clone(),
    };
    let (instrument, instrument_residual) = instrument_system(
        &instrument_input,
        live,
        &cycle.calculations,
        &packed.instrument.scales,
        &options.lebail,
    )?;
    if lattice.nrows() != instrument.nrows()
        || lattice_residual.len() != instrument_residual.len()
        || lattice_residual
            .iter()
            .zip(instrument_residual.iter())
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(TofMultiBankGeometryError::InternalInvariant);
    }
    let mut joint = DMatrix::zeros(lattice.nrows(), lattice.ncols() + instrument.ncols());
    joint.view_mut((0, 0), lattice.shape()).copy_from(&lattice);
    joint
        .view_mut((0, lattice.ncols()), instrument.shape())
        .copy_from(&instrument);
    Ok((joint, lattice_residual))
}

fn geometry_instrument_changes(
    input: &TofMultiBankGeometryInput,
    packed: &PackedInstruments,
    after: &[f64],
) -> Result<Vec<TofInstrumentParameterChange>, TofMultiBankGeometryError> {
    if packed.values.len() != after.len() || packed.values.len() != packed.scales.len() {
        return Err(TofMultiBankGeometryError::InternalInvariant);
    }
    let mut changes = Vec::new();
    let mut offset = 0;
    for model in &input.instrument_models {
        for bound in model.bounds() {
            if packed.values[offset].to_bits() != after[offset].to_bits() {
                changes.push(TofInstrumentParameterChange {
                    bank_id: model.bank_id().clone(),
                    parameter: bound.parameter,
                    before: packed.values[offset],
                    after: after[offset],
                    scaled_change: (after[offset] - packed.values[offset]) / packed.scales[offset],
                });
            }
            offset += 1;
        }
    }
    Ok(changes)
}

fn geometry_keys(input: &TofMultiBankGeometryInput) -> Vec<TofGeometryParameterKey> {
    let mut keys = Vec::new();
    for phase in &input.lattice.lattice_phases {
        for parameter_name in phase.parameterization().parameter_names() {
            keys.push(TofGeometryParameterKey::Lattice {
                phase_id: phase.phase_id().clone(),
                parameter_name: parameter_name.clone(),
            });
        }
    }
    for model in &input.instrument_models {
        for bound in model.bounds() {
            keys.push(TofGeometryParameterKey::Instrument {
                bank_id: model.bank_id().clone(),
                parameter: bound.parameter,
            });
        }
    }
    keys
}

fn geometry_diagnostics(
    input: &TofMultiBankGeometryInput,
    jacobian: &DMatrix<f64>,
    threshold: f64,
) -> Result<TofGeometryDiagnostics, TofMultiBankGeometryError> {
    let parameter_count = input.parameter_count()?;
    if jacobian.ncols() != parameter_count {
        return Err(TofMultiBankGeometryError::InternalInvariant);
    }
    let singular = jacobian.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0_f64, f64::max);
    let dimension = u32::try_from(jacobian.nrows().max(jacobian.ncols())).unwrap_or(u32::MAX);
    let tolerance = f64::from(dimension) * f64::EPSILON * maximum;
    let jacobian_rank = singular.iter().filter(|value| **value > tolerance).count();
    let keys = geometry_keys(input);
    let norms = (0..parameter_count)
        .map(|column| jacobian.column(column).norm())
        .collect::<Vec<_>>();
    let mut maximum_absolute_correlation = None::<f64>;
    let mut unresolved_correlations = Vec::new();
    for left in 0..parameter_count {
        if norms[left] == 0.0 {
            continue;
        }
        for right in left + 1..parameter_count {
            if norms[right] == 0.0 {
                continue;
            }
            let correlation = (jacobian.column(left).dot(&jacobian.column(right))
                / (norms[left] * norms[right]))
                .clamp(-1.0, 1.0);
            let absolute = correlation.abs();
            maximum_absolute_correlation =
                Some(maximum_absolute_correlation.map_or(absolute, |value| value.max(absolute)));
            if absolute >= threshold {
                unresolved_correlations.push(TofGeometryCorrelation {
                    left: keys[left].clone(),
                    right: keys[right].clone(),
                    correlation,
                });
            }
        }
    }
    Ok(TofGeometryDiagnostics {
        parameter_count,
        jacobian_rank,
        maximum_absolute_correlation,
        unresolved_correlations,
    })
}

fn checkpoint_from_state(
    input: &TofMultiBankGeometryInput,
    states: &[AcceptedBankState],
    cells: &[UnitCell],
    instruments: &[TofInstrument],
    history: &[TofMultiBankGeometryIterationRecord],
) -> TofMultiBankGeometryCheckpoint {
    TofMultiBankGeometryCheckpoint {
        completed_iterations: history.len(),
        banks: input
            .lattice
            .multibank
            .banks
            .iter()
            .zip(states)
            .zip(instruments)
            .map(
                |((bank, state), instrument)| TofMultiBankGeometryCheckpointBank {
                    bank_id: bank.bank_id.clone(),
                    instrument: *instrument,
                    phases: state.phases.clone(),
                    background: state.background.clone(),
                },
            )
            .collect(),
        lattice_phases: lattice_states(input, cells),
        history: history.to_vec(),
    }
}

fn lattice_states(
    input: &TofMultiBankGeometryInput,
    cells: &[UnitCell],
) -> Vec<TofSharedLatticeState> {
    input
        .lattice
        .lattice_phases
        .iter()
        .zip(cells)
        .map(|(phase, cell)| TofSharedLatticeState {
            phase_id: phase.phase_id().clone(),
            cell: *cell,
        })
        .collect()
}

fn instrument_states(
    input: &TofMultiBankGeometryInput,
    instruments: &[TofInstrument],
) -> Vec<TofBankInstrumentState> {
    input
        .lattice
        .multibank
        .banks
        .iter()
        .zip(instruments)
        .map(|(bank, instrument)| TofBankInstrumentState {
            bank_id: bank.bank_id.clone(),
            instrument: *instrument,
        })
        .collect()
}

fn recoverable_lattice_error(error: &TofMultiBankLatticeError) -> bool {
    matches!(
        error,
        TofMultiBankLatticeError::Lattice(_)
            | TofMultiBankLatticeError::InvalidModel(_)
            | TofMultiBankLatticeError::MultiBank(TofMultiBankError::LeBail(
                TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_)
            ))
    )
}

fn recoverable_instrument_error(error: &TofMultiBankInstrumentError) -> bool {
    matches!(
        error,
        TofMultiBankInstrumentError::Profile(_)
            | TofMultiBankInstrumentError::MultiBank(TofMultiBankError::LeBail(
                TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_)
            ))
    )
}

fn recoverable_multibank_error(error: &TofMultiBankError) -> bool {
    matches!(
        error,
        TofMultiBankError::LeBail(TofLeBailError::Profile(_) | TofLeBailError::InvalidPhase(_))
    )
}

impl TofMultiBankGeometryCheckpoint {
    /// Revalidate this continuation against every shared/local contract.
    ///
    /// # Errors
    ///
    /// Returns [`TofMultiBankGeometryError`] for stale identity, topology,
    /// bounds, geometry, instruments, backgrounds, or history.
    pub fn validate_for(
        &self,
        input: &TofMultiBankGeometryInput,
        options: &TofMultiBankGeometryOptions,
    ) -> Result<(), TofMultiBankGeometryError> {
        input.validate()?;
        options.validate()?;
        if self.completed_iterations != self.history.len()
            || self.completed_iterations > options.lebail.cycles
            || self.banks.len() != input.lattice.multibank.banks.len()
            || self.lattice_phases.len() != input.lattice.lattice_phases.len()
        {
            return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                "joint TOF geometry checkpoint counts differ from the request",
            ));
        }
        for (saved, model) in self
            .lattice_phases
            .iter()
            .zip(&input.lattice.lattice_phases)
        {
            if &saved.phase_id != model.phase_id() {
                return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                    "joint TOF geometry checkpoint lattice phase order changed",
                ));
            }
            model.validate_cell(saved.cell)?;
        }
        for (saved, original) in self.banks.iter().zip(&input.lattice.multibank.banks) {
            validate_checkpoint_bank(saved, original, input, &self.lattice_phases)?;
        }
        if self.history.iter().enumerate().any(|(index, record)| {
            record.iteration != index + 1
                || record.bank_metrics.len() != input.lattice.multibank.banks.len()
        }) {
            return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                "joint TOF geometry checkpoint history is not contiguous and bank-aligned",
            ));
        }
        Ok(())
    }
}

fn validate_checkpoint_bank(
    saved: &TofMultiBankGeometryCheckpointBank,
    original: &crate::TofLeBailBank,
    input: &TofMultiBankGeometryInput,
    cells: &[TofSharedLatticeState],
) -> Result<(), TofMultiBankGeometryError> {
    saved.instrument.validate()?;
    if saved.bank_id != original.bank_id || saved.phases.len() != original.input.phases.len() {
        return Err(TofMultiBankGeometryError::InvalidCheckpoint(
            "joint TOF geometry checkpoint bank identity or phase count changed",
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
            return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                "joint TOF geometry checkpoint bank topology changed",
            ));
        }
        if let Some((model, cell)) = input
            .lattice
            .lattice_phases
            .iter()
            .zip(cells)
            .find(|(model, _)| model.phase_id() == saved_phase.phase_id())
        {
            let expected = tof_lattice_geometry(
                model.parameterization(),
                cell.cell,
                saved_phase.hkl(),
                saved.instrument,
            )?;
            if saved_phase.d_spacing_angstrom() != expected.d_spacing_angstrom {
                return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                    "joint TOF geometry checkpoint d-spacings disagree with its cell",
                ));
            }
        } else if saved_phase.d_spacing_angstrom() != original_phase.d_spacing_angstrom() {
            return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                "fixed TOF phase d-spacings changed in a joint checkpoint",
            ));
        }
        for d_spacing in saved_phase.d_spacing_angstrom() {
            TofProfileParameters::from_instrument(*d_spacing, saved.instrument)?;
        }
    }
    validate_checkpoint_instrument(saved, original, input)?;
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
        _ => Err(TofMultiBankGeometryError::InvalidCheckpoint(
            "joint TOF geometry checkpoint background contract changed",
        )),
    }
}

fn validate_checkpoint_instrument(
    saved: &TofMultiBankGeometryCheckpointBank,
    original: &crate::TofLeBailBank,
    input: &TofMultiBankGeometryInput,
) -> Result<(), TofMultiBankGeometryError> {
    let selected = input
        .instrument_models
        .iter()
        .find(|model| model.bank_id() == &saved.bank_id);
    let saved_values = saved.instrument.values();
    let original_values = original.input.instrument.values();
    for parameter in TofInstrumentParameter::ALL {
        let bound = selected.and_then(|model| {
            model
                .bounds()
                .iter()
                .find(|bound| bound.parameter == parameter)
        });
        match bound {
            Some(bound)
                if saved_values[parameter.index()] >= bound.lower
                    && saved_values[parameter.index()] <= bound.upper => {}
            Some(_) => {
                return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                    "joint TOF geometry checkpoint instrument lies outside its bounds",
                ));
            }
            None if saved_values[parameter.index()].to_bits()
                == original_values[parameter.index()].to_bits() => {}
            None => {
                return Err(TofMultiBankGeometryError::InvalidCheckpoint(
                    "unselected TOF instrument coefficient changed in a joint checkpoint",
                ));
            }
        }
    }
    Ok(())
}

/// Invalid joint shared-cell/bank-local-instrument TOF state.
#[derive(Debug)]
pub enum TofMultiBankGeometryError {
    /// Shared lattice contract or calculation failed.
    Lattice(TofMultiBankLatticeError),
    /// Bank-local instrument contract or calculation failed.
    Instrument(TofMultiBankInstrumentError),
    /// Atomic multi-bank calculation failed.
    MultiBank(TofMultiBankError),
    /// Solver controls are invalid.
    InvalidOptions,
    /// Weighted joint normal equations could not be solved finitely.
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

impl Display for TofMultiBankGeometryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lattice(error) => Display::fmt(error, formatter),
            Self::Instrument(error) => Display::fmt(error, formatter),
            Self::MultiBank(error) => Display::fmt(error, formatter),
            Self::InvalidOptions => formatter.write_str("invalid joint TOF geometry options"),
            Self::LinearSolve => formatter.write_str("joint TOF geometry linear solve failed"),
            Self::InvalidCheckpoint(message) => formatter.write_str(message),
            Self::AllocationOverflow => {
                formatter.write_str("joint TOF geometry allocation overflow")
            }
            Self::Runtime(error) => Display::fmt(error, formatter),
            Self::InternalInvariant => {
                formatter.write_str("joint TOF geometry internal array invariant failed")
            }
        }
    }
}

impl Error for TofMultiBankGeometryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Lattice(error) => Some(error),
            Self::Instrument(error) => Some(error),
            Self::MultiBank(error) => Some(error),
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TofMultiBankLatticeError> for TofMultiBankGeometryError {
    fn from(value: TofMultiBankLatticeError) -> Self {
        Self::Lattice(value)
    }
}

impl From<TofMultiBankInstrumentError> for TofMultiBankGeometryError {
    fn from(value: TofMultiBankInstrumentError) -> Self {
        Self::Instrument(value)
    }
}

impl From<TofMultiBankError> for TofMultiBankGeometryError {
    fn from(value: TofMultiBankError) -> Self {
        Self::MultiBank(value)
    }
}

impl From<TofLeBailError> for TofMultiBankGeometryError {
    fn from(value: TofLeBailError) -> Self {
        Self::MultiBank(TofMultiBankError::LeBail(value))
    }
}

impl From<phasesmith_core::TofError> for TofMultiBankGeometryError {
    fn from(value: phasesmith_core::TofError) -> Self {
        Self::Instrument(TofMultiBankInstrumentError::Profile(value))
    }
}

impl From<crate::LatticeError> for TofMultiBankGeometryError {
    fn from(value: crate::LatticeError) -> Self {
        Self::Lattice(TofMultiBankLatticeError::Lattice(value))
    }
}

impl From<RuntimeError> for TofMultiBankGeometryError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
