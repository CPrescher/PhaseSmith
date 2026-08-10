//! Explicit and advisory staged workflows above the general Rietveld solver.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

use crate::{
    CancellationToken, CheckpointSink, Constraint, ConstraintError, ConstraintTransform,
    DiagnosticValue, LatticeBounds, ParameterKey, RefinementEvent, RefinementEventKind,
    RefinementEventSink, RefinementRuntime, RietveldCovarianceOptions, RietveldGeneralCheckpoint,
    RietveldGeneralParameterError, RietveldGeneralRefinementError, RietveldGeneralRefinementResult,
    RietveldInput, RietveldInstrumentParameter, RietveldParameterLayout,
    RietveldParameterSelection, RietveldRefinementOptions, RietveldStructuralSelection,
    TerminationReason, calculate_rietveld_pattern, refine_general_rietveld_with_runtime,
};

/// Origin of a staged Rietveld recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RietveldRecipeMode {
    /// Caller-authored stage sequence.
    Explicit,
    /// Deterministic advisory sequence proposed by the native planner.
    Intelligent,
}

impl RietveldRecipeMode {
    /// Return the stable scripting/wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Intelligent => "intelligent",
        }
    }
}

/// One caller-visible active parameter set and acceptance policy.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldStage {
    name: String,
    selection: RietveldParameterSelection,
    rationale: Vec<String>,
    options: Option<RietveldRefinementOptions>,
    covariance: Option<RietveldCovarianceOptions>,
    accepted_terminations: Vec<TerminationReason>,
}

impl RietveldStage {
    /// Construct one stage with the default converged-or-stagnated policy.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldRecipeError`] for an invalid name or rationale.
    pub fn new(
        name: impl Into<String>,
        selection: RietveldParameterSelection,
        rationale: Vec<String>,
    ) -> Result<Self, RietveldRecipeError> {
        let result = Self {
            name: name.into(),
            selection,
            rationale,
            options: None,
            covariance: None,
            accepted_terminations: vec![TerminationReason::Converged, TerminationReason::Stagnated],
        };
        result.validate()?;
        Ok(result)
    }

    /// Attach stage-specific numerical and covariance controls.
    #[must_use]
    pub fn with_options(
        mut self,
        options: RietveldRefinementOptions,
        covariance: RietveldCovarianceOptions,
    ) -> Self {
        self.options = Some(options);
        self.covariance = Some(covariance);
        self
    }

    /// Replace the accepted normal termination categories.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldRecipeError::InvalidAcceptedTerminations`] for an
    /// empty or duplicate policy.
    pub fn with_accepted_terminations(
        mut self,
        accepted: Vec<TerminationReason>,
    ) -> Result<Self, RietveldRecipeError> {
        self.accepted_terminations = accepted;
        self.validate()?;
        Ok(self)
    }

    /// Borrow the stable stage name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the cumulative active selection.
    #[must_use]
    pub const fn selection(&self) -> &RietveldParameterSelection {
        &self.selection
    }

    /// Borrow the human-readable planning rationale.
    #[must_use]
    pub fn rationale(&self) -> &[String] {
        &self.rationale
    }

    /// Borrow stage-specific solver controls, when supplied.
    #[must_use]
    pub const fn options(&self) -> Option<&RietveldRefinementOptions> {
        self.options.as_ref()
    }

    /// Return stage-specific covariance controls, when supplied.
    #[must_use]
    pub const fn covariance(&self) -> Option<RietveldCovarianceOptions> {
        self.covariance
    }

    /// Borrow normal termination categories accepted for state promotion.
    #[must_use]
    pub fn accepted_terminations(&self) -> &[TerminationReason] {
        &self.accepted_terminations
    }

    fn validate(&self) -> Result<(), RietveldRecipeError> {
        if !valid_label(&self.name) {
            return Err(RietveldRecipeError::InvalidStageName);
        }
        if self.rationale.is_empty() || self.rationale.iter().any(|value| !valid_label(value)) {
            return Err(RietveldRecipeError::InvalidRationale);
        }
        if self.accepted_terminations.is_empty()
            || self
                .accepted_terminations
                .iter()
                .enumerate()
                .any(|(index, value)| self.accepted_terminations[..index].contains(value))
        {
            return Err(RietveldRecipeError::InvalidAcceptedTerminations);
        }
        Ok(())
    }
}

/// Auditable explicit or intelligent stage sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldRecipe {
    name: String,
    stages: Vec<RietveldStage>,
    mode: RietveldRecipeMode,
    planner_notes: Vec<String>,
}

impl RietveldRecipe {
    /// Construct and validate a non-empty recipe.
    ///
    /// # Errors
    ///
    /// Returns [`RietveldRecipeError`] for malformed labels, empty recipes,
    /// duplicate stage names, or invalid stages.
    pub fn new(
        name: impl Into<String>,
        stages: Vec<RietveldStage>,
        mode: RietveldRecipeMode,
        planner_notes: Vec<String>,
    ) -> Result<Self, RietveldRecipeError> {
        let result = Self {
            name: name.into(),
            stages,
            mode,
            planner_notes,
        };
        result.validate()?;
        Ok(result)
    }

    /// Borrow the recipe name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Borrow the ordered stages.
    #[must_use]
    pub fn stages(&self) -> &[RietveldStage] {
        &self.stages
    }

    /// Return whether the recipe is explicit or planner-proposed.
    #[must_use]
    pub const fn mode(&self) -> RietveldRecipeMode {
        self.mode
    }

    /// Borrow human-readable planner notes.
    #[must_use]
    pub fn planner_notes(&self) -> &[String] {
        &self.planner_notes
    }

    fn validate(&self) -> Result<(), RietveldRecipeError> {
        if !valid_label(&self.name) {
            return Err(RietveldRecipeError::InvalidRecipeName);
        }
        if self.stages.is_empty() {
            return Err(RietveldRecipeError::EmptyRecipe);
        }
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            stage.validate()?;
            if !names.insert(stage.name.clone()) {
                return Err(RietveldRecipeError::DuplicateStageName {
                    name: stage.name.clone(),
                });
            }
        }
        if self.planner_notes.iter().any(|value| !valid_label(value)) {
            return Err(RietveldRecipeError::InvalidPlannerNote);
        }
        Ok(())
    }
}

/// Auditable outcome from one attempted recipe stage.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldStageResult {
    /// Stage contract that was executed.
    pub stage: RietveldStage,
    /// Rwp before this stage.
    pub starting_rwp: f64,
    /// Complete final result, including a rejected stopping state.
    pub result: RietveldGeneralRefinementResult,
    /// Whether the stage termination permits promotion to the next stage.
    pub accepted: bool,
}

/// Complete or safely stopped native staged workflow.
#[derive(Clone, Debug, PartialEq)]
pub struct RietveldWorkflowResult {
    recipe: RietveldRecipe,
    stages: Vec<RietveldStageResult>,
    completed: bool,
}

impl RietveldWorkflowResult {
    /// Borrow the recipe that was executed.
    #[must_use]
    pub const fn recipe(&self) -> &RietveldRecipe {
        &self.recipe
    }

    /// Borrow attempted stages in order.
    #[must_use]
    pub fn stages(&self) -> &[RietveldStageResult] {
        &self.stages
    }

    /// Return true only when every stage ran and the last stage was accepted.
    #[must_use]
    pub const fn completed(&self) -> bool {
        self.completed
    }

    /// Return the final attempted numerical result.
    #[must_use]
    pub fn final_result(&self) -> &RietveldGeneralRefinementResult {
        &self.stages[self.stages.len() - 1].result
    }

    /// Return the last state the recipe policy permits promoting.
    #[must_use]
    pub fn last_accepted_stage(&self) -> Option<&RietveldStageResult> {
        self.stages.iter().rev().find(|stage| stage.accepted)
    }
}

/// Cloneable host observers shared across every stage runtime.
#[derive(Clone, Default)]
pub struct RietveldRecipeSinks {
    event: Option<SharedEventSink>,
    checkpoint: Option<SharedCheckpointSink>,
}

impl RietveldRecipeSinks {
    /// Attach one synchronous structured-event consumer.
    #[must_use]
    pub fn with_event_sink(mut self, sink: impl RefinementEventSink + 'static) -> Self {
        self.event = Some(SharedEventSink(Arc::new(Mutex::new(Some(Box::new(sink))))));
        self
    }

    /// Attach one durable complete-checkpoint consumer.
    #[must_use]
    pub fn with_checkpoint_sink(
        mut self,
        sink: impl CheckpointSink<RietveldGeneralCheckpoint> + 'static,
    ) -> Self {
        self.checkpoint = Some(SharedCheckpointSink(Arc::new(Mutex::new(Box::new(sink)))));
        self
    }
}

type EventSinkBox = Box<dyn RefinementEventSink>;
type CheckpointSinkBox = Box<dyn CheckpointSink<RietveldGeneralCheckpoint>>;

#[derive(Clone)]
struct SharedEventSink(Arc<Mutex<Option<EventSinkBox>>>);

impl RefinementEventSink for SharedEventSink {
    fn emit(&mut self, event: &RefinementEvent) -> Result<(), String> {
        let mut sink = self
            .0
            .lock()
            .map_err(|_| "Rietveld recipe event sink lock is poisoned".to_owned())?;
        let Some(active) = sink.as_mut() else {
            return Ok(());
        };
        if let Err(error) = active.emit(event) {
            *sink = None;
            return Err(error);
        }
        Ok(())
    }
}

#[derive(Clone)]
struct SharedCheckpointSink(Arc<Mutex<CheckpointSinkBox>>);

impl CheckpointSink<RietveldGeneralCheckpoint> for SharedCheckpointSink {
    fn checkpoint(&mut self, checkpoint: &RietveldGeneralCheckpoint) -> Result<(), String> {
        self.0
            .lock()
            .map_err(|_| "Rietveld recipe checkpoint sink lock is poisoned".to_owned())?
            .checkpoint(checkpoint)
    }
}

/// Propose transparent cumulative stages from caller-authorized families.
///
/// This function only returns advice; it never starts numerical work.
///
/// # Errors
///
/// Returns [`RietveldRecipeError`] for invalid input or recipe metadata.
#[allow(clippy::too_many_lines)]
pub fn intelligent_rietveld_recipe(
    input: &RietveldInput,
    maximum: &RietveldParameterSelection,
    name: impl Into<String>,
) -> Result<RietveldRecipe, RietveldRecipeError> {
    input.validate()?;
    let mut stages = Vec::new();
    let mut current = None;
    append_stage(
        &mut stages,
        &mut current,
        "scale_background",
        selected(maximum, true, false, false, false, false, false, &[], true),
        vec![
            "Establish intensity scale and any differentiable background before correlated terms."
                .to_owned(),
        ],
    )?;
    let positions = [
        RietveldInstrumentParameter::WavelengthAngstrom,
        RietveldInstrumentParameter::ZeroShiftDeg,
        RietveldInstrumentParameter::SampleDisplacementMm,
        RietveldInstrumentParameter::DisplaceXMicrometre,
        RietveldInstrumentParameter::DisplaceYMicrometre,
    ];
    append_stage(
        &mut stages,
        &mut current,
        "positions",
        selected(
            maximum, true, true, false, false, false, false, &positions, true,
        ),
        vec![
            "Align reflection positions before refining peak widths or structural intensities."
                .to_owned(),
        ],
    )?;
    append_stage(
        &mut stages,
        &mut current,
        "structure",
        selected(
            maximum, true, true, true, true, true, false, &positions, true,
        ),
        vec![
            "Stabilize relative structural intensities after peak centers are aligned.".to_owned(),
            "Delay profile widths so they cannot initially mask intensity-model errors.".to_owned(),
        ],
    )?;
    append_stage(
        &mut stages,
        &mut current,
        "final_polish",
        maximum.clone(),
        vec![
            "Release authorized profile and sample broadening after positions and intensities."
                .to_owned(),
            "Finish with every caller-authorized parameter active together.".to_owned(),
        ],
    )?;
    if stages.is_empty() {
        stages.push(RietveldStage::new(
            "evaluate_only",
            maximum.clone(),
            vec![
                "No refinable parameter family was authorized; evaluate the supplied state."
                    .to_owned(),
            ],
        )?);
    }
    let mut notes = vec![
        "This plan is advisory workflow orchestration; the Rietveld solver remains general."
            .to_owned(),
        "Every stage is cumulative and limited to parameter families authorized by the maximum selection."
            .to_owned(),
    ];
    if !maximum.background {
        notes.push(
            "No differentiable background was authorized; the supplied background stays fixed."
                .to_owned(),
        );
    }
    if !maximum.structural.lattice {
        notes.push(
            "Lattice refinement was unavailable or disabled and was not proposed.".to_owned(),
        );
    }
    if maximum.instrument.iter().any(|value| {
        matches!(
            value,
            RietveldInstrumentParameter::DisplaceXMicrometre
                | RietveldInstrumentParameter::DisplaceYMicrometre
        )
    }) {
        notes.push(
            "Debye-Scherrer X/Y displacement was authorized and is proposed in the position-alignment stage before profile widths."
                .to_owned(),
        );
    }
    if maximum.structural.occupancy {
        notes.push(
            "Occupancy is delayed to the structural stage because it is strongly scale-correlated."
                .to_owned(),
        );
    }
    RietveldRecipe::new(name, stages, RietveldRecipeMode::Intelligent, notes)
}

/// Execute explicit stages while promoting only accepted physical states.
///
/// Intermediate covariance work is disabled; only the final attempted recipe
/// stage may use its selected covariance controls.
///
/// # Errors
///
/// Returns [`RietveldRecipeError`] for unauthorized selections, incomplete
/// constraint dependencies, invalid state, or numerical failures.
#[allow(clippy::too_many_arguments)]
pub fn run_rietveld_recipe(
    input: &RietveldInput,
    maximum: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    recipe: &RietveldRecipe,
    options: &RietveldRefinementOptions,
    covariance: RietveldCovarianceOptions,
    cancellation: Option<&CancellationToken>,
) -> Result<RietveldWorkflowResult, RietveldRecipeError> {
    run_rietveld_recipe_with_sinks(
        input,
        maximum,
        lattice_bounds,
        constraints,
        recipe,
        options,
        covariance,
        cancellation,
        None,
    )
}

/// Validate a complete staged recipe without calculating or refining a pattern.
///
/// This checks the input, authorized maximum selection, parameter layouts,
/// complete constraint state, and every stage's constraint dependencies. It is
/// intended for application review screens and other advisory orchestration.
///
/// # Errors
///
/// Returns [`RietveldRecipeError`] for the same recipe-contract failures that
/// would stop [`run_rietveld_recipe_with_sinks`] before numerical work begins.
pub fn validate_rietveld_recipe(
    input: &RietveldInput,
    maximum: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    recipe: &RietveldRecipe,
) -> Result<(), RietveldRecipeError> {
    input.validate()?;
    recipe.validate()?;
    let maximum_layout = RietveldParameterLayout::new(input, maximum, lattice_bounds)?;
    validate_constraint_contract(&maximum_layout, constraints)?;
    for stage in &recipe.stages {
        if !selection_subset(&stage.selection, maximum) {
            return Err(RietveldRecipeError::UnauthorizedSelection {
                stage: stage.name.clone(),
            });
        }
        let layout = RietveldParameterLayout::new(input, &stage.selection, lattice_bounds)?;
        let keys = layout
            .parameters()
            .specs()
            .iter()
            .map(|spec| spec.key().clone())
            .collect::<Vec<_>>();
        stage_constraints(constraints, &keys, &stage.name)?;
    }
    Ok(())
}

/// Execute a recipe with optional shared event and checkpoint consumers.
///
/// Each stage owns a fresh bounded runtime but forwards its numerical events
/// and accepted complete checkpoints through the same thread-safe sinks.
/// Recipe-level start/termination events use the caller-visible stage name.
///
/// # Errors
///
/// Returns [`RietveldRecipeError`] under the same contracts as
/// [`run_rietveld_recipe`], including durable checkpoint failures.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run_rietveld_recipe_with_sinks(
    input: &RietveldInput,
    maximum: &RietveldParameterSelection,
    lattice_bounds: &[Option<LatticeBounds>],
    constraints: &[Constraint],
    recipe: &RietveldRecipe,
    options: &RietveldRefinementOptions,
    covariance: RietveldCovarianceOptions,
    cancellation: Option<&CancellationToken>,
    sinks: Option<&RietveldRecipeSinks>,
) -> Result<RietveldWorkflowResult, RietveldRecipeError> {
    validate_rietveld_recipe(input, maximum, lattice_bounds, constraints, recipe)?;
    let mut current = input.clone();
    let mut current_rwp = calculate_rietveld_pattern(&current, &options.calculation)?
        .metrics
        .rwp;
    let mut results = Vec::with_capacity(recipe.stages.len());
    for (index, stage) in recipe.stages.iter().enumerate() {
        let layout = RietveldParameterLayout::new(&current, &stage.selection, lattice_bounds)?;
        let keys = layout
            .parameters()
            .specs()
            .iter()
            .map(|spec| spec.key().clone())
            .collect::<Vec<_>>();
        let stage_constraints = stage_constraints(constraints, &keys, &stage.name)?;
        let stage_options = stage.options.as_ref().unwrap_or(options);
        let mut stage_covariance = stage.covariance.unwrap_or(covariance);
        if index + 1 < recipe.stages.len() {
            stage_covariance.enabled = false;
        }
        let mut runtime = RefinementRuntime::new(stage_options.limits, cancellation.cloned())
            .map_err(RietveldGeneralRefinementError::from)?;
        if let Some(event) = sinks.and_then(|value| value.event.clone()) {
            runtime.set_event_sink(event);
        }
        if let Some(checkpoint) = sinks.and_then(|value| value.checkpoint.clone()) {
            runtime.set_checkpoint_sink(checkpoint);
        }
        runtime
            .emit(
                RefinementEventKind::Start,
                &stage.name,
                "native Rietveld recipe stage started",
                vec![
                    (
                        "recipe".to_owned(),
                        DiagnosticValue::String(recipe.name.clone()),
                    ),
                    (
                        "stage_index".to_owned(),
                        DiagnosticValue::Unsigned(u64::try_from(index).unwrap_or(u64::MAX)),
                    ),
                ],
            )
            .map_err(RietveldGeneralRefinementError::from)?;
        let result = refine_general_rietveld_with_runtime(
            &current,
            &stage.selection,
            lattice_bounds,
            &stage_constraints,
            stage_options,
            stage_covariance,
            None,
            &mut runtime,
        )?;
        let accepted = stage
            .accepted_terminations
            .contains(&result.termination_reason);
        let final_rwp = result.calculation.metrics.rwp;
        runtime
            .emit(
                RefinementEventKind::Termination,
                &stage.name,
                if accepted {
                    "native Rietveld recipe stage accepted"
                } else {
                    "native Rietveld recipe stage rejected"
                },
                vec![
                    ("accepted".to_owned(), DiagnosticValue::Bool(accepted)),
                    (
                        "reason".to_owned(),
                        DiagnosticValue::String(result.termination_reason.as_str().to_owned()),
                    ),
                ],
            )
            .map_err(RietveldGeneralRefinementError::from)?;
        results.push(RietveldStageResult {
            stage: stage.clone(),
            starting_rwp: current_rwp,
            result,
            accepted,
        });
        if !accepted {
            break;
        }
        current = results[results.len() - 1].result.input.clone();
        current_rwp = final_rwp;
    }
    let completed =
        results.len() == recipe.stages.len() && results.last().is_some_and(|stage| stage.accepted);
    Ok(RietveldWorkflowResult {
        recipe: recipe.clone(),
        stages: results,
        completed,
    })
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn selected(
    maximum: &RietveldParameterSelection,
    phase_scale: bool,
    lattice: bool,
    coordinates: bool,
    occupancy: bool,
    u_iso: bool,
    sample_physics: bool,
    instrument: &[RietveldInstrumentParameter],
    background: bool,
) -> RietveldParameterSelection {
    RietveldParameterSelection {
        structural: RietveldStructuralSelection {
            phase_scale: phase_scale && maximum.structural.phase_scale,
            lattice: lattice && maximum.structural.lattice,
            coordinates: coordinates && maximum.structural.coordinates,
            occupancy: occupancy && maximum.structural.occupancy,
            u_iso: u_iso && maximum.structural.u_iso,
        },
        instrument: maximum
            .instrument
            .iter()
            .copied()
            .filter(|value| instrument.contains(value))
            .collect(),
        background: background && maximum.background,
        sample_physics: sample_physics && maximum.sample_physics,
    }
}

fn selection_has_parameters(selection: &RietveldParameterSelection) -> bool {
    let structural = selection.structural;
    structural.phase_scale
        || structural.lattice
        || structural.coordinates
        || structural.occupancy
        || structural.u_iso
        || !selection.instrument.is_empty()
        || selection.background
        || selection.sample_physics
}

fn append_stage(
    stages: &mut Vec<RietveldStage>,
    current: &mut Option<RietveldParameterSelection>,
    name: &str,
    selection: RietveldParameterSelection,
    rationale: Vec<String>,
) -> Result<(), RietveldRecipeError> {
    if current.as_ref() == Some(&selection) || !selection_has_parameters(&selection) {
        return Ok(());
    }
    stages.push(RietveldStage::new(name, selection.clone(), rationale)?);
    *current = Some(selection);
    Ok(())
}

fn selection_subset(
    selected: &RietveldParameterSelection,
    maximum: &RietveldParameterSelection,
) -> bool {
    let selected_structural = selected.structural;
    let maximum_structural = maximum.structural;
    (!selected_structural.phase_scale || maximum_structural.phase_scale)
        && (!selected_structural.lattice || maximum_structural.lattice)
        && (!selected_structural.coordinates || maximum_structural.coordinates)
        && (!selected_structural.occupancy || maximum_structural.occupancy)
        && (!selected_structural.u_iso || maximum_structural.u_iso)
        && (!selected.background || maximum.background)
        && (!selected.sample_physics || maximum.sample_physics)
        && selected
            .instrument
            .iter()
            .all(|value| maximum.instrument.contains(value))
}

fn stage_constraints(
    constraints: &[Constraint],
    keys: &[ParameterKey],
    stage: &str,
) -> Result<Vec<Constraint>, RietveldRecipeError> {
    let available = keys.iter().collect::<BTreeSet<_>>();
    let mut selected = Vec::new();
    for constraint in constraints {
        if !available.contains(constraint.target()) {
            continue;
        }
        let sources = match constraint {
            Constraint::Fixed(_) => Vec::new(),
            Constraint::Affine(value) => vec![value.source()],
            Constraint::Linear(value) => value
                .terms()
                .iter()
                .map(crate::LinearTerm::source)
                .collect(),
        };
        if let Some(missing) = sources.into_iter().find(|key| !available.contains(key)) {
            return Err(RietveldRecipeError::MissingConstraintDependency {
                stage: stage.to_owned(),
                target: Box::new(constraint.target().clone()),
                missing: Box::new(missing.clone()),
            });
        }
        selected.push(constraint.clone());
    }
    Ok(selected)
}

fn validate_constraint_contract(
    layout: &RietveldParameterLayout,
    constraints: &[Constraint],
) -> Result<(), RietveldRecipeError> {
    let transform = ConstraintTransform::new(layout.parameters().clone(), constraints.to_vec())?;
    let constrained = transform.unpack(&transform.pack()?, false)?;
    for spec in layout.parameters().specs() {
        let value = constrained
            .get(spec.key())
            .copied()
            .ok_or(RietveldRecipeError::InternalInvariant)?;
        if (value - spec.value()).abs() > 2.0e-12 {
            return Err(RietveldRecipeError::UnsatisfiedConstraint {
                key: Box::new(spec.key().clone()),
            });
        }
    }
    Ok(())
}

fn valid_label(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

/// Invalid staged Rietveld workflow state.
#[derive(Debug)]
pub enum RietveldRecipeError {
    /// Recipe name is empty or contains surrounding whitespace.
    InvalidRecipeName,
    /// Stage name is empty or contains surrounding whitespace.
    InvalidStageName,
    /// A stage has no valid human-readable rationale.
    InvalidRationale,
    /// Accepted termination policy is empty or contains duplicates.
    InvalidAcceptedTerminations,
    /// A planner note is empty or contains surrounding whitespace.
    InvalidPlannerNote,
    /// Recipe has no stages.
    EmptyRecipe,
    /// Recipe has two stages with the same stable name.
    DuplicateStageName {
        /// Duplicate name.
        name: String,
    },
    /// Stage selects a parameter outside the caller-authorized maximum.
    UnauthorizedSelection {
        /// Invalid stage name.
        stage: String,
    },
    /// Selected constraint target is missing one selected dependency.
    MissingConstraintDependency {
        /// Stage name, when available.
        stage: String,
        /// Selected target.
        target: Box<ParameterKey>,
        /// Missing source.
        missing: Box<ParameterKey>,
    },
    /// Full authorized physical state does not satisfy its constraint graph.
    UnsatisfiedConstraint {
        /// First inconsistent physical identity.
        key: Box<ParameterKey>,
    },
    /// A validated layout unexpectedly omitted one key.
    InternalInvariant,
    /// Constraint graph or expansion failed.
    Constraint(Box<ConstraintError>),
    /// Complete parameter layout failed.
    Parameter(Box<RietveldGeneralParameterError>),
    /// Input or initial calculation failed.
    Rietveld(Box<crate::RietveldError>),
    /// One numerical stage failed.
    Refinement(Box<RietveldGeneralRefinementError>),
}

impl Display for RietveldRecipeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRecipeName => formatter.write_str("Rietveld recipe name is invalid"),
            Self::InvalidStageName => formatter.write_str("Rietveld stage name is invalid"),
            Self::InvalidRationale => formatter.write_str("Rietveld stage rationale is invalid"),
            Self::InvalidAcceptedTerminations => {
                formatter.write_str("Rietveld stage accepted terminations are invalid")
            }
            Self::InvalidPlannerNote => formatter.write_str("Rietveld planner note is invalid"),
            Self::EmptyRecipe => formatter.write_str("Rietveld recipe must contain a stage"),
            Self::DuplicateStageName { name } => {
                write!(
                    formatter,
                    "Rietveld recipe stage name {name:?} is duplicated"
                )
            }
            Self::UnauthorizedSelection { stage } => write!(
                formatter,
                "Rietveld stage {stage:?} selects parameters outside the authorized maximum"
            ),
            Self::MissingConstraintDependency {
                stage,
                target,
                missing,
            } => write!(
                formatter,
                "Rietveld stage {stage:?} selects constraint target {} without dependency {}",
                target.label(),
                missing.label()
            ),
            Self::UnsatisfiedConstraint { key } => write!(
                formatter,
                "initial physical value does not satisfy the constraint for {}",
                key.label()
            ),
            Self::InternalInvariant => {
                formatter.write_str("native staged Rietveld invariant failed")
            }
            Self::Constraint(error) => Display::fmt(error, formatter),
            Self::Parameter(error) => Display::fmt(error, formatter),
            Self::Rietveld(error) => Display::fmt(error, formatter),
            Self::Refinement(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RietveldRecipeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) => Some(error),
            Self::Parameter(error) => Some(error),
            Self::Rietveld(error) => Some(error),
            Self::Refinement(error) => Some(error),
            Self::InvalidRecipeName
            | Self::InvalidStageName
            | Self::InvalidRationale
            | Self::InvalidAcceptedTerminations
            | Self::InvalidPlannerNote
            | Self::EmptyRecipe
            | Self::DuplicateStageName { .. }
            | Self::UnauthorizedSelection { .. }
            | Self::MissingConstraintDependency { .. }
            | Self::UnsatisfiedConstraint { .. }
            | Self::InternalInvariant => None,
        }
    }
}

impl From<RietveldGeneralParameterError> for RietveldRecipeError {
    fn from(value: RietveldGeneralParameterError) -> Self {
        Self::Parameter(Box::new(value))
    }
}

impl From<ConstraintError> for RietveldRecipeError {
    fn from(value: ConstraintError) -> Self {
        Self::Constraint(Box::new(value))
    }
}

impl From<crate::RietveldError> for RietveldRecipeError {
    fn from(value: crate::RietveldError) -> Self {
        Self::Rietveld(Box::new(value))
    }
}

impl From<RietveldGeneralRefinementError> for RietveldRecipeError {
    fn from(value: RietveldGeneralRefinementError) -> Self {
        Self::Refinement(Box::new(value))
    }
}
