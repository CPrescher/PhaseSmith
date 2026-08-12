//! Application-neutral native refinement and project workflows.
//!
//! This crate is the shared host boundary for Python adapters and future
//! native application consumers. It contains no `PyO3`, Tauri, GUI, or
//! file-format types.
//! Applications normally use it through
//! [`phasesmith::workflows`](https://docs.rs/phasesmith/latest/phasesmith/).
//!
//! # Workflow families
//!
//! - **Backgrounds:** differentiable Chebyshev, polynomial, point, amorphous,
//!   and composite models.
//! - **Le Bail:** intensity extraction with optional profile/lattice
//!   refinement, covariance, checkpoints, and dynamic reflection domains.
//! - **TOF Le Bail:** typed microsecond-domain fixed-instrument calculation and
//!   nonnegative intensity extraction with fused d/instrument derivatives.
//! - **Rietveld:** calculation, prepared objective products, structural and
//!   general refinement, staged recipes, and joint multi-histogram refinement.
//! - **Quantitative analysis:** validated Hill–Howard phase fractions.
//! - **Infrastructure:** typed parameters, bounds, constraints, residuals,
//!   cancellation, limits, events, clocks, diagnostics, and checkpoints.
//!
//! # Selecting the level
//!
//! Use `calculate_*` functions for a deterministic forward calculation. Use a
//! `Prepared*Objective` when integrating a custom optimizer and consuming
//! values/JVP/VJP products directly. Use `refine_*` for the built-in solver, and
//! the corresponding `_with_runtime` variant when an application needs
//! cancellation, progress events, or checkpoint sinks.
//!
//! [`LeBailInput`] and [`RietveldInput`] own validated observations and domain
//! state. Result/checkpoint records and structured event diagnostics are the
//! stable programmatic output; human-readable event messages should not be
//! parsed.
//!
//! # Application integration
//!
//! Workflows are synchronous and application-neutral. A desktop or async host
//! runs them on its chosen worker mechanism, connects
//! [`RefinementRuntime`] to its cancellation/event bridge, and uses
//! `phasesmith-persistence` for native project state. The `joint_pbso4` example
//! in this package demonstrates a complete Rust-only X-ray/neutron refinement.
//! The facade provides collected mathematical references for
//! [refinement](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/refinement/)
//! and
//! [background models](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/backgrounds/).

mod backgrounds;
mod constraints;
mod lattice;
mod lebail;
mod parameters;
mod phase_scale_estimation;
mod profile_estimation;
mod quantitative;
mod residuals;
mod rietveld;
mod rietveld_general_objective;
mod rietveld_general_parameters;
mod rietveld_general_solver;
mod rietveld_joint;
mod rietveld_joint_solver;
mod rietveld_objective;
mod rietveld_parameters;
mod rietveld_project;
mod rietveld_recipe;
mod rietveld_solver;
mod runtime;
mod sample_physics;
mod tof_lebail;

pub use backgrounds::{
    AmorphousBackground, AmorphousPeak, BackgroundBasis, BackgroundError, BackgroundModel,
    ChebyshevBackground, CompositeBackground, DifferentiableBackground, PointBackground,
    PolynomialBackground,
};
pub use constraints::{
    AffineConstraint, Constraint, ConstraintDerivativeMatrix, ConstraintError, ConstraintTransform,
    FixedConstraint, LinearConstraint, LinearTerm,
};
pub use lattice::{
    CwLatticeGeometry, GeneratedLatticeDomain, LatticeBounds, LatticeError,
    LatticeParameterization, LatticeReflectionDomain, cw_lattice_geometry,
};
pub use lebail::{
    CoincidentReflectionGroup, CovarianceMatrix, IntensityExtractionResult, LeBailCalculation,
    LeBailCheckpoint, LeBailError, LeBailInput, LeBailIterationRecord, LeBailOptions, LeBailPhase,
    LeBailResult, ParameterChange, PhasePatternComponent, ReflectionIntensity,
    build_lebail_parameter_set, build_lebail_parameter_set_with_lattice, calculate_lebail_pattern,
    calculate_lebail_pattern_with_background, extract_lebail_intensities,
    initialize_lebail_intensities, iterate_lebail_once, lebail_background_parameter_key,
    lebail_instrument_parameter_key, lebail_lattice_parameter_key, lebail_phase_scale_key,
    lebail_reflection_position_key, refine_lebail, refine_lebail_with_runtime,
};
pub use parameters::{ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec};
pub use phase_scale_estimation::{
    PhaseScaleEstimationError, PhaseScaleEstimationResult, estimate_initial_phase_scales,
};
pub use profile_estimation::{
    ProfileEstimationError, ProfileEstimationInput, ProfileEstimationMode,
    ProfileEstimationOptions, ProfileEstimationResult, ProfileEstimationStage,
    ProfileEstimationStageKind, estimate_effective_profile, starting_profile_from_fwhm,
};
pub use quantitative::{
    PhaseWeightFraction, QuantitativeError, QuantitativePhase, QuantitativePhaseAnalysis,
    quantitative_phase_analysis, quantitative_phase_analysis_with_covariance,
};
pub use residuals::{
    ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals, evaluate_tof_residuals,
};
pub use rietveld::{
    RietveldCalculation, RietveldCalculationOptions, RietveldError, RietveldInput, RietveldPhase,
    RietveldPhaseCalculation, RietveldTopologyChange, calculate_rietveld_pattern,
};
pub use rietveld_general_objective::{
    DEFAULT_MAX_LINEARIZATION_ELEMENTS, PreparedGeneralFreeLinearization,
    PreparedGeneralRietveldObjective, RietveldGeneralObjectiveError,
};
pub use rietveld_general_parameters::{
    RietveldGeneralParameterError, RietveldInstrumentParameter, RietveldParameterLayout,
    RietveldParameterSelection,
};
pub use rietveld_general_solver::{
    RietveldCovarianceMatrix, RietveldCovarianceOptions, RietveldGeneralCheckpoint,
    RietveldGeneralRefinementError, RietveldGeneralRefinementResult, RietveldParameterCorrelation,
    refine_general_rietveld, refine_general_rietveld_with_runtime,
};
pub use rietveld_joint::{
    JointRietveldError, JointRietveldGradient, JointRietveldHistogram, JointRietveldLayout,
    JointRietveldProduct, PreparedJointRietveldObjective,
};
pub use rietveld_joint_solver::{
    JointRietveldCheckpoint, JointRietveldIterationRecord, JointRietveldMetrics,
    JointRietveldRefinementError, JointRietveldRefinementOptions, JointRietveldRefinementResult,
    JointRietveldTopologyChange, refine_joint_rietveld, refine_joint_rietveld_with_runtime,
};
pub use rietveld_objective::{
    PreparedRietveldLinearization, PreparedRietveldObjective, RietveldObjectiveError,
};
pub use rietveld_parameters::{
    RietveldParameterError, RietveldStructuralLayout, RietveldStructuralSelection,
    SiteCoordinateModel,
};
pub use rietveld_project::{RietveldAnalysis, RietveldProjectError, RietveldProjectState};
pub use rietveld_recipe::{
    RietveldRecipe, RietveldRecipeError, RietveldRecipeMode, RietveldRecipeSinks, RietveldStage,
    RietveldStageResult, RietveldWorkflowResult, intelligent_rietveld_recipe, run_rietveld_recipe,
    run_rietveld_recipe_with_sinks, validate_rietveld_recipe,
};
pub use rietveld_solver::{
    RietveldCheckpoint, RietveldIterationRecord, RietveldRefinementError,
    RietveldRefinementOptions, RietveldRefinementResult, refine_rietveld,
    refine_rietveld_with_runtime,
};
pub use runtime::{
    CancellationError, CancellationToken, CheckpointSink, DiagnosticValue, MonotonicClock,
    RefinementEvent, RefinementEventKind, RefinementEventSink, RefinementLimits, RefinementRuntime,
    RefinementStop, RuntimeClock, RuntimeError, TerminationReason,
};
pub use sample_physics::{
    EvaluatedSamplePhysics, RietveldSamplePhysicsModel, SamplePhysicsError, SamplePhysicsParameter,
};
pub use tof_lebail::{
    TofChebyshevBackground, TofChebyshevBasis, TofLeBailCalculation, TofLeBailCheckpoint,
    TofLeBailError, TofLeBailInput, TofLeBailIterationRecord, TofLeBailOptions, TofLeBailPhase,
    TofLeBailResult, TofReflectionIntensity, calculate_tof_lebail_pattern, refine_tof_lebail,
    refine_tof_lebail_with_runtime,
};
