//! Application-neutral native refinement and project workflows.
//!
//! This crate is the shared host boundary for Python adapters and future
//! desktop adapters. It contains no `PyO3`, Tauri, GUI, or file-format types.

mod backgrounds;
mod constraints;
mod lebail;
mod parameters;
mod residuals;
mod runtime;

pub use backgrounds::{
    AmorphousBackground, AmorphousPeak, BackgroundBasis, BackgroundError, BackgroundModel,
    ChebyshevBackground, CompositeBackground, DifferentiableBackground, PointBackground,
    PolynomialBackground,
};
pub use constraints::{
    AffineConstraint, Constraint, ConstraintDerivativeMatrix, ConstraintError, ConstraintTransform,
    FixedConstraint, LinearConstraint, LinearTerm,
};
pub use lebail::{
    CoincidentReflectionGroup, IntensityExtractionResult, LeBailCalculation, LeBailCheckpoint,
    LeBailError, LeBailInput, LeBailIterationRecord, LeBailOptions, LeBailPhase, LeBailResult,
    PhasePatternComponent, ReflectionIntensity, calculate_lebail_pattern,
    extract_lebail_intensities, initialize_lebail_intensities, iterate_lebail_once, refine_lebail,
    refine_lebail_with_runtime,
};
pub use parameters::{ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec};
pub use residuals::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals};
pub use runtime::{
    CancellationError, CancellationToken, CheckpointSink, DiagnosticValue, MonotonicClock,
    RefinementEvent, RefinementEventKind, RefinementEventSink, RefinementLimits, RefinementRuntime,
    RefinementStop, RuntimeClock, RuntimeError, TerminationReason,
};
