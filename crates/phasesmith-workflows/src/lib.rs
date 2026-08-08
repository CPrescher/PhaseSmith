//! Application-neutral native refinement and project workflows.
//!
//! This crate is the shared host boundary for Python adapters and future
//! desktop adapters. It contains no `PyO3`, Tauri, GUI, or file-format types.

mod backgrounds;
mod constraints;
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
pub use parameters::{ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec};
pub use residuals::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals};
pub use runtime::{
    CancellationError, CancellationToken, CheckpointSink, DiagnosticValue, MonotonicClock,
    RefinementEvent, RefinementEventKind, RefinementEventSink, RefinementLimits, RefinementRuntime,
    RefinementStop, RuntimeClock, RuntimeError, TerminationReason,
};
