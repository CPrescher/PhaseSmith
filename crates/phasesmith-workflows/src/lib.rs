//! Application-neutral native refinement and project workflows.
//!
//! This crate is the shared host boundary for Python adapters and future
//! desktop adapters. It contains no `PyO3`, Tauri, GUI, or file-format types.

mod constraints;
mod parameters;
mod residuals;

pub use constraints::{
    AffineConstraint, Constraint, ConstraintDerivativeMatrix, ConstraintError, ConstraintTransform,
    FixedConstraint, LinearConstraint, LinearTerm,
};
pub use parameters::{ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec};
pub use residuals::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals};
