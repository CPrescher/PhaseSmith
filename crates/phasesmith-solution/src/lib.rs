//! Experimental, bounded, file-independent X-ray structure solution.
//!
//! Equations and scope: `docs/experimental-structure-solution.md`.
//! Input symmetry expands intensities; candidates are full-cell P1 models.
//! No external crystallographic program is imported, executed, or required.

mod data;
mod fourier;
mod model;
mod solver;

use std::fmt::{Display, Formatter};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub use data::{Observation, PreparedData, prepare_data};
pub use fourier::{FlipOptions, FlipResult, charge_flip, extract_peaks};
pub use model::{ModelEvaluation, PolishedModel, evaluate_model, polish_model};
pub use solver::{Candidate, Solution, SolveOptions, solve};

/// Thread-safe cooperative cancellation between bounded numerical steps.
#[derive(Clone, Debug, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    /// Request a stop; true only for the first request.
    #[must_use]
    pub fn cancel(&self) -> bool {
        !self.0.swap(true, Ordering::AcqRel)
    }
    /// Whether a stop has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

fn stopped(deadline: Option<std::time::Instant>, cancellation: Option<&Cancellation>) -> bool {
    deadline.is_some_and(|end| std::time::Instant::now() >= end)
        || cancellation.is_some_and(Cancellation::is_cancelled)
}

/// Invalid input, numerical failure, or a declared resource limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolutionError(pub String);

impl Display for SolutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SolutionError {}

fn error(message: impl Into<String>) -> SolutionError {
    SolutionError(message.into())
}
