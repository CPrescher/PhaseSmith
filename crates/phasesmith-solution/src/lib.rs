//! Experimental, bounded, file-independent X-ray structure solution.
//!
//! Equations and scope: `docs/experimental-structure-solution.md`.
//! Supplied symmetry also constrains optional site/ADP refinement after P1 phasing.
//! No external crystallographic program is imported, executed, or required.

mod adp_refinement;
mod completion;
mod completion_orbit;
mod data;
mod dual_space;
mod fourier;
mod math3;
mod model;
mod peak_density;
mod shared_pool;
mod shared_search;
mod solver;
mod staged_refinement;
mod symmetry_model;

use std::fmt::{Display, Formatter};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub use adp_refinement::{
    AdpMode, PreparedSymmetricModel, RefinementSummary, SymmetricEvaluation, SymmetricRefinement,
};
pub use completion::{CompletionSummary, complete_candidate};
pub use completion_orbit::{
    DifferenceMap, OrbitEvaluation, PreparedOrbitSearch, project_fixed_sets,
};
pub use data::{Observation, PreparedData, prepare_data};
pub use dual_space::{DualSpaceOptions, DualSpaceResult, atomic_projection, dual_space};
pub use fourier::{FlipOptions, FlipResult, charge_flip, extract_peaks};
pub use model::{ModelEvaluation, PolishedModel, evaluate_model, polish_model};
pub use peak_density::{IntegratedPeakDensity, PeakIntegral, ball_transform, density_label_mask};
pub use shared_pool::PreparedElementPool;
pub use shared_search::{
    DensityCalibration, SharedMove, SharedSearchOptions, SharedSearchSummary, solve_elements,
};
pub use solver::{Candidate, PhasingMethod, PhasingTrial, Solution, SolveOptions, solve};
pub use staged_refinement::screening_shortlist;
pub use symmetry_model::{
    AsymmetricSite, SymmetryImage, SymmetryModel, recover_symmetry, recover_symmetry_hypotheses,
};

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
