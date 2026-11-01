//! Bounded, file-independent single-crystal X-ray structure search.
//!
//! Equations and scope: `docs/experimental-structure-solution.md`.
//! Supplied symmetry also constrains optional site/ADP refinement after P1 phasing.
//! No external crystallographic program is imported, executed, or required.

mod adp_refinement;
mod atomic_mask;
mod completion;
mod completion_orbit;
mod data;
mod dual_space;
mod execution;
mod fourier;
pub use execution::{WorkerStats, with_solution_workers};
mod linear_step;
mod math3;
mod memory;
pub use memory::{
    DEFAULT_DENSE_MEMORY_BUDGET_BYTES, DenseMemoryBudget, MAX_DENSE_MEMORY_BUDGET_BYTES,
    SolutionMemoryEstimate, estimate_solution_memory,
};
mod model;
mod model_consensus;
pub use model_consensus::model_agreement_distance;
mod peak_completion;
mod peak_density;
mod peak_scan;
mod refinement_families;
pub use peak_scan::PreparedAdaptivePeaks;
mod phase_checkpoint;
mod phase_interpolation;
mod phase_only;
mod phase_origin;
pub use phase_checkpoint::{PhaseCheckpoint, PhaseCheckpointSelector, checkpoint_map_cosine};
pub use phase_interpolation::interpolate_phase_density;
pub use phase_only::{
    PhaseOnlyPolicy, PhaseOnlyResult, phase_only, phase_only_with_policy, phase_only_with_workers,
};
mod phase_quality;
mod phase_schedule;
pub use phase_schedule::{late_phase_plateau_ready, phase_plateau_ready};
mod phase_score;
pub use dual_space::{PhasingExecution, dual_space_with_execution};
pub use phase_schedule::{PhaseSliceAudit, phase_cycle_quota, phase_cycle_quota_with_short_starts};
pub use phase_score::{PhaseQualityValue, PreparedPhaseScore};
mod reflection_prediction;
mod shared_pool;
mod shared_search;
mod solver;
mod staged_refinement;
mod symmetry_model;
mod symmetry_recycling;
mod tangent;

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
pub use dual_space::{
    AdaptivePeakHistory, CheckpointCallback, DualSpaceCheckpoint, DualSpaceOptions,
    DualSpaceResult, PeakSelection, adaptive_peaks, atomic_projection, atomic_projection_cached,
    dual_space, dual_space_monitored, dual_space_with_prediction,
};
pub use fourier::{
    FlipOptions, FlipResult, charge_flip, extract_peaks, real_density_transform, transform_3d,
};
pub use linear_step::{
    CachedNormalStep, blocked_normal_step, cached_normal_step, iterative_normal_step,
};
pub use model::{
    ModelEvaluation, PolishedModel, evaluate_model, evaluate_model_with_memory_budget,
    polish_model, polish_model_with_memory_budget,
};
pub use peak_completion::{
    AssignmentAlternative, AssignmentAlternatives, ChemicalScale, JointAssignment,
    JointAssignments, PeakCompletionCandidate, PeakCompletionResult, PreparedPeakCompletion,
    chemical_density_scales,
};
pub use peak_density::{IntegratedPeakDensity, PeakIntegral, ball_transform, density_label_mask};
pub use phase_origin::{
    PhaseOriginCandidate, PhaseOriginDensity, PhaseOriginEvaluation, PhaseOriginMap,
    PreparedPhaseOrigin,
};
pub use phase_quality::{
    PhaseQualityScore, PhaseQualityTrace, PreparedPhaseQuality, WilsonAudit, WilsonNormalization,
    WilsonShell, normalize_wilson,
};
pub use reflection_prediction::{
    PredictedCoefficient, PredictionAudit, PredictionStep, PreparedReflectionPrediction,
    prediction_coefficient,
};
pub use rustfft::num_complex::Complex as FourierComplex;
pub use shared_pool::PreparedElementPool;
pub use shared_search::{
    DensityCalibration, PhaseOriginAudit, PhaseSettingAudit, RecyclingAudit, SharedMove,
    SharedSearchOptions, SharedSearchSummary, solve_elements, solve_elements_from_phase_map,
};
pub use solver::{Candidate, PhasingMethod, PhasingTrial, Solution, SolveOptions, solve};
pub use staged_refinement::screening_shortlist;
pub use symmetry_model::{
    AsymmetricSite, SymmetryImage, SymmetryModel, recover_symmetry, recover_symmetry_hypotheses,
};
pub use symmetry_recycling::{RecycledDensity, recycle_density, recycle_density_guided};
pub use tangent::{
    NegativeQuartet, PhaseTerm, PhaseTriplet, PreparedTangent, TangentAudit, TangentEvaluation,
    TangentStep,
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
