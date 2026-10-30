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
mod linear_step;
mod math3;
mod model;
mod peak_completion;
mod peak_density;
mod peak_scan;
pub use peak_scan::PreparedAdaptivePeaks;
mod phase_checkpoint;
mod phase_origin;
pub use phase_checkpoint::{PhaseCheckpoint, PhaseCheckpointSelector, checkpoint_map_cosine};
mod phase_quality;
mod phase_schedule;
mod phase_score;
pub use dual_space::{PhasingExecution, dual_space_with_execution};
pub use phase_schedule::{PhaseSliceAudit, phase_cycle_quota};
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
    DualSpaceResult, PeakSelection, adaptive_peaks, atomic_projection, dual_space,
    dual_space_monitored, dual_space_with_prediction,
};
pub use fourier::{FlipOptions, FlipResult, charge_flip, extract_peaks, transform_3d};
pub use linear_step::{CachedNormalStep, cached_normal_step, iterative_normal_step};
pub use model::{ModelEvaluation, PolishedModel, evaluate_model, polish_model};
pub use peak_completion::{
    ChemicalScale, PeakCompletionCandidate, PeakCompletionResult, PreparedPeakCompletion,
    chemical_density_scales,
};
pub use peak_density::{IntegratedPeakDensity, PeakIntegral, ball_transform, density_label_mask};
pub use phase_origin::{
    PhaseOriginCandidate, PhaseOriginEvaluation, PhaseOriginMap, PreparedPhaseOrigin,
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
    SharedSearchOptions, SharedSearchSummary, solve_elements,
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
