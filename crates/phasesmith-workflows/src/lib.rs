//! Application-neutral native refinement and project workflows.
//!
//! This crate is the shared host boundary for Python adapters and future
//! desktop adapters. It contains no `PyO3`, Tauri, GUI, or file-format types.

mod backgrounds;
mod constraints;
mod lattice;
mod lebail;
mod parameters;
mod residuals;
mod rietveld;
mod rietveld_objective;
mod rietveld_parameters;
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
pub use lattice::{
    CwLatticeGeometry, GeneratedLatticeDomain, LatticeBounds, LatticeError,
    LatticeParameterization, LatticeReflectionDomain, cw_lattice_geometry,
};
pub use lebail::{
    CoincidentReflectionGroup, CovarianceMatrix, IntensityExtractionResult, LeBailCalculation,
    LeBailCheckpoint, LeBailError, LeBailInput, LeBailIterationRecord, LeBailOptions, LeBailPhase,
    LeBailResult, ParameterChange, PhasePatternComponent, ReflectionIntensity,
    build_lebail_parameter_set, build_lebail_parameter_set_with_lattice, calculate_lebail_pattern,
    extract_lebail_intensities, initialize_lebail_intensities, iterate_lebail_once,
    lebail_instrument_parameter_key, lebail_lattice_parameter_key, lebail_phase_scale_key,
    lebail_reflection_position_key, refine_lebail, refine_lebail_with_runtime,
};
pub use parameters::{ParameterBounds, ParameterError, ParameterKey, ParameterSet, ParameterSpec};
pub use residuals::{ResidualError, ResidualEvaluation, ResidualOptions, evaluate_residuals};
pub use rietveld::{
    RietveldCalculation, RietveldCalculationOptions, RietveldError, RietveldInput, RietveldPhase,
    RietveldPhaseCalculation, calculate_rietveld_pattern,
};
pub use rietveld_objective::{PreparedRietveldObjective, RietveldObjectiveError};
pub use rietveld_parameters::{
    RietveldParameterError, RietveldStructuralLayout, RietveldStructuralSelection,
    SiteCoordinateModel,
};
pub use runtime::{
    CancellationError, CancellationToken, CheckpointSink, DiagnosticValue, MonotonicClock,
    RefinementEvent, RefinementEventKind, RefinementEventSink, RefinementLimits, RefinementRuntime,
    RefinementStop, RuntimeClock, RuntimeError, TerminationReason,
};
