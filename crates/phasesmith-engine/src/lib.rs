//! Native composition boundary for crystallography and powder profiles.
//!
//! `phasesmith-engine` turns crystallographic definitions into calculated
//! powder patterns by composing `phasesmith-crystallography` with
//! `phasesmith-core`. It has no Python, file-format, persistence, refinement,
//! or GUI dependency. Applications normally use it through
//! [`phasesmith::engine`](https://docs.rs/phasesmith/latest/phasesmith/).
//!
//! # Choosing an entry point
//!
//! - [`calculate_structural_pattern`] is the direct one-phase, one-wavelength
//!   calculation.
//! - [`calculate_structural_tof_pattern`] is the direct one-phase, one-bank
//!   neutron TOF calculation with explicit bank geometry and correction.
//! - [`PreparedStructuralPhase`] validates and caches topology for repeated
//!   phase evaluations.
//! - [`PreparedStructuralSpectrum`] adds fixed wavelength components.
//! - [`PreparedStructuralMultiphase`] composes multiple structural phases.
//! - [`PreparedStructuralModel`] is the reusable high-level structural model.
//!
//! Values, dense-Jacobian, JVP, and VJP paths share the same physical model.
//! Prepared types are the intended choice for optimizers and interactive hosts:
//! construct them when topology changes, then reuse them while numerical
//! parameters vary.
//!
//! # Data boundary
//!
//! Inputs use borrowed array views and explicit physical definitions. Results
//! own their values, reflection metadata, and derivative products. File parsing
//! belongs to `phasesmith-io`; application records belong to
//! `phasesmith-model`; refinement orchestration belongs to
//! `phasesmith-workflows`.
//!
//! See the facade's
//! [pattern-composition mathematics](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/pattern_composition/)
//! for the multi-phase, multi-wavelength, sample-physics, and derivative sums.

pub use phasesmith_core as profile;
pub use phasesmith_crystallography as crystallography;

mod prepared_structural_phase;
mod structural_multiphase;
pub mod structural_pattern;
mod structural_spectrum;
pub mod structural_tof;

pub use prepared_structural_phase::{
    PreparedStructuralPatternInputView, PreparedStructuralPhase, StructuralPhaseDefinition,
};
pub use structural_multiphase::{
    PreparedStructuralModel, PreparedStructuralModelInputView, PreparedStructuralMultiphase,
    StructuralCalculationRequest, StructuralCalculationResult, StructuralModelInput,
    StructuralMultiphaseError, StructuralMultiphaseResult,
};
pub use structural_spectrum::{
    PreparedStructuralSpectrum, PreparedStructuralSpectrumInputView, StructuralSpectrumError,
};
pub use structural_tof::{
    StructuralTofDenseResult, StructuralTofError, StructuralTofInputView, StructuralTofJvpResult,
    StructuralTofResult, StructuralTofVjpResult, calculate_structural_tof_pattern,
    calculate_structural_tof_pattern_dense, calculate_structural_tof_pattern_dense_with_context,
    calculate_structural_tof_pattern_jvp, calculate_structural_tof_pattern_jvp_with_context,
    calculate_structural_tof_pattern_vjp, calculate_structural_tof_pattern_vjp_with_context,
    calculate_structural_tof_pattern_with_context,
};

pub use structural_pattern::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, MonochromaticReflectionGeometry,
    StructuralPatternDenseResult, StructuralPatternError, StructuralPatternInputView,
    StructuralPatternJvpResult, StructuralPatternResult, StructuralPatternVjpResult,
    calculate_monochromatic_reflection_geometry, calculate_structural_pattern,
    calculate_structural_pattern_dense, calculate_structural_pattern_dense_with_context,
    calculate_structural_pattern_jvp, calculate_structural_pattern_jvp_with_context,
    calculate_structural_pattern_vjp, calculate_structural_pattern_vjp_with_context,
    calculate_structural_pattern_with_context,
};
