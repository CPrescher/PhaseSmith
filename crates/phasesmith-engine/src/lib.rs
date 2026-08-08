//! Native composition boundary for crystallography and powder profiles.
//!
//! Structural pattern fusion is added in implementation unit 15. Re-exports
//! keep `PyO3` dependent on this facade rather than placing orchestration in the
//! binding crate.

pub use phasesmith_core as profile;
pub use phasesmith_crystallography as crystallography;

pub mod structural_pattern;

pub use structural_pattern::{
    BuiltInScatteringModel, MonochromaticPositionCorrection, StructuralPatternDenseResult,
    StructuralPatternError, StructuralPatternInputView, StructuralPatternJvpResult,
    StructuralPatternResult, StructuralPatternVjpResult, calculate_structural_pattern,
    calculate_structural_pattern_dense, calculate_structural_pattern_dense_with_context,
    calculate_structural_pattern_jvp, calculate_structural_pattern_jvp_with_context,
    calculate_structural_pattern_vjp, calculate_structural_pattern_vjp_with_context,
    calculate_structural_pattern_with_context,
};
