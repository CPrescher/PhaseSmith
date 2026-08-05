//! Native composition boundary for crystallography and powder profiles.
//!
//! Structural pattern fusion is added in implementation unit 15. Re-exports
//! keep `PyO3` dependent on this facade rather than placing orchestration in the
//! binding crate.

pub use rietveld_core as profile;
pub use rietveld_crystallography as crystallography;

pub mod structural_pattern;

pub use structural_pattern::{
    BuiltInScatteringModel, StructuralPatternError, StructuralPatternInputView,
    StructuralPatternJvpResult, StructuralPatternResult, StructuralPatternVjpResult,
    calculate_structural_pattern, calculate_structural_pattern_jvp,
    calculate_structural_pattern_vjp,
};
