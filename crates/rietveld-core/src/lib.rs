//! Numerical kernels for powder diffraction profile calculation.
//!
//! This crate is deliberately unaware of refinement projects, phases, files,
//! and GSAS-II data structures. It operates on explicit parameters and flat
//! numeric slices.

pub mod profile;

pub use profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, Peak, PeakBatchView, ProfileError,
    ProfilePoint, SupportJacobian, SupportPolicy, SupportRange, accumulate_batch, accumulate_peaks,
    accumulate_values_batch, symmetric_pseudo_voigt,
};
