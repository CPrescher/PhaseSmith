//! Numerical kernels for powder diffraction profile calculation.
//!
//! This crate is deliberately unaware of refinement projects, phases, files,
//! and GSAS-II data structures. It operates on explicit parameters and flat
//! numeric slices.

pub mod profile;

pub use profile::{
    Accumulation, Peak, ProfileError, ProfilePoint, accumulate_peaks, symmetric_pseudo_voigt,
};
