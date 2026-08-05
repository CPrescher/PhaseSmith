//! Numerical kernels for powder diffraction profile calculation.
//!
//! This crate is deliberately unaware of refinement projects, phases, files,
//! and GSAS-II data structures. It operates on explicit parameters and flat
//! numeric slices.

pub mod cw;
pub mod cw_fcj;
pub mod fcj;
pub mod profile;
pub mod tch;

pub use cw::{
    ConstantWavelengthInstrument, CwBatchError, CwError, CwProfileParameters,
    CwReflectionBatchView, accumulate_cw_batch,
};
pub use cw_fcj::{CwFcjBatchError, accumulate_cw_fcj_batch};
pub use fcj::{FcjError, FcjGeometry, FcjProfile, FcjProfilePoint};

pub use profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, Peak, PeakBatchView, ProfileError,
    ProfilePoint, SupportJacobian, SupportPolicy, SupportRange, accumulate_batch, accumulate_peaks,
    accumulate_values_batch, symmetric_pseudo_voigt,
};
pub use tch::{
    TchError, TchPeakBatchView, TchProfilePoint, TchShape, TchWidths, accumulate_tch_batch,
    tch_pseudo_voigt,
};
