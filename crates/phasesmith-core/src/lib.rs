//! Numerical kernels for powder diffraction profile calculation.
//!
//! This crate is deliberately unaware of refinement projects, phases, files,
//! and GSAS-II data structures. It operates on explicit parameters and flat
//! numeric slices.

pub mod background;
pub mod cw;
pub mod cw_components;
pub mod cw_contributions;
pub mod cw_fcj;
pub mod fcj;
pub mod profile;
pub mod radiation;
pub mod tch;
pub mod tof;

pub use background::{BackgroundError, smooth_bruckner};
pub use cw::{
    ConstantWavelengthInstrument, CwBatchError, CwError, CwProfileParameters,
    CwReflectionBatchView, accumulate_cw_batch,
};
pub use cw_components::{
    CwComponentsBatchError, accumulate_cw_components_batch, accumulate_cw_fcj_components_batch,
};
pub use cw_contributions::{
    CwContributionArrays, CwContributionsError, CwContributionsView,
    accumulate_cw_contributions_batch, accumulate_cw_contributions_batch_with_context,
    accumulate_cw_fcj_contributions_batch, accumulate_cw_fcj_contributions_batch_with_context,
};
pub use cw_fcj::{CwFcjBatchError, accumulate_cw_fcj_batch};
pub use fcj::{FcjError, FcjGeometry, FcjProfile, FcjProfilePoint};

pub use profile::{
    Accumulation, DenseJacobian, GridView, PatternDerivatives, Peak, PeakBatchView, ProfileError,
    ProfilePoint, SupportJacobian, SupportPolicy, SupportRange, accumulate_batch, accumulate_peaks,
    accumulate_values_batch, symmetric_pseudo_voigt,
};
pub use radiation::{WavelengthComponentsError, WavelengthComponentsView};
pub use tch::{
    TchError, TchPeakBatchView, TchProfilePoint, TchShape, TchWidths, accumulate_tch_batch,
    tch_pseudo_voigt,
};
pub use tof::{
    TOF_GLOBAL_PARAMETER_COUNT, TofError, TofInstrument, TofProfile, TofProfileParameters,
    TofProfilePoint, accumulate_tof_batch, accumulate_tof_batch_with_context,
};
