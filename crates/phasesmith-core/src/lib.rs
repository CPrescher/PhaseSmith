//! Numerical kernels for powder diffraction profile calculation.
//!
//! `phasesmith-core` is the lowest numerical layer. It operates on explicit
//! parameters and flat borrowed slices and is deliberately unaware of files,
//! phases, refinement projects, Python, and GUI state. Most applications should
//! depend on the [`phasesmith` facade](https://docs.rs/phasesmith/) and reach
//! this crate through `phasesmith::core`.
//!
//! # Capabilities
//!
//! - symmetric pseudo-Voigt and Thompson–Cox–Hastings profiles;
//! - constant-wavelength U/V/W/X/Y broadening;
//! - Finger–Cox–Jephcoat axial asymmetry;
//! - fixed wavelength components and neutron time-of-flight profiles;
//! - smooth Bruckner background estimation;
//! - fused pattern values and analytical derivative storage.
//!
//! # Quick start
//!
//! ```
//! use phasesmith_core::{Peak, accumulate_peaks};
//!
//! let x = [23.9, 24.0, 24.1];
//! let peaks = [Peak {
//!     position: 24.0,
//!     intensity: 100.0,
//!     fwhm: 0.1,
//!     eta: 0.4,
//! }];
//! let result = accumulate_peaks(&x, &peaks, 20.0)?;
//!
//! assert_eq!(result.y.len(), x.len());
//! assert_eq!(result.derivatives.local.parameter_count, 4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`GridView`] and the various `*BatchView` types validate borrowed arrays
//! once before entering hot kernels. Local derivatives use sparse finite-support
//! storage; call [`SupportJacobian::to_dense`] only when a dense allocation is
//! actually required. See [`profile`] for support and derivative conventions.
//! The facade's
//! [profile mathematics guide](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/peak_profiles/)
//! derives every implemented profile and broadening equation together.

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
    CwContributionArrays, CwContributionsError, CwContributionsView, OwnedCwContributionArrays,
    OwnedCwContributions, accumulate_cw_contributions_batch,
    accumulate_cw_contributions_batch_with_context, accumulate_cw_fcj_contributions_batch,
    accumulate_cw_fcj_contributions_batch_with_context,
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
    TOF_GLOBAL_PARAMETER_COUNT, TOF_GLOBAL_PARAMETER_NAMES,
    TOF_INCIDENT_SPECTRUM_COEFFICIENT_COUNT, TofBankGeometry, TofError, TofIncidentSpectrum,
    TofIncidentSpectrumError, TofIncidentSpectrumPoint, TofInstrument, TofInstrumentParameter,
    TofProfile, TofProfileParameters, TofProfilePoint, accumulate_tof_batch,
    accumulate_tof_batch_with_context,
};
