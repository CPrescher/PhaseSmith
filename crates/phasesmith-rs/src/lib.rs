#![doc = include_str!("crate.md")]

/// Task-oriented guides for native Rust consumers and application hosts.
pub mod guide;

/// Numerical profile and background kernels.
pub use phasesmith_core as core;
/// Crystallographic cells, symmetry, scattering, and structure factors.
pub use phasesmith_crystallography as crystallography;
/// Native structural-calculation composition.
pub use phasesmith_engine as engine;
/// Bounded deterministic execution policies.
pub use phasesmith_execution as execution;
/// Native powder-data and CIF input adapters.
pub use phasesmith_io as io;
/// Application-neutral project and pattern records.
pub use phasesmith_model as model;
/// Canonical native project persistence and reporting.
pub use phasesmith_persistence as persistence;
/// Structure search from corrected single-crystal intensities.
#[cfg(feature = "solution")]
pub use phasesmith_solution as solution;
/// Single-crystal detector measurements, masks, geometry and profile kernels.
#[cfg(feature = "spots")]
pub use phasesmith_spots as spots;
/// Le Bail, Rietveld, quantitative, residual, and validation workflows.
pub use phasesmith_workflows as workflows;

/// Optional bundled native HDF5 detector-frame input.
#[cfg(feature = "hdf5")]
pub use phasesmith_hdf5 as hdf5;

/// Independent equidistant-row grouping for lattice discovery.
#[cfg(feature = "row-search")]
pub use phasesmith_row_search as row_search;
