//! Public Rust entry point for `PhaseSmith`.
//!
//! The implementation remains split into focused crates. This facade keeps the
//! component boundaries visible while giving applications one stable dependency
//! and one version to select.

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
/// Le Bail, Rietveld, quantitative, residual, and validation workflows.
pub use phasesmith_workflows as workflows;
