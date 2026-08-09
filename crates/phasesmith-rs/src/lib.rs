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
/// Le Bail, Rietveld, quantitative, residual, and validation workflows.
pub use phasesmith_workflows as workflows;
