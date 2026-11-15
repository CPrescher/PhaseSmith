//! Native single-crystal diffraction kernels: detector measurements, structure
//! search and reciprocal-lattice row grouping. File adapters and complete workflow
//! orchestration remain separate. Select capabilities through Cargo features.

/// Position-only multi-grain lattice discovery and frozen-pool validation.
#[cfg(feature = "grains")]
pub mod grains;
/// Bravais-lattice search and symmetry-constrained cell refinement.
#[cfg(feature = "grains")]
pub mod lattice_symmetry;
/// Reflection prediction and intensity measurement for refined lattices.
#[cfg(feature = "grains")]
pub mod measurement;
/// Independent published DAFi equidistant-row grouping.
// Preserve the former row crate's lint policy during the module-only move.
#[cfg(feature = "row-search")]
#[allow(missing_docs, clippy::all, clippy::pedantic)]
pub mod row_search;
/// Bounded non-resonant X-ray structure search and candidate polishing.
#[cfg(feature = "solution")]
pub mod solution;
/// Detector, geometry, integration, profile and scaling kernels.
#[cfg(feature = "spots")]
pub mod spots;
