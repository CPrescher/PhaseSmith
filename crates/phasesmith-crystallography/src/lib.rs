//! File-independent crystallographic numerical kernels.
//!
//! This crate accepts explicit cells, Miller indices, atom arrays, and
//! scattering amplitudes. It contains no CIF parser, Python dependency,
//! refinement workflow, GUI state, or external-oracle integration.
//! Applications normally use it through
//! [`phasesmith::crystallography`](https://docs.rs/phasesmith/latest/phasesmith/).
//!
//! # From cell to reciprocal geometry
//!
//! ```
//! use phasesmith_crystallography::UnitCell;
//!
//! let geometry = UnitCell {
//!     a_angstrom: 5.0,
//!     b_angstrom: 5.0,
//!     c_angstrom: 5.0,
//!     alpha_deg: 90.0,
//!     beta_deg: 90.0,
//!     gamma_deg: 90.0,
//! }
//! .geometry()?;
//! let (d_angstrom, _) = geometry.d_spacing_and_derivatives([1, 0, 0])?;
//! assert!((d_angstrom - 5.0).abs() < 1.0e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Major API groups
//!
//! - [`cell`] derives direct and reciprocal metrics and analytical cell
//!   derivatives. Reciprocal bases omit the `2π` factor.
//! - [`symmetry`] represents exact rational symmetry operations, space groups,
//!   site expansion, and metric constraints.
//! - [`reflection`] performs bounded reflection generation and absence/family
//!   handling.
//! - [`scattering`] prepares tabulated X-ray and neutron scattering amplitudes.
//! - [`structure_factor`] provides values, dense Jacobians, JVPs, and VJPs for
//!   general symmetry; [`p1`] provides the specialized P1 path.
//! - [`intensity_correction`] applies explicit integrated-intensity models.
//!
//! The facade's
//! [crystallography mathematics guide](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/crystallography/)
//! collects the cell, symmetry, scattering, structure-factor, correction, and
//! derivative equations in one place.

pub mod cell;
pub mod intensity_correction;
pub mod p1;
pub mod powder_structure_factor;
pub mod reflection;
pub mod scattering;
pub mod structure_factor;
pub mod symmetry;

pub use cell::{CELL_PARAMETER_COUNT, CellError, CellGeometry, Matrix3, UnitCell};
pub use intensity_correction::{
    IntegratedIntensityCorrection, IntegratedIntensityCorrectionError,
    IntegratedIntensityCorrectionModel,
};
pub use p1::{
    P1BatchError, P1BatchView, P1DenseResult, P1JvpResult, P1ParameterLayout, P1Values,
    P1VjpResult, calculate_p1_dense, calculate_p1_intensity_vjp, calculate_p1_jvp,
    calculate_p1_values,
};
pub use reflection::{
    GeneratedReflection, PreparedReflectionGenerator, ReflectionGenerationError, ReflectionRange,
};
pub use scattering::{
    NEUTRON_TABLE_PROVENANCE, NeutronSpeciesMetadata, PreparedNeutronScattering,
    PreparedXrayScattering, ScatteringAmplitudeUnit, ScatteringBatch, ScatteringError,
    ScatteringTableProvenance, XRAY_MAX_S_INVERSE_ANGSTROM, XRAY_TABLE_PROVENANCE,
    XraySpeciesMetadata, neutron_species_metadata, xray_species_metadata,
};
pub use structure_factor::{
    StructureFactorBatchError, StructureFactorBatchView, StructureFactorDenseResult,
    StructureFactorJvpResult, StructureFactorValues, StructureFactorVjpResult,
    calculate_structure_factor_dense, calculate_structure_factor_dense_with_context,
    calculate_structure_factor_intensity_vjp,
    calculate_structure_factor_intensity_vjp_with_context, calculate_structure_factor_jvp,
    calculate_structure_factor_jvp_with_context, calculate_structure_factor_selected_with_context,
    calculate_structure_factor_values, calculate_structure_factor_values_with_context,
};
pub use symmetry::{
    CrystalSystem, ExpandedSites, MetricConstraints, Rational, ReflectionFamily, SpaceGroup,
    SymmetryError, SymmetryOperation,
};

pub use powder_structure_factor::{
    calculate_powder_structure_factor_dense_with_context,
    calculate_powder_structure_factor_intensity_vjp_with_context,
    calculate_powder_structure_factor_jvp_with_context,
    calculate_powder_structure_factor_selected_with_context,
    calculate_powder_structure_factor_values_with_context,
};
