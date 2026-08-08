//! File-independent crystallographic numerical kernels.
//!
//! This crate accepts explicit cells, Miller indices, atom arrays, and
//! scattering amplitudes. It contains no CIF parser, Python dependency,
//! refinement workflow, GUI state, or external-oracle integration.

pub mod cell;
pub mod intensity_correction;
pub mod p1;
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
    calculate_structure_factor_jvp_with_context, calculate_structure_factor_values,
    calculate_structure_factor_values_with_context,
};
pub use symmetry::{
    CrystalSystem, ExpandedSites, MetricConstraints, Rational, ReflectionFamily, SpaceGroup,
    SymmetryError, SymmetryOperation,
};
