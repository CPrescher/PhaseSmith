//! Bounded native CIF import into parser-independent crystallographic records.

mod import;
mod syntax;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use phasesmith_crystallography::{CellError, SpaceGroup, SymmetryError, UnitCell};

use crate::SpaceGroupLookupError;

pub use import::{parse_cif_text, read_cif_file};

/// Native CIF parser implementation identifier.
pub const NATIVE_CIF_BACKEND: &str = "phasesmith-native";
/// Version of the native CIF adapter contract.
pub const NATIVE_CIF_BACKEND_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Resource limits checked before and during CIF parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CifReadLimits {
    /// Maximum UTF-8 byte count.
    pub max_bytes: usize,
    /// Maximum number of data blocks.
    pub max_blocks: usize,
    /// Maximum row count of any loop.
    pub max_loop_rows: usize,
    /// Maximum atom or anisotropic-site rows.
    pub max_atom_sites: usize,
}

impl Default for CifReadLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_blocks: 100,
            max_loop_rows: 1_000_000,
            max_atom_sites: 100_000,
        }
    }
}

impl CifReadLimits {
    /// Validate that every limit is positive.
    ///
    /// # Errors
    ///
    /// Returns [`CifIoError::InvalidLimits`] when any limit is zero.
    pub fn validate(self) -> Result<(), CifIoError> {
        if self.max_bytes == 0
            || self.max_blocks == 0
            || self.max_loop_rows == 0
            || self.max_atom_sites == 0
        {
            return Err(CifIoError::InvalidLimits);
        }
        Ok(())
    }
}

/// Stable diagnostic severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CifDiagnosticSeverity {
    /// Recoverable condition visible to the caller.
    Warning,
    /// Non-recoverable condition retained in a partial record.
    Error,
}

/// One stable CIF import diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CifDiagnostic {
    /// Warning or error severity.
    pub severity: CifDiagnosticSeverity,
    /// Stable machine-readable code.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// Related CIF tag when applicable.
    pub tag: Option<String>,
    /// Zero-based loop row when applicable.
    pub row: Option<usize>,
}

impl CifDiagnostic {
    fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: CifDiagnosticSeverity::Warning,
            code: code.into(),
            message: message.into(),
            tag: None,
            row: None,
        }
    }

    fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    fn with_row(mut self, row: usize) -> Self {
        self.row = Some(row);
        self
    }
}

/// Source provenance retained after CIF parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CifStructureSource {
    /// Source format, currently `CIF`.
    pub format: String,
    /// Selected data-block name.
    pub block_name: String,
    /// Parser backend identifier.
    pub backend: String,
    /// Parser backend version.
    pub backend_version: String,
    /// Source path when read from a file.
    pub source_path: Option<PathBuf>,
}

/// Original CIF displacement convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplacementConvention {
    /// CIF `U_ij` components.
    CifU,
    /// CIF `B_ij` components converted to `U_ij`.
    CifB,
}

/// Fixed anisotropic displacement attached to one site.
#[derive(Clone, Debug, PartialEq)]
pub struct CifAnisotropicDisplacement {
    /// CIF U tensor in component order `11,22,33,23,13,12`.
    pub u_cif_angstrom2: [f64; 6],
    /// Convention present in the source file.
    pub source_convention: DisplacementConvention,
    /// Optional standard uncertainties in the same component order.
    pub standard_uncertainty: [Option<f64>; 6],
}

/// One independent atom site imported from CIF.
#[derive(Clone, Debug, PartialEq)]
pub struct CifAtomSite {
    /// Unique stable site identifier within the structure.
    pub site_id: String,
    /// Original CIF label before permissive duplicate renaming.
    pub source_label: String,
    /// Original atom type symbol.
    pub type_symbol: String,
    /// Parsed element symbol.
    pub element_symbol: String,
    /// Fractional coordinates in the selected cell.
    pub fractional_xyz: [f64; 3],
    /// Site occupancy.
    pub occupancy: f64,
    /// Optional isotropic U in square ångströms.
    pub u_iso_angstrom2: Option<f64>,
    /// Optional fixed anisotropic CIF U tensor.
    pub anisotropic_displacement: Option<CifAnisotropicDisplacement>,
    /// Parsed formal charge.
    pub charge: Option<i32>,
    /// Parsed isotope mass number.
    pub isotope: Option<u32>,
    /// Optional disorder group.
    pub disorder_group: Option<String>,
    /// Optional fractional-coordinate standard uncertainties.
    pub fractional_xyz_standard_uncertainty: [Option<f64>; 3],
    /// Optional occupancy standard uncertainty.
    pub occupancy_standard_uncertainty: Option<f64>,
    /// Optional isotropic-U standard uncertainty.
    pub u_iso_standard_uncertainty: Option<f64>,
}

/// Parser-independent native crystallographic structure.
#[derive(Clone, Debug, PartialEq)]
pub struct CifStructure {
    /// Stable ID derived from the selected block name.
    pub structure_id: String,
    /// Human-readable chemical/phase name.
    pub name: String,
    /// Validated direct unit cell.
    pub cell: UnitCell,
    /// Exact validated conventional symmetry operations.
    pub space_group: SpaceGroup,
    /// Independent atom sites in source order.
    pub sites: Vec<CifAtomSite>,
    /// Source provenance.
    pub source: CifStructureSource,
    /// Optional standard uncertainties for `a,b,c,alpha,beta,gamma`.
    pub cell_standard_uncertainties: [Option<f64>; 6],
    /// Import diagnostics also returned at the top level.
    pub diagnostics: Vec<CifDiagnostic>,
    /// Small textual metadata extracted from CIF.
    pub metadata: BTreeMap<String, String>,
}

/// One selected CIF structure plus block-selection context.
#[derive(Clone, Debug, PartialEq)]
pub struct CifReadResult {
    /// Imported native structure.
    pub structure: CifStructure,
    /// Visible import diagnostics.
    pub diagnostics: Vec<CifDiagnostic>,
    /// Selected display block name.
    pub selected_block: String,
    /// All display block names in source order.
    pub available_blocks: Vec<String>,
}

/// Native CIF syntax, limit, lookup, or domain failure.
#[derive(Debug)]
pub enum CifIoError {
    /// One or more configured limits are zero.
    InvalidLimits,
    /// UTF-8 input exceeds the configured byte limit.
    ByteLimitExceeded {
        /// Observed byte count.
        actual: u64,
        /// Configured maximum.
        maximum: usize,
    },
    /// Filesystem or UTF-8 reading failed.
    Io(std::io::Error),
    /// CIF tokenization or document structure is invalid.
    Syntax {
        /// Stable explanation.
        message: String,
        /// One-based source line when known.
        line: Option<usize>,
    },
    /// A parser or import resource limit was exceeded.
    Limit {
        /// Stable explanation including the limit name.
        message: String,
    },
    /// A syntactically valid CIF cannot form the requested structure.
    Import {
        /// Stable explanation.
        message: String,
    },
    /// A deliberately unsupported CIF feature was encountered in strict mode.
    Unsupported {
        /// Unsupported feature family.
        feature: String,
        /// Human-readable explanation.
        message: String,
    },
    /// Native space-group lookup failed.
    SpaceGroup(SpaceGroupLookupError),
    /// Unit-cell validation failed.
    Cell(CellError),
    /// Exact operation validation failed.
    Symmetry(SymmetryError),
}

impl Display for CifIoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("all CIF read limits must be positive"),
            Self::ByteLimitExceeded { actual, maximum } => {
                write!(
                    formatter,
                    "CIF input exceeds max_bytes: {actual} > {maximum}"
                )
            }
            Self::Io(error) => Display::fmt(error, formatter),
            Self::Syntax {
                message,
                line: Some(line),
            } => write!(formatter, "invalid CIF syntax at line {line}: {message}"),
            Self::Syntax {
                message,
                line: None,
            } => write!(formatter, "invalid CIF syntax: {message}"),
            Self::Limit { message }
            | Self::Import { message }
            | Self::Unsupported { message, .. } => formatter.write_str(message),
            Self::SpaceGroup(error) => Display::fmt(error, formatter),
            Self::Cell(error) => Display::fmt(error, formatter),
            Self::Symmetry(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for CifIoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::SpaceGroup(error) => Some(error),
            Self::Cell(error) => Some(error),
            Self::Symmetry(error) => Some(error),
            _ => None,
        }
    }
}

fn import_error(message: impl Into<String>) -> CifIoError {
    CifIoError::Import {
        message: message.into(),
    }
}
