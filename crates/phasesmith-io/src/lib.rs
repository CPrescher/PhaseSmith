//! Bounded native input adapters for application hosts.
//!
//! This crate parses external text into validated `phasesmith-model` and
//! crystallographic records. It owns syntax, provenance, diagnostics, format
//! detection, and pre-allocation limits; it does not own refinement or GUI
//! state. Applications normally use it through
//! [`phasesmith::io`](https://docs.rs/phasesmith/latest/phasesmith/).
//!
//! # Powder data
//!
//! ```
//! use phasesmith_io::{PowderFormat, PowderReadLimits, parse_powder_text};
//!
//! let data = parse_powder_text(
//!     "20.0 100.0 2.0\n20.1 120.0 2.5\n",
//!     PowderFormat::Columns,
//!     1,
//!     PowderReadLimits::default(),
//! )?;
//! assert_eq!(data.pattern.sample_count(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`read_powder_file`] and [`parse_powder_text`] support plain columns and
//! selected GSAS formats. [`read_cif_file`] and [`parse_cif_text`] implement the
//! native CIF path with structured diagnostics. Space-group lookup functions
//! expose database provenance and do not require a CIF parser.
//!
//! Keep default limits for ordinary trusted files; tighten them when accepting
//! untrusted uploads. Limit errors are distinct from syntax and domain errors.

mod cif;
mod powder;
mod space_groups;

pub use cif::{
    CifAnisotropicDisplacement, CifAtomSite, CifDiagnostic, CifDiagnosticSeverity, CifIoError,
    CifReadLimits, CifReadResult, CifStructure, CifStructureSource, DisplacementConvention,
    NATIVE_CIF_BACKEND, NATIVE_CIF_BACKEND_VERSION, parse_cif_text, read_cif_file,
};

pub use powder::{
    PowderData, PowderFormat, PowderIoError, PowderReadLimits, TofPowderData, parse_powder_text,
    parse_tof_powder_text, read_powder_file, read_tof_powder_file,
};
pub use space_groups::{
    SPACE_GROUP_DATABASE_PROVENANCE, SpaceGroupDatabaseProvenance, SpaceGroupInfo,
    SpaceGroupLookupError, space_group_by_hall_symbol, space_group_by_number,
    space_group_by_symbol,
};
