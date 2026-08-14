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
//! selected GSAS formats.
//!
//! # CIF structures
//!
//! [`read_cif_file`] and [`parse_cif_text`] implement a bounded native CIF 1.1
//! path. Import produces a parser-independent [`CifStructure`] containing a
//! validated cell, exact symmetry operations, independent atom sites,
//! uncertainties, metadata, diagnostics, and source provenance.
//!
//! ```
//! use phasesmith_io::{CifReadLimits, parse_cif_text};
//!
//! let cif = r#"
//! data_demo
//! _cell_length_a 5.431(1)
//! _cell_length_b 5.431(1)
//! _cell_length_c 5.431(1)
//! _cell_angle_alpha 90
//! _cell_angle_beta 90
//! _cell_angle_gamma 90
//! _space_group_IT_number 227
//! loop_
//! _atom_site_label
//! _atom_site_type_symbol
//! _atom_site_fract_x
//! _atom_site_fract_y
//! _atom_site_fract_z
//! Si1 Si 0 0 0
//! "#;
//! let result = parse_cif_text(cif, None, true, CifReadLimits::default())?;
//! assert_eq!(result.selected_block, "demo");
//! assert_eq!(result.structure.sites[0].element_symbol, "Si");
//! assert_eq!(result.structure.cell_standard_uncertainties[0], Some(0.001));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The block argument is the name after `data_`. Strict mode requires an
//! explicit selection for a multi-block document and rejects conflicting
//! scientific definitions. Permissive mode may make a documented recovery,
//! such as selecting the first block or renaming a duplicate site ID, but every
//! such decision is returned as a stable [`CifDiagnostic`]. Applications
//! should display and persist these diagnostics rather than treating a
//! successful parse as warning-free.
//!
//! Symmetry precedence is explicit operations, Hall symbol,
//! Hermann--Mauguin symbol, then International Tables number. If no definition
//! exists, P1 is returned with a warning. Atom sites require complete
//! fractional or Cartesian coordinates. Occupancy defaults to one; CIF B
//! displacement values are converted to U by `U = B/(8*pi^2)`. Magnetic,
//! modulated/superspace, and macromolecular feature families are not
//! interpreted.
//!
//! The facade's
//! [CIF input guide](https://docs.rs/phasesmith/latest/phasesmith/guide/cif_inputs/)
//! documents supported content, strict/permissive behavior, stable result
//! fields, and the subsequent conversion into refinement state.
//!
//! # Space groups
//!
//! Space-group lookup functions expose database provenance and do not require
//! a CIF parser. Use [`space_group_by_number`] or [`space_group_by_symbol`] for
//! a conventional setting, and [`space_group_from_hall_symbol`] when a general
//! Hall expression is already available.
//!
//! Keep default limits for ordinary trusted files; tighten them when accepting
//! untrusted uploads. Limit errors are distinct from syntax and domain errors.

mod cif;
mod powder;
mod space_groups;
mod tof_instrument;

pub use cif::{
    CifAnisotropicDisplacement, CifAtomSite, CifDiagnostic, CifDiagnosticSeverity, CifIoError,
    CifReadLimits, CifReadResult, CifStructure, CifStructureSource, DisplacementConvention,
    NATIVE_CIF_BACKEND, NATIVE_CIF_BACKEND_VERSION, parse_cif_text, read_cif_file,
};

pub use powder::{
    PowderData, PowderFormat, PowderIoError, PowderReadLimits, TofPowderData, TofPowderFormat,
    parse_powder_text, parse_tof_powder_text, parse_tof_powder_text_as, read_powder_file,
    read_tof_powder_file, read_tof_powder_file_as,
};
pub use space_groups::{
    SPACE_GROUP_DATABASE_PROVENANCE, SpaceGroupDatabaseProvenance, SpaceGroupInfo,
    SpaceGroupLookupError, space_group_by_hall_symbol, space_group_by_number,
    space_group_by_symbol, space_group_from_hall_symbol,
};
pub use tof_instrument::{
    GsasTofInstrumentData, GsasTofInstrumentIoError, GsasTofInstrumentReadLimits,
    parse_gsas_tof_instrument_text, read_gsas_tof_instrument_file,
};
