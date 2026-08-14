# CIF input: from text to native structure

`PhaseSmith` treats CIF as an external input format, not as live refinement
state. [`crate::io::parse_cif_text`] and [`crate::io::read_cif_file`] return an
owned, parser-independent [`crate::io::CifStructure`]. No parser object, CIF
tag lookup, or filesystem handle survives the import boundary.

## A complete minimal input

A structural calculation needs six cell parameters, a symmetry definition,
and atom sites with labels, chemical identities, and complete fractional or
Cartesian coordinates. Tag spelling is case-insensitive. Current dot-style
CIF names and the common underscore-style aliases shown below are accepted.

```
use phasesmith::io::{CifReadLimits, parse_cif_text};

let cif = r#"
data_silicon
_chemical_name_common 'Silicon'
_cell_length_a 5.431(1)
_cell_length_b 5.431(1)
_cell_length_c 5.431(1)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0 0 0 1 0.005
"#;

let imported = parse_cif_text(
    cif,
    None, // select by block name here when the document has several blocks
    true, // strict scientific conflict handling
    CifReadLimits::default(),
)?;

assert_eq!(imported.selected_block, "silicon");
assert_eq!(imported.structure.sites[0].element_symbol, "Si");
assert_eq!(imported.structure.cell_standard_uncertainties[0], Some(0.001));
assert!(imported.diagnostics.is_empty());
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a file, the same call is:

```no_run
use phasesmith::io::{CifReadLimits, read_cif_file};

let imported = read_cif_file(
    "phase.cif",
    Some("silicon"),
    true,
    CifReadLimits::default(),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The `block` argument is the name after `data_`, without the prefix. With one
block, `None` selects it. With multiple blocks, strict mode requires an
explicit name; permissive mode selects the first block and emits the stable
`multiple_blocks_first_selected` diagnostic.

## What is imported

The result separates content by responsibility:

| Result field | Meaning |
| --- | --- |
| `structure.cell` | Validated lengths in ångströms and angles in degrees |
| `structure.space_group` | Exact conventional symmetry operations used by reflection generation |
| `structure.sites` | Independent sites in source order, not symmetry-expanded atoms |
| `structure.cell_standard_uncertainties` | Optional uncertainties for `a,b,c,alpha,beta,gamma` |
| `structure.metadata` | Small retained strings such as formula, Z, formula mass, wavelength, and symmetry source |
| `structure.source` | Block, parser backend/version, and optional source path |
| `diagnostics` | Recoverable import decisions with stable code, tag, and zero-based loop row |
| `available_blocks` | All block names in source order, useful for a selection UI |

The imported phase name is the first available value among common chemical
name/formula tags and `_pd_phase_name`; otherwise the block name is used.
`structure_id` is a stable identifier derived from that block name. Applications
should preserve `site_id` as the durable parameter identity and use
`source_label` only when presenting the original CIF label.

## Cell and symmetry rules

All six cell values are required, including 90-degree angles. Standard
uncertainties such as `5.431(1)` are split from their values. The resulting
cell metric is validated immediately.

Symmetry is resolved in this order:

1. an explicit `_space_group_symop_operation_xyz` loop (legacy
   `_symmetry_equiv_pos_as_xyz` is also accepted);
2. a Hall symbol;
3. a Hermann--Mauguin symbol;
4. an International Tables number;
5. P1 with a visible `missing_space_group_assumed_p1` warning when no
   definition is supplied.

Every supplied definition is checked. If two valid definitions disagree,
strict mode rejects the file. Permissive mode keeps the higher-precedence
definition and reports `conflicting_space_group_definition`. A malformed
identifier is not guessed or silently converted to P1. Exact operations are
also used to validate the cell metric, so an incompatible cell/space-group
combination fails during import.

## Atom-site conventions

An atom loop requires a label plus all three fractional coordinates or all
three Cartesian coordinates. Fractional coordinates take precedence when both
sets exist; disagreement is an error in strict mode and a diagnostic in
permissive mode. Cartesian coordinates are converted through the direct cell
basis. Cartesian coordinate uncertainties are not transformed because the CIF
does not provide their covariance, and this decision is reported.

The type symbol is read independently from the label. If it is absent,
`PhaseSmith` attempts to infer it from the label. Isotopes and formal charges are
retained separately, so values such as `13C`, `Fe3+`, and `O2-` can later be
mapped to the correct built-in scattering key. Occupancy defaults to `1.0`
when absent, `.` (inapplicable), or `?` (unknown); the latter two remain visible
as diagnostics. Negative occupancy and displacement values are rejected.

Isotropic `_atom_site_U_iso_or_equiv` values are stored in square ångströms.
`B_iso` is converted using `U = B/(8*pi^2)`. Anisotropic loops must provide all
six components in CIF order `11,22,33,23,13,12`; anisotropic B values and their
uncertainties are converted to U. Simultaneous U and B definitions must agree.

## Strict and permissive modes

Use `strict = true` for reproducible scientific ingestion and curated project
files. It rejects conflicting cell/symmetry/coordinate/displacement
definitions, duplicate atom or anisotropic labels, multi-block ambiguity, and
unsupported magnetic, modulated/superspace, or macromolecular feature tags.

Use `strict = false` when building an import-review UI. It can select the first
block, rename duplicate stable site IDs (`C1`, `C1#2`), skip invalid site rows,
or keep a higher-precedence valid definition. These are not silent repairs:
each decision appears in [`crate::io::CifDiagnostic`]. Show those diagnostics
to the user and store them with provenance.

```
use phasesmith::io::{CifReadLimits, parse_cif_text};

let text = "data_a\n_cell_length_a 4\n_cell_length_b 4\n\
            _cell_length_c 4\n_cell_angle_alpha 90\n_cell_angle_beta 90\n\
            _cell_angle_gamma 90\n\
            data_b\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n\
            _cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n";
let imported = parse_cif_text(text, None, false, CifReadLimits::default())?;
assert_eq!(imported.selected_block, "a");
assert_eq!(imported.diagnostics[0].code, "multiple_blocks_first_selected");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Syntax and resource boundaries

The native bounded CIF 1.1 reader supports data blocks, scalar tags, loops,
bare and quoted values, comments, semicolon text fields, numeric exponents,
standard uncertainties, and the distinct `?`/`.` missing states. Save frames,
`stop_`, and `global_` are rejected. The input must be UTF-8.

[`crate::io::CifReadLimits::default`] allows 16 MiB, 100 blocks, 1,000,000 rows
in any loop, and 100,000 atom/anisotropic rows. `read_cif_file` checks file size
before allocating its contents. Tighten these values for upload services;
never disable the limits by passing unchecked user-derived maxima.

[`crate::io::CifIoError`] distinguishes filesystem/UTF-8, syntax, resource
limit, import-domain, unsupported-feature, cell, symmetry, and lookup failures.
An error means no scientifically valid structure was returned. Diagnostics,
by contrast, belong to a successful result and require an explicit caller
policy.

## Moving from CIF to refinement state

`CifStructure` intentionally does not choose radiation, angular range,
reflection multiplicities, intensity corrections, profile parameters, sample
physics, or which parameters to refine. Those are experiment/model decisions.
The normal conversion is:

1. generate a bounded reflection domain from `structure.cell` and
   `structure.space_group`;
2. map independent site coordinates, occupancy, U values, and scattering
   species into [`crate::engine::StructuralPhaseDefinition`];
3. choose X-ray or neutron scattering and the matching intensity correction;
4. wrap the definition in [`crate::workflows::RietveldPhase`] with stable
   phase/site IDs;
5. combine it with the observed pattern and instrument in
   [`crate::workflows::RietveldInput`].

[`crate::guide::real_data_rietveld`] shows these steps using the measured `PbSO4`
dataset. CIF import never selects refinement parameters and never imports a
GSAS-II project model.
