# Native CIF import and CIF-backed Le Bail

Status: implemented and internally validated in implementation unit 13.

CIF is an input format, not calculation state. The default pure-Rust adapter
returns immutable `CrystalStructure`, `UnitCell`, `SpaceGroup`, and `AtomSite` models.
It does not require CPython, Gemmi, or another native parser library. The same
`phasesmith-io` implementation can be called by Python and a Rust-only desktop
host.

## Installation and entry points

CIF import is part of the base package. The optional `phasesmith[cif]` extra
installs `gemmi>=0.7.5,<0.8` only for callers that explicitly use the retained
`GemmiCifBackend` or run differential validation.

```python
from phasesmith.io.cif import read_cif

result = read_cif("sample.cif", block="phase_a", strict=True)
structure = result.structure
print(structure.cell, structure.space_group.crystal_system)
for diagnostic in result.diagnostics:
    print(diagnostic.severity, diagnostic.code, diagnostic.message)
```

`read_cif()` accepts a `Path`, a local-path string, or CIF text. A multi-block
document requires an explicit block in strict mode. Permissive mode selects the
first block and returns a visible warning.

## Backend boundary

`CifBackend` is a small explicit Python protocol. A backend receives text plus
limits and returns `CifReadResult`; it may not leak parser objects. The default
backend calls the bounded native CIF 1.1 tokenizer/import policy. Callers can
still inject `GemmiCifBackend` for controlled comparison or unusual scripting
environments.

The native adapter owns tokenization, blocks, loops, quoted and semicolon text,
numeric uncertainties, operation-string parsing, and import policy. Conventional
space-group lookup uses the pinned pure-Rust Moyo 0.15.0 Hall database with all
530 settings. CIF Hall tags additionally use Moyo's general non-magnetic Hall
parser, so valid redundant translation spellings and explicit origin shifts do
not need to be present verbatim in that canonical table. Generated operations
are converted to exact denominator-12 translations. `SpaceGroup` construction
independently revalidates exact integer/rational operations and owns closure,
systematic absences, multiplicity, and reflection generation.

## Symmetry precedence

Definitions are resolved in this order:

1. explicit symmetry-operation loop;
2. Hall symbol;
3. Hermann–Mauguin symbol, including extended/non-standard settings;
4. International Tables number;
5. P1 with a visible warning when no definition exists.

Lower-priority definitions are also resolved. A disagreement is an error in
strict mode and a structured warning in permissive mode. An invalid supplied
definition is always an error in strict mode. In permissive mode it is ignored
with an `invalid_space_group_definition_ignored` warning only when another
supplied definition resolves valid symmetry; an invalid identifier never causes
an implicit P1 fallback. The chosen source and original identifiers are retained
as plain metadata. In particular, permissive mode may use valid explicit
operations while warning about a malformed secondary Hermann--Mauguin tag, but
it does not repair or guess the intended symbol.

Scripts that do not start from a CIF can use `space_group_by_number(1..230)` or
`space_group_by_symbol(...)`. These native lookup functions
return `SpaceGroupInfo` with the International number, Hermann--Mauguin and
Hall symbols, setting qualifier, and an engine-owned `SpaceGroup` made from
exact integer/rational operations.
Callers that already have a general Hall expression can use
`space_group_from_hall_symbol(...)` to obtain the exact `SpaceGroup` directly.

For a constructed structural Rietveld request,
`phasesmith.refinement.review_rietveld_input(request)` aggregates these retained
structure diagnostics with source/symmetry provenance and the active radiation,
scattering, intensity-correction, and specimen-geometry choices. It reports
contradictory optics and risky joint selections before refinement without
changing the request. `RietveldProject.review_readiness()` is the stateful
facade for the same operation.

## Numeric and atom-site conventions

- Cell lengths are ångströms and angles are degrees.
- Standard uncertainties such as `4.200(5)` are separated from the value and
  retained. `.` and `?` produce distinct `missing_cif_value` and
  `unknown_cif_value` diagnostics.
- Fractional coordinates take precedence. Complete Cartesian coordinates are
  converted through the native cell basis; simultaneous definitions must
  agree periodically within `1e-8`.
- Occupancy defaults to one when the dictionary value is absent or unknown.
- `B_iso` and anisotropic `B_ij` are converted by `U = B / (8 pi^2)`.
  Simultaneous U/B definitions must agree within local tolerances.
- Type symbols, element symbols, isotope mass numbers, charges, source labels,
  stable site IDs, and disorder groups are separate fields.
- Duplicate site labels are errors in strict mode. Permissive mode retains the
  source label and creates stable IDs such as `C1#2` with a warning.
- Anisotropic CIF U tensors are preserved in component order
  `(11,22,33,23,13,12)`, including component standard uncertainties. Duplicate
  anisotropic labels are errors in strict mode and retain the first row with a
  warning in permissive mode. The initial isotropic structure-factor conversion
  raises `NotImplementedError` rather than silently replacing the tensor.

Both current dot-style core dictionary names and common underscore-style legacy
aliases are accepted. Conflicting duplicate cell, coordinate, displacement, or
symmetry definitions are never silently merged.

## Resource and feature limits

`CifReadLimits` defaults to 16 MiB, 100 blocks, one million rows per loop, and
100,000 atom sites. File size is checked before parsing; block, loop, and atom
limits are checked immediately afterward. Callers can set smaller limits.

Magnetic, modulated/superspace, and macromolecular feature tags are unsupported
in this small-structure slice. Strict mode raises; permissive mode records a
warning without pretending the unsupported physics was interpreted.

## Parser-independent persistence

`structure_to_record()` returns a versioned JSON-compatible record containing
cell values, exact rational operations, sites, displacement records,
diagnostics, metadata, and provenance. `structure_from_record()` restores it
without importing a CIF parser. This is the persistence boundary future project
bundles will embed.

## CIF-to-Le Bail

Le Bail needs cell and symmetry but does not need atom sites or scattering
tables. `LeBailPhase` subclasses the existing generic `Phase`, so all normal
calculation and refinement functions accept it unchanged.

```python
from phasesmith.refinement.lebail import LeBailPhase

phase = LeBailPhase.from_cif(
    "sample.cif",
    phase_id="alpha",
    wavelength_angstrom=1.5406,
    two_theta_min_deg=10.0,
    two_theta_max_deg=120.0,
)
```

`from_structure()` provides the same path after inspection or modification.
Both call the prepared Rust reflection generator, remove exact systematic
absences, calculate monochromatic positions, and initialize one independently
extractable intensity per stable reflection family. No Python loop performs
per-reflection diffraction physics.

Pass `refine_lattice=True` to `LeBailPhase.from_cif()` to construct a guarded
domain, or use `LeBailInput.from_cif()` for a directly runnable request with
lattice refinement enabled by default. The latter derives the visible interval
from the observed pattern. Equations, finite bounds, topology regeneration,
and stable intensity transfer are documented in
[lattice-refinement.md](lattice-refinement.md).

## Validation

Authored MIT-project test CIFs cover quoted strings, loops, multiple blocks,
standard uncertainties, missing/unknown values, legacy tags, explicit/Hall/HM/
number symmetry, non-standard settings through exact operations, Cartesian
coordinates, disorder, duplicate definitions, anisotropic U/B input, resource
limits, parser-free records, lazy imports, and cell-only CIF-to-Le Bail.

Rust-only integration tests also import pinned Al2O3, CaF2, ZnO, and PbSO4 CIFs.
The Python validation suite compares their cells, exact operation sets, site
identities, fractional coordinates, and occupancies against the optional Gemmi
backend. A subprocess test blocks every Gemmi import and still performs the
default CIF import.

The adapter follows the official IUCr core dictionary for data semantics.
Gemmi remains an independent parsing oracle and does not supply numerical
diffraction results.

- IUCr core CIF dictionary: <https://www.iucr.org/resources/cif/dictionaries/cif_core>
- Gemmi CIF documentation: <https://gemmi.readthedocs.io/en/stable/cif.html>
- Gemmi small-structure documentation: <https://gemmi.readthedocs.io/en/stable/chemistry.html>
