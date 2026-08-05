# Optional CIF import and fixed-cell Le Bail

Status: implemented and internally validated in implementation unit 13.

CIF is an input format, not calculation state. The optional adapter returns
immutable `CrystalStructure`, `UnitCell`, `SpaceGroup`, and `AtomSite` models.
No Gemmi object is retained, serialized, passed to Rust, or required to reload
an imported structure.

## Installation and entry points

Gemmi is optional:

```bash
pip install 'rietveld-engine[cif]'
```

The supported adapter range is `gemmi>=0.7.5,<0.8`. Importing `rietveld` or
using explicit cells/reflections does not import Gemmi.

```python
from rietveld.io.cif import read_cif

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

`CifBackend` is a small explicit protocol. A backend receives text plus limits
and returns `CifReadResult`; it may not leak parser objects. Callers can inject a
backend directly for controlled environments. Default-backend loading is lazy
and raises an actionable optional-dependency error when Gemmi is unavailable.

The current Gemmi adapter is used for CIF tokenization, loops, quoted strings,
operation-string parsing, and space-group name/setting resolution. Native exact
`SpaceGroup` construction revalidates every operation and owns closure,
systematic absences, multiplicity, and reflection generation.

## Symmetry precedence

Definitions are resolved in this order:

1. explicit symmetry-operation loop;
2. Hall symbol;
3. Hermann–Mauguin symbol, including extended/non-standard settings;
4. International Tables number;
5. P1 with a visible warning when no definition exists.

Lower-priority definitions are also resolved. A disagreement is an error in
strict mode and a structured warning in permissive mode. The chosen source and
original identifiers are retained as plain metadata.

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
without importing Gemmi. This is the persistence boundary future project
bundles will embed.

## CIF-to-Le Bail

Le Bail needs cell and symmetry but does not need atom sites or scattering
tables. `LeBailPhase` subclasses the existing generic `Phase`, so all normal
calculation and refinement functions accept it unchanged.

```python
from rietveld.refinement.lebail import LeBailPhase

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

## Validation

Authored MIT-project test CIFs cover quoted strings, loops, multiple blocks,
standard uncertainties, missing/unknown values, legacy tags, explicit/Hall/HM/
number symmetry, non-standard settings through exact operations, Cartesian
coordinates, disorder, duplicate definitions, anisotropic U/B input, resource
limits, parser-free records, lazy imports, and cell-only CIF-to-Le Bail.

The adapter follows the official IUCr core dictionary for data semantics and
the public Gemmi CIF/small-structure documentation for parsing behavior. Gemmi
does not supply numerical diffraction results.

- IUCr core CIF dictionary: <https://www.iucr.org/resources/cif/dictionaries/cif_core>
- Gemmi CIF documentation: <https://gemmi.readthedocs.io/en/stable/cif.html>
- Gemmi small-structure documentation: <https://gemmi.readthedocs.io/en/stable/chemistry.html>
