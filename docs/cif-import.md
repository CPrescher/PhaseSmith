# Native CIF import and CIF-backed Le Bail

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

An International Tables number identifies the space-group type. When supplied
alone it selects the standard setting. With higher-priority symmetry, the reader
also checks the other settings tabulated for that number and retains the exact
deposited operations. The native tables also recognize centered triclinic cells, F-centered
settings of groups 9 and 139, and selected explicitly transformed settings of
groups 4 and 12. The reader matches their deposited basis expressions to Hall
generators in that same cell. For example, transforming group 4 to the
basis `(c,2*a+c,b)` places the screw axis along the deposited c axis and
adds the centering translation `(1/2,1/2,0)`. Its Hall symbol is `C 2c`.
The coordinates stay in the deposited basis.
Operation coordinates accept uppercase and lowercase X, Y and Z. Conventional
Hermann–Mauguin names accept explicit origin choices such as `I 41/a:2` and
the five historical double-glide names, including `Aba2` for `Aea2`, in
their tabulated axis permutations. A short H–M symbol such as `P b n m`
selects its own basis. When the symbol permits several origins or settings,
matching deposited operations or a Hall symbol disambiguate the choice.
See the [IUCr nomenclature report, section 3](https://www.iucr.org/who-we-are/commissions/commission-on-crystallographic-nomenclature/published-reports/symsym).

Historical cubic H–M names such as `Fd3m` and `Fm3m` are accepted.
An H–M name with no specified origin can agree with an explicit operation
list in the same axes at a different origin. The reader checks for one origin
shift on a grid of 24ths that gives exact equality of every rotation and
translation, using `t' = t + (I - R)s`. It retains the deposited operations.
Explicit origin choices, basis expressions and Hall declarations still require
the specified origin. Decimal translations in operation lists are exact
rationals, so `0.500` equals `1/2`. [IUCr H–M definition](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Ispace_group_name_H-M_alt.html).

Two incomplete archive spellings, `P42 mnm` and `P 42/n m c S`, require
independent Hall or explicit operations that exactly match the corresponding
complete symbol. A diagnostic records the interpretation. The latter spelling
also appears in the [VESTA manual, ICSD file formats](https://www.jp-minerals.org/vesta/archives/VESTA_Manual.pdf).

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
`phasesmith.refinement.readiness.review_rietveld_input(request)` aggregates these retained
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
  stable site IDs, and disorder groups are separate fields. An explicit type
  takes precedence over the site label and retains its source spelling.
  Element and scattering identities recognize chemical symbols regardless
  of case. If the type is missing, the longest valid element prefix of the
  label is used (`FE1` gives Fe and `CL1` gives Cl). D and T denote
  hydrogen isotopes when they are the element prefix.
- Repeated atom labels retain their source spelling and receive distinct IDs such
  as `C1#2`. Literal labels elsewhere in the site table are reserved. Strict
  import checks for indistinguishable symmetry-equivalent positions and for
  shared positions exceeding unit occupancy (tolerance `1e-10`).
- For geometry references to a pair of like-named sites, strict import uses the
  deposited coordinates in their listed cell image. One assignment must
  reproduce the bond length within one reported standard uncertainty, and every
  other assignment must differ by more than three. An affected angle needs a
  unique pair of verified bond legs and must agree within one reported standard
  uncertainty. Unknown symmetry codes, unsupported images and unresolved label
  references still raise an error. A diagnostic counts the resolved geometry rows.
- Anisotropic CIF U tensors retain all six components in the order
  `(11,22,33,23,13,12)` and their uncertainties. For two sites with a repeated
  label and explicit `Uani` or `Uiso` declarations, strict import can attach
  tensor rows to the `Uani` sites if each has exactly one owner. It computes
  $U_{\rm eq}=\tfrac13\sum_{ij}U_{ij}a_i^*a_j^*G_{ij}$, where $G$ is the
  direct-cell metric and $a_i^*$ are reciprocal-axis lengths without $2\pi$.
  Each calculated value must fall strictly within half a unit of the site's
  last reported decimal place. The assignment is independent of row order.
  [IUCr definition of equivalent isotropic displacement](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iatom_site.U_iso_or_equiv.html)
- An unmatched tensor label such as `C9A` can be redundant when `C9` is a
  fully occupied anisotropic site in a multigroup disorder assembly. Every
  other field in the two tensor rows must be identical, including uncertainties,
  and the extra label must have no other atom reference. A diagnostic names both
  rows. The imported structure contains the deposited coordinate sites.
- Permissive import retains repeated sites with a warning and keeps the first
  row for a repeated anisotropic label. The initial isotropic structure-factor
  conversion raises `NotImplementedError` for anisotropic tensors.

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

## Deposited values and empty tables

A single site row containing only bare missing markers (`?` or `.`) imports
as a cell-only record with an `empty_atom_site_table` diagnostic. This can
supply a Le Bail phase. A wholly empty anisotropic table likewise supplies no
displacement tensors. A populated row still requires its labels and coordinates.
A literal `#` inside an unquoted value belongs to that value.

An empty uncertainty suffix such as `.02554()` preserves the nominal value
and records a warning. The uncertainty remains unspecified. An unclosed suffix
is rejected because digits may have been truncated. For anisotropic labels,
the reader can match `C(1)` to an unclaimed, unique `C1` site and records
both spellings. A tensor labeled `Ti1/Mg1` can serve both named sites, or
a tensor labeled `M1` can serve sites such as `M1(Na)` and `M1(Ca)`
when each suffix matches its deposited species. The sites must declare
anisotropic displacement, share identical deposited fractional coordinates,
and have no competing tensor. Every link is recorded. Fractional average oxidation states such as `Mn3.5+` retain
their exact type string. The integer ionic charge is unset and X-ray lookup
requires that exact scattering state.

If the initial cell metric fails validation, the reader checks whether a
rotation makes two lengths or two angles equivalent. A finer deposited value
can supply their common value if it rounds to the coarser printed value, lies
within its reported uncertainty, and agrees with every other finer candidate.
This applies to fixed-decimal entries with uncertainty on the coarser
measurement and none on the finer one. The diagnostic records both original
literals, the reported uncertainty is retained, and the resulting cell must
pass the original metric check.

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
