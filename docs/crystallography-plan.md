# Crystallography and structure-factor implementation plan

Status: units 11 and 12 implemented and independently validated; live pinned
P1 and symmetry/reflection oracle fixtures pending; units 13 through 18
pending.

This plan adds CIF-driven Le Bail and structure-factor-driven Rietveld
calculation without moving file parsing, Python objects, or per-reflection
orchestration into the numerical hot path. The monochromatic X-ray path is the
first target. Non-magnetic neutron nuclear scattering follows through the same
typed structure-factor interface.

## Required user outcomes

The first completed crystallography program must support two short, fully
scripted workflows:

```python
structure = rietveld.io.cif.read_cif("silicon.cif")
phase = rietveld.phase.LeBailPhase.from_structure(
    phase_id="si",
    structure=structure,
    experiment=experiment,
    limits=rietveld.crystallography.TwoThetaLimits(10.0, 120.0),
)
result = rietveld.refinement.lebail.refine(
    rietveld.refinement.lebail.LeBailInput(pattern, experiment.instrument, (phase,))
)
```

and, after the structure-factor and refinement units are complete:

```python
structure = rietveld.io.cif.read_cif("silicon.cif")
phase = rietveld.phase.RietveldPhase(
    phase_id="si",
    structure=structure,
    scale=1.0,
    scattering=rietveld.scattering.XrayNonResonant(),
)
result = rietveld.refinement.rietveld.refine(
    rietveld.refinement.rietveld.RietveldInput(pattern, experiment, (phase,))
)
```

The final spelling may change during API review, but these ownership boundaries
must not: a CIF becomes an inspectable typed structure; Le Bail generates
allowed reflections from cell and symmetry; Rietveld calculates their
intensities from the structure; both reuse the existing Rust profile engine.

## Architectural boundary

```text
file or text
    |
    v
rietveld.io.cif                 optional parsing adapter
    |                           no diffraction calculation
    v
CrystalStructure               typed arrays and explicit symmetry operations
    |
    v
rietveld-crystallography       Rust
    |-- cell and reciprocal-space mathematics
    |-- symmetry application and special-position expansion
    |-- reflection enumeration, absences, orbits, and multiplicities
    |-- scattering factors and structure factors
    |-- analytical JVP/VJP operations
    v
rietveld-engine                Rust composition
    |-- integrated-intensity corrections
    |-- structure-factor/profile derivative chaining
    |-- one-call structural pattern calculation
    v
rietveld-core                  existing Rust profile kernels
    |
    v
NumPy values, diagnostics, and labeled derivatives
```

The proposed Cargo workspace is:

- `rietveld-core`: existing dependency-light profile primitives and fused
  support-limited accumulation;
- `rietveld-crystallography`: cells, symmetry, reflections, scattering, and
  structure-factor kernels, with no Python or file-I/O dependency;
- `rietveld-engine`: native composition of crystallographic intensities with
  profile accumulation, depending on the preceding two crates;
- `rietveld-py`: thin PyO3 validation and NumPy bindings.

`rietveld-core` must not depend on crystallography. This preserves its utility
as a small line-profile library and avoids a dependency cycle. `rietveld-py`
must not implement numerical orchestration that belongs in `rietveld-engine`.

The initial CIF adapter will use Gemmi behind an optional, lazily imported
Python boundary because it has mature CIF handling and an extensive
space-group-setting table. Gemmi objects never cross the adapter. The adapter
copies cell values, sites, and exact rotation/translation operations into the
public models. A parser backend can later be replaced without changing the
Rust calculation API. Gemmi remains a separately licensed dependency and is
not vendored or used as a structure-factor oracle.

## Canonical models and conventions

### Public domain models

`rietveld.crystallography` will own inspectable frozen models mirroring the
native array contracts:

- `UnitCell(a, b, c, alpha, beta, gamma)`;
- `SymmetryOperation(rotation, translation_numerator,
  translation_denominator)`;
- `SpaceGroupOperations(setting_id, hall_symbol, operations)`;
- `AtomSiteBatch(site_ids, species, fractional_xyz, occupancy, u_iso, ...)`;
- `CrystalStructure(structure_id, cell, symmetry, sites, metadata)`;
- `ReflectionSet(reflection_ids, hkl, d_spacing, multiplicity, orbit_offsets,
  orbit_hkl)`;
- `StructureFactorResult(f_real, f_imag, intensity, diagnostics)`.

`rietveld.phase` will distinguish supplied-intensity and calculated-intensity
workflows:

- the existing `Phase`/`ReflectionBatch` remains the direct profile and Le Bail
  compatibility model;
- `LeBailPhase` owns a generated `ReflectionSet` plus independently varied
  integrated intensities;
- `RietveldPhase` owns a `CrystalStructure`, a scattering model, scale, and
  typed intensity corrections.

This avoids a `Phase` with many mutually exclusive optional fields. Stable
phase, site, and reflection IDs connect persistence, diagnostics, constraints,
and derivative labels.

### Numerical conventions

- Public lengths are ångströms; public angles are degrees. Native trigonometry
  uses radians with conversions at the boundary.
- Fractional coordinates are the structural parameters. Coordinates differing
  by a lattice translation are physically equivalent; refinement state may be
  unwrapped while serialized/display coordinates are normalized explicitly.
- The reciprocal vector is `g = B h` in cycles per ångström,
  `d = 1 / |g|`, and `s = |g| / 2 = sin(theta) / wavelength`.
- Structure-factor phases use `exp(2 pi i h dot x)`.
- Isotropic displacement is stored as `U_iso` in square ångströms. The first
  vertical slice does not silently reinterpret `B_iso`; import converts it by
  the documented `B = 8 pi^2 U` convention.
- Symmetry rotations and translations are exact integers/rationals. Floating
  tolerances are permitted only when deduplicating equivalent site positions,
  and the tolerance is a named import/calculation option.
- Initial X-ray calculations are non-resonant and obey Friedel's law. Friedel
  pairs are combined when calculating powder multiplicity. Anomalous
  dispersion later changes this grouping explicitly.
- Peak intensity means integrated intensity before application of the
  unit-area profile, consistent with the current accumulator.
- Reflection ordering is deterministic: increasing position for a selected
  experiment, followed by canonical Miller-index order for exact ties.
  Reflection identity is based on phase and Miller family, never floating-point
  position or array index.

### Equations and derivative contract

For an asymmetric-unit site `j`, its unique symmetry-generated positions
`x_jq`, and a non-magnetic scattering amplitude `f_j(s)`, the initial kernel
uses

```text
F_h = sum_j occupancy_j f_j(s) T_j(h) sum_q exp(2 pi i h dot x_jq)
I_h = phase_scale multiplicity_h C_h |F_h|^2
```

where `T_j` is the displacement factor and `C_h` is a typed product of
Lorentz, polarization, absorption, preferred-orientation, and other explicitly
selected corrections. Correction ownership is unique: the existing sample
provider may supply preferred orientation, so the structure-factor layer must
not apply it a second time.

Every structural derivative is built from

```text
d|F|^2/dp = 2 Re(conjugate(F) dF/dp).
```

Coordinates, occupancies, isotropic displacement, phase scale, lattice
parameters, and every enabled correction need analytical derivatives before
they become refinable. Values and requested forward derivatives are evaluated
in the same atom/reflection pass. Reverse products may use a second native pass
over cached `F` values after the profile residual weights are known.

A dense `(sample, structural_parameter)` Jacobian is not the production
representation. The native engine provides:

- `J v` from structural tangents through `dF`, `dI`, peak position, and profile;
- `J^T w` from profile support blocks back through reflection intensities and
  structure factors;
- an explicitly requested, size-limited dense diagnostic matrix for tests and
  small examples.

This prevents memory from scaling as pattern samples times all atomic
parameters.

## GSAS-II behavioral study for parameter design

Before freezing each structural parameter family, run a controlled study with
the pinned GSAS-II validation environment. The purpose is to understand
observable conventions and refinement behavior, not to inherit its internal
data structures or implementation.

For cell, coordinate, occupancy, displacement, scale, scattering, and
correction parameters, the study must record:

1. the public/scripting name, displayed unit, starting value, refine flag,
   bound or constraint behavior, and any phase/histogram ownership visible to a
   user;
2. the corresponding reflection-list fields before and after calculation,
   including `hkl`, `d`, position, multiplicity, structure-factor magnitude,
   correction terms, and integrated intensity when exposed;
3. `Ycalc`, background, and phase contributions for minimal isolated test
   structures;
4. the numerical response to one-at-a-time positive and negative parameter
   perturbations, including centered finite-difference signs and scale factors;
5. special-position, space-group, occupancy, and displacement constraints as
   observed through public refinement results;
6. the exact pinned revision, input checksum, probe schema, and output hashes.

Use the public scripting API first. If a required intermediate is not public,
add one narrowly named, revision-gated probe returning a copied scalar, array,
or plain record. A probe may observe a value; it may not expose a GSAS-II
dictionary as our domain model. No GSAS-II object crosses into calculation code.

Each outcome becomes a reviewed mapping table:

| Concept | GSAS-II observation | Engine convention | Conversion | Source |
| --- | --- | --- | --- | --- |
| Isotropic displacement | Pinned study | `U_iso` in Å² | Explicit/tested | Published definition |

The Rietveld Engine convention is selected for physical clarity and
scriptability; it need not copy a GSAS-II parameter name. Published equations
or first-principles derivations remain the implementation source. The oracle
study supplies black-box numerical cases that verify conversions, derivative
signs, constraints, and final observables.

Required oracle cases include a one-atom P1 cell, a centring extinction, a
special-position structure, a multi-species structure, and a two-phase overlap.
Every case must run without importing `rietveld`; the resulting plain fixture
is then consumed by our tests. Normal builds, imports, calculations, and
refinements remain GSAS-II-free.

## Implementation unit 11: native foundation and P1 vertical slice

Status: implemented and internally reviewed on 2026-08-05. The external pinned
GSAS-II perturbation fixture remains pending because its checkout is not
available in the current environment; no oracle equivalence is claimed yet.

### Work

1. Add `rietveld-crystallography` and `rietveld-engine` workspace crates with
   the dependency direction described above.
2. Run the pinned GSAS-II P1 parameter study and review the physical-unit
   mapping before public structural parameter names are finalized.
3. Implement a general triclinic `UnitCell`, direct and reciprocal matrices,
   volume, `d(hkl)`, and derivatives with respect to six unconstrained cell
   parameters.
4. Add native structure-of-arrays validation for atom IDs, species IDs,
   fractional coordinates, occupancy, and `U_iso`.
5. Implement a P1 structure-factor kernel for caller-supplied real or complex
   scattering amplitudes.
6. Calculate values and analytical derivatives for coordinates, occupancy, and
   `U_iso`; expose bounded dense derivatives plus JVP/VJP primitives.
7. Add independent, readable NumPy reference equations and array-oriented PyO3
   bindings.
8. Benchmark values, values plus JVP, and VJP for a small inorganic structure
   and a larger molecular structure.

### Validation and exit gate

- One atom at the origin gives `F = occupancy f T` exactly.
- Translating any atom by an integer lattice vector leaves intensity unchanged.
- Shifting every atom and the origin together changes only the common phase,
  not intensity.
- Analytical coordinate, occupancy, `U_iso`, and cell-metric derivatives match
  centered finite differences away from domain boundaries.
- Rust/reference randomized comparisons cover triclinic cells, complex
  amplitudes, empty reflection batches, and invalid arrays.
- JVP, VJP, and the bounded dense Jacobian satisfy adjoint consistency.
- No Python callback occurs per atom or reflection.

## Implementation unit 12: symmetry and reflection generation

Status: implemented and internally reviewed on 2026-08-05. Exact conventions,
range semantics, validation, and performance are recorded in
`docs/symmetry-reflections.md`. The external pinned behavior fixture remains
pending an available checkout; no oracle equivalence is claimed yet.

### Work

1. Implement exact symmetry-operation validation, composition, closure checks,
   and reciprocal-index transformations.
2. Expand each asymmetric-unit site to unique equivalent positions, correctly
   handling special positions without double counting.
3. Generate reciprocal-lattice candidates inside a `d`, `Q`, CW `2theta`, or
   TOF range. Use the reciprocal-metric eigenvalue bound for safe triclinic
   index limits before optimization.
4. Detect systematic absences from exact translational phase cancellation.
5. Build symmetry orbits, canonical Miller representatives, multiplicities,
   and deterministic reflection IDs. Accidental equal-d families remain
   separately identified even though their profile contributions overlap.
6. Derive crystal-system cell parameterizations from the rotational group so
   later lattice refinement cannot violate required metric constraints.
7. Add a prepared reflection generator that caches group topology while
   recomputing metric-dependent values.
8. Probe the pinned oracle for reflection-family selection, multiplicity,
   absence, special-position, and non-standard-setting behavior and document
   every conversion without adopting its reflection-table layout.

### Validation and exit gate

- Cover P1, inversion, primitive screw/glide, body-centred, face-centred, and a
  non-standard setting.
- Verify known centring and screw/glide extinction rules from closed-form
  examples.
- Orbit size equals reported multiplicity; every generated member maps to the
  canonical family; no allowed family is emitted twice.
- Special-position multiplicities and structure factors agree whether computed
  from asymmetric sites or an explicitly expanded unit cell.
- Reflection sets are invariant under operation ordering and deterministic
  across runs.
- A brute-force reference enumerator agrees on randomized bounded cells and
  operation sets.

## Implementation unit 13: CIF import and fixed-cell CIF-to-Le Bail

### Work

1. Add an optional `cif` dependency group and `rietveld.io.cif`. Importing the
   base package must not require the parser backend.
2. Define `read_cif(path_or_text, *, block=None, strict=True)` and a backend
   protocol returning only typed public models and structured diagnostics.
3. Implement the first adapter with Gemmi. Pin a tested compatible version
   range; record its license and do not vendor or modify its source.
4. Support the current IUCr core CIF names plus common legacy aliases for cell,
   atom sites, occupancies, isotropic/anisotropic displacement input, and
   symmetry identifiers/operations.
5. Resolve symmetry in this precedence order: explicit operations, Hall
   symbol, unambiguous extended Hermann-Mauguin symbol plus cell setting, then
   space-group number. Ambiguity is an error in strict mode and a visible
   warning otherwise.
6. Parse standard uncertainties without treating them as part of the value;
   distinguish missing (`.`) and unknown (`?`) data; preserve source block,
   labels, chemical symbols, isotope/charge annotations, and unused metadata
   needed for diagnostics.
7. Convert `B`/`U` and Cartesian/fractional forms explicitly. Reject conflicting
   duplicate definitions unless their agreement is within a local documented
   tolerance.
8. Add file-size, row-count, and atom-count limits to the public adapter and
   clear errors for unsupported modulated, magnetic, or macromolecular cases.
9. Add `LeBailPhase.from_structure(...)` and a convenience CIF-to-Le Bail
   constructor for a fixed cell. It calls the Rust reflection generator from
   unit 12; the CIF adapter itself still performs no diffraction calculation.
10. Permit a cell-and-symmetry-only structure for Le Bail. Atom sites are not
    required because its integrated reflection intensities are independently
    extracted.

### Validation and exit gate

- Test single/multiple blocks, quoted values, loops, uncertainties, missing
  values, legacy tags, explicit operations, alternate settings, disorder, and
  anisotropic input.
- Compare typed output with hand-authored expected records, not live parser
  objects.
- Imported structures persist and reload without the CIF backend installed.
- CIF import performs no structure-factor or profile calculation.
- Test fixtures have explicit redistribution provenance.
- A user can run fixed-cell Le Bail directly from a CIF without manually
  calculating `hkl`, `d`, or `2theta`, before scattering tables are available.
- Anisotropic displacement input is preserved losslessly, but the initial
  isotropic structure-factor path must report it as unsupported unless the user
  explicitly requests and records an equivalent-isotropic conversion.

Gemmi is selected only for parsing and setting resolution. Its official
documentation describes CIF parsing, small-structure extraction, and explicit
space-group operations; the Rust engine independently applies those operations
and calculates diffraction.

## Implementation unit 14: scattering models and data provenance

### Work

1. Define a versioned `ScatteringModel` contract that evaluates all unique
   species over a contiguous `s` array and returns complex amplitudes plus
   `df/ds` where applicable.
2. Implement a built-in non-resonant neutral-atom X-ray form-factor model in
   Rust from a citable open parameter table.
3. Implement non-magnetic coherent neutron nuclear scattering in Rust with
   explicit element/isotope selection and citable open data.
4. Store table version, source, units, checksum, transformation script, and
   redistribution license under a source-data ledger. No table is extracted
   from GSAS-II source or fixtures.
5. Preserve ionic charge and isotope metadata even when a selected model cannot
   use it; strict mode errors and permissive mode diagnostics are explicit.
6. Permit vectorized Python scattering providers for research models. They run
   once per species/reflection batch before the native sum, never once per atom
   or peak sample. Built-in production models remain Rust-native.
7. Reserve typed additions for anomalous `f' + i f''`, electron scattering,
   magnetic neutron scattering, and wavelength-dependent edge models.
8. Probe the pinned oracle with one species at a time over a controlled
   scattering-vector grid to validate amplitude and derivative conventions;
   source tables still come only from reviewed publications/data sources.

### Validation and exit gate

- Reproduce every tabulated coefficient at serialization precision and test
  table checksums.
- Verify limiting values, `s` derivatives, interpolation boundaries, unknown
  species/isotopes, and vectorized provider validation.
- Rust values agree with an independent Python transcription of the published
  parameterization.
- Benchmarks demonstrate caching by unique species rather than repeated table
  evaluation per atom.

## Implementation unit 15: structural intensities and fused patterns

### Work

1. Combine symmetry-expanded sites, scattering models, displacement factors,
   multiplicity, and phase scale into `StructureFactorResult` and an integrated
   reflection-intensity batch.
2. Introduce typed diffraction geometry and correction models. Begin with one
   documented monochromatic X-ray Lorentz/polarization convention and an
   explicit neutral correction for callers supplying already corrected data.
3. Reuse the existing preferred-orientation provider through one named
   correction path; add tests preventing double application.
4. Chain `dI/dp` and cell-dependent peak-position derivatives into the existing
   support-limited profile derivatives.
5. Implement `rietveld-engine` one-call structural pattern calculation so
   Python submits structures and arrays once. Intermediate per-reflection
   arrays may be returned as diagnostics but never orchestrated in Python.
6. Add `RietveldPhase` and structural calculation result models with stable
   site/reflection/parameter labels.
7. Extend persistence with a new format version and an explicit migration path
   from reflection-only version 1 bundles.
8. Extend the external oracle adapter only with narrowly named structural
   outputs required to compare complex `F`, correction factors, integrated
   intensities, and parameter perturbations at the pinned revision.

### Validation and exit gate

- Closed-form P1, body-centred, face-centred, rock-salt-like, and
  diamond-like structures cover cancellations and multiplicities.
- Intensities are invariant to atom ordering, symmetry-operation ordering,
  unit-cell translations, origin shifts, and equivalent setting transforms.
- Structure-factor, intensity, peak-position, and end-to-end pattern
  derivatives pass finite differences and JVP/VJP adjoint checks.
- Sampled integrated peak areas match calculated integrated intensities within
  the documented finite-support loss.
- Independent NumPy and optional external-oracle comparisons cover reflection
  parameters, complex `F`, integrated intensities, and `Ycalc`. GSAS-II, when
  used, runs only outside the package to produce validation records.
- Benchmarks report structure-factor time, profile time, combined time, memory,
  reflection count, site count, and symmetry expansion count.

## Implementation unit 16: lattice-refining CIF-to-Le Bail

### Work

1. Extend the fixed-cell CIF-to-Le Bail path without changing the lower-level
   explicit-reflection API.
2. Generate reflections from experiment limits with a guard band derived from
   bounded lattice parameters. Hold the Miller-family topology fixed during a
   Jacobian evaluation and report/regenerate only between accepted iterations.
3. Map independent crystal-system lattice parameters to the full cell and
   analytically propagate `d`, CW `2theta`/TOF position, width, and profile
   derivatives.
4. Preserve reflection intensities by stable Miller-family ID when a prepared
   reflection domain is regenerated.
5. Extend the Dioptas adapter with CIF import, generated reflection markers,
   and Le Bail results without introducing GUI dependencies.

### Validation and exit gate

- A user can refine the allowed lattice parameters in a CIF-backed Le Bail run
  without manually regenerating reflections.
- Monochromatic X-ray, monochromatic neutron, and TOF position generation agree
  with their existing experiment conventions.
- Lattice derivatives pass finite differences for every crystal system.
- Peaks moving across the visible pattern boundary follow the documented guard
  and regeneration semantics without unstable IDs or silent intensity loss.
- Explicit-reflection Le Bail remains backward compatible.

## Implementation unit 17: first full Rietveld refinement

### Work

1. Implement typed `RietveldInput`, `RietveldOptions`, iteration/checkpoint
   records, and `RietveldResult` in the already reserved method module.
2. Begin with multi-phase, non-resonant monochromatic X-ray refinement of phase
   scale, allowed lattice parameters, fractional coordinates, occupancy, and
   `U_iso`, alongside the mature profile and background parameters.
3. Derive site-coordinate degrees of freedom from the site stabilizer so an
   atom on a special position cannot be refined away from its symmetry
   constraints accidentally.
4. Add typed constraints for shared occupancies, composition sums, shared
   displacement parameters, fixed atoms, and user-selected parameter groups.
5. Connect the optimizer to native structure-factor/profile JVP and VJP
   operations. A dense pattern Jacobian is allowed only below an explicit size
   threshold.
6. Detect and report scale/occupancy, scale/displacement, overlapping-phase,
   and special-position rank deficiencies. Do not hide singular directions by
   silently regularizing them.
7. Return final structures, reflection tables, phase curves, residual metrics,
   parameter uncertainties where identifiable, correlations, warnings, and a
   resumable checkpoint.
8. Add the neutron nuclear path after X-ray structural refinement passes the
   same contracts; scattering-model selection must not branch the optimizer.
9. Compare controlled one-parameter refinement steps and converged results with
   the pinned oracle to catch parameter scale, sign, constraint, and weighting
   differences. Optimizer iteration identity is not required when both methods
   satisfy the same physical objective and tolerances.

### Validation and exit gate

- Synthetic exact recovery covers coordinates, occupancy, `U_iso`, scale,
  lattice, profile, background, overlapping peaks, and multiple phases.
- Special positions and equality/composition constraints remain satisfied at
  every accepted iteration.
- Packed derivatives pass centered finite differences; native JVP/VJP pass
  adjoint consistency on full patterns.
- Parameter ordering, repeated runs, checkpoint/resume, and reflection-domain
  regeneration are deterministic.
- Published structures and optional external-oracle cases compare structure
  factors, intensities, `Ycalc`, residual trends, and refined parameters.
- A short script can refine a CIF-backed phase without dictionaries, project
  files, or per-reflection callbacks.

## Implementation unit 18: advanced crystallographic physics

These are separate reviewed increments after the first Rietveld exit gate:

1. symmetry-constrained anisotropic displacement tensors;
2. wavelength-dependent anomalous X-ray scattering and unmerged Friedel pairs;
3. absorption/extinction and additional instrument geometries;
4. magnetic neutron structures and magnetic CIF;
5. electron scattering;
6. modulated structures, stacking faults, diffuse scattering, and other
   specialist models through typed native/provider boundaries.

Each addition needs equations, independent references, analytical derivatives,
finite-difference tests, persistence, and a realistic benchmark. Unsupported
CIF features remain explicit diagnostics until their complete calculation path
exists.

## Review and commit discipline

Each implementation unit is its own reviewable commit series. Before merging a
unit:

1. review equations, coordinate conventions, units, and data provenance;
2. review public types for scriptability and GUI independence;
3. review array ownership and ensure no Python atom/reflection hot loop exists;
4. compare the Rust implementation with the independent NumPy reference;
5. run centered finite differences and JVP/VJP adjoint checks;
6. run `cargo fmt --check`, strict Clippy, all Rust tests, Ruff, and all normal
   Python tests;
7. run the new realistic benchmark and compare with the preceding committed
   result;
8. inspect persistence migrations and dependency/license changes;
9. commit only after review fixes are complete.

GSAS-II validation is optional, external, and pinned. It can demonstrate that
observable results agree, but it never supplies equations, tables, production
objects, runtime calls, or implementation code.

## Sources governing the import boundary

- The [IUCr core CIF dictionary](https://www.iucr.org/resources/cif/dictionaries/cif_core)
  defines the official crystallographic data names and versions accepted by the
  import adapter.
- The [Gemmi CIF documentation](https://gemmi.readthedocs.io/en/stable/cif.html)
  documents the proposed first parser backend.
- The [Gemmi symmetry documentation](https://gemmi.readthedocs.io/en/latest/symmetry.html)
  documents its setting table and export of explicit rotation/translation
  operations. Rust receives those operations as plain exact data and performs
  the calculations independently.
- Gemmi is available under MPL-2.0 or LGPL-3.0-or-later according to its
  [upstream repository](https://github.com/project-gemmi/gemmi). Dependency
  notices and packaging must preserve the selected upstream terms.

Equation and numerical-data publications will be added to the source ledger
before their corresponding implementation unit starts. A plan entry is not
permission to copy a proprietary table or another program's implementation.
