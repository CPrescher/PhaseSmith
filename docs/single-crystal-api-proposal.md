# Proposed single-crystal public API

**Historical preparation record.** The 0.8 implementation now follows the
[supported single-crystal guide](single-crystal.md). The evidence, blockers and
proposals below describe their recorded baseline; current implementation and
validation are tracked in the [0.8 release candidate](release-candidate-0.8.md).


Status: promotion proposal, 31 October 2026. These imports do not exist yet.
The reviewed implementation baseline is `54784c5`. See the
[promotion audit](single-crystal-promotion.md) for evidence and release gates.

## Namespace and ownership

Use `phasesmith.single_crystal` for the supported scientific workflow. Keep
`phasesmith.crystallography`, `phasesmith.symmetry`, `phasesmith.scattering`,
`phasesmith.execution` and `phasesmith.io.hkl` as shared owners of existing types.
Avoid duplicating `UnitCell`, symmetry operations or reflection-observation types.

Use explicit submodule imports. The namespace root exposes submodules and a
small documented set of shared records, avoiding eager optimizer/image imports.
Independent NumPy reference modules and search internals are validation support;
their functions do not automatically become stable exports.

| Proposed module | Existing implementation under `experimental` | Initial public surface to review |
| --- | --- | --- |
| `single_crystal.spots` | `spots` | `SpotOptions`, `SpotTable`, `SpotFlag`, `detect_spots`, `extract_spots` |
| `single_crystal.rotation` | `rotation`, `reciprocal` | Explicit panel/scan geometry, `RotationOptions`, `PositionTable`, `locate_series`, position-to-reciprocal conversion |
| `single_crystal.indexing` | `indexing`, `cell_metrics`, `orthorhombic`, `cell_comparison` | Typed lattice candidates, options, `find_lattices`, `compare_cells`; distinguish metric hypotheses from symmetry identification |
| `single_crystal.grains` | `grains`, `rows`, `row_grains`, `ownership`, `retention`, `recovery`, scheduling/cache helpers | `find_grains`, `find_grains_ensemble`, explicit FFT/Row search options and evidence records; keep consolidation/worker helpers internal |
| `single_crystal.prediction` | `prediction` | `Predictions`, `ScanPredictions`, `predict_spots`, `predict_scan`, positional `refine_crystal` with documented scope |
| `single_crystal.geometry` | `geometry_refinement` | `GeometryOptions`, `GeometryCalibration`, `refine_geometry`; named coordinate-map boundary and gauge |
| `single_crystal.integration` | `integration`, `integration_quality` | `IntegrationOptions`, `IntegrationFlag`, `ReflectionTable`, `integrate_series`, quality diagnostics |
| `single_crystal.profiles` | `profiles`, `profile_refinement` | `ProfileObservations`, `ProfileOptions`, `ProfileModel`, profile preparation/learning/fitting; strong fits and fixed weak estimators explicit |
| `single_crystal.reduction` | `reduction`, `reduction_io`, `reduction_cache`, intensity response/uncertainty and sensor models | `ReductionConfig`, `ReductionResult`, `reduce_observations`, portable case I/O and typed explicit correction models |
| `single_crystal.scaling` | `scan_scaling`, `empirical_scaling` | Explicit family-labelled relative/empirical scaling designs, options, fitted models and reserved-family diagnostics |
| `single_crystal.background` | `omega_background` | Supported array-oriented omega-background options and result contract |
| `single_crystal.solution` | `solution`, `solution_symmetry`, internal search/reference modules | `SolutionOptions`, `SolutionCancellation`, `SolutionResult`, `solve_structure`, `solve_structure_elements`, candidate export |

The table is a module/export proposal, not a claim that every non-underscore name
in the current modules is public. Final `__all__` lists should enumerate reviewed
types and functions only. Algorithm/cache/worker helpers remain private.

## Contract review before implementation

- Arrays: explicit shape/dtype/range checks; immutable result snapshots; no
  hidden copies of entire scans or mutation of caller arrays. Signed measured
  intensities survive integration, scaling, export and search preparation.
- Coordinates: integer image pixel centers, image `[y, x]`, panel axes,
  handedness, wavelength/length/angle units and exposure intervals are named.
  Library geometry never guesses an application's flip or detector hardware.
- Supports: preserve clipping, half-open pixel/angle intervals, support-union
  behavior, endpoint flags and conditional derivatives/uncertainties.
- Priors: record supplied cell, symmetry, element types, detector response,
  calibration and explicit intensity equivalences. Metric symmetry cannot
  silently supply intensity equivalences or certify a space group.
- Decisions: retain stage diagnostics, rejected/ambiguous candidates, masking,
  family partitions and termination reasons. An oriented lattice candidate
  does not directly certify a physical grain or a material identity.
- Runtime: bounded execution and memory policies, cancellation and application
  concurrency budgets remain explicit. No process-global worker pool.
- Optimization: retain current independent NumPy references and Rust fused
  values/derivatives; migrate namespace without changing arithmetic or defaults.

All promoted production operations must be usable through typed Rust entry
points, with Python as an optional frontend. The current experimental
shape/scaling optimizers still need native ports; geometry fitting is now native,
including robust optimization and fit diagnostics; their existing SciPy
requirement must not become the final production architecture. Detection,
summation, masking and structure search already have native kernels. HDF5
input now belongs to the separate optional `phasesmith-hdf5` adapter with bundled
HDF5/Bitshuffle/LZ4, exposed by the `hdf5` feature without h5py/hdf5plugin.
Importing the public namespace must require neither Dioptas, Qt nor an external
solver. See [native HDF5 input](native-hdf5.md).

## Structure-search configuration

The current element-search entry point exposes many research/performance flags;
the validated frozen combination differs from its defaults. Add a named, typed,
versioned configuration that expands to the exact reviewed options and records
that expansion in results. Separate scientific search choices, resource limits
and implementation optimizations in the configuration; preserve all explicit
legacy arguments during migration and reject conflicting specifications.

Do not silently make the most recent benchmark flags the new global defaults.
First compare accepted candidates, stop reasons, counters and deterministic
repeated runs against the validated configuration. Publicly supported methods
can still expose clearly scoped optional research strategies.

Keep the existing two contracts: known full-cell composition search and
element-type-only search. Supplied symmetry remains a prior in both. Return a
candidate even when a numerical target is missed if the existing method does
so, and preserve its quality/termination qualifications. Internal candidate
coordinate/ADP polishing is part of search. No
`single_crystal.refinement` module or independent full-refinement claim is
proposed in this release.

## Compatibility and persistence

Move each implementation to one canonical module and make old
`phasesmith.experimental.*` imports forward to the same classes/functions.
Do not maintain two independently imported copies of dataclasses: identity,
`isinstance`, saved recipes and application adapters must keep working.
Document one deprecation window in the release notes, with warning behavior
consistent with the existing package policy.

Preserve the old `python -m phasesmith.experimental.solution` and reduction CLI
entry points as delegating wrappers. Provide the new module entry points with
identical explicit output-ownership rules and exit statuses.

Existing result schema IDs such as `phasesmith.experimental-solution.v4/v5`
and recipe/cache versions are observable data contracts. Keep readable legacy
records and introduce any supported schema through an explicit version and
migration. Namespace movement changes implementation/cache identity; disclose
cache invalidation and retain original teacher/source provenance. Do not rewrite
archived results or historical frozen source hashes.

## Build and snapshot changes

Normal Python wheels must contain the reviewed native spot, row and solution
bindings. Rust facade defaults should include the corresponding supported
features, with names that describe capabilities, plus a tested minimal-build
option. Retain old experimental feature names as compatibility aliases if
feasible. Check every facade dependency against packaged crate publication.

Extend `scripts/public_api_snapshot.py` with explicit submodules: its current
fixed module list does not capture these experimental APIs. Add the new release
snapshot without modifying released snapshots. Test import identity through old
and new paths, CLI delegation, schema migration, missing optional dependencies
and the unchanged numerical suites from ordinary installed wheels.
