# PhaseSmith Project Brief

## Purpose and current release

PhaseSmith is a powder and single-crystal diffraction library with native Rust
numerical kernels and an optional typed Python/NumPy frontend. GUI applications
own presentation, acquisition configuration and application state. The current
source version is the **0.8 development candidate**, not a publication receipt.

Powder workflows include profile/background evaluation, crystallography,
CW/TOF structural calculation, Le Bail, Pawley, Rietveld and quantitative analysis.
Single-crystal workflows cover masks/shadows, spots and rotation positions,
reciprocal indexing, lattice/grain discovery, geometry fitting, prediction,
integration, learned profiles, explicit corrections/scaling and structure search.

The single-crystal endpoint is **structure search and candidate assessment**.
Positional detector/crystal fitting and internal coordinate/ADP polishing are
separate operations; no standalone single-crystal structure-refinement workflow
is promised. The interpreter-free coordinator now lives in
`phasesmith-workflows::single_crystal`, behind its optional `single-crystal`
feature and included by the facade single-crystal capability. It owns stage
coordination, independent family splits and native design construction; file I/O
stays in the HDF5/persistence adapters. The first native recipe uses a supplied
primitive-orthorhombic metric and explicit source/DAC assumptions. Existing Python
orchestration remains an independent regression path. See
[the native workflow](docs/single-crystal-native-workflow.md) for scope, execution
budgets and measured numerical differences; strict final-search parity is not
implied by successful labelled recovery.

Unknown symmetry is a native workflow mode. It must not use an exact
space group to filter predictions or define scaling families before discovery.
Retain all measured signed reflections; masks/unmeasured crossings never supply
absence evidence. Assess intensity symmetry before empirical scaling, enumerate
metric-compatible conventional Hall settings, and report ranked alternatives,
origin/model evidence and bounded-search omissions. Exact-symmetry input remains
an explicit supported mode. Automatic discovery is not proof of uniqueness or a
complete SHELXT implementation. Freeze data-run controls before frame reads.
The measured v33 MgSiO3 run withholds exact symmetry and Laue group, assesses
345 metric-compatible settings, phase-screens 64 and searches 12 models. It
ranks Pbca first and recovers Mg16 Si16 O48. Shared-map assignment and four
ordered independent P1 workers reduce observed full time from about 47.7 to
36.2–36.5 s. Its bounded coverage does not prove uniqueness; the supplied-group
control remains bitwise unchanged. See [the assessment guide](docs/space-group-search.md).
Further speed experiments preserve full-resolution final measurements and freeze
cache reuse, a published optional light-element centrosymmetric exit, and coarse
quality-controlled P1 phasing separately. See the
[speed experiment plan](docs/space-group-speed-plan.md); success and generality
must be measured rather than inferred from the reduced workload.
The current v40 example uses explicit `fast`/`auto`/`thorough` effort. `auto`
first attempts the short coarse path and only starts full-resolution phasing and
broader model search after numerical acceptance fails; both share the original
remaining deadline and retain complete pass evidence. The exposed MgSiO3 search
is about 0.5 s on the measured M4 Pro, versus about 0.65 s for the measured
10-worker SHELXT control. A forced fast-timeout test recovers the correct Pbca
structure through the full-resolution fallback. These timings and candidate
acceptance thresholds do not establish general speed/reliability or uniqueness.
See the [adaptive-effort report](docs/space-group-search-speed.md).

## Ownership and dependency direction

| Layer | Owner |
| --- | --- |
| Profile/background kernels | `phasesmith-core` |
| Cells, symmetry, scattering, structure factors | `phasesmith-crystallography` |
| Bounded worker/execution policy | `phasesmith-execution` |
| Structural pattern composition | `phasesmith-engine` |
| Parser-independent application records | `phasesmith-model` |
| CIF, powder, space-group and instrument adapters | `phasesmith-io` |
| Powder and single-crystal workflows | `phasesmith-workflows` |
| Versioned projects and lossless numeric archives | `phasesmith-persistence` |
| Detector, geometry, profile and scaling kernels | `phasesmith-spots` |
| Independent DAFi-based row grouping | `phasesmith-row-search` |
| Structure-search kernels | `phasesmith-solution` |
| Optional bundled detector-file input | `phasesmith-hdf5` |
| External comparisons and validation orchestration | `phasesmith-validation` |
| Native application facade | `phasesmith` (`crates/phasesmith-rs`) |
| Array translation and Python bindings | `phasesmith-py` and `python/phasesmith` |

Keep serialization, file adapters, FFT/search dependencies and application state
outside `phasesmith-core`. Do not collapse these dependency boundaries merely
to reduce the number of directories. Large binding/coordinator modules can be
split internally while preserving public signatures and operation order.

## Numerical and provenance invariants

- GSAS-II is a pinned external oracle only. No runtime import, implementation
  copying/mechanical translation or GSAS-II object model enters the kernel.
- Derive equations independently from publications/first principles. Keep the
  readable Python references independent of Rust production evaluation.
- Compute values and analytical derivatives in the same peak/sample pass.
  Finite support, clipping and endpoint semantics are observable contracts.
- Preserve deterministic scalar/reduction order unless a measured, documented
  change establishes explicit tolerances. Never loosen global tolerances.
- Preserve signed observations, uncertainties, flags, explicit equivalences,
  supplied priors and truthful termination. A low R1 is not structure proof.
- Keep source/teacher/geometry identity, immutable array ownership, bounded
  allocations and cancellation/result-order semantics explicit.
- Do not silently regenerate golden data, rewrite historical receipts or select
  scientific settings using source HKLs, reference coordinates/counts or R1.
- Scattering tables and vendored helpers retain exact provenance and notices.
  Generated table size alone is not a reason to hand-rewrite scientific data.

## Public APIs and compatibility

The Python owner is `phasesmith.single_crystal`, with explicit exports and lazy
submodule ownership. The maintainer chose to remove pre-release
`phasesmith.experimental` imports after migrating their Dioptas consumer.
Independent references live under `phasesmith.single_crystal.reference` as
validation support, outside the supported production exports.

The Rust facade defaults to `single-crystal` (`spots`, `solution`, `row-search`).
`--no-default-features` keeps a smaller facade; `hdf5` remains optional for Rust.
Experimental Cargo feature aliases are removed. Normal Python builds include
the native capabilities and bundled HDF5. Published powder compatibility aliases
from the 0.5 namespace transition remain scheduled for removal at 1.0.

`ElementSearchConfig.validated_v1` contains a literal, complete, typed expansion
of the reviewed controls. Changing a low-level default must not change that
named profile. Explicit low-level algorithm defaults are preserved.

New solution summaries use `phasesmith.single-crystal-solution.v1`; explicit
v4/v5 metadata migration and legacy summary output remain available. Historical
profile-case/reduction/cache schema identifiers remain readable. Saved portable
formats are pickle-free; retired Python module paths in arbitrary old pickles
are not automatically migrated. Implementation hashes intentionally invalidate
numerical caches after source changes, without rewriting their provenance.

## Dependencies and builds

The workspace requires **Rust 1.89.0**, matching the locked numerical dependency
floor; Python requires 3.11+ and NumPy 1.26+. The only mandatory Python runtime
dependency is NumPy. SciPy/Gemmi are explicit reference/interop/example extras,
not implicit production numerical fallbacks. Missing native omega-background
support raises an actionable error; `backend="scipy"` is an explicit reference.

HDF5/Bitshuffle/LZ4/zlib/LZF build from Cargo sources. Source builds need a C
compiler, CMake >=3.26 and Git; prebuilt wheels need no installed HDF5, h5py or
filter plugin. Keep the adapter optional for native Rust consumers. The direct
`libz-sys` static feature is intentional link configuration, not an unused import.

Keep Cargo/Python versions, the changelog, API snapshots and crate metadata
synchronized. Do not modify released snapshots to accept surface drift.
Document default, minimal and optional-feature builds truthfully.

## Evidence and remaining scientific limits

The MgSiO3 example is exposed development/regression data with supplied
cell/element priors and optional exact symmetry, provisional spindle angles and explicit aperture
assumptions. Its eight source files total 470,052,794 bytes and stay external.
No reference atomic count/coordinate, source HKL or archived observation enters
the numerical worker. Reference grading follows candidate save and worker exit.

Retained v11–v24 execution checks reproduce masks, signed intensities, candidate
records and phasing histories bitwise: R1 .04085182070733226; Mg16 Si16 O48;
4,125 observations including 220 negative values; maximum/RMS reference
errors .0075236/.0045385 Å. The full measured workflow is about 19 s on the
recorded M4 Pro, not a platform-independent guarantee. The example uses at most
ten workers per concurrent stage and explicit independent memory budgets.

The [fully Rust coordinator replay](docs/single-crystal-native-validation.md)
uses the same priors and scientific controls, with all I/O/output native. It
recovers Mg16 Si16 O48 and passes reduction tolerances in 16.24–16.57 s observed
full-process time. Small native matrix roundoff changes the retained internal
polish trajectory: R1 .041128354912261904; the original R1-difference and
maximum-candidate-coordinate parity gates fail. Keep that qualification separate
from labelled structure recovery; the existing ADP-bound limitation is not fixed
by this orchestration change.

Candidate polishing has a documented ADP-bound stagnation limitation; do not
call every polish converged. The 65-case corpus is now exposed regression data,
not an untouched holdout. New generalization claims need new untouched evidence.
Powder limitations include the frozen QARR 1h bounded-workflow failure and
qualified GSAS-II FCJ/finite-support differences; preserve their reports.

## Validation and documentation

For numerical changes, follow `AGENTS.md`: equations/conventions, independent
references, values/derivatives, boundary/invalid-input tests, deterministic
comparisons, finite differences, normalization/moments, oracle coverage or its
reasoned exclusion, and realistic benchmark evidence.

Required checks include:

```sh
ruff check .
ruff format --check .
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
pytest
python scripts/public_api_snapshot.py --check
python scripts/release_version.py
python scripts/sync_math_docs.py --check
python scripts/sync_skill.py --check
mkdocs build --strict
```

Also check the minimum Rust toolchain, default/minimal/focused features, normal
installed wheels and source reconstruction. Cross-platform CI/manual release
execution remain separate from local results. Never tag/publish as a side effect
of maintenance or benchmarking.

User guides are in `docs/`, implementation evidence is indexed by
[development.md](docs/development.md), and the current assessment is
[package-review-0.8.md](docs/package-review-0.8.md). Raw data, full arrays, trial
outputs and distribution artifacts stay outside Git. Remove regenerable build
caches only after preserving unique evidence and verifying recovery artifacts.

The previous 3,400-line brief is preserved exactly in
[the repository archive](docs/archive/project-brief-before-package-review-2026-10-31.md).
It is excluded from documentation search so historical status cannot be mistaken
for current guidance. Frozen protocols and historical numerical receipts remain
unchanged; use their recorded Git/source snapshots for historical replay.
