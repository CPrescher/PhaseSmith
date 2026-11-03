# 0.8 release dry-run record

This records the consolidated 0.8 preparation, separate from the earlier
[promotion receipt](release-candidate-0.8.md) and [package audit](package-review-0.8.md).
Version 0.8.0 remains unpublished until the reviewed release commit is tagged and
registry publication is explicitly performed.

## Changes under validation

- Add a runnable NumPy-only synthetic tutorial, full native file-workflow input,
  configuration/result/error guide, migration instructions and shared Rust/Python
  geometry, integration/scaling and structure-search mathematics.
- Consolidate three unpublished numerical component crates into
  `phasesmith-single-crystal`, with selectable `spots`, `solution` and `row-search`
  features. Keep facade/Python APIs, equations, protocols and reference data.
  HDF5 remains separate. The public workspace now has eleven crates.
- Correct minimum-NumPy positional-test startup: both parent and spawned workers
  inherit single-thread BLAS settings before NumPy import. Existing exact model
  bytes and audit comparisons remain the gates, with no relaxed tolerances.
- Add verified Rust archives, strict rendered docs and native Linux ARM wheel
  tests to the nonpublishing release workflow. Gate registry jobs on all checks.

## NumPy 1.26 investigation

The earlier CI run on Linux x86-64 failed one legacy positional-proposal test:
serial and spawned fits differed by roughly 2e-13 in the audit histories. Its
helper changed environment variables after NumPy was imported, while new workers
loaded BLAS with one thread. The documented process-search contract requires
limits before starting Python. On the same runner, the diagnostic with those
startup limits produced exact audit equality and exact model bytes, using NumPy
1.26.4/OpenBLAS 0.3.23.dev. Arm64 macOS also passed that configuration.

The diagnostic `benchmarks/positional_reproducibility.py` retains first-step
inputs/normal matrices, runtime thread information and complete comparisons.
Minimum-NumPy CI runs a checked proper-startup case and an unchecked late-setting
control. A late control may happen to match on another processor; that does not
make changing environment strings a supported way to reconfigure loaded BLAS.

## Execution receipt

The first consolidated source commit is `22f7369b62a5ae9e213c0ced801ccf64a79da96c`.
Its local checks passed: 567 Rust tests (34 explicitly ignored), 3,381 Python tests
(27 skipped and 34 external-data/oracle tests deselected), strict Clippy/rustdoc,
minimal and individual-feature builds, Ruff, strict MkDocs and API/version/skill/
mathematics checks. All 70 relocated numerical source files match their originals
after namespace substitution and Rust formatting.

A fresh installed macOS arm64 wheel with only NumPy passes native-capability and
HDF5-filter smoke, the synthetic tutorial and offline advisor-cycle example.
The wheel links system libraries only. Its source archive contains all three
consolidated modules and optimizer notices, without retired crate source paths.

A fresh original-count MgSiO3 run from that installed wheel accepts Pbca with
Mg16 Si16 O48. All 68 arrays in ten NPZ files, the complete candidate/result
record (including phasing histories), P1 CIF and selected-setting CIF are bitwise
identical to the earlier validated native control. The recorded elapsed time is
15.49 s; this is a regression control, not a new performance comparison or
independent scientific validation.

The first [remote dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/41698688678)
verified Rust package compilation but exposed two harness defects: the Rust
archive upload pointed above Cargo's `tmp-crate` directory, and external-working-
directory pytest could not import repository benchmark helpers. The corrected
workflow retains the verified archives and adds only the repository root to the
test helper search path; `python/` is not added, preserving installed-package
coverage. The installed wheel also passes all 3,381 tests from outside the
checkout, and all 21 explicit native/Python cross-comparison tests pass.
Package-content review adds the canonical PhaseSmith MIT license to each public
crate alongside applicable third-party notices; release metadata checks enforce parity.

The [second run](https://github.com/CPrescher/PhaseSmith/actions/runs/41699952961)
passes every target except Windows, where 3,377 tests pass and four benchmark-
harness checks fail. Fixes preserve the benchmark's owned process-tree timeout
on Windows, report unavailable load-average metadata as null, and force LF for
committed corpus/protocol JSON so Windows checkout conversion cannot invalidate
recorded byte hashes. Frozen JSON records and numerical thresholds are unchanged.
The 66 focused portability/provenance tests pass locally, including emulated
Windows checkout hashes. A final full-platform run verifies these corrections.

## Publication boundary

The new names `phasesmith-single-crystal` and `phasesmith-hdf5` need first uploads
with an owner credential before crates.io trusted publishing can cover them.
Follow the [reviewed bootstrap sequence](releasing.md#registry-configuration-and-new-crate-names).
No dry-run command creates the release tag or publishes packages. Public Read the
Docs `stable` remains on the published release until its release build completes.
The scientific scope remains bounded candidate search, with the exposed corpus
and provisional acquisition geometry documented in their original receipts.
