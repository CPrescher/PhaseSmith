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

Final commit, local checks and remote workflow results are recorded here after
completion. Until that receipt is filled, this page is a preparation record, not
an assertion that all release gates have passed.

## Publication boundary

The new names `phasesmith-single-crystal` and `phasesmith-hdf5` need first uploads
with an owner credential before crates.io trusted publishing can cover them.
Follow the [reviewed bootstrap sequence](releasing.md#registry-configuration-and-new-crate-names).
No dry-run command creates the release tag or publishes packages. Public Read the
Docs `stable` remains on the published release until its release build completes.
The scientific scope remains bounded candidate search, with the exposed corpus
and provisional acquisition geometry documented in their original receipts.
