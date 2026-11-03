# Unfinished-work reconciliation, 3 November 2026

## What was unfinished and why

The original `~/Programming/PhaseSmith` checkout was on
`codex/experimental-spots` at `297d157`, with seven modified tracked files and
44 untracked research files. The October integration deliberately preserved
that unfinished state separately: see the [original preservation boundary](structure-search-merge.md).
This was a scope boundary, not evidence of a failed Git merge.

The consolidated starting point is `codex/enstatite-raw-workflow` at `521aaa2`.
It includes both HKL branches, promotion to the 0.8 candidate, native workflows
and later search changes. The unfinished files cover three different subjects:

1. Row-staged annular threshold classification, with boundary/reference tests
   and a realistic detector benchmark. This changes execution, not the detector
   statistic, discovery population or intensity estimator.
2. Fe7C3/DIALS position, learned-profile, background and structural diagnostics,
   portable timing records and figures. Some measured intensity-agreement gates
   failed. These are retained research evidence, not release approval.
3. A deferred single-frame multigrain plan. It is now a separate
   [roadmap document](../single-frame-multigrain-plan.md), not an implemented feature.

## Resolution

The Rust detector patch and tests apply cleanly to the consolidated source.
The uncommitted `prediction.py` change was formatting already present in the
consolidated implementation. The rewritten modern project brief is retained;
recovered status and roadmap material are incorporated without restoring its
obsolete 0.7 architecture or experimental public namespace.

Research helpers now import `phasesmith.single_crystal`. Independent references
resolve from its `reference` subpackage. The rotation timing adapter has an
explicit namespace selector for preserved pre-promotion binaries, and its parent
benchmark selects that namespace only for the historical baseline. This does
not restore retired imports in the package. The two synthetic profile-control
scripts require a new output directory so reruns cannot overwrite old evidence.
Formatting-only repairs make the recovered research scripts pass repository lint.

Original portable JSON receipts and figure bytes are preserved. Historical
reports are labelled with a current reconciliation pointer; their measured
results, failed gates, calibration assumptions and intensity qualifications are
not regraded. Full ignored `runs/` directories and validation inputs remain in
their original worktrees. No raw datasets or generated distribution artifacts
are added to Git. The original seven modifications and 44 new files have a
byte-preserved external snapshot and SHA-256 inventory; the compact inventory
is [retained alongside this record](unfinished-work-inventory.json).

## Numerical contract and validation

The unchanged annular mean/noise equations and finite-support convention are in
[the detector guide](../experimental-spots.md) and the
[historical threshold report](../threshold-loop-optimization.md).
The outer/inner squares clip at image boundaries; both thresholds are strict.
The two passes, excluded strong pixels, rectangle subtraction order and all
extraction/centroid reductions are unchanged. The optimized implementation uses
row-sized scratch storage and independent interior lanes, with scalar clipped
columns and no unsafe or architecture-specific code.

The existing independent direct-sum NumPy reference remains unchanged. Added
cases cover nondefault noise/radii, masked NaNs, narrow images, interior/clipped
transitions and adjacent floating-point values at both strict thresholds.
Existing signed-extraction derivative, finite-aperture normalization/moment and
noise tests continue to apply. No profile term or derivative equation is added.
GSAS-II's powder oracle does not cover image thresholding.

Fresh local checks pass: formatting and strict workspace/all-target/all-feature
Clippy; **567 Rust tests/doctests** (34 intentionally ignored); **3,380 Python
tests** (27 unavailable external-input skips, 34 oracle/real-data deselections);
Ruff lint/format; the unchanged public API snapshot; synchronized version,
mathematical documentation and agent skill; and strict MkDocs. Python validation
uses `python -m pytest` from the repository root, which also exposes the research
`benchmarks` namespace to tests in a fresh editable installation.

Both recovered synthetic profile-control scripts execute successfully with the
current API into new external output directories. All 34 PhaseSmith import
statements in recovered research helpers resolve. Optional DIALS/FabIO/pyFAI
experiments and historical measured intensity comparisons are not rerun.

A fresh release build of the clean `521aaa2` baseline is compared with the
reconciled release extension in isolated processes. Complete detector records
match byte for byte on the 768-by-1024 multi-spot workload, with both noise paths.
The estimated-noise median falls from **31.159 to 8.618 ms**; supplied variance
falls from **32.351 to 9.060 ms** (one warmup, eight measured calls per build).
These are local observations, not a general speed guarantee. Native binary and
source hashes and every timing sample are in
[the portable receipt](https://github.com/CPrescher/PhaseSmith/blob/main/benchmarks/threshold-reconciliation-macos-20261103.json).
An initial comparison with the pre-existing installed baseline was preserved
separately and excluded from these fresh-build figures.

The complete synthetic rotation workflows also retain exact arrays, source/mask
hashes and ordered per-frame detector digests at one and four workers:

| Workload | Workers | Baseline median | Reconciled median |
| --- | ---: | ---: | ---: |
| 32 frames, 512² pixels | 1 | 0.348145 s | 0.107190 s |
| 32 frames, 512² pixels | 4 | 0.093797 s | 0.032361 s |
| 16 frames, 2048² pixels | 1 | 2.896137 s | 0.887280 s |
| 16 frames, 2048² pixels | 4 | 0.780557 s | 0.265642 s |

Each configuration has one excluded full warmup and three measured repeats;
baseline and candidate execute sequentially. Portable timing/provenance is in
[the rotation receipt](https://github.com/CPrescher/PhaseSmith/blob/main/benchmarks/rotation-reconciliation-macos-20261103.json).
Full arrays, logs, original file snapshots and preserved binaries remain in the
external `phasesmith-reconciliation` evidence directory for 3 November 2026.
No historical October timing receipt or golden data was regenerated.

## Remaining scope

This reconciliation does not publish a release, push branches, or claim general
Fe7C3 intensity accuracy. `main` remains the released branch until a separate
release review. The original checkout is advanced to a locally reviewed
consolidation branch only after validation, with the original unfinished state
preserved. Removing old worktrees and resolving the unrelated broken Documents
checkout are separate maintenance operations.
