# Validated structure-search integration and threshold Clippy fix

This integration combines the structure-search branch through `91ab07c` with
the experimental intensity, FFT and row-search commits through `297d157`, using
the reviewed resolutions in `e9475ad`. Documentation from both branches, the
optional native crates/features, attribution notices and the shared optional
SciPy test are preserved. Structure-search numerical sources are unchanged.

## Remaining lint fix and preservation boundary

The outstanding `needless_bitwise_bool` lint was present only in the other
checkout's unfinished threshold-loop optimization. The committed integration
already passed strict Clippy. The authorized correction changes that loop's
four Boolean predicates to eager `u8` masks, combines them with integer `&`,
and compares the result with zero. It preserves all four predicate evaluations,
floating-point operations, strict comparisons, masks and thresholds, without a
lint suppression.

The minimal correction is applied to
`/Users/clemens/Programming/PhaseSmith/crates/phasesmith-spots/src/lib.rs` and
recorded in the committed `threshold-clippy-fix.patch`. The surrounding unfinished
optimization is not committed or replaced by this task. Every other captured
tracked/untracked file in that checkout is preserved byte for byte; its branch,
index and installed environment are left in place. The patch retains the fix
without silently committing unrelated unfinished work.

## Checks

The committed integration passes formatting, strict workspace/all-target/all-
feature Clippy, 485 Rust tests (34 ignored) and the complete Python suite:
3,158 passed, 11 skipped, 34 deselected. An additional eight historical-freeze
tests pass after the final shallow-history handling change.

The isolated copy including the unfinished work and the lint correction also
passes strict workspace Clippy, 20 spot-crate Rust tests and all 99 detector
Python tests. Those tests include masks/NaNs, both noise conventions, clipped
supports, scalar/vector width transitions and exact/adjacent threshold values.

`benchmarks/threshold_predicate_clippy.py` compares preserved native builds on
768-by-1024 images with 300 spots, with both estimated noise and supplied
variances. All exported record arrays, dtypes/shapes and rejected-component
counts match byte for byte. With one warmup and eight measured calls per path,
the median times are 9.050-to-8.756 ms and 9.385-to-9.275 ms respectively.
These local observations show no slowdown; they are not a claim of a general
speed improvement. Raw samples and native hashes are retained in
`benchmarks/threshold-predicate-clippy-macos.json`.

## Frozen benchmark provenance

The historical holdout test now verifies the 402 frozen source hashes against
Git blobs from the recorded protocol commit `bc4c639`, and ties the frozen
native hash to the original result. It does not require later merged code to
match an old executable. In a shallow checkout without that historical commit,
the historical source-proof portion is explicitly unavailable; no results or
expected hashes are regenerated. The real holdout execution guard is unchanged:
it continues to reject any live source/binary drift.

The original release extension and Python package from the protocol commit are
archived at `runs/solver-freeze-870d61a/site/`, with hashes and provenance in its
parent directory. They remain available independently of the integrated build.
No holdout predictions, grading thresholds or scientific results are changed
or rerun. The 65 cases remain exposed validation evidence.

The user-authorized merge targets `codex/hkl-structure-solution`. It does not
overwrite the active experimental checkout or publish any branch. Detailed
check logs and preservation snapshots are retained under
`/tmp/phasesmith-integration-merge-20261031/`.

The merge completed by fast-forward to `63bb354`. The destination's own editable
native extension was rebuilt, and its full Python suite again passed all 3,158
tests. Post-merge formatting and strict Clippy pass. Its Rust tree is identical
to the integration tree that passed all 485 Rust tests. Final file/hash, archived
binary and check evidence is in `structure-search-merge-validation.json`.
