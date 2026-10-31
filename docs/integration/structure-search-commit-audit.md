# Structure-search commit and integration audit

## What is committed

All implementation, tests, method documentation, canonical benchmark reports,
the frozen holdout protocol and its results are committed on
`codex/hkl-structure-solution`. There are 49 committed changesets between the
common experimental fork `f166525` and the complete holdout result `bb6eab8`.
The numerical freeze remains `870d61a`, protocol freeze `bc4c639`, and holdout
results `bb6eab8`. This audit adds only review artifacts, without changing those
numerical sources or the benchmarked binary. All 402 frozen source hashes and
the native binary still match the freeze record.

Large raw observations, generated `runs/`, virtual environments and build outputs
remain intentionally ignored. Canonical result reports, input checksums and
reconstruction protocols are committed. This audit concerns local commits;
it does not imply that any branch has been published.

## Integration with the other development work

The other active checkout is `/Users/clemens/Programming/PhaseSmith`, on
`codex/experimental-spots`. At capture its committed head was `910bc22` and it
had additional staged/unstaged/untracked development work. The branch and
working files were not changed by this audit.

A non-mutating merge preview found two conflicts between committed tips:

- `PROJECT_BRIEF.md`: both branches prepend independent status descriptions.
  Preserve both descriptions in sequence.
- `tests/test_lebail.py`: both add the same `scipy.optimize` absence simulation,
  with different explanatory comments. Keep the identical test code once.

These are resolved in local integration-review commit `a215843` on
`codex/hkl-integration-review`, based on the complete structure-search work and
the other branch's committed intensity-reduction work. A subsequent formatting
commit `c4ee76d` sorts the combined Python extension module declarations. This
separate review branch leaves both original branches unchanged. It contains no
committed snapshot of the other checkout's unfinished work.

Six paths overlap with the other checkout's unfinished changes:
`Cargo.lock`, `Cargo.toml`, `PROJECT_BRIEF.md`, `THIRD_PARTY_NOTICES.md`,
`crates/phasesmith-py/Cargo.toml` and `crates/phasesmith-py/src/lib.rs`.
An isolated detached test copy combined the committed merge with a captured
snapshot of the unfinished work. Combining that snapshot needed three additional
text resolutions: retain both native dependencies in the lockfile, retain both
optional feature sets, and retain both status-description blocks. The other
three paths apply cleanly. All solution, row-search and spot/intensity features
remain enabled independently, and all attribution/dependency notices survive.

`structure-search-shared-files.patch` records the resulting six-file differences
against the captured other checkout. Its `git apply --check` passes against the
unchanged shared-file contents. It is a review aid for integrating the complete
branches; applying these manifest additions without the complete solver sources
would be incomplete. It has not been applied to the active checkout.

## Checks and limits

On the combined isolated snapshot:

- `cargo check --workspace --all-targets --all-features --locked` passes.
- `cargo test --workspace --all-features --locked` passes 485 tests, with
  34 ignored. This precedes a module-order-only formatting adjustment; no
  numerical expressions are changed by that adjustment.
- `cargo fmt --check` passes after that adjustment.
- A fresh release wheel with both experimental features builds and imports the
  solution, rows and intensity-reduction APIs together. It is extracted into a
  temporary package directory, leaving installed environments unchanged.
- Focused Python checks pass 189 tests: Le Bail optional-dependency behavior,
  exact repair-cache behavior, intensity reduction, row search and spot detection.
- Strict Clippy reports three `needless_bitwise_bool` diagnostics for one existing
  threshold expression in the captured `crates/phasesmith-spots/src/lib.rs:296`.
  That file is identical to the other checkout's captured version. This is an
  outstanding lint gate in that work, not a new structure-search merge conflict;
  its performance-oriented expression is left untouched.

No numerical benchmark is rerun or regraded. The integration runtime is a
separate build and is not substituted for the frozen benchmark binary.

The other checkout remains active. Two captured files,
`python/phasesmith/experimental/row_consolidation.py` and
`python/phasesmith/experimental/row_grains.py`, changed independently while the
audit ran. They do not overlap the structure-search changes. Test claims refer
to the captured snapshot, not to arbitrary subsequent edits; the six shared-file
preconditions were rechecked separately. Final integration should use committed
versions of that unfinished work and rerun the relevant checks.

The old `/Users/clemens/Documents/PhaseSmith` directory reports an invalid Git
HEAD. It was not repaired or used as the source of commits. The valid structure
and integration checkouts are under `/Users/clemens/Programming/`.

Hashes, exact parents, tested snapshot provenance, preserved source files and
check logs are summarized in `structure-search-commit-audit.json`. Raw audit
snapshots and logs remain in `/tmp/phasesmith-integration-audit-20261031/`.
