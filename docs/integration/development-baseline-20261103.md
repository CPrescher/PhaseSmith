# Complete development baseline, 3 November 2026

## Branch to continue from

`develop` is the combined 0.8 development candidate. The normal checkout is
`/Users/clemens/Programming/PhaseSmith`. `main` retains the 0.7 release baseline;
updating develop does not tag or publish a release.

The implementation baseline is reconciliation merge `eebeb17`, whose parents are
the consolidated single-crystal implementation `521aaa2` and the byte-preserved
unfinished working state `c0ac701`. This documentation follow-up changes no
numerical source, API, defaults or validation thresholds.

Before the update, GitHub develop was at `2e5edb7` and main at `447a869`.
Both are ancestors of the reconciliation: develop has zero unique commits and
is 197 commits behind it; main has zero unique commits and is 171 commits behind.
The update is a fast-forward and retains both histories.

For a new task, from the normal checkout:

```sh
git switch develop
git pull --ff-only
git switch -c codex/my-next-change
```

## Recent worktree accounting

Every local branch and all four Programming worktrees were checked. All were
clean, and every local branch tip is an ancestor of `eebeb17`:

| Worktree/branch | Tip included in the baseline | Status |
| --- | --- | --- |
| Original experimental spot/intensity/FFT/row search | `297d157` | Included through the single-crystal integration |
| HKL integration review | `63bb354` | Included |
| HKL structure solution | `54784c5` | Included |
| Single-crystal promotion and native workflow | `521aaa2` | Included, including subsequent search recovery and Dioptas adapter work |
| Unfinished working-state preservation | `c0ac701` | Included through the reviewed reconciliation |
| Memory-budget and promotion-audit branches | `4386027`, `c86b4bf` | Included |

The named old worktrees are historical checkouts of this same Git repository.
They contain no additional committed or uncommitted source needed to continue
the recent development. Their ignored research outputs, original binaries and
external validation inputs still require preservation before folder removal.

## What was superseded

| Earlier work | Current disposition |
| --- | --- |
| `phasesmith.experimental` namespace and experimental feature packaging | Superseded by the reviewed `phasesmith.single_crystal` API and ordinary native capabilities. Keep the current implementation; research helper imports were migrated. |
| Uncommitted `prediction.py` formatting | Already present in the later implementation. No duplicate change was needed. |
| Earlier Python production geometry/profile/scaling orchestration | Later commits move the production numerical fitting into Rust and add caching/ordered parallel work. Retain the later implementation and independent Python validation paths; do not restore older production code from research snapshots. |
| Unfinished row-staged detector threshold loop | Still useful and not supplied by the later frame-I/O/profile optimizations. Reconciled with current code, boundary tests and fresh exact-record/performance checks. |
| October Fe7C3/DIALS scripts, figures and timing receipts | Historical research evidence. They are committed for provenance and reproduction, with current status pointers; their old timings and failed intensity gates do not replace newer measurements or imply current release approval. |
| Single-frame multigrain proposal | Retained as a deferred plan, with no claim of implementation. |

Thus the older branch tips do not need new duplicate commits or another merge.
The useful unfinished code and research were already committed in `eebeb17`.
The [reconciliation report](unfinished-work-reconciliation.md) and
[per-file inventory](unfinished-work-inventory.json) account for all 51 original
modified/new files and preserve their original hashes.

## Intentionally separate historical refs

Some remote preservation refs have commits outside this history. They are not
additional recent single-crystal work:

- `codex/consolidation-snapshot-20260921` and
  `codex/feasible-width-experiment-20260917` share the original September snapshot.
  Its opt-in feasible-width experiment was deliberately excluded from the
  supported baseline; the reviewed integration already carries the accepted
  portions. See the [original consolidation decision](../repository-consolidation.md).
- `codex/pawley-evidence-snapshot-20260921` retains five extra historical fixture
  and diagnostic variants. These are provenance snapshots, not replacement golden
  data for the current Pawley implementation.
- `codex/fcj-ci-diagnosis` retains a September CPU-math diagnostic workflow/script.
  It changes no production implementation.
- `codex/pre-gui-history-cleanup-20260813` retains August GUI documentation commits
  and their reversions; GUI applications remain separate consumers.

These refs remain preserved remotely. They should be revisited deliberately
when their experiment or provenance is needed, rather than merged indiscriminately
into the continuation baseline.

## Validation and limits

The combined implementation passes 3,380 Python tests and 567 Rust tests/doctests,
with documented external-input skips/marker exclusions and ignored Rust tests.
Strict Clippy, Ruff, formatting, API/version/documentation/skill checks and strict
MkDocs pass. After installation in the normal checkout, another 125 detector,
public API, workflow and Dioptas integration checks pass. Fresh detector and
rotation comparisons retain complete scientific records exactly. The
[reconciliation report](unfinished-work-reconciliation.md) records benchmark
scope, hashes and local timings.

This branch is a practical continuation point, not a new scientific guarantee.
Existing Fe7C3 intensity limitations, unknown-group recovery failures and exposed
development-data qualifications remain documented. Raw data and ignored run
artifacts stay external. Remote CI is tracked separately from the local results.
