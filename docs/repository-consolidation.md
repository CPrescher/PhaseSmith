# Repository consolidation, 2026-09-21

## Rietveld accepted; this Pawley integration remains parked

The first consolidation preserves the existing six development commits, adds
version-matched skill distribution, and repairs the newly exposed canonical
QARR 1g termination failure without changing scientific gates or budgets.
The feasible-width API and implementation remain on the preserved experimental
branch. Pawley and the shared FCJ correction have a separate combined branch;
its macOS Intel convergence assertion and installed validation CLI paths remain
unresolved. See the [Pawley review](pawley-consolidation-review.md) for the
completed measurements and required fixes.

## Preservation and history

A verified complete Git bundle and SHA-256-verified copies of all 72 original
working files are retained outside the repository at
`/Users/clemens/PhaseSmith-preservation/20260921-consolidation/`.
Both original checkouts were converted to clean preservation branches only
after verifying their complete index trees matched the remote snapshots;
original file contents were unchanged by that operation.

Remote preservation refs:

- `codex/consolidation-snapshot-20260921` (`ada3541`): original develop worktree.
- `codex/develop-baseline-20260921` (`8d9d86e`): original committed baseline.
- `codex/feasible-width-experiment-20260917` (`ada3541`): intact experiment.
- `codex/pawley-implementation-plan` (`b9c6105`): original Pawley history.
- `codex/pawley-evidence-snapshot-20260921` (`a2933f0`): five additional variants.
- `codex/pre-gui-history-cleanup-20260813` (`3e871be`): historical local commits.

The duplicate-looking Pawley manifests represent different generation times
and source hashes. They were preserved as historical variants, not substituted
for canonical golden fixtures. The conda recipe and notes were deliberately
moved to a separate staged-recipes checkout; they target the published 0.5.0
sdist and are outside this numerical integration.

`main` contains a separate API cleanup and download-maintenance history.
This consolidation advances develop without rewriting or overwriting main.
Reconciliation with that API cleanup remains a distinct review before a
future merge to main. No release tag or package publication is part of this work.

## Numerical outcome and performance tradeoff

The [bounded recovery contract](refinement-bounded-recovery.md) keeps undamped
stopping verification and shares one existing line-search allowance across
coordinate proposals. Exhausting that search means `stagnated`; an actual
runtime guard still reports its own reason. Neither category is convergence.
The canonical QARR 1g workflow's original acceptance rule already permits
stagnation, and both native and explicit-execution entry points now run in
normal pull-request CI on Python 3.13.

The fixed protocol runs one warmup and two measured repetitions for each build
and case, sequentially after local builds/tests. All scientific records repeat
exactly. Candidate statuses match the committed baseline: seven passing cases
and the existing bounded QARR 1h failure. No gate was loosened or rerun with a
larger fit budget. Timings are whole-workflow measurements on a shared desktop.

| Case | Baseline → repaired gates | Median seconds, baseline → repaired |
| --- | --- | ---: |
| iucr-qarr-1g | passed → passed | 0.629 → 0.811 |
| iucr-qarr-1h | failed → failed | 0.617 → 0.618 |
| gsasii-pbso4-cw-x-ray | passed → passed | 4.045 → 4.403 |
| gsasii-pbso4-cw-neutron | passed → passed | 0.575 → 0.579 |
| rowles-1a | passed → passed | 3.119 → 3.672 |
| rowles-1e | passed → passed | 2.365 → 3.326 |
| ansto-echidna-lab6-cw-neutron | passed → passed | 0.008 → 0.008 |
| lanl-nickel-tof | passed → passed | 52.110 → 52.232 |

Truthful stopping has a measurable cost: QARR 1g increases by about 0.18 s,
Rowles 1a by about 0.55 s, and Rowles 1e by about 0.96 s (roughly 41%).
The transfer fit quality changes negligibly. This is an explicit correctness
tradeoff, not a speed improvement: the old solver could certify convergence
from damping alone. Recovery now has a fixed trial allowance independent of
parameter count and stays within existing overall runtime guards. Further
optimization requires separate evidence; no extra QARR-specific tuning was
introduced to hide this cost.

Raw evidence is retained in
`validation/results/consolidation-20260921-rietveld.json` and
`validation/results/consolidation-20260921-before.json`. The latter reproduces
the old candidate failure using its exact historical binary. Its preliminary
timings overlapped build/test work and are not the performance comparison.
The historical baseline binary was unavailable, so the baseline was rebuilt
from `8d9d86e`; hashes and complete installed Python sources are recorded.

## Completed checks for the Rietveld/packaging slice

- 345 Rust tests/doctests pass; opt-in Rust/Python differential tests pass.
- 890 Python tests pass, with 11 unavailable-data skips and 34 explicit marker
  deselections; both QARR entry-point tests additionally pass with pinned data.
- A fresh sdist install outside the checkout passes the same 890 tests,
  including skill resources, CLI discovery and the offline advisor cycle.
- Formatting, strict Clippy, strict documentation, mathematics/skill generation,
  public API and release metadata checks pass.
- [Remote CI](https://github.com/CPrescher/PhaseSmith/actions/runs/35577853523)
  and the [untagged distribution dry run](https://github.com/CPrescher/PhaseSmith/actions/runs/35578206362)
  pass. The latter covers Linux x86-64/AArch64, macOS Intel/ARM64, Windows,
  and sdist construction; the configured test jobs pass. No publishing ran.

Preserved historical snapshots intentionally retain their original defects.
A preservation branch is not a release-readiness claim.
