# Repository consolidation and package reliability plan

Prepared 2026-09-21. The original execution sequence is retained below.
Completed work, validation and remaining boundaries are recorded in the
[consolidation report](repository-consolidation.md).

## Objective and current evidence

Preserve all local work, make the development history reviewable and available
on the remote, and retain the package's useful numerical capabilities without
promoting known regressions. Repository cleanup must not change scientific
acceptance thresholds, silently regenerate fixtures, or discard experiments.

The inspected development head is `8d9d86e`, six commits beyond
`origin/develop` at `9b18acb`. The main checkout contains 28 modified tracked
files and 38 untracked files before adding this plan. The Pawley worktree is
at `b9c6105`; it has 14 commits absent from develop, while develop has one
commit absent from Pawley. Counts are a snapshot and must be refreshed before
execution. Three conda-related files disappeared during the preceding review;
their disposition needs to be established, not inferred.

The retained [assessment](refinement-assessment.md) identifies a new canonical
QARR 1g termination-gate failure in the uncommitted stopping/recovery change.
Its final Rwp and composition are unchanged. QARR 1h is an existing bounded-
recipe failure. The committed develop baseline also has a documented false-
convergence weakness; it is a comparison point, not a certified release.

## 1. Preserve the exact starting state

- Refresh remote refs, branch/worktree status, and the inventory of tracked,
  untracked, and relevant ignored evidence. Identify active writers before
  taking a consistent snapshot; work in an isolated integration worktree.
- Create an external Git bundle for committed refs and a checksum manifest
  plus copies of working changes and new files from both worktrees. Include
  relevant local-only baseline wheels, binaries and fixture evidence needed
  for reproduction; do not indiscriminately archive build caches.
- Verify that the preserved commit refs and working-file contents can be
  recovered. A Git bundle alone does not preserve uncommitted files.
- Preserve the current dirty source state in an explicitly named
  `codex/` snapshot branch in the isolated workspace. Keep original worktrees
  intact. Review its file inventory before pushing the preservation branch.
- Push the existing Pawley and historical cleanup histories under explicit
  branch names after confirming the intended refs. Never force-push or create
  a release tag for preservation.
- Inspect the five Pawley files with ` 2` suffixes individually and trace the
  missing conda files. Record whether each is superseded, distinct evidence,
  or owned by ongoing work. Retain uncertain files in the snapshot.

Exit: every relevant local file and commit has a verified recovery location;
preservation branches are clearly distinguished from validated integration.

## 2. Establish reproducible comparison builds

- Build clean installed packages from the committed develop head and the
  preserved candidate, with separate environments and recorded source,
  dependency, binary and dataset identities. Retain the historical baseline
  artifact used by the assessment if available and verify its hash.
- Run ordinary Rust/Python checks and the fixed eight-case assessment on the
  comparison builds. Record failures honestly; confirm the canonical QARR 1g
  regression before changing its implementation.
- Exercise both the native no-execution-argument validation entry point and
  the Python-assembled explicit-execution route. Their recipes differ, so
  neither substitutes for the other.
- Keep measurements sequential with fixed workers and the protocol's warmup
  and repeat policy. Record Rwp, applicable parameter/composition accuracy,
  stop reasons, evaluations, reproducibility, and complete-workflow runtime.

Exit: a reproducible before/after comparison and explicit known-failure list.
An unavailable historical binary is a recorded limitation, not permission to
claim a byte-identical reconstruction of it.

## 3. Organize changes into independently reviewable commits

Retain the six existing develop commits without rewriting shared history.
Review and validate their combined baseline before advancing remote develop.
Extract subsequent changes into this sequence, resolving overlapping files
by change rather than copying entire files between groups:

1. Assessment protocol, driver, retained evidence, and consolidation decision.
2. Agent-skill packaging, CLI, documentation, synchronization and tests.
3. Correct stopping behavior with the bounded-workflow repair and focused
   regressions. Review coordinate recovery's extra cost separately.
4. Feasible-width research and evidence on a separate experimental branch.
   Preserve all implementation and tests there; keep production defaults off
   and preserve existing persisted-project compatibility.
5. Any recovered conda packaging changes as a separate packaging review.

Every integration commit must build with the files it introduces. In
particular, CI references to `sync_skill.py`, packaged resources and generated
documentation must land with those files. Update the brief, implementation
status, changelog and API snapshot to describe the actual integrated scope.

Exit: coherent commits with validation records, and no source changes or
research evidence lost during separation.

## 4. Repair stopping without sacrificing scientific honesty

- Reduce the canonical QARR 1g failure to a focused reproducer. Distinguish
  stopping verification from recovery scheduling, rejection accounting and
  coupled physical-domain handling before selecting the smallest fix.
- Retain rejection of false convergence from excessive damping or a tiny
  step cap. A finite endpoint, small damped step, or final scale-only polish
  must not certify convergence of preceding nonlinear stages.
- Preserve existing recipes, physical models, bounds, support, scientific
  tolerances and iteration/evaluation/rejection budgets. Do not repair the
  test by suppressing the failure or accepting every endpoint.
- Implement the justified behavior independently in Rust and readable Python.
  Verify native/fallback behavior, runtime guards, constraints, cancellation,
  deterministic histories and checkpoint continuation where affected.
- Run the unchanged assessment panel. QARR 1g must pass its existing gate
  with truthful termination. Keep QARR 1h's existing failure visible and
  prevent further degradation. All previously passing transfer gates must
  remain passing. Inspect Rowles evaluation counts and recovery cost.
- Compare realistic multi-peak and full-workflow benchmarks. Flag material
  regressions against observed repeat variability and resolve or explicitly
  justify their correctness tradeoff before integration.

Exit: the new QARR 1g regression is resolved without false convergence or
relaxed gates, and the broader panel supports promotion. If that cannot be
achieved, preserve the repair branch and report the blocker; do not declare
the package release-ready merely because repository cleanup is finished.

## 5. Integrate Pawley as a separate numerical change

- Keep its remote preservation branch available while the development
  baseline is consolidated. Review the 14 branch-only commits and their
  dependencies before reconciling the divergence in an integration worktree.
- Review the shared FCJ geometric-Jacobian correction independently of the
  Pawley API. It can affect existing Rietveld and profile consumers and must
  follow the complete numerical-change checklist in `AGENTS.md`.
- Check published/first-principles equations, independent reference parity,
  analytical derivatives versus finite differences, finite-support boundaries,
  normalization/moments, deterministic randomized cases and pinned-oracle
  discrepancies with explicit provenance. Do not regenerate golden fixtures
  as an incidental merge action.
- Run Pawley CW, fixed-spectrum, TOF/joint, matrix-free and persistence checks,
  then rerun the affected existing Rietveld and multi-peak benchmarks. Review
  public API, checkpoint and project round-trip compatibility.

Exit: independently supported Pawley and shared-kernel changes, or an intact
separate branch with specific unresolved gates. Stable consolidation need
not wait for unrelated unfinished feature work.

## 6. Validate the installed package and finish the remote handoff

For each numerical integration, complete the repository's required checks:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
python -m pytest -q
```

Also run applicable Rust/Python differential contracts, Ruff, public API and
version checks, strict Rust/documentation builds, and generated-doc/skill
checks. Ordinary pytest excludes `real_data` and `external_oracle`, so invoke
the relevant explicit real-data and pinned-oracle checks separately. A suite
that skips those cases cannot replace the assessment gate.

Build a release wheel and sdist, install outside the checkout in clean
environments, and verify native import/calculation/refinement, persistence,
CLI and packaged skill discovery. Exercise the offline advisor-cycle example
and the existing skill evaluation requirements. Validate the sdist by building
and testing its installed package, including resource availability.

Push the reviewed integration commits, inspect remote CI, and use an untagged
release-workflow dry run to validate distribution artifacts across supported
platforms. Resolve failures before declaring consolidation complete. Keep
publication/version tagging as a separate release action.

Final handoff records the remote commit IDs, included capabilities, retained
experimental branches, test/assessment results, performance changes, and
remaining scientific limitations. The integration checkout must be clean;
original worktrees must have an explicit disposition for every remaining file.
No unique branch, duplicate-looking file, or historical result is deleted
merely to obtain a clean status.
