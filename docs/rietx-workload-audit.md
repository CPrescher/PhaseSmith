# Why the remaining QARR timings differ

Audit date: 2026-09-16. PhaseSmith release build from commit `28a10e5`,
rietx 1.4.0, Apple M4 Pro, Python 3.12.13. This follows the
[second optimization pass](refinement-performance-round2.md).

Subsequent opt-in optimizations and fresh measurements are documented in
[Choosing profile accuracy](profile-accuracy.md). The audit below describes
the pre-change defaults at the recorded commit.

The main remaining single-worker difference is **different profile work**.
In this real QARR case, rietx's default numerical policy treats **every
reflection as symmetric**, while PhaseSmith evaluates FCJ axial convolution.
rietx also uses smaller peak windows. Its optimizer needs fewer expensive
Jacobian evaluations in the second stage. Comparing default end-to-end times
alone therefore gives an incomplete picture of what needs optimizing.

## Controlled interventions

`benchmarks/investigate_rietx_workload.py` runs the existing measured QARR 1g
case with isolated, temporary interventions. It does not modify installed
packages or production defaults. Each case is warmed once, then measured five
times in alternating forward/reverse order. All workers, including numerical
library workers, are set to one. Inputs, preparation, fit, covariance and
result assembly remain inside the timed workflows. Profiling and workload
instrumentation run separately. Every run passes the unchanged real-data
quality gates, and scientific results repeat exactly within each case.

| Workflow | Median | Poisson Rwp | Maximum phase-fraction error |
|---|---:|---:|---:|
| PhaseSmith, unchanged FCJ and 30-FWHM support | 0.834 s | 19.8165% | 0.618 percentage points |
| PhaseSmith, explicitly symmetric; still 30 FWHM | 0.456 s | 19.8437% | 0.609 percentage points |
| rietx defaults | 0.411 s | 19.2601% | 0.913 percentage points |
| rietx, retain small axial corrections | 0.601 s | 19.2569% | 0.913 percentage points |
| rietx, 30-FWHM window multiplier | 0.470 s | 19.3151% | 0.863 percentage points |
| rietx, both interventions | 0.911 s | 19.3119% | 0.863 percentage points |

The PhaseSmith symmetric experiment removes axial geometry from each fit
request; initial scale preparation stays unchanged. Its initial crystalline
profile differs from full FCJ by relative L2 `0.000216618` (0.02166%). Its
fitted Rwp changes by about 0.0273 percentage points. This demonstrates a
useful approximation on this dataset, not a universal error bound.

The rietx 30-FWHM intervention retains its frozen windows, 0.3-degree movement
slack and FCJ-extent padding. PhaseSmith instead has per-node closed support
and updates support with parameters. Retaining small axial corrections also
retains rietx's quadrature rule, including zero-weight nodes in its equal-height
split. These interventions **do not establish identical models**. Their timing
effects include solver trajectory changes; individual savings cannot be added.

## Source audit and workload counts

The audited files are the installed rietx 1.4.0 package's
`model/profiles/fcj.py`, `model/forward.py`, `model/_kernels_numba.py`,
`model/compiled.py`, and `optimize/least_squares.py`.

1. **Small axial terms are skipped.** `fcj_node_count` returns zero when
   asymmetric extent is below `0.02 * FWHM`. At each actual optimizer entry,
   all 222 reflection/line rows have zero FCJ nodes in all three stages.
   Disabling only that threshold gives eight nodes for every row. Thus the
   previous description “both use FCJ” described configured geometry but
   missed an important runtime approximation. PhaseSmith has no small-extent
   symmetric shortcut: it uses eight nodes for small nonzero spans and 48
   for larger spans, per nonempty overlap interval.
2. **Peak tails use a different policy.** rietx's `WINDOW_AREA_TOL = 0.02`
   selects a mixing-dependent half-width, with movement and axial slack.
   Stage-one default windows visit 63,660 samples per profile pass; the
   30-FWHM intervention visits 235,226, about 3.69 times as many. With the
   small-axial shortcut disabled, those become 509,280 and 1,881,808
   node/sample visits. These are per-pass counts, not whole-fit timings.
3. **The solver schedules different work.** Default rietx's SciPy trust-region
   reflective solves report `(nfev, njev)` of `(13, 13)`, `(30, 14)`, `(3, 3)`.
   PhaseSmith records 31, 29 and 2 model evaluations with 7, 28 and 1 accepted
   steps. These counters have different meanings. PhaseSmith builds selected
   dense derivatives on first trials; rietx requests fewer Jacobians in stage
   two. Also, rietx maps `max_iter` to `max_nfev = 4 * max_iter`, so equal
   nominal budgets are not equal stopping contracts.
4. **rietx is compiled too.** Its hot profile/scatter loops use Numba with
   `fastmath=False`, fusion, cached compilation and shared worker pools.
   Python versus Rust is not a useful explanation of these measurements.

The 222 rietx rows versus PhaseSmith's 220 component/reflection rows also
disclose different reflection-domain preparation. Even with both interventions,
the initial relative L2 difference is `0.008494`, above the existing `0.001`
equivalence gate. The remaining mismatch is not resolved by these two policies.
No identical-calculation speed claim is justified.

## Next implementation priorities

1. **Reduce small-span FCJ cost with a validated numerical policy.** Investigate
   an independently derived lower-order or moment-based approximation that
   retains the axial shift and fused derivatives. State its error contract,
   preserve a reference mode, and test transitions, finite support, derivatives
   and multiple real datasets. The symmetric experiment shows headroom but is
   not the proposed implementation.
2. **Improve bounded steps and rejected trials.** Stage one still makes 31
   evaluations for seven accepted steps. The backtracking branch in
   `rietveld_general_solver.rs` calls the full calculator, including fixed
   axial diagnostic derivatives, rather than the selected cached objective.
   Final public diagnostics need those rows; rejected trials do not. A private
   selected calculation path can avoid that work while preserving fused values
   and required derivatives. Separately compare trust-region or bound-aware
   steps against current backtracking on the same physical objective.
3. **Evaluate an explicit area-error support option.** A caller-visible support
   policy could reduce tail work. Changing 30 FWHM silently would change
   integrated intensity. Validate QPA bias, moments, derivatives, support
   boundaries and reproducibility before adoption.

A separate eight-second native sample confirms FCJ evaluation and fused
profile accumulation, including exponential evaluation, dominate single-worker
execution. Small-system solves and allocation are secondary costs. This
supports optimizing profile work before more broad cache or Python tuning.

## Evidence and reproduction

- `validation/results/rietx-20260916-workload-ablation.json`: all timings,
  scientific gates, forward differences and instrumented rietx workloads.
- `validation/results/rietx-20260916-workload-phasesmith-stages.json`: current
  stage histories from a separate profiling run; times are diagnostic.
- `validation/results/rietx-20260916-workload-native-sample.txt`: native sample
  leaf-count summary; attribution evidence, not benchmark timing.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/investigate_rietx_workload.py \
  --repetitions 5 --json-output workload-ablation.json
```

Use an isolated environment with a release PhaseSmith wheel and `rietx==1.4.0`.
The script verifies dataset checksums, requires compiled rietx kernels, and
restores every policy patch. No runtime dependency, numerical default,
scientific threshold or oracle fixture changes.
