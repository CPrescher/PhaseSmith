# Checked normal caches for isotropic assignment refinement

This experiment starts from `25ef695`, the guarded ambiguity search recovering
19 of 35 development structures. Its measured extra work costs 6.073 seconds in
isotropic screening and 3.673 seconds in final refinement, relative to the prior
16-structure path. The present comparison keeps ambiguity handling enabled in
both modes. It changes execution of the short isotropic fits rather than removing
their competing assignments or relaxing correctness gates.

## Diagnosis

Two sequential five-second native profiles repeat complete solves for COD 2011393
and COD 1549173 with all current fast options and guarded ambiguity enabled.
The library bytes are hashed immediately before sampling. The first profile has
4,166 main-thread samples: constrained structure-factor evaluation has 1,031
collapsed self samples, while the original matrix-free normal solver and its
product construction have 557. Final-fit cached solving remains substantial too.
The second profile has 4,168 main-thread samples and a larger phasing fraction,
showing that this is a representative assignment hotspot, not a universal corpus
time fraction. Geometry preparation is much smaller than repeated evaluation
and linear solving in the first case.

Raw profiles, stack hashes, sampled image UUIDs, all printed collapsed counts,
driver scripts and actual native hashes are recorded in
`benchmarks/structure-solution-assignment-cache-profiles-macos.json` and
`runs/post-ambiguity-profile/`. Native final fits already use a checked cached
normal operator; short isotropic screens explicitly disabled that path.

## Execution change and equations

The experimental `cached_assignment_refinement` flag enables that same checked
normal cache for short isotropic screening passes. It is false by default and
requires `iterative_refinement` and `isotropic_assignment_refinement`. Existing
`cached_normal_refinement` remains the independent control for requested-mode
final fits. Both settings remain explicitly reported.

Let J contain parameter-major derivatives of normalized amplitude residual r,
D_ii=max(sum_h J_ih^2,1e-8), S_ii=1/sqrt(D_ii), and lambda be the original damping.
The damped Gauss-Newton equations are

    (J J^T + lambda D) step = -J r
    H = S J J^T S
    (H + lambda I) y = -S J r, step = S y.

For at least 32 parameters, the original solver repeatedly applies J and J^T
to vectors during CG. The enabled path constructs the bounded symmetric H once
per nonlinear linearization and reuses it during those CG iterations. It uses
the existing eight-lane f64 dot reduction without fast-math or reduced precision.
The current step is checked against the original Jacobian operator at the
unchanged scaled squared-residual bound 1e-16 times the squared RHS norm. Failed
or ineligible cache attempts fall back to the original matrix-free solver.

For 12–31 parameters, the existing blocked dense Gram/Cholesky path has the same
original-operator check and scalar dense fallback. Below 12 parameters, the
original small dense solve remains. Cache eligibility, the 8 MiB normal-matrix
bound, damping, CG limits and convergence thresholds are unchanged. Each new
nonlinear iterate recomputes its matrix; no stale Jacobian or matrix is reused
across changing coordinates, displacements or assignments.

All value/analytical derivative evaluations, input observations, family weights,
candidate lists/order, count proposals, six-extra-fit ceiling, eight-iteration
relaxations, conservative assignment guard, final anisotropic mode, requested
budgets and resource/correctness gates remain unchanged. No new dependencies,
threads or physical terms are introduced. Reordered floating-point reductions
can change nonlinear trajectories, so complete structure/count recovery must be
measured rather than inferred from the linear residual check.

## Diagnostics and validation

`assignment_cached_refinements` counts short fits that actually use at least one
cached step. `assignment_cached_normal_solves` counts accepted cached steps;
`assignment_normal_cache_fallbacks` counts attempted-cache failures returning to
the original solver. Existing final-fit counters retain their prior meaning and
do not absorb the new screening counts. A requested cache can record no use if
the gradient target is reached before a linear solve.

New short-fit comparisons cover dense and iterative isotropic cases in P1,
P-1 and Pm, including deterministic repeats, independent NumPy values and centered
finite differences. The independent normal-equation references, ill-conditioned
linear systems, original-system residual bounds and fallbacks remain covered by
the existing cache suite. Complete synthetic assignment searches require the
same inferred counts and final anisotropic mode. Existing tolerances are not
loosened. The original independent Python refinement equations remain applicable;
this activates an already independently verified execution path, not a new model.
Existing finite-support, moment and normalization conventions remain unchanged.
The GSAS-II powder oracle does not cover this single-crystal linear solver.

## Benchmark protocol

The matched comparison uses all 35 development cases with guarded ambiguity,
multiplicity-weighted final refinement and all previously enabled fast controls
in both modes. Only `cached_assignment_refinement` differs. Seed zero, four
starts, 800 requested cycles, native 13-second/outer 15-second budgets, final
anisotropic fitting and original input-only caps/gates remain fixed. Seven
above-cap references stay in every denominator. One numerical thread is used.
All 140 predictions precede both reference audit barriers, with leading mode
alternated by case. Adapter-only seal revision 25 preserves the registry and
35/65 split. No task-owned profiling, builds, tests or kernel benchmarks overlap
timed corpus collection. The 65 holdout outcomes remain unopened.

Criterion cases selected by `assignment_cache_` include preparation plus up to eight
isotropic iterations on eight/twelve-atom 1,054-row examples, and a complete
eight-atom Fe/C/N/O 32-cubed search on 2,108 expanded rows. These include realistic
multiple atoms/reflections rather than a scalar solve statistic.

```sh
.venv/bin/python benchmarks/solution_assignment_cache_interleaved.py --output runs/assignment-cache-new --seconds 15
```

Normalize completed panels with `benchmarks/solution_corpus_reaudit.py`, then
compare them with `benchmarks/solution_phase_ablation.py`.

## Validation and kernel measurements

The final feature-enabled suite passes 2,209 Python tests (11 external-data skips,
34 marker deselections) and 414 Rust tests (34 ignored), with strict workspace
all-target/all-feature Clippy, formatting and changed-file Ruff. Nine new focused
tests cover dense/iterative short fits, independent values, centered finite
differences, deterministic repeated fits and full synthetic model/count recovery.
The ablation report's feature-mismatch guard also covers the new option. No
existing numerical or correctness tolerance is loosened.

Same-executable Criterion sample medians are:

| Work | Original short solves | Checked assignment cache | Time reduction |
| --- | ---: | ---: | ---: |
| Prepare/eight-iteration ceiling, 8 atoms, 1,054 rows | 2.378 ms | 1.777 ms | 25.3% |
| Prepare/eight-iteration ceiling, 12 atoms, 1,054 rows | 4.221 ms | 2.904 ms | 31.2% |
| Complete 8-atom Fe/C/N/O 32-cubed search, 2,108 expanded rows | 216.586 ms | 215.853 ms | 0.3% |

The small whole-search example shows little change despite the isolated fit
gains. Both modes execute two starts and 192 phase cycles and retain the same
requested search work; actual fit stopping remains unchanged. These are separate
from the corpus timing, not a broad speedup claim. Raw samples, work counters,
source/executable hashes and the exact command are preserved in
`benchmarks/structure-solution-assignment-cache-kernels-macos.json`.

## Completed 35-case comparison

| Measure, all development cases | Original assignment solves | Checked assignment cache |
| --- | ---: | ---: |
| Correct labelled structures | 19/35 | 19/35 |
| Correct total non-H atom counts | 26/35 | 26/35 |
| Correct per-element non-H counts | 19/35 | 19/35 |
| Strict 0.02-angstrom structures | 15/35 | 15/35 |
| Raw median wall time, all outcomes | 1.815 s | 1.710 s |
| Summed solve wall time | 70.908 s | 69.393 s |
| Summed isotropic-screening time | 14.087 s | 12.646 s |
| Summed final-refinement time | 11.604 s | 11.570 s |
| Summed phase-search time | 31.052 s | 30.932 s |
| Starts / screening fits / final fits | 95 / 675 / 125 | 95 / 675 / 125 |
| Extra ambiguity fits / cheap ambiguity scores | 249 / 10,305 | 249 / 10,305 |

Screening time falls 10.23% (1.442 seconds); summed wall time falls 2.14%
(1.515 seconds). The median paired control/new wall ratio is only 1.00712x,
or 0.71% less time, with 23 cases faster and 12 slower. The median paired
difference is -0.006 seconds, and the native-time ratio is 1.01980x. Across the
19 common successes the median paired wall ratio is 1.00604x, about 0.6% less
time. Thus the added stage is cheaper, but a substantial typical-latency gain
is not established. Raw median ratios are different statistics from the typical
matched-case ratio. These are single-seed, single balanced collections on an
uncontrolled host, not repeated confidence intervals.

Every previously recovered development structure is retained, including the
three recent assignment gains and the protected COD 2011241 case. Structure,
total-count, per-element-count and strict recovery case sets are unchanged, as
are all returned inferred count vectors. Four no-model outcomes remain unchanged.
Thirteen selected outcomes compare equal in the recorded model fields, including
the four no-model cases;
the largest selected R1 difference is 2.29e-12. Complete recovery is assessed by
the original reference gates, not a bitwise-equality claim. There are no hard
PhaseSmith timeouts.

The new path uses cached steps in 615 of the 675 short fits, with 4,640 accepted
cached solves and zero screening-cache fallbacks. Final-fit cache counters remain
separate: 2,450/2,460 accepted steps and zero fallbacks. Remaining short fits use
the original small solve or stop before a linear step. Phase-scored cycles are
49,625/49,495 because the same wall backstops remain observable. Auxiliary
unresolved-map counts are three/four, while both modes request zero corroborating
starts. The extra six-fit search ceiling, candidate scores/count and conservative
dual-metric policy remain the same.

This is a modest reduction in the guarded search's additional cost. It does not
establish that the earlier cost of increasing recovery from 16 to 19 structures
has been eliminated: that would require a contemporaneous three-way comparison,
not mixing absolute timings from earlier runs. The profiles and remaining stage
costs show why: structure-factor/derivative evaluation and final polishing still
remain after the linear solve is accelerated.

Both panels use the same native binary and 43 matching runtime-source snapshots.
All 140 predictions precede the two audit barriers, with 18 control-leading and
17 cache-leading cases. The seven above-cap references remain in all denominators.
External SHELXT controls retain 13/35 structures and 29/35 total counts in both
panels; current PhaseSmith coverage still assumes the supplied space group.

Reports, paired comparison, current kernel samples and verified schedule are
`benchmarks/structure-solution-assignment-cache-{control,cache,comparison,kernels,schedule}-macos.json`
(each name is a separate file). Raw workers, normalized audits and sealed sources
remain in `runs/assignment-cache/`. Adapter-only revision 25 keeps the original
registry and 35/65 split. The option stays false by default. All 65 holdout
outcomes remain unopened.
