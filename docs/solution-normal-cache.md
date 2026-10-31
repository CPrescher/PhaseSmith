# Guarded normal-matrix caching in final refinement

## Diagnosis and scope

The final-refinement follow-up starts at `210de96`, with the latest batched FFT,
filtered peaks, work-sliced phasing and isotropic competing-assignment settings.
A native sample of repeated complete `cod-2021292` development solves finds
1,876 matrix-free CG samples inside 3,450 constrained-refiner samples (54%).
The fused symmetry evaluator has 2,324 samples across final and screening calls,
out of 6,690 main-thread samples. These are inclusive stack counts from one
representative case, not universal time fractions or a speed comparison.
`benchmarks/structure-solution-normal-cache-profile-macos.json` retains provenance,
script and raw-profile hash; the raw stack is in `runs/final-refinement-profile/`.
Friedel-pair reuse was considered, but this implementation targets the larger
linear-solve cost inside final refinement.

`cached_normal_refinement=True` is experimental and off by default. It requires
`iterative_refinement=True` and affects requested-mode final fits only. Isotropic
screening fits, competing-assignment policy, all original observations, analytic
values/derivatives, physical ADP bounds, site symmetry, damping, step clipping,
backtracking and acceptance/termination tests retain their existing definitions.

## Equations, memory and validation guard

Let J be the residual Jacobian in conventional reflection-by-parameter notation,
r the normalized amplitude residual, and

    D_j = max(sum_h J_hj^2, 1e-8), S_j = D_j^(-1/2).

The original damped system is

    (J^T J + lambda D) delta = -J^T r.

Writing delta=S x gives

    (S J^T J S + lambda I) x = b, b=-S J^T r.

The original CG operator streams the parameter-major Jacobian for J(S p) and
S J^T(J(S p)) on every CG iteration. The new path forms one triangle of the
preconditioned normal matrix once for the current Jacobian, mirrors it and uses
contiguous row-major matrix/vector products inside CG. A fixed eight-lane f64
reduction permits vectorization; tail elements retain explicit ordered handling.
No fast-math, reduced precision, external BLAS or numerical thread is introduced.
The matrix is rebuilt when the nonlinear iteration supplies a new Jacobian.

For CG, eligibility is numerical/resource based: at least 32 parameters, at least twice
as many reflection rows as parameters, and at most 8 MiB for the matrix. Other
shapes use the original solver. Original dense-Jacobian and input-derived atom,
reflection and parameter limits remain in force. This adds a bounded temporary
workspace, without expanding scientific/resource admissibility. It neither
compresses observations nor changes residual/Jacobian arrays or their ordering.

For 12–31 parameters, final fitting retains the original Cholesky solver. The
same fixed-lane reductions form its unscaled normal matrix; damping still uses
the original ordered column norms. The step is checked in the scaled original
Jacobian system with tau=1e-16. A failed check reconstructs the original ordered
normal matrix and repeats the original Cholesky solve. These matrices need less
than 8 KiB. Models with fewer than 12 parameters retain the original path, avoiding
normal-formation and validation overhead where there is little work to save.
The shared counters include returned checked steps from both final solver paths.

Normal formation changes floating-point reduction order; bitwise agreement with
the original CG trajectory is not claimed. Both paths remain deterministic for
fixed work. Before returning a cached result, the code reapplies the original
Jacobian operator and checks

    ||b - [S J^T J S + lambda I] x||^2 <= tau ||b||^2.

The original tolerances are retained: tau=1e-16 at normal convergence, or the
existing explicitly checked inexact bound 1e-10 at the unchanged final iteration
ceiling min(2*n,256). A failed check, nonpositive curvature or other cache failure
falls back to the original solver for that shape, with the original deadline and
cancellation token. No fit is accepted from a cached objective: the original
fused evaluator and line search decide acceptance. Successful cache-produced
linear solves and successful fallback solves are counted separately in the
shared-search audit. Those counters describe returned linear steps, rather than
accepted nonlinear updates or every unsuccessful cache attempt.

The independent NumPy reference forms and solves the damped equations, and a
separate preconditioned reference exposes the original-system residual check.
Tests cover scaled columns, zero/tiny columns, rank deficiency, several damping
levels, non-multiple-of-eight lengths, repeated determinism, cache shape/memory
boundaries, explicit cancellation/deadlines, invalid inputs and complete
anisotropic fits. The physical amplitude/profile evaluator is unchanged; existing
independent value/Jacobian and centered finite-difference tests continue to apply.
No support cutoff or normalization changes. GSAS-II's powder oracle does not
cover this experimental single-crystal linear solve.

## Comparison protocol

The microbenchmarks compare a real eight-atom, 2,108-reflection, 69-parameter
anisotropic Jacobian, a complete three-step final fit and a small complete search.
Both modes share the same final executable; only normal caching differs.

The corpus experiment uses all 35 development cases with seed zero, four starts,
800 requested cycles, the existing native 13-second/outer 15-second caps and
final anisotropic fitting. Both modes enable all current fast options, including
batched FFTs. Original exact non-H counts and geometry gates remain unchanged;
seven cases above native capacity stay in every denominator. A balanced serial
scheduler alternates the leading mode by case. All 140 predictions precede both
reference barriers. Adapter-only seal revision 19 preserves the registry,
source data and 35/65 split. No task-owned tests, builds or microbenchmarks overlap
timed collection. The 65 holdout outcomes remain unopened.

## Preserved CG-only first experiment

The first source-matched Criterion collection, before the small-model path was
added, records these sample medians:

| Kernel | Control | Cached | Time change |
| --- | ---: | ---: | ---: |
| Original-Jacobian linear step, 69 parameters | 2.214 ms | 0.936 ms | -57.7% |
| Complete three-step final refinement | 10.425 ms | 5.909 ms | -43.3% |
| Complete small search | 81.585 ms | 82.535 ms | +1.2% |

The small complete-search diagnostic records one final fit but zero returned
cached steps in either mode. It provides no evidence of a whole-search speedup;
the measured small slowdown is retained. These are one-host fixed-work kernel
results, separate from deadline-limited corpus timings. Raw samples, confidence
intervals, source/executable hashes and actual small-search counters are saved
in `benchmarks/structure-solution-normal-cache-cg-only-kernels-macos.json`.

Workspace validation passes 407 Rust tests (34 ignored) and the full 2,012-test
Python collection (11 external-data skips, 34 deselections). A subsequent focused
59-test cached-normal run includes the public-control check added after full
collection began. Strict all-target/all-feature Clippy, Cargo formatting and
Python Ruff checks pass. Existing physical-model value, derivative, normalization
and finite-support tolerances are unchanged. Reordered normal-equation reductions
are checked through local linear-system residual and complete-fit tolerances.

To reproduce the collection, build the experimental extension explicitly:

```sh
VIRTUAL_ENV="$PWD/.venv" maturin develop --release --features solution,spots
.venv/bin/python benchmarks/solution_normal_interleaved.py --output runs/normal-cache --seconds 15
```

The first complete 35-case collection retains 16 labelled structures, 26 total
counts, 16 per-element counts and 13 strict structures, with no changed recovery
sets. Its paired wall ratio is only 1.02304x (25 faster, 10 slower). Summed final
refinement falls 29.216 to 26.376 seconds. This modest full-search gain motivates
extending blocked normal formation to the small-model Cholesky path. The initial
reports, comparison, schedule and complete source patch are retained under
`benchmarks/structure-solution-normal-cache-cg-only-*`, with raw data in
`runs/normal-cache/`. The patch applies against starting commit `210de96`.
The final extended implementation is measured separately, with a fresh matched
control rather than reusing timings from this first collection.

## Final extended implementation: kernels and validation

The extended implementation adds guarded blocked normal formation with the
original Cholesky solver for 12–31 parameters. A fresh same-executable collection
gives these medians:

| Kernel | Control | Extended | Time reduction |
| --- | ---: | ---: | ---: |
| Linear step, 69 parameters | 2.262 ms | 0.943 ms | 58.3% |
| Complete three-step fit, 69 parameters | 10.680 ms | 5.930 ms | 44.5% |
| Complete three-step fit, 24 parameters | 2.192 ms | 1.744 ms | 20.5% |
| Complete small search | 82.124 ms | 81.509 ms | 0.7% |

The complete small search still records zero returned cached steps and provides
no useful evidence of an algorithmic search gain. Corpus effects are measured
separately. Raw samples, confidence intervals, current source/executable hashes
and diagnostics are in
`benchmarks/structure-solution-normal-cache-kernels-macos.json`.

The final feature-enabled Python suite passes 2,016 tests (11 external-data skips,
34 deselections), and the Rust suite passes 408 tests (34 ignored). Strict Clippy,
formatting and Ruff pass. New small-model complete-fit comparisons use local
relative/absolute tolerances of 2e-8/2e-10 for objectives, R1, residuals, Jacobians,
coordinates and tensors, with deterministic repeated results. Larger CG-fit and
independent original-system residual tolerances remain unchanged. The original
ordered Cholesky calculation remains the small-model numerical fallback.

## Final paired 35-case results

Both fresh panels use the extended source/native build, and differ only in
`cached_normal_refinement`. All 140 predictions precede both reference barriers.
The saved schedule verifies 18 control-leading and 17 treatment-leading cases,
39 identical runtime source hashes per panel and the same native binary.

| Metric | Fresh control | Extended normal path |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 16/35 |
| Correct total non-H atom counts | 26/35 | 26/35 |
| Exact per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-Angstrom structures | 13/35 | 13/35 |
| No model | 4/35 | 4/35 |
| Hard process timeouts | 0 | 0 |
| Wall-time median, all outcomes | 2.396 s | 2.240 s |
| Wall-time p95, all outcomes | 6.958 s | 6.739 s |
| Summed solve wall time | 101.487 s | 99.297 s |
| Summed final-refinement time | 27.556 s | 25.092 s |
| Summed P1 phase time | 50.253 s | 50.312 s |
| Quality-scored cycles | 48,976 | 48,990 |
| Screening / final fits | 425 / 98 | 425 / 98 |
| Returned checked normal-cache steps | 0 | 1,743 |
| Successful original-solver fallbacks | 0 | 0 |

The median of the 35 paired control/treatment wall ratios is **1.01876x** (1.84%
less time), with 24 faster and 11 slower cases. The paired kernel ratio is
1.01447x (26 faster, nine slower). Among the same 16 successful cases, the paired
wall ratio is only 1.00421x (nine faster, seven slower). These are paired case
statistics; they differ from the ratio of the two unpaired wall medians. Summed
wall time falls 2.16%, while summed final refinement falls 8.94%.

The labelled-structure, total-count and per-element recovery sets are identical
between modes, not just their totals. All 16 successes from the latest committed
batched-FFT panel are retained. Seven of 31 nonempty selected models are exactly
identical; normal-equation reduction order changes refined coordinates/tensors
in other cases. This is expected round-off behavior, separately constrained by
the independent linear-system and complete-fit checks. Correctness tolerances
and all input-derived caps remain unchanged, with seven above-capacity references
retained in every denominator. There are no case-specific branches.

The solve-time gain is modest. P1 phasing still costs about 50 seconds across the
panel, versus 25 seconds for final fitting, and structure-factor evaluation and
original-operator checks remain part of every refined trial. A 20–45% improvement
in selected fixed-work fit benchmarks does not establish a comparable full-search
speedup. The option remains experimental and false by default. These are single-
seed balanced development collections, with external host load uncontrolled;
no repeated-seed or holdout reliability is claimed. The 65 holdout outcomes remain
unopened.

The unchanged SHELXT comparator returns 13 labelled structures and 29 total
counts in both fresh panels, with one hard timeout and wall medians 0.911/0.906
seconds. PhaseSmith receives the deposited space group, so these results do not
establish general parity with SHELXT.

Final artifacts:

- `benchmarks/structure-solution-normal-cache-control-macos.json`
- `benchmarks/structure-solution-normal-cache-cached-macos.json`
- `benchmarks/structure-solution-normal-cache-comparison-macos.json`
- `benchmarks/structure-solution-normal-cache-schedule-macos.json`
- `benchmarks/structure-solution-normal-cache-kernels-macos.json`

Raw models, failures, load samples, source snapshots and pre-reference hashes
are under `runs/normal-cache-final/`. The collection/audits completed before a
comparison-output collision stopped the wrapper; the first comparison was
archived and the comparator rerun against the already saved final panels. No
prediction, timing, audit tolerance or case was changed by that report-only
recovery. Both final reports and the regenerated comparison pass their hash and
schedule checks.
