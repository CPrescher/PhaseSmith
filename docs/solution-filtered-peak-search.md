# Filtered adaptive peak search

`filtered_peak_search=False` retains the original execution path. The new true
option requires reusable Fourier workspaces and implements the same adaptive
peak decisions with fewer neighborhood tests and index conversions. It does
not change the requested work quotas, wall limits, equations, precision,
reflection support, phase ranking, initial seeds, candidate/refinement policy,
atom/Jacobian caps or correctness gates. Final fitting still uses the original
anisotropic mode and observations. The option remains experimental.

## Diagnosis and profile

The preceding independent-family assignment campaign spends 72.161 seconds in
P1 phase calls out of 121.523 total wall seconds over the 35 development cases.
Its typical paired solve time remains essentially unchanged after reducing
anisotropic work. Expensive P1 starts often reach their approximately 0.975-second
phase backstop; faster execution can be spent on extra cycles until the work
quota is reached. The work-sliced schedule already in use allows shorter starts
to return earlier when their fixed cycle quota finishes, so a substantial
per-cycle speedup can now reduce elapsed time.

A five-second macOS sampling profile of repeated element-only solves of the
already-audited development case COD 2010647 identified a concrete hot loop.
Of 4,205 main-thread samples, collapsed leaf counts include 1,178 in local-maxima
search and 486 in periodic grid-index conversion, about 40% combined. FFT work is
also substantial. The old selector visits all positive voxels and computes
periodic neighbor indices before applying the adaptive RMS threshold. This is
representative-case evidence, not a profile of every development input.

The raw sample and driver source, baseline native/source hashes, controls and
qualifications are retained in
`benchmarks/structure-solution-peak-profile-macos.{txt,json}`. The sampled wall
times are not performance comparisons. Only task-owned processes were sampled;
other user calculations were left untouched. No holdout data were used.

## Exact transformation and boundaries

The existing selector computes

    a = max_i abs(rho_i),
    rms = a * sqrt(sum_i (rho_i/a)^2 / N),
    theta_0 = sigma * rms,

with an explicit empty result at a=0. Peaks must be positive local maxima over
all 26 periodic neighbors. A tied neighboring value rejects index i when its
flat index is smaller than i. Order surviving maxima by decreasing density and
then ascending index. If the number above threshold exceeds the cap, multiply
sigma by 1.5 repeatedly. Equality at every threshold is included. The resulting
count is a data-selected count, not an input composition or forced atom count.

Since sigma only increases, no voxel below theta_0 can ever be returned. The new
plan computes exactly the old RMS arithmetic/reduction order first, skips such
voxels before testing neighbors, and still compares every candidate against all
26 original neighbor values. Below-threshold neighbors are not erased or altered.
The peak predicate, plateau tie rule and ordering are therefore unchanged.
The same existing sub-grid interpolation routine refines the selected indices.

Periodic previous/current/next coordinate offsets are prepared once for each
axis, including each axis's flat-array stride. Nested axis loops visit flat
indices in the original order. Neighbor indices require additions of cached
offsets, avoiding repeated signed remainders and coordinate decomposition.
Storage is O(Nx+Ny+Nz), at most 9 KiB of offsets for the current axis limits;
there is no 26-neighbor table for every voxel. A reusable vector holds surviving
maxima. The existing Fourier/map/mask buffers, Gaussian projection arithmetic,
three-sigma support and random-omit sequence are unchanged.

The original even axes in [8,128], at most 2,097,152 voxels, finite densities,
sigma in [0.1,10] and cap in [1,128] remain the boundary. Raised effective sigma
can exceed the starting-sigma range, as before. Zero and all-negative maps,
periodic-edge plateaus, non-cubic grids, non-power-of-two even axes, extreme
finite scales and exact threshold equality retain their conventions. All
floating-point values remain f64. No new differentiable objective is introduced;
the unchanged fused value/analytical-derivative refinement and density-integral
kernels keep their independent tests. The GSAS-II powder oracle does not cover
this peak-selection stage.

## Verification and declared experiment

The original native selector remains the execution control. Existing independent
NumPy peak/RMS/periodic-neighbor and complete dual-space references are reused,
with no call back into Rust. New tests compare every selected coordinate and
effective threshold bitwise against the original path, plus independent NumPy
values. Complete trajectories compare densities, every phase score, selected
cycles and adaptive/omit histories. Quality-ranked Rust trajectories and synthetic
P1/P-1/Pm complete model/count recovery also compare both paths. No global
tolerance is loosened.

Benchmarks cover peak scans on an eight-atom 32-cubed map, complete 64-cycle
Fourier work on 2,108 measured rows, and an eight-atom whole search with two
starts/128 requested cycles and generous wall backstops. The control already
uses scalar scoring, reusable workspaces, work-sliced phasing and competing
isotropic assignments; the new flag alone changes.

The matched campaign collects both programs on all 35 development cases,
serially interleaved between control/filtered modes with alternating case-leading
positions. All 140 predictions are saved before either reference barrier is
released. Native/source hashes, per-job load, chosen models, failures and actual
cycle/fit counters are retained. Native 13-second/outer 15-second caps, seed zero,
four starts, 800 requested cycles, one numerical thread, original input-only
resource caps and final anisotropic mode remain fixed. Exact non-H per-element
counts plus RMS <=0.025/max <=0.05 angstrom remain the correctness gate; strict
0.02-angstrom recovery is secondary. Seven above-cap references remain in the
denominator. The adapter-only revision-17 seal retains the registry and split.
The 65 holdout solver outcomes remain unopened. No task-owned builds, tests or
microbenchmarks overlap timed collection.

## Completed results

Both panels use native SHA-256
`ef6eb3ffcca222db482859eb98dd3d0c147bc91e8a60eccd9617a36a4abc826e`.
All 38 recorded implementation files match the two snapshots and current sources.
The schedule has all 140 predictions before both reference barriers, with 18
control-first and 17 filtered-first cases. Every mode/method includes exactly
the 35 development cases; seven references above native capacity remain in each
denominator. Final ADP mode, observations, caps, seeds and every other feature
flag match. The original isotropic-assignment success set is retained in full.

| All 35 development cases | Current control | Filtered peak search |
| --- | ---: | ---: |
| Correct structures with exact per-element counts | 16/35 | 16/35 |
| Correct total non-H counts | 26/35 | 26/35 |
| Correct per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-angstrom structures | 13/35 | 13/35 |
| Median wall time, including startup | 3.418 s | 2.543 s |
| Wall p95 | 6.937 s | 6.664 s |
| Sum of solve wall times | 122.078 s | 103.492 s |
| Sum of P1 phase time | 72.441 s | 55.259 s |
| Phase starts / scored cycles | 98 / 39,143 | 98 / 45,955 |
| Short quota completions | 26 | 43 |
| Isotropic / final anisotropic fits | 426 / 100 | 424 / 96 |

The primary matched wall ratio is **1.18703x control/filtered**, equivalent to
about 15.8% less typical paired elapsed time, with median paired difference
-0.61675 seconds. Thirty-four cases are faster and one slower. Native solver
time improves on all 35 cases, with paired median ratio 1.20878x. Among the 16
common successes the paired wall ratio is 1.20472x, about 17.0% lower elapsed
time; 15 are faster and one slower. Their unpaired wall medians are 1.401/1.212 s.
The all-outcome raw median drops 25.6%, but the matched ratio above is the
appropriate typical per-case effect; the two statistics are not interchangeable.

Cumulative solve time falls 15.2% and P1 phase time 23.7%. Both modes still use
98 starts. Faster cycles now finish 43 of 57 early work quotas instead of 26,
so the previous work-sliced policy can release time to model completion. Thirty
starts still reach the wall ceiling (52 before), spending some speed gain on
additional cycles. Scored work rises 17.4% even as phase time falls. No new
checkpoint or acceptance rule is used. Final-refinement stage time also changes
slightly, 26.649 to 25.166 seconds, because the additional deadline-limited
phase work can select a different best map.

The correctness, total-count and per-element-count success sets are each
identical between modes, not merely equal in aggregate. Twenty-seven of 31
returned models are exactly identical; four differ after extra phase work while
preserving the audited outcome classes. The four no-model outcomes remain
unchanged. Fixed-work tests establish bitwise equivalence; wall-limited solves
are not claimed identical when the faster path evaluates additional cycles.

Examples include correctly recovered COD 2108342, 1.362 to 0.963 seconds; COD
2011241, 3.636 to 2.543 seconds; and COD 2241796, 1.195 to 0.786 seconds. The
only wall regression is correctly recovered COD 2010746, 0.624 to 0.628 seconds;
its native solve still gets faster, so that small startup-inclusive difference
does not establish a kernel regression. Some large grids retain their wall
backstop, for example still-incorrect COD 2011181 at 6.761/6.745 seconds.

SHELXT controls preserve 13/35 structures and 29/35 total counts in both panels,
with wall medians 0.879/0.895 seconds and paired ratio 1.00133x. Median one-minute
host load samples are 4.38 in both modes (range 3.58–5.94). The external controls
and balanced order help assess host variation; this remains one interleaved,
single-seed development comparison, not a repeated timing or holdout reliability
estimate. PhaseSmith still receives the exact space group, so numerical recovery
scores do not establish general capability parity with SHELXT.

The realistic fixed-work benchmark corroborates the profiled mechanism:

| Benchmark median | Control | Filtered | Control/filtered |
| --- | ---: | ---: | ---: |
| Eight-atom 32-cubed peak scan | 322.491 us | 47.682 us | 6.76x |
| Complete 64-cycle P1 phasing, 2,108 rows | 68.958 ms | 40.675 ms | 1.70x |
| Eight-atom whole search, two starts/128 cycles | 117.350 ms | 89.963 ms | 1.30x |

Both phase benchmark modes actually complete 64 cycles. The 41.0% phase-kernel
and 23.3% whole-search reductions are fixed-work synthetic evidence, distinct
from the deadline-limited 35-case comparison. Raw samples, confidence intervals,
benchmark/numerical sources and executable hashes are retained in
`benchmarks/structure-solution-peak-search-kernels-macos.json`. Complete normalized
reports, guarded paired comparison and verified prediction schedule are the
`structure-solution-peak-search-{control,peaks,comparison,schedule}-macos.json`
files in `benchmarks/`. The snapshot/native/report hashes are verified, including
actual filtered-cycle counters matching adaptive workspace projections.

Validation passes **1,900 Python tests** (11 external-data skips, 34 marker
deselections), **404 Rust tests** (34 ignored), strict all-feature/all-target
Clippy, Rust formatting, repository Ruff and whitespace checks. The option stays
off by default pending broader seed/input validation. All 65 holdout outcomes
remain unopened.

Reproduce collection with a release experimental build and a fresh directory:

```sh
.venv/bin/python benchmarks/solution_peak_interleaved.py \
  --output runs/filtered-peak-search-new --seconds 15
```

The scheduler selects development only. Normalize each completed panel with
`solution_corpus_reaudit.py` and compare the complete reports using
`solution_phase_ablation.py`. The committed comparison retains exact input hashes.
