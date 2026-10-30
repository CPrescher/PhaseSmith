# Scalar phase scoring and reusable Fourier workspaces

Two independently controlled experimental options are added:
`score_only_evaluation=False` and `reusable_fourier_workspaces=False`.
They optimize execution of the existing numerical method. No observation,
normalization equation, peak threshold, precision, support convention, seed,
phase-map ranking, orbit edit, refinement equation or resource cap is changed.
Score-only requires phase-quality selection; workspaces require dual-space
phasing. Existing defaults and diagnostic full value/gradient evaluation remain.

## Scalar merits with prepared fixed statistics

`PreparedPhaseScore` borrows an immutable `PreparedPhaseQuality`. Plan arrays,
weak membership and shell indices are validated once per prepared scorer. The
scorer caches allowed-target mean and variance, weak count, shell indices and
shell populations. It owns normalized-amplitude and shell sum/mean work arrays,
reset in place for each map. Calculated amplitudes remain validated on each call.
The shared solver prepares one scorer per P1 trial and reuses it across cycles.

For maximum calculated magnitude Cmax, use c_i=C_i/Cmax. With allowed target
mean mt and calculated mean mc, the correlation is

    CC = sum_allowed((t_i-mt)*(c_i-mc))/sqrt(Vt*Vc).

Undefined correlation still returns zero when either variance is <=1e-28.
For shell s, the existing quantities are

    D_s = sum_s(c_i*c_i*w_i)/population_s,
    A_s = sum_(s,weak)(c_i*c_i*w_i),
    Rweak = sum_(s,D_s>0)(A_s/D_s/weak_count),
    CFOM = CC - Rweak.

The original row/bin iteration and f64 reduction orders are retained. A zero
shell contributes zero. Empty/invalid maps and a degenerate total second moment
retain the existing errors and 1e-250 convention. The old fused scalar/analytical-
gradient evaluator remains unchanged for derivative consumers and as the native
control. The new value type has no gradient field: score-only callers never
construct an unused gradient or silently return a partial derivative array.
No profile/refinement value/derivative pass is split.

Independent NumPy quality equations already cover these unchanged expressions.
New batch tests additionally check repeated scratch reuse, ordinary and extreme
positive scales, undefined correlation and zero shells, bitwise equality to the
full native evaluator, and equal invalid-input rejection. Native tests cover
malformed prepared plans, empty shell indices and repeated mixed zero shells.
Existing finite-difference quality-gradient tests continue to exercise the full
fused evaluator, whose equations have not changed.

## Reused P1 maps, masks and Fourier storage

The allocating atomicity kernel is deliberately retained as the control. The
workspace path prepares cell geometry, Gaussian support bounds and mask constants
once per trial. It resets the mask and overwrites the projected map each cycle,
using the same peak order, periodic indices and exponential operations. Gaussian
support remains r² <= (3*sigma)²*(1+1e-12), with the same unit-volume prefactor.
Original grid and support boundaries remain observable and unchanged.

The initial Fourier allocation becomes the recycled complex coefficient array.
Projection, coefficient and real-density buffers are reused across cycles. Best
maps own their storage: a newly selected best swaps its projected buffer with
the old best buffer, which returns to the workspace. Window/hand-off maps retain
independent copies. Quality and tangent input vectors also retain capacity.
Peak detection, omission and tangent internals may still allocate; this is not
an allocation-free whole solver.

Both paths already cache FFT plans and line scratch. With workspaces enabled,
the unit-stride FFT axis is transformed directly in contiguous grid slices,
avoiding its gather/scatter copies. Other axes, transform direction, f64 plans,
axis order and normalization remain identical. Supplied-symmetry recycling,
which already reuses full-grid buffers, uses the same contiguous-axis shortcut.
There is no new FFT dependency, real/complex layout change, float32 conversion,
fast-math reduction or altered missing-reflection support.

Counters record actual scalar-only evaluations and P1 workspace projections.
They do not claim that every helper allocation has disappeared. Default powder
builds and GSAS-II oracle boundaries are unchanged. The powder oracle does not
cover this single-crystal phasing stage; independent NumPy/Fourier references,
bitwise native ablations and full structure recovery are the relevant checks.

## Validation and benchmark protocol

Native fixed/adaptive peak tests compare all four execution modes through
complete trajectories and score traces, including Patterson/random starts and
omission. Python tests compare native workspace paths bitwise, and independent
NumPy random-start cycles on cubic and unequal grids. For Patterson starts with
random omission, exact double-image peak ties can be ordered differently after
NumPy/Rust FFT rounding; the independent comparison covers initialization/first
cycle, while the entire 40-cycle old/new native trajectory must remain bitwise
identical. No old/global tolerance is loosened to mask a branch difference.
Synthetic P1, P-1 and Pm full searches require the original correct counts and
coordinates and identical selected models/histories across all four modes.
Measured input arrays are preserved.

The predeclared development experiment compares current chemical/iterative/
phase-origin control, scalar-only, workspace-only and both on all 35 cases.
Missing-reflection prediction, extension, adaptive plateau stopping and independent
origin sampling remain off. Inputs, seed 0, four starts, 800 cycles, 13-second
native/15-second external ceilings and one numerical thread stay fixed. Seven
references exceeding the original input-derived model capacity remain in every
denominator. Correctness still requires exact full-cell non-H element counts,
maximum coordinate error <=0.05 angstrom and RMS <=0.025 angstrom. Strict 0.02-
angstrom recovery is a secondary result only.

Unlike earlier panel-by-panel collections, all four modes run serially interleaved
for each case/method. Mode position rotates by case (9/9/9/8 leading positions
across 35 cases). Every collector pauses at a reference barrier; all 280 selected
predictions (35 cases x two programs x four modes) must be saved before any
reference audit begins. Per-job load averages and a shared schedule are retained.
Existing external calculations are left untouched. This balances gradual host
load drift but cannot guarantee an idle machine or eliminate short load spikes.
Paired case ratios and unmodified SHELXT timings supplement overall medians.
No test/build/microbenchmark from this task overlaps timed corpus collection.

Adapter-only seal revision 14 retains the registry, split, source bytes and all
65 unopened holdout outcomes. The scheduler is included in each source snapshot.
`solution_execution_interleaved.py` is restricted to the development split; the
ordinary single-panel runner retains its existing behavior.

Validation before timing passes 1,693 Python tests (11 external-data skips,
34 deselected), 401 Rust tests (34 ignored), all-feature/all-target Clippy with
warnings denied, Rust formatting, Ruff and whitespace checks. No existing
numerical or structural tolerance has been loosened.

The measured native kernel case contains eight atoms, a 32-cubed Fourier grid
and 2,108 measured rows. All four benchmark trajectories complete exactly 64
cycles. Median times are 89.420 ms for control, 86.473 ms for score-only,
83.470 ms for workspaces and 81.780 ms combined (8.5% lower than control).
For the 1,054-family scalar merit itself, full value/gradient evaluation takes
12.494 us versus 4.976 us for scalar-only scoring (2.51x). These are observations
under active external host load, not quiet-host speed estimates. Raw Criterion
samples/intervals, source/executable hashes and cycle counts are retained in
`benchmarks/structure-solution-execution-kernels-macos.json`. Kernel measurement,
this task's tests and corpus collection did not overlap.

## Complete 35-case development results

All four interleaved panels use compiled native SHA-256
`254ae99573b5b1e41b104da45b47c9e9a86c51de38034e575bc5862f02cd1e35`.
Current numerical/adapter sources and all four saved source snapshots match the
collection hashes. The full chosen predictions and audits are retained in
`benchmarks/structure-solution-execution-{mode}-macos.json`, for control, score,
workspace and combined. The guarded derived comparison is
`benchmarks/structure-solution-execution-comparison-macos.json`.

| All 35 development cases | Control | Score only | Workspaces | Both |
| --- | ---: | ---: | ---: | ---: |
| Correct labelled structures | 14 | 14 | 14 | 14 |
| Exact total non-H counts | 26 | 26 | 26 | 26 |
| Exact per-element full-cell counts | 15 | 15 | 15 | 15 |
| Strict 0.02-angstrom recoveries | 11 | 11 | 11 | 11 |
| Median wall seconds, all outcomes | 3.405 | 3.334 | 3.336 | 3.291 |
| 95th percentile wall seconds | 7.742 | 7.725 | 7.731 | 7.740 |
| Median paired control/run wall ratio | 1.0000 | 0.9976 | 1.0041 | 1.0012 |
| Median paired control/run kernel ratio | 1.0000 | 1.0008 | 1.0025 | 1.0020 |
| Hard timeouts | 0 | 0 | 0 | 0 |
| Actual P1 starts | 95 | 95 | 95 | 95 |
| Quality-scored Fourier cycles | 44,928 | 45,009 | 45,707 | 45,989 |
| Isotropic screening fits | 184 | 184 | 184 | 184 |
| Requested-mode refinement fits | 104 | 104 | 104 | 104 |

There are no gained or lost structures or count outcomes. All 31 nonempty chosen
models, including species, coordinates and reported R1, are exactly identical to
the control in every mode. The same four cases return no model. All 35 cases,
including references exceeding the original input-derived capacity, remain in
every denominator. Correct total counts alone still do not certify element
assignments or coordinates.

The combined overall median is 3.4% lower, but **typical paired whole-solve time
is essentially unchanged**: the median of its 35 matched control/run ratios is
only 1.0012x, and the median matched wall-time difference is -5.9 ms. Combined
is faster in 19 cases and slower in 16. For the 14 common correct recoveries,
the paired median ratio is 1.0006x. Separate common-success medians are 1.500 s
for control and 1.541 s for both; a difference of unpaired medians is not the
median paired effect. The report helper explicitly computes matched ratios and
requires all declared cases; missing timing fields never silently reduce the
comparison denominator.

The kernel benefit therefore does not establish a material whole-search speedup
under the existing schedule. Most slow phase trials are limited by a wall-time
slice: saved per-cycle work permits additional Fourier iterations rather than an
earlier handoff. Combined completes 1,061 more quality-scored cycles (2.36%) with
the same starts, screening/refinement counts and selected models. Its scalar-only
counter records 45,989 actual calls. Workspace projection counters are 45,751
alone and 46,028 combined; they can include a final projection interrupted by a
deadline before reciprocal scoring, so they need not equal completed quality
cycles. No convergence threshold or stopping policy was changed to claim speed.

The external single-thread SHELXT control recovers 13 structures, 29 total counts
and 14 element-count sets in each panel, with one hard timeout. Its interleaved
medians stay close: 0.867/0.854/0.863/0.859 s, unlike the preceding study's large
panel-to-panel drift. Per-job load averages remain in the full reports. These
are one balanced collection, not repeated quiet-host timing estimates; external
host calculations were not stopped.

`benchmarks/structure-solution-execution-schedule-macos.json` retains the 280-job
schedule and checksum. Verification confirms 35 native case groups, rotating
mode order, and all 280 saved predictions before the four reference barriers.
The 65 holdout outcomes remain unopened. Both execution controls remain explicit
and false by default, allowing reproducible ablation without changing older
experimental callers.

After collection, 37 focused execution/report tests also pass, including checks
that paired ratios differ from ratios of unpaired medians, that missing timings
are not cherry-picked, and that feature/header mismatches are rejected. Final
all-target/all-feature Clippy, formatting, Ruff and whitespace checks pass.

Reproduce the balanced collection with
`python benchmarks/solution_execution_interleaved.py --output NEW_DIRECTORY
--seconds 15`. Normalize each complete panel with `solution_corpus_reaudit.py`
using `validation/solution-corpus-v5.json`, then compare the four saved reports
with `solution_phase_ablation.py`. For ordinary single-panel runs, add
`--score-only-evaluation`, `--reusable-fourier-workspaces`, or both to the
previous chemical/iterative/normalization/phase-origin controls.
