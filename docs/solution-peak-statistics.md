# Guarded peak statistics in P1 phasing

This experiment starts from `461aaa0`, the path recovering 19 of 35 development
structures. The experimental `streamed_peak_search` option is false by default
and requires `filtered_peak_search`. Both comparison modes retain every preceding
fast option, including specialized isotropic evaluation.

## Profile and experiment selection

Three sequential five-second profiles repeat full development solves for COD
2243618, 2010647 and 2241159 with the previous best settings. FFT arithmetic and
layout together dominate their P1 time; adaptive selection is the next major
individual kernel, with 519, 580 and 447 collapsed self samples, respectively.
Exact drivers, native library hashes, image UUIDs and collapsed sample counts
are in `benchmarks/structure-solution-peak-statistics-profiles-macos.json` and
`runs/post-isotropic-phasing-profile/`. These are representative samples rather
than full-corpus stage allocations. No own heavy workloads overlapped profiling.

A 32-line streaming FFT prototype passed 70 targeted tests and full validation,
but the broader same-executable benchmark showed only 1.92%/0.95% lower phasing
median at 32-cubed/64-cubed and 0.13% lower complete-search time. That prototype
is set aside instead of being incorporated into the solver. Its source patch,
raw Criterion samples and implementation hashes remain in
`benchmarks/structure-solution-wide-fft-prototype{,-kernels-macos.json}` (the
source patch has suffix `.patch`). Its unused adapter-only seal revision 27 is
retained unchanged. The current adapter revision 28 inherits the preceding
accepted revision 26 directly. No corpus outcomes were collected for the FFT
prototype. The present change targets full-map RMS statistics and candidate
scanning in the existing adaptive selector.

## Equations and conservative decision guard

The original map statistic is unchanged:

    M = max_i |rho_i|
    t_i = fl(fl(rho_i/M)^2)
    S_ordered = sum_i t_i in the original scalar order
    R_ordered = M sqrt(S_ordered/N).

For a nonzero finite map, 0 <= t_i <= 1 and at least one t_i equals 1. The new
path computes the same rounded t_i into eight fixed independent accumulators,
then combines them in fixed order. This exposes SIMD without fast-math or FMA.
The internal R estimate can differ slightly from R_ordered. It is not returned
as an observed statistic or used unchecked to change a peak decision.

Let epsilon be f64 EPSILON, u=epsilon/2, and gamma_k=k*u/(1-k*u). For these
nonnegative terms, each summation's absolute error relative to their exact sum
T is bounded by its gamma term times T, plus the negligible accumulated absolute
subnormal rounding error. The scalar chain has at most N additions; the lane
chain has at most ceil(N/8)+7. With N <= 2^21 and T >= 1, their difference and
the following division, square root and multiplication roundings fit well inside
this deliberately conservative relative envelope:

    radius = (32*N + 128) * epsilon.

The initial sigma lies in [0.1,10]. A normal initial cutoff therefore also bounds
any subnormal RMS multiplication error relative to that cutoff; the envelope's
128*epsilon allowance is more than sufficient in addition to the summation
allowance. Non-normal initial thresholds and overflow-adjacent intervals use the
original scalar RMS immediately. Later sigma increases do not change the RMS.

For an estimated cutoff c=sigma*R_estimated, the interval is

    lower = next_down(c*(1-radius))
    upper = next_up(c*(1+radius)).

Both endpoints are rounded outward by one ULP using the monotone IEEE-754 bit
patterns of positive finite values, preserving the Rust 1.85 minimum. An
unrepresentable or subnormal lower interval and a nonfinite upper interval
trigger scalar fallback. The radius also covers cutoff multiplication rounding.

The initial scan uses lower, so it retains every local maximum that the original
scalar cutoff could accept. All 26 neighbors are compared against the original,
unthresholded density, with the same positive-value and smaller-flat-index tie
rules. Scanning streams contiguous density rows; periodic Z offsets are loaded
only after threshold filtering. The two adjacent central-row neighbors are
checked first, then the remaining 24. Reordering these finite Boolean comparisons
does not change which voxels are maxima. Sorting and peak-coordinate refinement
are unchanged.

At each adaptive sigma, a maximum inside the inclusive uncertainty interval
causes one recomputation of R_ordered. The original scalar value then controls
all remaining threshold/cap decisions. Outside that interval, every maximum has
the same inclusion decision under both cutoffs. Sigma is raised by the same
factor 1.5 until the same cap is satisfied. The retained maxima already include
every scalar survivor, so fallback needs no second map scan. These steps retain
exact peak indices, effective sigma and refined coordinates, rather than merely
using a small tolerance on structure outputs.

Zero maps, invalid inputs, finite support, interpolation, mask construction,
omission seeds, Fourier equations, phase quality, reflection handling, count
selection, refinement and scheduling are unchanged. No grid-sized workspace,
new input cap, reduced precision, external dependency or numerical thread is
introduced. There are no new fit parameters or physical derivative terms.
`streamed_peak_cycles` records actual valid adaptive selections on this path;
`peak_rms_fallbacks` records actual scalar fallback calls separately from the
requested option. The native diagnostic selector and phase helper expose the
same choice and counters.

## Independent validation

The independent NumPy equations in
`python/phasesmith/experimental/solution_count_free_reference.py` remain unchanged
because the statistical equation and selected peaks are preserved. Native
outputs are checked bitwise against the scalar selector and numerically against
that independent implementation over noise, plateaus, periodic corners,
negative/zero/constant maps, tiny/large finite scales, threshold raising and caps.
Adversarial thresholds are placed within one ULP of a scalar inclusion boundary
and must exercise fallback. Subnormal maps and overflow thresholds are covered.
Rust interval-enclosure tests include the maximum 128-cubed grid.

Complete 80-cycle phase histories, peak counts, effective sigmas, omission
decisions and selected maps match bitwise for three seeds, random/Patterson
starts and omitted/non-omitted cycles. Three complete synthetic element-only
searches preserve exact models, correct counts and final anisotropic mode.
Existing analytical-derivative and physical/refinement tests remain in the full
workspace suite. The pinned GSAS-II powder-profile oracle does not cover this
single-crystal execution path; no GSAS-II or decompiled external code is used.

## Development protocol

`benchmarks/solution_peak_statistics_interleaved.py` collects both modes serially
on all 35 development cases. Only `streamed_peak_search` differs. Seed zero,
four starts, 800 requested cycles, native 13-second/outer 15-second limits,
requested anisotropic final fits and original input-only caps stay fixed. Leading
mode alternates by case, 18/17. All 140 predictions precede both reference audit
barriers. Seven above-cap references remain in every denominator. Adapter-only
seal 28 preserves the registry and 35/65 split without reopening source/reference
data. The 65 holdout outcomes remain closed. No task-owned tests, builds,
profiling or kernel benchmarks overlap timed corpus collection.

Criterion `peak_statistics_` cases cover a realistic 32-cubed multi-peak scan,
complete 64-cycle 32-cubed/64-cubed phasing, and a complete eight-atom Fe/C/N/O
search with 2,108 expanded rows. Actual cycles, transforms, fallback counts and
search work are printed in the raw log. Comparisons use modes within one
executable, not Criterion's historical build-to-build deltas.

```sh
cargo bench -p phasesmith-single-crystal --features solution --bench solution -- peak_statistics_ --sample-size 20 --warm-up-time 1 --measurement-time 2
.venv/bin/python benchmarks/solution_peak_statistics_interleaved.py --output runs/peak-statistics-new --seconds 15
```

Results are added after complete collection, normalized development auditing,
and comparison with the exact preceding set of 19 successful structures.

## Complete development results

Both panels recover the exact same **19/35 structures**, **26/35 total non-H
counts**, **19/35 per-element counts**, and **15/35 strict 0.02-Angstrom structures**.
All four case sets match both the fresh control and the preceding best report.
All 35 inferred count vectors are unchanged. All 31 returned model records,
including coordinates and reported R1, are exactly identical between modes;
the four no-model outcomes are identical too. No PhaseSmith hard timeouts occur,
and all seven above-cap references remain in the denominator. Holdout outcomes
remain unopened.

| Measurement | Scalar control | Guarded statistics | Change |
| --- | ---: | ---: | ---: |
| Summed wall time, all 35 outcomes | 69.51143 s | 67.60997 s | 2.74% less |
| Paired median control/new time ratio | — | 1.01977 | about 1.94% less typical time |
| Raw wall-time median | 1.75560 s | 1.74788 s | descriptive, not a paired estimate |
| P1 phase-search stage sum | 30.97753 s | 29.63369 s | 4.34% less |
| Isotropic screening stage sum | 11.82041 s | 11.78424 s | essentially unchanged |
| Final refinement stage sum | 11.94596 s | 11.65283 s | host-sensitive variation |

Twenty-nine cases are faster and six slower. This is a single balanced paired
collection on the current host, not a platform-wide performance guarantee. The
phase-stage reduction is the relevant targeted change; the implementation does
not alter final refinement, so its timing variation should not be attributed to
a new final-fit algorithm.

Both modes perform 95 phase starts, 675 isotropic screens, 125 final fits, 249
ambiguity relaxations and 10,305 cheap ambiguity scores. Checked assignment and
final steps remain 4,640 and 2,460, without normal-cache fallback. The new selector
is actually used 49,876 times. No RMS fallback is needed on this corpus; the
adversarial boundary and subnormal tests separately require and verify fallback.
Existing per-start wall-clock safeguards allow 219 additional completed Fourier
workspace cycles in three cases (49,657 control versus 49,876 guarded), with no
changes to scheduling rules, quotas or caps and no change to final model records.
These extra cycles partly consume the throughput gain under the fixed deadlines.

Same-executable Criterion median comparisons are:

| Kernel/workflow | Control | Guarded | Time reduction |
| --- | ---: | ---: | ---: |
| 32-cubed multi-peak scan | 47.960 us | 31.808 us | 33.68% |
| 32-cubed complete 64-cycle phasing | 19.658 ms | 18.447 ms | 6.16% |
| 64-cubed complete 64-cycle phasing | 230.204 ms | 152.295 ms | 33.84% |
| Complete eight-atom synthetic search | 216.871 ms | 214.677 ms | 1.01% |

Kernel gains are not substitutes for the full-corpus result. Raw samples,
confidence estimates, benchmark/source/executable hashes and actual completed
work are preserved in the kernel artifact.

Artifacts use prefix `benchmarks/structure-solution-peak-statistics-` followed by
`{control,cache,comparison,kernels,profiles,schedule,retention}-macos.json`.
The retention artifact records all 35 count vectors, all exact success/count/strict
case sets, source-report hashes, timings/work/stage totals, cycle variations and
its verifier source. The schedule audit verifies 140 predictions before both
reference barriers, alternating 18/17 leading cases, matching current 44-file
source snapshots and the identical native library hash in both modes. The
set-aside FFT prototype patch applies to baseline `461aaa0` in an isolated Git
index; its historical source/executable hashes are distinct from the accepted
implementation.

Final checks pass: formatting, clippy with warnings denied, all-feature workspace
Rust tests (**416 passed, 34 ignored**), and pytest (**2,392 passed, 11 skipped,
34 deselected**). External oracle and real-data tests retain their normal opt-in
requirements. The execution option stays false by default.
