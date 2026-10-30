# Adaptive phase checkpoints and independent origin sampling

Two experimental controls preserve the existing path when false:
`adaptive_checkpoints=False` and `independent_origin_sampling=False`.
The first requires phase-quality, phase-origin and phase-first selection plus
`dual_checkpoints=True`. The second requires phase-origin selection.
Neither control reads atom counts, formulas, reference coordinates or holdout
outcomes. They do not alter measured intensities, normalization, scoring families,
atom/Jacobian limits, refinement equations or correctness gates.

## Adaptive handoff of P1 maps

The current control's phase-first path does **not** perform full-model completion
at every legacy 4/16/64/... checkpoint: it completes the selected map after its
phase trajectory ends, with an existing early stop at CFOM >= 0.9 after cycle 32.
That distinction matters when interpreting the number of requested-mode fits.
The new policy changes when the phase-first trajectory hands off its best map;
it does not claim to eliminate legacy callbacks that were already disabled.

Checkpoint the live non-omitted atomic map at 16, 64, 256 and the cycle ceiling.
The scheduled 16*4^n checkpoints are non-omission cycles; the existing global
best-quality target is also tested on every cycle from 32 onward.
Keep the existing high-quality test on every cycle from 32 onward. At a scheduled
checkpoint from cycle 64 onward, a stable-map handoff is allowed if:

- live measured-only CFOM is at least 0.65 and no more than 0.015 below the
  running-best CFOM;
- its absolute change since the previous live checkpoint is at most 0.015;
- the live-map amplitude SSE changes by at most 2% of the preceding live SSE
  (denominator floor 1e-12);
- positive-scale-invariant map cosine is at least 0.995.

The cosine is `(a dot b)/sqrt((a dot a)*(b dot b))`, evaluated after separately
scaling each map by its largest absolute value. Empty, zero, nonfinite or
mismatched maps fail. Values are clipped to [-1,1] only for floating-point rounding.
This is a policy statistic, not a differentiated refinement objective. All existing
value/analytical derivative kernels remain unchanged.

A handoff ends the current P1 start and tests its best map with the same supplied
symmetry, chemical assignment and complete refinement. Failure advances to the
next independent start, as the existing high-quality handoff does. This is an
empirical scheduling choice, not proof of convergence or a recovered SHELXT rule;
a plateau can still be the wrong structure. Actual decisions record start, cycles,
live and running-best quality, live SSE, cosine, selected flag and reason. A selected map is not a certified
structure. Deadline/cancellation termination remains cooperative and unchanged.
The selector retains just the preceding live checkpoint map, not every Fourier
iterate. The stored champion map is never used to infer trajectory stability.

## Origin sampling independent of density-grid dimensions

The origin objective is the existing exact measured-phase equivalence loss.
For measured equivalent indices h and hm, its harmonics are q = hm - h. Use
nonzero-weight pairs from the existing deterministic 2048-family selection and
choose each seed FFT axis as

    N_a = clamp(next_power_of_two(2*max(abs(q_a)) + 1), 16, 128).

Thus frequencies below the cap fit strictly within the sampling Nyquist boundary.
The 128 cap is observable: higher frequencies may alias during seeding, but the
subsequent loss, gradient, Hessian and Newton/backtracking polish still use the
original exact harmonics. This does not promise exhaustive global optimization.
No density-grid expansion or extra reciprocal coefficients are introduced.
No atom model enters sampling selection. The configured origin seed shape and harmonic extents are audited
independently of the P1 density shape. A group with no informative phase pairs
(such as P1) returns its free-origin candidate directly, without an origin FFT. Legacy behavior continues to use the density
axis sizes clamped to 16..64. All previous local derivative and phase-wrap
conventions remain unchanged.

The published SHELXT workflow motivates selecting phase maps before atom fitting
and FFT-based origin alignment (Sheldrick, 2015,
[DOI 10.1107/S2053273314026370](https://doi.org/10.1107/S2053273314026370)).
The thresholds and frequency-grid policy here are independently specified
PhaseSmith experiments, not copied or mechanically translated SHELX code.
The GSAS-II powder oracle does not cover these single-crystal scheduling rules.

## Validation and development experiment

Independent NumPy references check map cosine/checkpoint decisions and measured
harmonic sampling extents. Existing origin finite-difference tests check unchanged
loss derivatives. Tests check padding-independent seed shapes, origin/hand recovery,
invalid inputs, strict full-cell counts and coordinates for synthetic P1, P-1
and Pm structures, both controls separately and together. A realistic 8-peak,
2108-row benchmark compares old and independent origin searches and times the
checkpoint policy on a 32-cubed map.

The predeclared four-way campaign uses all 35 development cases: current control,
adaptive only, independent sampling only, and both. It retains the current combined
chemical/iterative/normalization/phase-origin features, missing prediction off,
resolution extension 1.0, seed 0, four starts, 800 cycles, 13-second internal and
15-second outer budgets, and one numerical thread for both programs. Every panel
saves all chosen predictions before its reference audit. Seven cases exceeding
current native capacity remain in the denominator. Exact per-element full-cell
non-H counts plus maximum coordinate error 0.05 angstrom and RMS 0.025 angstrom
remain the structural gate; strict 0.02-angstrom recovery is secondary.
Adapter-only seal revision 13 retains the same registry, splits and source bytes.
The 65 holdout solver outcomes remain unopened.

Initial validation before timed corpus collection passed 1,665 Python tests (11 external-
data skips, 34 deselected), 398 Rust tests (34 ignored), Clippy for all features
and targets with warnings denied, Rust formatting, Ruff and whitespace checks.
No existing numerical or structural tolerance was loosened.

The isolated realistic kernel benchmark reports a median 4.185 ms for legacy
origin search versus 2.620 ms for independent sampling on the selected mirror-
symmetry 8-peak/2108-row 32-cubed case. The checkpoint comparison costs 50.87 us.
Criterion mean estimates and intervals are 4.183 ms (4.167--4.207), 2.633 ms
(2.616--2.659) and 51.52 us (51.02--52.10), respectively. These are kernel costs
for one fixed synthetic case, not a general end-to-end speedup. Raw samples,
estimates and implementation/executable hashes are retained in
`benchmarks/structure-solution-adaptive-sampling-initial-kernels-macos.json`. Kernel
measurements and corpus collection did not overlap tests or other benchmarks.


## Retained initial implementation and correction

The first four-way comparison is retained in full as
`benchmarks/structure-solution-adaptive-sampling-initial-{mode}-macos.json`
for control/adaptive/sampling/combined, plus the initial comparison report.
Its control/adaptive/sampling/combined medians were 3.447/2.632/3.507/2.569 s,
with 14 structures, 26 total counts and 15 element-count sets in every mode.
Adaptive and combined gained COD 4326074 but lost COD 2108342. The apparent
25.5% combined speed gain was therefore not a no-loss result.

Inspection found a policy error: two checkpoints compared the same stored
champion, so cosine 1.0 and unchanged champion scores were interpreted as
convergence even while live phases kept moving. For COD 2108342, the champion's
CFOM stayed 0.829734 from checkpoint 64 to 256 while live CFOM at 256 was only
0.741775. COD 4326074 likewise had live quality below its champion. This is an
algorithmic diagnosis from solver traces, not a reference-dependent threshold.

The corrected policy above compares actual live checkpoint maps and their live
quality/SSE, requires live quality close to the champion, and uses an absolute
quality change. All thresholds retain their predeclared values. Regression tests
separate stale champions from moving or degraded live maps, and integration tests
check that audited live CFOM equals the actual Fourier-cycle quality trace.
The best-map >=0.9 handoff is retained unchanged. All four complete development
panels are recollected with the corrected native build, preserving the initial
results and the same unopened holdout.

Final validation after the live-map correction passes 1,668 Python tests (11
external-data skips, 34 deselected), 398 Rust tests (34 ignored), all-target/
all-feature Clippy with warnings denied, Rust formatting, Ruff and whitespace
checks. The corrected isolated kernel medians are 4.116 ms for
legacy origins, 2.559 ms for independent origins and 60.62 us for
the live checkpoint comparison. Full raw estimates, samples and source/executable
hashes are in `benchmarks/structure-solution-adaptive-sampling-kernels-macos.json`.

## Corrected four-way development comparison

All four corrected panels use native SHA-256
`f568af67f4d7b784e3b5cc3ced61cd99a96d2bf7982be58cc208fd7fc6ce7b3a`.
Current numerical sources and each saved source snapshot match their collection
hashes. Full chosen models/audits are in
`benchmarks/structure-solution-adaptive-sampling-{mode}-macos.json` and the
complete guarded comparison is
`benchmarks/structure-solution-adaptive-sampling-comparison-macos.json`.

| All 35 development cases | Control | Adaptive only | Independent origins | Both |
| --- | ---: | ---: | ---: | ---: |
| Correct labelled structures | 14 | 14 | 14 | 13 |
| Exact total non-H counts | 26 | 26 | 26 | 26 |
| Exact per-element full-cell counts | 15 | 15 | 15 | 15 |
| Strict 0.02-angstrom recoveries | 11 | 11 | 11 | 11 |
| Observed median wall seconds | 3.393 | 3.819 | 3.973 | 4.558 |
| Observed 95th percentile seconds | 7.629 | 8.224 | 9.151 | 10.816 |
| Hard timeouts | 0 | 0 | 0 | 0 |
| Actual stable-map early handoffs | 0 | 0 | 0 | 0 |
| Quality-scored Fourier cycles | 45,227 | 40,653 | 40,737 | 32,774 |
| Requested-mode refinement attempts | 104 | 104 | 107 | 105 |

Adaptive-only and sampling-only preserve exactly the same 14 structures. Both
loses COD 2108342 with no gained structure; full atom counts remain correct,
while maximum coordinate error grows from 0.02373 to 3.51532 angstrom. This is a
real failed structural result under the declared time limit, not a threshold
change. The other 34 per-case recovery outcomes stay unchanged.

**The wall-time columns are contended-host observations, not isolated algorithmic
slowdown estimates.** Unchanged SHELXT medians in the same serial panels rise
from 0.843 to 0.952, 1.046 and 1.606 seconds. Its recovery remains 13 structures,
29 total counts and 14 element-count sets, with one hard timeout per panel.
CPU-heavy calculations in other user projects were observed immediately after
collection and were left running. A read-only post-collection load snapshot is
retained in `benchmarks/structure-solution-adaptive-sampling-host-load-macos.json`;
it is not a continuous record of load throughout the campaign. No test, build or
other benchmark from this task overlapped the timed collections.

In particular, none of the corrected adaptive runs selects a stable-map early
handoff. Their smaller Fourier-cycle totals therefore do **not** demonstrate an
adaptive speed gain: time slices expire sooner in terms of completed cycles.
Common-success medians also rise (14 adaptive/control successes: 1.511 to 1.536 s;
14 independent/control: 1.511 to 1.689 s; 13 combined/control: 1.525 to 1.871 s).
These single-seed serial panels cannot separate host contention from kernel cost.
Neither option has established a useful end-to-end speed/recovery improvement.

Independent sampling uses larger configured origin grids in 29 of 32 cases with
recorded informative phase relations: six grow by 2x in voxels, eight by 4x and
15 by 8x; two are unchanged and one shrinks by 8x. This explains why a fast
mirror-symmetry kernel example does not imply a general search speedup. The
current policy prevents ordinary seed aliasing below its cap, but does not
exploit shorter origin periods to minimize FFT work.

### Fixed-work diagnosis of the combined loss

For COD 2108342, the control completes 400 Fourier cycles within its phase time
slice; adaptive completes 327, sampling 349 and both only 258. The successfully
selected phase map appears at cycle 310 (CFOM 0.839593). At cycle 258 the stored
best is still the earlier cycle-44 map (CFOM 0.829734), which leads to the wrong
coordinates. Every corrected checkpoint explicitly rejects premature convergence
at cycles 16, 64 and 256; the lost combined run ends on its wall deadline.

A selected-case diagnostic therefore compares **exactly 400 Fourier cycles and
one start** in all four modes. It retains the original 13-second total native
limit and input-derived atom cap; one start gives the phase portion more time
than the headline four-start schedule. Each mode is required to complete the
declared identical work. All four chosen predictions are saved before opening
the registered development reference. All four then recover the correct counts
and coordinates, with identical maximum error 0.02373 and RMS 0.01381 angstrom.
The full result and pre-reference snapshot checksum are retained in
`benchmarks/structure-solution-adaptive-sampling-work-probe-macos.json`.
This supports a deadline-sensitive failure, not an intrinsic coordinate loss
from independent origin sampling. It is one selected development case, not a
fifth full corpus panel, independent generalization test or quiet-host timing
measurement.

Reproduce the diagnostic with
`benchmarks/solution_checkpoint_work_probe.py --control RUN/control --case
cod-2108342 --cycles 400 --output NEW_DIRECTORY`. The helper enforces the
registered development split, original native build, count-free input whitelist,
fixed work and prediction-before-reference ordering.

Both experimental controls remain false by default. The earlier apparent speed
gain depended on the rejected stale-champion convergence rule. The corrected
rule supplies a useful guard/audit but has not saved actual phase work here;
frequency-derived origin grids supply a sampling option but no demonstrated
additional recovery. Quiet-host, preferably interleaved paired measurements would
be required for an isolated speed claim. All 35 cases and the original capacity
limits remain in every denominator, and the 65 holdout outcomes remain unopened.

To reproduce a full panel, use `solution_corpus_benchmark.py run --manifest
validation/solution-corpus-v5.json --seconds 15 --wilson-normalization
--phase-quality-selection --phase-origin-selection --chemical-assignment
--iterative-refinement --output NEW_DIRECTORY`. Add `--adaptive-checkpoints`,
`--independent-origin-sampling` or both for the other modes. Normalize with
`solution_corpus_reaudit.py` after collection and compare complete reports with
`solution_phase_ablation.py`. Both initial and corrected collections, including
failed outcomes, are retained separately.
