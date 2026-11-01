# Three-stage structure-search recovery

This follows the [candidate fitting and remaining-case diagnostics](candidate-fitting-speed.md).
The same exposed 14-case comparison is replayed, with unchanged numerical HKL
values, cells, element lists and matched Laue/centring priors. Exact group, atom
counts and coordinates remain withheld. This is development/regression evidence.

The [frozen policy](https://github.com/CPrescher/PhaseSmith/blob/main/validation/three-step-recovery-policy-v1.json) precedes the
first revised numerical run. Baseline and final use fast-three, auto-one,
SHELXT-four-one and SHELXT-ten-three repetitions per case; intermediate stages use
the same native repetitions. All failures are retained. Every stage saves all
predictions before reference grading. Timing collection is sequential and never
overlaps builds, tests, profiling or other numerical collection.

## Bounded tensor and element recovery

The native coordinator uses an explicit `ElementSearchPolicy`. Original solution
entry points and named Python profiles retain their historical defaults. In the full fallback pass, an
isotropic parent with all required elements, validated recycling, no assignment
disfavour and R1 <=0.18 can undergo final
anisotropic polishing and existing coupled categorical repair. It shares the
original deadline and is capped at two additional seconds and 20 iterations per
fit. Successful fast searches bypass this extra work. Within full effort, promising
models are treated equally even when one group already meets its own target;
otherwise different displacement classes could bias group selection.
For these promoted models the categorical repair can run below its legacy 0.04
R1 work cutoff: a low residual after adding tensor flexibility must not suppress
the requested element-reassignment attempt. This changes work eligibility only.

The tensor model and fused analytical derivatives are unchanged:

```text
F(h) = sum_j f_j(h) exp(-2 pi² q^T U_j q) exp(2 pi i h.x_j)
q = B h,  U_image = Q U_site Q^T
U_site = mean_images(Q^T U_image Q).
score = n log(normalized amplitude SSE) + p log(n)
```

U is Cartesian, in angstrom squared. Existing positive-definite tensor
parameterization and eigenvalue limits remain unchanged. Values and Jacobians
still accumulate in one reflection/image pass; no support convention changes.
The original parent remains available. An improvement must survive the existing
penalized-score and replacement-supported rules; lower R1 alone is insufficient.
Full common-data scoring now uses the candidate's actual displacement mode and
actual coordinate/tensor/scale parameter count. Independent-family admission and
all acceptance checks continue to apply. The tensor pullback uses image source
identities and validates topology/finiteness before existing seed checks.

## Centred intensity groups

For an intensity processing group, let T be its exact pure-translation subgroup.
Permit a symmorphic intensity group only when each operation translation lies in
T. Closure then supplies a zero-translation representative for every rotation.
This admits independently supplied centring and still excludes unknown screw/glide
translations. The existing support/preparation kernels retain exact centring
extinctions; unmeasured rows never become measured zeros. Common scoring still
includes every selected genuinely measured family, including forbidden rows.

## Candidate frames and consensus

A saved refinement declares aligned x = hand * input x + shift (modulo one).
Validate that frame with the original same-element bijective symmetry matching
and projection tolerance before retrying arbitrary-origin recovery. A recorded
frame is a proposal, not permission to skip validation. Retain rejected models
and frame diagnostics for audit. Native consensus stopping requires R1 no larger
than both its historical 0.1 ceiling and the requested target; acceptance itself
is unchanged.

The initial frame-first replay exposed a selection regression: an acentric model
with lower residual displaced a correct centrosymmetric approximation, while the
unchanged inversion-consistency gate still kept that acentric model unaccepted.
An unrestricted preference for inversion compatibility then exposed a second
regression: a poor centrosymmetric approximation displaced a better atomic
framework. The final [selection policy](https://github.com/CPrescher/PhaseSmith/blob/main/validation/three-step-recovery-policy-v3.json)
requires that a preferred compatible model also satisfy the existing 0.18
promising-model R1 work ceiling on the full measured common data. Otherwise
score ordering applies between implausible/phase-blocked models. This ceiling
controls selection preference, never acceptance. It does not establish atomic
centrosymmetry. All attempts remain visible, and lower-scoring alternatives
remain in ambiguity assessment. Both earlier regressing runs are preserved.

The primary aim is reliable recovery, using SHELXT as a reference. Runtime is a
measured cost and tradeoff; outperforming SHELXT is not a requirement.

## Interpreting the comparison

Both programs receive the cell, non-hydrogen element **types**, measured HKLF4
intensities and matched Laue/centring priors. Neither receives the published atom
counts, atomic coordinates or exact space group as solution input. SHELXT's UNIT
line is the same dummy one-per-type input as in the frozen earlier comparison.
Reference CIFs are used only after every worker in a stage has exited and the
prediction barrier has been saved. Grading evaluates the retained model, without
choosing another candidate using the reference.

The primary labelled-structure test requires exact element counts, one common
origin/inversion, RMS position error <=0.025 angstrom and maximum error <=0.05
angstrom. Published group number is checked separately; the stricter position
diagnostic uses a 0.02 angstrom maximum. Position-only matching is secondary and
cannot turn an incorrectly labelled framework into a solved structure.

Native acceptance is stricter than reference-based coordinate recovery. Its
own-group R1 <=0.06, full common-set R1 <=0.08, origin alpha <=0.3, independent
family/parameter admission, measured symmetry contradictions, unresolved higher
Laue symmetry, inversion evidence and close group alternatives remain unchanged.
The acceptance source file is byte-identical to the baseline; its SHA256 is in
the frozen policy. A correct but unaccepted model remains an approximation.

Times include process startup and output. Each fast and ten-worker SHELXT case
uses the median of three runs; auto and four-worker SHELXT have one run per
stage. Case medians are then summarized across all 14 cases, including failures.
Stages are sequential, not randomized/interleaved performance trials. Native
runs use four ordered P1 workers and arm64 code; the pinned SHELXT 2018/2 runs
under Rosetta with four or ten workers on the same Apple M4 Pro. These platform,
sampling and exposed-data limits prevent a general solver ranking.

## Measured results

The final fast and auto outputs recover **12/14 labelled structures**, versus
11/14 in the fresh baseline, and **11/14 with the published group**, versus 10/14.
No previously correct labelled model is lost. Position-only matching passes all
14 frameworks; nickel and lithium still have incorrect element labels. Thirteen
group numbers match, but that count alone does not establish a structure.

Automatic effort accepts **5/14**, versus 4/14 before. All five accepted models
pass both labelled and group grading. The additional accepted case is COD 2203173,
the rhombohedral Sc/O/C structure (R-3c, Sc6 O36 C18): maximum/RMS source-position
errors are 0.002313/0.001654 angstrom and own R1 is 0.0382501. Nine cases still
escalate naturally; one equivalent-group ambiguity remains explicitly deferred.

| Stage | Fast median, s | Auto median, s | Auto total, s | Auto labelled | Auto labelled + group | Auto accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh baseline | 1.083 | 8.040 | 117.019 | 11/14 | 10/14 | 4/14 |
| 1. Tensor/element recovery | 1.079 | 8.403 | 120.565 | 11/14 | 10/14 | 4/14 |
| 2. Centred intensity maps | 0.938 | 8.590 | 112.594 | 11/14 | 10/14 | 5/14 |
| Initial frame-first regression | 0.939 | 8.565 | 116.905 | 10/14 | 9/14 | 5/14 |
| Unrestricted inversion-preference regression | 0.932 | 8.431 | 115.140 | 11/14 | 11/14 | 5/14 |
| 3. Final frames, stopping and selection | 0.944 | 8.533 | 116.802 | 12/14 | 11/14 | 5/14 |

Stage 1 retains every baseline candidate, CIF, R1, map and acceptance flag
bitwise. Its added tensor work costs time without changing retained outcomes.
Centred processing enables the additional Sc/O/C acceptance. The final selection
recovers chromium, while stored-frame validation also makes the lithium framework
available in the normal budget. The first frame-first run lost the correct
Tl/Mg/Cl model to an ambiguous acentric alternative. The unrestricted inversion
preference restored that model but lost the correct Hg/I framework to a poor
centric model. Both completed regressions are retained; the final plausibility
qualification restores Hg/I as well. No accepted wrong model occurs in any stage.

These are fixes developed on exposed cases, not independent generalization tests.
No reference-based candidate selection or acceptance-threshold adjustment is used.

### Same final comparison

`L+G` means the labelled structure and published group pass; `L` means only the
labelled structure passes; `fail` means the labelled structure fails. These are
reference grading outcomes, separate from PhaseSmith acceptance. Native fast and
auto have the same per-case L/G outcomes here. Times are seconds.

| COD | Native fast | Native auto | SHELXT four | SHELXT ten |
| --- | ---: | ---: | ---: | ---: |
| 2200381 | 1.842 / L+G | 17.917 / L+G | 1.762 / L+G | 1.331 / L+G |
| 2215687 | 0.262 / L+G | 0.270 / L+G | 0.434 / fail | 0.480 / fail |
| 2243618 | 1.085 / L+G | 17.714 / L+G | 2.756 / L+G | 0.901 / L+G |
| 2017324 | 0.452 / L+G | 0.452 / L+G | 0.452 / L+G | 0.450 / L+G |
| 1549038 | 0.835 / L+G | 13.472 / L+G | 0.437 / fail | 0.446 / fail |
| 2010875 | 2.335 / fail | 24.195 / fail | 0.455 / fail | 0.660 / fail |
| 2241300 | 0.665 / L+G | 0.657 / L+G | 0.870 / L+G | 0.451 / L+G |
| 2021040 | 3.296 / L+G | 3.279 / L+G | 1.272 / fail | 0.877 / fail |
| 2207083 | 0.241 / L+G | 2.644 / L+G | 0.453 / L | 0.232 / fail |
| 2233659 | 0.505 / L | 9.001 / L | 0.458 / L+G | 0.456 / L+G |
| 2011241 | 1.469 / L+G | 8.611 / L+G | 0.238 / L+G | 0.449 / L+G |
| 2010993 | 0.343 / L+G | 0.347 / L+G | 0.240 / L+G | 0.236 / L+G |
| 2203173 | 1.197 / L+G | 8.454 / L+G | 0.451 / fail | 0.668 / fail |
| 2201295 | 1.054 / fail | 9.789 / fail | 0.882 / L+G | 0.868 / L+G |

Final median times are 0.944 s fast, 8.533 s auto, 0.454 s SHELXT-four and
0.468 s SHELXT-ten. SHELXT-four recovers 9 labelled structures, 8 with their
published group; SHELXT-ten recovers 8 with their groups. Native fast CIFs are
bitwise repeatable across all three repetitions; SHELXT-ten L/G outcomes also
repeat across its three runs on these cases. These small exposed-set scores do
not establish that one solver is generally better.

### Remaining cases and deeper controls

| Case | Final normal auto result | Six-run deeper replay |
| --- | --- | --- |
| 2011241, Cr/O/N/C | Correct P-1, Cr2 O8 N4 C10; max error 0.012395 angstrom; R1 0.0852244; unaccepted | Both seeds correct, max 0.012319 angstrom; tensor R1 0.0675929 still above the unchanged own-target gate |
| 2010875, Ni/O/N/C | Correct P21 framework, incorrect C/N/O counts; unaccepted | Both seeds retain frameworks but incorrect labels |
| 2201295, Li/O/N/C | Correct P3121 framework, Li6 O30 N12 C18 instead of Li6 O30 N6 C24; unaccepted | Both seeds retain the same C/N error, R1 0.0871950 |
| 2233659, Hg/I | Correct Hg4 I4 positions, group 121 instead of 139; R1 about 0.225; unaccepted | Not part of the previously frozen six-job diagnostic |

The six deeper jobs use exactly the earlier packets: two seeds per Ni, Cr and Li
case, eight workers, eight starts of 1,600 cycles and unchanged 30 s candidate /
180 s total / 210 s process limits. All six finish without process timeout. Two
now pass labelled and group grading (chromium); none is accepted. Total time is
147.076 s, versus 129.444 s in the earlier diagnostic. This is
additional recovery at a higher cost, not a speed improvement. Neither nickel
nor lithium is solved completely at these budgets.

Example full-pass tensor proposals explain why lower R1 alone is insufficient.
Scores below use the actual fitted parameter count; a smaller score is better.
Parameter counts exclude the separable scale, which common scoring also counts.

| Model | Parent R1 → tensor R1 | Parent score → tensor score | Tensor parameters | Outcome |
| --- | ---: | ---: | ---: | --- |
| Ni, P21 | 0.09292 → 0.07280 | -9871.7 → -9390.0 | 314 | Not adopted; 20-step limit |
| Cr, P-1 | 0.08522 → 0.06759 | -8333.3 → -8396.4 | 108 | Adopted within the full-pass group fit; gradient stop |
| Li, P3121 | 0.08720 → 0.06770 | -3720.3 → -3657.6 | 100 | Not adopted; gradient stop |

None of these example proposals hits an ADP eigenvalue bound. Nickel reaches the
iteration ceiling; chromium and lithium report the gradient criterion. The
previously documented ADP-bound limitation is not an explanation for these
specific proposals. Across normal auto runs, 45 bounded polishing calls take
3.997 s and 20 improve their group-local parent; this does not mean 20 new
structures are accepted.

### Where the normal auto time goes

These are totals over the fourteen searches. Parallel time is wall time inside
its owning stage; the rows below are disjoint. Independent phasing, assignment
and nonlinear fitting remain combined inside model runs.

| Area | Seconds |
| --- | ---: |
| Shared phase maps | 5.746 |
| Phase-origin screening | 7.862 |
| Model runs: independent phasing, assignment and fitting | 101.978 |
| Common assessment and attempt coordination | 0.192 |
| Other search coordination | 0.042 |
| Process startup, output and other host work | 0.982 |
| **Total** | **116.802** |

The 3.997 s polishing cost is already inside model runs. Final model runs use
308 independent starts, versus 360 before; stored frames validate successfully
102 times and four common-score rejections retain their candidate diagnostics.
Common assessment/attempt coordination falls from 0.860 to 0.192 s, but it is
a small part of the total. The paired ten-atom / 4,168-row common-score benchmark
measures 1.2793 ms with arbitrary-origin recovery versus 1.1691 ms with a validated
frame (8.6% lower). Preparation plus eight ordinary isotropic fitting iterations
is 7.4939 ms in the same benchmark collection.

The fast median falls by 12.8%. Overall auto time is essentially unchanged:
117.019 → 116.802 s summed (0.2% lower), while its median rises 6.1%. There is
**no meaningful overall auto speed gain** from this revision. Most remaining time
is in phasing/assignment/fitting; speeding the small common-score stage further
would have little overall effect. Reliability, rather than a race with SHELXT,
is the reason to keep the changes.

### MgSiO3 and forced fallbacks

All eight controls pass labelled/group grading and their expected effort path.
Three normal MgSiO3 runs with matched priors take 0.507–0.514 s; three without
Laue/centring priors take 0.505–0.515 s. All six accept Pbca, Mg16 Si16 O48, retain
R1 **0.0514043045582282**, and preserve their previous candidate CIF, density and
phase-map arrays bitwise.

Forced fast-timeout controls perform all four full-resolution 800-cycle starts
and select an accepted thorough result: MgSiO3 in 20.727 s, R1 0.0416448, and
acentric COD 2021040 in 27.132 s, R1 0.0418996. Their full-pass candidate CIFs
change with the new polishing policy; maximum source errors are 0.007030 and
0.012824 angstrom, respectively. Their density and phase-map arrays remain
bitwise unchanged. This is labelled recovery and accepted fallback validation,
not a claim of identical internal refinement trajectories. These controls reuse
saved reduced intensities; this revision does not rerun original-frame reduction.

## Validation

These changes reuse existing structure-factor/ADP equations and independent
Python derivative references. New references/tests cover tensor pullback,
penalized replacement, lattice-translation eligibility, frame matching and
stopping policy. Existing finite-difference and invalid-input checks remain.
GSAS-II's powder-profile oracle does not cover this single-crystal coordinator.
The realistic candidate benchmark and staged corpus runs measure the added work.

Local validation passes 561 workspace Rust tests (34 existing ignored tests),
four native-host tests, two focused release-mode frame tests, strict all-feature
Clippy and Rust documentation, and 3,330 installed-wheel Python tests. Eleven
external-data Python tests remain skipped and 34 slow tests are deselected by
the unchanged default marker policy. Rust 1.89 all-feature and minimal facade
checks pass. The ordinary source distribution rebuilds and passes the installed
NumPy-only capability smoke with optional numerical imports blocked; the complete
frozen native source archive also passes its all-feature facade check. No runtime
dependencies are added. API, version, reference-doc, skill, formatting and strict
site checks accompany the machine-readable receipt.

The compact [results receipt](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/three-step-recovery-v1-macos.json)
pins policies, binaries, source archives, all prediction barriers, grading,
repeatability and controls. One initial partial polishing run overlapped source
archive compression and was stopped; its output and exclusion reason are kept,
and the identical frozen inputs were rerun before grading. Neither completed
selection regression is excluded from the report.

Complete data, source snapshots, executables, models, barriers and logs stay
outside Git at
`/Users/clemens/ChatGPT/PhaseSmith/three-step-search-recovery-0.8-20261101/`.
After source reconstruction and retained-artifact hash checks, the two regenerable
Rust target directories were removed: 11,117,424,640 allocated bytes (10.35 GiB).
Physical free-space gain was not measured. All original data, numerical evidence,
failed trials, source archives, executables, wheels and benchmark estimates remain;
other worktrees and shared dependency caches were not cleaned.
