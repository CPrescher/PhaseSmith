# Chemical assignment and iterative refinement

This follow-up adds two independent opt-in controls to the element-only solver:
`chemical_assignment` (requires `phase_origin_selection`) and
`iterative_refinement`. Both default to false. The input remains element types,
cell, supplied symmetry and HKLF4 measurements; no formula counts or reference
coordinates enter the solver. The original algorithm remains available.

## Chemical proposals

Sheldrick (2015), section 4, describes density calibration using carbon networks,
boron cages and oxyanions before a heaviest-atom fallback:
https://journals.iucr.org/a/issues/2015/01/00/sc5086/ .
PhaseSmith retains its existing heavy/carbon proposals and adds the following
bounded motif proposals independently from those descriptions:

- With oxygen and a heavier allowed element, three or more neighbors at
  0.9–2.3 angstrom, within 12% of one common distance and 20% of one integrated
  mass, support an oxygen-shell proposal. Central mass must exceed the ligand
  mass by 10%. The relative scale is median ligand mass divided by 8.
- With boron supplied, three or more neighbors at 1.65–1.8 angstrom with masses
  within 20% of the center support scale = central mass divided by 5.

Distances use exact closest periodic images, including nonorthogonal cells.
At most 512 full-cell images are inspected. Proposals are sorted by support,
kind and scale; scales within 10% are deduplicated and at most two motifs kept.
Each offers scale factors 0.9,1,1.1, preserving the eight-scale total budget.
These are empirical proposal windows, not universal coordination rules or
chemical correctness tests. They neither force labels nor add unspecified types.
Diffraction ranking decides which candidates receive actual refinement.

After short isotropic refinement, the best two screens can each undergo two
rounds of reassignment. Cache the updated representative coordinates and each
site's Uiso. An orbit column and its fused coordinate jet are both multiplied by
exp(-8*pi²*U_j*s²). The label-dependent sum is then

    F_h = sum_j f(label_j,s_h) exp(-8*pi²*U_j*s_h²) P_j(h).

Profile the positive common amplitude scale and use normalized amplitude SSE,
as documented in `solution-phase-origin.md`. For fixed U_j the coordinate
derivatives retain precisely the same attenuation factor. Independent NumPy
values, jets and finite differences verify this extension.

Try single-element changes and paired swaps among at most 24 sites prioritized
by their best single-change SSE. Unique required types may prevent a single
change; a paired swap can preserve coverage and escape that restriction. Up to
three improving greedy moves are generated. Atom topology/total count remains
fixed, but per-element counts may change. At most two alternatives receive four
isotropic steps seeded with the existing site U values. A refined alternative
must improve the actual screening rank to replace its parent. Final ranking
still uses the original full observations and requested displacement model.
These coupled post-fit changes are PhaseSmith proposals, not a recovered SHELXT
algorithm. The parent is retained when the alternatives fail. Timers include all
attempts; per-setting motif and reassignment counters are exported. Cached
change counts include proposals later rejected by actual refinement; they are
not counts of accepted changes in the final model. The joint post-fit cache and
reassignment work is included in the screening-stage timer.

Similar scatterers can be confounded with amplitude scale and displacement.
In particular, swapping the identities of an isolated two-atom pair can preserve
all amplitudes. A regression test requires such a homometric swap to remain
equal rather than claiming an identifiable improvement.

## Iterative normal equations

The original refiner solves the damped Gauss–Newton system

    (JᵀJ + lambda D) step = -Jᵀr,
    D_ii = max(sum_h J_hi², 1e-8).

The opt-in solver uses conjugate gradients with q=D^(1/2)*step. Its operator is

    A(q) = D^(-1/2) Jᵀ (J D^(-1/2) q) + lambda q,
    b = -D^(-1/2) Jᵀr.

It applies the same existing analytical Jacobian using two streaming products,
without forming or factorizing the dense normal matrix. The Jacobian itself is
retained: this is a matrix-free normal operator, not a Jacobian-free evaluator.
All original Jacobian/atom caps remain unchanged for matched validation.
Fits with fewer than 32 parameters retain the dense solver. Maximum CG work is
min(2P,256) steps; the scaled squared residual target is 1e-16 times its initial
value. At the iteration ceiling an inexact step is accepted only below 1e-10.
Failure increases damping under the existing policy. Coordinate/tensor step
limits, line search, scale profiling and monotonic acceptance are unchanged.
Deadline/cancellation checks run inside CG. Values and derivatives retain their
original fused reflection/image evaluation. Independent dense NumPy solves
check random scaled, correlated systems; a complete multi-parameter refinement
checks accepted steps and objective parity. GSAS-II's powder oracle does not
cover this single-crystal search stage.

## Development-only experiment

Collect all 35 development cases before auditing references, with the same
seed, four starts, 800-cycle ceiling, one thread, 13/15-second budgets, input-only
atom caps and exact count/geometry gates as the preceding campaign. Compare
old phase-origin path, chemical-only, iterative-only and combined controls from
one compiled native build. The 65 holdout outcomes remain unopened. Adapter-only
seal revision 11 preserves the revision-10 data registry and split.

All four collections use native SHA-256
`7ecf121beeea37017937e6d0ff0326671e13eced8528623ecce0aae1fd6cfb82`.
Runtime sources remained unchanged throughout, and all source hashes still match
the saved snapshots. Each collection saves both programs' 35 predictions before
its initial reference audit; the normalized registry re-audits run after all
four collections finish. All gates, observations, caps and requested final ADP
mode remain unchanged. Predictions are not selected using references.

The complete reports are `benchmarks/structure-solution-chemical-{mode}-macos.json`
for modes `control`, `chemistry`, `iterative` and `combined`. Their guarded paired
comparison is `benchmarks/structure-solution-chemical-comparison-macos.json`.

| All 35 development cases | Control | Chemistry | Iterative | Combined |
| --- | ---: | ---: | ---: | ---: |
| Correct labelled structures | 12 | 14 | 12 | 14 |
| Exact total non-H atom counts | 26 | 26 | 26 | 26 |
| Exact per-element full-cell counts | 13 | 15 | 13 | 15 |
| Strict 0.02-angstrom recoveries | 9 | 11 | 9 | 11 |
| Median wall seconds, all outcomes | 4.308 | 4.302 | 3.535 | 3.348 |
| 95th percentile wall seconds | 11.181 | 11.355 | 7.658 | 7.607 |
| Hard timeouts | 0 | 0 | 0 | 0 |

Chemistry and combined gain COD 2242657 and 2243618, with no lost recoveries.
Both fix element assignments at an already correct total count. COD 2242657
has oxygen-shell scale proposals; COD 2243618 has no oxygen/boron motif, but
post-fit reassignment is active and recovers Mg/Cl. These controlled contrasts
support the usefulness of the combined chemistry path; this campaign does not
separately ablate motifs and post-fit reassignment against each other.

The iterative-only path preserves exactly the control's 12 successes. Chemistry
alone adds approximately one second of aggregate screening/cache work without
an overall speed improvement. In the combined path the recorded requested-mode
refinement time drops from 63.895 to 34.825 seconds, despite 104 final-fit attempts
versus 100 in control. Short-screen attempts rise from 146 to 184; combined
performs 19,219 cached reassignment scores, 32 cheap changes and 31 reassignment
fits, with 69 motif proposals. These counts include rejected work, and timers
cover those stages rather than all wall time. Faster fits allow two additional
P1 starts under the same deadlines (95 versus 93).

The combined all-outcome median is 22.3% lower. Among the 12 common successes,
median time changes only from 1.485 to 1.428 seconds. Comparing chemistry with
combined over all 14 common successes gives 1.687 versus 1.517 seconds. These
are single-seed serial collections, not paired repeated performance estimates;
small differences and deadline-dependent paths should not be overinterpreted.

SHELXT in the combined collection recovers 13 structures, 29 total counts and
14 exact element-count sets, at a 0.843-second median and with one hard timeout.
PhaseSmith now exceeds that particular structural/element-count score but is
about four times slower by median. It still receives the exact space group;
this is not equal capability or general superiority. Seven reference-sized
structures remain beyond the unchanged input-derived cap, and all stay in the
denominator. The 65 holdout outcomes remain unopened. Both new controls remain
opt-in pending broader deterministic/seed validation.

Validation passes: 1,601 Python tests (11 external-data skips, 34 deselected),
398 Rust tests (34 ignored), all-feature/all-target Clippy with warnings denied,
Rust formatting, Ruff over Python/tests/benchmarks and `git diff --check`.
The final focused equation/trajectory/report guards also pass. No original
correctness gates or global tolerances were loosened.

The multi-peak microbenchmark uses eight atoms, 2,108 reflections and 69
anisotropic parameters for three refinement steps: dense takes 10.566 ms
(10.497–10.703 ms interval), iterative 9.725 ms (9.699–9.756 ms).
Cached coupled reassignment of eight atoms over 1,054 rows takes approximately
0.164 ms, with a broad 0.149–0.199 ms interval and a high outlier. These are
kernel costs, not repeated end-to-end speed estimates. Full raw samples,
confidence intervals, benchmark/numerical source hashes and executable hash
are retained in `benchmarks/structure-solution-chemical-kernels-macos.json`.


Reproduce with `benchmarks/solution_corpus_benchmark.py run --manifest
validation/solution-corpus-v5.json --seconds 15 --wilson-normalization
--phase-quality-selection --phase-origin-selection --output NEW_DIRECTORY`.
Add `--chemical-assignment`, `--iterative-refinement`, or both for the other
modes. After collection, use `solution_corpus_reaudit.py` for the normalized
reference audit and `solution_phase_ablation.py` for complete-panel comparisons.
