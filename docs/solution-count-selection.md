# Development count-selection follow-up

## Diagnosis

The frozen 35-case screen is the development set; the 65-case holdout remains
unopened. Predictions, source intensities, cells, supplied element types/symmetry,
input-derived atom ceilings, seed 0, 15-second outer limit, 13-second internal
limit, correctness gates and one-thread execution are preserved.

The baseline returns the right total non-H count in 9/35 cases, the exact
per-element counts in 6/35, and the correct element-labelled coordinates in 5/35.
Those are different tests. Seven true structures exceed the input-derived dense
model ceiling; a count-selection change cannot recover them under that ceiling.

Most other count failures are incomplete high-R1 models, not subtle disagreements
between nearly equivalent complete structures. Almost every unsolved case spends
its time on the first phase start. At the early four-cycle handoff, a small strong
framework monopolizes constrained fits. A single density batch adds at most eight
orbits once, followed by one-orbit births/deletions/relabels. Hundreds of isotropic
and dozens of anisotropic fits often exhaust the budget before enough weak atoms
are included. An absolute integrated-mass cutoff can also discard weak candidates
while the strong-atom scale is still provisional.

## Implemented policy

`solve_structure_elements(..., adaptive_counts=True)` now:

1. Retains the original initial density-batch procedure.
2. Revisits the difference map after the current framework is refined. Its peaks
   and supplied-symmetry recycled peaks generate symmetry-compatible whole-orbit
   candidates without atom-count quantities or reference coordinates.
3. Tests one- and two-orbit prefixes together, screening both isotropically and
   promoting the preferred prefix to the requested final ADP mode. A promoted
   improvement advances a complete count step before ordinary single-orbit edits.
   If no batch improves the score, the original deletion/relabel/birth round runs.
4. Treats the integrated-density range as a guide during these residual proposals:
   out-of-range peaks receive a diffraction-based alternative label instead of
   unconditional rejection. Required elements, overlap checks, existing resource
   limits and final requested-mode score comparisons still apply.
5. Limits poor early phase-map handoffs to two seconds before cycle 64 and four
   seconds thereafter, within the unchanged total deadline. An atomic model below
   R1=0.30 receives the remaining global budget. These are explicit scheduling
   heuristics, not correctness tests. Full requested-mode candidates and the
   ordinary stability/required-element gate are still required for a target stop.

The new policy also retains up to `max(32, maximum_atoms)` recycled peaks rather
than a universal 32-peak cap. Existing peak-height, duplicate-image, 0.5 Å
non-overlap and 128-atom bounds remain unchanged. The per-setting `batch_orbits`
diagnostic describes its most recent proposal; accumulated
`count_prefix_screenings`, `bounded_map_handoffs` and accepted lineage moves
record the additional work.

`adaptive_counts=False` preserves the original scheduling, peak caps, mass filter
and one-time batch. The feature is independent of composition input: none is
accepted by the element-only solver. Count editing depends on symmetry recycling;
local map budgets apply to checkpointed dual-space phasing. Default powder builds
and fixed-composition solves are unaffected.

For all count alternatives the existing profiled objective is

\[
 k={\sum_h F_{o,h}|F_{c,h}|\over\sum_h |F_{c,h}|^2},\qquad
 J={\sum_h(k|F_{c,h}|-F_{o,h})^2\over\sum_h F_{o,h}^2}.
\]

The unchanged descriptive score is

\[
 S=N_f\log\max(J,10^{-300})+(p+1)\log N_f,
\]

where `N_f` is the independent symmetry/Friedel family count, `p` the actual
symmetry-constrained coordinate/ADP parameter count, and one scale is added.
Count prefixes compete under the same isotropic mode before final refinement;
only full requested-mode states enter the retained beam. This BIC-style score
is not a calibrated posterior probability for correlated crystallographic
residuals. It cannot certify correct composition or geometry.

The numerical value/analytic-derivative kernels are unchanged. Existing fused
orbit, coordinate, U/ADP, Fourier and periodic-frame checks remain applicable.
The new independent NumPy score reference checks native retained-state scores,
and synthetic P1, inversion and mirror cases test both policy settings without
input counts. No external oracle fixture is regenerated: GSAS-II's powder oracle
does not cover count-free phasing, while SHELXT remains an external comparison.

The published [SHELXT method, section 4](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html)
motivates interpreting integrated density after supplied-symmetry recycling.
The residual-prefix and budget policy here is an independent bounded design.

## Comparison protocol and results

The full campaign reruns all 35 development cases in both programs, serially,
with alternating program order and the same pinned executable and observation
hashes. References are audited only after all 70 predictions are saved. The
previous frozen baseline remains unchanged; this is a new collection, not paired
repeated timing. All unsuccessful, capacity-limited and timed-out outcomes stay
in the denominator. The complete source snapshot and installed-native hash are
part of each collection.

The contemporaneous ablation uses the **same native binary**, disabling only
`adaptive_counts` in the legacy run. Each policy and SHELXT was evaluated on all
35 cases. These are serial collections rather than paired repeated timings.

| Measure | Legacy policy, current build | Adaptive policy | SHELXT, adaptive campaign |
| --- | ---: | ---: | ---: |
| Correct total non-H count | 7/35 | **13/35** | 29/35 |
| Exact per-element non-H counts | 5/35 | **7/35** | 14/35 |
| Correct labelled coordinates and counts | 4/35 | **7/35** | 13/35 |
| Maximum coordinate error ≤0.02 Å | 4/35 | **7/35** | 10/35 |
| Median time, all outcomes | 13.357 s | 13.390 s | 1.762 s |
| 95th-percentile time, all outcomes | 13.473 s | 13.575 s | 3.425 s |
| Hard timeouts | 0 | 0 | 1 |

The new policy adds `cod-4326074`, `cod-2011038` and `cod-2011451` to the
contemporaneous legacy successes and loses none. Relative to the **original
frozen screen**, the change is 9→13 correct totals and 5→7 correct structures;
all five originally recovered structures remain recovered. The current legacy
control misses one originally successful, budget-sensitive case. Both older
records and the new controls remain available; they are not interchangeable.

The two additional structures over the original screen are Rb–Zn–As (Rb1 Zn4 As3,
correcting Rb2 Zn4 As2) and Ba–Bi–K–O (Ba16 Bi12 K4 O4, correcting O12). The
former illustrates a wrong per-element count despite the correct total and low
R1; the latter illustrates overcounting a weak species around a heavy framework.

**There is no overall speed improvement.** The all-outcome medians are nearly
identical because most unsuccessful searches consume the budget. On the four
structures recovered by both contemporary policies, median time changes from
1.074 to 1.791 s: an overhead on already solved cases. The expanded peak proposals
and repeated count screens cost time. Some difficult cases benefit: the Ni–Cl–C–N
case is recovered in the adaptive campaign while the contemporary legacy run
remains incomplete at its limit. This is not a matched successful-solve speedup.

Contemporary undercounts fall from 26 to 20 (24→20 relative to the original
screen). There remains one overcount and one no-model case. Six adaptive cases
have the right total but wrong element counts. The seven capacity-exceeding
references remain in the primary denominator; among the 28 within capacity,
structure recovery is 4→7 for the contemporary policies. General SHELXT parity
is not established. No random-seed robustness claim is made for the new panel.

SHELXT recovers 13 structures and 29 total counts in both contemporary campaigns;
its median changes from 1.150 to 1.762 s. Together with the older 0.889-s record,
this illustrates host/timing variability. Timings are observations, not a precise
hardware-normalized ranking.

Committed records:

- [Complete adaptive campaign](../benchmarks/structure-solution-count-adaptive-macos.json).
- [Complete same-build legacy campaign](../benchmarks/structure-solution-count-legacy-macos.json).
- [Same-build per-case comparison](../benchmarks/structure-solution-count-policy-macos.json).
- [Comparison with original frozen screen](../benchmarks/structure-solution-count-historical-macos.json).

Each complete report retains all 70 program jobs, models/failures, actual
collection source hashes and source snapshots. Final validation adds a guard
respecting `integrated_peak_density=False` and equivalent lint-only condition
spelling/binding names. Both timed policies used density guidance **enabled**, so
the guard does not change their configured numerical path. The final-build replay
of all seven already recovered cases is recorded separately, with both native
hashes; it is a regression replay, not an additional 35-case screen.

Intermediate development experiments are retained locally. A first local-budget
trial cut off a previously solved model; that motivated the promising-model
continuation rule. An oversized 32-orbit batch variant exhausted screening budgets
and was interrupted rather than graded as a complete campaign. The final policy
uses two-orbit residual steps and preserves the original initial batch. These are
development decisions; no held-out outcomes informed them.

## Cores and timing interpretation

PhaseSmith's current experimental solver runs phase starts, FFTs, count edits and
refinements serially. Releasing Python's interpreter lock permits another caller
to do work but does not parallelize this solve. The numerical crate has no Rayon
or worker-thread search. Both benchmark runs limit numerical environment threads
to one and launch SHELXT with `-t1`; its output confirms one thread.

SHELXT itself supports multicore work. Its published examples describe four
Patterson starts in parallel on a four-core processor, and its flowchart shows
parallel solution, symmetry and refinement stages. Native PhaseSmith arm64
versus x86_64 SHELXT through the host compatibility runtime remains an architectural
qualification. A multicore comparison would be a separately declared experiment.

## Reproduce

Build the release `experimental-solution` feature, then use a new directory (the current adapter seal is revision 5):

```sh
.venv/bin/python benchmarks/solution_corpus_benchmark.py run \
  --manifest validation/solution-corpus-v5.json \
  --output runs/count-selection/new-development --seconds 15
.venv/bin/python benchmarks/solution_corpus_reaudit.py \
  --manifest validation/solution-corpus-v5.json \
  --run runs/count-selection/new-development \
  --output runs/count-selection/new-development/comparison-verified.json
```

For the explicit same-build legacy ablation, add `--legacy-counts` to `run` and
choose another new output directory. `benchmarks/solution_count_comparison.py`
compares complete reports, refusing changed panels, reference registries, timing
budgets, score policies, executable identity, thread policy or resource ceilings.
When both native hashes match, it checks that the policy is disabled/enabled in
all 35 respective native records.

The corpus/input/reference bytes remain unchanged. Adapter seal revision 5 adds
only the explicit legacy-policy flag and records it for reproducibility/resume;
older immutable seals remain available. The holdout guard remains active; these
commands select development cases only.

## Validation

The complete suite passes **1,497 Python tests** (11 external-data skips and
34 normal marker deselections), **394 Rust tests** (34 opt-in ignores), formatting,
strict all-target/all-feature Clippy, repository Ruff and whitespace checks.
Twelve new count-policy/reference tests cover both settings, independent score
agreement, three symmetry types, invalid flags and disabled density guidance.

The final-build replay recovers all seven previously successful development
structures ([record](../benchmarks/structure-solution-count-final-replay-macos.json)).
An additional seed-0 regression on the previously exposed K[BrF4], BiSeI and
thiourea examples recovers all three, with and without tangent refinement, correct
counts/coordinates and both CIF-frame fit checks
([record](../benchmarks/structure-solution-count-original-three-macos.json)). These
six solves are regression checks, not additions to the 35-case panel.

The realistic full-search benchmark uses eight atoms and 2,108 reflection rows.
Ten Criterion samples give 2.002 s for legacy and 1.763 s for adaptive search
([estimates](../benchmarks/structure-solution-count-hotloop-macos.json)). Legacy
hits the two-second search ceiling, so this is a bounded synthetic workload
measurement, not a successful-structure speedup claim. Experimental timing
conclusions use the complete development campaigns above.
