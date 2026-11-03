# Short early phase starts

The preceding `01b5a20` matrix-reuse improvement reduces summed solve time but
barely affects the typical paired case. Its saved work exists in only eight of
the 35 development cases: 27 have zero matrix reuses. COD 2242589 alone accounts
for 71.18% of that collection's total saving. This is why a useful refinement
kernel improvement did not produce a broad typical-case speedup. The measured
case table and source-report hash are recorded in
`benchmarks/structure-solution-short-phase-diagnosis-macos.json`.

Fresh sequential five-second profiles of COD 2243618, 2011241 and 2011451, all
with zero reuses in that preceding collection, show Fourier arithmetic and
layout as a common substantial cost. The raw profiles, actual native hashes,
image UUIDs and drivers are retained in
`benchmarks/structure-solution-short-phase-profiles-macos.json` and
`runs/post-normal-reuse-profile/`. All preceding fast options were enabled and
no own heavy work overlapped sampling. These representative self samples are
not a full-corpus stage allocation.

An eight-row packed-real last-axis FFT prototype was tested first. It saves
about 2% at 32-cubed and has no useful gain at 64-cubed in the same-executable
64-cycle benchmark. That Rust-only prototype is set aside before Python or
corpus integration. Its source patch and raw measurements are retained as
`benchmarks/structure-solution-packed-z-prototype.patch` and
`benchmarks/structure-solution-packed-z-prototype-kernels-macos.json`. It was not
used for any structure-recovery claim.

## Work allocation

The next experiment changes phase work allocation rather than shaving another
small fraction from an FFT call. The opt-in `short_phase_starts` control is false
by default and requires `work_sliced_phasing`, with its existing phase-first,
phase-origin, phase-quality and non-adaptive-checkpoint dependencies.

Let C be requested cycles, T the maximum number of starts and j a zero-based
start. The existing allocation is ceil(C/2) for the first floor(T/2) starts,
then C for all remaining starts. The short-start policy uses

    early_starts = floor(T/2)
    early_quota = min(C, max(32, ceil(C/4)))
    quota(j) = early_quota if j < early_starts else C.

For the standard C=800,T=4 campaign the quotas change from 400,400,800,800 to
200,200,800,800. At least half the independent starts retain full work; one-start
searches retain all C cycles. The 32-cycle floor preserves the established
quality warm-up, and smaller requested budgets are never increased. Original
ranges C in [1,20000], T in [1,128], and valid j<T are retained. The quota depends
on these public controls alone, with no atom-count, composition, case-ID or
reference-coordinate branch.

The original per-start wall safety ceiling, overall deadline and native resource
caps remain. A short result goes through the same origin/symmetry validation,
count/element search, ambiguity checks and requested-mode final refinement.
Existing target/ambiguity rules determine whether later starts are needed. Later
starts keep their original independent seeds and full-work ceiling. This policy
does not resume an earlier seed at full length; it is a changed bounded allocation
of search work that must be judged by complete development recovery, not by a
low R1 or phase plateau alone.

All Fourier, atomicity, tangent, weak-reflection, normalization and refinement
equations, analytical derivatives, precision, finite support and random-omission
sequences are unchanged. Equal-seed trajectories have identical prefixes when
neither execution hits the wall guard. Their selected endpoint/map and resulting
model can change, so exact final model identity is not an assumption of this
experiment. Default behavior and all preceding execution controls are retained.
The requested flag is exported separately from every actual `phase_slices` quota,
completed-cycle count, elapsed time and stop reason.

## Validation and comparison

The independent Python integer reference in
`python/phasesmith/experimental/solution_schedule_reference.py` implements both
allocations without native calls. Native/Python tests cover small/odd budgets,
single/odd/many starts, invalid controls, the warm-up floor and full-work later
starts. Three complete synthetic P1/P-1/Pm searches check counts, geometry,
anisotropic final mode, unchanged source intensities and bitwise equal phase
prefixes. Full independent value/derivative, support and numerical tests remain
applicable because no physical equation changes. The pinned GSAS-II powder
oracle does not cover this scheduling experiment.

`benchmarks/solution_short_phase_interleaved.py` compares two serial modes on all
35 development cases, differing only in `short_phase_starts`. Both retain the
preceding best options, including matrix reuse, seed zero, four requested starts,
800 requested cycles, native 13-second/outer 15-second limits and final anisotropic
refinement. The changed early quota is the declared treatment, not a hidden equal-
work claim. Seven above-cap references remain in all denominators. Leading mode
alternates 18/17 by case. All 140 predictions precede both reference audit barriers.
Adapter-only seal 30 inherits 29 without changing the registry or the 35/65 split
or reopening raw/reference data. The 65 holdout outcomes remain unopened.
No task-owned tests, profiling, builds or kernel benchmarks overlap timed corpus
collection.

The acceptance check compares the exact previous 19-success case set, total and
per-element counts, strict secondary geometry, and every case's predicted count
vector. Failed or regressed experiments must remain labelled as such. This is
development tuning, not a claim of general unseen-data reliability.

Criterion `short_phase_` measures a complete eight-atom Fe/C/N/O search on a
32-cubed grid and 2,108 expanded rows, with both modes' actual work and outcomes
recorded. It does not reduce this policy comparison to a scalar integer call.

```sh
cargo bench -p phasesmith-single-crystal --features solution --bench solution -- short_phase_ --sample-size 20 --warm-up-time 1 --measurement-time 2
.venv/bin/python benchmarks/solution_short_phase_interleaved.py --output runs/short-phase-new --seconds 15
```

Results are added after complete development collection and auditing.

## Complete development results

The short-start policy retains the exact same **19/35 successful structures**,
**26/35 correct total non-H counts**, **19/35 correct per-element counts**, and
**15/35 strict 0.02-Angstrom structures**. All four successful-case sets match both
the fresh control and the preceding best report: there are no lost or gained
cases under any of those gates. No PhaseSmith hard timeouts occur. All seven
above-cap references remain in the denominator, and no holdout outcomes are opened.

| Measurement | Half-work early starts | Quarter-work early starts | Change |
| --- | ---: | ---: | ---: |
| Summed wall time, all 35 cases | 64.24765 s | 58.27780 s | 9.29% less |
| Median paired control/new ratio | — | 1.10358 | 9.39% less typical time |
| Raw wall-time median | 1.53559 s | 1.27855 s | descriptive, not a paired estimate |
| P1 phase-search stage sum | 29.27494 s | 23.47391 s | 19.82% less |
| Completed Fourier workspace cycles | 49,893 | 39,676 | 20.48% fewer |
| Actual phase starts | 95 | 95 | unchanged |

Thirty-two cases are faster and three slower. This addresses the earlier typical-
case problem: work is reduced in the common phasing path, rather than only in
rare repeated normal-matrix preparations. This is one balanced development
collection; timing remains host-sensitive and does not establish unseen-data
reliability. Only matched-mode ratios from this collection are used, not absolute
time differences between historical builds.

Downstream work stays comparable rather than being assumed identical: isotropic
screenings change 675-to-667, final fits remain 125, and ambiguity relaxations
change 249-to-250. Their stage times are 11.15048-to-10.75918 seconds for screening
and 9.85051-to-9.96399 seconds for final refinement. Thus the phase saving is not
consumed by a large increase in model-validation work.

All 35 total atom counts are unchanged. Per-element vectors change in one
previously unsolved case, COD 2020988: S4 O10 N12 C14 becomes S4 O10 N10 C16.
Both vectors are incorrect; the case remains failed and in every denominator.
Every previously correct total-count and element-count case is retained. Twenty-
four saved model records (including four no-model outcomes) are exactly equal;
eleven records change with the selected map. Those changes are preserved in the
artifacts, and the primary and strict structural gates still retain exactly the
same cases. This experiment therefore does not claim bitwise final-model identity.

The same-executable synthetic full-search median falls from 202.527 to 188.337 ms
(7.01%). Both modes use two starts, eight ambiguity relaxations, 224 assignment
cached steps, four prepared large systems and 52 reuses. Completed Fourier cycles
fall from 192 to 160, as expected for C=128,T=2 (early quotas 64 versus 32). Raw
samples, confidence estimates, executable/benchmark/source hashes and actual
work counters are retained in the kernel artifact.

Artifacts use prefix `benchmarks/structure-solution-short-phase-` followed by
`{control,cache,comparison,diagnosis,kernels,profiles,schedule,retention}-macos.json`.
The retention artifact contains every case's count vectors, exact successful-case
sets, gains/losses, model-record differences, source-report hashes, timing/work
and stage totals, and its verifier source. The schedule audit confirms all 140
predictions before both reference barriers, 18/17 leading cases, actual 200/800
versus 400/800 quotas, matching current 44-file source snapshots, and the same
native library hash for both modes. The rejected packed-Z source patch exactly
reconstructs its measured Rust and benchmark sources from `01b5a20`.

Validation passes: formatting, clippy with warnings denied, all-feature workspace
Rust tests (**419 passed, 34 ignored**), and pytest (**2,487 passed, 11 skipped,
34 deselected**). External oracle and real-data tests retain their normal opt-in
requirements. The execution option remains false by default.
