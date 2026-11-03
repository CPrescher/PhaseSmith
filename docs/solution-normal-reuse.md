# Reusing unchanged normal linearizations

This experiment starts from `edbc218`, the guarded peak-statistics path recovering
19 of 35 development structures. `reuse_normal_matrices` is experimental and false
by default. It requires iterative refinement and at least one of the existing
final-fit or isotropic-assignment normal-cache controls. All preceding fast
options, including guarded peak statistics, remain enabled in both comparison
modes.

## Diagnosis

Three sequential five-second native profiles repeat full development solves for
COD 2243618, 2011393 and 2021292 using the preceding best controls. The two
refinement-heavy examples show checked cached-normal solving as their largest
individual collapsed self kernel: 898 and 674 samples. Structure-factor and
analytical derivative evaluation remain substantial. FFT work continues to
dominate the phasing-heavy example, showing that the remaining costs depend on
the case. These profiles are not a full-corpus stage allocation. Exact drivers,
actual native hashes, sampled library UUIDs and collapsed counts are preserved in
`benchmarks/structure-solution-normal-reuse-profiles-macos.json` and
`runs/post-peak-statistics-profile/`. No own heavy work overlapped sampling.

Inspection identifies redundant work in refinement: a rejected/backtracked step
changes damping but leaves the last accepted residual and Jacobian unchanged.
The existing cached-CG solver nevertheless rebuilds the same scaled Gram matrix,
scales and right-hand side on the next damping attempt. Reusing this unchanged
linearization avoids that work without changing the mathematical step.

## Equations and lifetime

J is parameter-major, r is the normalized amplitude residual, and

    D_ii = max(sum_h J_ih^2, 1e-8), S_ii = 1/sqrt(D_ii)
    H = S J J^T S, b = -S J r
    (H + lambda I) y = b, step = S y.

Only lambda changes when the model trial is rejected. H, b, S and the original
right-hand-side norm depend on J and r, so their existing f64 values can be
retained exactly. The implementation separates this preparation from the same
CG solve. It retains the original summation orders, eight-lane dot products,
damping, iteration ceilings, stopping tolerances and scaling. The ordinary
non-reuse wrapper moves its newly prepared RHS into the CG residual as before;
the retained system clones that RHS for each solve because CG mutates its
residual. No factorization, precision change or arithmetic approximation is added.

A cache belongs to one invocation of one constrained fit. It is initialized empty
and is cleared immediately before replacing the accepted evaluation after any
improving backtracking trial. Rejected trials and solver retries preserve it
because they do not replace J or r. Failed evaluations of a trial cannot mutate
the accepted evaluation. Element reassignment, changed data, mode changes and
separate fits always create a fresh cache. No cache is attached to a reusable
public model, shared across candidates, or serialized in an output model.

Every candidate linear step still undergoes the original matrix-free residual
check using the current J, r and damping. A failed cached solve still falls back
to the original matrix-free solver. Input validation, cancellation/deadline
checks in CG, the line search, gradient gate, accepted-step rules and termination
reasons are retained. No solve, residual check or candidate is skipped.

Eligibility remains at least 32 parameters, at least twice as many reflections,
and a Gram matrix within the existing 8-MiB cache budget, after the original
2,000,000-entry Jacobian validation. Smaller cached Cholesky fits and unsuitable
shapes keep their existing execution. Zero RHS keeps its zero-step behavior
without allocating a Gram matrix. No input cap is tightened or loosened.

The retained matrix can remain live during trial evaluation, adding up to the
existing 8-MiB matrix bound plus O(parameters) vectors to peak memory compared
with dropping it before backtracking. At most one such system is retained per
fit; accepting a model drops it before the next preparation. It is released when
the fit returns. No dependency, numerical thread or full-grid workspace is added.

The requested option and actual execution are distinct. `normal_matrix_builds`
counts completed preparations in the optional reuse path; `normal_matrix_reuses`
counts subsequent attempts using those retained preparations. Both are zero when
the option is disabled, and do not count the existing non-reuse path's ordinary
matrix construction. Existing successful cached-step and fallback counters remain
independent. Fit-level and whole-search diagnostics expose the new counters.

## Independent checks

Rust tests compare original and reused cached steps bitwise over damping changes,
small/unsupported and eligible shapes, and verify explicit clearing before new
data. They cover zero RHS, invalid inputs, already-stopped systems and cancellation
on an existing retained cache. Six 40-step isotropic/anisotropic fits across P1,
P-1 and Pm exercise actual rejected damping attempts and accepted-state
invalidation. Parameters, every evaluation field, accepted steps and termination
match bitwise, and repeated API calls reproduce fresh-cache counts.

The independent Python equations in
`python/phasesmith/experimental/solution_reference.py` and
`solution_origin_reference.py` remain unchanged: this alters cache lifetime,
not the equations. New tests compare fit values with that independent reference
and analytical Jacobian rows with centered finite differences. Existing cached
linear-step tests compare the same solver with independent NumPy damped systems,
including rank deficiency and tiny columns. Three complete synthetic element-only
searches check exact models, correct counts, final anisotropic mode, requested
controls and invalid dependencies.

No new physical term, derivative, finite-support rule, data normalization, count
proposal, assignment alternative or ranking gate is introduced. The pinned
GSAS-II powder-profile oracle does not cover this single-crystal optimization;
no GSAS-II or decompiled external implementation code is used.

## Benchmark protocol

The balanced serial runner `benchmarks/solution_normal_reuse_interleaved.py`
collects all 35 development cases with only `reuse_normal_matrices` differing.
It retains seed zero, four starts, 800 requested cycles, native 13-second/outer
15-second limits, requested anisotropic final fits and all previous fast flags.
Leading mode alternates by case, 18/17. All 140 predictions precede both reference
audit barriers. The seven above-cap references remain in every denominator.
Adapter-only seal 29 inherits revision 28 without changing the registry or 35/65
split or reopening raw/reference data. The 65 holdout outcomes remain closed.
No task-owned tests, builds, profiling or kernel benchmarks overlap timed corpus
collection.

Criterion `normal_reuse_` cases include a complete eight-atom, 2,108-row constrained
40-step fit and a full eight-atom Fe/C/N/O search on a 32-cubed grid. Both modes
use the existing checked normal solver. Actual accepted steps, cached steps,
preparations, reuses, phase starts and cycles are printed in the raw log. All
comparisons use modes within one executable, not historical build-to-build deltas.

```sh
cargo bench -p phasesmith-single-crystal --features solution --bench solution -- normal_reuse_ --sample-size 20 --warm-up-time 1 --measurement-time 2
.venv/bin/python benchmarks/solution_normal_reuse_interleaved.py --output runs/normal-reuse-new --seconds 15
```

Complete development results are recorded after prediction collection, normalized
reference auditing and exact comparison with the previous 19 successful cases.

## Complete development results

Both modes recover the exact same **19/35 structures**, **26/35 total non-H
counts**, **19/35 per-element counts**, and **15/35 strict 0.02-Angstrom structures**.
All four case sets match the preceding best report as well as the fresh control.
Every inferred atom-count vector is unchanged across all 35 cases. All 31 returned
model records, including coordinates and reported R1, are exactly identical
between modes; the four no-model outcomes match too. There are no PhaseSmith
hard timeouts, and all seven above-cap references remain in the denominator.
The holdout remains unopened.

| Measurement | Control | Matrix reuse | Change |
| --- | ---: | ---: | ---: |
| Summed wall time, all 35 outcomes | 69.78278 s | 67.43709 s | 3.36% less |
| Paired median control/new time ratio | — | 1.00539 | about 0.54% less typical time |
| Raw wall-time median | 1.76577 s | 1.58776 s | descriptive, not a paired estimate |
| Isotropic screening stage sum | 12.11471 s | 11.67770 s | 3.61% less |
| Requested-mode final refinement sum | 11.99769 s | 10.31851 s | 14.00% less |
| P1 phase-search stage sum | 30.33859 s | 30.34123 s | unchanged |

Twenty-six cases are faster and nine slower. The paired typical-case improvement
is small; the larger change in the two unpaired medians should not be advertised
as a typical 10% gain. This is one balanced paired collection on the current host.
The stage improvement and matrix-reuse counts substantiate the targeted saving,
while timings remain host-sensitive. The P1 FFT-dominated cost is unaffected.

Both modes perform 95 phase starts, 675 isotropic screens, 125 final fits, 249
ambiguity relaxations and 10,305 cheap ambiguity scores. Successful assignment
and final cached steps remain 4,640 and 2,460, without normal-cache fallback.
The reuse path makes 5,333 preparations and reuses them on 882 subsequent
attempts. Thus 882 of 6,215 eligible preparation attempts are avoided (14.19%),
while every linear step and original-operator residual check still occurs.
Smaller cached Cholesky steps are outside those new counters. The disabled
control reports zero for the new reuse-path counters, as documented above.

Existing per-start wall-clock safeguards yield minor phasing variation in three
cases: 49,676 versus 49,692 completed Fourier workspace cycles. No scheduling
rule, cycle quota or cap changes, and all final model records remain identical.

Same-executable Criterion medians fall from 63.272 to 55.172 ms (12.80%) for the
complete 40-step, eight-atom/2,108-row fit, and from 217.428 to 211.658 ms (2.65%)
for the full synthetic search. The fit retains 28 accepted steps and 40 cached
solves. The synthetic search retains two starts, 192 Fourier workspace cycles,
eight ambiguity relaxations and 224 assignment cached steps, with four prepared
large systems and 52 reuses in the enabled mode. Raw samples, confidence
estimates and actual work counters are retained in the kernel artifact.

Artifacts use prefix `benchmarks/structure-solution-normal-reuse-` followed by
`{control,cache,comparison,kernels,profiles,schedule,retention}-macos.json`.
The retention artifact includes all 35 count vectors, exact structure/count/strict
case sets, source-report hashes, work/stage totals, timing and cycle variations,
and its verifier source. The schedule audit verifies all 140 predictions before
both reference barriers, alternating 18/17 leading cases, matching current
44-file source snapshots, and the same native library hash for both modes.

Final checks pass: formatting, clippy with warnings denied, all-feature workspace
Rust tests (**418 passed, 34 ignored**), and pytest (**2,406 passed, 11 skipped,
34 deselected**). The disk filled during an intermediate lint invocation;
rebuildable debug/incremental caches from this checkout were removed and the
lint check then passed before benchmark collection. Release benchmark artifacts
and source/data were retained. External oracle and real-data tests retain their
normal opt-in requirements. The new execution option stays false by default.
