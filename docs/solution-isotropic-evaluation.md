# Specialized canonical isotropic evaluation

This experiment starts from `e9cf62b`, the checked-assignment-cache path recovering
19 of the 35 development structures. The experimental
`specialized_isotropic_refinement` option is false by default. It changes the
execution of the cached canonical isotropic value/derivative kernel; it does not
remove competing assignments or change counts, candidate ranking, phase starts,
iteration limits, final ADP mode, numerical tolerances, resource caps or threads.

## Diagnosis and rejected prototypes

Three sequential five-second native samples repeat complete development solves
for COD 2011393, 2021292 and 2243618, with the previous fast controls and cached
assignment refinement enabled. The first two show constrained model evaluation
as a major individual refinement kernel (1,066 and 532 collapsed self samples),
with cached normal solving also substantial (807 and 616 samples). FFTs dominate
the third, phasing-heavy example. These are representative profiles, not a
measurement of the full corpus time fraction. The original profiles, sampled
native hashes and image UUIDs, exact drivers and printed collapsed counts are
recorded in `benchmarks/structure-solution-isotropic-evaluation-profiles-macos.json`
and `runs/post-assignment-cache-profile/`.

An eight-reflection tile prototype matched all evaluation fields exactly in 20
native fixtures, but was slightly slower in preliminary Python/native evaluation
timing on 12 atoms and 2,108 reflections. A four-product Gram prototype also lost
in Criterion (0.939 ms control versus 0.966 ms prototype). Neither was integrated
into the structure search. Their patches and qualified preliminary measurements
are retained under `benchmarks/structure-solution-*-rejected.*`; these are not
corpus comparisons. The retained implementation specializes the scalar isotropic
loop instead.

## Equations and conventions

For reciprocal Cartesian vector q and a canonical site U=u I, each orthogonal
symmetry image has displacement

    w = -2 pi^2 (q dot q)
    T = exp(w u)
    dT/dp = T w (dU/dp)_00.

Here p is the existing invariant log-tensor coordinate, not U itself. The
specialized path uses the same prepared physical tensors and analytical matrix
exponential derivatives as the general evaluator. For cached image phase phi,
scattering factor f, and coordinate derivative c=dphi/dp:

    re = f T cos(phi), im = f T sin(phi)
    dRe/dp_coordinate -= c im; dIm/dp_coordinate += c re
    dRe/dp_ADP += w (dU/dp)_00 re
    dIm/dp_ADP += w (dU/dp)_00 im.

The existing amplitude derivative, profiled global scale, family square-root
weights, normalized residual and scale derivative are unchanged. Values and all
analytical derivatives are accumulated together in the original reflection,
site, image and parameter reduction order. The q coefficient is hoisted once
per reflection; tensor-basis and cache dispatch are hoisted outside the hot loop.
There is no fast-math, FMA reassociation or precision change. The original
unweighted amplitude threshold of 1e-14 remains before family weighting. No
finite-support or reflection selection convention changes.

The path requires isotropic mode, exact canonical tensor bases, and a geometry
cache that still matches the public symmetry metadata. Unsupported or mutated
geometry/bases, cache-budget fallback and anisotropic models use the original
evaluator. It uses the same two parameter scratch arrays and the existing full
Jacobian; it adds no input rejection or new scratch budget. The public Rust model
builder and private diagnostic Python constructor also expose this execution
choice. `specialized_isotropic_fits` records actual eligible fits separately from
the requested option.

The independent NumPy implementation in
`python/phasesmith/experimental/solution_reference.py` already evaluates these
same equations without calling Rust. Tests compare it and centered finite
differences across five space groups, both ADP modes, family weighting, partial
families/duplicates, cached/direct fallback, short reflection lists, F000 and
physical U boundaries. Mutated Rust geometry and tensor bases test fallback.
GSAS-II's pinned powder-profile oracle does not cover this single-crystal
execution kernel; no GSAS-II implementation code is used.

## Validation protocol

The two modes share one native executable and all previous fast options,
including ambiguity handling and checked assignment caches. Only
`specialized_isotropic_refinement` differs. The balanced serial runner
`benchmarks/solution_isotropic_evaluation_interleaved.py` runs all 35 development
cases, seed zero, four starts, 800 cycles, native 13-second/outer 15-second limits,
and anisotropic final fits. Leading mode alternates by case, 18/17. All 140
predictions are saved before either reference audit barrier. The seven
above-cap references remain in every denominator. Adapter-only seal revision 26
inherits the unchanged registry and 35/65 split from revision 25; no raw or
reference data are opened by sealing. The 65 holdout outcomes remain closed.
No task-owned tests, builds, profiling or kernel benchmarks overlap timed corpus
collection.

Criterion `isotropic_evaluation_` cases measure complete eight- and twelve-atom
isotropic fits (preparation plus up to eight iterations on 1,054 rows) and a full
eight-atom Fe/C/N/O search on a 32-cubed grid with 2,108 expanded rows. Both controls
use the checked normal solver. Full workspace formatting, clippy, Rust tests,
Python tests, independent equations and derivative checks precede collection.

```sh
.venv/bin/python benchmarks/solution_isotropic_evaluation_interleaved.py --output runs/isotropic-evaluation-new --seconds 15
```

Results are added only after complete collection and exact recovery-set audits.

## Complete development results

Both panels recover the **same 19/35 structures**, **26/35 total non-H counts**,
**19/35 per-element counts**, and **15/35 strict 0.02-Angstrom structures**. All
four case sets match both the fresh control and the preceding best
`assignment-cache-cache` report, not merely their totals. Every inferred count
vector is unchanged across all 35 cases. All 31 returned model records are
exactly identical between modes; the four no-model outcomes are identical too.
There are no PhaseSmith hard timeouts. All seven above-cap references remain in
the denominator. No holdout outcomes were opened.

| Measurement | Control | Specialized | Change |
| --- | ---: | ---: | ---: |
| Summed wall time, all 35 outcomes | 70.0443 s | 69.4376 s | 0.87% less |
| Paired median control/new time ratio | — | 1.00961 | about 0.95% less typical time |
| Raw wall-time median | 1.81368 s | 1.76469 s | descriptive, not a paired estimate |
| Isotropic screening stage sum | 12.69490 s | 11.86001 s | 6.58% less |
| Final refinement stage sum | 11.62834 s | 11.64555 s | essentially unchanged |
| P1 phase-search stage sum | 31.21287 s | 31.23286 s | essentially unchanged |

Twenty-three cases are faster and twelve slower. These are measurements from one
balanced paired collection; the sub-percent whole-solve saving is modest and
host-sensitive. The stronger result is the isolated kernel and stage improvement,
with exact model preservation. This does not establish a large overall speedup.

Both modes perform 95 phase starts, 675 isotropic screens, 125 final fits, 249
ambiguity relaxations and 10,305 cheap ambiguity scores. The new evaluator is
actually used in all 675 isotropic fits. Both modes take 4,640 checked assignment
normal steps and 2,460 final cached steps, without fallback. Existing per-start
wall-clock safeguards yield small cycle differences in three cases (49,610
versus 49,567 total Fourier workspace cycles); neither the scheduling rules nor
cycle quotas changed, and the final models remain identical.

In same-executable Criterion comparisons, eight/twelve-atom value/Jacobian
medians fall 16.48%/16.69%. Full short-fit medians fall 7.20%/8.85%. The complete
synthetic search is essentially unchanged (222.53 versus 223.35 ms), showing why
kernel gains must not be advertised as whole-search gains. P1 phase search still
accounts for about 31 seconds of the development run and requested anisotropic
final refinement about 11.6 seconds; this change affects neither kernel.

Artifacts are `benchmarks/structure-solution-isotropic-evaluation-` followed by
`{control,cache,comparison,kernels,profiles,schedule,retention}-macos.json`.
The retention report includes all 35 count vectors, exact successful-case sets,
source report hashes, timing/stage/work totals, and its verifier source. The
schedule audit verifies 140 predictions before both reference barriers, 18/17
leading cases, matching current 44-file source snapshots and the same native
library hash in both modes. Both rejected patches apply to `e9cf62b` in an
isolated Git index. Revision 26 changes adapter hashes only; the corpus registry
and development/holdout split are unchanged.

Final validation: `cargo fmt`, clippy with warnings denied, all-feature workspace
Rust tests (**414 passed, 34 ignored**), and pytest (**2,259 passed, 11 skipped,
34 deselected**). External oracle and real-data tests retain their normal opt-in
requirements. The execution option remains false by default.
