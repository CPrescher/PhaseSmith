# Phase-origin alignment and cached peak completion

The phase-origin kernel is now used by the bounded native
[unknown-symmetry coordinator](space-group-search.md). This page records the
earlier supplied-setting experiment and its controls; its opt-in/default remarks
describe that experiment, rather than the current v33 workflow.

This opt-in experimental path changes the search order: measured phase
equivalences determine origins before any atom assignment or coordinate fit.
Supplied-symmetry density recycling then feeds shared, coordinate-fixed count
and element proposals. Short independent-family isotropic screens select the
few complete models promoted to full requested-mode refinement. No formula,
expected counts, reference coordinates or holdout outcomes enter the workers.
The previous heavy-framework search is retained with
`phase_origin_selection=False`, the default until this experiment is evaluated.

## Origin equations and conventions

The positive structure-factor convention is F(h)=sum_j f_j exp(2*pi*i*h*x_j).
For supplied operation x'=R*x+t, h_m=h*R. Aligned coordinates are y=hand*x+s.
For an allowed measured pair, use the published phase-equivalence residual

\[
 \eta=\operatorname{wrap}_{[-\pi,\pi]}
 [\phi_{source}(hand\,h_m)-\phi_{source}(hand\,h)
   +2\pi(h.t+(h_m-h).s)].
\]

The source density is transformed once for each hand. No atoms are needed.
Each independent symmetry/Friedel family contributes its observed F-squared
weight, divided equally among its distinct nontrivial rotated mates. Reflection
stabilizers with h_m=h provide no origin information and are excluded. Duplicate
centring rotations are removed; allowed centring phases agree modulo 2*pi.
Unmeasured and absent families are not manufactured as phase observations.
The strongest 2,048 independent families bound the phase search, with canonical
HKL tie ordering. The merged signed observations themselves remain unchanged.

The wrapped diagnostic alpha and smooth search loss C are

\[
 \alpha={3\over\pi^2 W}\sum w\eta^2,
 \qquad C={1\over W}\sum w(1-\cos\eta).
\]

Both have random-phase expectation one. The use of measured phase differences
for origin/symmetry selection follows
[Sheldrick (2015), section 3](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html).
The circular surrogate, family/selection bounds and search details below are
independent PhaseSmith choices. Only the
supplied operation set is searched; this does not add automatic space-group
determination or lattice reindexing.

Let q=h_m-h. Values and analytical derivatives are accumulated together:

\[
 \nabla C={2\pi\over W}\sum wq\sin\eta,\qquad
 H_C={(2\pi)^2\over W}\sum wqq^T\cos\eta,
\]
\[
 \nabla\alpha={12\over\pi W}\sum wq\eta.
\]

Alpha derivatives apply away from phase-wrap boundaries. A zero calculated
coefficient has undefined phase and its pair is excluded at relative magnitude
1e-12. An all-zero informative map fails explicitly. In P1 (no rotational
origin information), origin zero/hand +1 is a deterministic free-gauge choice.

Aggregate w*exp(i*delta) at reciprocal difference q. An inverse FFT evaluates the
smooth merit on a periodic fractional grid with each axis bounded to 16–64.
Folding q into that grid evaluates its discrete samples exactly; it does not
claim to resolve every continuous local minimum. Retain up to eight local
maxima of the cosine sum and refine each by a damped 3x3 Newton/pseudoinverse
step with backtracking, at most 24 iterations. The maximum fractional step is
0.1. Continuous origin gauges are handled by the pseudoinverse.

Shifts whose conjugated operation sets agree are equivalent, including free
origin directions and allowed alternative origins. These are deduplicated
before completion. Both hands are tested for acentric groups; centrosymmetric
groups need only +1 because inversion is already a group operation. Return at
most four hypotheses in production; only alpha<=0.3 settings earn completion.
An alpha pass is a compatibility test, not a correctness certificate.

## Completion and selective refinement

Equivalent hand/origin representations are also deduplicated when the affine
inversion conjugates the supplied operation set onto itself. This preserves
genuinely different origin choices instead of spending both completion slots on
an inversion-equivalent pair; enantiomorphic groups retain their distinct hands.

For each of at most two compatible origins, apply the origin and ten requested
cycles of positivity/symmetry phase projection. This precedes the first peak
assignment. Extract up to 128 periodic maxima, project near-special peaks onto
their stabilizer constraints and deduplicate complete orbits. Redundant stabilizer
operations are reduced to at most three rank-increasing affine constraints; the
projected point must satisfy all nearby stabilizer operations. Require 0.5 angstrom
physical separation, preserve the input-derived full-cell count cap, and bound
the proposed union to four times that cap and at most 128 asymmetric proposals.
Keep positive 0.7-angstrom integrated-density masses above 2% of the strongest.
These thresholds are explicit proposal heuristics, not supplied atom counts.

For orbit j, cache the phase column P_j(h)=sum_images exp(2*pi*i*h*x_image) and
all three representative-coordinate derivatives in the same image/reflection
pass. Neutral scattering factors are prepared once for every allowed type.
Candidate labels and orbit omissions update the complex sum by changing only a
cached column, with no coordinate/ADP fit. Cache budgets are 262,144 orbit/row
entries and two million image/row terms. At most 2,048 canonical screening
families are used, sampled uniformly across canonical order when necessary.
Full fits still use all original prepared observations and the original scores.

For included orbit labels z_j (an omitted orbit contributes zero), define

\[
 F_h=\sum_j f_{z_j}(s_h)P_j(h),\quad
 A_h=|F_h|\exp(-8\pi^2 U s_h^2),\quad
 k={\sum_h A_h F_{o,h}\over\sum_h A_h^2},\quad
 J={\sum_h(k A_h-F_{o,h})^2\over\sum_h F_{o,h}^2}.
\]

For image x=R*p+t, the cached jet is
dP_j/dp_a=sum_images (2*pi*i*(h*R)_a)*exp(2*pi*i*h*x).
Then dA_h/dp_a=exp(-8*pi²*U*s_h²)*Re(conj(F_h)*f_z*dP_j/dp_a)/|F_h|,
and dJ/dp_a=2*k*sum_h((k*A_h-Fo_h)*dA_h/dp_a)/sum_h(Fo_h²).
Profiling k eliminates its derivative by stationarity. At calculated modulus
below 1e-100 its coordinate derivative is assigned zero. The objective rejects
an all-zero model. These derivatives hold for fixed orbit topology; at a special
position only tangent directions preserve multiplicity. They are not derivatives
through an orbit birth, deletion, categorical label change or phase-wrap boundary.

For fixed labels, profile the common amplitude scale and compare global Uiso
seeds 0, .005, .01, .02, .04 and .08 angstrom-squared. Initial density scales
bracket the heaviest expected atom at the strongest integrated peak. If carbon
is supplied, similar integrated masses connected at 1.25–1.65 angstrom provide
an additional carbon-based scale proposal. Diffraction scores decide among
these alternatives; the carbon heuristic is not imposed on other chemistries.

Nested peak prefixes provide count proposals, with least-cost density rescues
for required types. Up to four distinct cheap models undergo four categorical
whole-orbit birth/deletion/relabel sweeps. Their BIC-like descriptive rank is
M*log(profiled amplitude SSE)+(4*N_orbits+2)*log(M). This is a screen, not a
probabilistic count posterior. No combinatorial count-table solves are restarted.

At most three candidates per setting with raw normalized SSE<=0.25 receive up
to eight isotropic iterations. The best two screens with R1<=0.3 may receive
requested-mode fits using all observations. These model fits never determine
the initial origin. Screen and full scores are kept separate, source-frame
transforms are composed explicitly, and only full candidates enter final ranking.
Timers and counters record origin, recycling, categorical completion, screening
and final fitting work, including failed attempts.

With phase-quality selection enabled, validate each completed P1 start through
this sequence as soon as it is available. A high P1 CFOM alone cannot end the
entire search if the resulting model fails. Independent phase starts continue
within the original global budget. Each map initially receives four seconds;
the former R1<0.3 rule granting an incorrect map the entire budget is not used.

## Validation and development comparison

Independent NumPy phase-equivalence values/derivatives and orbit-column
values/coordinate jets are tested, including centring, screw translations,
free gauges, hand inversion and special-position tangent boundaries. The
GSAS-II powder oracle does not cover this structure-search stage. The prior
experimental solver and the pinned SHELXT binary remain external comparisons.

The fixed 35 development cases retain their input hashes, seed 0, four starts,
800-cycle ceiling, 13-second internal/15-second outer budget, one-thread policy,
input-derived atom caps and exact count/geometry correctness gates. All solver
predictions are saved before references are audited. The 65 holdout outcomes
remain unopened. Adapter-only seals advance without reopening raw/reference
data or changing the registry.

The same-native-build comparison (`b53b97a35adcaa8057a63434b070b6d9c7a16a807bd3c481da95cce1528b75db`)
is retained in
`benchmarks/structure-solution-phase-origin-comparison-macos.json`, with full
chosen predictions and audits in the adjacent `phase-origin-control` and
`phase-origin` reports.

| Outcome, all 35 development cases | Previous path, matched control | Phase origin/completion | SHELXT in new collection |
| --- | ---: | ---: | ---: |
| Correct labelled structures | 8 | 12 | 13 |
| Exact total non-H atom counts | 14 | 26 | 29 |
| Exact per-element full-cell counts | 8 | 13 | 14 |
| Strict 0.02-angstrom recovery | 8 | 9 | 10 |
| Median wall seconds, all outcomes | 13.217 | 4.376 | 0.877 |
| 95th percentile wall seconds | 13.266 | 11.297 | 1.882 |
| Hard timeouts | 0 | 0 | 1 |

Six recoveries are gained: COD 2010705, 2010901, 2011590, 2021199, 2107296 and
2108342. Two are lost: COD 4326074 has the right element counts but wrong
coordinates despite R1=0.0242; COD 2243618 has the right total but wrong Mg/Cl
assignments. The six common successes take median 1.365 versus 8.709 seconds.
The new all-outcome median is about three times faster, but these are single-seed,
serial deadline-bounded collections, not paired repeated timing estimates.
The previous committed campaign recovered 9/35 structures and 11 totals; this
fresh matched control recovers 8 and 14. In particular COD 2010705 changes outcome
under the deadline. Report the fresh matched comparison rather than combining
its timings with the older outcome counts.

The previous path attempts 2,443 isotropic screens and 712 final fits; the new
path attempts 146 screens and 100 final fits. It performs 70,525 cheap cached
categorical scores. Across the panel, recorded origin search takes 1.418 seconds,
symmetry recycling 1.572, cached completion 1.720, isotropic screening 3.073 and
final refinement 66.720 seconds. These timers cover those stages only; P1
phasing, preparation, startup and other work remain in wall time. The control
has no equivalent per-stage timing instrumentation, represented by null rather
than an inferred zero in the comparison.

The input-derived cap excludes seven reference-sized structures in both modes;
all seven remain in the denominator. The new path recovers 12/28 within-cap
structures. It is still about five times slower than SHELXT by all-outcome
median, on this development panel and host, and is not established as equally
reliable. PhaseSmith is native arm64 and the supplied SHELXT is x86_64 under
the host compatibility runtime. Neither program uses multiple numerical threads.

Thirteen cases have correct total counts but wrong element-count sets, mostly
confusion among C/N/O and also O/F or Mg/Cl. The next algorithmic target is joint
light-element assignment after limited coordinate/displacement relaxation,
including coupled label moves. Low amplitude residuals and phase compatibility
must continue to be distinguished from actual structural correctness. The new
path remains opt-in because it loses two former recoveries; no reference-based
fallback or outcome-selected hybrid is used.

The new microbenchmarks use eight peaks on a 32-cubed map and 2,108 measured
rows (1,054 for the cached independent-family screen). Criterion point estimates
are 4.175 ms for origin search, 7.928 microseconds for fused origin
loss/gradient/Hessian, 118.94 microseconds for peak-cache construction,
21.61 microseconds for the cached objective/coordinate gradient and 610.07
microseconds for count/label search. Raw samples, confidence intervals, source
and executable hashes are in
`benchmarks/structure-solution-phase-origin-kernels-macos.json`. These measure
new-kernel costs; the complete panel provides the old/new performance comparison.

Validation: the focused origin/completion/report checks pass (42 tests), full
Python validation passes (1,570 passed, 11 external-data skips, 34 deselected),
and `cargo test --workspace --all-features` passes (397 passed, 34 ignored).
Workspace Clippy with all targets/features and warnings denied, Rust formatting,
Ruff over Python/tests/benchmarks and `git diff --check` all pass. No tolerances
or corpus correctness gates were loosened.

Both timed collections used identical compiled native code and unchanged runtime
sources throughout. The only subsequent difference to a collected implementation
file is Python API docstrings, verified by comparing the parsed syntax trees
with docstrings removed. Production Rust sources still match the saved snapshots.
Tests, documentation, standalone comparison reporting and microbenchmarks were
completed outside the timed collection interval.

The first complete prototype collection is retained in
`benchmarks/structure-solution-phase-origin-prototype-failed-macos.json`, with
local source snapshots and raw outputs under `runs/phase-origin/development-a`.
It recovered only 1/35 structures because two
integration calls exceeded existing kernel bounds: requesting more than 128
maxima, and passing redundant high-symmetry stabilizer operations into the
three-constraint projector. The corrected collection clamps the first call and
uses independent stabilizer equations checked against the full stabilizer set.
Default-cap and cubic-special-position end-to-end regressions cover these errors.
The failed collection is evidence of an implementation failure; its shorter
failure times are not treated as a performance improvement.

Reproduce the development collection with `solution_corpus_benchmark.py run`,
`--manifest validation/solution-corpus-v5.json --seconds 15
--wilson-normalization --phase-quality-selection --phase-origin-selection` and
a new `--output` directory. The matched control omits only the last flag.
After all predictions finish, run `solution_corpus_reaudit.py` on each saved
collection. `solution_phase_ablation.py` compares the resulting complete reports,
checks policy/cap/seed consistency and sums actual stage counters. These commands
default to development; neither collection opens holdout outcomes. Adapter-only
seal revision 10 preserves the revision-9 data registry and split hashes.
