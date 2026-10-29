# Composition-free normalization and phase-quality selection

This is the historical experiment at `1be278b`. The subsequent
[separate ablations and calculated-normalization correction](solution-phase-ablation.md)
retain these results and explain the changed current options. On the current
build, `phase_first_selection=True` plus
`calculated_shell_normalization=False` selects the earlier method; use the
historical checkout and adapter seal for exact source-level reproduction.

This opt-in experimental follow-up implements resolution-dependent normalization
and phase-map selection from published equations and first principles. It does
not translate decompiled program code, implement complete SHELXT, change the
supplied-space-group contract, or introduce a learned model.

## Inputs and normalization

Only corrected measured HKLF4 observations, their uncertainties, the cell, known
symmetry and non-H element types enter numerical workers. No formula quantities,
expected counts, reference coordinates or holdout outcomes enter this path.
Inverse-variance symmetry/Friedel merging retains signed intensity means. The
original observations and all final refinement/scoring objectives are unchanged.

For independent allowed family h, let x_h=(sin(theta)/lambda)^2 and epsilon_h be
the number of supplied operations whose reciprocal rotation fixes h (including
centring). Define C(x)=sum_e f_e(x)^2 using one copy of each supplied type. This
specifies a radial form-factor shape, **not** an assumed chemical formula. An
empirical radial envelope absorbs unknown quantities and overall attenuation.

Fit the expectation

\[
 \widehat I_h=\varepsilon_h C(x_h)\exp[a+b x_h+r(x_h)].
\]

Canonical symmetry/Friedel families contribute once, including nonpositive signed
means. Sort by resolution with a 1e-12-relative resolution-key convention and
canonical-HKL tie breaking. Use floor(N/80) equal-population bins, bounded to
2–20 and at most N. Coincident shell centres merge with population weights.
For each shell compute the **signed arithmetic mean** of I/(epsilon*C). Fit the
log of positive shell means against shell centres with population weights.
Nonpositive shell means remain recorded; they use the fitted envelope without a
log residual. If no positive shell mean exists, normalization fails explicitly
and the solver records its legacy-sharpening fallback diagnostic.

For a positive shell with n families, shrink its log residual toward the Wilson
line by n/(n+80). Linear interpolation gives r(x), with constant residual
extrapolation outside shell centres. Scale intensities before accumulation to
avoid overflow. Expectations must be finite and greater than 1e-250. The slope
b is an effective radial-envelope estimate; -b/2 should not be interpreted as a
physical sample ADP when composition and its scattering mix are unknown.

The separate derived amplitudes are

\[
 E_h=\sqrt{\max(I_h,0)/\widehat I_h}.
\]

A final positive common rescale makes the independent-family mean E^2 exactly
one. Shell means need not be exactly one after smoothing/interpolation. Negative
measurements are never removed from the source data or shell estimation; only
the derived phasing amplitude is clipped at zero. The mixed phasing coefficient
is G=E^(1/2)*Fo^(1/2), with Fo=sqrt(max(I,0)). Charge-flipping can also opt into the
normalized amplitudes; phase-quality selection currently requires dual space.

This is an isotropic empirical Wilson/shell normalization for unknown atom
quantities. It is not chemical-composition inference, anisotropic scaling, a
Bayesian weak-intensity amplitude estimator, or SHELXT's exact normalization
implementation. Severe data anisotropy, sparse shells and non-random scattering
can still limit the estimate.

Primary basis: the independent-family epsilon normalization and shell expectation
in [Read (1999), section 3](https://journals.iucr.org/d/issues/1999/10/00/ba0032/),
Wilson normalization in the [IUCr GENEV documentation](https://www.iucr.org/__data/iucr/cif/software/xtal/xtal372htmlman/html/genev-desc.html),
and the mixed coefficients in [Sheldrick (2015), section 2](https://journals.iucr.org/a/issues/2015/01/00/sc5086/index.html).
The bin bounds, regularization and failure conventions above are explicit
PhaseSmith choices rather than a recovered SHELX implementation.

## Phase quality and derivatives

The prepared comparison uses one representative per independent measured allowed
family, plus actually observed systematic-absence families within the measured
allowed-resolution sphere. There are no fabricated unmeasured reflections in the
score. Observed forbidden-family intensities are independently variance-merged
for weak-set ranking, retaining their signed means. They do not enter CC, because
known supplied symmetry excludes them from the phasing target. Families outside
the measured sphere are excluded from this score to avoid FFT aliasing.

Let g be the mixed observed G and c be back-transformed atomic-map magnitudes.
CC is the Pearson correlation of g and c on allowed families, as a fraction.
Targets have a harmless common maximum rescale to keep the flat-variance test
independent of intensity units. The weak set is the lowest 10% of normalized
observed E, rounded upward, including all ties within relative 1e-12 at the
boundary. Its actual size is recorded; many zero/negative observations may make
it larger than 10%.

Because G=Fo/(expected I)^(1/4), calculated Ec^2 is proportional to
c^2/sqrt(expected I). When testing quality selection with legacy phasing
coefficients instead, the conversion is c^2*sqrt(C)/expected I. Independently
rescale calculated E^2 to have unit mean on the comparison set, then compute

\[
 R_{weak}=\langle E_c^2\rangle_{weak},\qquad Q=CC-R_{weak}.
\]

This is the published default 0.01*CC_percent-Rweak criterion. The organic-only
CHEM alternative is not imposed on minerals. A flat target/calculated variance
reports CC=0 with `correlation_defined=False`; a zero/nonfinite map is rejected.
Quality is invariant to positive common scaling of calculated magnitudes.

Values and dQ/dc are evaluated together. With centred allowed vectors u=g-mean(g)
and v=c-mean(c),

\[
 {\partial CC\over\partial c_i}={u_i\over\sqrt{(u.u)(v.v)}}
  -CC\,{v_i\over v.v}.
\]

It is zero for absence-only rows. If w converts c^2 to raw Ec^2,
D=mean(c^2*w), W is the weak-set size and M the comparison size,

\[
 {\partial R_{weak}\over\partial c_i}=
 {2c_iw_i\over D}\left({1_{i\in weak}\over W}-{R_{weak}\over M}\right).
\]

The score is a selection statistic, not an additional coordinate/ADP refinement
term or a correctness probability. Existing fused structure-factor derivatives
and finite Gaussian support are unchanged. Independent NumPy values and centred
finite differences test these analytic derivatives, including intensity-unit and
calculated-scale invariance.

## Phase-first schedule

`wilson_normalization=True` and `phase_quality_selection=True` are independent,
explicit options, both off by default while the experiment is evaluated. The
latter prepares normalization for its weak statistic even when the former leaves
the phasing coefficients on the legacy path.

With phase-quality selection active, cheap P1 phasing precedes atom-count edits:

1. Each requested deterministic Patterson/random start receives up to
   0.30*total_seconds/trials of the unchanged global deadline. Its non-omitted
   atomic maps compete by Q. A map with Q>=0.90 after at least 32 cycles ends that
   phasing start, and reaching that threshold also ends start exploration early.
2. Keep at most `beam_width` complete density maps, sorted by Q with trial-index
   ties. This bounds retained FFT-map memory independently of the trial count.
3. Only then pass maps to the existing shared count search, in quality order.
   Each initially receives four seconds within the global deadline. The existing
   promising-model continuation rule can grant the remaining global budget.
   All final candidates use the original observations and requested refinement.

The 30% allocation, Q=0.90 threshold, 32-cycle minimum and four-second initial
model budget are explicit development scheduling choices. They do not certify
phase correctness. No atom count is used for phase-map ranking or early stopping.
The legacy checkpointed mixed-amplitude SSE path remains available when quality
selection is disabled. Legacy histories remain amplitude-SSE diagnostics;
separate typed quality traces record actual CC/Rweak/Q values, while
`phase_model_trials` records the ranked start order actually sent to count search.

An initial development pilot replaced only the window score. It preserved its
three previously recovered examples but still spent nearly the whole budget
refining four-cycle partial models on harder cases. That evidence motivated
moving phase selection ahead of count refinement. Pilot outputs are retained
locally and are not substituted for a complete 35-case campaign.

This change adds neither automatic space-group selection nor missing-data
resolution extension beyond the original sphere. Tangent recycling retains its
existing independently normalized relation weights. Those are separate methods,
not claimed as newly reproduced here.

## Development experiment

The exact 35 development cases, input hashes, element-only contract, input-derived
atom ceilings, original correctness gates, seed 0, four starts, 800-cycle ceiling,
13-second internal/15-second outer budgets and one-thread policy remain fixed.
Both programs run serially with alternating order. Every chosen prediction is
saved before reference auditing; failures, timeouts and the seven known capacity
boundaries remain in the denominator. The 65 holdout solver outcomes stay unopened.

The complete new campaign recovers **7/35 structures**, the same seven as the
preceding adaptive-count campaign. All seven pass the strict 0.02 angstrom gate.
Total counts are correct in **12/35**, versus 13/35 previously; exact per-element
counts remain 7/35. No normalization fallback occurs, and the largest deviation
of the recorded independent-family mean E-squared from one is 4.22e-15.

The all-outcome median is 13.374 seconds versus the preceding 13.390 seconds,
which is not evidence of a speed improvement. Among the seven common successes,
the median increases from 2.960 to 10.320 seconds. Selecting several complete
phase trajectories before any model handoff adds work on easy cases; a better
phase statistic alone does not eliminate count/label-search failures. For
example, the former count-only successes COD 2243618 and 1549173 become
over-/undercounts, while COD 2011590 gains the right total without the right
element-labelled structure. These historical comparisons use different native
builds; the same-build control below is the primary ablation.

The primary ablation uses the **same installed native executable** with both
options disabled/enabled, retaining adaptive counts and all other controls:

| Measure, all 35 cases | Same-build control | Normalization + quality | SHELXT, new campaign |
| --- | ---: | ---: | ---: |
| Correct element-labelled structures | 8 | 7 | 13 |
| Exact per-element counts | 8 | 7 | 14 |
| Correct total non-H atom count | 13 | 12 | 29 |
| Strict 0.02 angstrom recovery | 8 | 7 | 10 |
| Median wall time, all outcomes (s) | 13.331 | 13.374 | 1.440 |
| 95th percentile wall time (s) | 13.454 | 13.646 | 3.259 |
| Hard process timeouts | 0 | 0 | 1 |

There are no gained structures. COD 2243618 is recovered in this control but
the new path returns 38 atoms instead of 30. The seven common recoveries take
median 1.776 seconds in the control and 10.320 seconds with the new options
(5.81 times slower). Full-panel medians are dominated by the 13-second internal
budget and hide that regression. The 8-versus-7 difference between the current
control and preceding campaign also illustrates sensitivity of deadline-bounded
search to host/compiler timing. These are serial complete collections, not
paired repeated timing estimates or uncertainty intervals. The interrupted
control collection resumed from saved completed predictions; its one unfinished
worker directory was preserved separately and rerun before any control audit.

Both solvers use one thread. PhaseSmith runs natively on arm64, while the pinned
SHELXT 2018/2 executable uses the x86_64 compatibility runtime. The same supplied
HKLF4 observations, cell, space group and element types enter both, and neither
receives reference atom quantities or coordinates. SHELXT's exact-element count
does not by itself establish coordinate recovery.

Retained reports:

- [Complete new campaign](../benchmarks/structure-solution-phase-quality-macos.json)
- [Complete same-build control](../benchmarks/structure-solution-phase-control-macos.json)
- [Verified case-by-case comparison](../benchmarks/structure-solution-phase-comparison-macos.json)

The implementation is therefore retained as an explicit experimental option,
**not promoted to the default or described as a recovery/performance gain**.
The combined experiment changes both normalization and selection/scheduling;
it cannot attribute the regression to one component. Separate fixed-cycle
normalization-only and quality-only ablations would be needed for that claim.
The input-derived resource caps still exclude seven reference sizes. Automatic
space-group selection, anisotropic normalization and resolution extension remain
unimplemented; these results do not establish parity with SHELXT.

Validation passes: 396 Rust tests, 1,519 Python tests (11 skipped and 34
deselected under the existing external-data policy), Clippy with warnings
denied, Rust formatting, Ruff and whitespace checks. The 22 new focused Python
cases cover six symmetry settings, centring/epsilon factors, signed input,
coincident shell boundaries, native/independent-NumPy agreement, centred
finite-difference gradients, positive-scale invariance, invalid element types,
explicit fallback and three synthetic count-search regressions. Two native
unit tests check weak-evidence ranking and flat correlation semantics.

The GSAS-II profile oracle does not cover count-free structure phasing or this
map-selection statistic. Independent equations/derivatives and the pinned
black-box SHELXT campaign provide the relevant checks; neither comparison is
claimed as agreement with SHELXT's unpublished implementation details.

On the eight-atom Si/O benchmark (2,108 measured rows, 1,054 independent Friedel
families), ten Criterion samples give 605.50 microseconds for normalization
and 7.463 microseconds for a fused quality value/gradient evaluation. Their
95% estimate intervals are 604.06–606.84 and 7.428–7.508 microseconds. These
small kernel costs do not predict the whole-search speed: the phase-first
schedule demonstrably spends more time before successful model recovery.
[Raw kernel samples and provenance](../benchmarks/structure-solution-phase-kernels-macos.json)
are retained with the full campaigns. No older golden data or benchmark result
has been regenerated or replaced.

## Commands

Use a release build with `experimental-solution` and a fresh output directory:

```sh
.venv/bin/python benchmarks/solution_corpus_benchmark.py run \
  --manifest validation/solution-corpus-v5.json --seconds 15 \
  --wilson-normalization --phase-quality-selection \
  --output runs/phase-quality/new-development
.venv/bin/python benchmarks/solution_corpus_reaudit.py \
  --manifest validation/solution-corpus-v5.json \
  --run runs/phase-quality/new-development \
  --output runs/phase-quality/new-development/comparison-verified.json
```

Omit both flags for the same-build control; each flag can also be tested alone.
Corpus adapter seal revision 6 adds these explicit flags and resume validation;
source datasets, splits, input packets and prior immutable seals are unchanged.
No held-out run is needed for this development comparison.
