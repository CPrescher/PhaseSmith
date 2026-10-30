# Multiplicity-weighted final refinement

The experiment starts from `0440339`. The prior best execution path enables
packed real FFTs, fused phase workspaces, filtered peaks, scalar phase scoring,
work-sliced phasing, isotropic assignment competition and guarded normal caches.
The new experimental `multiplicity_weighted_refinement` flag is false by default
and changes only final requested-mode constrained fits in element-only search.

## Equations and conventions

The original preparation variance-merges symmetry/Friedel observations and
expands their means to P1. Under the supplied symmetry and non-resonant real
scattering model, equivalent calculated amplitudes and their parameter
derivatives agree mathematically. Let m_f count actual supplied rows belonging
to family f, o_f its observed amplitude and a_f(theta) its calculated amplitude.
The existing expanded-data amplitude objective is preserved by

    k = sum_f m_f a_f o_f / sum_f m_f a_f^2
    N = sqrt(sum_f m_f o_f^2)
    r_f = sqrt(m_f) (k a_f - o_f) / N
    dk/dtheta = sum_f m_f da_f/dtheta (o_f - 2 k a_f) / sum_f m_f a_f^2
    J_f = sqrt(m_f) (k da_f/dtheta + a_f dk/dtheta) / N
    objective = sum_f r_f^2
    R1 = sum_f m_f |k a_f-o_f| / sum_f m_f o_f.

The implementation scales each representative amplitude, observed amplitude
and fused analytical derivative by sqrt(m_f), then uses the original scale,
normalization and derivative equations. R1 has a second sqrt(m_f) factor to
retain its original absolute-residual weighting. The independent NumPy reference
implements the explicit weighted sums directly, without calling Rust.

Families contain only supplied rows, including duplicates and incomplete orbits;
no missing observations are fabricated. First supplied members define a stable
representative order. Members must have bit-identical observed amplitudes;
otherwise the weighted constructor rejects the input, leaving the original
general expanded evaluator available. The shared search's merged data satisfy
this requirement by construction. The plan checks its operation set at evaluation
and rejects a mutated group. It assumes the same valid symmetry-constrained
site/image/tensor model as the original evaluator.

The original input-row, parameter-times-original-row and input-derived atom caps
apply before reduction. This change must not expand admissibility or remove
above-cap references from benchmark denominators. No observations, signed source
intensities, phase support classes, grid support, resolution, f64 precision,
peak-count choices, assignment candidates, seeds, deadlines, iteration ceilings,
ADP mode, refinement stopping gates or model-rank sample count are changed.
Isotropic screens retain their existing independent-family data and weights.
P1 phasing continues to use its original expanded arrays.

This reduces scattering/geometry preparation, structure-factor passes and dense
Jacobian/normal-operator rows in final fits. It adds a per-fit bounded family plan;
preparation is included in the benchmark. No new runtime dependency, thread or
global cache is introduced. Mathematical equivalence does not imply bitwise
equivalence: selecting representatives and changing reduction order changes
round-off, which can affect iterative trajectories and close candidate rankings.

Audit fields distinguish requested execution from actual work:
`weighted_refinement_fits`, `weighted_refinement_input_rows` and
`weighted_refinement_family_rows` sum prepared fits and row counts once per fit,
not once per nonlinear iteration or Fourier cycle.

## Validation and comparison protocol

Tests compare isotropic and anisotropic native values and complete profiled
derivatives with independent direct NumPy calculations in P1, P-1, Pm, P3121 and
Fm-3m, with direct/cached geometry, noncubic metrics, centring and special sites.
They compare expanded/weighted objectives, scales, R1, gradients and normal
matrices, and check centered finite differences at randomized parameters. Unequal
multiplicities, incomplete families, repeated rows and shuffled inputs exercise
actual-count semantics. Invalid unequal family amplitudes and original memory
limits are tested explicitly. Complete synthetic searches require the same
structure, per-element counts and final anisotropic mode. No existing tolerance
is loosened. This introduces no new peak profile, support convention or physical
term; existing structure-factor and ADP equations remain the reference. The
GSAS-II powder-profile oracle does not cover this single-crystal refiner.

The complete comparison uses all 35 development cases and the same native binary
for control/weighted modes, alternating leading mode by case. Both enable all
previous best execution controls, including fused phase workspaces. Only the new
refinement flag differs. Seed zero, four starts, 800 requested cycles, native
13-second/outer 15-second budgets, final anisotropic refinement, source data,
input-derived caps and original correctness/count gates remain fixed. Seven
above-cap references stay in all denominators. All 140 predictions precede both
reference audit barriers. Adapter-only seal revision 22 retains the same registry
and 35/65 split. No task-owned tests, builds or microbenchmarks overlap collection.
The 65 holdout outcomes remain unopened.

To collect a new panel in a fresh directory:

```sh
.venv/bin/python benchmarks/solution_family_interleaved.py --output runs/refinement-families-new --seconds 15
```

Normalize both completed panels with `benchmarks/solution_corpus_reaudit.py`, then
compare them using `benchmarks/solution_phase_ablation.py`. Fixed-work kernel
benchmarks are selected by `family_refine_` in the solution Criterion benchmark.

## Completed validation and fixed-work measurements

Full validation passes 2,168 Python tests (11 external-data skips and 34 marker
deselections) and 413 Rust tests (34 ignored), with strict all-target/all-feature
Clippy, Rust formatting and changed-file Ruff checks. Thirty new focused tests
cover native/reference equations, centered finite differences, original resource
limits and complete synthetic structure/count recovery; the report's feature
mismatch guard also tests the new flag.

Same-executable Criterion sample medians on the eight-atom, 2,108-row example are:

| Work | Expanded control | Multiplicity-weighted | Time reduction |
| --- | ---: | ---: | ---: |
| Symmetry recovery, preparation and three anisotropic iterations | 6.092 ms | 3.344 ms | 45.1% |
| Complete small count-free search | 70.002 ms | 69.822 ms | 0.3% |

The small whole-search example spends little time in final refinement. It does
not establish a broad end-to-end speedup. Both modes execute one phase start and
53 Fourier cycles; the weighted mode performs one weighted final fit on 1,054
families versus 2,108 input rows. Preparation costs are included rather than
amortized away. Raw sample sets, actual-work metadata and numerical-source,
benchmark-source and executable hashes are preserved in
`benchmarks/structure-solution-refinement-families-kernels-macos.json`.

## Completed 35-case comparison

| Measure, all development cases | Expanded control | Multiplicity-weighted |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 16/35 |
| Correct total non-H atom counts | 26/35 | 26/35 |
| Correct per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-angstrom structures | 13/35 | 13/35 |
| Median wall time, all outcomes | 1.627 s | 1.524 s |
| Wall p95 | 5.831 s | 4.196 s |
| Summed solve wall time | 81.998 s | 63.808 s |
| Summed final-refinement time | 26.386 s | 8.333 s |
| Summed phase-search time | 31.978 s | 31.928 s |
| Starts / screening fits / final fits | 96 / 433 / 96 | 96 / 433 / 96 |

The median paired wall ratio is 1.15807x, or 13.65% less time. Thirty cases are
faster and five slower; the median paired difference is -0.224 seconds. Summed
wall time falls 22.18%, and final-refinement time falls 68.42%. On the 16 common
successes the median paired wall ratio is 1.08368x (7.72% less time), with raw
wall medians 0.879/0.802 seconds. The ratio of the raw all-case medians is a
different statistic from the typical paired ratio. These are single-seed,
single balanced collections on an uncontrolled host, not repeated confidence
intervals.

The recovery case sets, all inferred count vectors and strict recovery case sets
are unchanged. There are no gained/lost correct structures and no hard PhaseSmith
timeouts. Four cases return no model in both modes. The 31 returned numerical
models are not bitwise identical, as expected from weighted reduction and changed
iterative rounding; preserved recovery is established by the original reference
gates, not exact float equality.
The largest selected-model R1 difference is 9.53e-11.

All 96 final fits use multiplicity weighting in the treatment, reducing summed
per-fit input rows from 627,106 to 182,331 evaluated family rows. Those are
preparation counts, not totals multiplied by iteration count. Actual cached
normal solves are 1,963/1,946, with zero fallbacks in both modes. Phase-scored
cycles are 50,179/50,233; six starts still encounter their phase time backstop.
The new feature reduces final-fit work without changing the requested budgets
or final-refinement count. Phasing is now the largest measured remaining stage.

Large savings include still-incorrect COD 2020988 (8.024 to 4.549 seconds) and
COD 2242589 (5.841 to 3.992 seconds). Three of the five small regressions are
no-model searches that perform no final fit; the largest regression is 0.060
seconds. These observations qualify the distribution rather than demonstrating
new structure recovery. External SHELXT controls retain 13/35 structures and
29/35 total counts in both panels; current PhaseSmith coverage still assumes
supplied space-group symmetry.

Both reports have the same native hash and 42 matching runtime-source snapshots.
The verified schedule contains 140 prediction events followed by two reference
audit barriers, with 18 control-leading and 17 weighted-leading cases. Reports,
paired comparison, kernel samples and verified schedule are preserved as
`benchmarks/structure-solution-refinement-families-{control,weighted,comparison,kernels,schedule}-macos.json`
(each name is a separate file). Raw workers, audits and sealed source snapshots
remain in `runs/refinement-families/`. Adapter revision 22 preserves the original
registry and split. The new flag stays false by default. All 65 holdout outcomes
remain unopened.
