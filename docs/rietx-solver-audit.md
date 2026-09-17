# rietx solver and QARR recipe audit

The installed rietx 1.4.0 source was inspected on 2026-09-16, following the
[Rwp attribution experiments](powder-friedel.md#rwp-attribution-experiments).
Source hashes are in the [audit record](../validation/results/rietx-20260916-solver-source-audit.json).
This is a source audit and diagnostic experiment, not a production solver change.
The subsequent [same-objective solver comparison](solver-isolation.md) finds
that directly substituting SciPy TRF does not recover rietx's residual advantage;
instrument-width feasibility and parameter redundancy remain central.

## What the code does differently

| Area | rietx 1.4.0 | PhaseSmith in this benchmark |
|---|---|---|
| Optimizer | SciPy `least_squares(method="trf")`, with explicit bounds | Rust damped normal-equation solve, capped scaled step, then backtracking |
| Positive variables | Smooth `softplus` transforms for scales, W, sample width coefficients and March ratio | Scaled physical variables and domain/bound validation |
| Sample width variables | Lorentzian size coefficient proportional to 1/D; Gaussian strain coefficient proportional to epsilon² | Crystallite size D and RMS strain epsilon directly |
| Instrument Gaussian width | Add sample variance before validating/flooring the combined width | Require instrument-only variance positive before adding sample variance |
| Trial recovery | TRF adjusts its trust region; convergence tests include cost, step and gradient | Halve the same proposed step up to eight times, increase damping after exhausting those trials |
| Numerical budgets | `max_nfev = 4 * max_iter`; `xtol = gtol = 1e-12` | Explicit iteration/evaluation/rejection limits; objective/step convergence |

The primary rietx sources are `optimize/least_squares.py` (driver at line 1088,
TRF call at 1190), `params/transforms.py` (forward/inverse mappings at 23/35),
`params/vector.py` (initial vector, bounds and decode at 1566/1680/1747),
`schemas/structure.py` (sample width fields near 528),
`schemas/instrument.py` (Caglioti parameter bounds near 921), and
`model/profiles/caglioti.py` (combined Gaussian width near 269).

PhaseSmith's corresponding logic is in
[`rietveld_general_solver.rs`](../crates/phasesmith-workflows/src/rietveld_general_solver.rs),
[`rietveld.py`](../python/phasesmith/refinement/rietveld.py),
[`cw.rs`](../crates/phasesmith-core/src/cw.rs), and
[`cw_contributions.rs`](../crates/phasesmith-core/src/cw_contributions.rs).

These differences are not all proven causes of the residual gap. In particular,
rietx also uses approximate scalar finite differences for many reflection-level
chains, whereas PhaseSmith has analytical chains; replacing those analytical
derivatives is not a suggested improvement. Neither the rietx variance floor
nor its parallel worker pool is active in the recorded final QARR fits.

The 8/28/10 staged recipe and background freeze are choices of our comparison
driver. They are not evidence of an automatic staging algorithm inside rietx.
The existing comparison starts both programs from the shared initial model.

## Width initialization and identifiability

The PhaseSmith Gaussian variance includes

`[U + C epsilon_phase²] tan²(theta) + V tan(theta) + W`,

where `C = (2 * 180 / pi)²` in degrees squared. Thus a shared change in sample
strain variance can be offset by the opposite change in instrument U. The
powder fit alone cannot determine that shared decomposition. rietx uses the
same sum in FWHM-squared units, with an additional factor `8 ln(2)`.

The original first stage fixes every phase's RMS strain at 0.0008. PhaseSmith
cannot compensate that broadening with a negative instrument-only variance;
rietx can, provided its combined width remains valid. That explains why an
apparently matched first-stage recipe exposes different reachable widths.

A PhaseSmith-only diagnostic changes the initial strain to 0.0002 and adds
`C (0.0008² - 0.0002²)` to instrument U. This preserves the starting profile
to floating-point precision, while allowing the first stage to reach narrower
total widths. In stage two, holding U fixed removes the shared U/strain
ambiguity. This defines an empirical decomposition; it is not independent
evidence about physical instrument resolution or microstrain. A calibrated
instrument profile would be the stronger basis for physical interpretation.

## Recovery and staging experiments

The current solver reduces damping by 0.3 after every accepted step, down to
1e-18. It multiplies damping by 10 only after exhausting a backtracking sweep.
Every failed trial counts against a default limit of 20 consecutive rejections.
Consequently a stalled direction can exhaust the guard after roughly two
complete nine-trial sweeps, before damping changes substantially. This is a
recovery limitation worth testing, not proof that simply increasing the guard
is a satisfactory solver design.

All candidates below use PhaseSmith only; no rietx fitted state is imported.
The complete records retain failed experiments as well as successful ones.

| Candidate | Rwp | Existing QARR acceptance |
|---|---:|---|
| Original recipe, both speed options | 19.9888% | Pass |
| Initial variance redistribution only | 19.8179% | Pass |
| Redistribution + free background, original budgets | 20.3319% | Fail: Rwp |
| Above + fixed U in stage two, original budgets | 22.7293% | Fail: residual metrics |
| Longer runs and 200-rejection allowance only | 19.9824% | Pass |
| Redistribution + free background + fixed U + longer runs/recovery, both speed options | 18.8716% | Fail: stage-one repeated rejections |
| Same combined recipe, default profile accuracy | 18.9115% | Pass; all three stages report convergence |

The combined candidate uses 100 iterations and 3000 evaluations as the limits
for each of the first two stages, with 200 consecutive rejected trials allowed.
Its second stage has 21 free parameters rather than 19: three background
coefficients are added and instrument U is held fixed. Scale polish is unchanged.
The default-accuracy candidate uses 330/406/2 model evaluations; that considerable
extra work prevents any speed claim based on these results. The default
20-rejection limit was not changed in production.

Lower residual is not the only objective. Maximum phase-fraction error for the
successful candidate is **1.863 percentage points**, versus **0.656** for the
original combined-policy PhaseSmith recipe and **0.913** for rietx. It passes
the existing 2-point limit, but its phase quantification is worse. Convergence
here means the solver's stopping criterion fired, not an independently verified
stationary point or globally optimal fit.

## Recommended implementation order

1. **Represent width fitting more directly.** Investigate internal variance
   and inverse-size coordinates while keeping public RMS strain and size units.
   Anchor the common instrument/sample contribution or expose its ambiguity.
   Test zero-width limits and carry analytical derivatives through each mapping.
2. **Make rejected-step recovery effective.** Use damping scaled to the local
   linear system or a bounded trust-region step, and account for coupled width
   positivity before evaluating a trial. Preserve explicit runtime limits;
   do not adopt a 200-rejection default as the solution.
3. **Keep background adjustable until widths stabilize.** Prefer a bounded
   joint polish or a deliberately staged linear scale/background solve to
   freezing the first-stage background unconditionally. Test parameter
   correlations and phase fractions alongside Rwp.
4. **Compare time to a common quality target.** Match profile policy, free
   parameter roles and physical bounds as closely as possible. Include
   phase-fraction accuracy and convergence checks, then repeat on independent
   real datasets before changing a default recipe.

Copying rietx's variance floor or its fitted negative instrument coefficient
is not required. Its use of transformed width coordinates and a mature bounded
optimizer provides useful design comparisons; a PhaseSmith implementation
should retain explicit numerical contracts and independently tested derivatives.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/investigate_rietx_solver_recipe.py --threads 8 \
  --json-output solver-recipe.json
```

The [driver](../benchmarks/investigate_rietx_solver_recipe.py) verifies dataset
hashes and unchanged initial profiles, preserves all scientific checks, and
uses the existing native solver. The [record](../validation/results/rietx-20260916-solver-recipe-diagnostic.json)
contains all cases, stage counters and damping histories. Its times are
diagnostic only. Production code and defaults are unchanged.
The successful candidate was repeated at one worker; its full scientific
record and all non-time stage diagnostics match the eight-worker result
exactly. See the [one-worker record](../validation/results/rietx-20260916-solver-recipe-1worker.json).
Initial profile relative L2 difference under the default policy is 5.79e-17.
Ruff formatting/lint and diff checks pass for this diagnostic-only addition.
