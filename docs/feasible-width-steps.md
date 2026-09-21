# Experimental feasible CW width steps

This experiment is preserved on `codex/feasible-width-experiment-20260917`.
Its option is not included in the consolidated develop API. The evidence below
records the original experiment and does not establish a production default.

Development status, 2026-09-17: **parked** pending broader demonstrated benefit.
The option remains off by default. The fixed-workflow assessment and promotion
criteria in [refinement-assessment.md](refinement-assessment.md) supersede any
proposal to continue tuning this solver against QARR alone.

`RietveldOptions(feasible_width_steps=True)` enables an optional constrained
quadratic proposal in the complete native CW solver and independent NumPy
orchestration. The default is **False**. This is a solver experiment, not a
new production recipe or a guarantee of a lower final Rwp. In a staged fit,
a different early endpoint changes the frozen background and can worsen the
final fit. Keep profile and composition checks separate.

## Model and constraints

For residual `r = W(ycalc-yobs)` and Jacobian `J` in scaled free coordinates,
solve the local convex problem

```
min_d  g.T d + 0.5 d.T (J.T J + lambda I) d,    g = J.T r
subject to A d >= b.
```

The existing CW instrument domain requires, at each calculated reflection
and each fixed spectral component, with theta half the corrected two-theta:

```
G(theta) = U tan(theta)^2 + V tan(theta) + W > 0
L(theta) = X sec(theta) + Y tan(theta) >= 0.
```

These are the base instrument widths, before sample contributions. The
proposal requires `grad(G).T d >= -0.99 G` and likewise for L. The 0.99
fraction keeps a margin from the strict Gaussian boundary; it does not impose
a new physical minimum width. A zero Lorentzian width remains valid. Width
derivatives are `[tan², tan, 1, 0, 0]` and `[0, 0, 0, sec, tan]` for UVWXY.
For an additive zero shift measured in degrees, append

```
dG/dzero = (2 U tan(theta) + V) sec(theta)^2 pi/360
dL/dzero = (X sec(theta) tan(theta) + Y sec(theta)^2) pi/360.
```

Pull these rows back through the complete physical-to-free derivative matrix.
Also include every finite physical box bound, including affine dependent
parameters: `D_i d >= lower_i-p_i`, `-D_i d >= p_i-upper_i`.
With zero shift these width constraints are only a local linearization.
All trials still undergo the complete existing profile/domain checks and
strict actual objective-descent acceptance. Profile equations, analytical
profile derivatives, finite support and sample weights are unchanged.

## Bounded implementation and fallback

The active-set subproblem starts from the feasible zero step, whitens variables
by the square roots of the Hessian diagonal, and normalizes constraint rows.
A QR null space of active rows provides the tangent direction; Cholesky solves
the reduced Hessian. Blocking rows enter the working set and negative dual
multipliers leave it. The returned step must pass primal and stationarity
residual checks. Numerical safeguards include a 1e-10 QR rank threshold,
1e-9 relative direction threshold, 1e-8 dual threshold, and 1e-10 scaled
primal tolerance. Stationarity uses the caller's `cg_tolerance`.

This implementation is bounded to dense fits with at most 64 free parameters
and at most 2,048 reflections plus physical parameters (at most 4,096 rows).
It requires some refined UVWXY width and fixed cells, wavelengths and
displacements; an additive zero shift is supported. Matrix-free, larger and
unsupported position models retain the ordinary solver. Singular or
indefinite tangent Hessians, dependent active rows, failed residual checks,
or more than `20*(p+1)` active-set iterations also fall back. No pseudoinverse,
extra dependency or unbounded search is introduced.

The normal step cap and backtracking remain in force. The candidate-stop
verification also tries an effectively undamped constrained subproblem
(`lambda=1e-18`) before the established recovery checks. Remaining stagnation
is reported, not converted into convergence. Runtime limits still govern
model evaluations and outer iterations; the small subproblem has its own
fixed iteration cap.

The option persists through native and Python project files; missing fields
in old projects mean False. Checkpoints need no additional state. Use the same
option and controls for exact resume; changing it intentionally changes the
subsequent search. Separate legacy and TOF solvers are unchanged.

## Validation and QARR findings

Rust and independent NumPy subproblems are checked against exhaustive active
set KKT solutions on 32 deterministic positive-definite cases. Tests also
cover centered finite differences for width/zero/affine rows, monochromatic
and fixed-doublet profiles, dependent physical bounds, unsupported and
unverifiable fallbacks, native/Python synthetic recovery, exact checkpoint
continuation, project persistence and unchanged ineligible solves. Existing
profile, derivative, normalization and pinned oracle fixtures are retained;
there is no new forward equation requiring regenerated oracle data.

The real-data diagnostic retains the original width model, full profiles,
support 30, weights, selections, physical bounds and scientific gates. It
compares the existing verified-stop/recovery solver with this new option.
All Rwp numbers below are percentages; composition errors are maximum absolute
weight-fraction errors in percentage points.

| Search policy | 1g Rwp | 1g composition error | 1h Rwp | 1h composition error |
| --- | ---: | ---: | ---: | ---: |
| Default | 19.690019 | 0.597903 | 19.689517 | 1.411222 |
| Feasible steps in all stages | 19.676589 | 0.600633 | 19.908181 | 1.301200 |
| Feasible steps only in joint stage 2 | 19.690119 | 0.596558 | 19.689482 | 1.412861 |
| Tight feasible continuation of default endpoint | 19.690019 | 0.598053 | 19.689469 | 1.411094 |

The all-stage experiment improves 1g but worsens final 1h Rwp despite improving
its initial fit. That initial stage's background is frozen afterward. Enabling
only stage 2 preserves it and changes final fit quality very little.
Continuation accepts only improvements from the existing default endpoint,
but its gains are tiny. All four profile/composition checks pass in these
experiments. The joint stages still report **stagnated**; no claim of full
constrained convergence or global optimality is supported.

Model evaluation counts also show why this is not a universal optimization:

| Policy | 1g total evaluations | 1h total evaluations |
| --- | ---: | ---: |
| Default | 7,340 | 4,541 |
| Feasible steps in all stages | 3,166 | 1,800 |
| Feasible steps only in joint stage 2 | 4,572 | 5,825 |

All-stage proposals reduce evaluations by roughly 57–60%, but change the
final attained quality. Stage-two-only proposals reduce 1g work while
increasing 1h work. Constrained proposals also cost more per linear solve.
Do not equate these evaluation reductions with wall-time speedups.

Two timing-only repetitions per policy, eight workers, release build, no
warmup repetitions, include preparation, all stages, final covariance and
reporting. Numerical outputs repeat exactly:

| Policy | 1g seconds | 1h seconds |
| --- | ---: | ---: |
| Default | 30.46–33.14 | 18.58–20.44 |
| Feasible steps in all stages | 11.61–12.55 | 5.85–6.46 |
| Feasible steps only in joint stage 2 | 13.68–14.86 | 19.70–23.43 |

Policies ran sequentially on a shared machine. Validation, other project
builds and system workloads overlapped parts of the sequence, so these are
exploratory ranges, not controlled speedup ratios. The all-stage 1h timing
also ends at a worse Rwp and cannot establish a speed win at equal accuracy.
The stage-two-only policy takes more evaluations and time on 1h.

Retained release-build evidence:

- `validation/results/feasible-width-all-stages-20260917.json`: original
  real-data fits and coordinate probes, including exact one/eight-worker
  numerical repeatability. Its times include probes and concurrent validation;
  they are not isolated timing measurements.
- `validation/results/feasible-width-continuation-20260917.json`: tighter
  continuation, with strict non-increasing actual objective.
- `validation/results/feasible-width-timing-{default,joint,all}-20260917.json`:
  timing-only repetitions and all stage outcomes, including the joint-only
  comparison in the tables above.
- `validation/results/feasible-width-default-regression-20260917.json`:
  scientific reports are identical to the pre-option build for the tested
  explicit-execution QARR 1g/PbSO4 calls and native QARR 1h. This verifies that
  the disabled option preserves those workflows. It does not clear the earlier
  stopping/recovery changes: the later consolidated assessment identifies a
  canonical native QARR 1g termination regression against the pre-recovery build.

Final checks: `cargo fmt`, warning-free all-target/all-feature Clippy,
346 passing Rust tests (34 ignored), and 881 passing Python tests (11 skipped,
33 deselected). The public API snapshot changes only to append the optional
boolean with default False.

Reproduce the standard-policy comparison with
`benchmarks/qarr_convergence.py --case standard --feasible-stages 1 2 3`
(or `--feasible-stages 2`, or omit it for default). Use
`benchmarks/qarr_convergence_polish.py --feasible-width-steps` for continuation.
Both require a release build, the registered QARR datasets, an output path and
BLAS/OpenMP thread variables set to 1. Coordinate probes (`--probe`) are
diagnostic work and must not be counted as timing-only benchmarks.
