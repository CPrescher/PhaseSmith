# Solver isolation on the real QARR dataset

On 2026-09-17, SciPy trust-region reflective least squares was compared with
the native PhaseSmith solver on the **same PhaseSmith objective**. A direct
replacement did not reproduce rietx's residual advantage. The evidence points
to width parameterization and coupled numerical-domain constraints as the
priority, while leaving room to improve the native solver's recovery behavior.

## Experimental contract

The original QARR recipe was run once for each profile policy to capture the
input to all three stages. Each subsequent pair of solver runs used the exact
same captured stage input. Stage two therefore starts from the original native
stage-one result for both solvers; this is **not** a sequential SciPy workflow.

Both solvers use:

- The same 7,251 observations, uncertainties, mask, fixed/residual background,
  structures, reflection domain, Friedel averaging and wavelength components.
- The same PhaseSmith profile policy, either default or both optional speed
  controls. No rietx forward calculations or fitted parameters enter the test.
- The same scaled physical parameters (`physical value / ParameterSpec.scale`),
  initial vector, free variables, and lower/upper bounds. The driver checks
  native-returned parameter contracts against the adapter's input contract.
- Rust profile/structural calculations and analytical derivatives. The SciPy
  adapter assembles the existing PhaseSmith weighted Jacobian in sample-major
  order and checks it using centered finite differences.
- A 300-evaluation ceiling, with 300 as the native iteration/rejection ceiling
  so the original 8/28/10 budgets and 20-rejection guard are not the comparison's
  dominant stopping rule. Native stage-specific step caps remain solver controls.

The primary SciPy call uses `method="trf"`, `x_scale=1`, `ftol=1e-10`,
`xtol=1e-7`, and disables `gtol` because the native algorithm does not use an
equivalent gradient stopping rule. The objective/step tolerances match the
native numerical settings, but the algorithms' stopping formulas and evaluation
accounting are not identical. Tighter stopping tolerances are tested separately.
No equal-time or equal-work claim follows from these counters.

PhaseSmith's valid domain is stricter than its scalar parameter bounds:
instrument Gaussian variance must remain positive at every evaluated reflection.
For a rejected-domain SciPy proposal, the adapter returns an infinite trial
residual. SciPy's existing TRF implementation rejects such a proposal and shrinks
its trust radius; no derivative is requested there. Finite valid residuals and
derivatives are unchanged. No variance clipping, synthetic finite penalty, or
additional fitted parameter is introduced. This is a straightforward domain
rejection adapter, **not** a solver that represents coupled width inequalities
explicitly inside its trust-region subproblem.

## Results

These are final **isolated-stage** Rwps, not complete independently staged
refinements. All rows use 300 as the evaluation ceiling; a solver may stop
earlier on its own convergence criterion.

| Profile policy / isolated stage | Native PhaseSmith Rwp | SciPy TRF Rwp |
|---|---:|---:|
| Default, stage one | 26.3882% | 25.7419% |
| Default, stage two | 19.8776% | 25.8992% |
| Default, scale polish | 19.8855% | 19.8855% |
| Both speed options, stage one | 26.3958% | 25.7493% |
| Both speed options, stage two | 19.9825% | 25.8439% |
| Both speed options, scale polish | 19.9888% | 19.9888% |

Native stage one hits the 300-evaluation ceiling for both policies. Default
stage two also reaches that ceiling; combined-policy stage two reports
convergence after 61 evaluations. Both native scale-only runs converge in two
evaluations.

SciPy's nonlinear runs stop on step tolerance after 28–46 residual evaluations.
They contain 18–23 invalid-width trial proposals. Its scale-only runs contain
no invalid trials and agree with native Rwp to floating-point precision.
Its reported stopping status should not be interpreted as proof of an optimal
fit under the coupled numerical-domain constraints.

The adapter reproduces native final calculated profiles with relative L2
differences below 2.1e-16, and final Rwp differences below 6.2e-16. Maximum
initial analytical-Jacobian versus centered-difference column error is
7.87e-7 relative, below the diagnostic's local 1e-5 limit. This checks every
free column, rather than relying on a few selected derivatives. Finite support
can make such checks discontinuous at other points; this is a local check at
the captured starting states, not a global smoothness claim.

## Controls

1. **Tighter stopping tests:** `ftol=xtol=gtol=1e-12` leaves SciPy's nonlinear
   Rwps essentially unchanged. Default stage two remains 25.8992%; the combined
   policy remains 25.8439%. The coarse stopping tolerance is not the main
   explanation for the failed replacement.
2. **Automatic scaling and alternative linear algebra:** Jacobian-based
   scaling and TRF's regularized LSMR subsolver were each tested separately.
   Default stage-two Rwps are 26.3598% and 26.0991%; combined-policy values are
   26.3632% and 26.0301%. Neither recovers the native result in this adapter.
3. **Fixed-instrument control:** the same U/V/W and zero-shift variables are
   removed from both solvers' stage-two selection, with their values held at
   the shared starting state. The remaining 15 variables and objective are
   identical between the two runs. No invalid-width proposals occur. Default
   Rwp becomes 20.19947% native versus 20.19934% SciPy; combined-policy Rwp is
   20.2676320156844% versus 20.2676320156825%. The restricted problem is not
   proposed as a better fit—it is a control showing close agreement once the
   problematic instrument motion is removed.

The original stage-two weighted Jacobian has a smallest singular value of
about 1e-15. This is consistent with the known instrument-U/sample-strain
ambiguity: their coefficients all contribute to tan²(theta). Both the exact
redundancy and instrument-only positivity constrain how a solver can move,
even though the initial profile and scalar bounds match.

Raw gradient norms and singular values are retained in the records. Raw
gradient size alone does not establish constrained nonstationarity at a
domain boundary; it is diagnostic information, not a replacement for a
feasible/projected optimality test.

## Consequence for implementation

This does not establish that either optimizer is universally better, nor that
PhaseSmith's solver needs no improvement. It does rule against treating the
observed rietx advantage as evidence that adopting its default SciPy optimizer
alone will fix PhaseSmith's current recipe.

The next experiment should express a valid, identifiable width model to both
solvers: remove or anchor the common instrument/sample contribution, and make
coupled positivity part of the parameterization or constrained step calculation.
Then repeat the same-objective comparison before choosing a production solver
change. Background staging and phase-fraction accuracy remain separate checks.

## Artifacts and reproduction

The [driver](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/benchmarks/compare_solvers_qarr.py) uses SciPy only in the
optional benchmark environment. Production code and dependencies are unchanged.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/compare_solvers_qarr.py --suite primary --budget 300 \
  --json-output primary.json
```

Repeat with `--suite controls` and `--suite scaling`. Recorded artifacts:

- [Primary comparison](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/solver-isolation-20260917.json)
- [Tolerance and fixed-instrument controls](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/solver-isolation-20260917-controls.json)
- [Scaling and linear-solver sensitivity](https://github.com/CPrescher/PhaseSmith/blob/8d9d86e44901c336a6b0a6e7c35ea8ad82d6da6d/validation/results/solver-isolation-20260917-scaling.json)

Records include dataset hashes, the installed native-extension hash, SciPy/
NumPy versions, individual stage results and rejected trial reasons. Timings
are diagnostic only: the adapter prepares dense derivatives on every valid
SciPy trial and is not a production-throughput implementation.
