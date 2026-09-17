# Empirical Gaussian instrument/sample decomposition

`EmpiricalGaussianConvention` is an explicit Python preparation step for CW
Rietveld input when an independent instrument calibration is unavailable. It
anchors one phase's isotropic RMS strain and transfers the shared Gaussian
variance into instrument U. The starting calculated profile is preserved; the
reference strain remains fixed when subsequent stages select sample physics.
Existing defaults and solver policies do not change.

```python
import phasesmith as ps

convention = ps.EmpiricalGaussianConvention(
    reference_phase_id="ZnO",
    reference_rms_microstrain=2e-4,
)
request = convention.apply(request)
project = ps.RietveldProject(request)
result = project.refine()
```

The reference ID and strain are required decisions. The value `2e-4` above is
an illustrative convention, **not a measured ZnO strain or a recommendation for
other samples**. Use independent instrument calibration when absolute sample
broadening matters. Choosing an empirical reference changes the feasible model
and potentially the fitted phase fractions; inspect sensitivity to the reference.

## Equation and interpretation

In PhaseSmith's degree-squared Gaussian **variance** convention, the U and
isotropic-strain terms for phase p are

\[
\sigma_{G,p}^2(\theta)
 = (U + C\epsilon_p^2)\tan^2\theta + V\tan\theta + W,
\qquad C=(2\cdot180/\pi)^2.
\]

Here epsilon is dimensionless RMS delta-d/d, not Gaussian FWHM or rietx's
Gaussian width coefficient. The factor follows by differentiating Bragg's law:
`delta(2 theta) = -2 tan(theta) delta(d)/d` in radians. Independent Gaussian
variances add. Other width terms, if present, are untouched.

For reference phase r, initial strain epsilon-r and chosen anchor a, apply

\[
U' = U + C(\epsilon_r^2-a^2),\qquad
\epsilon_p'^2 = \epsilon_p^2-\epsilon_r^2+a^2.
\]

Thus `U' + C epsilon-p'^2 = U + C epsilon-p^2`. In an unconstrained simultaneous
U/strain fit, this shared coefficient admits a redundant direction. Fixing the
reference strain removes that particular direction; it does not guarantee
identifiability of all other parameters.

The implementation evaluates differences of squares as `(x-y)*(x+y)` and sets
the reference strain exactly to a. The anchor must have a positive finite
square. A positive anchor also avoids initializing all initially equal-strain
phases at zero, where the RMS-strain derivative vanishes. Nonreference phases
may have zero strain; their existing nonnegative bound and zero derivative
still apply.

## Domain, constraints and numerical behavior

- Every phase must have exactly one built-in `IsotropicMicrostrainBroadening`,
  directly or in a flat `CompositePhysicsProvider`.
- Reject a reference that would give any phase negative strain variance.
  No width floor or variance clipping is introduced.
- The normal CW instrument-width domain still applies. The transformed model
  is preflighted through the existing calculation, and an invalid instrument
  profile is rejected even if sample broadening could compensate for it.
- Existing constraints touching U or transferred strains, and custom parameter
  scales/bounds/refine settings for those variables, are rejected. Unrelated
  parameter settings and constraints survive unchanged. Canonical parameter
  scales are rebuilt for the transferred values.
- The reference gets a normal `FixedConstraint` whenever its strain parameter
  is selected. Stage selection may remove and restore the parameter; the
  convention persists and reinstates the constraint. Conflicting constraints
  are errors. Reapplying the same convention is a no-op; silently changing the
  reference on an already labelled input is rejected.
- Production values and analytical derivatives still use the existing Rust
  kernels and constraint transform. No alternative solver or profile term is
  added, and no GSAS-II implementation is used.
- Finite-support rules remain those of the selected profile accuracy policy.
  The unchanged total starting widths preserve support endpoints to floating
  point rounding. Existing endpoint inclusion semantics still apply; this
  feature does not promise differentiability at moving support boundaries.

This is a Python domain-translation API; the Rust numerical core continues to
receive ordinary parameters and fixed constraints.

## Persistence and reporting

The convention travels with `RietveldInput`, checkpoints and recipe workflows.
Resume rejects a different or missing convention. `RietveldProject.save/load`
preserves it on both native monochromatic and generic spectrum paths. Native
projects use the existing string metadata map under
`phasesmith.empirical_gaussian`; the value is a versioned JSON record. The
numerical fixed constraint is also stored normally. Generic checkpoints carry
the record explicitly. Existing projects without it retain their old behavior.

The result JSON, readiness review and fit-report advice state the empirical
interpretation. Instrument Gaussian widths, sample strains and conditional
uncertainties must not be presented as independent measurements. For custom
serialization outside `RietveldProject`, preserve `convention.to_record()` and
restore it with `EmpiricalGaussianConvention.from_record()` when constructing
an input or checkpoint.

## Measured QARR 1g benchmark

The benchmark uses the existing measured 7,251-point, three-phase, Cu doublet
QARR acceptance case. ZnO is the reference with RMS strain fixed to 0.0002.
Shared background/scale preparation honors the requested worker count in both
libraries. Timings include complete preparation, transfer/preflight, refinement
and QPA, with one warmup and three alternating-order timed runs per case.
BLAS thread limits are one; rietx is pinned to 1.4.0 with compiled kernels.
Worker budgets are not measurements of simultaneously active cores.

The `fast` policy enables FCJ small-span quadrature and 0.01 tail-area tolerance.
The normal bounded recipe uses its existing 8/28/10 iteration budgets. Extended
cases explicitly allow 100 iterations, 1,500 evaluations and 200 consecutive
rejections in each stage; these are experimental recipe settings, not defaults.

| Workflow | Rwp (%) | Max phase-fraction error (pp) | 1 worker (s) | 8 workers (s) | Dataset gates |
|---|---:|---:|---:|---:|---|
| Baseline, default profiles | 19.8855 | 0.7129 | 0.808 | 0.351 | pass |
| Baseline, fast profiles | 19.9888 | 0.6558 | 0.439 | 0.262 | pass |
| Empirical, default profiles | 19.6301 | 0.7289 | 0.666 | 0.292 | pass |
| Empirical, fast profiles | 19.6776 | 0.6909 | 0.385 | 0.227 | pass |
| Empirical extended, default | 19.2199 | 0.8730 | 3.564 | 1.182 | pass |
| Empirical extended, fast | 19.2322 | 0.8354 | 3.157 | 1.335 | **fail: rejection termination** |
| rietx original recipe | 19.2601 | 0.9125 | 0.427 | 0.406 | pass |

Both ordinary empirical cases use 9/29/2 evaluations. Their first two stages
end at intentional iteration limits, as in the baseline. The extended default
case converges in all stages, using 330/45/1 evaluations. Extended fast uses
275/374/2, and its first stage ends in repeated rejections; its otherwise good
profile/QPA metrics do not override the failed termination gate.

For this case, the normal empirical recipe improves runtime and Rwp relative
to the matching baseline policy, with slightly larger phase-fraction error.
Extended default obtains a slightly lower Rwp than rietx, but is much slower.
No equal-model speed claim follows: conventions, constraints, profile policies
and stopping behavior differ. The existing stricter cross-engine forward
profile equivalence failure is not waived by this feature. Scientific results
are deterministic across repeats and both worker budgets.

Reproduce in the existing optional benchmark environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/empirical_gaussian_qarr.py --threads 8 --repetitions 3 \
  --json-output validation/results/empirical-gaussian-20260917-8workers.json
```

Use `--threads 1` for the serial budget. Raw results, environment details and
dataset checksums are retained in `validation/results/empirical-gaussian-20260917-*.json`.

## Validation

Tests cover randomized coefficient/profile invariance for monochromatic and
fixed-doublet inputs, direct/composite providers, integrated grid intensity and
moments, invalid references and conflicting settings, stage reselection,
native/Python deterministic continuation, both project formats, and disclosed
interpretation in reports. A full-profile Jacobian test demonstrates removal
of the U/strain null direction and checks remaining analytical columns with
centered finite differences away from support boundaries. Existing profile
oracles remain applicable: this is an algebraic reparameterization using the
same kernels, so no new oracle fixture or golden-data regeneration is needed.

## Follow-up: scaling at fixed converged quality

A fresh sweep uses the extended default-profile recipe at exactly the same
19.2199357747% Rwp and 0.8730298168 pp maximum phase-fraction error, with all
three stages converged. One warmup and three timed runs per worker budget use
alternating order; shared preparation uses the same budget and BLAS stays at
one. Entire scientific records match exactly across every run and budget.

| Workers | Median full workflow (s) | Speedup over one worker |
|---:|---:|---:|
| 1 | 3.644 | 1.00x |
| 2 | 2.490 | 1.46x |
| 4 | 1.495 | 2.44x |
| 8 | 1.111 | 3.28x |

Two workers reduce elapsed time by about 32%, but moving from two to eight
reduces it by another 55% on this workload. Two remains a conservative resource
budget rather than the fastest measured setting. Parallel work includes phase,
wavelength-component and reflection calculations through the shared Rust pool;
ordered accumulation preserves deterministic results. Preparation, reductions
and optimizer control limit scaling. Requested workers do not imply that every
core stays busy throughout the fit.

This fresh sweep should be compared internally, not mixed with timings from the
preceding run. Reproduce with `benchmarks/empirical_gaussian_scaling.py
--json-output validation/results/empirical-gaussian-20260917-quality-scaling.json`
in the same environment and with the three numerical-library thread limits
set to one. The raw record contains all samples, stage limits and results.
See also [rietx multicore](rietx-multicore.md) for the forced-thread control.

Later [rejected-step recovery improvements](refinement-quality-recovery.md)
compare this same converged model at unchanged numerical settings. The timing
tables above describe the preceding solver build.
