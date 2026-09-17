# Powder intensities with anomalous scattering

Powder pattern calculations now use the mean intensity of the two opposite
reflections in each Friedel family. This corrects the previous use of one
representative's intensity with the merged multiplicity. It is mandatory
physics, independent of `ProfileAccuracy`.

For real displacements and scalar scattering factors depending on scattering
vector magnitude, write the site orbit sum as S_j(h). Then S_j(-h) is the
complex conjugate of S_j(h), while the complex atomic factor f_j is unchanged.
The independently derived intensity convention is

    F(h) = sum_j occupancy_j * f_j * S_j(h)
    P(h) = (abs(F(h))^2 + abs(F(-h))^2) / 2
    I(h) = scale * multiplicity * correction * P(h).

Multiplicity includes the whole represented family; it is never doubled by
averaging. An unmerged list containing both mates gives the same total powder
intensity when each mate has its own crystallographic multiplicity and the
same scalar correction. Powder entries describe family-averaged intensities;
use the individual structure-factor API for separate anomalous intensities.

The implementation evaluates the second amplitude using conjugated atomic
factors: abs(sum_j conj(f_j) S_j(h))^2 = abs(F(-h))^2. The same operation applies
to df/ds. Every intensity derivative is the arithmetic mean of the two
individual-reflection derivatives, including cell, coordinates, occupancy,
displacement, scale and correction chains. Dense, selected, JVP and VJP paths
use the same convention. Complex F and its derivatives remain those of the
representative: under powder averaging, `f_squared` need not equal `abs(f)^2`.

The existing individual-reflection Rust/Python APIs preserve their behavior.
Low-level Python `calculate_structure_factor_values` and
`calculate_structure_factors` accept `powder_average=True` when family
intensities are wanted. Structural CW and TOF engines and the Python provider
fallback always select powder intensities. This is not a user-selectable
accuracy approximation in a Rietveld fit.

Centrosymmetric groups and entirely real scattering batches retain their
existing arithmetic without an extra evaluation. A nonzero imaginary df/ds
also activates the derivative correction even where the imaginary amplitude
itself is zero. For the affected batches the additional work is at the
reflection/site level; peak evaluation and fused sample accumulation still
run once. Finite-support conventions and peak normalization are unchanged.

The independent NumPy reference sums both opposite orbit phases explicitly.
Tests compare randomized non-centrosymmetric cases, centered finite differences,
dense/selected and matrix-free products, sign reversal, inversion symmetry,
zero dispersion, zero scale, masks, checkpoint continuation and native/provider
paths. No GSAS-II implementation is used and no golden fixture is regenerated:
the missing anomalous-pair case is validated against the explicit independent
Fourier sum, with existing oracle fixtures retained as regression coverage.

This numerical correction can change fitted states and optimizer trajectories
in old anomalous non-centrosymmetric projects. Recalculate them and start a
fresh refinement from their accepted parameters; previously published benchmark
Rwps describe the old calculation. Cross-version exact continuation is not
promised. Corrected real-data benchmarks are recorded separately.

## Corrected initial-profile comparison (2026-09-16)

The production result reproduces the earlier diagnostic pair average. Input
scales are now also initialized with the corrected calculation, accounting for
the small difference between the diagnostic's 0.0885% and this run's 0.0887%.

| PhaseSmith policy | rietx policy | Initial relative L2 difference | 0.1% gate |
|---|---|---:|---|
| Default | Default | 0.4318% | Fail |
| Default | Retain small axial terms, 30-FWHM windows | 0.0887% | Pass |
| Fast FCJ + 1% tail budget | Default | 0.3555% | Fail |
| Fast FCJ + 1% tail budget | Retain small axial terms, 30-FWHM windows | 0.1893% | Fail |

With the default PhaseSmith policy and the adjusted rietx policy, per-phase
differences are Al2O3 0.3463%, ZnO 0.0543%, and CaF2 0.0524%. The total gate
uses the norm of the summed three-phase pattern, so an individual phase can
exceed it while the combined pattern passes. The residual Al2O3 discrepancy
has not been fully attributed. Matching the support multiplier does not make
the per-node and frozen-window support conventions identical.

Reproduce with `benchmarks/friedel_equivalence.py --json-output forward.json`.
The [recorded forward results](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-friedel-corrected-forward.json)
include input hashes and both profile policies. The successful row establishes
the initial forward gate only, not final-fit equivalence or identical math.

## Corrected measured-data fits

Release build on the same Apple M4 Pro host, one warmup and five measured
runs per policy at one, two and eight workers, alternating order; BLAS/OpenMP
workers fixed to one. Scientific records repeat exactly within each policy
and across worker counts. Existing acceptance gates remain unchanged.

| Policy | 1 worker | 2 workers | 8 workers | Poisson Rwp | Maximum phase error |
|---|---:|---:|---:|---:|---:|
| Default | 0.808 s | 0.591 s | 0.350 s | 19.8855% | 0.713 percentage points |
| Fast FCJ only | 0.586 s | 0.474 s | 0.312 s | 19.9189% | 0.708 pp |
| 1% tail budget only | 0.561 s | 0.425 s | 0.290 s | 19.9828% | 0.654 pp |
| Both | 0.437 s | 0.359 s | 0.267 s | 19.9888% | 0.656 pp |

All pass the existing QARR checks, but stages one and two stop at their
iteration budgets; only scale polish reports convergence. The tail-budget
cases are close to the 20% Rwp acceptance limit. Correcting the physics does
not guarantee a smaller residual under the old bounded optimizer recipe.

The default fitted fractions are Al2O3 30.848%, ZnO 34.019%, CaF2 35.133%;
with both optional approximations they are 30.931%, 33.994%, 35.076%.
The weighed values remain 31.370%, 34.210%, 34.420%. These replace the
pre-correction QARR figures; no old results were overwritten.

PbSO4 X-ray and neutron scientific records are unchanged by Friedel averaging,
as expected for these unaffected cases. Default Poisson Rwp is 10.3446% and
4.2172%, respectively. All four accuracy policies pass their existing gates.
The combined-policy X-ray fit still stops with `repeated_rejections`, which
its existing acceptance rule permits after sufficient improvement; this is
not reported as convergence. Its Rwp is 10.3466%; the combined neutron fit
converges at 4.2889%.

The [corrected accuracy sweep](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-friedel-corrected-accuracy.json)
contains all repetitions, stage timings, scientific checks, termination reasons
and data hashes. Reproduce with `benchmarks/profile_accuracy.py --repetitions 5
--json-output corrected.json` in a release environment.

A separate interleaved comparison against the retained pre-correction release
wheel measured default QARR times of 0.7890 -> 0.7950 s at one worker and
0.3449 -> 0.3487 s at eight: approximately 0.8% and 1.1% additional time in
that run. These are small whole-workflow differences, including the changed
physical intensities and fit trajectory, rather than isolated kernel timings.
The [before/after record](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-friedel-before-after.json)
retains both scientific results. The optional profile speed controls remain
available on top of the corrected powder intensities.

Validation: 342 Rust tests passed (34 ignored); 844 Python tests passed
(11 skipped, 33 deselected). The final reference/ABI check reran the 20 targeted
Friedel and TOF-reference tests successfully. Workspace/all-feature Clippy,
formatting, Ruff and the public API snapshot checks pass.

## Paired rerun and actual worker use

The corrected, interleaved ten-repetition comparison uses both optional
PhaseSmith speed controls and rietx 1.4.0 defaults. Imports and JIT compilation
are warmed outside timing. Worker counts below are requested budgets, not
claims about the number of active cores.

| Requested workers | PhaseSmith median | rietx median | PhaseSmith Rwp | rietx Rwp |
|---|---:|---:|---:|---:|
| 1 | 0.4217 s | 0.4082 s | 19.9888% | 19.2601% |
| 8 | 0.2493 s | 0.3999 s | 19.9888% | 19.2601% |

The benchmark now makes shared initial-scale preparation honor the requested
budget, using a shared execution policy. Earlier records without
`shared_preparation_threads` used PhaseSmith's default two workers during
this preparation, including inside the rietx workflow. Those records are
therefore not strictly single-worker whole-workflow measurements. The updated
benchmark retains the same input model, and its complete scientific records
and equivalence checks exactly match the earlier corrected-physics runs at
both worker counts. See the [one-worker record](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-controlled-paired-1worker.json)
and [eight-worker record](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-controlled-paired-8workers.json).

Both workflows pass their existing scientific acceptance gates. Strict
cross-engine equivalence still fails: initial relative L2 difference is
0.3555% against a 0.1% limit, and final Rwp differs by 0.7287 percentage
points against a 0.5-point limit. These are bounded workflow timings with
different profile policies and optimizer behavior, not equal-quality or
identical-math throughput measurements.

### Why rietx does not scale here

The installed rietx 1.4.0 `model/compiled.py` uses serial Numba kernels and
dispatches disjoint row ranges through a shared thread pool only when a
bucket has at least 512 rows. Scatter accumulation stays serial. Runtime
instrumentation of this real-data workflow observed only 36-, 58-, and
128-row batches, with 99 calls at each size during a complete warmup workflow.
At both one and eight requested workers there were zero pool requests; no
rietx kernel pool existed after the ten measured workflows either.

Measured process CPU time divided by elapsed time after shared preparation
was 0.9997 at one requested worker and 0.9996 at eight (median of ten runs).
Because process CPU time sums time across threads, this supports approximately
one busy core during rietx refinement at both settings. It does not establish
CPU affinity or rule out every brief overlap. BLAS/OpenMP environment limits
were set to one. `threadpoolctl` found no recognized pools; NumPy uses Apple's
Accelerate library here, so that empty inventory alone is not proof of
single-thread execution.

The small timing difference between rietx's one- and eight-worker runs is
consistent with run-to-run variation on the same serial path, not evidence
of parallel overhead or scaling. Larger batches can take its parallel path;
this result is specific to the QARR case.

The [audit script](https://github.com/CPrescher/PhaseSmith/blob/main/benchmarks/audit_rietx_threads.py) records source hashes,
dispatch counts and individual CPU/wall measurements. Its
[one-worker](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-thread-audit-1worker.json) and
[eight-worker](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-thread-audit-8workers.json)
records measure preparation separately, preserving the original default
two-worker preparation for diagnosis. Run each setting in a fresh process:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/audit_rietx_threads.py --threads 1 --repetitions 10 \
  --json-output audit-1worker.json
```

The optional benchmark environment additionally needs `threadpoolctl`.
Dispatch instrumentation is confined to warmup; timed workflows retain the
original rietx dispatcher. Ruff and diff checks pass for these benchmark-only
changes; no production numerical behavior was modified by the thread audit.

## Rwp attribution experiments

Fresh corrected-physics diagnostics isolate fit quality from speed. The
19.9888% versus 19.2601% Rwp gap is 0.7287 percentage points, about 3.78%
relative Rwp or 7.71% more weighted residual sum of squares for PhaseSmith
(same observed data and weights). It is not an equal-quality speed comparison.

| Diagnostic | Final Poisson Rwp |
|---|---:|
| PhaseSmith default, existing recipe | 19.8855% |
| PhaseSmith both speed options, existing recipe | 19.9888% |
| PhaseSmith default, 8x stage-one/two iteration limits | 19.8776% |
| PhaseSmith both options, 8x iteration limits | 19.9825% |
| rietx default | 19.2601% |
| rietx retaining axial corrections and 30-FWHM windows | 19.3119% |
| PhaseSmith default, extended stage two also frees background | 18.9117% |
| PhaseSmith both options, extended stage two also frees background | 18.8716% |
| PhaseSmith default, stage two seeded from reparameterized rietx stage one | 19.3396% |
| PhaseSmith both options, same seed intervention | 19.3407% |

The speed controls account for 0.1033 percentage points under the existing
recipe. Extending iteration limits alone barely helps; the default extended
run reaches repeated rejections, and the combined extended run satisfies its
objective-change convergence criterion. This does not prove stationarity.
Changing rietx's two profile policies also leaves most of the original gap.
Both extended-budget workflows fail the existing termination acceptance
check because stage one reaches repeated rejections; their lower residuals
are diagnostic values, not accepted replacements for the original recipe.

The first-stage problem is much larger: PhaseSmith reaches 26.3906% Rwp
(default), versus rietx's 21.4893%. Both recipes freeze the fitted background
after this stage. Their three residual background coefficients are respectively
approximately (0.0407, -0.0558, 0.0133) and (3.2901, -4.0014, 0.7073).
Allowing PhaseSmith's background to move during an extended second stage
lowers its residual below the original rietx recipe. These are **22-parameter
second stages instead of 19**, take more iterations, and stop with
`repeated_rejections`; only subsequent scale polish converges. They demonstrate
a staging limitation, not a new equal-workload win or a fully converged fit.
Both free-background workflows also fail that existing termination check;
their other scientific checks pass. The seed experiments pass all existing
checks, but remain hybrid workflows rather than independent PhaseSmith fits.

There is also a real difference in feasible parameter decompositions.
PhaseSmith validates positive instrument-only Gaussian variance before adding
sample broadening. rietx adds instrument and sample terms first, then floors
the combined Gaussian FWHM squared at 1e-8 deg². Its first-stage fit has
instrument U=-0.0398428 deg² and sample Gaussian strain coefficient
0.0466016 deg² for each phase. Instrument-only variance is negative at all
222 reflection/line rows, but **combined variance is positive at every row**.
The floor is unused in the fitted states of all three stages; it must not be
blamed for the residual advantage. The final instrument-only variance is
negative at 202 of 222 rows, with sample broadening again making all totals
positive.

This is partly an identifiability issue: instrument U and isotropic sample
strain both multiply tan²(theta). The fitted widths can be represented by
setting instrument U to zero and adding its former negative value to each
sample strain coefficient, which remains positive here. This preserves total
Gaussian widths and yields a valid PhaseSmith decomposition; it does not
establish independently measured instrument resolution or microstrain.

The seed experiment uses precisely that transformation of rietx's first-stage
fit, and transfers its scales, zero and background into the start of PhaseSmith
stage two. The remaining PhaseSmith budgets stay 28/10. The resulting 19.3407%
combined-policy Rwp closes roughly 89% of the original gap. This identifies
the first-stage outcome and staging as major contributors. It does not isolate
one transferred parameter or prove that all remaining differences are solver
errors. It is a hybrid diagnostic, not a standalone PhaseSmith benchmark.

Other remaining differences include trust-region reflective optimization
versus scaled damped steps/backtracking, parameter transformations and bounds,
frozen versus parameter-dependent profile windows, component-dependent size
broadening, and 222 versus 220 reflection/line rows. Equal `max_iter` values
also mean different budgets: rietx permits four residual evaluations per
nominal iteration. Both fits have the same parameter-role counts, 10/19/3,
in the original recipe.

The next quality-focused change should address initialization, instrument/
sample width identifiability and when background is frozen, then compare time
to a shared Rwp target under a documented common model. Another profile-kernel
speedup would not resolve the demonstrated staging issue.

Reproduce with [investigate_rietx_rwp.py](https://github.com/CPrescher/PhaseSmith/blob/main/benchmarks/investigate_rietx_rwp.py)
in the optional benchmark environment, with BLAS/OpenMP environment limits
set to one. The [diagnostic record](https://github.com/CPrescher/PhaseSmith/blob/main/validation/results/rietx-20260916-rwp-diagnostic.json)
contains individual stage histories, acceptance checks, fitted parameters,
width checks and dataset hashes. Timings are diagnostic only; no production
numerics, recipe defaults or acceptance thresholds were changed.
