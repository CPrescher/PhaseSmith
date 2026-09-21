# Fit evidence and advisory review

`phasesmith.build_fit_report(result, original_pattern)` reviews an accepted
CW Rietveld result without recalculating it or changing any parameter.
`RietveldProject.fit_report()` provides the same operation for the last fit.

```python
result = project.refine()
report = project.fit_report(region_count=20)
for region in report.residuals.worst_regions(3):
    print(region.lower, region.upper, region.chi_square_fraction)
for action in report.advice:
    print(action.code, action.message, action.parameter_labels)

import json

print(json.dumps(report.to_record(), allow_nan=False))
```

The separately versioned `phasesmith.fit-report.v1` record does not change the
existing result-report or automation schemas. No advice changes the authorized
parameter selection, approves a plan, or executes another fit.

## Equations and conventions

Let `r_i = calculated_i - observed_i`, and let `z_i` be the actual weighted
residual used by the fit (`r_i / sigma_i`, or `r_i` for unit weighting).
Only included observations contribute. For each equal-coordinate interval B:

- `chi_square_B = sum(z_i**2 for i in B)`;
- `chi_square_fraction_B = chi_square_B / sum(z_i**2)`;
- `weighted_rms = sqrt(sum(z_i**2) / N_included)`;
- `mean = sum(r_i) / N_included`.

Fractions are zero for an exact fit. Empty intervals have zero count and
contribution; `worst_regions()` excludes them. Intervals are left-closed and
right-open, except that the final interval includes the last coordinate.
Ranking is descending chi-square with coordinate order retained for ties.

The reported Durbin–Watson statistic is `sum((z_i-z_(i-1))**2) / sum(z_i**2)`.
Its numerator includes only pairs adjacent in the original grid with **both**
samples included. It never bridges an excluded interval. The denominator uses
all included samples. The value is unavailable when no pair exists or the
denominator is zero. With gaps this is a descriptive, segmented statistic;
ordinary significance tables do not automatically apply. No pass/fail
threshold is imposed, and weighted residuals are called sigma units only when
the caller's uncertainty model warrants that interpretation.

These are sums, norms and the definition of the serial-difference statistic,
implemented independently from first principles in `phasesmith-workflows`.
There are no new profile terms, derivative conventions or support policies.
An independent NumPy implementation and randomized differential tests validate
the calculation; a GSAS-II oracle fixture is unnecessary for these elementary
array statistics. Native and Python tests cover masks, interval edges,
nonfinite input, overflow, exact fits and unavailable evidence.

The retained `validation/results/rietx-20260916-qarr-fit-report.json` applies
this API to the final scale polish of the unchanged measured IUCr QARR 1g
validation. Its native diagnostic chi-square agrees with the fit's own
metric at relative tolerance `1e-14`.

## Interpretation limits

The report identifies where weighted error accumulates. It cannot distinguish
a missing phase from a background error, peak-shape mismatch or an incorrect
structure. `attribution_available` is explicitly false. There is no inferred
parameter-gain score or automatic fit acceptance.

Advice carries stable codes for non-converged termination, unassessed rank,
missing covariance, unresolved parameter correlations, exact active bounds,
and inspection of residual regions. Rank and covariance being unavailable are
not treated as proof of singularity. Bound flags mean exact equality, not an
undocumented closeness heuristic.

The original coordinate grid is the caller's responsibility: observations and
masks are checked against the result, but the existing result does not store a
grid identity. Diagnostic arrays are validated for shape, numeric dtype,
finiteness and strictly increasing coordinates. The standalone
`diagnose_residuals(x, residual, weighted_residual=..., included=...)` API also
works on TOF grids without reinterpreting microseconds as angles.
