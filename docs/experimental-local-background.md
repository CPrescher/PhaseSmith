# Guarded signal summation and local background planes

This opt-in detector-count estimator improves separation of spatial signal and
local background. It does not fit a peak profile or extrapolate missing signal.
The original `background_model="mean"` API remains byte-compatible. Select
`IntegrationOptions(background_model="plane", background_gap=3,
background_margin=6, pixel_radius=11)` explicitly; radii are experimental choices,
not universal spot sizes. Dioptas exposes the plane and gap controls, displays
both the guard and background perimeter, and exports the fitted diagnostics.

For a half-open signal box S, expand it by gap pixels to form a guard G, then
by another margin pixels to form an outer rectangle O. Background candidates
are O outside the union of **all active guards**, excluding masks and saturated
pixels. The gap is never integrated as signal or used as background. Overlap
flags refer to actual signal boxes. Spatial and angular supports remain fixed
for weak and negative measurements; no significance threshold selects their
signal pixels. Every full exposure intersecting the angular window contributes.

## Independently derived estimator

Use centred, scaled pixel coordinates u=(x-cx)/sx and v=(y-cy)/sy, with cx/cy
the signal-box pixel-centre midpoint and sx/sy half its width/height, at least one.
Let each background design row be Xj=(1,u,v). Solve the unweighted least-squares
plane beta=(X'X)^-1 X'd. Unweighted fitting avoids an inverse-observed-count
weight in the background estimate. The raw-count noise model still supplies
pixel variances V for residual standardisation and covariance propagation.

Fit, standardise residuals by sqrt(Vj), subtract their median, and reject
absolute residuals above clip times max(1,1.4826*MAD). Rejection is symmetric,
monotone, deterministic, and repeated for at most eight rejection rounds followed
by a final fit. The default clip is four. No rejected background pixel is removed
from a signal sum. Too few retained samples or a singular design gives BACKGROUND
and undefined counts; there is no fallback to an unidentifiable plane. Broad
curvature, an unpredicted peak inside the signal support, or extensive contamination
can still invalidate the planar model.

For usable signal pixels, accumulate q=sum(1,u,v) and raw counts D in the same
ordered pixel pass. The signed intensity is I=D-q'beta. With A=(X'X)^-1,
Cov(beta)=A X' V X A'. Then Var(I)=sum(Vsignal)+q'Cov(beta)q. The conditional
analytical derivatives are dI/dbeta=-q, dI/ddsignal=1 and
dI/ddbackground=-X A q. The native pass accumulates q with the signal, and uses
the same linear response for covariance propagation. Centered finite differences
check these equations away from clipping changes; an independent NumPy reference
uses `lstsq` rather than the native small-matrix solver.

These errors condition on the clipping membership, support and independent
pixel/frame model. They exclude clipping-selection uncertainty, background-model
bias, detector correlations, shared-background cross-reflection covariance and
geometry/support uncertainty. Residual inflation is reported rather than silently
rescaling uncertainties. This Gaussian-residual clipping is not a calibrated
low-count Poisson background algorithm.

## Observable quality and data products

BOUNDARY=2048 means the outermost one-pixel signal strip has signed net signal
above three conditional standard deviations and above 2% of the aperture total,
with the total itself above three sigma. Its background covariance is propagated
using that strip's own design sum. The test is advisory: it neither changes the
aperture nor declares weaker spots complete. BACKGROUND_MODEL=4096 means over
25% of candidate background pixels were rejected or the final standardised
residual RMS exceeds three. Existing mask, saturation, overlap, edge and angular
flags remain visible. The quality report also maps the new flags to actions.

`ReflectionTable.background_models` holds one immutable 12-column row per
measurement: plane coefficients, dI/dcoefficients, candidate/rejected background
counts, standardised residual RMS, and boundary intensity/sigma/fraction. The
coefficient coordinates are normalised as above. `background` in the existing
measurement table is the fitted background averaged over the usable signal
pixels. `background_sigma` is its conditional uncertainty. Original signed
intensities, per-exposure rocking records and all flags are retained.

## Validation scope and source basis

The Dioptas adapter uses the lower of an explicit detector saturation limit and
the original integer array's storage ceiling. Ceiling hits are retained in signal
sums with SATURATED and excluded from background candidates. This is not a
measurement of the physical detector's saturation threshold. Mixed original
storage types are rejected rather than silently sharing an incompatible limit.

`assess_support(nominal, alternatives)` returns immutable relative-change,
reason and pass arrays, requiring at least two matched alternate measurements.
It checks raw/noise provenance, >5-sigma nominal signal, clear flags/angular
endpoints, and <=5% changes by default. These are configurable descriptive checks
on correlated estimates, not proof of profile completeness. All original arrays
remain intact. The TIFF study's five controls and 31 passing crossings are
documented in `docs/single-crystal-integration-methods.md`.

Tests cover independent randomized equations, asymmetric background masks,
isolated background contamination, known broad-peak counts, conditional pixel
derivatives/covariance, deterministic repeats, signed zero-signal simulations,
singular backgrounds, invalid options and unchanged flat-background results.
The powder GSAS-II oracle does not cover image summation; no GSAS-II code or
objects are used. Synthetic injected counts are the numerical truth fixture.
The TIFF scan is a measured support-sensitivity and contamination diagnostic,
not absolute corrected I(hkl) truth.

For crystallographic context, the [DIALS methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5947772/)
describes background estimation followed by signed summation and distinguishes
weak-reflection profile fitting. Its [documented options](https://dials.github.io/documentation/programs/dials_integrate.html)
include local linear planes and outlier rejection. The equations and simple
clipping implementation here are independently derived; they do not claim
equivalence to DIALS's robust Poisson GLM, profile models or corrections.
