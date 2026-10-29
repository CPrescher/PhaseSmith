# Predicted rotation-reflection integration

Opt-in `experimental-spots` only. `phasesmith.experimental.integration` is a
script-first workflow; Dioptas supplies file loading, masks and inspection.
No material, composition or structure is required. Supply the selected indexed
cell/orientation through `predict_scan`, or an equivalent `ScanPredictions`
record. One row is one predicted crossing; no symmetry merging is performed.
Optional integer grain IDs preserve domain ownership. Pass competing domains
together so their apertures are excluded from each other's background.

## First milestone and equations

This section describes the original `background_model="mean"` estimator.
The optional guarded plane, fitted-background covariance and spatial-boundary
checks are documented in `experimental-local-background.md`. The comparison
with established packages and the actual DIALS TIFF probe are documented in
`single-crystal-integration-methods.md`. Empirical profile-intensity fitting is
not included in the current PhaseSmith implementation.

The estimator is a fixed-support linear sum, derived directly by independent
pixel-error propagation. For each prediction (x,y,omega), round x and y using
floor(v+0.5). Use the inclusive square of `pixel_radius` pixel centres around
that centre, clipped to the detector. Pixel bounds are half-open. The same
square is used in every contributing exposure (no scan-varying trajectory or
profile extrapolation). Include an exposure iff its angular interval has
positive intersection with [omega-omega_half_width,omega+omega_half_width).
Counts already represent the full exposure: never multiply them by angular
width or fractional overlap. Different exposure widths therefore still produce
summed observed counts, not an angular density. Overlapping exposures are
rejected, since this estimator does not normalize repeated angular coverage.
Forward and reverse scans and gaps are supported.

In each frame, exclude all active prediction signal apertures from the local
background ring, plus supplied masks and saturated background pixels. With m
usable signal pixels, n usable background pixels and variances V:

    B = sum_background(D)/n
    I_frame = sum_signal(D) - m*B
    var_frame = sum_signal(V) + (m/n)^2*sum_background(V)
    I_crossing = sum_frames(I_frame)
    var_crossing = sum_frames(var_frame)

For fixed support, derivatives are 1 on signal and -m/n on background;
intensity has derivative 1 with respect to each frame intensity. No fitted
profile parameter or new nonlinear derivative is introduced. Independent
Python pixel equations, finite differences, injected signals and randomized
comparisons validate these identities. Rust owns aperture evaluation and ordered
cross-frame reduction. Python loops over frames, never individual measurements.

Use strictly positive supplied variances for processed data. The raw default is
V=max(max(D,0)/gain+read_noise^2,1), with gain in electrons/photons per image
count. Sigma is conditional on independent pixels/frames and fixed supports.
Shared backgrounds cause cross-reflection covariance, which is not returned.
Fitted-background covariance, calibration/assignment uncertainty, detection
selection and truncation bias are not included. Negative intensities are retained.

## Quality and observable support

Retain EDGE=1, MASKED=2, SATURATED=4, OVERLAP=8, BACKGROUND=16 from aperture
measurement. NO_CENTROID is irrelevant to signed prediction-driven integration.
ANGULAR=256 means the requested angular window is not completely covered by
the union of exposures (including scan edges and internal gaps). NO_DATA=1024
means no contributing exposure. Incomplete masks/edges still return observed
counts with flags, without missing-signal scaling. Any undefined frame intensity
makes the total undefined; valid frames are never silently substituted.
All per-frame boxes, signal/background pixel counts, I, sigma, background and
flags are returned. A fixed finite window is not a claim of physical completeness;
inspect rocking curves and repeat with larger windows to assess tail sensitivity.

No Lorentz, polarization, absorption, detector-efficiency or partiality correction,
profile fitting, deblending, structure-factor conversion or symmetry merging is
applied. This first milestone measures finite-support detector counts. GSAS-II's
powder-profile oracle does not cover this image estimator. Synthetic injected
counts and independent pixel equations are the numerical oracle; Fe7C3 is a
measured workflow/sensitivity example, not absolute-intensity truth.

## Python example

```python
from phasesmith.experimental.prediction import predict_scan
from phasesmith.experimental.integration import IntegrationOptions, integrate_series

predictions = predict_scan(reciprocal_basis, geometry, omega_bounds,
                           image_shape, qmax=qmax)
reflections = integrate_series(
    frame_loader, omega_bounds, predictions, mask_loader=mask_loader,
    options=IntegrationOptions(pixel_radius=5, omega_half_width=1.0),
    gain=1.0, read_noise=0.0,
)
intensities = reflections.column("intensity")
errors = reflections.column("sigma")
curve = reflections.rocking_curve(0)
# curve contains omega, exposure bounds, signed intensity, sigma, background,
# frame indices, usable pixel counts and per-frame quality flags.
```

Windows are explicit experimental choices, not inferred mosaicity or guaranteed
complete support. The library is serial by default and accepts caller-owned
loading callbacks, without a file-format, GUI, DIALS or GSAS-II dependency.

## Fe7C3 P1 example (25 October 2026)

`benchmarks/reflection_integration.py` reads the preserved position-only Fe7C3
comparison case. It selects the accepted native candidate by training support,
then refines its positions using the correct one-based grain labels; source
structures and published cells/intensities are never consulted. The P1 example
uses 76 raw exposures from the user-authorized Dropbox archive. Zero padding
and nonfinite pixels are explicitly excluded. No calibrated seat mask,
saturation threshold or detector gain calibration was supplied in this example.

The corrected run (`runs/reflection-integration-20261025-v3/`) contains 3,285
predicted crossings. Radius 5 px / omega half-width 1 degree / background ring
4 px gives 9,749 frame contributions, 2,220 unflagged crossings, and 1,174
unflagged observations above 3 sigma. Radius 7 px / half-width 1.5 degrees /
ring 6 px gives 12,947 contributions and 2,130 unflagged crossings. On the
1,018 common unflagged observations above 5 sigma in the first window, the
median absolute intensity change is 22.89% (90th percentile 58.53%). These
simultaneously changed windows are a joint sensitivity check, not attribution
to an individual width or a completeness certificate. Unflagged means absence
of the enumerated flags; broad unmeasured tails remain possible.

Measured single-pass loading/integration times were 1.96 and 2.23 seconds on the
local M4 Pro. They are descriptive workflow timings, taken while other checks
could be active, not a controlled speed comparison. The standalone synthetic
benchmark covers 400 predictions on eight 512x512 frames and checks byte-exact
repeatability after warm-up. There is no pre-existing rotation-integration
baseline against which to claim a regression or speedup.

The initial exploratory v2 run is explicitly invalidated: its benchmark selected
zero (unassigned positions) rather than one-based candidate label 1 during
positional refinement. Its original payloads remain preserved and are not
scientific evidence. No production indexing code or preserved source result
was changed by that benchmark repair.


## Verification at this checkpoint

`cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features`, and
`cargo test --workspace --all-features` pass. The ordinary Python suite passes
1,297 tests (11 missing-data skips; 34 optional external/data tests deselected).
The 13 integration tests include signed weak reflections, injected count and
rocking-centroid recovery, independent random pixel equations, fixed-support
finite differences, masks/saturation, competing apertures, scan gaps/reversal,
exact angular boundaries, cancellation, invalid inputs and reused loader buffers.

The final experimental wheel is installed in Dioptas' single-crystal development
environment. Replaying both Fe7C3 windows through that installed wheel reproduces
all arrays and frame fingerprints byte-for-byte; the separate installed-wheel
validation record retains its actual extension hash. GUI/adapter tests cover
original counts for corrected scans, full per-frame masks, CSV exports, plotting,
frame selection and stale selected-cell rejection. No standard release, Git
commit or publication is part of this checkpoint.

Dioptas' full test rerun passes 2,130 tests (15 skips; nine subtests). The added
cell-change cancellation regression also passes against the installed wheel;
cancelled work cannot replace a newly selected cell or its position records.

## Next milestone: image-level integration QC and intensity reduction

User clarification: the requested inspection is broader than a rocking curve.
It should assess whether reflection intensities measured from each image are
usable, identify problematic images/reflections, and show which correction or
re-integration is supported. Rocking curves remain per-reflection diagnostics.
Do not normalize images to equal total brightness: rotating a crystal changes
which HKLs diffract. Distinguish pixel-integration failures (truncated supports,
background contamination, saturation/overlap and shadow masks) from relative
scaling and orientation-dependent corrections. Corrections need identifiable
redundant observations; a frame-scale trend alone does not diagnose its cause.
Use confirmed intensity symmetry only, not merely a proposed metric symmetry.
Retain original counts, unscaled observations, proposed corrections and excluded
measurements separately. The current estimator implements none of those new
scaling/absorption/error-model fits.

The user identified the preferred eventual CrysAlis comparison archive as:
`~/Dropbox/Science/_Archive/Fe7C3/X-ray diffraction/`.
An inventory confirms real CrysAlis reduction exports and project information
under `Single Crystal Diffraction Data/ESRF_OCT2011/`, including Fe7C3d projects
with `.hkl`, `.sum`, `.eqv`, `.rint`, `.par`, `.run`, `.rrpprof` and `.pcorr` files.
For example `Fe7C3d_p1/Fe7C3d_p1.sum` identifies CrysAlis171
1.171.32.29 (10 June 2008), and its `.eqv` explicitly reports agreement before
and after outlier rejection. This is an inspected reference candidate, not yet
an executed intensity comparison. Raw frame/project association, axes and HKL
basis transformations, unmerged/merged status, scales, corrections and rejection
policies must be established before asserting numerical agreement. No source
cell, structure, HKLs or CrysAlis ranking may enter a benchmark described as
blind indexing; use reference information only in the separate comparison.
