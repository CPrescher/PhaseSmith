# Experimental detector spots: first milestone

This branch adds an opt-in `experimental-spots` native feature and the unstable
`phasesmith.experimental.spots` Python module. Default extension and Rust facade
builds omit the engine. There is no standard release, tag, GUI dependency, new
runtime dependency, or change to powder behavior. The numerical crate is
unpublished. No cell, elements, atom model, hkl, detector calibration or simulator
truth is accepted by this milestone.

## Build and use

```sh
uv venv
uv pip install maturin numpy pytest
uv run maturin develop --release --features experimental-spots
uv run pytest tests/test_experimental_spots.py
```

```python
from phasesmith.experimental.spots import SpotOptions, detect_spots, extract_spots

result = detect_spots(image, mask=mask, options=SpotOptions(threshold_sigma=5.0))
signed = extract_spots(image, boxes, variance=independent_pixel_variance)
print(result.column("intensity"), result.column("sigma"), result.flags)
```

`mask=True` excludes pixels. Images are row-major `image[y,x]`; integer x/y
refer to pixel centers. Boxes are half-open `[x0,y0,x1,y1]`. GUI renderers whose
pixel centers are half-integers add 0.5 only for the display overlay. Bounds
and arrays retain detector orientation after any caller-applied flip/rotation.
Nonfinite unmasked values fail; masked nonfinite samples are omitted. Input
arrays are not modified, and native work uses owned snapshots outside the GIL.

## Detection and finite support

For pixel i the background region is an outer square of radius R minus an
inner square of radius r, both including their endpoints. The default radii
are R=8, r=2. Windows clip to the detector. Masked and saturated background
pixels are excluded. Let n be the number of usable background samples:

    B_i = sum(D_j)/n
    Var(B_i) = sum(V_j)/n^2
    strong_i = (D_i - B_i > absolute_threshold)
               and (D_i - B_i > sigma_threshold * sqrt(V_i + Var(B_i)))

Strict inequalities define threshold boundaries. A first pass uses all usable
background pixels; a second pass excludes first-pass strong pixels from the
background only. Summed-area tables implement these sums. There are exactly
two passes; this is an independently implemented local-mean detector, not a
claim to reproduce CrysAlisPro or DIALS. Eight-connected components are found
in row-major order. Components below `min_pixels` are counted separately;
components above `max_pixels` are returned with a LARGE flag. Exceeding
`max_spots` fails explicitly, without silent truncation.

Each accepted component's bounding box is padded by `padding` pixels, clipped
to the detector and used as its finite integration aperture. Every signal box
is excluded from every background estimate. The extraction background is a
constant local mean in the surrounding `background_margin`-pixel ring. There
is no profile fit, extrapolation, deblending, robust background plane, powder
ring rejection, gap filling or selection-bias correction in this milestone.
Connected touching reflections can form one component and remain unresolved;
OVERLAP detects overlapping *apertures*, not all physical spot overlaps.

## Intensities and conditional errors

For m valid signal samples and n usable background samples:

    I = sum_signal(D_i) - m*B
    Var(I) = sum_signal(V_i) + m^2 * sum_background(V_j)/n^2

This follows propagation of an independent-pixel linear sum. Its analytical
data derivative for fixed supports is 1 on signal pixels, -m/n on background
pixels and zero elsewhere. The tests check this identity by finite differences.
No nonlinear profile or profile-parameter derivative is introduced.

The caller may supply strictly positive per-pixel variances in squared count
units. Otherwise V_i = max(max(D_i,0)/gain + read_noise^2, 1 count^2), an
explicit Poisson-plus-read-noise plug-in approximation. `gain` is photons or
electrons per image count, and read noise is in image count units. The floor
avoids zero variance; it is not detector calibration. Corrected/resampled
images require appropriate propagated variances and potentially covariance
that this API cannot represent. Cross-spot covariance from shared background
pixels is not returned. Background gradients, profile tails, assignments and
threshold selection are outside these conditional uncertainties.

Centroids and RMS widths use positive background-subtracted weights inside
the aperture; these are moments rather than fitted parameters or standard
errors. With no positive excess, the center defaults to the aperture center
and NO_CENTROID is set. Signed I is preserved, including negative results in
caller-supplied boxes. Detecting only positive spots still selects observations
and cannot support an unbiased systematic-absence or weak-intensity claim.

Flags: EDGE=1, MASKED=2, SATURATED=4, OVERLAP=8, BACKGROUND=16, LARGE=32,
NO_CENTROID=64. Masked signal is omitted without a correction. Saturated
signal contributes its clipped counts and is flagged; its intensity is not
treated as a valid complete measurement. Insufficient background yields NaN
intensity/error; an aperture with no usable signal samples does likewise.
All flagged records remain available. No crystallographic
corrections, merging, partiality, grain-volume estimate or |F|^2 is inferred.

## Validation and performance

The NumPy reference uses direct local slices and independent connectivity;
it never calls native kernels or summed-area helpers. Tests cover native/reference
parity, deterministic records, known areas/centroids, signed extraction, finite
support, Poisson noise, supplied variance, flags, masks, thresholds, invalid
inputs and uncertainty propagation. GSAS-II's powder-profile oracle does not
cover detector aperture photometry; no oracle-equivalence claim is made.
The equations follow elementary linear error propagation and the summation
integration approach described by [Leslie (1999)](https://doi.org/10.1107/S090744499900846X).
No external implementation code was copied.

Detection is O(P), using three integral arrays (two f64 and one u32), an O(P)
strong mask and an O(P) worst-case component queue. Extraction takes O(P+A),
where A is the total area of signal boxes and background rings, including
overlaps. It is not automatically O(P) for arbitrary overlapping caller boxes.
One native thread is used; no hidden fitting-worker pool is created.
`benchmarks/spots.py` measures realistic multi-spot images and records complete
parameters, seed, spot count, flags and repeated timings. Cross-platform CI
builds/tests the experimental feature on Linux, macOS and Windows. CI configuration
alone does not establish a passing run on those platforms.

Later milestones can add learned profiles, frame linking, unknown-cell
indexing and multi-grain scaling separately after this observable contract is
validated. Experimental code remains on its branch and behind explicit build
and UI switches; it must not be merged into a standard release by accident.

## Initial checks and measured-data scope

The first local validation passes 41 detector tests and all 1,023 normal Python
tests (11 external-input tests skipped and 34 external-oracle/real-data tests
deselected), plus workspace Rust tests and strict linting. The first synthetic
2048x2048, 1024-spot benchmark is preserved in
`benchmarks/experimental-spots-macos.json`; it precedes the compensated-moment
repair and is not a timing claim for the later implementation.

User-supplied Fe7C3 archive frames are tested only in ignored `runs/` directories.
`benchmarks/fe7c3_spots.py` freezes the frame list, settings and source snapshot
before its run. The initial 13-frame pilot completed, with three independent
reference checks failing on 1.3e-8 to 2.9e-8 pixel centroid discrepancies in very
large apertures. Those failed checks remain preserved. Compensated centroid and
width sums address that accumulation error without changing detection thresholds
or intensity arithmetic; tolerances are unchanged. A new all-series study
examines every TIFF independently, including the other phase-labelled frames,
with no phase label entering detection. A separately recorded zero-padding mask
sensitivity must not replace the unmasked outcomes.

Digital gain/read noise have not been calibrated for this archive. The 65535
flag is a conservative uint16 storage-ceiling check, not a detector full-well
measurement. No indexed text, UB matrix or known structure enters these tests.
There is no manually labelled reference peak set, so completion and numerical
agreement do not establish physical spot precision/recall or intensity accuracy.
