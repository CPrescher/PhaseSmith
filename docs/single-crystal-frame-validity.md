# Shared single-crystal shadow masks

`phasesmith.experimental.frame_validity` provides portable frame validity,
original-image seat-shadow estimation, geometric DAC masks and composition.
It uses NumPy arrays and loader callbacks, with no Dioptas, Qt, HDF5 reader,
display flip or application cache dependency. Native detector/profile kernels
remain unchanged. The API is staged under `experimental` with the existing
single-crystal APIs; no public namespace/build promotion is implied.

## Original-image estimator

```python
import numpy as np
from phasesmith.experimental.frame_validity import (
    SeatShadowOptions, estimate_seat_shadows, flag_invalid_tracks,
)
from phasesmith.experimental.rotation import locate_series
from phasesmith.experimental.profiles import prepare_profiles

# Loader returns original nonnegative counts in one declared array orientation.
# Supply ordered identities and actual exposure bounds; the library reads no files.
ids = tuple(f"frame-{i}" for i in range(len(omega_bounds)))
validity = estimate_seat_shadows(
    load_original_counts, len(ids), frame_ids=ids, omega_bounds=omega_bounds,
    mask_loader=load_detector_exclusions, fixed_mask=manual_mask,
    options=SeatShadowOptions(reference_first=20, reference_last=40), workers=4,
)
validity.save("/external/run/frame-validity.npz")  # exclusive, pickle-free archive

def load_checked(i):
    image = load_original_counts(i)
    validity.verify_frame(i, image)
    return image

def mask(i):
    # Dynamic source exclusions, e.g. EIGER reserved values, still belong to
    # the reader. Static/manual masks are already captured in validity.
    return validity.mask(i) | load_detector_exclusions(i)

positions = locate_series(load_checked, omega_bounds, mask_loader=mask)
positions = flag_invalid_tracks(positions, validity)
observations = prepare_profiles(
    load_checked, omega_bounds, predictions, mask_loader=mask,
)
```

The estimator alone requires optional SciPy (`pip install 'phasesmith[masking]'`).
Snapshot access and geometric masks require NumPy only. Existing detector and
profile operations retain their own native feature requirements. Source loaders,
source-mask loaders and progress callbacks run in order on the calling thread.
At most `workers` owned frame snapshots are in flight (default 1, maximum 32).
Every source is read once during estimation. Retained working data are coarse
frames/masks, never a detector-sized mask cube. The coarse cell budget is
50 million; detector masks are limited to 100 million pixels.

The estimator adapts the MIT-licensed Dioptas `seat-illumination-ratio-v2` behavior
at revision `651c27f98108fb19f60db9a8cb227755b2f9d6a2`, with its notice retained in
`THIRD_PARTY_NOTICES.md`. Its steps are:

1. Form each original frame's block medians (default 16x16), ignoring supplied
   exclusions/nonfinite samples and padding partial edge blocks with NaN.
   Counts must be nonnegative and <=1e30; background-subtracted residuals are
   not an input. Real zero pixels remain valid counts.
2. Apply spatial 3x3 median and sigma=1 coarse-pixel Gaussian smoothing, without
   temporal smoothing. Form the reference as the median of up to five evenly
   spaced illuminated views in an inclusive reference interval. Defaults are
   rounded 37%--59% of the source frame-index range.
3. Compute q=smoothed/max(reference,1). Divide q by the frame's 90th percentile
   over reference pixels >40 counts. This is detection-only flux normalization;
   measured counts and subsequent intensities are never normalized here.
4. At reference support >25 counts, identify q<threshold (default 0.85), close
   and fill holes, then retain >=20-cell connected components touching the
   declared relative left/right detector-side anchors (350/2048 and 1700/2048).
   Dilate by ceil(guard_pixels/block) coarse cells; default guard is 32 pixels.
   Every True coarse cell excludes its whole native detector block, including
   partial detector-edge blocks. Guard=0 adds no dilation.

Uniform dimming with positive measurable illumination is diagnostic only and
never sufficient to mask an entire frame. A wholly dark *unmasked* exposure has
unknown visibility and raises `SeatShadowUnavailable`, as does an insufficient
illuminated reference. There is no silent all-valid fallback. Entire frames
already explicitly unavailable in the source mask remain excluded. This is a
bounded broad-shadow estimator with absolute background/side-component assumptions,
not universal segmentation or a classification of why a frame is dim.

## Snapshot and source identity

`FrameValidity` owns read-only coarse boolean arrays, an optional precise static
mask, ordered unique `frame_ids`, detector shape, exposure bounds and copied JSON
provenance. `True` means unavailable. Accessing `provenance` returns an independent
copy. `mask(i)` returns one full-resolution mask; `rows(start,end)` returns a
bounded flattened-pixel-by-frame tile for temporal background consumers.
`boxes_valid(i, boxes)` checks full half-open integer rectangles; `valid_points`
checks nearest pixel centres within the half-open detector footprint and is
only a centre diagnostic, not a full-aperture acceptance test.

Generated snapshots retain original image and source-mask hashes, estimator
settings, illuminated reference indices and detection scales. `verify_frame`
checks the supplied array against its original-image hash when that hash is
present. Manually declared/geometric masks check layout and bind to their
explicit frame/angle identities instead. `validate` checks ordered identifiers,
shape and, when supplied/stored, exact exposure bounds. No basename rebinding
or implicit orientation conversion is performed. Reader-specific dynamic masks
must continue to be composed at each read; their bytes are fingerprinted during
estimation, but are not inflated into a stored high-resolution mask cube.

`save` refuses an existing path. `load(..., frame_ids=..., shape=...,
omega_bounds=...)` validates schema, identity, metadata and the snapshot
fingerprint with `allow_pickle=False`. The fingerprint covers coarse/static
arrays, angular bounds and provenance. Source freshness still requires verifying
actual loaded frames; the library does not stat or reopen application files.

## Geometric masks and composition

```python
from phasesmith.experimental.frame_validity import DacApertureMask, CombinedValidity

aperture = DacApertureMask(
    geometry, omega_bounds, detector_shape, frame_ids=ids,
    axis_at_omega_zero=np.array([0., 0., 1.]),
    half_opening_degrees=35.,  # explicit measured/assumed hardware parameter
)
combined = CombinedValidity((validity, aperture))
# Pass combined.mask as the common mask_loader in every stage.
```

No universal opening angle or DAC axis is inferred. The provider uses an affine
`PanelGeometry`; non-affine coordinate maps require caller-supplied mapped masks.
The DAC axis is fixed in the sample frame and specified in lab coordinates at
omega zero. For unit lab ray s and spindle axis a, visibility requires
`|s dot R(a,omega)n| >= cos(half_opening)` in one consistent symmetric cone.
This applies to both the incident beam and every outgoing pixel footprint.

For the entire closed exposure [lo,hi], write the dot product as
`C+A*cos(omega)+B*sin(omega)`. Evaluate both endpoints and every contained
stationary extremum at atan2(B,A) and that angle+pi, modulo 2pi. Forward/reverse
bounds have identical semantics. All four affine pixel corners must remain
in the same cone over the whole exposure; convexity then covers the pixel
interior. The closed angular boundary allows eight float64 epsilons in cosine.
Thus neither an illuminated midpoint nor an illuminated pixel centre can hide
partial exposure/pixel blocking. Evaluation uses bounded row tiles, without
caching a full mask cube. This is conservative exclusion, not absorption or
partiality correction, and has no intensity significance threshold.

`CombinedValidity` ORs masks from compatible ordered frames, detector shapes and
angular bounds; it propagates source verification to its providers. Geometric
parameters are exposed as JSON-compatible `metadata()`; source-array snapshots
are saved separately rather than silently expanding a composite scan into a
full-resolution cube.

## Consistent finite supports

Apply the same frozen mask before detection, position calibration, profile
teaching, foreground/background integration and scaling. `flag_invalid_tracks`
preserves all position rows, values and observations, ORing MASKED into tracks
whose bounding rectangle intersects an excluded pixel in a contributing or one
adjacent contiguous exposure. Angular gaps are not bridged. This protects
position fitting from truncated rocking tracks.

Native integration consumes full pixel masks in foreground and background fits.
Profile learning rejects masked teachers; profile fitting retains masked/support
flags, including missing rocking tails. Retain those flags and finite-coverage
checks when selecting observations. Never turn blocked measurements into zero
intensities, infer masks from weak-reflection significance, or interpolate counts
across shadows. The existing explicit MISMATCH-warning policy is distinct from
mask/partiality failures. Temporal backgrounds can consume `validity.rows` with
`arpls_masked_rows`, which keeps missing intervals separate.

## Validation

Tests cover moving/partial shadows, uniform dimming, explicit unavailable inputs,
fixed-mask composition, partial blocks, source order, reused buffers, worker
bitwise equality, immutable ownership, archive corruption and changed sources.
Independent Rodrigues sampling checks continuous-angle extrema; pixel-corner,
closed-boundary, reverse-scan and tile-equality tests cover geometric supports.
A masked rocking-tail fixture verifies teacher exclusion and bitwise invariance
of background/profile results when hidden pixel values are changed by orders of
magnitude. Temporal background tests use the same snapshot tiles.

`benchmarks/benchmark_frame_validity.py` exercises 40 detector frames with 64
narrow peaks per 512x768 grid and moving shadows. Its optional `--dioptas-source`
adapter compares the unmodified upstream estimator numerically with only its
reader/resource imports supplied by the harness. Native profile equations and
analytical derivatives are unchanged; no new GSAS-II fixture covers this
image-mask preprocessing. All ordinary native/reference tests remain required.
The MgSiO3 v6 protocol tests the original-image mask throughout a fresh raw run,
without the earlier example-only 35-degree centre-ray postselection.
