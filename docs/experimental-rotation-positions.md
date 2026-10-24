# Experimental rotation-series positions

This is a position-first peak hunt, before cell finding or intensity integration.
It is private branch work under the existing `experimental-spots` feature, with
no change to standard builds, powder APIs or release dependencies. The UI and file
readers remain in Dioptas. PhaseSmith accepts arrays and caller-supplied frame
loaders; the dependency-free spot crate owns the linking and centroid kernels.

## Input and angle conventions

`phasesmith.experimental.rotation.locate_series(loader, omega_bounds)` reads each
frame once. Images/masks obey the existing experimental spot contract. The caller
supplies an explicit floating array of exposure start/end bounds in degrees,
shape `(n_frames,2)`. Both forward and reverse monotonic scans are supported.
Starts must be strictly ordered in that direction; frame widths must be nonzero
and have consistent signs. Overlapping angular exposures are allowed for
position hunting. Angular gaps exceeding `1e-8` degrees split linking segments.
A missing/empty detector frame is not bridged. Nothing derives omega from frame
numbers, exposure time, chi, or detector motor readings in the numerical core.

Dioptas groups files by their common prefix and numeric suffix. Per-frame
`omega0` and `omegaR` sidecars require explicit start/width interpretation at
import. Alternatively, the user supplies first start, step, and exposure width
separately. Missing numeric frame IDs retain their gaps under manual angles.
Original sidecar fields and checksums survive manual overrides and export.

## Detection, linking and finite support

The existing two-pass local detector supplies frame centroids and finite padded
boxes. Observations in frames f and f+1 are connected when their Euclidean x/y
distance is <= `link_radius` pixels and the frames belong to the same angular
segment. There are no same-frame edges. Connected components are numbered by
first observation in original frame/spot order. A deterministic x-coordinate
sweep limits candidate comparisons, with an explicit candidate budget. There is
no velocity model, gap bridging, lattice prediction, one-to-one assignment,
profile fitting or automatic rejection of persistent detector features.

Components containing more than one detection from a frame are marked
AMBIGUOUS=128; touching a scan/segment edge sets SCAN_BOUNDARY=256. All original
per-frame quality flags are retained by bitwise OR. No flagged candidate is
silently promoted or removed. The returned `PositionTable` contains track
means, bounding boxes, half-open observed frame spans and **every** individual
`[track,frame,x,y,omega]` observation. These frame observations are important
because detector position can change during rotation.

For each detected frame observation j, let w_j=max(I_detected,j,0), and let
z_j=(x_j,y_j,(omega_start,j+omega_end,j)/2). The track summary is

    z_mean = sum(w_j z_j) / sum(w_j)
    omega_rms = sqrt(sum(w_j (omega_j-omega_mean)^2) / sum(w_j))

Nonfinite position/weight rows do not contribute to moments. Zero total weight
yields NaN coordinates, retaining the underlying quality flags. Compensated
sums and centered angular deviations preserve numerical stability. Positive
background-subtracted detector sums are **internal centroid weights**, not
integrated rocking intensities, |F|² or corrected I(hkl). Only detected frames
enter this centroid, so threshold selection can bias angular means. A one-frame
spot gets its exposure midpoint; zero angular RMS does not imply exact omega.
No position uncertainties or sub-frame rocking profiles are inferred.

These are first-principles weighted moments, not a new fitted profile. For fixed
weights, d(mean_z)/d(z_j)=w_j/sum(w); deterministic centered-difference tests cover
these derivatives. Detection/connectivity boundaries are discrete and are not
differentiated. GSAS-II's powder-profile oracle does not cover rotation spot
positions. No external implementation code was copied.

## Cost and validation

For F frames of P pixels and K detected observations, frame detection takes
O(FP + A), where A covers finite extraction apertures/rings as in the existing
spot contract. Linking sorts by x per frame, then checks Q candidate pairs:
O(K log K + Q alpha(K)) time, O(K+F) metadata memory. In dense cases Q can be
quadratic in observations per neighbouring frame; the explicit budget fails
rather than truncates. Grouping and moment summaries are O(K log K + K), apart
from per-track NumPy convenience allocations. Detector arrays are streamed, so
memory is O(P+K+F), never the full F-by-P image cube.

The independent Python reference uses an all-pairs graph with explicit traversal
and direct NumPy moment sums. Tests cover deterministic repeated records,
randomized graph/moment parity, finite boundaries, ambiguous branches, empty
frames, gaps, forward/reverse scans, distinct half-degree steps and one-degree
widths, mask/saturation flags, invalid inputs, safety budgets, translations and
fixed-weight derivatives. `benchmarks/rotation_positions.py` exercises a
multi-frame/multi-spot detector grid and checks exact repeatability.

Real-data completion is assessed on the three available Fe7C3 narrow scans using
recorded sidecars and the reviewed scan convention. Their handwritten logbook
entry for P2 lists 0.5 seconds per 1 degree, not a 0.5 degree step. Raw datasets,
logbook renders and measured results stay in ignored local `runs/` folders.
Measured completion is not physical precision/recall validation: there is no
manually labelled position truth. No known cell, indexed text or UB file is
used. Cell finding, reciprocal-space geometry validation and symmetry decisions
remain subsequent milestones. Positions can constrain a cell/Bravais lattice;
a full space-group claim needs additional diffraction evidence.

## Local validation checkpoint (2026-10-24)

All three 76-frame Fe7C3 narrow scans completed with exact repeated records
under the recorded zero-padding mask and uint16 storage-ceiling flag policy.
P1/P2/P7 respectively produced 2,125/2,421/6,436 linked position candidates from
4,703/4,882/11,787 frame detections. Of those candidates, 863/949/2,361 combine
multiple frames. Ambiguous counts are 36/39/125; quality flags are retained.
These are heuristic candidates, not validated unique reflections. No physical
spot labels, UB matrix, known cell or symmetry prior enters the peak hunt.

The final grouping implementation preserves all measured positions exactly.
The local synthetic 32x512x512 benchmark contains 225 linked candidates and
7,200 frame observations, with exact repeats; recorded timing is about 0.4 s
on this desktop and is not a controlled speed comparison. The final environment
passes 1,037 Python tests (11 external-data skips, 34 deselected) and native
workspace checks. The first full Python attempt had three subprocess failures
from the desktop's Xcode/Git tool selection; selecting Command Line Tools and
Homebrew Git passed the three failures and then the complete suite. No test
threshold or scientific assertion was weakened. Hosted cross-platform results
must be checked separately from these local results.


## Optional frame-completion observer

`locate_series(..., frame_finished=callback)` calls the observer with a zero-based
frame index and immutable `SpotTable` after detection/fingerprinting and the
observation-budget check, before that frame's progress notification. These are
per-frame candidates, before cross-frame linking and track-level flags. The
observer runs synchronously on the caller's thread; an exception aborts the job.
Without an observer the numerical pipeline is unchanged. Live consumers may
retain only the latest source image/candidate snapshot and marshal presentation
to their own GUI thread. Enabling preview must not alter positions or hashes.
