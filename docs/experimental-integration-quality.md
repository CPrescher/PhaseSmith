# image-by-image integration quality

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


`phasesmith.single_crystal.integration_quality.assess_integration` is a pure,
non-mutating report over `ReflectionTable`. It does not change peak equations,
intensities, uncertainties, masks, supports, reflection selection or scale.
Numerical profile evaluation and fused measurement remain in Rust. NumPy
summarizes contiguous measurement records; only frames are iterated in Python.

Each exposure retains its own row, including exposures with no predictions.
Report counts of finite, strong, flagged, masked, saturated, overlapping,
background-deficient and edge-touching measurements; excluded signal-pixel
fraction; median local background, I/sigma and strong-spot centroid distance
from the prediction. Median brightness and I/sigma are descriptive only:
different frames contain different HKLs and are never equalized by this report.
Centroid offsets use unflagged positive observations above an explicit 3-sigma
threshold; the default review trigger requires at least five offsets and a
median >=2 pixels. Excluded pixels remain missing counts, never zero intensity.

For each crossing with at least three measurements, all finite I and positive
sigma, set E=max(I_first,I_last)/sum(max(I_frame,0)). If E>=0.10 and either
endpoint is above 3 sigma, advise wider angular inspection. This tests signal at
a measured endpoint, not physical completeness, a fitted mosaicity, a p-value,
or proof of lost intensity. With fewer than three observations the endpoint
metric is undefined. Invalid endpoint observations are never replaced by inner
valid observations. Gaps, masks and scan coverage still retain the raw flags.

An optional alternate-window result must have exactly identical HKLs,
predictions, grain IDs, exposure bounds, input fingerprints and noise model.
On unflagged finite pairs where baseline I>3 sigma, compute
D=(I_alternate-I_baseline)/abs(I_baseline); advise inspection for abs(D)>=0.10.
The two estimates share pixels and their errors are correlated: D has no
independent-error significance interpretation. Thresholds are caller-owned
advisory settings in QualityOptions, never rejection or scientific success gates.
No per-frame scale fit, absorption/decay model, uncertainty inflation, automatic
outlier rejection or symmetry assumption is introduced. Those require
comparable, appropriately corrected repeated observations and adequate redundancy.
A metric cell candidate alone does not establish intensity symmetry.

The report preserves every observation and links frame review actions to
reflection IDs. The independent test equations use direct loops over small
synthetic records. Existing integrated intensity and its derivatives do not
change; no new nonlinear profile or profile derivative is introduced. GSAS-II's
powder oracle does not cover these descriptive image statistics. Fe7C3 is a
measured usability/window-sensitivity example, not absolute detector truth.

## Dioptas review and window comparison

After integration, **Intensities → Image quality** shows every exposure and
advisory actions. Selecting a row browses its detector frame. **Inspect
measurements in selected image** filters the reflection table to that exposure;
**Show all reflections** restores the full list. Selecting a reflection outlines
its signal aperture (yellow) and outer background support (dashed cyan) on the
image. The background estimator excludes all active signal apertures within
that support and applies the full masks; the outlined rectangle is not a claim
that every pixel inside it entered the background mean. For corrected scans,
intensities use verified original counts; the main series viewer can still show
the corrected image, explicitly labelled as such.

**Check wider windows** repeats the measurement with signal radius +2 pixels,
omega half-width x1.5 and background ring +2 pixels, bounded by the supported
maximums. Both results survive; the quality view reports the comparison and
the reflection plot overlays the alternate intensity trace. Input fingerprints,
noise parameters, predictions and exposure bounds must match. Changes to
selection/settings invalidate the review and stale work is discarded. Export
includes baseline and optional wider-window reflection/frame CSVs, quality
frame/reflection CSVs, thresholds, actions and full provenance. No review action
automatically rejects an image or changes a correction.

## Measured example and calibration boundary

The preserved 0.3100-A PVT-EOS Fe7C3 P1 example provides 76 frame rows. At the
explicit default advisory thresholds, 942 crossings have significant endpoint
signal and 954 change by >=10% in the alternate-window comparison among the
eligible strong unflagged pairs. These counts use a baseline >3-sigma gate;
the earlier 22.89% median example used >5 sigma and is a different disclosed
population. All 76 images contain measurements needing some inspection; that
does not establish that any image is unusable. No intensity was changed.

`benchmarks/integration_quality.py` summarizes 49,700 measurements from 10,000
predictions on 200 synthetic exposures with byte-identical repeat reports.
The local warmed median is approximately 11 ms; the saved 76-frame example is
approximately 5 ms. These are reporting-only timings, not diffraction processing
or physical-validation performance claims.

The user supplied a separate Fit2D calibration screenshot for an older
measurement: pixels 139 um, direct distance 309.6355 mm, wavelength 0.414915 A,
beam centre (1313.766,1534.685), tilt-plane rotation -114.9476 degrees, tilt
-0.176961 degrees. It is converted with pyFAI's public `Geometry.setFit2D` into
`runs/integration-quality-20261025/calibration/fe7c3-user-fit2d.poni`, with the
original image, exact transcribed inputs and a reload/roundtrip record. The
canonical positive-tilt/180-degree-shifted plane rotation is geometrically
equivalent; the supplied signed values remain in provenance. Fit2D/MAR/TIFF
row orientation still requires explicit matching, with no assumed centre offset.

This screenshot is not silently assigned to every old archive run. The initially
inspected CrysAlis Fe7C3d_p1 records ~399.54 mm and a different centre; its raw
MAR headers instead contain 1.55 A / 100 mm, incompatible with that reduction.
The read-only `benchmarks/crysalis_calibration_audit.py` records the 96 named
P1 frames and these source conflicts. The older archive also contains other
geometries, including a p14 log with ~309.635 mm distance but a different centre.
The user subsequently confirmed that the screenshot applies at least to the
first pressure points. Use it as their supplied starting calibration and check
raw-image alignment; do not extrapolate it to later points. Differences from
archived project geometry still require reconciliation before an old-data
intensity comparison. No CrysAlis numerical agreement or validated
conversion of its private geometry/scan-axis conventions is claimed.

## Checkpoint verification

The complete PhaseSmith Python suite passes 1,311 tests (11 missing-data skips;
34 optional external/data tests deselected). Required workspace Rust formatting,
clippy and all-feature tests pass. The installed Dioptas development environment
passes 2,149 tests (15 skips and nine subtests). Focused tests exercise quality
navigation, filtering, aperture placement, preserved baseline/wider-window
results, provenance rejection, advisory exports and stale/cancelled work.
The installed wheel reproduces every saved Fe7C3 quality array byte-for-byte.
No Git commit, standard release, old-data intensity comparison, or automatic
scale/absorption correction is included in this checkpoint.


## Vertical-row audit on the older P1 candidate scan

The user flagged possible vertical inversion. Dioptas `read_frame` already
reverses native MAR/TIFF rows once; calibration rows and any extra image
transforms remain separate metadata. Do not add another undocumented flip.
`benchmarks/older_fe7c3_orientation.py` freezes 96 P1 sources, the supplied Fit2D
calibration and uppercase MAR phi exposure bounds before testing both row
conventions and both signs of the assumed vertical rotation axis. No reference
UB, cell, structure or HKLs enter the inverse. The same 1,539 tracks are used
in all four cases, selected from 4,346 linked positions with whole rotation
blocks reserved for checking. No lattice candidate is returned in those four
bounded searches; this is not a general proof that the calibration is wrong.

The separately frozen `older_fe7c3_geometry_control.py` uses only archived
instrument distance/beam centre, with explicit zero detector tilt and unverified
pixel-origin conventions. It returns a passing positional model for each row
convention at the negative rotation-axis sign, and none at the positive sign.
Thus neither flipping alone nor positional support in this approximate control
establishes the correct row convention. Calibration-to-scan association and
sample-axis geometry remain to be reconciled. A passing lattice is not a phase
identification, primitive-cell proof or absolute geometry validation.

All arrays and hypotheses are preserved under
`runs/older-fe7c3-orientation-20261025/`; the preview overlays supplied and archived
beam centres on native and vertically reversed image composites. The centres
remain labelled as different, unconverted source conventions. The benchmark's
four ray geometries agree with Dioptas' existing public adapter. No production
calibration, image, mask, integrated intensity or default row convention changes.


## User clarification: geometry authority and CrysAlis coordinates

Use the user-supplied Fit2D calibration (converted through public pyFAI APIs to
PONI) as the Dioptas geometry for the first older pressure points. CrysAlis uses
its own instrument-coordinate convention. Its unconverted beam-centre and
angle fields must not be compared directly to Fit2D values or substituted into
the application as a competing calibration. The archived-scalar control above
is exploratory only; it does not establish a calibration mismatch or justify
replacing the supplied geometry. The failed bounded indexing tests leave row
mapping, assumed sample-axis geometry and search limitations unresolved.
CrysAlis remains the reference for reduced reflection intensities, after explicit
HKL-basis, scale, correction and merging-convention reconciliation. Coordinate
conversion between CrysAlis instrument fields and Dioptas is a separate problem,
not a prerequisite for using the supplied Fit2D/PONI file. The conclusion and
calibration provenance records carry this correction; original numerical probe
results remain preserved.

## Active example: return to the original TIFF scan

The user has set aside the older MAR images because of image-format issues.
Continue integration development with `Fe7C3_P1_C3_D1s_010.tif` through `_085.tif`
from the separate PVT-EOS archive: 76 exposures, 2048 by 2048 pixels, and the
preserved `LaB6_p49_001.poni` calibration at 0.3100 A. The older supplied Fit2D
calibration at 0.414915 A is not applied to these TIFFs. The MAR investigations
above remain historical diagnostics and do not change this example's geometry.

`benchmarks/tiff_integration_windows.py` remeasures the frozen 3,285 predicted
crossings with radii 5/7/9 pixels and angular half-widths 1/1.5/2.5 degrees,
holding the background-ring width at four pixels. Two six-pixel background-ring
controls retain comparability with the earlier wider-window result. It checks
the calibration hash and every loaded image/mask hash, and reproduces every
baseline and original wider-window record byte-for-byte. No indexing or cell
fit is repeated. The calibration, source TIFF hashes, frozen settings, per-window
arrays and per-reflection comparison CSV are retained under
`runs/tiff-integration-windows-20261025-v2/`. The earlier directory retains
the first measurement run; the second adds explicitly labelled plots and
detector cutouts/rocking curves at fixed baseline-intensity ranks. Repeated
measurements are byte-identical in every saved array.

There are 1,043 baseline unflagged >5-sigma crossings; 818 remain finite and
unflagged in all eleven windows. On this common population, changing only the
angular half-width from 1 to 1.5 degrees at radius five changes intensities by
a median absolute 2.35%. Changing only the radius from five to seven at
half-width one changes them by a median absolute 13.34%. Increasing radius
seven to nine still changes them by a median absolute 8.66%. Changing only
the background-ring width from four to six at the baseline support gives
2.78%. These are descriptive correlated comparisons, not significance tests;
the common population excludes reflections that become flagged in any window.
Pair-specific populations and results also remain in the report.

This example directs attention to spatial support, local background and
contamination before selecting an aperture. Increasing radius also moves the
background annulus, so the radius result is not a pure missing-signal estimate.
Endpoint-review counts decline with wider angular coverage but do not establish
complete reflections. No automatic aperture choice, rejection, scale correction,
absorption correction or CrysAlis intensity-agreement claim follows. Focused
verification on returning to TIFFs passed 27 PhaseSmith integration/quality tests
and 11 Dioptas adapter/widget tests; no production numerical behavior changed.

The subsequent guarded-plane increment adds spatial-boundary and poor-background
actions while preserving the original reports. Its TIFF support study, stored
ceiling checks, conservative support assessment and external DIALS integration
probe are described in `experimental-local-background.md` and
`single-crystal-integration-methods.md`. The installed Dioptas adapter reproduces
all six nominal/alternate plane measurements byte-for-byte. All 3,285 rows and
31 passing support checks remain exported under the v4 local-background run.
