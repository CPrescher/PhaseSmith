# Integration methods and the next PhaseSmith milestone

The [profile-bias follow-up](fe7c3-profile-bias-investigation.md) records the
implemented scattering-plane/frozen-shape method, known-count controls,
CCD/background audit, timings and limited structure check. Summation and profile
fitting remain distinct programmable choices. Strong-only shape learning avoids
per-weak-spot width-selection bias. A universal recommended default remains open.

Source review and local TIFF probe, 25 October 2026. This compares published
methods with the current experimental implementation; it does not establish
numerical equivalence between packages or the internal algorithms of proprietary
software. No external implementation code is copied into the numerical library.

## What the mature workflows add

The main step beyond a fixed-aperture sum is an empirical profile learned from
well-measured reflections. A reflection occupies detector x, detector y and
rotation angle. Geometry helps transform these observations into a coordinate
system in which neighbouring reflections have comparable shapes. A template
then provides both a foreground/background boundary and an intensity estimator
for weak observations. Background, missing-pixel handling and validation remain
essential even with an accurate profile.

| Package and source | Documented method | Consequence for PhaseSmith |
| --- | --- | --- |
| [DIALS methods, §3.4](https://journals.iucr.org/d/issues/2018/02/00/di5011/index.html), [current integration interface](https://dials.github.io/documentation/programs/dials_integrate.html) | Rotation-profile extents are estimated from strong indexed spots. Reference profiles vary across detector regions/image blocks; summation and profile fitting are distinct outputs. The documented Gaussian reciprocal-space model is supported alongside newer profile options. | Learn shape from strong reflections and preserve both estimators. A reciprocal-space Gaussian support model and an empirical fitted intensity profile are related but distinct layers. |
| [XDS INTEGRATE documentation](https://xds.mr.mpg.de/html_doc/xds_program.html), [parameter definitions](https://xds.mr.mpg.de/html_doc/xds_parameters.html) | Strong, fully recorded reflections supply templates for detector regions and rotation batches. A profile cutoff defines support; a second pass performs profile fitting. Missing contributions can be estimated from learned profiles subject to an observed-fraction limit. | Integration support should follow measured shapes. Missing counts require an explicit validated profile and coverage policy, not a larger sum or an undocumented correction. |
| [MOSFLM guide, integration details](https://www.mrc-lmb.cam.ac.uk/mosflm/mosflm/mosflm_user_guide.html), [Leslie 2006, §5.4](https://journals.iucr.org/d/issues/2006/01/00/ba5082/) | Local standard profiles are built from stronger spots across detector regions and image groups. A background plane is fitted with outlier handling; two-dimensional profile fitting improves weak measurements. Profile quality and insufficient local reference data are checked. | Pool information locally rather than thresholding each weak spot. Include profile-quality checks and a documented fallback to signed summation. |
| [Rigaku CrysAlisPro application note SMX004](https://rigaku.com/products/crystallography/x-ray-diffraction/application-notes/smx004-using-crysalispro-to-elucidate-twinned) | Rigaku documents simultaneous profile fitting/integration of twin components, smart background and bad-profile rejection. | Overlapping domains eventually need a joint intensity model. This public note does not establish exact template construction, objective functions or weighting rules. Current features must not be assumed present in the archived 2011 reduction. |

For a normalised profile P (sum P=1), a simplified pixel model is
expected counts = I times P + background. Here I is the reflection's integrated
intensity. Fitting its amplitude can use the strong central pixels without
discarding weak peripheral observations. Its accuracy depends on alignment and
profile transferability; a poor shape model can introduce systematic error.
Summation remains an independent, inspectable cross-check.

Background treatment also depends on the detector. [Parkhurst et al. 2016](https://journals.iucr.org/j/issues/2016/06/00/ap5008/ap5008.pdf)
demonstrates that inappropriate clipping of low-count Poisson backgrounds can
underestimate background and overestimate weak intensities. DIALS documents a
robust Poisson GLM for that regime. PhaseSmith's current clipped plane is an
unweighted least-squares/Gaussian-residual method with conditional covariance;
it is not that Poisson GLM. Detector gain, read noise and correlations still
need their own validation for the TIFF measurements.

## Implemented TIFF increment

PhaseSmith now has guarded local planes, background-outlier rejection, fitted
background covariance and spatial-boundary diagnostics, independently checked
against NumPy equations and known-count fixtures. Dioptas exposes the controls,
overlays the guard/background perimeter and exports diagnostics. Its raw-count
adapter also flags the integer storage ceiling when no lower detector limit was
supplied. This dataset contains 28,679 ceiling pixels across 63 of 76 frames;
41 nominal predicted crossings encounter such pixels at the tested support.
These are file-ceiling observations, not a separately calibrated detector limit.

The fixed TIFF cell/calibration supplies 3,285 predicted crossings. After the
pilot, a deliberately conservative descriptive assessment compares radius 11
with radii 13/15, background gaps 1/5 versus 3, and angular half-width 4.5 versus
2.5 degrees, with a six-pixel background ring. It requires >5-sigma nominal
signal, clear measurement/endpoint flags in every variant and <=5% relative
change in each comparison. Thirty-one crossings pass these checks. All other
measurements and individual failure reasons remain exported; they are not
silently removed. This is support stability, not corrected-I(hkl) truth.

The separate radial sweep fails its predeclared training-only universal-radius
criterion. No recommended radius is invented. These results make empirical
profiles and explicit contamination/partiality handling the next useful work.
See `runs/tiff-local-background-20261025-v4/` for full records, the support CSV,
raw-count exports and installed-adapter verification. The original flat-background
TIFF baseline reproduces byte-for-byte.

## Actual DIALS integration probe

`benchmarks/dials_integration_probe.py` runs the isolated DIALS installation on
the same preserved TIFF pixel arrays and affine calibration. It uses DIALS's
previously discovered crystal and 605 reference observations; it does not give
DIALS a published Fe7C3 cell or PhaseSmith intensities. Stored 65,535 values are
outside the trusted range. Default profile/background integration is retained,
with one worker and unintegrated rows preserved.

The probe returns 3,371 predictions, 2,291 sum-integrated rows, 2,303
profile-integrated rows and 60 overloaded rows. These populations can overlap;
an integration flag is not a correctness certificate. The run also returns
partiality, background and LP fields. Their conventions, HKL basis and exact
crossing identities must be reconciled before numerical intensity comparisons.
Different prediction/flag populations cannot rank the two methods' accuracy.
Outputs and parameters are under `runs/dials-tiff-integration-20261025/`.

## Profile implementation and remaining targets

1. Select isolated, unsaturated strong reference spots and verify centroids.
2. Learn detector-local and rotation-local three-dimensional profiles, keeping
   an independent validation population. Preserve asymmetry and test shape drift.
3. Derive explicit foreground, guard and background masks from those profiles;
   retain the summed estimator and fit profile amplitudes independently.
4. Return observed fractions, profile residuals and failed-fit reasons. Do not
   repair clipped/missing counts without a separately validated policy.
5. Add multi-domain joint fitting only with identifiability and overlap tests.
6. Compare matched unmerged measurements with DIALS, then assess detector and
   crystallographic corrections/scaling separately. The older MAR/CrysAlis archive
   remains set aside at the user's request.

The empirical three-dimensional profile fitter is included in ordinary builds, with signed zeta-normalised rocking coordinates, local
profile banks, a global fallback and exact pixel/exposure overlap integration.
It retains the guarded rectangular background ring and flags actual learned
foreground overlaps; joint domain fitting remains open. The frozen-geometry
Fe7C3 comparison fails the intensity-agreement gate, so this is not yet a
validated replacement for established integration software. See
[implementation equations and API](experimental-3d-profiles.md) and
[measured comparison](fe7c3-3d-profile-comparison.md).
