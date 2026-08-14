# Profile transferability gate

## Current status

The first external empirical Unit-29 transfer test is complete. It uses the
IUCr-sponsored ceria size/strain round robin: annealed narrow-line CeO2 is the
calibration specimen and the broadened CeO2 specimen is the untouched holdout.
Both were measured on the same University of Birmingham sealed-tube instrument
with an incident Ge(111) monochromator.

This is independent historical laboratory evidence, not in-house evidence, and
is labelled that way. The six numbered Birmingham files are three contiguous
angular ranges per specimen, not repeat scans. Their original bytes remain
external because the source page states no redistribution licence.

The current evidence supports a negative production decision: do not add LPSD
defocusing, tube-tail, continuum, or coupled-dispersion profile terms. The
Rowles physical-profile compression worsens both real holdouts, the citrate
forensics do not isolate a transferable missing term, and the ceria archive
lacks the physical metadata required to test those terms fairly.

## Reviewed external result

The checksum-pinned driver is
`benchmarks/validate_ceria_transferability.py`; its reviewed finite report is
`validation/results/2026-08-14-ceria-transferability.json`.

- The sharp calibration uses 8,726 samples. Three dispersed FWHM starts select
  the same W-only empirical profile to `3.45e-8` relative width spread. Median
  Poisson Rwp is `0.192590`, background-subtracted correlation is `0.953870`,
  and each accepted fit reports weighted rank `1/1` with zero pairwise
  parameter correlation.
- That profile is frozen before the 4,126-sample broadened holdout. The
  no-sample-broadening baseline gives Rwp `0.553990`; the existing isotropic
  size plus Gaussian microstrain model gives `0.058097` and correlation
  `0.985574`, reducing weighted SSE by `98.9002%`.
- Three dispersed size/strain starts agree within `0.0692 nm`, `4.79e-5`
  microstrain, and `3.92e-6` Rwp. Twelve deterministic Poisson resamples put
  the maximum transfer-improvement variation at `0.001786`, far below the
  observed improvement.
- Peak-wise centroid, FWHM, area, normalized-L1, and first-/second-moment
  diagnostics are retained rather than hidden by the full-pattern metrics.
  They show that the selected empirical profile is not a high-fidelity
  fundamental-parameters description, especially at high angle.

The fitted approximately `30.9 nm` Scherrer-convention size is a profile-model
parameter, not a replacement certification for the round-robin material. The
published study uses different size-weighting/profile conventions.

The source documents Cu K-alpha1/K-alpha2 wavelengths, intensity ratio, and
polarization, but not the radius, apertures/detector geometry, axial lengths,
Soller angles, monochromator passband, or specimen mounting metrology. The
current Le Bail estimator also uses the dominant K-alpha1 wavelength rather
than treating the documented 1.6% K-alpha2 component as a complete fixed
spectrum. Consequently this case passes only the empirical existing-model
transfer gate. It cannot authorize a specialized optics term.

## Minimum input package for reopening specialized terms

The next campaign addition needs one calibration/holdout pair acquired on the
same laboratory instrument without fitting the holdout first. It must provide:

- raw pattern files with the original x/count columns, masks, and any counting
  uncertainties or repeat scans;
- immutable file sizes and SHA-256 hashes plus permission/license notes;
- source wavelengths and relative line intensities;
- goniometer radius, scan range/step, receiving aperture or detector geometry,
  axial dimensions, Soller angles, and any monochromator/filter configuration;
- specimen geometry/preparation and enough metadata to distinguish reflection
  from transmission/capillary use;
- a citable calibrant structure or certification and an independently supplied
  holdout structure/composition; and
- an explicit statement of which pattern is calibration and which is untouched
  holdout before any candidate model is fit.

Unknown physical inputs remain `unknown`; they are not silently filled from a
GSAS-II or vendor default. If the missing value prevents evaluation of a
candidate term, the case is classified as blocked rather than tuned around it.

## Protocol

1. Register the input bytes and metadata in the external-dataset ledger.
2. Fit the existing production U/V/W/X/Y, fixed spectrum, and FCJ subset to the
   calibration pattern only. Record peak-wise centroid, FWHM, area, normalized
   L1/moment errors, full-pattern Rwp/correlation, rank, correlations, and
   repeat/start sensitivity.
3. Freeze those profile parameters. On the holdout, allow only the
   predeclared scale, background, and physically justified position nuisance
   terms; do not refit profile shape.
4. Repeat steps 2–3 for one named candidate term implemented first in the
   independent reference path. Use identical masks, weights, nuisance terms,
   and stopping rules.
5. Publish both successful and failed transfer results as plain finite records.

## Promotion gate

A specialized production term advances only when all of these conditions hold:

- its physical inputs are measured or independently documented;
- its calibration parameters are identifiable and stable across at least three
  dispersed starts (or across repeat scans when available);
- freezing it improves holdout weighted SSE by at least 10% relative to the
  current model and by more than three times the observed repeat/run or
  uncertainty-resampling variation;
- the improvement is not purchased by a material centroid, integrated-area,
  phase-fraction, or background regression;
- the same sign/convention is supported by the physical model and the
  independent implementation; and
- values and analytical derivatives can be added to the Rust peak/sample pass
  with finite-difference, normalization, boundary, and benchmark coverage.

If neither repeat scans nor a defensible count-uncertainty resampling model is
available, the variation clause cannot pass and the result remains diagnostic.
A failed gate is a valid conclusion and leaves the current production model
unchanged.

## Handoff

The external empirical gate is no longer input-blocked. The production decision
remains unchanged: do not add LPSD, tube-tail, continuum, or coupled-dispersion
terms. Reopening any one of them requires a new calibration/holdout pair with
the term's measured physical inputs and the full promotion protocol above. A
future in-house pair would provide stronger evidence but must remain labelled
separately from this historical round robin.
