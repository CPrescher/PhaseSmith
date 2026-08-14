# In-house profile transferability gate

## Current status

Unit 29 is not complete. The workspace inventory on 2026-08-14 found no new,
independent in-house calibration/holdout pair beyond the public/reviewed cases
already in the benchmark campaign. External, synchrotron, historical, or
already tuned patterns are not relabelled as new in-house evidence.

The current evidence supports a negative production decision: do not add LPSD
defocusing, tube-tail, continuum, or coupled-dispersion profile terms. The
Rowles physical-profile compression worsens both real holdouts, and the citrate
forensics do not isolate a transferable missing term. This decision can be
reopened only with the prospective evidence below.

## Minimum input package

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

## Prospective protocol

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

The missing external input is the only blocker for executing this prospective
gate. Once the files are supplied, the first work item is registration and a
readiness report—not numerical tuning. The existing machine-readable campaign
runner remains the aggregation boundary after the new pair has its own reviewed
driver and result contract.
