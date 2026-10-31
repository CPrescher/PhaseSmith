# MgSiO3 raw-frame-to-structure-search example

This explicit example coordinates PhaseSmith's existing detector, reciprocal,
indexing, position-refinement, integration, profile, scaling and structure-search
APIs. It reads original HDF5 counts and generates new measurements throughout.
The first protocol is `validation/enstatite-raw-workflow-v1.json`, committed
before executing the measured workflow. The current default is version 5;
earlier versions and failures are retained as described below. Version 5 passes
complete labelled recovery; see the
[reduction diagnosis](enstatite-reduction-diagnosis.md). This is exposed
development/regression data with declared cell and symmetry priors, not a new blind holdout.

## Run and storage

Install a feature-enabled PhaseSmith build and the optional dependencies:

```sh
python -m pip install 'phasesmith[single-crystal-example]'
# Development build, until single-crystal features join ordinary wheels:
maturin develop --release --features experimental-spots,experimental-solution
python benchmarks/enstatite_raw_workflow.py \
  --data-root /path/to/MgSiO3-files \
  --output /path/outside/the/repository/enstatite-raw-run
```

The output directory must be new and outside the checkout. All large observation
arrays, shoeboxes, profiles, detector positions, maps and candidate CIFs are
written there. The repository contains code, input/protocol hashes, and small
result summaries only. No input file is modified, downloaded or uploaded.
Original input identities are pinned in `enstatite-mgsio3-local-v1.json`.

The script uses no Dioptas/Qt, vendor HKL/peak table, archived intensity/profile
array, atomic starting model, SHELXT or SHELXL. Supplied cell, group and element
types are explicit priors. The parent verifies input hashes, then passes only
whitelisted raw/calibration paths and priors to the numerical worker. Reference
coordinates and counts are read only after a candidate is saved and that worker
has stopped. Failures and original stage outputs are retained, never replaced.

`--through geometry` supports an explicitly partial positional check. It does not
pass or represent the complete workflow. A completed search process returns
exit status 0 even when no candidate is found. Inspect `report.json` for the separate `raw_to_structure_search_passed`
gate; execution success and numerical R1 are separate from labelled coordinate
recovery. Process errors/timeouts return status 2.

## Fixed acquisition assumptions

- Original zero-based frames 2--143 are used. Source notes place frames 0/1 on
  the DAC seat/body; they are explicitly excluded.
- Bounds start at -36 degrees for original frame 0 and advance by 0.5 degrees
  with 0.5-degree exposures. HDF5 records exposure time but no spindle angles.
  This carries forward the previously adopted provisional angle convention.
- Reverse raw rows once into the legacy PONI's already reversed array
  convention. Pixel centers are zero-based. The existing PONI vector conversion
  uses the HDF5's 75-micrometre pixels and declared spindle axis `(-1,0,0)`.
- Exclude only reserved uint32 values 4,294,967,294 and 4,294,967,295. Real zeros
  and high finite EIGER counts remain. Noise uses declared gain=1/read-noise=0.
- The supplied CeO2 PONI is the initial affine plane. Its wavelength remains
  fixed; an indexed-position calibration uses the declared source cell, never
  atomic coordinates or reflection-intensity targets.

## Stages and gates

1. Stream each selected frame through `locate_series`, retaining all individual
   detections/links and source-frame hashes. The example holds at most the
   bounded detector worker queue in memory, not the full image cube.
2. Correct linked observations to reciprocal space. Use unflagged tracks with
   at most eight observations and q RMS <=0.03 inverse angstrom. Whole five-frame
   blocks and tracks remain in training or withheld sets.
3. Run the generic multi-grain FFT search. Select a primitive orthorhombic
   setting by training support and the explicit supplied metric prior; align
   axes by integer settings/permutations. Retain every candidate record.
4. Refine detector pose and crystal orientation from indexed positions, fixing
   the supplied cell. Complete reserved HKL families and discovery-withheld
   blocks never teach this calibration. Require >=100 training and >=30 withheld
   assignments, convergence without a bound, withheld xy RMS <=1 pixel and omega
   RMS <=0.3 degrees. These are conditional position gates, not metrology proof.
5. Preserve up to two independently position-supported competing orientations.
   Refit their bases on training q after calibration; require 30 training/10
   withheld assignments and withheld reciprocal RMS <=0.015 inverse angstrom.
   They are alternative orientation models, not assigned physical grains/phases.
6. Predict target crossings to q=2 inverse angstrom; supplied symmetry determines target
   absences. Restrict the declared rocking-coordinate support to |zeta|>=0.1.
   Collect target predictions and measured competing tracks together so their finite apertures exclude each other's
   backgrounds and flag overlaps. Stream the original counts again and require
   identical per-frame hashes to the detection pass. Use the existing guarded
   plane background and finite collection windows from the protocol.
7. Select target shoeboxes **after** joint collection, preserving its original
   fitted backgrounds/covariance. Learn 31x31x23 finite profiles from complete
   training families only. Require at least 40 strong teachers. Declared strong
   observations may fit their centers/widths; weak observations retain fixed
   estimators. Signed aperture and profile counts are retained separately.
8. Apply declared rotation Lorentz/polarization factors. The first recipe uses
   horizontal polarization fraction 0.98 and neutral detector efficiency; no
   diamond absorption or vendor transfer chain is inferred. Fit the predeclared
   frame/panel/ray empirical model from training equivalents only, with neutral
   strength 1,000. Retain quality flags and the optional equivalent-error floor.
   Shapes, sensor or scale models never select settings using structural R1/ADPs.
9. Export >=1,000 finite, positive-sigma, quality-controlled signed observations.
   Exclusions are by explicit flags/support, not intensity significance. Run
   PhaseSmith element-only search with seed 0, four starts, 800 cycles, 128 MiB
   dense-buffer budget and (version 4) a 60-second native safeguard/75-second process cap.
   Input-only preflight sets atom capacity; reference counts do not set it.
10. Save the chosen candidate before reference grading. The established independent
    periodic element-preserving auditor requires exact counts, maximum coordinate
    error <=0.05 A and RMS <=0.025 A. A strict 0.02 A maximum is secondary.
    Original outcomes remain visible even when a later diagnostic improves them.

Internal candidate polishing is part of search. This example adds no standalone
single-crystal structure-refinement workflow or calibrated atomic uncertainties.
The known sample, supplied priors and previous inspection preclude unseen-data
claims; independent transfer validation still needs other experiments.

## Array adapters

`prepare_profiles(..., grain_ids=...)` forwards typed domain IDs to the existing
joint integration kernel. Omitting them retains the previous behavior.
`profiles.select_observations(observations, ids)` copies sorted unique in-range
reflection rows, preserving original all-domain backgrounds, sample/covariance
groups and signed values. It does not refit or reread measurements. Synthetic
tests check domain-ID propagation, unchanged numerical counts, covariance/row
selection and malformed inputs, alongside raw HDF5 orientation/mask and family
partition checks. No numerical kernel, physical equation or support changed.


## Preserved first outcome and bounded follow-up

Version 1 detected 3,308 tracks from 4,483 frame detections on 142 raw frames.
It recovered the primitive enstatite-like cell and passed calibration gates:
1,421 training/782 withheld assignments, 0.3342 pixel withheld xy RMS and
0.16395-degree omega RMS. It stopped before integration because one provisional
competing cell required a 310,005-HKL enumeration, above the default 250,000 cap.
No intensity or atomic comparison was produced or consulted.

Version 2 uses the measured unassigned competing tracks as exclusion supports.
It does not invent unobserved reflections from provisional large-cell models.
The target remains fully prediction-driven, including weak/negative intensities.
Its explicitly bounded 128-MiB dense budget accommodates the volume-derived atom
cap for potentially broader new q<=2 coverage without a reference count. Seed,
cycle/time budgets and scientific gate thresholds are unchanged. The frozen
version-1 protocol/run are retained; the small record is
`benchmarks/enstatite-raw-workflow-attempts-macos.json`.

Version 2 completed integration of 7,286 target crossings and learned 278
training profiles, then stopped before scaling/search: nine already EDGE-flagged
centers were within the predictor's half-pixel footprint but outside the scaling
mesh's pixel-center rectangle. Version 3 retains those rows in edge diagnostics
and omits only those pre-flagged centers from profile learning/reduction. No
unflagged measurement, mesh bound, scientific threshold or atomic target selects
this repair. Earlier outcomes and both immutable protocols are retained.

Version 3 completed raw reduction and search: 5,708 observations (including
1,233 negative intensities), 278 training teachers and matching raw-frame hashes.
All four starts failed the existing phase-origin gate `alpha <= 0.3`; no atomic
candidate was produced and no reference model was opened. Starts 2/3 hit their
per-start phase allowance at 396/398 of 800 cycles. Version 4 is one predeclared
longer-time diagnostic: 60-second native safeguard/75-second process cap, with
the same seed, four starts, 800 cycles, signed data and scientific gates. It
reruns the full original-frame workflow in a new external directory.

## Recorded complete run

Version 4 completed all stages in 103.44 seconds on the recorded macOS machine.
It read original frames twice with matching hashes, retained 7,277 target rows
after the nine flagged edge-center exclusions and exported 5,708 quality-valid
observations, including 1,233 negative intensities. It inferred an admissible
105-atom cap from input volume/coverage and the explicit dense budget.
The complete reduced arrays are bitwise identical to version 3.

The search took 5.27 seconds. Its scheduled 200/200/800/800 cycles all finished;
each start failed the existing phase-origin `alpha <= 0.3` check. No atomic
model was produced, so neither the atomic reference nor its counts was read.
The output reports process success and **failed structure recovery** separately.
The longer allowance rules out the earlier per-start time truncation for this
recorded run; it does not identify extraction or phasing as the cause.

The small machine-readable report is
`benchmarks/enstatite-raw-workflow-macos.json`. Original files, large shoeboxes,
profiles, maps and logs remain outside Git at the local path recorded there.
No scientific gate, phase-origin threshold or intensity selection was relaxed.
The next scientific investigation is the fresh reduction/phasing discrepancy
relative to the previously successful archived-intensity search, using saved
measurements and equivalent-family diagnostics before changing the recipe.
This run does not pass the promotion gate.

## Measurement-validity repair (version 5)

The retained v4 failure comes from two interacting observation-selection errors.
First, `flags == 0` excludes positive, high-signal profile fits solely because
the residual model is imperfect: MISMATCH is set for I > 5 sigma and reduced
chi-square >4. This discards 711 rows containing 71.3% of the positive intensity
in otherwise usable rows, while accepting background-like counterparts.
Second, detector coverage does not guarantee visibility through the DAC: strong
equivalent measurements collapse above an exit angle of approximately 35 degrees
in the sample frame. Those occulted crossings were exported as measured zeros.

Version 5 declares MISMATCH as a retained warning; every other flag still excludes.
All signed values, sigmas and original flags remain in the saved NPZ. It also
declares a conservative symmetric cone of 35 degrees, with its axis along the
incident beam at omega zero. For unit ray s and spindle axis a, Rodrigues rotation
R(a,-omega) places incident and outgoing rays in that sample frame. A centre is
visible when both |R(a,-omega)s dot n| >= cos(35 degrees), where n is the declared
DAC axis. The boundary is closed, allowing eight float64 epsilons in cosine.
The mask uses geometry only and is applied to export after unchanged reduction.
Blocked measurements and their entry/exit angles remain available in diagnostics.
This is not an absorption or finite-rocking partiality correction.

The aperture is an exposed-data, raw-equivalent diagnostic assumption, consistent
with acquisition notes that remove seat/body frames. It was declared before the
diagnostic model searches and was not selected by structural R1 or coordinates.
It needs independent calibration/validation on other acquisitions. A source-HKL
membership control was used only to isolate the cause; neither vendor row
membership nor vendor intensities enter the v5 numerical worker. Existing
profile/scaling reference selection, search settings and coordinate gates remain
unchanged. Original v1--v4 outcomes remain frozen.

The frozen v5 full raw rerun passes: 4,310 exported observations, inferred
Mg16 Si16 O48, R1=0.0413175 and maximum/RMS reference errors
0.0078273/0.0050056 A. Every integrated value/sigma/flag is bitwise identical to
v4; eligibility changes only. Search takes 0.608 seconds in the 100.42-second
complete worker. This establishes the qualified MgSiO3 example, not general
mask calibration or public API promotion. The retained v4 summary is now
`benchmarks/enstatite-raw-workflow-v4-macos.json`; the current successful summary
is `benchmarks/enstatite-raw-workflow-macos.json`.
