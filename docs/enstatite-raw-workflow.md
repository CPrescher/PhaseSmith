# MgSiO3 raw-frame-to-structure-search example

This explicit example coordinates PhaseSmith's existing detector, reciprocal,
indexing, position-refinement, integration, profile, scaling and structure-search
APIs. It reads original HDF5 counts and generates new measurements throughout.
The first protocol is `validation/enstatite-raw-workflow-v1.json`, committed
before executing the measured workflow. The current default is version 11,
which adds native profile and empirical-scaling fits after version 9 validated
native geometry and version 8 validated bundled HDF5 input and Rust shadow masks.
The scientific models and downstream settings remain unchanged. Version 10
recovers the structure but fails strict numerical equivalence; its first outcome
is retained, and version 11 restores the previous bounded-BFGS optimizer family. Earlier versions and
failures are retained as described below. Version 5 first passed complete labelled recovery; see the
[reduction diagnosis](enstatite-reduction-diagnosis.md). This is exposed
development/regression data with declared cell and symmetry priors, not a new blind holdout.

## Run and storage

Install a feature-enabled PhaseSmith build and the optional dependencies:

```sh
python -m pip install 'phasesmith[single-crystal-example]'
# Development build, until single-crystal features join ordinary wheels:
maturin develop --release --features experimental-spots,experimental-solution,hdf5
python benchmarks/enstatite_raw_workflow.py \
  --data-root /path/to/MgSiO3-files \
  --output /path/outside/the/repository/enstatite-raw-run
```

The output directory must be new and outside the checkout. All large observation
arrays, shoeboxes, profiles, detector positions, maps and candidate CIFs are
written there. The repository contains code, input/protocol hashes, and small
result summaries only. No input file is modified, downloaded or uploaded.
Original input identities are pinned in `enstatite-mgsio3-local-v1.json`. The
frame reader now uses bundled native HDF5 and Bitshuffle/LZ4, with no h5py or
hdf5plugin dependency. See [native HDF5 input](native-hdf5.md) for build/runtime
contracts. See [native profile/scaling fits](native-profile-scaling-fitting.md)
for the numerical ports and remaining Python orchestration/design scope.

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

## Bundled native HDF5 validation (v8)

The complete v8 run uses the bundled Rust HDF5/Bitshuffle reader and Rust shadow
kernels, with v7 scientific settings unchanged. In a clean environment without
h5py/hdf5plugin and with dynamic filter loading disabled, it recovers
Mg16 Si16 O48 at R1 0.0411159 and maximum/RMS coordinate errors
0.0065804/0.0045658 A. All reduced arrays, positions and original-frame hashes
are bitwise identical to v7; all eight source files remain unchanged.
The full worker takes 95.85 s. See
`benchmarks/enstatite-bundled-hdf5-macos.json` and
[native HDF5 input](native-hdf5.md) for the artifact, contracts and validation.
The earlier v5/v6/v7 outcomes remain retained. This verifies the complete
existing workflow through native input, while Python orchestration and the
experimental geometry/profile/scaling optimizers still required native ports
at that checkpoint. Their later migrations are described below; Python
orchestration and design construction remain before a standalone Rust
raw-to-structure executable can be claimed.

## Native single-crystal geometry fitting (v9)

The complete v9 rerun uses native geometry parameter chains, robust bounded
optimization and fit diagnostics, with v8 scientific settings unchanged. All
predeclared comparison gates pass. The same 4,125 observations, export mask,
HKLs, flags and inferred Mg16 Si16 O48 are retained. Maximum indexed-prediction
changes are 2.13e-7 pixels and 6.18e-8 degrees. Intensities differ by 9.21e-6 in
relative L2 norm, at most .009495 baseline sigma. Search R1 is .0410522 versus
.0411159, and maximum labelled reference error is .0069006 A. The largest
origin/hand-aligned coordinate change from v8 is .0005351 A. Full time is 94.53 s.

Geometry fitting now runs without SciPy; standalone Rust and clean NumPy-only
wheel tests cover that contract. All source hashes remain unchanged and large
products stay outside Git. At v9, profile fitting and empirical scaling
remained Python numerical optimizers; the subsequent native migration is
described in [native profile/scaling fits](native-profile-scaling-fitting.md).
Python raw-workflow orchestration remains. See [native geometry calibration](experimental-crystal-geometry.md)
and `benchmarks/enstatite-native-geometry-macos.json` for methods, tolerances,
artifact identities and the full comparison.

## Native profile fitting and empirical scaling (v10/v11)

Version 10 first migrated these numerical paths but its simplified projected
quasi-Newton optimizer missed strict reduction equivalence. It still recovered
the correct structure. The complete first outcome is retained in
`benchmarks/enstatite-native-profile-scaling-v10-macos.json`. A separate
frozen-v9-shape diagnostic matches raw GLS intensities exactly, joint sigma to
3.4e-16 relative L2, and final scaled intensity/sigma to 2.91e-9/1.49e-7.
That diagnostic intentionally reads archived arrays and is not a fresh raw run.
It isolates the difference to optimizer trajectories.

Version 11 uses native limited-memory BFGS generalized Cauchy/free-subspace
steps and a safeguarded More–Thuente search. Its fresh full raw run passes
structure recovery: Mg16 Si16 O48, 246 teachers, 4,125 exported signed
observations including 220 negative, R1 .0408518207, and maximum/RMS labelled
reference errors .0075236/.0045385 A. The worker takes 95.24 s. All source files
remain unchanged; geometry/positions, exported HKLs and use mask match v9.
The reduction passes every inherited numerical tolerance: intensity/sigma
relative L2 differences 2.42e-5/2.15e-5, maximum intensity change .02153
baseline sigma, 95th percentile 1.024e-5 sigma. One MISMATCH warning changes;
the existing warning policy retains that row and the export mask is unchanged.

Overall strict end-to-end equivalence remains **false**: search R1 changes by
.00020038 (limit .0001), and the maximum aligned candidate-coordinate change
is .0017993 A (limit .001). Both searches stop at the existing R1 target and
recover the same labelled structure well inside the scientific recovery gates.
These two stricter migration limits are preserved, not loosened or hidden.
No search seed, effort, scientific threshold or optimizer tolerance was chosen
by atomic-reference score or R1. The final record is
`benchmarks/enstatite-native-profile-scaling-macos.json`.

Both reduction and search workers reject SciPy, h5py and hdf5plugin imports.
A clean wheel also runs the profile/scaling benchmark with only NumPy installed.
Rust entry points own profile objectives/optimization, expected-count updates,
joint sigma, frozen weak filters, family elimination, bounded empirical solves
and Huber IRLS. Python still orchestrates the example and constructs designs;
a standalone Rust raw-to-structure executable and standalone structural
refinement remain outside this change. Methods and independent validation are
in [native profile/scaling fitting](native-profile-scaling-fitting.md).
