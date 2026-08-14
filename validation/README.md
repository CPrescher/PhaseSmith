# External real-data validation

PhaseSmith does not bundle tutorial patterns and does not import or execute
GSAS-II. The command below explicitly downloads files from commit-pinned HTTPS
locations, verifies their byte sizes and SHA-256 hashes, and runs only
PhaseSmith code:

```bash
python tools/validate_real_data.py --fetch \
  --include-diagnostics \
  --data-directory validation/data \
  --output validation/results/local.json
```

The IUCr ceria size/strain round robin has a separate empirical
calibration/holdout gate because it does not require GSAS-II:

```bash
python -c "from phasesmith.validation import fetch_validation_dataset; \
fetch_validation_dataset('iucr-ceria-size-strain-round-robin', \
'validation/data/iucr-ceria-size-strain-round-robin')"
python benchmarks/validate_ceria_transferability.py \
  --dataset-directory validation/data/iucr-ceria-size-strain-round-robin \
  --json-output validation/results/local-ceria-transferability.json
```

The fetch is explicit, source bytes remain ignored, and an existing report is
not overwritten unless `--overwrite` is passed.

The reviewed public-API comparison with XRD-Rust is preserved in
`results/2026-08-13-xrd-rust-performance.json`. Its calculation scope,
numerical gate, interpretation, and reproduction command are documented in
`docs/xrd-rust-performance.md`; XRD-Rust remains a benchmark-only dependency.

The canonical suite uses native runners. Add `--interactive-qarr` to run QARR
1g through the independent Python path with structured progress and cooperative
cancellation: press `q` or Ctrl+C once to stop at the next safe batch boundary;
a second Ctrl+C forces interruption.

## Unified GSAS-II comparison campaign

The broader black-box comparison campaign is declared in
`validation/benchmark-campaign.json`. It composes the existing case-specific
drivers; it does not duplicate their numerical workflows or import GSAS-II into
PhaseSmith. List the reviewed cases without configuring an oracle:

```bash
python tools/run_benchmark_campaign.py --list
```

Run the complete campaign with the normal PhaseSmith interpreter while passing
the isolated pinned GSAS-II interpreter and checkout explicitly:

```bash
python tools/run_benchmark_campaign.py \
  --gsas-python /path/to/gsasii-python \
  --gsas-root /path/to/GSAS-II \
  --binary-dir /path/to/GSAS-II/bindist \
  --data-directory validation/data \
  --output validation/results/campaign-YYYY-MM-DD.json \
  --continue-on-error
```

Before any scientific driver runs, the command verifies the checkout against
`oracle/PINNED_GSASII.json` and verifies every selected dataset's exact byte
sizes and SHA-256 hashes. Each generated case result must carry the same oracle
revision and satisfy the case-level assertions in the manifest. The aggregate
JSON embeds every driver report and its canonical SHA-256. Existing output is
never replaced unless `--overwrite` is supplied; reviewed goldens are not
regenerated implicitly.

Use repeated `--case CASE_ID` options for a focused rerun. `--fetch` downloads
missing registered inputs from their pinned HTTPS locations before verification.
The command exits nonzero for a pin, checksum, driver, provenance, or declared
scientific-gate failure. Diagnostic cases are successful only when their
reviewed diagnostic executes and records the pinned oracle provenance; they are
not relabeled as scientific parity passes.

The default command contains only accepted, complete runners. Two additional
diagnostic cases are intentionally invoked separately because one is a failing
transferability holdout and the other exposes an unsupported workflow boundary:

```bash
cargo run -p phasesmith-validation --bin phasesmith-validation -- run \
  iucr-qarr-1h validation/data/iucr-qarr-1h
cargo run -p phasesmith-validation --bin phasesmith-validation -- run \
  powgen-lab6-tof-calibration validation/data/powgen-lab6-tof-calibration
```

`validation/data/` and local result files are ignored. The current cases
have deliberately different meanings:

- `iucr-ceria-size-strain-round-robin` pins six University of Birmingham
  laboratory X-ray ranges. Annealed narrow-line CeO2 calibrates the empirical
  profile; broadened CeO2 is not inspected until that profile is frozen. Three
  calibration starts select the same W-only profile. The existing isotropic
  size/Gaussian-microstrain model lowers holdout Rwp from 0.553990 to 0.058097,
  reaches correlation 0.985574, and reduces weighted SSE by 98.9002%. Three
  dispersed sample-width starts and twelve deterministic Poisson resamples pass
  the stability gates. Peak-wise metrics qualify the result, and incomplete
  physical optics metadata blocks LPSD, tube-tail, continuum, and
  coupled-dispersion evaluation. No specialized term is promoted.
- `aps-sucrose-11bmb` exercises the supported monochromatic FXYE → background
  → symmetry/reflection generation → Le Bail → analytical profile-refinement
  path on the official APS 11-BM sucrose tutorial pattern. Its `Rwp <= 0.22`
  threshold is a regression smoke gate for the present model, not a claim of
  numerical equivalence with GSAS-II. The tutorial's lower final residual uses
  additional staged background-peak, crystallite-size, microstrain, lattice,
  and repeated extraction refinements. The matched oracle comparison gives
  both methods the identical fixed Smooth Bruckner array and one refinable
  constant Chebyshev residual. It also overrides GSAS-II's legacy instrument
  import so both start from the same symmetric U/V/W/X/Y state. The reviewed
  PhaseSmith Rwp is 0.14184 with correlation 0.99070; matched GSAS-II returns
  0.14535 and 0.97300. GSAS-II's internal calculation floor reports SH/L as
  0.0005 even though the shared input value is zero.
- `ansto-echidna-lab6-cw-neutron` exercises a second independent
  constant-wavelength neutron instrument using the deposited three-column LaB6
  pattern and its uncertainties. On the selected 20–125.5° range, the native
  Le Bail/profile runner uses 2,111 samples and 13 reflections, lowers Rwp from
  0.51822 to 0.40858, and reaches correlation 0.90621. Its fixed ten-term
  Chebyshev background is initialized from twenty regular Smooth-Bruckner
  startup anchors. The deposited 2.047 Å
  wavelength is explicitly approximate, so this is a profile smoke gate rather
  than a wavelength or detector-zero calibration claim.
- `iucr-qarr-1g` verifies the 7,251-point 5–150° input and its explicit Cu Kα1/
  Kα2 instrument metadata, runs a native three-phase fixed-spectrum structural
  refinement, and converts the final polished scales with the Hill--Howard
  relation. The reviewed native baseline returns Al2O3 30.460%, ZnO 34.113%,
  and CaF2 35.427% against weighed targets of 31.37%, 34.21%, and 34.42%. Its
  largest absolute error is 1.007 weight-percentage points. The profile gates
  distinguish Poisson-weighted Rwp (0.19828, limit 0.20) from unit-weight Rwp
  (0.13179, limit 0.15); profile correlation is 0.99062. The final scale-only
  covariance propagates to a maximum phase-fraction standard uncertainty of
  0.000852.
- `iucr-qarr-1h` is an independent mixture from the same round robin and uses
  the unchanged 1g setup as a transferability holdout. QPA remains within the
  0.02 absolute-fraction gate (maximum error 0.01725), but Poisson Rwp 0.27896,
  unit-weight Rwp 0.24149, and correlation 0.97763 fail the accepted profile
  gates. The runner reports `failed`; the thresholds
  are not relaxed to turn this diagnostic into a pass.
  A separate GSAS-II-parity workflow matches the ten-term background,
  size/microstrain model, refinement blocks, and trace-mean isotropic
  displacement representation in both programs. Without tuning against 1h,
  it reaches 18.358% Poisson Rwp versus GSAS-II's 18.636%; the maximum
  cross-program phase-fraction delta is 0.00415. This does not rewrite the
  historical acceptance result above.
- `curtin-rowles-qpa-topas` pins two laboratory Bruker D8 patterns (`1a` and
  `1e`) plus their deposited TOPAS v6 inputs and include file. The narrow
  converter emits neutral XY, CIF, GSAS-II instrument, and JSON records while
  retaining every omitted or approximated source term. The accepted
  PhaseSmith common-doublet workflows recover the weighed fractions within
  0.74 and 2.16 percentage points respectively. PhaseSmith and pinned GSAS-II
  give 8.782%/9.085% Rwp for `1a` and 8.264%/8.195% for `1e`; the largest
  cross-program phase-fraction delta is below 0.25 percentage points. This
  parity came from the correct Lorentzian `Mustrain` convention, a unit shape
  factor for the shared size convention, and an exact alternating linear block
  for phase scales and background. A separate pinned GSAS-II FPA diagnostic
  compresses the deposited geometry successfully on synthetic peaks (6.952%
  Rwp) but worsens the real-pattern Rwp from 9.085% to 13.928% for `1a` and
  from 8.195% to 11.710% for `1e`. TOPAS-only source and optics terms remain
  excluded and are listed in `docs/topas-rowles-model.md`.
- `nist-srm660c-lab6-xray` validates all 20 pdCIF specimens in NIST's official
  SRM 660c archive: 106,640 measured points, aligned released calculated
  profiles, the certified lattice interval, aggregate correlation 0.99933, and
  weighted Rwp 0.06083. Every specimen also clears correlation 0.999 and Rwp
  0.065 gates. This validates the external oracle archive. It is not a
  pointwise PhaseSmith equivalence claim because the released fit uses a Cu
  emission spectrum and fundamental-parameters optics model.
  A separate specimen-100a empirical holdout compares the physically valid
  common subset with pinned GSAS-II. PhaseSmith/GSAS-II Poisson Rwp values are
  18.912%/13.984%, and correlations are 0.96932/0.98983, so the strict profile
  parity gate remains failed. Both independently recover the released NIST
  reference metrics exactly. U/V/W and zero microstrain are fixed because the
  unconstrained GSAS-II solution drives those terms into nonphysical negative
  widths; those states are not accepted as a PhaseSmith parity target.
  The untouched 100b scan gives the same outcome: 18.777%/13.830% Rwp versus
  the 6.142% released NIST reference.
- `powgen-lab6-tof-calibration` pins the official GSAS-II POWGEN tutorial bank
  and calibration, converts 6,825 SLOG FXYE bin-boundary/integrated-intensity
  rows into 6,824 bin-center intensity densities in microseconds, and exercises
  the production TOF position law with Zero=4.41 µs, DIFC=22581.63 µs/Å, and
  DIFB=0. The legacy GSAS `ICONS` record is translated as DIFC, DIFA, Zero,
  unused; the mapping is checked against the pinned GSAS-II import rather than
  inferred from the four-column layout.
  The native reader returns a typed `TofPatternRecord`, preventing the CW
  reader's centidegree conversion from being silently applied. The complete
  fixed-instrument TOF Le Bail run generates 330 cubic LaB6 reflection
  families, returns all 15 shared derivative rows, extracts finite nonnegative
  intensities, jointly updates a 16-term Chebyshev residual above the fixed
  Smooth Bruckner baseline, reaches
  uncertainty-weighted Rwp 0.26192, and reaches background-subtracted profile
  correlation 0.96763. The public structural file composer independently
  checks the real 6,824-sample request, 90-degree geometry, 330 generated
  families, and complete source/reduction hashes. The staged native structural
  solve reaches Rwp 0.14415, correlation 0.97638, a=4.15792396 A, and
  x(B)=0.19954072 against the published POWGEN SRM-660b values.
  Position, centered finite-difference derivative, calibration-range,
  profile-improvement, coverage, fused-profile-derivative, and analytical
  background-basis gates all pass.
  The live pinned-oracle benchmark additionally matches all 329 GSAS-II
  reflection positions exactly, matches derived variance/rate terms to floating
  point precision, and reconstructs GSAS-II's peak-only Le Bail pattern from
  its extracted intensities with correlation above 0.99999 and relative L2
  error below 0.006. With both native workflows using a 16-term Chebyshev
  component on 6,824 centers, their uncertainty-weighted Rwp values are
  0.26192 (PhaseSmith) and about 0.26297 (GSAS-II), an absolute delta below
  0.0011; their profile-correlation delta is below 0.0005.
- `lanl-nickel-tof` also gates structural TOF transfer independently of
  POWGEN. The bounded calibration adapter reads each bank's explicit type-4
  vanadium incident spectrum; raw count densities and uncertainties are divided
  by that positive spectrum before fitting. One Fm-3m Ni phase shares its cubic
  cell and isotropic displacement across banks 2--4 while scale and Zero remain
  bank-local. The 13,293-point native result has Rwp 0.03278208, minimum bank
  correlation 0.99749010, a=3.52373113 A, and Uiso=0.00396507 A^2 versus
  3.5234 A. The isolated pinned GSAS-II worker creates temporary one-bank views
  so the public legacy loader cannot silently reuse the first RAW dataset. Its
  13,290-center result has Rwp 0.03367362, minimum correlation 0.99730908,
  a=3.52368699 A, and Uiso=0.00401813 A^2. The shared cell and Uiso differ by
  only 0.00004414 A and 0.00005306 A^2. Each implementation must clear its own
  Rwp/correlation gate; their Rwp delta is diagnostic because their background
  and optimizer contracts differ.
  The public one-bank file composer separately gates bank 2 with an explicit
  1101.6--8189.6 µs interval applied before incident normalization. It returns
  4,431 samples, 88.05-degree geometry, and 100 reflection centers inside the
  selected interval with complete source/range provenance. Combining public
  requests for banks 2--4 with the declared 0.2--3.0 Å range produces the same
  186-family shared topology used by the native structural acceptance and
  retains three ordered provenance records. The report's separate 102-family
  count belongs to its bank-2 Le Bail gate.
  The public end-to-end smoke gate estimates each Smooth Brückner background
  after incident normalization and declares the background domain explicitly.
  Its scale-only three-bank solve converges in four accepted steps to scales
  0.04029/0.04012/0.04481 and bank Rwp values 0.1163--0.1243. This does not
  replace the stricter native structural cell/Uiso/Zero gate.

  Run the two pinned multi-bank comparisons with an exact GSAS-II checkout:

  ```bash
  python benchmarks/compare_gsasii_nickel_tof_multibank.py \
    --gsas-python "$GSASII_PYTHON" --gsas-root "$PHASESMITH_GSASII_ROOT" \
    --binary-dir "$PHASESMITH_GSASII_BINARY_DIR"
  python benchmarks/compare_gsasii_nickel_tof_structural.py \
    --gsas-python "$GSASII_PYTHON" --gsas-root "$PHASESMITH_GSASII_ROOT" \
    --binary-dir "$PHASESMITH_GSASII_BINARY_DIR"
  ```
- `gsasii-pbso4-cw` adds official packed-GSAS X-ray and neutron patterns for
  the same PbSO4 specimen. The established Python validation keeps its staged
  per-probe comparison for continuity, while the Rust-only joint benchmark
  shares one structural parameter state across both probes. The paired pinned
  GSAS-II benchmark follows the official joint refinement stages.
  The supplied reference cell is a=8.480, b=5.398, c=6.958 Å. PhaseSmith's
  neutron result is a=8.47045, b=5.39170, c=6.95148 Å; its X-ray path currently
  keeps the supplied lattice fixed. These distinctions are recorded rather
  than treating a fixed starting model as an independently recovered result.
  A fixed Smooth Bruckner curve supplies the broad baseline. A three-term
  Chebyshev polynomial is initialized by weighted linear least squares against
  the starting structural profile, then refined to correct residual slope and
  curvature.
  The optional intelligent workflow reports cumulative scale/background,
  position, structure, and final-polish stages. It reaches 10.346% X-ray Rwp
  and 4.217% neutron Rwp with profile correlations above 0.995. The neutron
  position stage refines typed Debye--Scherrer X/Y displacement at a fixed
  650 mm radius; the report records the geometry and units.
- `iucr-sodium-dihydrogen-citrate-si-standard` is the preferred-orientation
  common-subset holdout. Its official IUCr pdCIF contributes 4,701 raw counts,
  a contiguous 4,452-point deposited range, both structures, the Cu doublet,
  and the deposited GSAS curve. The converter recomputes deposited
  `Rwp=0.08432525` and pins every source byte. The accepted comparison fixes
  the complete silicon-derived U/V/W/X/Y profile, equal-height `SH/L=0.0187`,
  calibrated displacement, and all size/strain terms; it refines phase scales,
  one residual constant, and March--Dollase (001). PhaseSmith/GSAS-II return
  `Rwp=0.18183/0.18945`, correlations `0.95589/0.94997`, Si fractions
  `0.22173/0.21651`, and March ratios `0.63851/0.63237`. All seven
  cross-implementation gates pass. The deposited 18.74 wt% Si and 8.433% Rwp
  remain richer-model references because spherical-harmonic orientation,
  Stephens anisotropy, and Suortti roughness are outside this common subset.
- `iucr-tripotassium-citrate-si-standard` is an independent laboratory
  transferability holdout. Its official IUCr pdCIF contributes 3,217 raw counts, the contiguous
  2,696-point deposited range, both structures, and the deposited legacy-GSAS
  profile. The converter recomputes `Rwp=0.04852875` and `Rp=0.03808683`,
  retains the unequal S/L=0.0168 and H/L=0.0200 axial terms, and discloses the
  phase-specific profiles, Stephens broadening, second-order texture, and
  Suortti roughness. The source identifies a silicon internal standard with
  `a=5.43105` Å but not a particular NIST SRM. On the fixed-geometry common
  subset, PhaseSmith and pinned GSAS-II give `Rwp=0.23965/0.23933`, profile
  correlations `0.61846/0.62079`, and Si fractions `0.04450/0.04343`; all four
  parity gates pass. Their separate 128-point Si-window fits also have nearly
  equal Rwp, but infer specimen displacements of `-0.02492/+0.00180 mm`.
  The millimetre values are conditional on the disclosed 141.5 mm PhaseSmith
  goniometer-radius assumption because the source does not deposit that radius.
  Because that `0.02672 mm` disagreement exceeds the predeclared
  identifiability threshold, the 1.44 wt% Si calibration is diagnostic and is
  not applied to either full-pattern fit. The result is a qualified parity pass
  and expected model/anchor failure, not reproduction of the deposited 4.853%
  Rwp or 1.44 wt% Si result.
- `iucr-trirubidium-citrate-si-standard` is the next independently converted
  laboratory holdout. Its official pdCIF contributes 4,701 raw counts and an
  exact contiguous 4,106-point mask after the deposited 5–17° beam-spillover
  exclusion. It contains both structures, 2.15 wt% NIST SRM 640b Si, a
  source-deposited 141.5 mm radius, equal S/L=H/L=0.0097, and no absorption or
  roughness correction. The converter recomputes `Rwp=0.02458338` and
  `Rp=0.01950460`. Both phases share the same isotropic base profile, while
  their mixing and Stephens terms remain explicitly source-only. A forensic
  correction maps legacy `LX=3.634` to the current Lorentzian size axis and
  `shft=-8.7503` to `-0.1080505 mm` in the current PhaseSmith/GSAS-II
  displacement convention at the deposited radius; the legacy manual's
  physical shift variable has the opposite sign. On that corrected common
  subset, PhaseSmith/pinned-GSAS-II give `Rwp=0.09579/0.09735`,
  correlations `0.94732/0.94248`, and Si fractions `0.02575/0.02548`; all
  full-pattern parity gates pass. The 128-point Si-window fits infer
  `-0.07519/-0.11031 mm`; their `0.03512 mm` disagreement keeps that calibration
  diagnostic and does not replace the direct source translation. The separate
  16-case oracle factorial records that the signed deposited transparency term
  is detrimental and Stephens negligible, while most remaining SSE lies below
  30 degrees. The legacy equation confirms the numerical sign and shows that
  positive `trns=1.30` has formally negative effective absorption; the manifest
  therefore labels its GSAS-II value as an equation field, not physical `1/mu`.
  The subsequent 643-sample 17–30° audit finds that no isolated background,
  intensity, position, symmetric-width, or axial-profile probe closes half of
  the local source-to-deposited weighted-SSE gap. Their combined best physical
  shape plus background closes 80.47% but remains at `Rwp=0.04977` versus the
  deposited `0.02711`. The pinned GSAS-II FCJ scan reaches its effective
  `SH/L=0.002` floor; this is recorded as a compression-fidelity diagnostic,
  not zero physical source divergence. The reviewed forensic artifact is
  `results/2026-08-13-citrate-rubidium-low-angle-forensics.json`.
  The converter also preserves all 1,197 source reflection rows in
  `source_reflections.csv`. The 600 unique calculated F² values provide a
  model-independent conversion check. On the 20 low-angle rubidium reflections,
  the deposited Cromer–Mann table plus converted structure gives 0.0289%
  source-weighted L1 error; PhaseSmith's production Waasmaier–Kirfel table gives
  0.301%. Thus intensity-table choice is visible but not large enough to explain
  the profile residual. The reviewed record is
  `results/2026-08-13-citrate-rubidium-source-reflection-fidelity.json`.
  The follow-up isolated axial-profile audit confirms that PhaseSmith preserves
  the deposited `S/L=H/L=0.0097` geometry exactly. Its source-native profile is
  bit-identical to its formal equal-height `SH/L=0.0194` mapping, but pinned
  GSAS-II at that documented sum differs by up to 17.53% normalized L1 and
  0.00847 degrees in centroid for three low-angle source reflections. A tested
  `SH/L=0.0097` is closer but does not represent the documented sum and is not
  adopted. The conversion is cleared; exact deposited-shape recovery requires
  a legacy two-parameter oracle. The reviewed record is
  `results/2026-08-13-citrate-rubidium-source-axial-profile-fidelity.json`.

The QARR checkpoint evaluates supplied CIF anisotropic displacement tensors
directly and keeps them fixed; sites without displacement values start from a
documented isotropic value and refine. It uses fixed Cu Kα1 dispersion offsets
for both doublet components and maps SH/L=0.002 to the documented equal-height
FCJ geometry. Absorption is not yet active. Those limitations are emitted in
the report rather than hidden.

The QARR case comes from the [IUCr quantitative phase analysis round
robin](https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm), with files
retrieved from a commit-pinned public mirror because the legacy IUCr asset URL
does not permit automated retrieval. The sucrose files come from the official
[GSAS-II tutorial repository](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/tree/e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data).
The PbSO4 patterns, instrument files, CIF, and staged recipe come from the
commit-pinned official [combined-refinement tutorial](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/e2485148a3d7ee4757239b1ba40653f1f715bba5/CWCombined/Combined%20refinement.htm).
The Echidna pattern is deposited in the
[ANSTO powder-diffraction Zenodo record](https://zenodo.org/records/14286343).
The sodium-citrate/Si holdout comes from the official
[IUCr article and supplementary pdCIF](https://journals.iucr.org/e/issues/2016/06/00/hb7585/index.html).
The tripotassium-citrate/Si holdout comes from its official
[IUCr article and supplementary pdCIF](https://journals.iucr.org/e/issues/2016/08/00/wm5301/index.html).
The anhydrous and monohydrate trirubidium-citrate/Si holdouts come from their
official IUCr articles and supplementary pdCIFs
([anhydrous](https://journals.iucr.org/e/issues/2017/02/00/vn2123/index.html),
[monohydrate](https://journals.iucr.org/e/issues/2017/02/00/hb7648/index.html)).
The LaB6 certification scans and reference fits come from the official
[NIST SRM 660c data release](https://data.nist.gov/od/id/mds2-2315).
The TOF bank and instrument parameters come from the commit-pinned official
[GSAS-II TOF calibration tutorial](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/tree/e2485148a3d7ee4757239b1ba40653f1f715bba5/TOF%20Calibration/data).

The original schema-1 result remains at
`validation/results/2026-08-07-baseline.json`. The reviewed schema-2 suite is
`validation/results/2026-08-10-baseline-v2.json`; it records expected-outcome
policy and deterministic scientific fingerprints for all eight cases. Elapsed
time is diagnostic host timing and is excluded from each fingerprint. Compare
a candidate without overwriting either file:

```bash
python tools/compare_validation_results.py \
  validation/results/2026-08-10-baseline-v2.json \
  validation/results/local.json
```

Run the Python-free joint workload directly through the native release binary:

```bash
cargo run --release -p phasesmith-workflows --example joint_pbso4 -- \
  validation/data/gsasii-pbso4-cw
```

The reviewed one-worker fingerprint selects 8,378 observations, decreases the
summed objective from `4.057058531851e5` to `3.888630080197e5`, reports joint
`Rwp=0.2793263122`, and installs the identical shared cell
`8.4795406429 × 5.3976687489 × 6.9575866497 Å` in both histogram states. Its
40 accepted iterations are an intentional fixed workload; `max_iterations` is
therefore an expected benchmark termination. Set
`PHASESMITH_BENCHMARK_THREADS` to a positive worker count when comparing native
execution policies. The executable gates sample count, finite `Rwp <= 0.30`,
objective decrease, shared-cell identity, and 0.5% proximity to the supplied
reference cell.

## Native real-data benchmark and regression tests

After building an optimized extension, benchmark all checksum-pinned native
workflows with cold/warm timing and deterministic scientific fingerprints:

```bash
uv run python benchmarks/real_data.py --require-release \
  --data-directory validation/data \
  --json-output validation/results/local-real-data-benchmark.json
```

QARR runs with one and two workers by default; repeat `--threads` to choose
other fixed worker counts, or pass `--threads 0` for automatic selection. Use
`--dataset` repeatedly to benchmark only selected datasets. Timing is never
part of the SHA-256 scientific fingerprint, and every warm-up and measured run
must reproduce the cold scientific record exactly.

This benchmark measures PhaseSmith configurations. Paired sucrose, Echidna,
QARR, and PbSO4 comparisons against the exact pinned GSAS-II revision provide
separate cross-implementation gates:

```bash
uv run python benchmarks/compare_gsasii_qarr.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --sample 1g \
  --data-directory validation/data/iucr-qarr-1g \
  --phasesmith-threads 2

uv run python benchmarks/compare_gsasii_nist_srm660c.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --specimen 100a \
  --data-directory validation/data/nist-srm660c-lab6-xray

uv run python benchmarks/compare_gsasii_real_lebail.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --case ansto-echidna-lab6-cw-neutron \
  --data-directory validation/data/ansto-echidna-lab6-cw-neutron

uv run python benchmarks/compare_gsasii_pbso4.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/gsasii-pbso4-cw

uv run python benchmarks/compare_gsasii_rowles_qpa.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/curtin-rowles-qpa-topas \
  --sample all

uv run python benchmarks/compare_gsasii_rowles_fpa.py \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/curtin-rowles-qpa-topas \
  --sample all
```

The NIST comparison runs both contracts by default: a matched SH/L=0.002 FCJ
parity case and an expected-failure SH/L=0.02 large-asymmetry stress case. Use
`--case matched-small-fcj` or `--case large-fcj-stress` to run either one alone.
The worker converts the pdCIF displacement from millimetres to GSAS-II's
micrometre sample-shift convention before refinement.

It rejects mismatched sample or reflection counts and requires differences no
larger than 0.02 in any phase fraction, 0.03 in Poisson-weighted Rwp, 0.02 in
unit-weight Rwp, and 0.01 in profile correlation. These deliberately broader
workflow tolerances acknowledge the documented non-matched background and
refinement parameterizations; they do not replace the tighter matched-kernel
oracle tests.

The PbSO4 comparison requires identical selected sample counts, bounds the Rwp
and profile-correlation deltas separately for X-ray and neutron, and requires
the independently refined PhaseSmith neutron cell to remain within 0.2% of the
joint GSAS-II cell. Reflection counts are reported but not gated because the
two APIs expose different family/list conventions. The JSON report explicitly
labels PhaseSmith's independent fits, GSAS-II's joint fit, and the supplied
reference cell. Probe-specific Rwp deltas are limited to at most 0.006 and
profile-correlation deltas to at most 0.002.
The live oracle test job runs both Le Bail cases, both QARR mixtures, and
PbSO4. The matched QARR parity workflow passes both 1g and the untouched 1h
holdout with maximum phase-fraction deltas of 0.00355 and 0.00415 and Poisson
Rwp deltas of 0.00169 and 0.00278. The historical acceptance workflow and its
reviewed 1h failure remain intact as a separate transfer diagnostic. NIST 660c
is checked against its certification release as the primary oracle.
The JSON also records both implementations' Debye--Scherrer radius and X/Y
values. They are not equality-gated because PhaseSmith currently fits the
neutron histogram independently while GSAS-II shares one structure between
the X-ray and neutron histograms. A PhaseSmith joint comparison awaits the
first-class multi-histogram objective described in `docs/rietveld.md`.

The complete real-pattern regression tests are deliberately excluded from the
normal unit-test command because the inputs are external and the workflows are
comparatively slow. Run them explicitly after fetching or verifying the data:

```bash
uv run pytest -m real_data
```

On the 2026-08-08 Apple Silicon development host, one discarded warm-up and
three measured release runs gave the following complete-workflow driver-wall
times. These are observations, not cross-machine acceptance thresholds:

| Dataset and configuration | Cold (ms) | Warm median (ms) | Warm p95 (ms) |
| --- | ---: | ---: | ---: |
| APS sucrose | 532.6 | 520.5 | 524.2 |
| QARR 1g, one worker | 1238.5 | 1234.3 | 1239.7 |
| QARR 1g, two workers | 935.4 | 936.5 | 938.4 |
| PbSO4 X-ray, intelligent recipe, one worker | 5848.4 | 5827.2 | 5829.0 |
| PbSO4 neutron, intelligent recipe, one worker | 2353.4 | 2359.5 | 2365.8 |

The two QARR configurations produced the same scientific fingerprint,
`a0afa830220b800bfaf71985efcf71385fc88ef221c9191c74470b67805f061e`;
the observed two-worker median speedup was 1.32x. The sucrose fingerprint was
`c20baeac2a814eb82b4147ef7ccaa796e7129a80c2e1c8896d751aa2f1cb207b`.
The staged PbSO4 fingerprints were
`f2025354f99aee4219af511e5fba188bd2fe6e40c1399988bb246314d2bce6fd`
for X-ray and
`f22e00b4ec1fbe99c28a5362dbb90c60ad26bae3c84bbcbcebbba8649ccd4423`
for neutron; all three measured repetitions reproduced them exactly.
