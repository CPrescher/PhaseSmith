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

The canonical suite uses native runners. Add `--interactive-qarr` to run QARR
1g through the independent Python path with structured progress and cooperative
cancellation: press `q` or Ctrl+C once to stop at the next safe batch boundary;
a second Ctrl+C forces interruption.

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
  for phase scales and background. TOPAS-only source and optics terms remain
  excluded and are listed in `docs/topas-rowles-model.md`.
- `nist-srm660c-lab6-xray` validates all 20 pdCIF specimens in NIST's official
  SRM 660c archive: 106,640 measured points, aligned released calculated
  profiles, the certified lattice interval, aggregate correlation 0.99933, and
  weighted Rwp 0.06083. Every specimen also clears correlation 0.999 and Rwp
  0.065 gates. This validates the external oracle archive. It is not a
  pointwise PhaseSmith equivalence claim because the released fit uses a Cu
  emission spectrum and fundamental-parameters optics model.
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
  uncertainty-weighted Rwp 0.26461, and reaches background-subtracted profile
  correlation 0.96729.
  Position, centered finite-difference derivative, calibration-range,
  profile-improvement, coverage, fused-profile-derivative, and analytical
  background-basis gates all pass.
  The live pinned-oracle benchmark additionally matches all 329 GSAS-II
  reflection positions exactly, matches derived variance/rate terms to floating
  point precision, and reconstructs GSAS-II's peak-only Le Bail pattern from
  its extracted intensities with correlation above 0.99999 and relative L2
  error below 0.006. With both native workflows using a 16-term Chebyshev
  component on 6,824 centers, their uncertainty-weighted Rwp values are
  0.26461 (PhaseSmith) and about 0.26297 (GSAS-II), an absolute delta below
  0.0017; their profile-correlation delta is below 0.0005.
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
```

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
PbSO4. QARR 1h remains a reviewed PhaseSmith failure while still being compared
numerically with GSAS-II; its weight fractions remain within 0.00979, but its
Poisson Rwp, unit-weight Rwp, and correlation deltas fail at 0.09434, 0.11009,
and 0.01297. The test requires that explicit reviewed failure, so expected
failure never means “skip the oracle.” NIST 660c is checked against its
certification release as the primary oracle.
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
