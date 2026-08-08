# External real-data validation

PhaseSmith does not bundle tutorial patterns and does not import or execute
GSAS-II. The command below explicitly downloads files from commit-pinned HTTPS
locations, verifies their byte sizes and SHA-256 hashes, and runs only
PhaseSmith code:

```bash
python tools/validate_real_data.py --fetch \
  --data-directory validation/data \
  --output validation/results/local.json
```

The QARR stages emit structured progress to standard error. In an interactive
terminal, press `q` or Ctrl+C once to request a graceful stop at the next safe
batch boundary; a second Ctrl+C forces interruption. A cooperative stop returns
a machine-readable `blocked` report with the last accepted Rwp and a nonzero
CLI exit status.

`validation/data/` and local result files are ignored. The current cases
have deliberately different meanings:

- `aps-sucrose-11bmb` exercises the supported monochromatic FXYE → background
  → symmetry/reflection generation → Le Bail → analytical profile-refinement
  path on the official APS 11-BM sucrose tutorial pattern. Its `Rwp <= 0.22`
  threshold is a regression smoke gate for the present model, not a claim of
  numerical equivalence with GSAS-II. The tutorial's lower final residual uses
  additional staged background-peak, crystallite-size, microstrain, lattice,
  and repeated extraction refinements.
- `iucr-qarr-1g` verifies the 7,251-point 5–150° input and its explicit Cu Kα1/
  Kα2 instrument metadata, runs a native three-phase fixed-spectrum structural
  refinement, and converts the final polished scales with the Hill--Howard
  relation. The reviewed baseline returns Al2O3 33.250%, ZnO 32.936%, and CaF2
  33.814% against weighed targets of 31.37%, 34.21%, and 34.42%. Its largest
  absolute error is 1.881 weight-percentage points. The profile gates distinguish
  Poisson-weighted Rwp (0.19679, limit 0.20) from unit-weight Rwp (0.13282,
  limit 0.15); profile correlation is 0.99069.
- `gsasii-pbso4-cw` adds official packed-GSAS X-ray and neutron patterns for
  the same PbSO4 specimen. PhaseSmith runs the probes independently, while the
  paired pinned-GSAS-II benchmark follows the official joint refinement stages.
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

The QARR checkpoint explicitly approximates anisotropic displacement with
trace-mean isotropic values, uses fixed Cu Kα1 dispersion offsets for both
doublet components, and does not yet apply the supplied SH/L=0.002 FCJ
asymmetry or absorption. Those limitations are emitted in the report rather
than hidden.

The QARR case comes from the [IUCr quantitative phase analysis round
robin](https://www.iucr.org/__data/iucr/powder/QARR/data-kit.htm), with files
retrieved from a commit-pinned public mirror because the legacy IUCr asset URL
does not permit automated retrieval. The sucrose files come from the official
[GSAS-II tutorial repository](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/tree/e2485148a3d7ee4757239b1ba40653f1f715bba5/LeBail/data).
The PbSO4 patterns, instrument files, CIF, and staged recipe come from the
commit-pinned official [combined-refinement tutorial](https://github.com/AdvancedPhotonSource/GSAS-II-Tutorials/blob/e2485148a3d7ee4757239b1ba40653f1f715bba5/CWCombined/Combined%20refinement.htm).

The committed `validation/results/2026-08-07-baseline.json` records the first
reviewed run. Elapsed time is diagnostic host timing, not a cross-machine
performance acceptance threshold.

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

This benchmark measures PhaseSmith configurations. Paired QARR and PbSO4
comparisons against the exact pinned GSAS-II revision provide separate
cross-implementation gates:

```bash
uv run python benchmarks/compare_gsasii_qarr.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/iucr-qarr-1g \
  --phasesmith-threads 2

uv run python benchmarks/compare_gsasii_pbso4.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --data-directory validation/data/gsasii-pbso4-cw
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
