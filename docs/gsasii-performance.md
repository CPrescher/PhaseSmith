# GSAS-II performance comparison

`benchmarks/compare_gsasii.py` and
`benchmarks/compare_gsasii_structural.py` provide reproducible speed
comparisons between Rietveld Engine and the exact GSAS-II revision recorded in
`oracle/PINNED_GSASII.json`. GSAS-II remains a separately installed external
validation oracle and is never imported by the normal package.

## Profile-only workload

The default case contains 200 monochromatic constant-wavelength reflections on
a sorted 5,001-sample grid. Both implementations return all of the following:

- the accumulated symmetric CW profile;
- support-block local derivatives for integrated intensity and position;
- dense shared derivatives for U, V, W, X, and Y.

Each timed operation includes reflection-dependent width calculation, TCH
support calculation, binary support lookup, output allocation, profile and
analytical derivative evaluation, and accumulation. Warmup calls, interpreter
startup, process creation, and NPZ transfer are outside the timed region.
Rietveld Engine uses one fused native batch call. The pinned GSAS-II worker uses
one `GSASIIpwd.getdPsVoigt` call per active reflection and performs batch
orchestration in Python, which is the interface GSAS-II exposes for this
profile-level comparison.

This scope is intentionally narrower than a complete powder calculation or
refinement. It does not compare structure factors, reflection generation,
background models, constraints, optimizers, project-file I/O, or GUI work. The
reported ratio must therefore be described as profile-and-derivative
throughput, not as total Rietveld-refinement speed.

## Structural-intensity workload

The separate structural benchmark uses a synthetic 32-site P1 crystal with C,
O, Si, and Fe atoms, 256 GSAS-II-generated powder families, and monochromatic
neutron radiation. It reports two independently timed cases:

- structure-factor values through integrated reflection intensities;
- the same structural calculation composed with a 20,001-sample symmetric CW
  profile and analytical intensity, position, U, V, W, X, and Y derivatives.

Both programs receive the same atom coordinates, occupancies, isotropic
displacements, cell, HKLs, multiplicities, wavelength, instrument parameters,
finite-support convention, and sample grid. The GSAS-II project, reflection
generation, dictionary preparation, process startup, and NPZ transfer are done
once outside the timed region. The Rietveld Engine phase and prepared pattern
are likewise constructed outside the timed region. The first Rietveld timing
uses the public values-only structure-factor API; the second uses the fused
native `PreparedStructuralPattern` path.

The structural comparison deliberately uses neutron scattering and a neutral
integrated-intensity correction. GSAS-II tabulates coherent scattering lengths
in `10^-12 cm`, while Rietveld Engine's independently sourced table uses fm;
the adapter therefore applies the exact squared-unit conversion factor of 100.
Because the tabulations are independently sourced and rounded, normalized
maximum errors for F-squared and intensity must remain below `1.25e-4`, not
machine precision. Reflection geometry is checked at `2e-14`; the unweighted
profile kernel derivative remains gated at `6e-6`; intensity-weighted profile
and derivative arrays include the table difference and are gated at `1.5e-4`.

This is a controlled structure-to-profile throughput comparison, not a full
Rietveld refinement. It excludes background evaluation, constraints,
optimization, project I/O, and GUI work. It also does not time reflection-list
generation because the list is prepared input to both structural kernels.

## Run it

Build Rietveld Engine in release mode, then use GSAS-II's separate interpreter
and compatible binary directory:

```shell
maturin develop --release --uv
uv run python benchmarks/compare_gsasii.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --json-output gsasii-speed-comparison.json

uv run python benchmarks/compare_gsasii_structural.py --require-release \
  --gsas-python /path/to/gsas/python \
  --gsas-root /path/to/pinned/GSAS-II \
  --binary-dir /path/to/compatible/GSASII-bin/platform-directory \
  --json-output gsasii-structural-speed-comparison.json
```

The equivalent `GSASII_PYTHON`, `RIETVELD_GSASII_ROOT`, and
`RIETVELD_GSASII_BINARY_DIR` environment variables are supported. The external
worker refuses any GSAS-II checkout other than revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`.

Before printing a speed ratio, each runner performs its documented numerical
checks. JSON reports record raw timings, median, p95, software versions,
platform, workload dimensions, numerical errors, and the ratio `GSAS-II median
/ Rietveld Engine median`. A ratio above one means Rietveld Engine was faster
for that specific workload on that machine.

The self-hosted pinned-oracle workflow runs both commands and uploads text and
machine-readable JSON reports. Results should be compared on the same
otherwise-idle host; VM type, CPU power policy, Python/NumPy versions, and build
mode are part of the result, not noise to omit.

## Reference structural run

On 2026-08-06, a 100-repetition release run on the development Apple Silicon
macOS host produced:

| Timed case | Rietveld median | GSAS-II median | GSAS-II / Rietveld |
| --- | ---: | ---: | ---: |
| Structure-factor values and intensities | 0.292 ms | 0.731 ms | 2.508x |
| Structure through profile and CW derivatives | 0.633 ms | 5.211 ms | 8.231x |

The corresponding p95 values were 0.308/0.747 ms and 0.649/5.285 ms,
respectively. The normalized maximum errors were `4.683e-5` for complex F,
`9.134e-5` for both F-squared and integrated intensity, `9.222e-5` for the
profile, and at most `9.385e-5` across the reported derivative groups. These
numbers characterize this workload and host; the committed runner and JSON
schema, not the observed ratio, are the durable performance contract.
