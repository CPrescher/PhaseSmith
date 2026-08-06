# GSAS-II performance comparison

`benchmarks/compare_gsasii.py` provides a reproducible speed comparison between
Rietveld Engine and the exact GSAS-II revision recorded in
`oracle/PINNED_GSASII.json`. GSAS-II remains a separately installed external
validation oracle and is never imported by the normal package.

## Compared workload

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
```

The equivalent `GSASII_PYTHON`, `RIETVELD_GSASII_ROOT`, and
`RIETVELD_GSASII_BINARY_DIR` environment variables are supported. The external
worker refuses any GSAS-II checkout other than revision
`c0bc79b259cdf0065480b5fbd57674ddf12c4a23`.

Before printing a speed ratio, the runner requires identical sparse support
indices and checks profile, local derivative, and global derivative arrays
against the pinned implementation. Each normalized maximum error must remain
below the existing CW oracle tolerance of `6e-6`. The JSON report records raw
timings, median, p95, software versions, platform, workload dimensions, active
sample count, numerical errors, and the ratio
`GSAS-II median / Rietveld Engine median`. A ratio above one means Rietveld
Engine was faster for this specific workload on that machine.

The self-hosted pinned-oracle workflow runs the same command and uploads both
the text output and machine-readable JSON report. Results should be compared on
the same otherwise-idle host; VM type, CPU power policy, Python/NumPy versions,
and build mode are part of the result, not noise to omit.
