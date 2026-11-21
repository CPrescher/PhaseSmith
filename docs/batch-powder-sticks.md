# Batch X-ray powder sticks

PhaseSmith 0.9.0 adds `calculate_powder_sticks_batch` for calculating discrete monochromatic X-ray patterns from a collection of structures. Pass an iterable of `CrystalStructure` objects and a common scan. Existing scripts can adopt it in place of a Python loop over structures.

```python
from pathlib import Path

from phasesmith import CwTwoThetaRange, calculate_powder_sticks_batch
from phasesmith.io.cif import read_cif

paths = sorted(Path("structures").glob("*.cif"))
structures = [read_cif(path, strict=True).structure for path in paths]
scan = CwTwoThetaRange(min_deg=2.0, max_deg=60.0, wavelength_angstrom=0.71073)
patterns = calculate_powder_sticks_batch(structures, scan)

for path, pattern in zip(paths, patterns, strict=True):
    print(path.name, pattern.structure_id, len(pattern.two_theta_deg))
```

The returned tuple follows input order, including repeated structures. Each `PowderSticks` contains the structure ID and two one-dimensional NumPy arrays of equal length, `two_theta_deg` and `integrated_intensity`. An empty batch yields an empty tuple. A scan with no reflection families yields two empty arrays for that structure. A cell without atom sites raises an error.

## What the sticks represent

The calculation uses the supplied space-group setting, sites, scattering species, occupancies and displacement parameters. It generates symmetry-allowed reflection families with Friedel mates merged, evaluates the non-resonant X-ray structure factors and applies the unpolarized Bragg–Brentano Lorentz–polarization correction. For each family the position is \(2\arcsin[\lambda/(2d)]\) in degrees and the unscaled intensity is \(m |F|^2 LP\), where \(m\) is its multiplicity. The scan includes both endpoints and requires \(0 \leq 2\theta_\min \leq 2\theta_\max < 180^\circ\) and a positive finite wavelength in ångströms.

Sticks are sorted by angle. Reflections less than \(10^{-5}\) degrees from the lowest angle in a group contribute to the same stick. A reflection exactly at that separation starts a new stick. The reported position is the lowest angle and the intensity is the sum. For a sampled, broadened diffraction profile use the existing pattern-calculation API. [Reflection and symmetry conventions](symmetry-reflections.md)

## Scheduling independent structures

The default runs a single structure inline. For two or more it uses up to four Python threads, capped by batch size and available CPUs. Each structure uses one native calculation thread. Threads share input structures in memory. Set `workers=1` when your application already schedules structures, or set an explicit worker count to tune for your machine.

`backend="processes"` starts a fresh pool for one call. Structures are transferred as portable records and the calculated arrays return to the parent. For scripts using this backend, put the call inside `if __name__ == "__main__":` in an importable file. All workers finish or shut down before the call returns. The earlier 5,232-pattern test measured persistent workers with models already resident in memory. [Full-collection comparison](xrd-rust-performance.md#one-pattern-or-many-patterns-in-parallel)

## End-to-end timings for small batches

Three disjoint, seeded panels of 256 qualifying inorganic COD structures were drawn from the corrected collection. Each smaller batch used a prefix of its panel. Three repetitions per panel started from in-memory structures in an already-imported Python session. Worker creation, input submission and transfer, preparation of reflection generators, calculation, return of both arrays and pool shutdown were timed on every call. CIF reading and output verification were outside the timer. Process workers used the cross-platform spawn method. Runs used the neutral-atom, zero-displacement model from the full comparison, a PhaseSmith 0.9.0 release build, an Apple M4 Pro and Python 3.13.7.

The table gives milliseconds per complete API call. Each entry is the median of the three panel medians. Worker configurations took turns in seeded random order. A dash means the batch had fewer structures than the worker count.

| Structures | Inline | 2 threads | 4 threads | 8 threads | 2 processes | 4 processes | 8 processes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.17 | — | — | — | — | — | — |
| 2 | 3.83 | 2.37 | — | — | 169.44 | — | — |
| 4 | 6.24 | 4.54 | 4.01 | — | 162.62 | 190.30 | — |
| 8 | 17.13 | 11.14 | 9.07 | 10.42 | 179.40 | 217.85 | 229.47 |
| 16 | 40.62 | 30.40 | 27.10 | 29.33 | 229.52 | 210.26 | 272.17 |
| 32 | 168.04 | 91.79 | 61.55 | 72.94 | 272.39 | 297.73 | 330.08 |
| 64 | 320.40 | 186.08 | 152.15 | 159.53 | 412.47 | 397.07 | 425.77 |
| 256 | 993.08 | 583.06 | 384.28 | 352.23 | 883.46 | 590.38 | 516.99 |

Four threads gave the lowest median at 4, 8, 16, 32 and 64 structures. Eight threads led at 256 structures (352 ms versus 384 ms). That modest gain on the largest batch and the cost of starting fresh Python processes motivated the four-thread default. The pool uses fewer workers for smaller batches.

### Independent check of the default

Three further panels contributed 768 new structures. The same timing procedure compared the automatic setting with inline calculation. Every timed result in both experiments matched the serial reference bit for bit in position and intensity, in input order.

| Structures | Inline (ms) | Automatic (ms) |
|---:|---:|---:|
| 1 | 5.58 | 5.99 |
| 2 | 13.13 | 13.25 |
| 4 | 14.95 | 13.47 |
| 8 | 20.23 | 13.81 |
| 16 | 40.99 | 29.79 |
| 32 | 123.65 | 63.67 |
| 64 | 341.03 | 157.63 |
| 256 | 905.88 | 418.93 |

On these panels the two-structure batch was effectively tied. At 64 and 256 structures, the median per-panel speedups were 2.16× and 2.31× respectively. Other machines and structure sizes can shift these timings.

[Scheduling measurements and COD IDs](downloads/batch-powder-sticks-m4-pro.json) · [Independent default measurements and COD IDs](downloads/batch-powder-sticks-default-m4-pro.json)

To reproduce both runs, use the checksum-matched CIFs from the corrected collection and run from the source checkout with a release build installed. Set the numerical-library thread limits shown in the receipts before launching Python.

```sh
python benchmarks/batch_powder_sticks.py /path/to/cifs timings.json
python benchmarks/batch_powder_sticks.py /path/to/cifs holdout.json --check-default
```
