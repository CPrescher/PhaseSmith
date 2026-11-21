# PhaseSmith versus XRD-Rust

This benchmark records a direct public-API speed comparison with XRD-Rust,
the implementation described in the 2026 IUCr Journal of Applied
Crystallography [paper](https://journals.iucr.org/j/issues/2026/04/00/hat5023/index.html).
The reproducible driver is `benchmarks/compare_xrd_rust.py`; the complete
timings and provenance from the reviewed run are retained in the source tree at
`validation/results/2026-08-13-xrd-rust-performance.json`.

## Result

The 2026-08-13 release-wheel run used an Apple M4 Pro with 14 CPU cores and
48 GB RAM on macOS 26.5.2. The software environment was Python 3.13.5,
PhaseSmith 0.3.0, XRD-Rust 0.3.5, pymatgen 2026.5.4, and NumPy 2.5.2.
Times are medians in milliseconds; lower is better.

| Case | Atoms | PhaseSmith families | PhaseSmith 1 thread | XRD-Rust 1 thread SIMD | PhaseSmith 1t vs XRD-Rust 1t | XRD-Rust 8-thread SIMD | PhaseSmith 1t vs XRD-Rust 8t | PhaseSmith 2 threads | PhaseSmith 2t vs XRD-Rust 8t | PhaseSmith threading speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 8 | 7,673 | 5.360 | 24.659 | 4.60x | 19.602 | 3.66x | 4.863 | 4.03x | 1.10x |
| Medium | 64 | 39,790 | 63.167 | 335.342 | 5.31x | 142.351 | 2.25x | 51.428 | 2.77x | 1.23x |
| Large | 256 | 114,320 | 512.380 | 2,902.616 | 5.66x | 749.487 | 1.46x | 388.279 | 1.93x | 1.32x |

The single-thread comparison puts both implementations on the same worker
budget: PhaseSmith completes these cases 4.60–5.66 times faster than XRD-Rust's
serial SIMD path. Even PhaseSmith's single-thread path is 1.46–3.66 times faster
than XRD-Rust configured for eight threads and SIMD. Giving PhaseSmith's public
values-only structure-factor path a reusable two-thread `ExecutionPolicy`
improves its complete calculation by another 10–32%; that two-thread path is
1.93–4.03 times faster than XRD-Rust's eight-thread SIMD path. These are
workload- and machine-specific measurements, not a claim about every structure
or platform.

## What is timed

Each call starts from an already constructed, deterministic P1 triclinic
structure and includes:

- reciprocal reflection generation over 2–60 degrees two theta;
- non-resonant X-ray structure factors for Mo K-alpha, 0.71073 angstrom;
- multiplicity and unpolarized Bragg-Brentano Lorentz-polarization correction;
- construction of the public stick-pattern output.

CIF parsing, sampled peak-profile evaluation, derivatives, plotting, and
serialization are excluded. Structure construction and calculator preparation
are outside the timed region. The implementations return different public
objects, so the benchmark compares equivalent scientific work rather than an
identical allocation and formatting contract.

PhaseSmith is measured with one and two threads. XRD-Rust is measured in its
public serial-SIMD and eight-thread-SIMD modes. Calls are interleaved with a
rotating order after warmup to reduce systematic ordering effects. Explicit
PhaseSmith policies are created once per case so the native worker pools are
reused rather than rebuilt for every repetition.

## Numerical gate

Timing is reported only after the driver verifies the scientific outputs.
Every XRD-Rust peak matched a PhaseSmith reflection within the configured
1e-5-degree gate; the observed maximum error was at floating-point roundoff.
The normalized intensity correlation remained above the 0.999 gate.

| Case | XRD-Rust peaks | Matched peaks | Maximum position error (degrees) | Normalized intensity correlation |
| --- | ---: | ---: | ---: | ---: |
| Small | 7,631 | 7,631 | 4.26e-14 | 0.9999992 |
| Medium | 39,216 | 39,216 | 3.55e-14 | 0.9999885 |
| Large | 110,406 | 110,406 | 4.26e-14 | 0.9997993 |

XRD-Rust reports fewer sticks because its public calculator filters very weak
intensities. PhaseSmith retains those families. For XRD-Rust 0.3.5, the public
path also evaluates Friedel-related reciprocal points before merging its final
sticks, while PhaseSmith performs symmetry/Friedel family reduction before the
structure-factor accumulation. That difference is part of each library's
public end-to-end design and helps explain why raw inner-kernel throughput alone
does not predict this table.

## Reproduce it

Install XRD-Rust and pymatgen only in a benchmark environment; neither is a
PhaseSmith runtime dependency. Build the current PhaseSmith checkout in release
mode, then run:

```shell
uv pip install 'xrd-rust==0.3.5' 'pymatgen==2026.5.4'
maturin develop --release --uv
uv run python benchmarks/compare_xrd_rust.py --require-release \
  --json-output validation/results/local-xrd-rust-performance.json
```

Use `--case small`, `--repetitions`, `--warmups`,
`--phasesmith-threads`, and `--xrd-rust-threads` for controlled scaling
experiments. The driver records all individual samples, package versions,
platform information, calculation scope, thread counts, numerical checks, and
derived ratios in JSON. It exits without writing a timing report if numerical
agreement fails.

## Corrected inorganic COD collection, November 2026

A paired release-wheel run on an Apple M4 Pro compared PhaseSmith 0.8.5 at
`98bd223`, the cubic reflection-generation optimization, and XRD-Rust 0.3.5.
All 5,552 files were attempted. Both PhaseSmith builds produced bitwise-identical
reflection lists, d-spacing derivatives, and powder sticks on all 5,319 computed
structures at both one and eight threads. The original peak-position and intensity
agreement criteria admitted 5,232 structures, including 892 cubic structures.

| Eight threads, qualifying cubic structures | Previous PhaseSmith | Optimized PhaseSmith | XRD-Rust |
|---|---:|---:|---:|
| Median powder calculation (ms) | 16.675 | 2.159 | 7.989 |
| Cubic files faster than XRD-Rust | 186 / 892 | 885 / 892 | — |

Median paired speedup over previous PhaseSmith was 7.35× on cubic structures
and 1.0005×–1.0070× across the six other crystal-system medians. Optimized
PhaseSmith was 2.92× faster than XRD-Rust on the median cubic pair. Each
ratio was calculated per file before taking the median. Summing one
median-timed calculation per qualifying file across all seven systems gave
a 1.70× speedup over previous PhaseSmith at eight threads.

Profiling 32 cubic entries attributed a median 96.9% of the previous
eight-thread time to serial reflection generation. Cubic point groups map
many candidate indices to one family. Discarding noncanonical indices before
building their full orbits reduced median generation time on those entries
from 16.67 to 2.73 ms. The exact representative and extinction rules are
described in [symmetry and reflection generation](symmetry-reflections.md).

Each library received the same physical model with neutral atoms and zero
displacements. Timings covered Mo Kα powder sticks over 2–60° in 2θ from
prepared structures. The release builds took turns on each file with alternating
order, one warm-up and three timed repetitions per mode. The native
`reflection_generation` benchmark includes an Fm-3m case and a C-centred
orthorhombic control for repeating the generator measurements.

### Extending the early symmetry check to all crystal systems

A second paired release-wheel run used the cubic-optimized commit `6acbffc`
as its baseline and repeated the full corrected collection on the same M4 Pro.
The new generator checks equivalent indices early whenever the rotation and
search-box bounds guarantee representable indices. Both builds used the same
common inputs, scan limits, alternating order and three timed repetitions.

| Crystal system | Qualifying structures | Median speedup over cubic-only build, 1 thread | Median speedup over cubic-only build, 8 threads |
|---|---:|---:|---:|
| Triclinic | 306 | 1.09× | 1.13× |
| Monoclinic | 1,228 | 1.36× | 1.46× |
| Orthorhombic | 1,286 | 1.88× | 2.03× |
| Tetragonal | 583 | 2.85× | 2.80× |
| Trigonal | 484 | 2.34× | 2.35× |
| Hexagonal | 453 | 3.23× | 3.13× |
| Cubic | 892 | 1.00× | 1.00× |
| All | 5,232 | 1.65× | 1.74× |

The paired eight-thread median over XRD-Rust 0.3.5 rose from 3.15× to 5.52×.
Across all 5,552 CIFs, both builds produced identical per-file outcomes. All
5,319 computed structures had identical reflection lists, d-spacing
derivatives, integrated intensities and powder sticks at each PhaseSmith thread
setting. The same 87 timed pairs missed the cross-library agreement criteria.
