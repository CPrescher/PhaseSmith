# Batched FFTs with blocked line staging

The experimental `batched_fft=False` default retains the existing Fourier path.
The true option requires reusable Fourier workspaces. It applies to P1 phasing
and the supplied-symmetry recycling used by phase-origin completion. It retains
the same complex f64 RustFFT plans, z/y/x axis order, normalization, grid limits,
observations, reciprocal support, seeds, iteration equations and model validation.
No additional dependency or numerical thread is introduced.

## Layout and transform conventions

For C-order XYZ input with dimensions Nx,Ny,Nz, the forward transform is

    F(h,k,l) = sum_xyz rho(x,y,z) exp(-2*pi*i*(h*x/Nx+k*y/Ny+l*z/Nz)).

The inverse uses the positive exponent. Both directions remain unnormalized;
an inverse/forward round trip gives N=Nx*Ny*Nz times the input. Physical density
normalization remains in the existing callers. All original complex coefficients,
including measured, free, fixed-zero and Nyquist locations, retain their layout
and meaning at the boundary.

The old execution path calls the planned one-dimensional FFT on each contiguous
z line, then gathers/scatters each y and x line separately. The final new path
batches all contiguous z lines. For y and x, it packs at most eight adjacent
strided lines into a small lane-by-coordinate array, invokes the same FFT plan
on the contiguous batch, then scatters those lines back to their original XYZ
locations before processing the next group. Axes still execute in z/y/x order.

For axis length L, stride S, group offset O and lane l, the permutation is

    batch[l*L+r] = values[O+r*S+l],

where r is the coordinate along a line. The inverse copy restores exactly that
location. The y stride is Nz, with one outer block per x; the x stride is Ny*Nz.
Packing/scattering uses eight-row tiles and adjacent lanes so the source/destination
values for a grid row share cache lines. Partial groups and row tiles remain
bounded by their actual sizes. Copies perform index arithmetic only, with no
floating-point arithmetic on the values. The backend receives full contiguous
FFT batches in the original independent-line order.

The extra staging buffer is allocated once per active transform plan and reused.
It holds at most 8*max(Nx,Ny,Nz) complex f64 values, at most 16 KiB under the
existing 128-point axis limit. It does not scale with the full grid volume.
Original scratch and scalar-line buffers retain their conventions. No grid,
atom or Jacobian cap increases. The buffer is privately owned by the FFT plan,
with no shared mutable cache. Even axes in [8,128], at most 2,097,152 voxels,
non-cubic grids and non-power-of-two shapes retain the original boundary.

Bitwise equivalence is checked rather than assumed: FFT backends can in general
choose different arithmetic for batches. On this installed backend the direct
complex transforms, complete phase trajectories and synthetic complete models
match the original path bitwise. Independent NumPy forward/inverse transforms,
round trips, complex and real inputs, impulses, constants, extreme finite scales
and invalid boundaries provide separate numerical checks. Equations and fused
value/analytical derivative kernels are unchanged; their existing independent
tests continue to apply. The GSAS-II powder oracle does not cover this FFT stage.
`batched_phase_transforms` records actual batched 3D calls inside P1 phasing;
recycling also uses the layout but is not included in that P1 counter.

## Benchmarks and comparison protocol

The realistic fixed-work benchmark compares complete 64-cycle P1 phasing on a
32-cubed map with eight atoms and 2,108 measured rows. A complete eight-atom search
uses two starts and 128 requested cycles with a generous 60-second backstop.
Both modes already enable fast scalar scoring, reusable Fourier workspaces,
filtered peak search, work-sliced phasing and competing isotropic assignments.
Only the batch/layout execution option differs. Raw samples, confidence intervals,
source and executable hashes are retained separately from corpus timings.

The complete matched experiment uses all 35 development cases, seed zero, four
starts, 800 requested cycles, native 13-second/outer 15-second caps and one
numerical thread. Final anisotropic fitting still uses all original prepared
observations. Counts, element types and coordinates are never taken from
references by a solver worker. Original input-derived resource caps and exact
non-H per-element counts with RMS <=0.025/max <=0.05 angstrom remain the scoring
gate; strict 0.02-angstrom recovery is secondary. Seven references above native
capacity remain in every denominator.

The balanced serial two-mode scheduler alternates the leading mode by case.
All 140 predictions precede both reference barriers. Both panels use the same
native build and retain full source snapshots, per-job load samples, models,
failures and actual phase/fit effort. The adapter-only revision-18 seal preserves
registry/source data and splits. No task-owned tests, builds or microbenchmarks
overlap timed collection. The 65 holdout solver outcomes remain unopened.

## Preserved initial layout experiment

The first implementation rotated whole-grid XYZ/ZXY/YZX buffers through three
16-by-16 blocked matrix transposes and a final copy. It was numerically correct
and passed all fixed-work equivalence tests, but the staging cost outweighed
batching savings in the realistic kernel benchmark:

| Initial design median | Control | Whole-grid batching |
| --- | ---: | ---: |
| Complete 64-cycle phasing | 42.796 ms | 65.021 ms |
| Eight-atom complete search | 93.669 ms | 113.769 ms |

No 35-case collection was run with that slower prototype. Its raw samples and
source/executable hashes remain in
`benchmarks/structure-solution-batched-fft-prototype-kernels-macos.json`. Its
complete implementation, adapter, independent reference and tests are preserved
as `benchmarks/structure-solution-batched-fft-prototype.patch`, verified to apply
against starting commit `f871aea` using an isolated temporary Git index. The
final bounded-slab design replaces the whole-grid workspace and copies; its
benchmark and corpus results are recorded separately.

## Final layout kernel measurements

The final source-matched Criterion collection gives these sample medians:

| Final design median | Control | Small-slab batching | Time reduction |
| --- | ---: | ---: | ---: |
| Complete 64-cycle phasing | 41.322 ms | 30.511 ms | 26.2% |
| Eight-atom complete search | 92.113 ms | 81.146 ms | 11.9% |

These are fixed-work microbenchmarks, not the 35-case latency result. Raw samples,
intervals and source/executable hashes are in
`benchmarks/structure-solution-batched-fft-kernels-macos.json`. External host
activity is uncontrolled; no task-owned workload overlaps these measurements.

Build the Python extension with both experimental features before collecting:

```sh
VIRTUAL_ENV="$PWD/.venv" maturin develop --release --features experimental-solution,experimental-spots
.venv/bin/python benchmarks/solution_fft_interleaved.py --output runs/batched-fft --seconds 15
```

A preliminary collection was stopped after a feature-disabled rebuild returned
import errors. It provides no solve-time or correctness evidence; its partial
logs remain in `runs/batched-fft-aborted-feature-build/`. The reported complete
collection starts afresh after verifying the experimental search entry points.

## Validation

- `cargo fmt --check` and strict workspace/all-target/all-feature Clippy pass.
- `cargo test --workspace --all-features`: 405 passed, 34 ignored.
- Feature-enabled `.venv/bin/pytest -q`: 1,953 passed, 11 skipped,
  34 deselected (external-data/oracle markers retain their normal opt-in status).
- The new FFT tests cover 40 shape/input/direction combinations at three finite
  scales, six invalid boundaries, three complete 80-cycle dual-space trajectories
  and three complete symmetry/count/model recoveries with final anisotropic fits.
- Rust direct transforms and ranked phase tests also require bitwise agreement.
  The independent Python reference uses NumPy rather than wrapping Rust.

No numerical approximation, changed support, reduced observation set, lower
precision, new early stopping rule or extra numerical thread is introduced.
Wall-clock backstops may still change how much otherwise identical work finishes;
fixed-work bitwise equivalence and deadline-limited corpus outcomes are separate
claims.

## Complete 35-case development comparison

Both fresh panels use the same final source/native build; the only execution
option difference is batching. All 140 predictions precede both reference
barriers. The saved schedule verifies balanced leading positions (18 control,
17 batched), 39 runtime source files per snapshot and matching native hashes.
All 35 development cases, including seven above native capacity, remain scored.

| Outcome or timing | Control | Batched |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 16/35 |
| Correct total non-H counts | 26/35 | 26/35 |
| Exact per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-Angstrom structures | 13/35 | 13/35 |
| No model | 4/35 | 4/35 |
| Hard process timeouts | 0 | 0 |
| Wall-time median, all outcomes | 2.582 s | 2.362 s |
| Wall-time p95, all outcomes | 6.990 s | 6.740 s |
| Summed wall time | 106.100 s | 98.776 s |
| Summed P1 phase time | 56.300 s | 49.132 s |
| Quality-scored phase cycles | 45,563 | 49,092 |
| Early quotas completed | 43/57 | 49/57 |
| Starts reaching phase wall backstop | 30/98 | 12/98 |
| Actual batched P1 transforms | 0 | 98,273 |

The median of the 35 paired control/batched wall ratios is **1.07339x** (6.84%
less time), with 30 faster and five slower cases. The paired kernel ratio is
1.08064x, with 33 faster and two slower. Among the same 16 successful cases,
the paired wall ratio is 1.07720x. These paired statistics differ from the ratio
of the two unpaired wall medians. Summed wall time falls 6.90%, and P1 phase time
falls 12.73% while completing 7.75% more scored cycles.

All three recovery sets (labelled structure, total count and per-element count)
are identical between panels. The latest committed development panel's 16
structure successes are also retained. Twenty-eight of 31 nonempty selected
models are byte-identical. The three other selected models occur in cases with
different completed phase budgets: `cod-2241159`, `cod-2011038` and `cod-2011346`.
Their correctness/count classifications remain unchanged; the middle case is
still correctly recovered. Fixed-work bitwise equivalence does not imply that
wall-clock-limited runs select identical models.

The corpus gain is smaller than the fixed-work FFT gain: the scheduler spends
some savings on additional phase cycles, while final refinement remains about
26.3/26.5 seconds. Starts stay at 98, screening fits change 424 to 425 and final
fits 97 to 98. Neither caps nor the final anisotropic validation are relaxed.
The flag remains experimental and false by default. This is one balanced serial
seed-zero development experiment, with external host load uncontrolled; it does
not establish repeated-seed or holdout reliability. The 65 holdout outcomes
remain unopened.

SHELXT is rerun in both panels as the unchanged external comparator: 13/35
labelled structures and 29/35 total counts in each, with one hard timeout and
wall medians 0.895/0.898 seconds. PhaseSmith receives the deposited space group;
these restricted comparisons do not establish general SHELXT parity.

Artifacts:

- `benchmarks/structure-solution-batched-fft-control-macos.json`
- `benchmarks/structure-solution-batched-fft-batched-macos.json`
- `benchmarks/structure-solution-batched-fft-comparison-macos.json`
- `benchmarks/structure-solution-batched-fft-schedule-macos.json`

Raw models, load samples, both source snapshots and pre-reference hashes remain
under `runs/batched-fft/`. Reports retain their complete audits, failed models,
source/native hashes and actual work counters. Re-auditing uses the existing
normalized development validator without changing its tolerances.
