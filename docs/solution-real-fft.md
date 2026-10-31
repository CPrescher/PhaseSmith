# Packed real-density Fourier cycles

## Why the previous solve-time gain was small

The preceding `6756789` development comparison saves 2.46 seconds of final
refinement across all 35 cases. That is only about 2.4% of its 101.49-second
control wall total, so the modest full-search gain is consistent with the actual
work removed. Its optimized panel still spends 50.312 seconds in P1 phasing,
25.092 in final refinement and 99.297 in total solve wall time. These summed
stage times explain the workload; the median paired case speedup is a separate
statistic, not a ratio of unpaired medians.

An eight-second native sample of repeated current element-only solves of the
phase-heavy development case `cod-2010647` records 6,698 main-thread samples.
Collapsed leaf counts include 1,908 in Fourier execution/staging itself, 1,766
in RustFFT kernels/transposes, 787 in atomic-mask updating and 634 in peak scans.
This is representative-case evidence, not a universal CPU fraction. The driver,
baseline native hash, raw-profile hash and counts are preserved in
`benchmarks/structure-solution-fft-staging-profile-macos.json`; raw stacks remain
under `runs/main-bottleneck-profile/`.

A first fixed-eight-line copy specialization reduces a 32-cubed transform by
about 6%. It is retained as a separate prototype patch and raw sample artifact
(`structure-solution-fft-staging-prototype.*`), but was not used for a corpus
collection. The final path instead reduces both FFT arithmetic and staging by
using the real-valued nature of the density maps. The copy prototype is not part
of the final implementation.

## Scope and Fourier equations

`real_fft=True` is experimental and false by default. It requires batched FFTs
(and therefore reusable workspaces and dual-space phasing). It changes the two
Fourier operations inside P1 cycles only. Initialization, Patterson starts,
supplied-symmetry recycling, observations, support codes, phase merit, peaks,
omissions, requested cycle/work budgets, chemical assignment and final fitting
retain their existing definitions. The full XYZ reciprocal array is restored
before scoring and constraints; no measured, weak or missing reflection is
removed. No resolution, precision or atom/Jacobian cap changes.

Let Nz=2M, and use the existing unnormalized DFT convention:

    F(k) = sum_j rho(j) exp(+2*pi*i*j*k/Nz),
    rho'(j) = sum_k F(k) exp(-2*pi*i*j*k/Nz).

The density-to-Fourier operation is positive-exponent. Pack even and odd density
samples into a complex M-vector and transform it with the existing RustFFT plan:

    A(k) = DFT_M^+(rho(2j) + i*rho(2j+1)),
    B(k) = conj(A(M-k)), W(k)=exp(+2*pi*i*k/Nz),
    F(k) = (A(k)+B(k))/2 - i*W(k)*(A(k)-B(k))/2, 0<k<M,
    F(0) = Re A(0) + Im A(0), F(M) = Re A(0) - Im A(0).

Transform y then x on only M+1 z frequencies. Fill the other half of the full
XYZ array from F(-kx,-ky,Nz-kz)*. The Nyquist and zero planes remain present.

The return operation needs the real part of the negative-exponent DFT, as in
the original solver. Reciprocal updates need not be assumed exactly Hermitian.
For arbitrary complex C, first form its Hermitian component:

    H(k) = [C(k) + conj(C(-k))]/2.

Then Re DFT^-(C) = DFT^-(H) mathematically. Transform x then y on the stored
half spectrum, giving row coefficients G(k). Reconstruct the packed M-vector:

    A(k) = [G(k)+conj(G(M-k))]/2
           + i*conj(W(k))*[G(k)-conj(G(M-k))]/2, 0<k<M,
    A(0) = [Re G(0)+Re G(M)]/2 + i*[Re G(0)-Re G(M)]/2.

After the negative M-point transform, write twice the real and imaginary parts
as even and odd real samples. The factor two preserves the original full Nz
normalization. The existing caller still divides by Nx*Ny*Nz. Zero-plane endpoint
averages use overflow-safe midpoints. This applies to general complex return
arrays, rather than silently discarding an assumed small imaginary component.

All values remain f64. The original x/y FFT plans are shared; only the half-length
z plans are additional. Fourier axis order in the return operation and rounding
order change. Bitwise agreement with the original complex Fourier trajectory is
not claimed; fixed-work repeated packed runs remain deterministic. The independent
NumPy reference checks both mathematical operations, including deliberately
non-Hermitian spectra. Short complete trajectories, longer repeated trajectories
and synthetic complete structure/count recoveries are separate checks.

The main reusable workspace has Nx*Ny*(Nz/2+1) complex values. There is also one
opposite-row index per x/y row, a half-length z line, twiddles, small eight-line
staging and FFT scratch. Array workspace is below 18 MiB at the existing maximum
128-cubed grid (checked natively), in addition to the original buffers and FFT
plan metadata. The half spectrum itself uses 16.25 MiB at that maximum. No global
cache, numerical thread or dependency is added. All original even axes in [8,128]
and the 2,097,152-voxel bound remain. The DFT has no finite-support cutoff; the
Gaussian mask's existing three-sigma convention and derivatives are unchanged.
The fused physical refinement evaluator is untouched. The GSAS-II powder oracle
does not cover this experimental single-crystal Fourier path.

## Validation and measured kernels

The full feature-enabled Python suite passes 2,077 tests (11 external-data skips,
34 deselections); Rust passes 409 tests (34 ignored). Strict all-target/all-feature
Clippy, formatting and Ruff pass. New checks cover 40 combinations of shapes,
input types and directions at scales 1e-150, 1 and 1e150; zero/Nyquist modes;
round trips and DC mass; arbitrary complex return maps; seven invalid boundaries;
eight short trajectories; two repeated 256-cycle trajectories; three complete
P1/P-1/Pm count/structure recoveries and the maximum workspace bound.
Direct-transform tolerances are local (2e-12 relative, 8e-12 times input scale
absolute), with short-trajectory tolerances 2e-10/2e-12. Existing tolerances are
unchanged. Original independent value/Jacobian and centered-difference tests
continue to cover the unchanged physical-model derivatives.

| Matched kernel median | Complex batched | Packed real | Time reduction |
| --- | ---: | ---: | ---: |
| Complete 64-cycle phasing, 32 cubed / 2,108 rows | 30.246 ms | 21.710 ms | 28.2% |
| Complete small count-free search | 80.273 ms | 73.438 ms | 8.5% |

Both phase calls complete all 64 cycles. The full search completes 53 scored
cycles/one start in both modes, with 105 actual real transforms in the treatment.
Raw samples, confidence intervals, actual work and source/executable hashes are
in `benchmarks/structure-solution-real-fft-kernels-macos.json`. No task-owned tests,
builds or corpus jobs overlap measured intervals; external host load is uncontrolled.

## Complete development comparison protocol

The balanced serial collector runs all 35 development cases in control/real modes,
with alternating case-leading position. Both enable the latest batched FFT,
filtered peak, scalar scoring, reusable workspace, work-sliced phase, isotropic
assignment and guarded normal-refinement controls. Only packed real P1 cycles
differ. Seed zero, four starts, 800 requested cycles, native 13-second/outer
15-second caps, all original prepared observations and final anisotropic fitting
remain fixed. Exact per-element non-H counts plus RMS <=0.025/max <=0.05 angstrom
are the scoring gate; strict 0.02-angstrom recovery is secondary. Seven above-cap
references stay in every denominator. All 140 predictions precede both reference
barriers, with source/native hashes, load samples, failures and actual effort
saved. Adapter-only seal revision 20 preserves the registry and split. No own
heavy workloads overlap timed collection. The 65 holdout outcomes remain unopened.

## Completed 35-case development results

Both fresh panels use the same compiled native library and the same 40 runtime
source files; all hashes match their saved snapshots and current files. The
verified schedule has 18 control-leading and 17 real-leading cases, with every
one of the 140 predictions saved before either reference barrier.

| Metric | Complex batched control | Packed real |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 16/35 |
| Correct total non-H atom counts | 26/35 | 26/35 |
| Exact per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-Angstrom structures | 13/35 | 13/35 |
| No model | 4/35 | 4/35 |
| Hard process timeouts | 0 | 0 |
| Wall-time median, all outcomes | 2.163 s | 1.805 s |
| Wall-time p95, all outcomes | 6.595 s | 6.074 s |
| Summed solve wall time | 96.077 s | 86.990 s |
| Summed P1 phase time | 49.084 s | 36.964 s |
| Summed final-refinement time | 23.955 s | 26.636 s |
| Starts / scored phase cycles | 98 / 49,230 | 96 / 49,802 |
| Early quotas completed | 48/57 | 51/56 |
| Starts reaching phase wall backstop | 11 | 6 |
| Screening / final fits | 425 / 98 | 433 / 96 |
| Actual packed-real P1 transforms | 0 | 99,596 |

The median of the 35 paired control/treatment wall ratios is **1.13421x** (11.83%
less solve time), with 30 faster and five slower cases. The paired kernel ratio
is 1.14618x (32 faster, three slower). Among the same 16 successful cases, the
paired wall ratio is 1.16179x (13.93% less time; 13 faster, three slower). These
paired statistics differ from the ratio of the two unpaired wall medians.
Summed wall time falls 9.46%; phase time falls 24.69% while scored cycles rise
1.16%. The gain is therefore visible in successful solves as well as failures.

The labelled-structure, total-count, per-element-count and strict-geometry case
sets are identical between panels. All 16 successes from the preceding guarded-
normal and batched-FFT panels are retained. Only two of 31 nonempty selected
models are exactly identical: rounding changes accumulate through the nonlinear
phase search and can select different maps/origins. The original stopping and
shortlist rules can therefore take different branches without changing their
settings. This is reflected in 96 versus 98 starts, more screening fits and
fewer final fits; final-refinement time actually rises 11.2% and offsets part of
the Fourier savings. It is not a reduction in validation work or a new shortcut.
The original observations, tolerances, physical bounds and final anisotropic
mode remain authoritative for every returned candidate.

The packed path remains opt-in. These are balanced serial single-seed development
measurements with external host activity uncontrolled, not repeated-seed or
holdout reliability evidence. The unchanged SHELXT comparator still returns
13/35 labelled structures and 29/35 total counts in both panels, with one hard
timeout and wall medians 0.938/0.884 seconds. PhaseSmith receives the deposited
space group; this does not establish general SHELXT parity. All 65 holdout
outcomes remain unopened.

Artifacts:

- `benchmarks/structure-solution-real-fft-control-macos.json`
- `benchmarks/structure-solution-real-fft-real-macos.json`
- `benchmarks/structure-solution-real-fft-comparison-macos.json`
- `benchmarks/structure-solution-real-fft-schedule-macos.json`
- `benchmarks/structure-solution-real-fft-kernels-macos.json`

Raw models, failures, load samples, source snapshots and pre-reference hashes
remain under `runs/real-fft/`. The profile's native checksum is explicitly
attributed to the preceding benchmark report; its actual sampled image UUID is
recorded from the raw stack file. The copy-only prototype patch applies against
starting commit `6756789` and has separate samples; it contributes no corpus
correctness evidence and is not part of the final path.

```sh
VIRTUAL_ENV="$PWD/.venv" maturin develop --release --features solution,spots
.venv/bin/python benchmarks/solution_real_fft_interleaved.py --output runs/real-fft-new --seconds 15
```
