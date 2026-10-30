# Fused phase workspaces and exact Gaussian geometry

## Diagnosis and scope

The follow-up starts at `f534544`, with packed real FFTs and every previously
measured execution optimization enabled. In its last development panel, P1
phasing still costs 36.964 seconds and final refinement 26.636 seconds out of
86.990 total wall seconds. Faster Fourier transforms expose mask construction,
peak scanning and data movement as larger fractions of the remaining work.

Two sequential five-second native profiles cover a phase-heavy failed case
(`cod-2010647`) and a refinement-heavy case (`cod-2021292`). AtomicWorkspace update
has 630/366 collapsed self samples out of 4,194/4,206 main-thread samples; peak
scanning has 483/254. Fourier work remains distributed across several kernels.
Normal solving/refinement is also substantial in the second case. These identify
representative hotspots, not universal corpus percentages. Native bytes were
hashed immediately before sampling. The raw stack hashes, sampled image UUIDs,
driver scripts and complete leaf counts are preserved in
`benchmarks/structure-solution-phase-workspace-profiles-macos.json`; raw profiles
remain under `runs/post-real-fft-profile/`.

An initial geometry-only implementation improves the 64-cycle phasing benchmark
by about 3%. Its separate raw kernel samples and full source patch are retained
as `structure-solution-phase-workspace-mask-only-*`. No corpus was collected for
that intermediate version. The final implementation also removes whole-grid
transfers between density and complex Fourier buffers.

`fused_phase_workspace=True` is experimental and false by default. It requires
reusable Fourier workspaces. It caches Gaussian axis calculations inside P1
projection and, when `real_fft=True`, fuses the real density input/output transfers.
It changes no phase equations, observed/free/fixed-zero reflections, resolution,
precision, seeds, thresholds, omission sequence, requested work/deadlines,
assignment shortlist, anisotropic fitting, resource admissibility or correctness
gates. The original execution remains the control.

## Exact Gaussian factorization and support

For peak fractional coordinate p, grid dimensions N, rounded grid centre o,
and stencil offset d, the original calculation is

    c_j = p_j N_j, o_j = round(c_j),
    delta_j(d_j) = [o_j + d_j - c_j]/N_j,
    q2 = delta^T G delta,
    M[index] += (2*pi)^(-3/2) sigma^(-3) exp[-0.5*q2/sigma^2],
    rho_projected = max(rho,0) M.

G is the original Cartesian direct metric. The finite stencil bounds come from
the existing reciprocal metric and are at most 16 steps per axis. Acceptance is
inclusive at q2 <= (3*sigma)^2*(1+1e-12), exactly as before. Rounded centres,
periodic wrapping, negative density clipping, mask normalization, Gaussian
overlap and addition order are unchanged. No stencil image is deduplicated.

For each peak, prepare at most 33 entries per axis containing delta, the wrapped
flat-index contribution and three products G_ij*delta_j. The x/y pair prepares
its first two matrix-vector terms once, using the same Iterator::sum identity
and order as the original three-term dot product. The z loop adds the third
term and evaluates the original outer dot product, division, exponential and
mask addition. This is common-subexpression removal, not reassociation of the
floating-point reduction. Integer flattening remains exact under the same grid
bounds. The projected map is still materialized by the original full-array
multiply after all overlapping peaks have accumulated.

The private boxed geometry workspace is at most 4.5 KiB (checked natively), with
three reusable fixed-size arrays. Entries used by every new peak are overwritten;
there is no rounded-coordinate lookup or reuse of a previous peak's Gaussian.
No per-voxel table, global mutable cache, new dependency or numerical thread is
introduced. All existing finite-density, cell, sigma, peak-count and grid limits
remain in force. Because the projection equation is unchanged, the independent
NumPy reference is reused directly. Existing mass/moment and support tests
remain applicable, with added bitwise comparisons at boundary/extreme cases.
The physical value/analytical-derivative evaluator is untouched; this adds no
new differentiable profile term. GSAS-II's powder oracle does not cover P1 masks.

## Fused Fourier boundaries

The old packed-real path first writes Complex(rho_projected,0) into a full-grid
buffer, then reads its real component to pack the half-length FFT lines. The new
path reads rho_projected directly into those same lines. All Fourier arithmetic,
full reciprocal spectrum restoration, constraints and scoring remain identical.

On return, the old path writes Complex(2*line_component,0), then reads the real
part and divides by the voxel count. The new path writes 2*line_component/N
straight into the existing density array, with multiplication followed by the
same division. Both paths share the same Fourier routines through monomorphized
read/write callbacks. The recycled complex buffer can retain its reciprocal
contents: the next positive transform overwrites every coefficient before it
is used again. The existing best-map/workspace ownership swaps are preserved.
No extra full-grid buffer is allocated, and the packed-real workspace bound
remains unchanged.

Native tests compare both direct-transfer operations bit for bit on cubic and
non-cubic/non-power-of-two grids, including arbitrary non-Hermitian return data.
Complete 128-cycle random/Patterson trajectories also match bitwise for both
legacy-complex and packed-real execution. Three synthetic P1/P-1/Pm searches
recover the same element counts, coordinates and final anisotropic mode exactly.
These fixed-work statements are separate from deadline-limited search outcomes.

## What the published SHELXT method still suggests

[Sheldrick (2015), sections 2–5](https://journals.iucr.org/a/issues/2015/01/00/sc5086/)
describes Gaussian/random-omit masks, phase-quality-driven escalation, retaining
plausible symmetry alternatives, isotropic refinement and parallel independent
work. Useful next experiments are quality-guided restart allocation and pooling
cheap candidates before expensive polishing, with ambiguity retained. Its
organic CHEM criterion is unsuitable as a general mineral default. Space-group
inference would be a separate feature: current PhaseSmith benchmarks supply
symmetry. Parallel execution needs its own reproducibility/CPU-budget comparison.
These are published principles and proposed experiments. This change independently optimizes existing
algebra and data movement; it does not add those scheduling/search features.

## Validation and comparison protocol

The final feature-enabled full Python suite passes 2,137 tests (11 external-data
skips, 34 deselections), and Rust passes 411 tests (34 ignored). Strict workspace,
all-target/all-feature Clippy, formatting and Ruff pass. Projection tests compare
40 shape/cell/cycle combinations at three finite scales against the original
kernel; ordinary-scale values also match the independent NumPy reference at the
unchanged local 2e-12/2e-13 tolerance. Empty, negative, tied and signed-zero maps,
invalid inputs, the maximum axis-cache length and periodic/support boundaries
are covered. New bitwise tests do not loosen any existing numerical tolerance.

All 35 development cases are compared in balanced serial control/fused modes,
with alternating leading position. Both enable packed real FFTs, scalar scoring,
filtered peaks, existing scheduling and guarded normal refinement. Only the fused
phase-workspace option differs. Seed zero, four starts, 800 requested cycles,
native 13-second/outer 15-second caps, final anisotropic fits, original observations
and input-derived atom/Jacobian caps remain fixed. Exact per-element non-H counts
plus RMS <=0.025/max <=0.05 angstrom remain the gate; strict 0.02-angstrom recovery
is secondary. Seven above-cap references remain in every denominator. All 140
predictions precede both reference barriers. Adapter-only seal revision 21 keeps
the registry and 35/65 split unchanged. No own tests, builds or kernel benchmarks
overlap timed collection. The 65 holdout outcomes remain unopened.

## Measured results

The same-executable fixed-work measurements use a multi-atom 32-cubed map and
2,108 measured reflections. Criterion sample medians are:

| Work | Control | Fused | Time reduction |
| --- | ---: | ---: | ---: |
| Complete 64-cycle P1 phasing | 23.217 ms | 20.888 ms | 10.0% |
| Complete small count-free search | 77.120 ms | 75.198 ms | 2.5% |

Both phasing modes execute 64 cycles. Both small searches execute one start and
53 cycles; the fused mode records 53 cached-mask cycles and 105 direct Fourier
operations. The last cycle exits after scoring, before its return transform.
These timings are separate from the earlier geometry-only probe: do not compare
their absolute times across collections. Raw samples, benchmark source and
executable hashes are in
`benchmarks/structure-solution-phase-workspace-kernels-macos.json`. The initial
geometry-only patch applies cleanly to `f534544` using an isolated Git index.

The fresh, balanced development comparison gives:

| Measure, all 35 development cases | Control | Fused |
| --- | ---: | ---: |
| Correct labelled structures | 16/35 | 16/35 |
| Correct total non-H atom counts | 26/35 | 26/35 |
| Correct per-element non-H counts | 16/35 | 16/35 |
| Strict 0.02-angstrom structures | 13/35 | 13/35 |
| Median wall time, all outcomes | 1.874 s | 1.803 s |
| Summed wall time | 90.347 s | 85.260 s |
| Summed phase-search time | 38.270 s | 32.962 s |
| Summed final-refinement time | 27.781 s | 27.922 s |
| Quality-scored cycles | 49,629 | 50,191 |
| Starts / screening fits / final fits | 96 / 433 / 96 | 96 / 433 / 96 |

The median paired wall ratio is 1.06094x: 5.74% less time, with 31 cases faster
and four slower. Summed wall time falls 5.63%, and summed phasing time falls
13.87% despite 1.13% more scored cycles before deadlines. On the 16 common
successes, the median paired wall ratio is 1.07511x, with raw wall medians
1.002/0.940 seconds. These are a single-seed, single balanced collection on an
uncontrolled host, not repeated timing confidence intervals.

Correctness and both atom-count recovery case sets are unchanged. All 31
nonempty selected models match; four cases have no model in both modes. There
are no gained or lost structures and no hard PhaseSmith timeouts. Both panels
use the same native binary and 41 matching runtime-source snapshots. The
fused panel records 50,193 cached projection cycles and 100,374 direct Fourier
operations, exactly matching its eligible operation counts. Remaining
refinement/screening costs explain why the phasing gain does not translate into
an equally large whole-search gain. This improves execution, not recovery.

The paired reports are `benchmarks/structure-solution-phase-workspace-{control,fused}-macos.json`;
the comparison and verified 140-prediction/two-barrier schedule are
`benchmarks/structure-solution-phase-workspace-{comparison,schedule}-macos.json`.
Raw workers, audits and sealed source snapshots remain in `runs/phase-workspace/`.
The feature stays false by default, and the 65 holdout outcomes remain unopened.

To collect a new development-only panel in a new directory:

```sh
.venv/bin/python benchmarks/solution_phase_workspace_interleaved.py --output runs/phase-workspace-new --seconds 15
```

After both modes finish, independently audit each mode using
`benchmarks/solution_corpus_reaudit.py --manifest validation/solution-corpus-v5.json --run MODE_DIRECTORY --output REPORT.json`,
then compare the normalized reports with `benchmarks/solution_phase_ablation.py`.
The preserved collection uses adapter seal revision 21.
