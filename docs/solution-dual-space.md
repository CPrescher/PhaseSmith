# Patterson-seeded atomic dual-space search

This experimental alternative implements published mechanisms in Sheldrick (2015),
*Acta Cryst.* A71, 3–8, sections 2.1–2.2,
[10.1107/S2053273314026370](https://doi.org/10.1107/S2053273314026370).
It is independently written from the equations, without SHELX program code.
It is not full SHELXT intrinsic phasing or automatic space-group determination.

## Equations and explicit choices

Let `E=Fo/sqrt(sum_j f_j(s)^2)` be the existing composition-sharpened amplitude,
and `G=E^q Fo^(1-q)` with fixed `q=0.5`. This is composition sharpening, not a
fitted Wilson normalization. All `G` are scaled by a common positive factor
so their RMS is one; the factor changes map units only. FFT conventions remain
`rho=FFT(F).real/N`, `F=N IFFT(rho)`.

A Patterson start uses `P=FFT(G^2).real/N`, zero at unmeasured coefficients,
and a strong non-origin positive local-maximum grid vector `u` to form
`rho(x)=max(0,min(P(x),P(x-u)))`. Vectors whose translation leaves the measured
Fourier support invariant are excluded (centring-origin peaks). Up to 128
positive periodic maxima are inspected; opposite vectors share one canonical
choice. Start rank chooses successive remaining vectors deterministically.
If no vector survives, fall back to explicitly recorded random-phase initialization.
Every fourth dual-space trial instead starts with the supplied random phases.

For each iteration, find the strongest positive periodic local maxima in `rho`.
Use up to `min(2*Natoms,128)` peaks during the first 50 cycles and `Natoms`
thereafter. Build `M(x)` from unit-integral Cartesian Gaussian peaks centred on
sub-grid interpolated maxima, with `sigma=0.35*d_min`. Gaussians have **finite
support `r <= 3 sigma`**, including the boundary within relative `1e-12` rounding.
Sigma must lie in `[1e-4,5]` angstrom. Mask construction uses the cell metric and a bounded periodic neighbourhood;
a radius reaching half a fractional axis or more than 16 grid steps is rejected.
The projected map is `rho_a=max(rho,0)*M`. Every third iteration omits a
seeded selection of `floor(0.1*Npeaks)` peaks (at least one when there are two
or more). Stateless SplitMix64 rankings make omission independent of global RNGs.
The count, width, omission rules and support are implementation choices; the
paper does not specify an identical parameterization.

Calculate `Gc=N IFFT(rho_a)`. Profile a positive common scale
`k=sum(G*abs(Gc))/sum(abs(Gc)^2)` over measured coefficients. With `m=3`, use
`Fnext=(m*G-(m-1)*k*abs(Gc))*exp(i arg(Gc))` at measured coefficients;
use `k*Gc` for missing coefficients **inside the original measured sphere**
(including 000), and zero for absences/outside. Negative hybrid coefficients are
retained as signed coefficients, equivalent to a pi phase change. There is no
unvalidated resolution extension. Both Friedel mates are present and the real
map/omission process preserves Hermitian coefficients.

The iteration score is normalized amplitude least-squares after profiling scale.
Keep the lowest-score atomic map after the first 50 cycles (or all cycles for
short trials); omission cycles remain diagnostics but are not eligible as the
final complete map. If interrupted before an eligible
map exists, the latest Fourier map remains a diagnostic. A degenerate initial
projection returns an empty map and a rejected atomic trial. Candidate
extraction, element assignment, supplied-symmetry alignment and ADP refinement
then follow the same path as charge flipping. Cancellation/time checks occur
between FFT and mask passes. Values remain deterministic without time interruption.

`phasing="charge_flip"` preserves the previous default. `"dual_space"` selects
this alternative; `"portfolio"` alternates charge-flip and dual-space starts
within the caller's existing total trial/time budget. Per-trial method, start,
Patterson vector, parameters and completed cycles are exported with histories.
The input remains corrected HKLF 4, supplied cell/group and full-cell composition.

## Verification

Independent NumPy equations check complete cycles, masks, Patterson vectors,
normalization, support, omission determinism and score selection. Fourier maps
are global iterative operations: no new smooth model parameter or analytical
profile derivative is introduced. Existing fused coordinate/ADP derivatives and
their finite-difference tests are unchanged. GSAS-II does not cover phasing, so
its powder fixtures are not regenerated. Experimental results must include the
known difficult K[BrF4] case and keep reference coordinates out of all searches.

## Use and recorded comparison, 28 October 2026

Build with `maturin develop --release --features experimental-solution`, then set
`SolutionOptions(phasing="dual_space")` or CLI `--phasing dual_space`.
`--phasing portfolio` is also available. Keep using externally corrected HKLF 4
and supplied cell, symmetry and complete-cell composition.

The explicit serial comparison can be rerun from the repository root:

```sh
python benchmarks/solution_dual_comparison.py --download-data \
  --output runs/dual-comparison
```

All output folders must be new. Downloads occur only with the explicit flag and
are checked against the existing exact-size/SHA-256 manifests. The three cases
are **reused development evidence**, not new validation holdouts. Each method
gets the same seed 0, eight-start/1,200-cycle limits, 60 s budget, anisotropic
polishing and 0.06 R1 target. Reference atom tables are parsed after saving blind
results. No per-case phase-search parameters were tuned.

| Case | Charge-flip refined R1 | Dual-space refined R1 | Recovery |
| --- | ---: | ---: | --- |
| Thiourea | 0.049409 | 0.049409 | Both recover 16 non-H atoms; max error 0.01037 Å |
| BiSeI | 0.052285 | 0.052285 | Both recover 12 atoms; max error 0.00252 Å |
| K[BrF4] | 0.091212 (P1) | 0.080879 (P1) | Both fail the supplied symmetry and geometry checks |

Thiourea's first Patterson start reaches the target; the serial run takes 3.59 s
versus 3.87 s for charge flipping. BiSeI succeeds on the fourth, random-phase
start (3.98 s versus 0.46 s). K[BrF4] exhausts eight starts (34.90 s versus
24.37 s): its best dual-space candidate still has a 1.786 Å maximum and 1.062 Å
RMS element-matched error. A smaller residual is not a correct solution. These
single-run timings do not establish a general speed advantage or increased solve
coverage. Charge flipping therefore remains the default.

The fused FFT/mask benchmark uses a 32³ grid, four point atoms, 2,108 measured
reflections and 50 cycles. Ten-sample Criterion slope intervals are
54.393–54.651 ms for dual space and 28.065–28.450 ms for charge flipping; the new
atomicity passes cost additional time. The historical charge-flip record remains
unchanged. The frozen comparison is
`benchmarks/structure-solution-dual-space-macos.json`, with input manifests,
implementation/native hashes, actual starts, geometric audits and raw timing
estimates. No external SHELX executable is invoked by this comparison.

The full Python suite passes: **1,223 passed**, 11 skipped, 34 deselected.
Workspace all-feature Rust tests pass: **387 passed**, 34 ignored. Formatting,
strict Clippy and the fresh default-wheel crystallography/import smoke check pass.
The default wheel omits the optional phasing engines. The new tests include
complete independent Fourier cycles, stable repeats and global amplitude scale,
Cartesian masks in nonorthogonal cells, finite-support boundaries, Gaussian
volume/periodic moments, the 50-cycle peak-count transition, omitted masks,
centring-vector fallback, malformed boundaries, degenerate masks, stop handling,
and typed portfolio provenance/JSON export. No statistical model uncertainties
or broader crystallographic validity are inferred from this test count.
