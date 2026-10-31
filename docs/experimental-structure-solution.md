# Experimental structure solution from integrated HKL data

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.


This opt-in capability accepts **externally integrated and corrected single-crystal
X-ray intensities**, independently of PhaseSmith's detector workflow. It is a
first implementation of published charge-flipping methods, not a claim of
SHELXT-level coverage. No external solver is a runtime dependency.

## Contract and conventions

Dense reflection-buffer admission has a configurable byte policy with an
unchanged 16,000,000-byte default. See the [memory budget contract](solution-memory-budget.md)
for input-only capacity preflight, larger explicitly budgeted searches and its
scope separate from total process memory.

The input is a SHELX HKLF 4 table (`h k l I sigma(I)`, optionally a positive
batch number), a cell in angstrom/degrees, a known space group, and **non-hydrogen
atom counts in the complete conventional cell**. Twinned HKLF 5 observations,
amplitudes masquerading as intensities, neutron data, disorder, partial occupancy,
and anomalous phasing are outside this first slice. No atom coordinates enter
the solver. An HKL file alone generally does not contain the cell or composition.

Signed source intensities and positive uncertainties are retained unchanged.
Symmetry-equivalent observations and Friedel mates are inverse-variance merged;
negative merged intensities become zero **only in the derived phasing amplitudes**,
`Fo = sqrt(max(Imean, 0))`. This is an explicitly biased initial phasing
approximation, not a replacement for a likelihood treatment of weak data.
Systematically absent observations are excluded and counted, not silently fitted.
Input symmetry supplies intensity equivalences and absences. After P1 phasing,
the default experimental path also recovers the supplied origin/hand and refines
an asymmetric-unit model with site-symmetry constraints and positive-definite
anisotropic displacement tensors. This does not discover or prove a space group.
Use `refinement="isotropic"` for constrained per-site isotropic U, or
`refinement="none"` for the historical P1/common-U path. Failed alignment retains
the P1 hypothesis and an explicit diagnostic; it cannot meet a requested
symmetry-refined target. Successfully constrained candidates rank ahead of P1
fallbacks, then by amplitude objective. See [equations](solution-symmetry-adp.md).

## Equations and independent implementation

The independently implemented algorithm follows Oszlanyi & Suto (2004),
*Acta Cryst.* A60, 134–141, DOI
[10.1107/S0108767303027569](https://doi.org/10.1107/S0108767303027569), and their
2008 tutorial, A64, 123–134, especially sections 3–6 and 8, DOI
[10.1107/S0108767307046028](https://doi.org/10.1107/S0108767307046028).
No SHELX, SUPERFLIP, or GSAS-II implementation code is used.

For non-resonant scattering, `F(h) = sum_j f_j(s) exp(2 pi i h.x_j)` and
`s = 1/(2d)`. Phasing uses sharpened amplitudes
`A(h) = Fo(h) / sqrt(sum_j f_j(s)^2)`, with the existing independently sourced
Waasmaier–Kirfel scattering table. Source intensities are not altered.
On an `Nx,Ny,Nz` periodic grid with `N = Nx Ny Nz`, the discrete convention is
`rho = FFT(F).real / N`, `F = N IFFT(rho)` (NumPy signs). Map values are arbitrary
units, not calibrated electrons per cubic angstrom. Both FFT signs and this
normalization are checked against an independent NumPy implementation.

Random starting phases obey Hermitian/Friedel symmetry. `F(000)` starts at zero
and is subsequently free. With fixed `delta = delta_sigma * std(rho_initial)`,
one cycle changes `rho < delta` to `-rho` and leaves `rho >= delta` unchanged.
Measured magnitudes are imposed after transformation; unmeasured coefficients
inside the measured resolution sphere are free, known absences and coefficients
outside the sphere are zero. The sphere includes its boundary within a relative
rounding tolerance of `1e-12`. No measured index may alias a grid/Nyquist boundary.
The optional weak-reflection variant retains calculated weak magnitudes and
shifts Friedel-pair phases by opposite `pi/2`, following section 6 of the tutorial.
Its mask is fixed from the lowest measured sharpened amplitudes.

Candidates are extracted from periodic 26-neighbour maxima with deterministic
tie breaking and parabolic sub-voxel interpolation. The requested atom count
selects the strongest maxima; initial element assignment follows descending
atomic number. This assignment is a hypothesis. Coordinate polishing fits
unsharpened amplitudes using a common bounded isotropic U and an analytically
profiled positive amplitude scale. The objective is unweighted amplitude least
squares; uncertainties are used for merging, not as objective weights.

For calculated amplitudes `a`, `k = (a.Fo)/(a.a)` and `r = k a - Fo`.
The Jacobian includes `dk/dp`,
`dk/dp = ((da/dp).Fo - 2 k a.(da/dp))/(a.a)`, and
`dr/dp = k da/dp + a dk/dp`. Existing native fused structure-factor values and
analytical coordinate/U derivatives supply `da/dp`. The first atom fixes the
translational gauge. Damped Gauss–Newton steps use a bounded backtracking search.
No covariance, uncertainty on atom positions, or statistical confidence is claimed.

## Acceptance and validation

Every run retains trial diagnostics and an explicit budget/termination reason.
Meeting the caller's amplitude-R1 target means only that the candidate meets
that numerical target; it is **not proof of a correct or unique structure**.
Density maps from unsuccessful trials remain diagnostic, not solved structures.
Synthetic recovery is checked up to origin, inversion, and same-element
permutations, separately from amplitude fit. Experimental comparisons must
preserve source provenance and keep archived coordinates out of solver inputs.
SHELXT is an optional black-box comparison; GSAS-II does not cover this phasing
algorithm. Its powder oracle fixtures are therefore unchanged.

The numerical crate and binding use the explicit `experimental-solution` feature.
The default powder build and stable public API remain unchanged.

## Use

Build the experimental feature in a development environment:

```sh
maturin develop --release --features solution
python -m phasesmith.single_crystal.solution sample.hkl \
  --cell 4.913 4.913 5.405 90 90 120 \
  --space-group 'P 31 2 1' --composition Si=3 O=6 \
  --output runs/sample-solution
```

The output directory must be new. It contains `result.json`,
`density-and-observations.npz`, and, if a candidate exists, `candidate-p1.cif`.
A successful constrained refinement additionally writes `candidate.cif` with
asymmetric sites, explicit supplied operations and anisotropic CIF tensors.
`candidate-p1.cif` carries the equivalent full-cell model in the density frame,
including individual ADPs. `candidate.as_structure(cell)` returns the constrained
asymmetric model when available and the P1 fallback otherwise. `result.json`
schema v4 records exact rational operations, the transformation, both models,
per-trial phasing and completion provenance (v2 added symmetry/ADPs; v3 added
phasing strategies).
The compatibility scalar `u_iso` is the full-cell mean U-equivalent after
constrained refinement; physical per-atom tensors live in `candidate.refinement`.
CLI exit status 0 means the requested numerical R1 target was met; 3 means it
was not met; 2 means invalid input or an execution error. Both 0 and 3 retain
candidate diagnostics. The input HKL file is never overwritten.

```python
from phasesmith.crystallography import UnitCell
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure

result = solve_structure(
    read_hkl("sample.hkl"),
    cell=UnitCell(4.913, 4.913, 5.405, 90, 90, 120),
    space_group=space_group_by_symbol("P 31 2 1").space_group,
    composition={"Si": 3, "O": 6},  # all atoms in the complete cell
    options=SolutionOptions(seed=0, trials=8, cycles=800, max_seconds=60),
)
candidate = result.best  # None when no atomic candidate was constructed
structure = None if candidate is None else candidate.as_structure(result.cell)
```

`SolutionCancellation` can be supplied separately as `cancellation=token`.
Another thread may call `token.cancel()` while Rust runs without the GIL.
Cancellation and wall-clock limits are cooperative, checked between FFT cycles,
model evaluations and assignment proposals; setup/one numerical operation is
not preempted. The last available candidates and histories are retained.
The first slice does not provide checkpoint/resume or live progress callbacks.
Seeds reproduce a run on the same numerical build/platform when a wall-clock
or cancellation limit does not intervene; chaotic phasing trajectories can
differ across FFT implementations/platforms.

Element assignment can use up to `assignment_rounds` (default 4, maximum 8)
composition-preserving swaps, scoring at most 256 unlike-element pairs per
round, ordered by atomic-number difference and then site index. Each accepted
swap reduces the amplitude objective and is followed by coordinate/U polishing.
The number of accepted swaps is recorded. This is a bounded heuristic, not an
exhaustive chemical search. Common-U polishing has a first-atom translational
gauge and does not impose special positions or equality of symmetry-related atoms.

## Alternative dual-space phasing

The default remains `phasing="charge_flip"`. Select `"dual_space"` to use
Patterson superposition starts, hybrid difference-Fourier updates and Gaussian
atomicity masks with random omission. Select `"portfolio"` to alternate the two
methods within the existing total trial/time budget. This is an independent
implementation of selected published mechanisms, not full SHELXT intrinsic
phasing. Cell, supplied symmetry and composition are still required. See
[equations and measured comparison](solution-dual-space.md).

```sh
python -m phasesmith.single_crystal.solution sample.hkl \
  --cell 4.913 4.913 5.405 90 90 120 \
  --space-group 'P 31 2 1' --composition Si=3 O=6 \
  --phasing dual_space --output runs/sample-dual-space
```

`delta_sigma`, `weak_fraction` and `cleanup_cycles` apply to charge flipping.
The alternative's sharpening exponent, hybrid multiplier, Gaussian width and
omit schedule are fixed, documented and recorded; no per-material tuning is
performed. `result.phasing_trials` contains immutable typed records aligned
with histories and candidate trial indices. Score definitions differ between
methods and are labelled; candidate ranking still uses the common atomic fit.

## Partial-framework completion

[Partial-framework completion](solution-completion.md) is enabled by default for
symmetry-refined experimental searches. If the complete candidate is insufficient,
it keeps the stronger species, tests compatible framework settings and searches
for one missing weakest-species orbit using difference Fourier and fixed-set
projection. Disable it with `SolutionOptions(completion=False)` or CLI
`--no-completion`. Existing frozen benchmark definitions default to disabled;
`--completion` explicitly enables the follow-up. The original total runtime,
trial and cancellation budgets continue to apply. A completion must improve the
candidate objective and its effort/outcome remains in `candidate.completion`.
Only the single-missing-orbit slice is implemented; general fragment completion
and automatic space-group determination remain outside scope.

## Experimental development case

The benchmark-only reference audit requires Gemmi and SciPy (the `test` and
`refinement` extras); the native solver itself requires neither package.

The explicit benchmark `benchmarks/structure_solution.py` uses Antonio L.
Llamas-Saiz's [thiourea 130 K deposit](https://doi.org/10.5281/zenodo.18910433)
(8 March 2026, CC-BY-4.0). Exact HKL and reference-CIF members and local SHA-256
hashes are pinned in `validation/structure-solution-thiourea.json`. The whole
318 MB archive was not downloaded or locally MD5-verified. Only bounded ZIP
ranges and the two required members were retrieved.

```sh
# Network retrieval happens only with the explicit --download-data flag.
python benchmarks/structure_solution.py --download-data --output runs/thiourea
# Optional local black-box comparison; exact binary hash is checked:
python benchmarks/structure_solution.py --output runs/thiourea-with-oracle \
  --shelxt-executable ~/.local/opt/shelx/versions/2026-10-28/bin/shelxt
```

Only the corrected HKL, known cell/symmetry and complete non-H composition enter
the solve. Reference atom coordinates are read afterwards, solely for a separate
origin/inversion/element-permutation audit. This is an exploratory development
case, not an untouched validation holdout. It is a molecular crystal and does
not establish performance across inorganic materials. The deposited model has
hydrogens and anisotropic displacement, while the historical P1 baseline did not.
Published and native residuals therefore do not represent the same objective.

### Historical P1 baseline, recorded 28 October 2026

- Independent FFT-cycle comparisons cover plain/weak-perturbed phasing, free
  missing coefficients, fixed zeros, Friedel symmetry and normalization. Native
  amplitude residuals/Jacobians match independent complex sums on fixed and
  deterministic randomized cases; centered finite differences check polishing.
- Blind synthetic quartz recovery finds all 9 non-H atoms with amplitude
  R1 about `8.4e-12`; coordinate matching is checked independently up to origin,
  inversion and equal-element permutations, with a 0.005 angstrom maximum gate.
- On the 9,047-observation thiourea file, 8 starts recover all 16 non-H atom
  positions **and element labels**: maximum/RMS reference distance
  `0.01527/0.01095` angstrom. Native amplitude R1 remains `0.101923`, so the
  requested `0.06` target is **not met**, and the stop is `trial_budget`.
  The recorded wall time is 24.75 seconds on the development Apple Silicon host.
  A separate pinned SHELXT 2018/2 run also returns a structure, in the equivalent
  Pmc21 setting with permuted axes. Its search/model and reported R1 differ;
  this is not an equal-objective accuracy or speed comparison.
- The frozen numerical/environment record is
  `benchmarks/structure-solution-thiourea-macos.json`; it includes input and
  implementation hashes, native-extension hash, options, all best-candidate
  coordinates, the separate reference audit, and external executable provenance.
- A 32-cubed Fourier field with 4 point atoms and 2,108 reflection constraints
  takes about 30 ms for 50 native cycles (Criterion, 10 samples; measured interval
  28.8–33.6 ms). No established production solver is replaced by this new kernel.
- `cargo fmt`, strict workspace Clippy, and workspace tests with all features
  pass: **383 Rust tests/doctests passed**, 34 ignored. The full Python suite
  passes: **1,199 passed**, 11 skipped, 34 deselected. External/opt-in test
  exclusions remain visible. A freshly built default wheel passes a native
  crystallography smoke check and correctly omits the experimental solver.

The frozen baseline above is preserved. The following experimental follow-up
adds asymmetric reconstruction and individual displacement refinement; hydrogen
placement, automatic space-group selection, anomalous data, disorder and twinning
remain future work.

## Symmetry/ADP follow-up, 28 October 2026

`benchmarks/structure_solution.py` retains `--refinement none` as its default,
so rerunning the historical benchmark does not silently select the new model.
For the follow-up, run:

```sh
python benchmarks/structure_solution.py --refinement anisotropic \
  --output runs/thiourea-adp
python benchmarks/solution_adp_transfer.py --download-data \
  --output runs/adp-transfer
```

The transfer source manifest is
`validation/structure-solution-adp-transfer.json`: exact sizes/SHA-256s of COD
CIF reflection loops and separate reference CIFs, original authors/DOIs, metadata,
retrieval date and CC0 attribution. Measured signed F² and sigma are translated
without conversion or clipping. Reference atom tables are parsed only after the blind
result is saved. Two compact ternary non-H cases were selected before solver
outcomes and share seed 0, eight starts, 1,200 cycles, a 60 s budget and 0.06 R1
target. Neither failure nor success was used to retune these settings.

| Case | P1 R1 | Constrained R1 | Geometric check after solving |
| --- | ---: | ---: | --- |
| Thiourea, development | 0.101923 | 0.049409 | 16 non-H atoms; max/RMS error 0.01037/0.00771 Å |
| BiSeI, COD 2010577 | 0.071034 | 0.052285 | 12 atoms; max/RMS error 0.00252/0.00175 Å |
| K[BrF4], COD 1549359 | 0.091212 | unavailable | Wrong P1 candidate; max/RMS error 3.24564/1.52739 Å; no symmetry match |

Thiourea reaches its numerical target on the first start with six asymmetric
sites and 41 independent coordinate/ADP parameters (13 coordinate parameters,
one free polar-origin direction removed). The first validated run took 4.67 s (the frozen repeat took 4.73 s). BiSeI uses
three mirror sites and 18 parameters; the initial recorded transfer takes 0.55 s.
K[BrF4] exhausts eight starts and remains an explicit failure. Its moderately low
P1 residual demonstrates why geometry and symmetry are assessed separately.
Times describe these builds/runs; no equal-objective SHELX speed claim is made.
Hydrogens and wavelength-dependent anomalous terms are still absent, and the
native unweighted amplitude objective differs from deposited refinement metrics.

The new frozen reports are `benchmarks/structure-solution-thiourea-adp-macos.json`
and `benchmarks/structure-solution-adp-transfer-macos.json`; previous records are
untouched. Implementation/input/native-extension hashes accompany the results.
The fused eight-atom, 2,108-reflection, 69-parameter ADP value/Jacobian evaluation
takes 0.764–0.768 ms in a 10-sample Criterion run on this host. There was no previous
constrained-ADP hot loop; the earlier Fourier benchmark remains separate.

Tests cover independently computed complex amplitudes and centered differences
in P1, inversion, mirror, nonorthogonal trigonal and cubic settings, repeated ADP
eigenvalues, special-position multiplicity, polar origin gauges, invariant tensors,
bounds, F000, closest periodic images, cancellation and export equivalence.
The full Python suite passes: **1,209 passed**, 11 skipped, 34 deselected.
Workspace all-feature Rust tests pass: **386 passed**, 34 ignored.
Formatting and strict Clippy pass; the default
wheel remains independently checked without the optional solver.
