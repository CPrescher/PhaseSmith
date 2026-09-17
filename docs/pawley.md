# CW Pawley refinement

`phasesmith.refinement.pawley` fits independent powder-family integrated areas
with bounded least squares. Atomic coordinates and structure factors are not
required. Selected cell, CW profile and linear background parameters can join
the same fit. Production calculations and refinement run in Rust.

The matrix-free measured acceptance gate passes for sucrose and neutron LaB6,
including convergence under the declared budgets and exact repeatability.
Sucrose converges locally at a hard-support boundary. Use matrix-free mode for
large joint fits; dense mode remains the default small-problem reference.
See the [validation report](pawley-validation.md) for evidence and limitations.

Use Pawley when the cell and symmetry are known but individual reflection
intensities should remain free. Le Bail uses iterative intensity redistribution;
Rietveld predicts relative intensities from a structural model. A good Pawley
fit does not establish a unique structure, and its area totals are not
quantitative phase mass fractions.

## Supported boundary

| Capability | Status |
| --- | --- |
| One CW histogram, multiple phases, fixed X-ray or neutron wavelength | Supported |
| Symmetric TCH; fixed FCJ axial geometry | Supported |
| Independent areas; optional signed areas | Supported |
| U/V/W/X/Y and bounded symmetry-independent cell variables | Supported |
| Fixed plus polynomial/Chebyshev/point/composite linear background | Supported |
| Fixed, affine and linear parameter ties, including dependent bounds | Supported |
| Native/Python cancellation, accepted checkpoint continuation, standalone and shared-bundle persistence | Supported |
| Dense full analytical Jacobian and full-rank interior covariance | Supported within explicit allocation limit |
| Matrix-free bounded solving with analytical JVP/VJP | Supported; rank and covariance omitted |
| Fixed detected-area CW spectra | Implemented, including union domains and restart |
| Single-/multi-bank TOF | Supported in the [TOF workflow](tof-pawley.md) |
| Structural restraints and automatic recipe integration | Outside Pawley scope |
| Live pinned GSAS-II optimizer comparison | Fixed-cell and cell-refinement agreement tested; strict profile equivalence not established |

The implementation uses deterministic serial kernels and bounded dense or
matrix-free solvers. `max_elements` is a conservative estimate of native floating-point
workspace elements, not a total process RSS limit. Python result arrays,
serialization and allocator overhead consume additional memory. Requests above
the estimate fail before allocating the dense workspace. Default 50 million
f64 elements is approximately 400 MB; raise it deliberately for larger problems.

## Start from explicit families

```python
import numpy as np
from phasesmith import ConstantWavelengthInstrument, PowderPattern
from phasesmith.refinement import pawley

x = np.linspace(39.0, 41.0, 1001)
instrument = ConstantWavelengthInstrument(1.5406, 0.0, 0.0, 0.001, 0.002, 0.0)
phase = pawley.PawleyPhase("sample", ("100", "110"), np.array([40.0, 40.06]), np.array([3.0, 7.0]))
truth = pawley.PawleyInput(PowderPattern(x), instrument, (phase,))
y = pawley.calculate(truth).calculated_y

phase = pawley.PawleyPhase("sample", ("100", "110"), np.array([40.0, 40.06]), np.zeros(2))
request = pawley.PawleyInput(PowderPattern(x, observed_y=y), instrument, (phase,))
result = pawley.refine(request)
print(result.termination_reason, result.calculation.rwp, result.intensities)
```

For measured data supply `uncertainty` as one standard deviation, an optional
`mask` with true meaning included, and a fixed `background` on `PowderPattern`.
The optional input `background` model is an additive refinable residual, never
a replacement for that fixed array. Negative observations are preserved.

`PawleyPhase.from_cell(...)` accepts a typed `UnitCell`, `SpaceGroup`, fixed
wavelength, a two-theta reflection range, and optional `LatticeParameterBounds`.
`from_cif(...)` extracts the same cell/symmetry metadata from a CIF. Neither
constructor uses atom sites to calculate areas. Include a profile-tail margin
in the requested reflection range; an explicit family list makes no completeness
claim. The generated domain is conservative over its supplied cell bounds.
Keep those bounds tight enough to avoid unnecessary guard families. Inaccessible
Bragg reflections or out-of-domain trials are rejected, not silently removed.

## Matrix-free calculations and fits

Set `PawleyOptions(solver="matrix_free")` to retain support-block derivatives
and solve bounded steps with projected, preconditioned conjugate gradients.
`calculation.jacobian` is then `None`; use `calculation.jacobian_operator.jvp(v)`
and `.vjp(u)` for analytical products in the same scaled-free coordinates.
Both modes expose this operator and its `shape` and `storage_elements`.
Dense mode remains the default and a reference for small problems.

The iterative solver checks the recomputed linear residual against
`linear_tolerance * (1 + initial_residual_norm)` and respects
`max_linear_iterations`. Failure to satisfy that check is a numerical failure,
not convergence. Both controls and the solver mode are bound into checkpoints.
The preconditioner uses 32-column diagonal Gram blocks with a positive diagonal
stabilizer; this changes neither the objective nor the checked normal operator.
Dense physical/free Jacobians are absent in this mode, but constraint transforms
and active-face work still have quadratic parameter storage. `max_elements`
includes those workspaces and the exact native support count.

Matrix-free results return `rank=None` and no covariance rather than claiming
full rank from an iterative solve. Exact coincident-family and unobserved-column
diagnostics remain available. `diagnostics["krylov_iterations"]` distinguishes
inner product iterations from active-set iterations.

## Joint parameter selection

```python
from dataclasses import replace

parameters = pawley.build_parameter_set(request, profile_parameters=("w_deg2",))
request = replace(request, parameters=parameters)
result = pawley.refine(request)
```

Set `lattice=True` only for phases constructed with lattice domains. Modify
individual `ParameterSpec` records to change bounds or selection; preserve their
keys and units. `parameter_key("intensity", phase_id, reflection_id)` identifies
an area. The profile owner is `"instrument"`; cell parameters use the phase ID.
Pass existing `FixedConstraint`, `AffineConstraint` or `LinearConstraint`
objects in `PawleyInput.constraints` to impose explicit ties. Phase scales are
absent: free areas already absorb scale, multiplicity and fixed amplitude
corrections. Unknown parameter families or widened domain bounds are rejected.

Refinement first estimates areas/background with geometry held fixed, then
moves all selected variables jointly. The implementation uses column scaling,
QR reduction and box-face solves, with SVD-derived constraint row spaces and
explicit orthonormal null-space bases for coupled faces. Constraint chains are
exact and backtracking evaluates the complete objective. Severe backtracking
increases damping for the next step. Convergence checks use the
normalized feasible step, undamped projected gradient, or small relative cost
reduction with agreement between the actual and predicted reduction, modest
damping and at least one tenth of the proposed step. A step
made small solely by heavy damping reports stagnation. A zero starting area can
leave its lower bound. `signed_intensities=True` explicitly removes the default
non-negative area bound; it does not change the observation weights.

## Areas, support and uncertainty

Areas have units of observed-y times degrees and refer to the complete powder
family before finite support truncation. They are not F-squared. Do not multiply
by multiplicity or Lorentz/polarization again. The wavelength and axial geometry
are fixed in this release.

Support follows the existing CW/FCJ kernels, including physical endpoints. No
normalization to the observed grid occurs. Jacobians differentiate the profile
with support membership held fixed; derivatives at moving cutoffs are undefined.
For symmetric fixed-position CW fits, positive objective jumps at cutoffs can
form local barriers. The solver checks those finite jumps separately from the
smooth derivatives, optimizes on the permitted side, and reports a `support_`
convergence criterion when appropriate. This is local convergence of the exact
truncated objective; covariance is omitted there. The treatment is deliberately
conservative for simultaneous events and does not certify moving-position or
axial-profile cutoffs. See the [mathematical contract](mathematics/refinement.md#pawley-family-area-least-squares).

Mask and uncertainty weights are applied once; least squares adds no bin-width
factor. Integration diagnostics do use physical grid spacing.

`result.diagnostics` reports evaluation and active-set iteration counts,
stage wall times, rejected backtracks, the last normalized step and projected
gradient when checked, and the convergence criterion. Timings are not part of
scientific checkpoint identity.

Inspect `rank`, `active_bounds`, `active_width_bounds`, `calculation.unobserved_reflections` and
`calculation.coincident_groups`. Each exact-coincidence record contains stable
family identities and their area sum; an arbitrary split is not a measurement.
Other near-dependencies are reflected in numerical rank when dense mode is selected. Damping does not enter
the rank estimate. Unobserved free columns preserve their accepted values.

The Jacobian and covariance use **scaled free coordinates**, in
`ConstraintTransform(request.parameters, request.constraints).free_keys` order.
`result.parameters` contains fitted physical values. To propagate covariance,
use the exact transform derivative matrix `D`: `C_physical = D @ C_free @ D.T`.
Full covariance includes intensity/profile/background cross terms and is only
returned after convergence for an interior full-rank solution with positive
residual degrees of freedom. Otherwise `covariance_limitation` explains its
absence. Known sigmas give unscaled inverse information; unit weights apply
reduced chi-square. Bound-constrained reduced chi-square remains an approximate
statistic; nominal observable free-parameter count and numerical rank are
reported separately.

## Runtime and persistence

```python
project = pawley.PawleyProject(request)
result = project.refine(max_iterations=100, max_evaluations=1000)
project.save("sample.pawley.json")  # new file only
restored = pawley.PawleyProject.load("sample.pawley.json")
y_accepted = restored.calculate().calculated_y
```

`project.stop()` or an explicit shared `CancellationToken` requests cooperative
cancellation. `progress=` receives structured native boundary events.
Budgets are cooperative: an ongoing factorization and final diagnostics can
overrun a wall-clock target. A returned budget stop or cancellation retains the
last accepted state;
`converged`, `stagnated`, `max_evaluations` and numerical failures are distinct.
Do not accept a stopped scientific fit solely because it returned finite arrays.

Pass `result.checkpoint` to `refine(..., checkpoint=...)` or resume a loaded
project. Data, masks, background, identities, bounds, ties, support and numerical
controls must match. Runtime budgets may change; iteration continuation starts
from the accepted history count. Rejected attempts are discarded on restart.

The standalone `phasesmith-pawley` JSON format version 2 is shared by Rust and
Python and is distinct from the existing multi-histogram JSON+NPZ format. It
contains plain finite arrays and typed records; nullable bounds denote infinity.
Version-1 projects remain readable, including their original checkpoint digest.
Checkpoints bind the canonical request/options with SHA-256. Load checks byte
limits, version, unknown fields, identities, array and constraint contracts;
resuming also recomputes the saved objective. No pickle or matrix factorization
is stored. Saving never overwrites an existing file. Pawley analyses also integrate with the [shared native bundle](native-persistence.md#mixed-analyses-and-pawley)
format 7; use its `ProjectBundle` API to preserve multiple methods together.

Rust consumers use `phasesmith::workflows::{PawleyInput, refine_pawley}` and
`phasesmith::persistence::{PawleyProject, save_pawley_project, load_pawley_project}`.
The workflow has no Python dependency. A complete runnable native example is
`crates/phasesmith-workflows/examples/pawley.rs`.

## Validation and provenance

The independent NumPy implementation is `phasesmith.pawley_reference`.
Tests cover randomized CW/FCJ values and derivatives, analytical cell chains,
finite differences, an exhaustive small bounded least-squares oracle, exact
coincidences, explicit ties/dependent bounds, signed fits, masks, support
endpoints, area/centroid behavior, and exact checkpoint continuation.
The existing pinned GSAS-II CW fixture checks extracted area conventions;
a live pinned optimizer fixture now compares fixed states, fitted profiles,
cells, isolated areas and overlap sums. Declared model-agreement checks pass;
strict profile equivalence is not claimed.
See [Pawley validation and performance](pawley-validation.md) for measured gates
and [the implementation plan](pawley-plan.md) for the completed implementation scope.
[TOF Pawley](tof-pawley.md) provides the single- and multi-bank workflow.

Method: G. S. Pawley (1981), *J. Appl. Cryst.* **14**, 357–361,
[doi:10.1107/S0021889881009618](https://doi.org/10.1107/S0021889881009618).

## Fixed wavelength spectra

Pass `fixed_spectrum=WavelengthComponents([1.5406, 1.5444], [1.0, 0.5])`
to both `PawleyPhase.from_cell` (or `from_cif`) and `PawleyInput`. Component
zero must match the instrument reference wavelength. One fitted family area
multiplies the normalized weighted sum of all component profiles; the weights
are fixed effective **detected areas**, not source fluxes awaiting LP or
structure-factor corrections. Neither wavelength nor component ratios are
refined. Native generation takes the union of component-visible family domains;
one-component requests retain the monochromatic ordering and numerical result.
Each component uses its own Bragg position and composed U/V/W/X/Y widths.
Analytical cell and profile chains, bounds, matrix-free products and both
standalone/shared-bundle persistence retain this contract.

Every retained family must be Bragg-accessible at every supplied wavelength,
including zero-weight components, as required by the existing component kernel.
Inaccessible combinations fail explicitly. Reflection-domain margins and
finite support remain caller-visible; components are not renormalized to the
observed range. The support-local boundary certificate currently applies only
to monochromatic symmetric fixed-position fits. Spectral fits use ordinary
full-objective acceptance and may report stagnation at moving cutoffs.

`python -m phasesmith.validation.pawley_spectrum --data-root PATH --output REPORT`
compares the pinned low/middle/high-angle doublet profiles under their existing
FCJ tolerances and runs the measured Birmingham ceria regression with and without
the disclosed 1.6% secondary detected-area component. This is an exploratory
regression, not independent holdout qualification. At support 10000 the spectrum
fit converges at Rwp 0.20952 with exact repeated arrays/history. A support-100
probe stagnates at 0.20997 and is not labelled converged. The wide-support recipe
avoids moving cutoffs within these observations; it does not change the kernel's
support convention. The report retains both measured recipes and provenance.
