# TOF Pawley refinement

`phasesmith.refinement.tof_pawley` fits independent reflection-family areas on
one or more neutron time-of-flight banks. The Rust workflow owns the objective,
analytical derivatives, bounded least-squares steps and accepted-state restart.
No atoms, structure factors or GSAS-II runtime are required.

## Data and area conventions

Each bank owns an increasing finite `tof_us` array, observed **density per
microsecond**, optional positive one-sigma density uncertainties, mask and
fixed background. Nonuniform sampling is supported directly. Convert bin
counts and their uncertainties to densities together before constructing the
request; the native TOF importer preserves the relevant bin boundaries.
`normalization` is required provenance text describing those external steps.
No incident-spectrum, multiplicity, Lorentz, phase-scale or structure-factor
amplitude is applied inside Pawley. These factors are absorbed in each fitted
bank-local family area, whose unit is observed density times microseconds.
Areas from different banks are independent by default.

The objective is one half the sum of squared residuals over every included
sample in every bank. Residuals are `(calculated-observed)/sigma`, or unweighted
when uncertainties are absent or `use_uncertainty=False`. With uncertainty
weighting enabled, either all banks supply sigmas or none do; mixed availability
is rejected. Bin widths are not additional least-squares weights. Negative
observations remain valid; fitted areas are nonnegative unless explicitly
requesting `signed_intensities=True`.

## Minimal extraction

This self-contained example creates two synthetic peaks and recovers their
areas from zero. Replace the synthetic observations with imported densities
for a real experiment.

```python
from dataclasses import replace
import numpy as np
from phasesmith import TofInstrument
from phasesmith.pattern import TofPowderPattern
from phasesmith.refinement.pawley import PawleyOptions
from phasesmith.refinement.tof_pawley import (
    TofPawleyBank,
    TofPawleyInput,
    TofPawleyPhase,
    TofPawleyProject,
    calculate,
)

instrument = TofInstrument(
    0,
    5000,
    0,
    0,
    0.18,
    0.04,
    0,
    0,
    1,
    10,
    0,
    0,
    0.3,
    0,
    0.4,
)
x = np.linspace(7000, 12000, 1001)
phase = TofPawleyPhase("sample", ("family-a", "family-b"), [1.7, 2.0], [30, 70])
bank = TofPawleyBank(
    "bank1",
    TofPowderPattern(x),
    instrument,
    (phase,),
    "Synthetic density per microsecond; no incident-spectrum correction",
)
truth = calculate(TofPawleyInput((bank,))).calculated_y
bank = replace(
    bank,
    pattern=TofPowderPattern(x, observed_y=truth),
    phases=(replace(phase, intensities=[0, 0]),),
)
project = TofPawleyProject(
    TofPawleyInput((bank,)),
    PawleyOptions(solver="matrix_free"),
)
fit = project.refine()
assert fit.termination_reason == "converged"
np.testing.assert_allclose(fit.intensities, [30, 70], atol=1e-7)
```

For Python-free use, run:

```sh
cargo run -p phasesmith-workflows --example tof_pawley
```

The native entry points are `TofPawleyInput`, `evaluate_tof_pawley`,
`refine_tof_pawley` and `refine_tof_pawley_with_runtime` in
`phasesmith::workflows`. Dense and matrix-free modes use the same physical
objective. Matrix-free storage holds support blocks and evaluates JVP/VJP
without a samples-by-all-parameters dense Jacobian; its iterative steps retain
bounds and exact constraints. `max_elements` guards workspace allocation.

## Shared cells and bank-local parameters

Add banks to `TofPawleyInput.banks` for one joint fit. Their `bank_id` values must
be unique. Use a consistent `phase_id` across banks for a shared cell, and pass
`TofSharedLatticePhase` records through `shared_lattice`. Each record supplies a
`LatticeParameterization`, explicit `LatticeParameterBounds` and initial cell.
Matching phases must provide integer HKLs. Without a shared cell, supplied
d-spacings remain fixed. Reflection lists and family identities never change
during a fit or restart.

Generate indexed families with the existing
`PreparedReflectionGenerator(space_group).generate(cell, DSpacingRange(...))`.
Choose the d-spacing domain to cover the full allowed-cell and profile-tail
margins for every bank; this explicit TOF constructor does not automatically
expand or regenerate your list. Preserve systematic absences and one entry per
powder family. Bank and phase IDs cannot contain `:` because their composite
parameter owner is `bank_id:phase_id`.

Select parameters with:

```python
from phasesmith.refinement.tof_pawley import build_parameter_set

request = project.input
parameters = build_parameter_set(
    request,
    profile_parameters={"bank1": ("zero",)},
    lattice=False,
)
request = replace(request, parameters=parameters)
```

Set `lattice=True` on a request with shared cells to select symmetry-independent
cell parameters. Area and supplied Chebyshev background coefficients are free
by default; calibration/profile coefficients and cells are initially fixed.
Profile selection names are `zero`, `difc`, `difa`, `difb`, `alpha`, `beta0`,
`beta1`, `betaq`, `sigma0`, `sigma1`, `sigma2`, `sigmaq`, `x`, `y`, `z`.
Units follow `TofInstrument`; the calibration is
`TOF = zero + DIFC*d + DIFA*d² + DIFB/d` in microseconds.
`TofPawleyBackground` requires an explicit microsecond Chebyshev domain.

When a shared cell's lengths vary, keep DIFC fixed in at least one bank
containing that phase. The workflow
rejects an all-free DIFC/cell scale gauge; unrelated banks cannot anchor it. This anchor is a necessary modeling
policy, not a promise that every remaining coefficient is identifiable.
Calibration must increase over the retained d-spacing interval, and every
trial must produce valid positive composed profile widths and tail rates.
Invalid trials are rejected. Add scientifically appropriate explicit coefficient
bounds when selecting strongly correlated terms. TOF uses physical parameter
bounds; the CW-specific `active_width_bounds` diagnostic is not a TOF composed-
width active-set certificate.

Result sample arrays are concatenated in bank order; `request.sample_offsets`
gives their boundaries. Area and position arrays follow bank, phase, then
family order. The reused result type's `positions` are **microseconds** here.
Dense mode provides scoped numerical-rank and covariance diagnostics; matrix-free
mode explicitly omits global rank/covariance. Exactly coincident families are
reported within each bank, and individually unobserved columns retain their
accepted values. Overlap can make individual areas non-identifiable even when
the calculated profile is excellent.

## Finite support and derivatives

The existing native TOF kernel convolves the symmetric TCH basis with a
normalized, truncated back-to-back exponential. The outer support is inclusive:
`[position - support_fwhm*H - tail_log/alpha,
position + support_fwhm*H + tail_log/beta]`. Values outside are zero. The retained
symmetric TCH area is not renormalized to one, and finite support is not
renormalized to the observed range. Area and centroid tests use these conventions.

Values and analytical area, profile, calibration and shared-cell derivatives
are accumulated together. Independent NumPy quadrature, all-coefficient finite
differences, JVP/VJP and adjoint tests validate the chains. The converged NumPy
reference differs from the existing native quadrature; the deterministic test's
per-column normalized derivative tolerance is explicitly `2e-6`.

Derivatives hold support membership fixed away from cutoffs. TOF does not use
the CW fixed-position hard-support stationarity certificate: it accepts steps
against the complete finite-support objective and can report stagnation if
moving cutoffs prevent progress. Check the termination reason instead of
interpreting every returned fit as converged.

## Restart and shared bundles

`project.refine(cancellation=token, progress=callback, ...)` updates only an
accepted checkpoint. Budget or cancellation termination retains a resumable
joint state. `project.save("sample.tof-pawley.json")` writes standalone TOF
format 1 atomically and refuses to overwrite an existing path. `load` validates
the full scientific request digest, bounds, solver controls and accepted history.
Changing data, calibration, normalization, topology or support invalidates a
checkpoint. Runtime budgets may change on continuation.

```python
from phasesmith.project_bundle import ProjectBundle

bundle = ProjectBundle.from_tof_pawley(project, analysis_id="joint")
bundle.save("sample.psproj")
restored = ProjectBundle.load("sample.psproj")
analysis = restored.tof_pawley("joint")
result = analysis.refine()
restored.with_tof_pawley("joint", analysis).save("resumed.psproj")
```

Native format 7 stores observations once in shared TOF histograms and retains
all six analysis families. Formats 1–6 migrate with no TOF Pawley analyses.
Use the full bundle API to preserve other methods; method-specific views select
only their own family. The standalone schema is
[`tof-pawley-project-v1.schema.json`](https://github.com/CPrescher/PhaseSmith/blob/main/schemas/tof-pawley-project-v1.schema.json).

## Scientific coverage

The predeclared TOF acceptance manifest reuses established import, background,
calibration and quality conventions. POWGEN LaB6 extraction converges with
330 families at Rwp 0.21680 and correlation 0.96900. Joint LANL nickel fits
three banks, their independent zero offsets/areas/backgrounds and one shared
cubic cell: joint Rwp 0.02099, every bank below 0.03, and a = 3.523669 Å within
the existing 0.0005 Å reference tolerance. Repeated arrays and histories are exact.
These are measured regressions, not automatic structure validation.

The three pinned GSAS-II TOF profile fixtures pass through this Pawley objective
and area extraction under their existing `2.5e-4` tolerance. A live external
TOF Pawley optimizer comparison is not claimed; the live optimizer fixture is
CW. See [validation](pawley-validation.md) for reproducible commands and reports.
