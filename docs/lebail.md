# Le Bail extraction

`phasesmith.refinement.lebail` is the first complete refinement workflow. It
consumes `PowderPattern`, `ConstantWavelengthInstrument`, and one or more
identified `Phase` objects through the same calculation API used by ordinary
scripts.

## Redistribution equation

For reflection `k`, current integrated intensity `I_k`, normalized sampled
profile `p_ki`, observed intensity `y_i`, and supplied background `b_i`, one
redistribution step is

```text
P_i      = sum_j I_j p_ji
r_i      = max(y_i - b_i, 0) / P_i
I'_k     = I_k * sum_i(w_i p_ki r_i) / sum_i(w_i p_ki)
```

Only included mask samples and the exact finite support block of a reflection
participate. Bin-width integration weights are always used. If uncertainties
are enabled, `w_i` additionally contains `1 / uncertainty_i**2`. A damping
factor may interpolate between `I_k` and `I'_k`. Intensities remain
non-negative.

Samples whose calculated profile is below `minimum_calculated` do not form a
ratio. An explicit reflection with no included support is set to zero and
reported in the iteration warnings. A generated guard-only reflection retains
its current intensity so it can cross into the visible interval without silent
information loss.

Exactly coincident profiles preserve their starting intensity ratio because
their multiplicative factors are identical. Set
`LeBailOptions(diagnose_rank_deficiency=True)` to run the quadratic
support-block correlation analysis and return unresolved groups with stable
`(phase_id, reflection_id)` keys and numerical rank. It is opt-in so ordinary
large reflection sets do not pay an O(N²) diagnostic cost.

## Short script

```python
from phasesmith.refinement import lebail

request = lebail.LeBailInput(
    pattern=pattern,  # observed_y and background are explicit arrays
    instrument=instrument,
    phases=(alpha, beta),  # reflection IDs remain stable
)
result = lebail.refine(request)

print(result.termination_reason, result.metrics.rwp)
for value in result.intensities:
    print(value.phase_id, value.reflection_id, value.integrated_intensity)
```

Starting intensities come from each `ReflectionBatch`. If every supplied value
is zero, initialization divides the non-negative integrated net pattern area
equally and applies a small positive floor.

## Optional profile updates

`build_parameter_set` selects any combination of CW `U/V/W/X/Y`, phase scales,
individual reflection positions, or symmetry-independent lattice variables
without requiring users to pack a vector. Lattice parameters and independent
reflection positions are deliberately mutually exclusive.
Instrument coefficients are not individually forced positive: their composed
Gaussian variance and Lorentzian width must be valid over the evaluated
reflection range. Phase scales are non-negative and positions remain in
`(0, 180)` degrees.

```python
parameters = lebail.build_parameter_set(
    instrument,
    (alpha,),
    instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"),
    reflection_positions=True,
)
result = lebail.refine(
    lebail.LeBailInput(pattern, instrument, (alpha,), parameters),
    lebail.LeBailOptions(max_iterations=40),
)
```

Intensity extraction alternates with bounded, damped analytical profile steps.
Rejected invalid-width or non-improving steps are shortened by backtracking.
Covariance is returned only when the final selected profile Jacobian has full
rank and its interpretation is valid. Phase scale and freely extracted
intensities are inherently correlated; a selected phase scale therefore emits
an explicit identifiability warning and covariance is not reported for that
parameterization.
The same uncertainty convention is used for covariance as in Rietveld:
supplied active uncertainties yield the unscaled inverse weighted normal
matrix, while unit-weight fits scale it by the final reduced chi-square.
Phase and reflection IDs cannot contain `/`, which keeps the documented
`phase/reflection` parameter identities unambiguous.

CIF-backed lattice refinement, including the one-call constructor and guarded
reflection-domain semantics, is documented in
[lattice-refinement.md](lattice-refinement.md).

The application-neutral Rust workflow now implements the same bounded lattice
path without importing or embedding Python. It owns setting-aware lattice
parameters, analytical pattern columns, accepted-step reflection regeneration,
stable-ID intensity transfer, structured topology warnings, and compatible
checkpoint continuation. The Python module remains the public scripting façade
and numerical differential oracle until its built-in orchestration delegates to
this native workflow.

## Checkpoints and diagnostics

Every result contains an immutable checkpoint. Passing it back to `refine`
continues from the next iteration and is bitwise-equivalent to an uninterrupted
run for the same input and options. The result also carries final `Ycalc`,
background, phase/reflection identities, residual metrics, history, warnings,
termination reason, unresolved groups, parameters, and covariance where valid.

For notebook loops and custom application scheduling,
`lebail.iterate_once(request, options, checkpoint=previous)` advances exactly
one iteration and returns the ordinary result/checkpoint types. This is the
same implementation used by the one-call workflow, not a second orchestration
path.

The pinned `oracle/fixtures/lebail_v1` case uses GSAS-II only in an external
environment. It compares profile parameters, grouped extracted intensities,
`Ycalc`, residual improvement, and convergence trend. The reflection-table
conversion to this library's degree-density integrated intensity is explicitly
`0.01 * Fobs^2 * intensity_correction`.
