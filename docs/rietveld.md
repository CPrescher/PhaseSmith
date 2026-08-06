# Structural Rietveld refinement

The first structural refinement vertical slice is implemented in
`rietveld.refinement.rietveld`. It is a script-first, monochromatic
constant-wavelength workflow. It accepts one or more typed `RietveldPhase`
objects and obtains every reflection intensity from their crystal structures;
GSAS-II is never imported or called.

## Objective and numerical method

For included observations `i`, the residual used by the optimizer is

```text
r_i = (Ycalc_i - Yobs_i) / sigma_i
```

when uncertainties are enabled and supplied, and `Ycalc_i - Yobs_i`
otherwise. The minimized objective is `0.5 sum_i r_i^2`. A mask value of
`False` removes that sample from both the objective and all derivative
products.

The optimizer is damped Gauss--Newton. It solves

```text
(J^T J + lambda I) delta = -J^T r
```

with deterministic conjugate gradients. It never constructs the dense
sample-by-parameter Jacobian. Each matrix product calls the Rust structural
pattern JVP and VJP; peak accumulation remains fused and exactly
support-limited. Bounds are applied in scaled free-parameter coordinates and
trial states are installed only after their complete pattern improves the
objective.

The implemented families are CW U/V/W/X/Y profile coefficients, an additive
grid-normalized polynomial background, phase scale, symmetry-independent
lattice values, symmetry-allowed fractional coordinates, occupancy, and
isotropic displacement `U_iso` in square ångströms. Lattice trials regenerate
the conservative guarded reflection topology by stable Miller-family ID.
General positions expose `x`, `y`, and `z`. Special positions expose only the
null-space coordinates allowed by their exact site stabilizer; a fixed special
position exposes no coordinate parameter.

Fixed, affine, and multi-source linear constraints use `ParameterKey`
identities. The exact
physical-to-scaled-free derivative of a constraint graph is shared across
refinement methods and is applied before every native JVP and after every VJP.

## Minimal CIF script

```python
from rietveld import PowderPattern
from rietveld.refinement import rietveld

observed = PowderPattern(
    two_theta,
    observed_y=intensity,
    uncertainty=sigma,
    mask=included,
    background=background,
)
request = rietveld.RietveldInput.from_cif(
    observed,
    experiment,
    "phase.cif",
    phase_id="alpha",
    selection=rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=True,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2"),
        background=True,
    ),
    background=rietveld.PolynomialBackground("main", (0.0, 0.0, 0.0)),
)
result = rietveld.refine(request)
print(result.termination_reason, result.metrics.rwp)
print(result.phases[0].structure.cell)
```

`ConstantWavelengthExperiment.x_ray(...)` selects built-in non-resonant X-ray
scattering by default. `ConstantWavelengthExperiment.neutron(...)` selects the
built-in nuclear neutron model through the identical refinement API. The
workflow is explicitly monochromatic; wavelength-component refinement and TOF
structural refinement are later method extensions, not implicit mode switches.

Multiple phases are a tuple in one `RietveldInput`. Background is owned by the
pattern and is added once after summing phase profiles. Phase IDs, site IDs,
and reflection-family IDs remain durable across calculations, logs,
constraints, and checkpoints.

## Safety, logging, and recovery

`RietveldOptions.limits` bounds iterations, total model products, wall time,
and consecutive rejections. `refine` accepts a thread-safe cancellation token,
a structured logger, and a checkpoint callback. Cancellation and recoverable
budgets return the last accepted state. Every accepted step emits a complete
`RietveldCheckpoint`; an unexpected exception attempts an emergency callback
with the last accepted state and then preserves the original traceback.

```python
from rietveld import CancellationToken
from rietveld.refinement import ConsoleRefinementLogger, rietveld

stop = CancellationToken()
result = rietveld.refine(
    request,
    cancellation=stop,
    logger=ConsoleRefinementLogger(),
    checkpoint_callback=save_checkpoint,
)
```

The result includes accepted parameter changes, residual metrics, evaluation
count, termination message, and the final checkpoint. For a small enough free
parameter set it also builds the parameter-space normal matrix, reports its
rank and nearly collinear columns, and returns physical-parameter covariance
only when the normal matrix has full rank. It does not report a misleading
inverse for a singular problem.

Persistence format 4 stores the pattern, refined experiment and background,
structural phases,
guarded domains, parameter selection, options, constraints, and checkpoint.
`PersistenceBundle.to_rietveld_input()` reconstructs the request; pass the
restored checkpoint to `rietveld.refine` for deterministic continuation.

## Extension boundary

Calculation-time reflection-physics providers remain explicit, batch-oriented
plugins: Python prepares one contribution array per reflection and Rust owns
the hot support accumulation. Structural JVP/VJP refinement currently requires
the built-in fused provider configurations whose full lattice/position chains
are defined. A third-party calculation provider is therefore not silently
treated as structurally differentiable. A future provider capability record
will advertise the additional structural derivative chains required for
refinement; until then unsupported configurations fail explicitly.

Preferred-orientation parameters, anisotropic displacement, wavelength
components, magnetic scattering, and TOF structural refinement are
intentionally outside this first slice. They can be added as typed parameter
families without changing the accepted-state runtime or crystal-structure
ownership model.
