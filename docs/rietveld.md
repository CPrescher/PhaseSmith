# Structural Rietveld refinement

The first structural refinement vertical slice is implemented in
`phasesmith.refinement.rietveld`. It is a script-first, monochromatic
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
from phasesmith import PowderPattern
from phasesmith.refinement import rietveld

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
from phasesmith import CancellationToken
from phasesmith.refinement import ConsoleRefinementLogger, rietveld

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

## Reference optimizer benchmark

`benchmarks/structural_refinement.py` is the realistic hot-loop benchmark for
this slice. On the development Apple Silicon host, the release wheel processed
423 reflections, eight sites, 20,001 samples, and 29 free parameters for three
matrix-free iterations (79 combined calculation/JVP/VJP evaluations) in
152.393 ms median over seven repetitions; the minimum was 145.738 ms. The
synthetic Rwp after those deliberately capped three iterations was
`2.197e-5`. This measures the engine optimizer, not a complete comparison with
another program.

The reusable native-linearization checkpoint reduces that reviewed median from
148.7 ms to 38.3 ms, model evaluations from 79 to 4, and preserves final Rwp to
2.6e-11 absolute. Built-in paths materialize one parameter-major Rust Jacobian
per accepted or trial state and reuse it for every optimizer JVP/VJP.

The default memory ceiling is 10,000,000 floating-point elements.
RietveldOptions.max_linearization_elements set to zero forces the matrix-free
path; requests above the ceiling and third-party physics providers fall back
automatically. QARR stage 2 uses the bounded reusable linearization. After the
fixed-anisotropic and adaptive-FCJ checkpoints, three repeated release runs
produced exactly matching scientific records, reduced stage-2 evaluations from
the 1,500 matrix-free budget to one linearization per accepted/trial state, and
improved the final Rwp. The earlier validation-only matrix-free override is
therefore removed rather than retained as an unmeasured compatibility path.

The separately validated pinned GSAS-II structural benchmark remains the
like-for-like speed comparison documented in `gsasii-performance.md`. The
locally installed GSAS-II checkout currently has revision `e88e61f`, not the
pinned `c0bc79`, so the runner correctly refused a new comparison rather than
publishing an unreviewed ratio.

Preferred-orientation parameters, refinement of symmetry-constrained
anisotropic tensors, wavelength components, magnetic scattering, and TOF
structural refinement are intentionally outside this first slice. Fixed CIF
anisotropic tensors are calculated directly. The remaining capabilities can be
added as typed parameter families without changing the accepted-state runtime
or crystal-structure ownership model.
