# Structural Rietveld refinement

The first structural refinement vertical slice is implemented in
`phasesmith.refinement.rietveld`. It is a script-first, monochromatic
constant-wavelength workflow. It accepts one or more typed `RietveldPhase`
objects and obtains every reflection intensity from their crystal structures;
GSAS-II is never imported or called.

For native applications, the first Python-free Rietveld boundary now lives in
the `phasesmith-workflows` Rust crate. It owns structural phases, experiment
state, fixed background composition, phase-resolved calculation results, and
residual metrics, so a future Tauri command can calculate and plot a complete
multiphase pattern without a Python process. The scripting optimizer described
below remains the production refinement orchestrator until its parameter,
solver, topology, runtime, and staged-recipe contracts have each migrated and
passed cross-interface validation.

The native boundary additionally provides stable phase/site parameter IDs,
setting-aware lattice bounds, deterministic symmetry-allowed special-position
coordinate bases, and exact transforms to the Rust structural JVP/VJP layout.
These are solver-neutral records: a desktop inspector can list and constrain
them without importing Python, while the remaining native solver work can use
the same identities without changing the command contract.

The native prepared objective applies those transforms directly to structural
JVP/VJP products and exposes the weighted gradient and damped normal operator
needed by a matrix-free solver. Masks, uncertainties, fixed background, and
stale-layout rejection are handled at this application-neutral layer, so a
Tauri command will not need to reproduce numerical objective rules.

The native workflow now includes a bounded structural solver with guarded
reflection topology.
It uses scaled conjugate gradients and deterministic backtracking over the
matrix-free objective, with shared cancellation, budgets, events, accepted
history, and restart checkpoints. Dynamic cells regenerate HKLs and
multiplicities after bounded motion and record added/removed families on
accepted history rows. Attached built-in sample-physics records are evaluated
again for the accepted cell and reflection list, rather than relying on stale
transferred arrays. Every accepted history row owns `Rp`, `Rwp`, chi-square,
and reduced chi-square alongside its objective and parameter/topology changes,
so application adapters do not need to replay accepted states to construct
complete result records.

An optional native `BackgroundModel` may already be attached to the request.
Its calculated values are added once to the pattern's fixed supplied background
and returned as the display-ready background series; coefficient selection and
motion are introduced by the general native parameter layout.

That general layout now covers CW U/V/W/X/Y, wavelength, zero shift,
Bragg--Brentano displacement, Debye--Scherrer X/Y displacement, every attached
background coefficient, built-in isotropic size, isotropic microstrain,
March--Dollase ratio, and all structural families. Sample parameters use stable
phase-owned keys. March--Dollase cell chains pass through the setting-aware
lattice transform, including all six reciprocal-metric derivatives. Its
complete JVP/VJP reuses matrix-free structural products and only stores the
small global, sample, and background columns. Wavelength installation updates
matching correction models and guarded domains atomically. The
constraint-aware solver entry point now consumes this complete layout.

`refine_general_rietveld` is the Python-free complete solver boundary. It
applies fixed, affine, and multi-source constraints through the exact
physical-to-scaled-free derivative matrix, freezes solver scales from the
initial contract, and rebuilds only accepted physical values and guarded
topology. Its checkpoint owns the complete accepted request, selection,
lattice bounds, constraints, fixed scale template, damping, and accepted
history, so continuation rejects changed contracts and unselected fixed state.
The final optional small-matrix diagnostic reports weighted Jacobian rank and
near-collinear free columns. It returns a full physical-parameter covariance
only when the scaled-free normal matrix is full rank; fixed and constrained
rows are propagated through the constraint derivative rather than inverted as
independent parameters. The native staged-recipe layer now consumes this
solver; Python façade delegation remains the next migration slice.

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

The implemented families are CW U/V/W/X/Y profile coefficients, constant zero
shift, Bragg--Brentano sample height, Debye--Scherrer X/Y displacement, an
additive grid-normalized polynomial background, phase scale, symmetry-independent
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

## Explicit and intelligent staged recipes

The application-neutral Rust boundary provides `RietveldStage`,
`RietveldRecipe`, `intelligent_rietveld_recipe`, and `run_rietveld_recipe`.
It validates every stage against the caller-authorized maximum selection and
the full initial constraint graph, then filters constraints only when their
target and all dependencies are active in that stage. Only policy-accepted
states advance. Intermediate stages skip covariance work, while the last
attempted recipe stage may produce the complete solver diagnostics. The same
cooperative cancellation token is observed by every stage. The optional
`RietveldRecipeSinks` boundary forwards structured events and complete accepted
checkpoints to a host such as Python or Tauri.

The solver always refines exactly the active parameters in one
`RietveldInput`; it never inserts a hidden sequence. Staging is an optional
workflow layer in `phasesmith.refinement`:

```python
from phasesmith.refinement import intelligent_rietveld_recipe, run_rietveld_recipe

# Advisory only: inspect this record before deciding whether to run it.
proposal = intelligent_rietveld_recipe(request)
print(proposal.to_record())

workflow = run_rietveld_recipe(request, proposal, options=options)
for stage in workflow.stages:
    print(
        stage.stage.name,
        stage.starting_rwp,
        stage.result.metrics.rwp,
        stage.result.termination_reason,
    )
```

The intelligent planner is deterministic and transparent. It can activate
only families already authorized by `request.selection`. Its default cumulative
advice establishes scale/background, aligns positions, stabilizes structural
intensities, and finally releases the remaining profile/sample parameters. Each
stage includes human-readable rationale and the resulting record lists every
active `ParameterKey`. Iteration exhaustion is not accepted by default.

Callers can construct `RietveldRecipe` and `RietveldStage` directly to change
the order, options, or acceptable termination reasons. This is the appropriate
place for experiment-specific judgment; the numerical solver remains
unchanged.

`RietveldProject.propose_intelligent_recipe()` returns advice without running
anything. `RietveldProject.refine_intelligently()` is the explicit opt-in that
plans and executes it, while `refine_recipe()` runs a caller-owned recipe.

`ConstantWavelengthExperiment.x_ray(...)` selects built-in non-resonant X-ray
scattering by default. `ConstantWavelengthExperiment.neutron(...)` selects the
built-in nuclear neutron model through the identical refinement API. The
workflow is explicitly monochromatic; wavelength-component refinement and TOF
structural refinement are later method extensions, not implicit mode switches.
For ordinary constant-wavelength neutron powder intensities, pass
`ConstantWavelengthNeutronLorentz(wavelength)` explicitly. It evaluates
`1/(sin(theta) sin(2 theta))` with analytical reciprocal-metric and wavelength
derivatives; a neutral correction remains available for controlled calculations.
For capillary/transmission data, attach `DebyeScherrerGeometry` to the
experiment and authorize `displace_x_micrometre` and
`displace_y_micrometre`. The intelligent planner places those parameters in
its position-alignment stage, but only when the caller selected them.

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

Persistence format 12 stores the pattern, refined experiment and background,
structural phases,
guarded domains, parameter selection, options, constraints, and checkpoint.
`PersistenceBundle.to_rietveld_input()` reconstructs the request; pass the
restored checkpoint to `rietveld.refine` for deterministic continuation.

## Multi-histogram boundary

One `RietveldInput` currently owns one observed pattern and one experiment.
The PbSO4 validation therefore runs independent X-ray and neutron fits and
labels its GSAS-II comparison accordingly. A scientifically equivalent joint
fit requires a first-class multi-histogram input: structure, cell, atomic
coordinates, occupancies, and displacement parameters are shared, while each
histogram retains its own radiation/scattering model, scale, background,
profile, limits, zero/displacement geometry, weights, and derivatives. The
combined objective must sum every histogram's residual norm before accepting a
trial. Alternating independent refinements is not treated as joint refinement.

This typed shared/local parameter ownership is the next Rietveld slice. Until
it exists, PhaseSmith does not claim that its independently fitted X/Y
displacements should equal values from GSAS-II's coupled X-ray/neutron fit.

## Extension boundary

Calculation-time reflection-physics providers remain explicit, batch-oriented
plugins: Python prepares one contribution array per reflection and Rust owns
the hot support accumulation. Structural JVP/VJP refinement currently requires
the built-in fused provider configurations whose full lattice/position chains
are defined. A third-party calculation provider is therefore not silently
treated as structurally differentiable. A future provider capability record
will advertise the additional structural derivative chains required for
refinement; until then unsupported configurations fail explicitly.

## Execution policy

`RietveldOptions.execution` is an immutable `ExecutionPolicy`. Its bounded
default is `threads=2`; GUI hosts and larger applications that already schedule
their own work can explicitly select `threads=1`. A positive integer selects a
fixed CPU budget; `threads=None` selects the available logical CPU count.
`minimum_parallel_tasks` defaults to two and avoids creating a pool for a
single independent phase.

The scheduler flattens phase/wavelength leaves and assigns the fixed budget
without nesting native pools. Single native leaves use internal reflection or
parameter-row parallelism where deterministic ownership is available. Native
calls release the GIL. Results return in input order and final additions remain
serial in that order, so values, analytical derivatives, accepted history, and
final metrics are bitwise identical between tested worker counts.

Cancellation remains cooperative at safe evaluation boundaries. An in-flight
native phase calculation completes before the pool closes; the refinement then
returns its last accepted checkpoint. No worker mutates shared phase state.

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

For the three-phase QARR workload, reusing guarded coordinate models across
trial states reduces the release median from 1.659 to 1.245 seconds on one
thread. `ExecutionPolicy(threads=2)` reduces it further to 0.961 seconds, a
1.30x multicore speedup. Three threads and automatic selection measure about
0.965--0.966 seconds because the three phase costs are unequal. This evidence
keeps two threads as the measured QARR choice and leaves within-phase Rust
tiling as a future single-phase benchmark task rather than nesting another
pool.

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
