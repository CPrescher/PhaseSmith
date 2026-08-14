# Operating a refinement safely

This guide starts after a validated [`crate::workflows::RietveldInput`] has
produced a plausible forward calculation. The scientific model and parameter
selection remain the caller's responsibility; the workflow layer supplies a
bounded deterministic solve and complete accepted-state records.

## Walkthrough 1: release parameters in stages

Refinement is usually more stable when parameters are released cumulatively:

1. establish phase scales and a small analytical background;
2. align instrument/position terms that the experiment actually supports;
3. release lattice and structural intensity terms;
4. release profile and sample-physics terms only after peak positions and
   intensities are credible.

Construct [`crate::workflows::RietveldParameterSelection`] directly when you
already know the sequence. Its structural flags cover lattice, symmetry-allowed
coordinates, occupancy, isotropic U, and phase scale. Its instrument list uses
typed [`crate::workflows::RietveldInstrumentParameter`] values. Background and
sample-physics selections apply only to analytical/native models already
attached to the input.

For application-generated advice, call
[`crate::workflows::intelligent_rietveld_recipe`]. It returns an inspectable
[`crate::workflows::RietveldRecipe`]; it does not run anything. The proposal is
bounded by a caller-authorized maximum selection and cannot introduce a new
parameter family. Show the stage names, rationale, and active keys before
passing the accepted recipe to [`crate::workflows::run_rietveld_recipe`].

After each stage, use its returned accepted input as the next starting point
only when the stage termination policy accepts it. A decreasing Rwp is useful
but insufficient: inspect parameter bounds, correlations, difference curves,
and whether a newly released family merely compensated another modelling
error.

## Walkthrough 2: set numerical and work limits

[`crate::workflows::RietveldRefinementOptions`] separates convergence controls
from hard [`crate::workflows::RefinementLimits`]. A typical construction is:

```no_run
use phasesmith::execution::ExecutionPolicy;
use phasesmith::workflows::{
    RefinementLimits, RietveldCalculationOptions, RietveldRefinementOptions,
};

let calculation = RietveldCalculationOptions::new(
    30.0, // exact finite support: 30 FWHM on each peak
    true, // use supplied one-sigma uncertainties
    ExecutionPolicy::new(Some(1), 256)?,
)?;
let options = RietveldRefinementOptions::new(
    calculation,
    RefinementLimits::new(40, 3_000, Some(120.0), 100)?,
    3,       // minimum accepted iterations before convergence
    1.0e-7,  // relative objective tolerance
    1.0e-8,  // scaled parameter-step tolerance
    1.0e-6,  // initial damping
    10.0,    // damping increase after rejection
    0.3,     // damping decrease after acceptance
    1.0e-8,  // conjugate-gradient tolerance
    40,      // CG iterations per attempted step
    0.15,    // maximum scaled step norm
    12,      // half-step backtracks
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

These values mirror the shape of the real PbSO4 workload, not universal
defaults. Finite profile support is observable numerical behavior. Changing it
can change normalization tails, derivatives, and the accepted path, so treat it
as part of the analysis contract. Worker count is also explicit; application
hosts should normally give one outer task a bounded worker budget.

A limit stop or cancellation is a normal result containing the last accepted
state. It is not the same as convergence. Numerical/domain failures return an
error instead of inventing a partial state.

## Walkthrough 3: connect events, cancellation, and checkpoints

The convenience solver accepts an optional
[`crate::workflows::CancellationToken`]. For progress events and durable
checkpoints, create a typed [`crate::workflows::RefinementRuntime`] and call the
matching `_with_runtime` function.

```ignore
let stop = CancellationToken::default();
let mut runtime = RefinementRuntime::<RietveldGeneralCheckpoint>::new(
    options.limits,
    Some(stop.clone()),
)?;
runtime.set_event_sink(|event: &RefinementEvent| {
    println!("{}: {}", event.kind().as_str(), event.message());
    Ok(())
});
runtime.set_checkpoint_sink(|checkpoint: &RietveldGeneralCheckpoint| {
    save_checkpoint(checkpoint)?;
    Ok(())
});

let result = refine_general_rietveld_with_runtime(
    &input,
    &selection,
    &lattice_bounds,
    &constraints,
    &options,
    covariance,
    None,
    &mut runtime,
)?;
```

Callbacks run synchronously at orchestration boundaries. Keep them short and
move UI or network work onto the host's own queue. Event messages are for
humans; branch on [`crate::workflows::RefinementEventKind`] and structured
diagnostics. A failing event sink is detached and recorded rather than allowed
to corrupt numerical state. A checkpoint sink receives a complete accepted
checkpoint, never a half-installed trial.

Calling [`crate::workflows::CancellationToken::request`] from another thread
requests a cooperative stop and records the first non-empty reason. An
in-flight calculation reaches its safe boundary first. The result's termination
reason records cancellation and its checkpoint remains the last accepted
model.

## Walkthrough 4: continue the exact accepted contract

Pass `Some(&previous.checkpoint)` to the same solver family. Continuation
validates the request, phase/site identities, selected parameters, bounds,
constraints, and fixed state. A changed contract is rejected instead of being
silently merged.

```ignore
let continued = refine_general_rietveld(
    &previous.input,
    &selection,
    &lattice_bounds,
    &constraints,
    &longer_options,
    covariance,
    Some(&previous.checkpoint),
    None,
)?;
```

Increase limits explicitly when continuing after a budget stop. For a
caller-owned runtime, call `resume_accepted(checkpoint.completed_iterations)`
before the `_with_runtime` solver. Evaluation and rejection budgets restart
for the continuation call; accepted iteration numbering continues.

Persist the checkpoint together with its full input and analysis contract via
[`crate::persistence`]. A checkpoint alone is not a portable project file: it
does not authorize guessing the observed pattern, parameter selection,
constraints, or external providers.

## Walkthrough 5: interpret the final result

[`crate::workflows::RietveldGeneralRefinementResult`] returns the display-ready
calculation, accepted input, complete physical parameters, free-key order,
accepted history, termination category, checkpoint, and evaluation count. When
requested and small enough, it also returns the rank of the final weighted
normal matrix, near-collinear key pairs, and physical covariance.

Covariance is returned only for a full-rank free system. Fixed and constrained
physical rows are propagated through the constraint derivative. If covariance
is absent or correlations are unresolved, report that fact; do not substitute
zero uncertainty or invert a singular matrix in presentation code.

Use the returned `calculation.y`, `profile_y`, `background_y`, and per-phase
components for plots. Use `result.input` for the accepted physical model. Use
`history` for an audit table, not to recreate state. Always report the
termination reason and weighting convention alongside Rwp or chi-square.

For a complete measured-data construction leading into these operations, see
[`crate::guide::real_data_rietveld`].
