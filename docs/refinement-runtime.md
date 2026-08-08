# Refinement runtime, logging, cancellation, and recovery

This contract applies to every iterative refinement method. Numerical methods
may add diagnostics, but they may not bypass the safety boundaries described
here.

## Ownership boundaries

The numerical refinement function never reads standard input, changes terminal
mode, installs signal handlers, configures Python's global logger, or exits the
process. It accepts plain callback protocols and a thread-safe cancellation
token. A terminal helper, notebook, GUI, or remote service owns user input and
requests cancellation through that token.

The optional terminal controller maps `q` to a graceful stop. The first
`Ctrl+C` also requests a graceful stop; a second `Ctrl+C` invokes Python's
previous interrupt handler. Signal and terminal state are restored when the
controller exits. A GUI Stop button calls the same token without importing any
terminal module.

Cancellation is cooperative at declared orchestration boundaries, never once
per atom, reflection, or pattern sample. The maximum cancellation latency is
therefore one bounded native calculation or derivative product.

The same safety shell now exists in the Python-free `phasesmith-workflows`
crate. Its cloneable cancellation token uses first-request-wins shared state,
publishes the reason before the atomic requested flag, and can be owned by a
Tauri command registry while numerical work runs on another thread. Terminal
and signal handling remain presentation adapters and are intentionally not
part of the native workflow crate.

## Structured event stream

`RefinementEvent` is an immutable plain-data record. It includes:

- event kind and method stage;
- attempted and accepted iteration counts;
- model-evaluation count;
- monotonic elapsed time;
- a short stable message;
- finite string, Boolean, integer, or floating-point diagnostics.

Events cover start, trial, accepted/rejected steps, iteration summaries,
checkpoint completion, warnings, termination, and failures. Human-readable and
JSON-lines loggers consume the same records. They write only to caller-supplied
streams and never configure process-global logging.

The native runtime exposes the same finite event categories and diagnostics
through a synchronous sink trait. A failing event sink is detached and its
message recorded without invalidating numerical state, which lets a desktop
adapter forward events without making webview delivery part of the solver.

The numerical result separately retains deterministic iteration history.
Wall-clock timestamps and presentation text are deliberately not part of that
restart state.

## Bounded execution

Every refinement has finite limits for iterations, model evaluations,
backtracking attempts, and consecutive rejected steps. Runtime may additionally
have a finite wall-clock limit. At every boundary the runtime checks, in order:

1. cancellation;
2. elapsed-time budget;
3. evaluation budget;
4. method-specific divergence, stagnation, and physical-validity conditions.

Trial parameters are never installed as current state before their model has
passed validation and improved the selected objective. Non-finite kernel
outputs, non-finite derivatives, invalid physical models, and worsening trial
steps are rejected. Repeated failure terminates with a typed reason instead of
looping indefinitely.

Invalid initial input raises before refinement starts. Cancellation and
recoverable numerical termination return an ordinary result containing the
last accepted model. An unexpected internal exception triggers an emergency
checkpoint callback with the last accepted state and is then re-raised with
its original traceback.

## Checkpoint durability

The method emits a complete immutable checkpoint after every accepted
iteration. Checkpoint callbacks run only after the current state and history
agree. A filesystem implementation writes a new generation completely,
flushes it, and only then atomically advances a small current-generation
pointer. It never overwrites the sole recoverable generation in place.

Restart validates method version, parameter identities and bounds, experiment,
phase/reflection-domain contracts, array hashes, and accepted history before
performing another calculation. Resuming an interrupted deterministic run must
produce the same accepted-state sequence as running continuously.

Native checkpoint sinks are typed by the workflow's checkpoint record. The
runtime increments accepted state before calling the sink; a sink error is
returned structurally while accepted counters remain advanced. Continuations
restore attempted/accepted counts only before new work and restart evaluation
and rejection budgets, matching the existing workflow policy.

## Termination reasons

Public results distinguish at least:

- `converged`;
- `cancelled`;
- `max_iterations`;
- `max_runtime`;
- `max_evaluations`;
- `stagnated`;
- `diverged`;
- `repeated_rejections`;
- `numerical_failure`.

These are outcomes, not generic exceptions. Input errors and unexpected
programming failures remain exceptions so applications cannot accidentally
present them as scientifically valid convergence.

## Required validation

Tests use fake clocks and injected failures; they do not depend on timing races.
They cover cancellation before the first iteration, cancellation after an
accepted iteration, first/second interrupt behavior, `q`, callback isolation,
every execution budget, rejected/non-finite trials, checkpoint callback
failure, restart equivalence, and restoration of signal/terminal state.
The native suite additionally launches cancellation from another OS thread and
runs a configured stop-sequence differential check against the Python runtime.
