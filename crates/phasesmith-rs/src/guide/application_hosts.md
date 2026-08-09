# Desktop, service, and CLI integration

`PhaseSmith` is synchronous native Rust. An application host owns scheduling,
presentation state, authorization, and transport DTOs; `PhaseSmith` owns
scientific state and computation.

## Recommended host state

Create one bounded execution policy and share it with prepared models and
workflow requests. Fixed policies cap themselves to the logical CPUs available
to the process.

```
use phasesmith::execution::ExecutionPolicy;

#[derive(Debug)]
struct ScientificService {
    execution: ExecutionPolicy,
}

let service = ScientificService {
    execution: ExecutionPolicy::new(Some(4), 2)?,
};
assert!(service.execution.resolved_budget() >= 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Tauri or another GUI toolkit

A Tauri command should translate its serializable request into validated
[`crate::model`] or [`crate::workflows`] values, run the synchronous operation
off the UI thread, and translate the result into a presentation DTO. Store
Tauri handles, window identifiers, and event-channel types in the application
crate—not in `PhaseSmith`.

For long refinements, connect application cancellation and progress delivery
through [`crate::workflows::RefinementRuntime`]. Do not cancel by abandoning a
thread while it is writing a project; use the workflow cancellation contract
and persist only validated checkpoints.

## Persistence

Use [`crate::persistence::save_project`] and
[`crate::persistence::load_project`] for calculation/domain projects, or their
Rietveld counterparts for runnable refinement state. The bundle is a directory
containing a versioned `manifest.json` and typed `arrays.npz`. Read limits are
part of the load call and should remain explicit at trust boundaries.

Application settings, recent-file lists, window layout, credentials, and other
presentation state belong in the host's own storage.

## Python is optional

The Python scripting interface can coexist with a native application, but it
is not on the application's dependency path. Native hosts should depend on
`phasesmith`, not `phasesmith-py`, and do not need to ship `CPython`.
