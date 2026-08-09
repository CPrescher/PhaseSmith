# Python-free desktop adapter

`phasesmith-desktop` is the application-state boundary intended for the Tauri
host. It depends on the native model, workflows, and persistence crates, and it
does not depend on Tauri, a webview, PyO3, NumPy, or CPython. This keeps the
scientific and project-state contract independently testable and lets the final
Tauri crate remain a thin command/event translation layer.

## Revision contract

`DesktopProjectStore` owns one optional `Arc<RietveldProjectState>` behind a
short-lived read/write lock. `snapshot()` returns an immutable shared snapshot;
numerical work and file I/O operate after releasing that lock.

Edits use compare-and-swap semantics. `replace_project()` checks the caller's
revision and increments it exactly once. Background work retains the original
`ProjectSnapshot` and completes through `replace_snapshot()`, which compares
both the numeric revision and the exact `Arc` identity. The identity check
prevents a late result from overwriting a project that was closed or reopened
at the same revision. Failed validation and stale updates never mutate current
state.

Creation, native load, summary, exact-revision save, replacement, and close are
workflow-sized methods with serializable stable error records. Native load is
fully validated before installation; failed load preserves the current project.
The job layer builds event delivery, cancellation, and accepted-result
installation on top of these immutable snapshots.

## Refinement jobs and events

`JobManager` runs each native Rietveld analysis on a named Rust worker thread.
Every progress/completion/failure event carries the process-local job ID,
histogram ID, and immutable source revision. Native event diagnostics are
translated into finite JSON scalar records; non-finite fit metrics are omitted
rather than producing invalid JSON. A failed presentation event sink is
isolated by the native runtime and cannot invalidate accepted numerical state.

Completion only retains a result. `accept_refinement()` is a separate command
that updates the analysis, histogram experiment, native phase definitions, and
checkpoint through the exact source-snapshot compare-and-swap boundary. A
newer edit or reopened project therefore wins deterministically. Shared phases
across multiple histograms are rejected here until the joint objective owns the
required shared/local parameter split. `cancel_refinement()` uses the native
first-reason-wins token, and `discard_job()` releases retained result arrays.

## Binary display series

Large plotting arrays never enter JSON command/event payloads. Project and
completed-job catalog commands return `BinarySeriesDescriptor` records with a
stable series ID, semantic role, physical unit, owner revision/job, scalar
count, exact byte count, and dtype. A second command returns `BinaryPayload`
bytes for one descriptor. Project payload lookup checks the revision again, so
an edit between catalog and fetch is a conflict rather than a mixed plot.

`float64_le` uses explicit IEEE-754 little-endian encoding on every host;
`uint8` masks contain only zero or one. The refinement catalog includes grid,
observed/calculated/profile/background/residual values, inclusion mask, and
per-phase profile/reflection position/reflection intensity series. Job disposal
invalidates all of its descriptors and releases the retained native result.
The Tauri command returns these bytes as a native IPC response body rather than
serializing a numeric JSON array.
