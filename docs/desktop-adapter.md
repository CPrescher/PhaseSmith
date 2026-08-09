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
The next adapter slice adds job IDs, refinement event delivery, cancellation,
and accepted-result installation on top of these immutable snapshots.
