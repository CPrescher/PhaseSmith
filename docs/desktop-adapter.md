# Python-free application adapter

`phasesmith-desktop` is the application-state boundary intended for a future
native GUI host. It depends on the native model, workflows, and persistence
crates, and it does not depend on Tauri, a webview, PyO3, NumPy, or CPython.
This keeps the scientific and project-state contract independently testable and
lets a GUI crate remain a thin command/event translation layer.

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

Native powder import accepts bounded plain-column, GSAS FXYE, and GSAS STD
files together with a complete monochromatic experiment record. Parsing and
nested validation happen outside the state lock; exact-snapshot replacement
then appends the histogram and increments the project revision once. Parse,
resource-limit, experiment, phase-reference, duplicate-ID, and concurrent-edit
failures leave the open snapshot unchanged. A GUI host should run file parsing
on a blocking worker and expose only serializable request/response records.

Native CIF phase import selects one bounded CIF block, preserves parser
diagnostics and source provenance, and generates the phase's reflection
families from the exact target histogram range and monochromatic wavelength.
Built-in X-ray charge/table identities and neutron isotope identities are
derived with the same explicit rules as the scripting layer. The request owns
Friedel merging, candidate limits, phase scale, coordinate tolerance, and an
explicit probe-compatible integrated-intensity correction. Import refuses to
silently rewrite a histogram that already has a native analysis; attaching a
new phase there requires a later analysis-edit workflow.

Standalone calculation builds neutral built-in `RietveldPhase` inputs directly
from a histogram's ordered project phase records and reuses the same native
calculation path as refinement. A GUI command can await this work on its
blocking worker, return only a finite scalar summary and retained calculation
ID, and never mutates the project. Plot catalogs and raw payloads include the
grid, observed/calculated/profile/background/residual/mask arrays and all
per-phase profiles and reflection sticks. Each retained result owns its exact
source snapshot and remains viewable after later edits until explicit disposal;
bulk encoding occurs after releasing the result-map lock. This initial command
does not offer cooperative cancellation, so the UI should use refinement jobs
for cancellable iterative work.

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
These bytes are suitable for a native IPC response body rather than
serializing a numeric JSON array.

## Future GUI host

No GUI framework or executable application is included in this repository.
A future Tauri or other native host should live in its own crate or repository,
depend on `phasesmith-desktop`, and translate these project, refinement, event,
and binary-series contracts. Application state and refinement workers remain in
`phasesmith-desktop`, project codecs remain in `phasesmith-persistence`, and no
scientific crate needs to depend on the presentation framework or Python.

Histogram radiation may be monochromatic or a validated fixed spectrum. The
desktop calculation boundary constructs the corresponding native structural
model directly, and fixed-spectrum project state round-trips through native
persistence. No validation or calculation command crosses a Python process
boundary.

Project report export is revision-owned as well. It writes the canonical
versioned, array-free native summary JSON on the blocking pool and returns its
absolute destination. Protected creation uses an atomic `create_new` policy:
an existing caller-owned file is never truncated unless the command explicitly
sets `overwrite`. Stale revisions are rejected before any filesystem change.
This completes the reusable workflow surface defined for delivery step 16;
product UI and application packaging remain separate work.
