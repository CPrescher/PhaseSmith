# Native application and desktop integration plan

## Decision

PhaseSmith's desktop application will use a Rust-only runtime. It will not
embed CPython or ship Python as a sidecar. The public Python package remains a
first-class scripting interface, but built-in application workflows will move
behind shared Rust APIs that both PyO3 and a future Tauri adapter call.

This plan does not put GUI state, widgets, or Tauri dependencies into the
scientific crates. It adds an application-neutral native boundary between the
validated numerical kernels and presentation-specific adapters.

## Target dependency direction

```text
Tauri frontend -> Tauri adapter ----+
                                     v
                              phasesmith-workflows
                                     |
Python API -> phasesmith-py ---------+
                                     |
                    phasesmith-model / phasesmith-io
                                     |
          phasesmith-engine / crystallography / core
```

Dependencies point downward only. In particular:

- `phasesmith-core`, `phasesmith-crystallography`, and `phasesmith-engine`
  never depend on PyO3, Tauri, or GUI types;
- `phasesmith-workflows` never depends on Tauri or Python;
- `phasesmith-py` translates NumPy/domain objects to the shared native API;
- the desktop adapter never depends on `phasesmith-py`, NumPy, or CPython;
- independent NumPy reference implementations remain independent and readable.

## Runtime and feature contract

Desktop version 1 supports built-in native scattering, intensity-correction,
profile, background, and sample-physics models. Python-defined provider
objects remain supported by the Python scripting API, but are not silently
accepted by the native desktop runtime. Persisted projects expose a capability
diagnostic when a requested provider is unavailable to a host.

CPU work uses explicit bounded execution policies and deterministic ordered
reductions. Refinement remains cooperatively cancellable at declared safe
boundaries and returns the last accepted checkpoint. Presentation adapters own
threads, user input, dialogs, and event dispatch.

## Owned native boundaries

### Prepared numerical objects

Move the owned structural phase currently implemented inside the PyO3 crate to
`phasesmith-engine` as a validated `PreparedStructuralPhase`. Move fixed
wavelength-component and multiphase scheduling/composition into native code so
arrays do not cross a language boundary merely to be combined.

### Domain and wire models

Add application-neutral owned records for patterns, instruments, radiation,
structures, phases, backgrounds, parameters, constraints, options, events,
checkpoints, results, recipes, and projects. Persistence records are explicit
wire types; serialization is not derived directly from internal kernel enums.
Physical units remain visible in field names.

The project record is multi-histogram-aware from its first native version. An
initial solver may reject more than one dataset, but project storage and stable
IDs must not encode a one-pattern assumption.

### I/O and persistence

Native I/O covers plain columns, supported GSAS powder formats, CIF import,
JSON+NPZ persistence, and reports. CIF import includes a native space-group
lookup source with reviewed provenance. The optional Python Gemmi backend may
remain as a scripting alternative and differential oracle.

Rust and Python must read one another's project bundles. All supported legacy
formats receive committed migration fixtures. New schemas describe nested
records completely rather than treating them as unconstrained JSON objects.

### Refinement workflows

Port parameter graphs, analytical backgrounds, residuals, bounded runtime,
Le Bail, Rietveld, staged recipes, and project operations to Rust in that
dependency order. Python's built-in workflows retain their public signatures
and result types while delegating to the native implementation. The optional
SciPy optimizer remains Python-only.

## Desktop adapter contract

The Tauri adapter exposes workflow-sized commands for project creation,
loading, import, calculation, refinement, cancellation, result acceptance,
saving, and report export. It stores revisioned immutable project snapshots and
tags every asynchronous result with the revision it evaluated. Completion
never overwrites a newer edited state implicitly.

Large numerical arrays use binary IPC. JSON payloads contain settings, stable
IDs, metrics, diagnostics, errors, and display-series descriptors. Dense
Jacobians and optimizer workspaces remain native unless explicitly exported.

The desktop distribution gate inspects produced artifacts and rejects any
dependency on `phasesmith-py`, `libpython`, Python frameworks, or bundled
wheels.

## Delivery sequence

1. Correct current execution-policy and documentation inconsistencies.
2. Extract `PreparedStructuralPhase` from PyO3 into `phasesmith-engine`.
3. Add a persistent native `ExecutionPolicy`.
4. Move fixed-component composition into Rust.
5. Move multiphase composition into Rust.
6. Add owned calculation request/result records.
7. Add a Rust-only import-to-calculation integration test.
8. Delegate Python built-in structural calculation to the shared Rust path.
9. Re-run QARR numerical and performance gates.
10. Add owned domain/project records and structured errors.
11. Port powder input, then native CIF input and space-group lookup.
12. Implement the canonical Rust persistence/reporting codec.
13. Port constraints, residuals, backgrounds, and the bounded runtime.
14. Port Le Bail, followed by Rietveld and staged recipes.
15. Delegate built-in Python refinement and persistence to the native APIs.
16. Add the separate Tauri adapter and packaging gates.
17. Implement the joint multi-histogram objective.

Each numbered change is reviewed after implementation, passes its focused
tests plus the relevant full gates, and is committed independently before the
next change starts.

## Cross-interface validation

Every migrated workflow adds deterministic Rust-versus-existing-Python
comparisons before Python delegates to Rust. Validation covers values,
derivatives, accepted histories, termination reasons, diagnostics,
checkpoints, invalid inputs, serialization, cancellation, worker-count
determinism, and realistic performance.

The migration may remove duplicated application orchestration only after the
shared native path passes those comparisons. It does not remove the independent
Python numerical references used to validate kernel equations.

## First architectural exit gate

A Rust-only program can construct or import one built-in structural request,
calculate a complete multiphase fixed-spectrum pattern, and return
display-ready arrays without importing Python. The existing Python calculation
API produces the same scientific result through that native workflow. No
refinement migration begins until this gate passes.

Status on 2026-08-08: the construction-to-calculation portion of this gate is
complete. An external Rust integration target constructs a two-phase request
containing a fixed spectrum, calculates display-ready arrays through the owned
native boundary, and verifies bitwise one-/two-thread agreement. The public
Python structural values and derivative products now call that same native
multiphase workflow, while custom providers retain the Python fallback. The
pinned QARR regression passed with an identical scientific fingerprint across
worker counts and no measured two-thread performance regression.

Powder import completed its native portion on 2026-08-08. The standalone
`phasesmith-io` crate now reads bounded plain-column, GSAS FXYE, and packed
constant-step GSAS STD data into `PatternRecord` without Python. Its contract
suite includes the pinned 49,494-sample APS sucrose file. The public Python
reader delegates both file and text input to this crate while preserving its
script-facing `PowderData` result. Native CIF import remains the outstanding
part of the I/O milestone.
