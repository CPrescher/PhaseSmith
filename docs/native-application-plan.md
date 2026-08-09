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
17. Implement the joint multi-histogram objective. Complete: the native layout,
    matrix-free products, constrained solver, cancellation/checkpoint boundary,
    aggregate metrics, and pinned Rust-only PbSO4 workload share structure while
    preserving histogram-local experiment state.

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
script-facing `PowderData` result.

Native CIF import and space-group lookup completed on 2026-08-08. The same
crate now provides a bounded CIF 1.1 tokenizer/import policy and exact lookup
over all 530 conventional Hall settings using pinned Moyo 0.15.0 data. Rust-only
tests import four real small-structure CIFs, and Python delegates its default
`read_cif()` and lookup functions through stable parser-independent records.
The optional Gemmi backend remains injectable and is used as a differential
oracle; default CIF import succeeds when Gemmi imports are blocked. This closes
the native I/O portion of delivery step 11.

Canonical native project persistence and summary reporting completed on
2026-08-08. The Python-free `phasesmith-persistence` crate stores the
multi-histogram `ProjectRecord` as an explicit version-1 JSON wire manifest and
bounded NPZ arrays, validates hashes and all reconstructed domain invariants,
and emits stable array-free reports. Its complete JSON Schema and a configured
NumPy rewrite test lock down cross-interface compatibility. The existing
Python format-12 checkpoint remains unchanged until its refinement-only state
has native records and can migrate without loss. This completes delivery step
12 without introducing PyO3 or Tauri into the scientific dependency graph.

The parameter/constraint portion of delivery step 13 completed on 2026-08-08.
The new Python-free `phasesmith-workflows` crate owns stable structured
parameter identities, bounds, scaling and selection plus ordered fixed, affine,
and multi-source linear constraint transforms. Public constructors prevent
invalid state, dependency cycles and duplicate targets fail structurally, and
the exact row-major chain matrix agrees with the existing Python implementation
in a configured differential test. Residuals, analytical backgrounds, and the
bounded runtime remain the next independently reviewed substeps before solver
or Python-refinement delegation begins.

Native residual evaluation completed next. `phasesmith-workflows` now consumes
the validated owned pattern directly, applies inclusion masks and optional
one-sigma weights, and returns sample-aligned residual arrays plus `Rp`, `Rwp`,
chi-square, and reduced chi-square under the documented denominator and
degrees-of-freedom rules. Invalid observations/calculated arrays are structured
errors, and a configured differential test matches the existing Python arrays
and metrics. Analytical backgrounds and bounded runtime remain before step 13
is complete.

Analytical native backgrounds completed next. The workflow crate now owns
power, explicit-domain Chebyshev, fixed-knot point, normalized broad-Gaussian
amorphous, and ordered composite records with checked row-major analytical
bases. Constructor validation prevents invalid state, invariant bases are
explicitly cacheable, nonlinear amorphous derivatives pass centered
differences, and a mixed composite agrees with the existing Python
implementation. A 20,001-sample release benchmark covers values and the full
basis. The bounded runtime is the remaining delivery-step-13 substep.

The bounded native runtime completed delivery step 13. It provides the full
stable termination/event enums, finite diagnostics, positive iteration,
evaluation, elapsed-time and consecutive-rejection budgets, typed checkpoint
sinks, isolated event sinks, checkpoint-resume counters, and a cloneable
thread-safe first-reason cancellation token. Fake-clock and injected-failure
tests cover every boundary, a real OS thread requests cancellation, and the
normal stop sequence agrees with the existing Python runtime. This gives a
future Tauri adapter a direct Rust cancellation/event/checkpoint contract;
terminal key and signal handling remain optional Python/CLI presentation code.

The fixed-reflection portion of native Le Bail is now implemented as the first
substep of delivery step 14. It owns validated phase/reflection records,
display-ready calculations, non-negative redistribution, residual history,
rank diagnostics, typed checkpoints and one-iteration orchestration. It uses
the native bounded runtime directly, including cross-thread cancellation,
structured events and host checkpoint sinks. Mask, uncertainty, unobserved
support, restart, worker determinism and every accepted history row agree with
the current Python workflow. The committed release benchmark for eight
iterations over 85 reflections and 3,001 samples has a current central estimate
of 865.96 microseconds. This first entry point deliberately owns fixed
reflection topology; the subsequent substeps add parameter motion around that
validated extraction core.

Analytical fixed-topology profile motion completed the next native Le Bail
substep. Native typed keys now cover CW U/V/W/X/Y coefficients, phase scales,
and independent reflection positions; exact constraint chains feed a bounded
regularized Gauss--Newton solve with deterministic backtracking. Accepted
changes, scaled step norms, live parameters, covariance and the refined
instrument are carried through results and restart checkpoints. Every trial
calculation consumes the host evaluation budget. Constrained positions,
instrument width, checkpoint continuation and covariance have native tests,
while the complete accepted position-refinement history agrees with Python.
The setting-aware lattice foundation is now native as well. It owns exact
crystal-system parameter mappings, finite bounds, analytical CW d-spacing and
two-theta derivatives, conservative bound-wide reflection guards, physical
Bragg filtering, multiplicities, visibility masks, and stable-ID intensity
transfer. Cells outside the declared box and invalid transferred intensities
fail explicitly. The accompanying crystallography fix distinguishes proper
sixfold hexagonal axes from cubic threefold topology and recognizes exact
rhombohedral metrics in rhombohedral settings. Native Le Bail now consumes this
contract end to end. Lattice variables use the same typed bounds and constraint
transform as profile variables; their analytical Bragg-law columns feed the
same bounded solve. A successful line-search trial regenerates each accepted
dynamic domain, transfers intensities by stable ID, refreshes guard visibility,
reports added/removed families, and charges any topology-driven recalculation
to the evaluation budget. Restart permits changed reflection lists only when
the complete domain records compare equal. Exact synthetic recovery and the
complete accepted history agree with the Python implementation. Native Le Bail
is complete; Python delegation and native Rietveld are the next delivery
boundaries.

The first native Rietveld delivery boundary is now complete at calculation
level. `phasesmith-workflows` owns validated, adapter-safe structural phases,
observations, experiment state, sample-physics contributions, execution
controls, display-ready phase/profile/background arrays, crystallographic
intermediates, and residual metrics. Calculation revalidates public record
fields so a JSON/Tauri adapter cannot bypass constructors by mutating decoded
state. Six Rust integration contracts cover multiphase composition, bitwise
worker determinism, masks and uncertainties, position corrections,
sample-physics contributions, structural intermediates, invalid and mutated
requests, and configured parity with the existing Python structural workflow.
This is deliberately a forward-calculation boundary, not yet the native
Rietveld solver; typed structural parameter motion, matrix-free refinement,
topology regeneration, checkpoints, staging, and Python delegation remain
separate reviewed substeps.

The next Rietveld substep now supplies the stable structural parameter layer.
Phases can carry explicit asymmetric-site IDs (with deterministic generated IDs
for compatibility), and typed selections pack symmetry-independent lattice,
symmetry-allowed coordinate, occupancy, isotropic displacement, and phase-scale
parameters into the shared parameter model. Special-position coordinate bases
use deterministic pivot-ordered stabilizer elimination so durable `q0`, `q1`,
and related GUI/constraint identities do not depend on an SVD implementation.
Exact forward and reverse transforms connect these physical parameters to the
engine's native per-phase derivative layout; their composed JVP/VJP satisfies
the adjoint identity. Instrument, position, sample-physics, and background
families join this layout with the solver rather than being approximated here.

The reusable structural objective is now native as well. It prepares accepted
phase state once and exposes physical-parameter JVP, VJP, weighted gradient,
and damped `J^T W J` products without allocating a dense sample Jacobian. Mask
and uncertainty semantics are shared with residual evaluation, fixed background
participates in the gradient residual but not derivatives, and stale layouts
with different phase/site/symmetry identities fail before engine evaluation.
Native tests compare the matrix-free normal product with explicit analytical
columns and check both derivative adjoint identities.

Physical accepted-state installation is explicit as well. Ordered bounded
values rebuild cloned phase definitions through stable keys, including
setting-aware cells, general or symmetry-constrained coordinates, occupancy,
isotropic displacement, and phase scale. Out-of-bounds or non-finite vectors
fail before structural preparation. This closes the value-to-domain half of
the native solver boundary; no adapter needs to edit engine array offsets.

The bounded native structural Rietveld solver now supports guarded reflection
topology. Scaled conjugate gradients consume the matrix-free normal operator;
step norms, physical bounds, deterministic half-step backtracking, Levenberg
damping, model-product budgets, cancellation, structured events, accepted-only
history, typed checkpoint sinks, and exact restart all use the shared runtime.
Synthetic phase-scale and cubic-cell recovery pass natively, as do cancellation,
evaluation exhaustion, all-excluded observations, corrupt checkpoints, worker
determinism, and event/checkpoint delivery. Dynamic phases regenerate guarded
HKLs and multiplicities when lattice values move, transfer all sample-physics
values and derivative rows by stable reflection ID, and attach added/removed
families to accepted history. Changed-topology checkpoints require exact domain
identity; fixed-list checkpoints require exact reflection identity. The
non-structural parameter families are the next Rietveld substep.

The request boundary now also owns an optional native analytical background.
It composes with fixed supplied background during calculation and survives
accepted structural state replacement, providing the state half of subsequent
background-parameter refinement without introducing adapter-side arithmetic.

The general native parameter layout now adds selected CW profile, wavelength,
zero/displacement, and analytical-background coefficients around the existing
structural layout. The complete objective keeps structural products matrix-free
and materializes only the small experiment/background columns already emitted
by the engine. Full accepted-state installation updates instrument, correction,
guarded wavelength topology, background, and phases through stable keys.

Built-in native Rietveld sample physics completed next. Phase-owned Rust
records now cover isotropic size, isotropic microstrain, March--Dollase
preferred orientation, and ordered composition. They evaluate against the
engine's exact corrected reflection geometry, regenerate with dynamic
reflection topology, expose stable refinable scalar records, and install
accepted values without Python. Analytical parameter, position, and all six
March reciprocal-metric cell chains feed the complete matrix-free objective;
composition follows additive width and multiplicative intensity product rules.
Centered-difference, adjoint, topology, mixed-phase, and configured
Rust-versus-Python tests cover the boundary. The constraint-aware complete
solver is the next Rietveld substep.

The constraint-aware complete Rietveld solver completed next. It maps the full
native instrument, background, sample, and structural layout through the
existing fixed/affine/multi-source constraint transform, keeps scaled-free
solver coordinates fixed from the initial contract, and uses the complete
matrix-free JVP/VJP normal operator. Accepted checkpoints contain the complete
request state plus exact selection, lattice-bound, constraint, scale, damping,
and history contracts. Changed or initially unsatisfied graphs fail before
numerical work. Final diagnostics explicitly materialize only the bounded
free-parameter Jacobian, report rank and unresolved correlations, and emit a
full physical covariance only for a full-rank normal matrix. Native tests cover
complete-family recovery, affine covariance propagation, singular covariance
suppression, cancellation, exact continuation, and stale checkpoint rejection;
a configured Python differential matches accepted objective histories.

Native staged Rietveld recipes now sit above that unchanged solver. Typed
explicit stages own cumulative selections, optional numerical controls,
rationale, and accepted termination policies. The deterministic intelligent
planner proposes only caller-authorized families and never runs implicitly.
Execution validates the complete maximum constraint graph before numerical
work, rejects stage-local missing dependencies, promotes only accepted
physical states, disables covariance on intermediate stages, and shares one
cooperative cancellation token across the workflow. Optional shared sinks
forward structured numerical events, caller-visible stage boundaries, and
complete accepted checkpoints without adapter-owned orchestration. Rust tests
cover planner parity with the configured Python oracle, cumulative recovery,
final-only diagnostics, invalid recipes, constraint contracts, cancellation,
and the evaluate-only case.

The built-in monochromatic Python Rietveld façade now delegates complete
refinement and native checkpoint continuation through PyO3 while releasing the
GIL. Python-defined providers and Python callback requests retain the scripting
orchestrator explicitly. The adapter bridge reconstructs the established
Python result types from native metrics, histories, covariance, and accepted
state, and Rust/Python regression suites cover structural families,
backgrounds, instrument motion, sample physics, constraints, and restart.

Native project refinement ownership now has a validated application-neutral
record as well. `RietveldProjectState` joins the multi-histogram
`ProjectRecord` to at most one runnable analysis per histogram and checks exact
pattern, experiment, phase-order, phase-definition, provider, bounds,
constraint, solver-option, and checkpoint contracts. This is the domain input
for the later revisioned Tauri state; neither adapter needs to invent its own
cross-record validation rules. Native project format 2 now persists this
complete state through explicit Rust-only wire records, including guarded
lattice domains, built-in sample physics and backgrounds, constraints, options,
and accepted checkpoints. Format 1 remains readable as project-only state.
The PyO3 adapter now exposes that same save/validate/load boundary, releases
the GIL during file I/O, and carries restart checkpoints as independent native
handles. The public Python project facade now uses that bridge for representable
built-in monochromatic projects and reconstructs the scripting request,
options, and native continuation handle on load. Component radiation, Python
extensions, Python-only optimizer controls, and non-native checkpoints retain
the backward-compatible scripting format. Fixed-dispersion, neutron, geometry,
background, sample-physics, constraint, option, corruption, and restart tests
cover this translation. This completes delivery step 15.
The scripting `CancellationToken` now also owns the shared Rust token used by
detached native refinement, so `RietveldProject.refine()` no longer falls back
to Python merely to preserve its concurrent `stop()` contract.

Delivery step 16 has started with the Python-free `phasesmith-desktop` adapter.
Its project store owns immutable native snapshots, checks every edit/save/close
against an expected revision, and uses exact snapshot identity to reject the
close/reopen ABA race for asynchronous completions. Native load validates
before installation and failed commands preserve current state. The adapter's
job/event/binary transport layer and thin Tauri host remain subsequent atomic
substeps.

A further desktop-adapter substep adds detached native Rietveld jobs, structured
finite progress/completion/failure events, first-reason cooperative
cancellation, polling, explicit result acceptance, and result disposal.
Completion never mutates project state; acceptance compares the exact source
snapshot and revision. Event-delivery failure is isolated from numerical work,
and shared multi-histogram phase acceptance remains explicitly unavailable
until delivery step 17 provides the joint shared/local model. Binary series
transport and the thin Tauri host remain.

Binary display transport is now complete in the desktop adapter. JSON catalogs
describe stable revision/job-owned `float64_le` and `uint8` series, while a
separate payload call explicitly encodes only the requested vector. Catalogs
cover observed project data and complete refinement plots, residuals, masks,
phase profiles, and reflection sticks. Revision rechecks prevent mixed project
snapshots and job disposal invalidates retained result series. The thin Tauri
host is now present as a separate pinned application crate. It translates the
existing project, refinement, event, and binary-series contracts without
linking Python, and a release audit checks its dependency tree, executable
links, and bundle contents for Python artifacts. The checked-in static frontend
is only a runtime probe. Bounded powder import is now exposed as an atomic
revision-checked command for columns, GSAS FXYE, and GSAS STD input with a
complete native monochromatic experiment record. Native CIF phase import now
also parses with bounded native limits, generates target-histogram-specific
reflections, preserves X-ray/neutron scattering identity and diagnostics, and
attaches the phase through exact snapshot comparison. Standalone calculation
now evaluates ordered project phases through the native structural/Rietveld
path, retains revision-owned results without mutating project state, and uses
the same binary-series transport as refinement. Report export now writes the
canonical versioned project-summary JSON for one exact revision, refuses
existing destinations unless overwrite is explicit, and performs file I/O off
the Tauri event loop. Delivery step 16 is complete; product UI design remains
separate from this native adapter plan.
