# Rust API

Native applications depend on the `phasesmith` facade crate. This is the
recommended boundary for a Tauri desktop application or any other Rust
consumer; Python is not embedded or launched.

```shell
cargo add phasesmith
```

For reproducible application builds, commit `Cargo.lock`. Libraries can select
an explicit compatible range in `Cargo.toml`:

```toml
[dependencies]
phasesmith = "0.1"
```

## Facade modules

| Facade path | Component crate | Responsibility |
| --- | --- | --- |
| `phasesmith::core` | `phasesmith-core` | Profile and background kernels. |
| `phasesmith::crystallography` | `phasesmith-crystallography` | Cells, symmetry, scattering, and structure factors. |
| `phasesmith::engine` | `phasesmith-engine` | Native structural-calculation composition. |
| `phasesmith::execution` | `phasesmith-execution` | Bounded deterministic execution policies. |
| `phasesmith::io` | `phasesmith-io` | Powder and CIF input adapters. |
| `phasesmith::model` | `phasesmith-model` | Application-neutral project and pattern records. |
| `phasesmith::workflows` | `phasesmith-workflows` | Le Bail, Rietveld, quantitative, and validation workflows. |
| `phasesmith::persistence` | `phasesmith-persistence` | Native JSON+NPZ project persistence and reports. |

The complete generated Rust reference is published by docs.rs:

- [`phasesmith` facade documentation](https://docs.rs/phasesmith/0.1.0/phasesmith/)
- [`phasesmith-workflows`](https://docs.rs/phasesmith-workflows/0.1.0/phasesmith_workflows/)
- [`phasesmith-persistence`](https://docs.rs/phasesmith-persistence/0.1.0/phasesmith_persistence/)

Starting with the next release after 0.1.0, the facade reference also contains
task-oriented native guides alongside the generated item reference:

- [getting started](https://docs.rs/phasesmith/latest/phasesmith/guide/getting_started/)
- [architecture and ownership](https://docs.rs/phasesmith/latest/phasesmith/guide/architecture/)
- [scientific conventions](https://docs.rs/phasesmith/latest/phasesmith/guide/scientific_conventions/)
- [implemented mathematics](https://docs.rs/phasesmith/latest/phasesmith/guide/mathematics/)
- [calculations and workflows](https://docs.rs/phasesmith/latest/phasesmith/guide/workflows/)
- [desktop and service integration](https://docs.rs/phasesmith/latest/phasesmith/guide/application_hosts/)

The examples embedded in these pages are compiled as Rust doctests. Component
crates also provide their own overview, capability map, and boundary guidance
before the generated function/type listings.

## Application boundary

A GUI crate should own window state, commands, presentation-specific records,
and background-task coordination. It should call the typed facade directly and
persist domain state through `phasesmith::persistence`. This keeps the Python
scripting interface available to scientists without making CPython a desktop
sidecar or packaging dependency.

Long-running work should use `phasesmith::execution::ExecutionPolicy` and the
workflow runtime/cancellation contracts. Do not pass Tauri handles or UI types
into PhaseSmith domain crates.

For neutron TOF structural integrations, `phasesmith::engine` exposes the
single-bank value/dense/JVP/VJP primitive. The guarded native multi-bank
objective is `phasesmith::workflows::PreparedStructuralTofMultiBankObjective`;
it sums shared structural rows across banks and keeps scale, selected instrument
coefficients, and optional Chebyshev backgrounds bank-local. The corresponding
bounded entry points are `refine_structural_tof_multibank` and
`refine_structural_tof_multibank_with_runtime`; the latter accepts application
cancellation, event, and typed checkpoint sinks and returns the last atomically
accepted state at a normal bound. These remain Rust integration boundaries, not
yet the public Python workflow or persisted project format. Their correction
and bank-geometry requirements are documented in
[Structural TOF readiness](tof-structural-readiness.md).
Application hosts can retain runnable analyses in
`StructuralTofMultiBankProjectState`. Its validation binds bank IDs to exact
project TOF histogram arrays/instruments, enforces disjoint histogram
ownership, matches the one shared built-in phase, and revalidates optional
checkpoints. Versioned serialization of this state is documented separately
when available.
