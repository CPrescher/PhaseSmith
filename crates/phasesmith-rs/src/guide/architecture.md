# Architecture and ownership

`PhaseSmith` separates scientific kernels, application-neutral state, workflows,
and adapters. The facade re-exports every supported native layer while keeping
dependency direction visible.

```text
application / CLI / Tauri host
        |       |       |
        |       |       +---- phasesmith::persistence
        |       +------------ phasesmith::workflows
        +-------------------- phasesmith::model
                                  |
                         phasesmith::engine
                          /              \
             phasesmith::core     phasesmith::crystallography

input adapters: phasesmith::io       worker budget: phasesmith::execution
```

## Layer responsibilities

- [`crate::core`] owns one-dimensional profile, broadening, background, and
  derivative kernels. It knows nothing about files or refinement projects.
- [`crate::crystallography`] owns cells, exact symmetry, reflection generation,
  scattering tables, and structure factors. It does not parse CIF syntax.
- [`crate::engine`] composes crystallography and profile kernels into structural
  patterns, including JVP/VJP paths.
- [`crate::model`] owns validated application records and stable identifiers.
  These are live domain values, not persistence or GUI transfer objects.
- [`crate::workflows`] owns refinement parameters, constraints, objective
  products, solvers, checkpoints, and runtime events.
- [`crate::io`] converts bounded external text into domain records.
- [`crate::persistence`] converts validated live records to explicit versioned
  JSON+NPZ wire data.
- [`crate::execution`] owns reusable, bounded worker pools.

## Borrowed calculation inputs, owned application state

Hot numerical APIs generally accept validated borrowed views such as
[`crate::core::GridView`] and [`crate::core::PeakBatchView`]. This avoids
copying large arrays at adapter boundaries. Results own their arrays so they
can safely leave the calculation scope.

Projects and workflows instead use owned records such as
[`crate::model::PatternRecord`] and [`crate::model::ProjectRecord`]. Hosts may
translate those records into their own command or presentation DTOs, but GUI
framework types should not be added to `PhaseSmith` crates.

## Adapter boundary

The `phasesmith-py` workspace crate is an adapter and is intentionally not
re-exported here. It converts `NumPy` arrays and Python models to these same Rust
types. A native host does not link `CPython`, initialize Python, or communicate
with a Python process.
