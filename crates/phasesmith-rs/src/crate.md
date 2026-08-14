# `PhaseSmith`

`PhaseSmith` is a native Rust library for powder-diffraction profile calculation,
crystallography, Le Bail extraction, Rietveld refinement, and versioned project
persistence. This facade is the recommended dependency for applications. It
exposes the focused implementation crates under one version without embedding
Python, `PyO3`, a GUI toolkit, or a process runtime.

The Python distribution is a separate adapter over the same native kernels. A
Rust program, including a Tauri desktop application, calls this crate directly
and does not need a Python interpreter or sidecar.

## Which API should I start with?

| Task | Start here |
| --- | --- |
| Evaluate profiles or subtract a smooth background | [`core`] |
| Work with cells, symmetry, reflections, or scattering | [`crystallography`] |
| Compose crystal structures into calculated patterns | [`engine`] |
| Read powder data, CIFs, or space-group metadata | [`io`] |
| Own application-neutral patterns, phases, and projects | [`model`] |
| Run Le Bail, Rietveld, or quantitative workflows | [`workflows`] |
| Save native projects and write stable reports | [`persistence`] |
| Bound worker threads for application integration | [`execution`] |

The [`guide`] module adds task-oriented explanations that span those crate
boundaries:

- [`guide::getting_started`] contains small, runnable native examples;
- [`guide::architecture`] explains ownership and dependency direction;
- [`guide::scientific_conventions`] defines units, layouts, and derivatives;
- [`guide::mathematics`] documents the implemented equations and analytical
  chains;
- [`guide::workflows`] maps calculation and refinement entry points;
- [`guide::cif_inputs`] explains accepted CIF content, strict/permissive
  import, diagnostics, and conversion into a refinement phase;
- [`guide::real_data_rietveld`] walks through the repository's measured PbSO4
  X-ray/neutron refinement from files to validated results;
- [`guide::refinement_operations`] shows parameter staging, bounded execution,
  checkpoint continuation, and result interpretation;
- [`guide::application_hosts`] covers desktop/GUI integration.

## From files to a refinement

A complete analysis crosses several deliberately separate crates. The types
make each scientific decision visible instead of hiding it in a project-file
dictionary:

| Step | Input | Output | Guide |
| --- | --- | --- | --- |
| Import | powder text and CIF | [`model::PatternRecord`] and [`io::CifStructure`] | [`guide::cif_inputs`] |
| Prepare | structure, radiation, range | reflections and [`engine::StructuralPhaseDefinition`] | [`guide::real_data_rietveld`] |
| Calculate | pattern, instrument, phases | [`workflows::RietveldCalculation`] | [`guide::real_data_rietveld`] |
| Refine | parameter selection, bounds, constraints | accepted state, history, metrics, checkpoint | [`guide::refinement_operations`] |
| Persist | validated project and analysis | versioned native bundle and reports | [`persistence`] |

If you want a working program before reading the individual types, start with
the real-data walkthrough. It names the exact repository command, explains
each construction step, and shows which returned fields should be checked
before accepting a refinement.

## Quick start: calculate a profile

The lowest-level profile API borrows a strictly increasing grid and
structure-of-arrays peak parameters. The result owns the calculated values and
analytical derivatives.

```
use phasesmith::core::{
    GridView, PeakBatchView, SupportPolicy, accumulate_batch,
};

let x = (0..=1_000)
    .map(|index| 20.0 + f64::from(index) * 0.01)
    .collect::<Vec<_>>();
let positions = [24.0, 26.0];
let intensities = [100.0, 80.0];
let fwhms = [0.08, 0.10];
let etas = [0.30, 0.50];

let grid = GridView::new(&x)?;
let peaks = PeakBatchView::new(&positions, &intensities, &fwhms, &etas)?;
let result = accumulate_batch(
    grid,
    peaks,
    SupportPolicy::FwhmMultiple(20.0),
)?;

assert_eq!(result.y.len(), x.len());
assert_eq!(result.derivatives.local.peak_count(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For values without derivative storage, use
[`core::accumulate_values_batch`]. Higher-level structural calculations and
refinements build on the same kernels rather than reproducing their formulas.

## Design promises

- **Explicit units:** public physical fields include unit suffixes where
  practical; see [`guide::scientific_conventions`].
- **Validated boundaries:** borrowed numerical views and owned domain records
  validate shapes, finiteness, ordering, and resource limits.
- **Analytical products:** calculation APIs expose values, sparse/dense
  Jacobians, JVPs, and VJPs instead of relying on finite differences.
- **Bounded execution:** application hosts choose an [`execution::ExecutionPolicy`]
  rather than modifying Rayon's global thread pool.
- **Adapter independence:** Python objects, Tauri handles, and persistence wire
  records do not enter the numerical or domain crates.

## API status

`PhaseSmith` is pre-1.0. Scientific conventions and validated boundaries are
intentional, but Rust type names and composition APIs may still evolve between
minor releases. Commit `Cargo.lock` in applications and consult the versioned
[PhaseSmith documentation](https://phasesmith.readthedocs.io/) when upgrading.
