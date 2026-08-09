# Native project persistence

`phasesmith-persistence` is the Python-free storage and reporting boundary for
the Rust application model. A future Tauri adapter can call it directly; the
crate has no PyO3, NumPy, CPython, webview, or Tauri dependency.

## Bundle contract

A native project is a directory with two library-owned files:

- `manifest.json` contains format version 2, explicit wire records, array
  descriptors, units in field names, and SHA-256 hashes;
- `arrays.npz` contains only contiguous little-endian `float64`, `int32`,
  `uint64`, and boolean NPY members.

The complete manifest contract is
[`schemas/native-project-v2.schema.json`](../schemas/native-project-v2.schema.json).
Version 1 remains readable and migrates to a project with no Rietveld analyses.
Internal kernel enums are not serialized directly. Every persisted record has
an explicit conversion to and from the validated `phasesmith-model` domain.

Rust callers use:

```rust,ignore
use phasesmith_persistence::{
    ProjectReadLimits, ProjectSaveOptions, load_project, save_project,
};

save_project("analysis.psproj", &project, ProjectSaveOptions::default())?;
let restored = load_project("analysis.psproj", ProjectReadLimits::default())?;
```

The version-2 API persists the complete Python-free refinement boundary as
well:

```rust,ignore
use phasesmith_persistence::{load_rietveld_project, save_rietveld_project};

save_rietveld_project(
    "analysis.psproj",
    &state,
    ProjectSaveOptions::default(),
)?;
let restored = load_rietveld_project(
    "analysis.psproj",
    ProjectReadLimits::default(),
)?;
```

`RietveldProjectState` stores at most one analysis per histogram, including
built-in phase/sample-physics models, guarded lattice domains, analytical
backgrounds, parameter selections, constraints, execution and covariance
options, and the last accepted restart checkpoint with its full iteration
history. Opaque external-provider arrays are rejected because they cannot be
reconstructed by a Python-free desktop process.

Loading is bounded before domain construction. It checks manifest and archive
sizes, version, archive filename and hash, exact NPZ member set, every member's
dtype, shape, element count and content hash, finite floating-point values,
boolean encoding, record consistency, stable IDs, and final domain invariants.
NPY v1, v2, and v3 headers are accepted; generated archives use deterministic
NPY v1 members and deflate compression.

Saving an existing directory requires `overwrite: true`. Only
`manifest.json` and `arrays.npz` are replaced, so caller-owned notes remain.
The archive is installed before its manifest. An interrupted two-file update
can therefore be rejected by its archive hash, but cannot silently load mixed
state.

## Reports

`ProjectSummaryReport::from_project`, `project_summary_json`, and
`write_project_summary_json` produce a stable, versioned, array-free summary.
Reports include project revision, histogram and phase IDs, probes, sample
counts, phase links, required provider capabilities, and metadata. They are
safe JSON payloads for a desktop command response; display arrays remain in
binary storage or later binary IPC.
`write_project_summary_json_with_options` adds a protected-create mode for
desktop export: an existing destination is rejected unless overwrite is
explicit. The legacy convenience writer retains its replacement behavior.

## Python format distinction

`RietveldProject.save()` writes native format 2 when its monochromatic request,
built-in providers, numerical controls, and optional checkpoint can be
represented by the Rust application model. `RietveldProject.load()` translates
that validated native state back to the public scripting dataclasses, including
an independently resumable native checkpoint. Fixed X-ray dispersion, neutron
identities, every built-in sample-physics/background model, constraints,
geometry, and native numerical controls round-trip through this path.

`phasesmith.persistence.PersistenceBundle` format 12 remains readable and
writable. The scripting facade deliberately uses it for component radiation,
Python provider extensions, Python-created checkpoint state, and the
Python-optimizer-only `max_linearization_elements` option. Format 12 also
retains parser provenance, source labels, uncertainties, disorder metadata,
and other scripting metadata outside the native scientific application model.
Thus existing Python projects remain compatible while desktop projects never
require a Python interpreter.

## Cross-interface gate

The contract suite can run a real NumPy compatibility round trip when
`PHASESMITH_NUMPY_PYTHON` names a Python executable with NumPy. Rust writes the
bundle, NumPy verifies its dtypes and shapes and rewrites the NPZ, and Rust
loads the result into an equal validated project. This gate ensures the file
format remains usable from scripting without putting NumPy in the Rust runtime.
