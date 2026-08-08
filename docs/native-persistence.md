# Native project persistence

`phasesmith-persistence` is the Python-free storage and reporting boundary for
the Rust application model. A future Tauri adapter can call it directly; the
crate has no PyO3, NumPy, CPython, webview, or Tauri dependency.

## Bundle contract

A native project is a directory with two library-owned files:

- `manifest.json` contains format version 1, explicit wire records, array
  descriptors, units in field names, and SHA-256 hashes;
- `arrays.npz` contains only contiguous little-endian `float64`, `int32`,
  `uint64`, and boolean NPY members.

The complete manifest contract is
[`schemas/native-project-v1.schema.json`](../schemas/native-project-v1.schema.json).
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

## Python format distinction

The existing `phasesmith.persistence.PersistenceBundle` format version 12 is
the current Python scripting checkpoint format. It contains solver types that
have not all moved to Rust yet and remains supported unchanged. Native project
format 1 instead stores the application-neutral, multi-histogram
`ProjectRecord` already owned by Rust.

These are deliberately separate during migration. Once constraints,
backgrounds, refinement options, checkpoints, and results have native domain
records, the Python API can delegate its built-in persistence to this codec
through explicit migrations. No Python interpreter is required by the native
format now or after that delegation.

## Cross-interface gate

The contract suite can run a real NumPy compatibility round trip when
`PHASESMITH_NUMPY_PYTHON` names a Python executable with NumPy. Rust writes the
bundle, NumPy verifies its dtypes and shapes and rewrites the NPZ, and Rust
loads the result into an equal validated project. This gate ensures the file
format remains usable from scripting without putting NumPy in the Rust runtime.
