# Native project persistence

`phasesmith-persistence` is the Python-free storage and reporting boundary for
the Rust application model. A future Tauri adapter can call it directly; the
crate has no PyO3, NumPy, CPython, webview, or Tauri dependency.

## Bundle contract

A native project is a directory with two library-owned files:

- `manifest.json` contains format version 7, explicit wire records, array
  descriptors, units in field names, and SHA-256 hashes;
- `arrays.npz` contains only contiguous little-endian `float64`, `int32`,
  `uint64`, and boolean NPY members.

The complete manifest contract is
[`schemas/native-project-v7.schema.json`](https://github.com/CPrescher/PhaseSmith/blob/main/schemas/native-project-v7.schema.json).
Versions 1 through 6 remain readable. Versions 1–5 have no CW Pawley field;
version 6 adds CW Pawley, and version 7 adds joint TOF Pawley analyses. Versions 1 and 2 have no TOF histograms
or TOF Le Bail analyses; version 1 also has no Rietveld analyses; versions 1
through 3 have no joint multi-bank TOF geometry analyses; versions 1 through 4
have no structural multi-bank TOF analyses.
Internal kernel enums are not serialized directly. Every persisted record has
an explicit conversion to and from the validated `phasesmith-model` domain.

## Mixed analyses and Pawley

Use native `ProjectBundle`, `load_project_bundle` and `save_project_bundle` to
retain **all** supported methods together. Method-specific loaders remain
validated views selecting one family; saving such a view writes that family.
The mixed bundle API preserves Rietveld, single-bank TOF Le Bail, joint TOF
geometry, structural TOF, CW Pawley and joint TOF Pawley analyses, including accepted checkpoints.
All families validate against the same owned histogram records. Unknown future
formats and fields are rejected rather than silently dropped.

Pawley analyses own typed cell-only phase/domain metadata and stable family IDs;
they do not require fictitious structural atoms. When a histogram references
structural phases, Pawley phase IDs must match their ordered identities. The
histogram pattern and experiment must match exactly. Observations are stored
once in the shared NPZ arrays; native format 7 hydrates the standalone scientific
codec from that histogram before checking the checkpoint request digest.

TOF Pawley analyses have a stable analysis ID and reference one or more TOF
histogram IDs. They retain shared cells and fixed HKL topology, bank-local
areas/backgrounds, normalization provenance, selected calibration/profile
parameters, exact ties and one atomic joint checkpoint. Each bank hydrates
its observations and instrument from the corresponding shared histogram.
Python `from_tof_pawley`, `tof_pawley` and `with_tof_pawley` mirror the CW methods
below; see the [TOF example](tof-pawley.md#restart-and-shared-bundles).

Python exposes `phasesmith.project_bundle.ProjectBundle`:

```python
from phasesmith.project_bundle import ProjectBundle
from phasesmith.radiation import RadiationProbe

bundle = ProjectBundle.from_pawley(pawley_project, probe=RadiationProbe.NEUTRON)
bundle.save("sample.psproj")
restored = ProjectBundle.load("sample.psproj")
analysis = restored.pawley("histogram")
result = analysis.refine()
updated = restored.with_pawley("histogram", analysis)
updated.save("sample-resumed.psproj")
```

The Python bundle is an immutable snapshot. Retrieved Pawley projects are
independently editable; explicit replacement prevents unnoticed mutation of
shared state. Load an existing native Rietveld/TOF bundle and use `with_pawley`
to add the method to an existing CW histogram while preserving other analyses.
`analysis_counts` reports every retained family. Saving has an explicit
`overwrite=True` option and preserves unrelated application files.

`crates/phasesmith-persistence/examples/pawley_bundle_exchange.rs` is an
executable Python-free load/resume/save adapter. Tests exercise Python-save /
Rust-load-resume / Rust-save / Python-load-resume, exact profiles and histories,
mixed-method retention, legacy migration, corruption and stale shared inputs.

Rust callers use:

```rust,ignore
use phasesmith_persistence::{
    ProjectReadLimits, ProjectSaveOptions, load_project, save_project,
};

save_project("analysis.psproj", &project, ProjectSaveOptions::default())?;
let restored = load_project("analysis.psproj", ProjectReadLimits::default())?;
```

The native API persists the complete Python-free Rietveld refinement boundary:

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

Format 3 adds explicit microsecond-domain TOF records and resumable
fixed-instrument Le Bail state:

```rust,ignore
use phasesmith_persistence::{
    load_tof_lebail_project, save_tof_lebail_project,
};

save_tof_lebail_project(
    "tof-analysis.psproj",
    &tof_state,
    ProjectSaveOptions::default(),
)?;
let restored = load_tof_lebail_project(
    "tof-analysis.psproj",
    ProjectReadLimits::default(),
)?;
```

`TofHistogramRecord` stores `tof_us` separately from the CW `x_deg` record and
owns the validated 15-coefficient TOF calibration/profile model. A
`TofLeBailProjectState` binds at most one analysis to each TOF histogram and
checks the project pattern, instrument, active phase order, and phase labels.
Its NPZ members retain reflection HKLs, d-spacings and intensities plus every
accepted checkpoint's phase intensities and sample-aligned residual history.
The manifest retains scalar options and optional microsecond-domain Chebyshev
background state. This is fixed-instrument, fixed-cell Le Bail persistence.

Format 4 adds complete joint multi-bank TOF geometry state:

```rust,ignore
use phasesmith_persistence::{
    load_tof_multibank_geometry_project,
    save_tof_multibank_geometry_project,
};

save_tof_multibank_geometry_project(
    "joint-tof.psproj",
    &joint_state,
    ProjectSaveOptions::default(),
)?;
let restored = load_tof_multibank_geometry_project(
    "joint-tof.psproj",
    ProjectReadLimits::default(),
)?;
```

`TofMultiBankGeometryProjectState` binds disjoint sets of TOF histograms to
stable joint analysis IDs. Bank IDs are the project histogram IDs; pattern,
initial instrument, phase order, phase labels, initial shared cells, and exact
symmetry settings are cross-checked against project state. The manifest stores
the bounded shared-cell and bank-local instrument selections plus all solver
controls. Typed NPZ members store every input and checkpoint HKL, d-spacing,
intensity, inclusion, residual, and weighted-residual array. The checkpoint
also retains accepted instruments, cells, backgrounds, aggregate metrics, and
the complete lattice/instrument change history. This remains Le Bail geometry
refinement with fixed reflection topology, not structural TOF Rietveld.

Format 5 adds the complete Python-free structural multi-bank TOF boundary:

```rust,ignore
use phasesmith_persistence::{
    load_structural_tof_multibank_project,
    save_structural_tof_multibank_project,
};

save_structural_tof_multibank_project(
    "structural-tof.psproj",
    &structural_state,
    ProjectSaveOptions::default(),
)?;
let restored = load_structural_tof_multibank_project(
    "structural-tof.psproj",
    ProjectReadLimits::default(),
)?;
```

`StructuralTofMultiBankProjectState` reuses project-owned pattern arrays,
initial instruments, and the shared structural definition. Its explicit wire
record stores stable site IDs, structural selections and lattice bounds,
bank-local angle/correction/scale/background/instrument contracts, bounded
execution and solver controls, and the complete last-accepted checkpoint.
Checkpoint state includes accepted cell/site values, instruments, scales,
background coefficients, the original solver parameter contract, and every
accepted iteration/change record. Loading reconstructs the domain objects and
revalidates the checkpoint against the exact original request before returning
it. Infinite open scale endpoints are represented by JSON `null`; all physical
values remain finite. No structural TOF bulk array is duplicated outside the
project-owned NPZ members.

Loading is bounded before domain construction. It checks manifest and archive
sizes, version, archive filename and hash, exact NPZ member set, every member's
dtype, shape, element count and content hash, finite floating-point values,
boolean encoding, record consistency, stable IDs, and final domain invariants.
NPY v1, v2, and v3 headers are accepted; generated archives use deterministic
NPY v1 members and deflate compression.

Saving an existing directory requires `overwrite: true`. Only
`manifest.json` and `arrays.npz` are replaced, so caller-owned notes remain.
Overwrite uses synced temporary files plus private backups. The archive and
manifest are installed as one recoverable pair; a later load or save restores
the previous pair after an interrupted partial install, while a fully installed
pair commits and removes the backups. Future format versions are probed before
strict manifest decoding, so they return `UnsupportedVersion` even when they
add unknown fields.

## Reports

`ProjectSummaryReport::from_project`, `project_summary_json`, and
`write_project_summary_json` produce a stable, versioned, array-free summary.
Reports include project revision, histogram and phase IDs, probes, coordinate
conventions (`two_theta_deg` or `tof_us`), sample counts, phase links, required
provider capabilities, and metadata. They are
safe JSON payloads for a desktop command response; display arrays remain in
binary storage or later binary IPC.
`write_project_summary_json_with_options` adds a protected-create mode for
desktop export: an existing destination is rejected unless overwrite is
explicit. The legacy convenience writer retains its replacement behavior.

## Python format distinction

`RietveldProject.save()` writes the current native format when its monochromatic request,
built-in providers, numerical controls, and optional checkpoint can be
represented by the Rust application model. `RietveldProject.load()` translates
that validated native state back to the public scripting dataclasses, including
an independently resumable native checkpoint. Fixed X-ray dispersion, neutron
identities, every built-in sample-physics/background model, constraints,
geometry, and native numerical controls round-trip through this path.

`phasesmith.persistence.PersistenceBundle` format 13 is the current writable
schema and format 12 remains readable. The scripting facade deliberately uses
it for component radiation,
Python provider extensions, Python-created checkpoint state, and the
Python-optimizer-only `max_linearization_elements` option. Format 13 also
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

## CW Pawley workflow

[CW Pawley refinement](pawley.md) provides independent bounded family areas,
selected analytical cell/profile/background refinement and accepted-state restart.
Use `phasesmith.refinement.pawley` in Python or `phasesmith::workflows` in Rust.
Its [standalone version-1 JSON format](pawley.md#runtime-and-persistence) is separate
from existing mixed-method project bundles. See the [validation report](pawley-validation.md)
for current scientific coverage and limitations.
