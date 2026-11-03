# Migrating to 0.8

Version 0.8 adds supported single-crystal APIs alongside the existing powder
workflows. Python requires 3.11+ and NumPy 1.26+; source builds require Rust
1.89+. Ordinary Python wheels include single-crystal kernels and bundled HDF5,
including the supported count filters. No h5py, SciPy, Gemmi, Dioptas or external
structure solver is required for the native workflow.

## Imports and Cargo features

Replace pre-release imports under `phasesmith.experimental` with the corresponding
`phasesmith.single_crystal` modules. The former experimental wrappers are removed;
independent Python equations live under `phasesmith.single_crystal.reference`.
Shared `UnitCell`, structure, symmetry and file-reader types retain their existing
owners. See the [API map](api-reference.md#single-crystal-workflows).

The three unpublished development crates (`phasesmith-spots`,
`phasesmith-solution`, `phasesmith-row-search`) are consolidated into one
`phasesmith-single-crystal` crate with `spots`, `solution` and `row_search`
modules. Direct development-crate imports must use those module paths. Existing
facade paths `phasesmith::spots`, `phasesmith::solution` and
`phasesmith::row_search` remain available. HDF5 is a separate adapter crate.
Historical diagnostic/backend labels such as `phasesmith-spots Rust` remain
unchanged for result-format compatibility; they identify a component, not an
additional published crate.

Rust facade features are `single-crystal` (default), `spots`, `solution`,
`row-search`, `hdf5`, and `single-crystal-files`. Retired experimental feature
aliases are removed. HDF5 remains optional for Rust consumers:

```sh
cargo check -p phasesmith --no-default-features
cargo check -p phasesmith --no-default-features --features spots,solution,row-search
cargo check -p phasesmith --features single-crystal-files
```

Published powder compatibility aliases keep their separate pre-1.0 policy;
see [the 0.6 migration](migration-0.6.md). They are not removed as part of the
single-crystal promotion.

## Persisted data and scientific inputs

Portable data readers remain supported. `solution.upgrade_solution_record`
explicitly upgrades supported historical search-summary schemas; it does not
rerun a search or invent observations. Historical protocols and their expected
results remain frozen. Arbitrary pickles referring to removed Python modules
are not a portable migration format.

Preserve signed intensities and positive finite sigma(I). Full-cell composition
counts, element-type-only inputs, supplied symmetry and unknown-group assessment
are distinct contracts. A candidate file or `target_met` does not imply the full
workflow's `accepted` assessment or a unique structure.

For new scripts, start with the [NumPy-only example](single-crystal-quickstart.md).
For original frames, use [WorkflowConfig](single-crystal-workflow-api.md): its
current composed recipe accepts EIGER-style HDF5, a matching PONI calibration,
explicit acquisition/DAC geometry and a primitive orthorhombic metric. Lower-level
methods support other inputs; those are not automatically supported by this recipe.

## Exact positional-search reproducibility

Legacy lattice-proposal fitting uses a NumPy optimizer around native predictions.
Its spawned search contract requires a single-thread numerical environment in
both parent and workers. Start Python as follows (POSIX shell):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python analysis.py
```

In PowerShell, set `$env:OPENBLAS_NUM_THREADS="1"`,
`$env:OMP_NUM_THREADS="1"` and `$env:VECLIB_MAXIMUM_THREADS="1"` before launching
Python. Use the multiprocessing `if __name__ == "__main__":` guard.
Changing `os.environ` after importing NumPy cannot reconfigure an already-loaded
BLAS library. A multithreaded parent and single-thread spawned workers can differ
in reduction rounding, including on NumPy 1.26. CI and release tests now set these
limits before interpreter startup and retain exact model/audit equality checks.
The library does not mutate the host's BLAS configuration. The complete native
file workflow uses its own bounded Rust execution policy.
