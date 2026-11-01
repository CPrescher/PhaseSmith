# Dioptas native structure workflow

Try the integrated application with a local PhaseSmith 0.8 wheel before publishing
0.8. The first Dioptas consumer now calls the same Rust coordinator as the
standalone original-frame example. Python owns configuration, Qt notifications
and presentation; masks, detection/linking, geometry fitting, integration,
profile learning/fitting, scaling and structure search remain native Rust.
Full single-crystal structure refinement remains outside this workflow.

## Supported first recipe

The current recipe accepts one original EIGER-style HDF5 count dataset, its pixel
size metadata, a uniform-detector PONI geometry, supplied primitive orthorhombic
cell lengths and non-hydrogen element types. Declare exposure indices, omega
start/step/width, the laboratory rotation axis, detector row reversal and the DAC
axis/opening. It searches symmetry without an exact group or Laue prior.

This preserves the frozen v40 scientific controls, gain 1/read noise 0 and
combined illumination/DAC masks, with up to ten workers and independent memory
budgets. It does not inherit masks, image corrections or lattice choices from
Dioptas's older blind-lattice workflow. Unsupported PONI orientation/distortion
configuration and incompatible pixel sizes fail explicitly. General detector
formats, arbitrary cell metrics and structure refinement are separate work.

## Python application boundary

The ordinary wheel exports `phasesmith.single_crystal.workflow`:

```python
from phasesmith.single_crystal.workflow import WorkflowConfig, run_workflow

config = WorkflowConfig.from_record(saved_recipe)
result = run_workflow(config, cancellation=stop_token, progress=completed_stage)
print(result.accepted, result.candidate_cif, result.output)
```

`WorkflowConfig` records physical inputs and file paths, without atomic counts,
coordinates or a reference model. PONI supplies wavelength. The native reader
handles HDF5 and filters; h5py is not a processing dependency. The frozen protocol
and null rotations are bundled source assets, not runtime checkout lookups.

The binding releases the GIL throughout native work. Completed-stage callbacks
run on the worker thread and must use GUI signals. `SolutionCancellation` is a
cooperative native token; cancelled work joins bounded tasks and retains partial
products. In-process hosts never install the standalone CLI's process-exit guard.

Large products go into a new external directory, with exclusive creation and
source hashes checked before and after processing. `input.json`,
`source-identity.json`, per-stage products and timings, signed observations,
search alternatives, complete candidate records and CIFs retain provenance.
`WorkflowResult.accepted` is the existing bounded search assessment, not proof
of uniqueness or completed structure refinement.

## Dioptas behavior

The **Structure search** tab supplies the recipe and runs the complete native
pipeline asynchronously. It shows six completed stages, timings, selected group,
element counts, R1 and acceptance status. Unaccepted candidates remain explicitly
marked for review. Native processing and the older lattice/integration workflow
cannot be started together. Cancellation, changed inputs and shutdown prevent
stale worker results from becoming the current candidate.

The result has an independent raw-frame viewer. It reads original counts through
the native HDF5 reader, applies only the declared display orientation and overlays
saved native spot observations. Frame payload hashes must match the completed
workflow. One preview is in flight, with only the latest requested frame queued;
changed inputs discard old previews. This viewer does not overwrite the older
workflow or the powder image/calibration state. Its mask overlays are not shown.

Candidate CIF and the lossless signed-observation NPZ can be exported, and the
complete result directory opened. A small JSON recipe can be saved/reopened;
loading it does not restore old numerical results. Existing `.scw` recipes and
Dioptas's separate application feature setting are unchanged. The package APIs
remain in the main `phasesmith.single_crystal` namespace.

## Local installation before publication

Build the ordinary wheel outside the repositories and install it into the
Dioptas development environment without replacing its other dependencies:

```sh
CARGO_TARGET_DIR=/path/to/build-cache maturin build --release \
  --interpreter /path/to/Dioptas/.venv/bin/python --out /path/to/wheels
uv pip install --python /path/to/Dioptas/.venv/bin/python --no-deps \
  /path/to/wheels/phasesmith-0.8.0-PLATFORM.whl
```

Use Dioptas's `.venv/bin/dioptas` or `uv run --no-sync dioptas` for this pilot.
Do not run a normal lock/sync expecting an unpublished 0.8 registry package.
The lock remains tied to real published packages; no fabricated checksums or
machine-specific dependency paths are committed. After publication, upgrade the
PhaseSmith lock entry and test the released wheels on supported platforms.

The [complete MgSiO3 application test](dioptas-native-validation.md) records
bitwise parity, GUI responsiveness, exports, timing and remaining scope limits.
