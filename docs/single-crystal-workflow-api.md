# Original-frame workflow API and input contract

`phasesmith.single_crystal.workflow.run_workflow` calls the same native Rust
coordinator as the [standalone example](single-crystal-native-workflow.md).
Python validates/translates the configuration and notifications; numerical stages
and HDF5/NPZ/CIF products run natively. The [first calculation](single-crystal-quickstart.md)
is a smaller runnable example that needs no measured data.

## Supported recipe

Workflow v1 accepts one EIGER-style HDF5 unsigned-count dataset with detector
pixel-size metadata, a uniform-detector PONI calibration, a supplied primitive
orthorhombic metric and 1–8 non-H element types. It uses the frozen v40
unknown-symmetry recipe, explicit illumination/DAC masks, gain 1/read noise 0,
up to ten workers and separate internal memory budgets. General TIFF/multigrain
input, arbitrary cell metrics and standalone structure refinement are outside
this composed recipe. Lower-level kernels have their own broader contracts.

The [HDF5 reader](native-hdf5.md) accepts owned uint16/uint32 frames in `(frame,y,x)`
order and rejects unvalidated virtual datasets and unavailable filters. It does
not infer orientation or angles. PONI supplies wavelength; incompatible HDF5/PONI
pixel sizes and unsupported detector orientation/distortion overrides fail.
The cell alone is not evidence for intensity symmetry or a unique group.

## Complete configuration

The following is the exposed MgSiO3 recipe, not a generic acquisition default.
Replace paths and independently establish every physical input for another scan.
Neither atomic coordinates/counts nor an exact group enter this configuration.

```python
import json
from pathlib import Path
from phasesmith.single_crystal.workflow import WorkflowConfig, run_workflow

config = WorkflowConfig(
    raw_path="/data/mgsio3/scan.h5",
    dataset="/entry_0000/ESRF-ID11/eiger/data",
    poni_path="/data/mgsio3/calibration.poni",
    output="/data/results/new-mgsio3-search",  # new, outside any Git checkout
    cell=(8.8101, 5.1824, 18.2288, 90, 90, 90),
    elements=("Mg", "Si", "O"),
    first=2,
    last=143,
    omega_zero=-36.0,
    omega_step=0.5,
    exposure=0.5,
    rotation_axis=(-1.0, 0.0, 0.0),
    dac_axis=(0.0, 0.0, 1.0),
    dac_half_opening=35.0,
    vertical_flip=True,
    effort="auto",
    provenance="Exposed MgSiO3 control; provisional spindle angles and declared DAC geometry.",
)
Path("recipe.json").write_text(json.dumps(config.to_record(), indent=2))
config = WorkflowConfig.from_record(json.loads(Path("recipe.json").read_text()))
result = run_workflow(config, progress=lambda stage, seconds: print(stage, seconds))
print(result.accepted, result.candidate_cif, result.output)
```

Verify the acquisition/DAC values against the [frozen source protocol](https://github.com/CPrescher/PhaseSmith/blob/develop/validation/enstatite-raw-workflow-v40.json)
when reproducing that example. Geometry is user-supplied evidence, not a value
to select by whether the expected structure is recovered.

| Configuration field | Units and validation |
| --- | --- |
| `raw_path`, `poni_path` | Existing original count file and matching calibration; source identity checked before/after processing |
| `dataset` | Explicit HDF5 dataset path; no directory/dataset search |
| `output` | New external directory; existing paths and Git-checkout output locations are rejected |
| `cell` | Six values: a/b/c in Å, angles in degrees; positive lengths and exactly 90° angles in v1 |
| `elements` | 1–8 distinct canonical non-H symbols; element types, not full-cell counts or site labels |
| `first`, `last` | Zero-based original indices, **both inclusive**; 5–225 selected exposures and `last < frame_count` |
| `omega_zero` | Degrees at original frame zero, not at `first` |
| `omega_step`, `exposure` | Positive degrees per original frame / exposure width; distinct quantities even when numerically equal |
| `rotation_axis` | Three finite unit-vector components in the common right-handed laboratory frame |
| `dac_axis` | Unit direction of the DAC opening at omega zero in that same frame |
| `dac_half_opening` | Degrees, strictly between 0 and 90; not a full cone angle |
| `vertical_flip` | Explicit Boolean reversal of raw rows; no inferred array orientation |
| `effort` | `fast`, `auto` or `thorough`; default `auto`, both passes share the original deadline |
| `provenance` | Nonempty origin/assumption record for supplied geometry and priors |

Exposure i starts at `omega_zero + i*omega_step` and has the declared width.
The raw array uses row/column order; detector observations use x=column/y=row.
Pixel centres are zero-based. PONI coordinates must describe the declared
orientation; a display flip is not an independent calibration correction.

The record schema is `phasesmith.single-crystal-workflow-config.v1`.
`to_record()` contains small JSON-safe configuration only. `from_record()`
rejects another schema or unexpected fields. Loading a recipe does not load
cached numerical results or authorize reuse of changed source data.

## Cancellation and application hosts

Pass `SolutionCancellation` from `phasesmith.single_crystal.solution` to
`run_workflow(..., cancellation=token)`, then call `token.cancel()` from the
application's control thread. Cancellation is cooperative, joins bounded tasks,
raises `InterruptedError` and retains written partial products. A pre-cancelled
token stops before file work. Native work releases the GIL. Progress callbacks
receive completed-stage `(name, seconds)` on the calling worker thread; GUI
updates must be marshalled through application signals. Callback exceptions stop
the workflow. The in-process library never terminates the GUI process.

The standalone executable additionally has a hard process guard. This is a host
policy, separate from the library's cooperative deadline. See the
[native boundary](single-crystal-native-workflow.md#typed-library-boundary).

## Results and failures

`WorkflowResult.output` is the durable output `Path`; `report` is a read-only
top-level mapping of the native report. `accepted` is true only when the saved
acceptance value is literally true. `candidate_cif` returns the selected-setting
CIF, falls back to the P1 candidate if present, or returns `None`.

| Product | Inspect it for |
| --- | --- |
| `input.json`, `source-identity.json` | Actual recipe, count/calibration identity and hashes |
| Stage NPZ/JSON products | Masks, spot/link assignments, geometry, teachers, signed measurements and quality flags |
| `preliminary-symmetry.json`, `space-group-evidence.json` | Measured evidence, contradictions and insufficient constraints |
| `space-group-search.json` | Actual passes, acceptance/rejection, budget omissions and alternative settings |
| `candidate/space-group.cif`, `candidate/best.cif` | Selected setting and retained full-cell model; presence is not acceptance |
| `workflow-result.json`, `timings.json`, `worker-complete.json`, `failure.json` on failure | Stop reasons, included observations and actual elapsed stages; nested timings are not additive |

An accepted candidate satisfies this bounded method's checks; it does not prove
uniqueness or complete refinement. A correct atomic model can remain unaccepted
because its group is ambiguous. An unaccepted or missing candidate is a retained
scientific outcome, not an instruction to discard weak/negative observations.
Quality flags and unmeasured crossings are not observed zero intensities.

| Symptom | Check |
| --- | --- |
| Missing native capability | Installed version/path; use an ordinary 0.8 wheel, not an old editable 0.7 package |
| HDF5/filter/pixel-size failure | Explicit dataset, reader limits, EIGER metadata and matching PONI; no silent h5py fallback |
| Too few tracks or no lattice | Orientation, exposure convention, masks, coverage and competing hypotheses before changing scientific thresholds |
| Memory admission failure | The reported input-only dense/decoded/payload bound; a configured dense budget is not total RSS |
| Existing output rejected | Select a new external directory; results are not overwritten |
| Source changed / partial output | Keep diagnostics, establish input identity and start a new run |
| Candidate rejected / budget exhausted | Read both passes, missing evidence and alternatives; low R1/count alone is insufficient |

The [limitations and measured validation](dioptas-native-validation.md) describe
the exposed MgSiO3 control. The [recovery panel](structure-search-recovery.md)
records remaining chemistry and group failures. Use the
[lower-level HKL guide](experimental-structure-solution.md) for supported supplied-
symmetry/composition searches independent of this original-frame recipe.
