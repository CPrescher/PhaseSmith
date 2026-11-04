# Single-crystal diffraction

Starting with release 0.8, single-crystal analysis is a normal
PhaseSmith capability. Python uses `phasesmith.single_crystal`; native consumers
use `phasesmith::spots`, `phasesmith::row_search`, `phasesmith::solution` and
`phasesmith::workflows::single_crystal`.
Ordinary wheels include the kernels and the bundled native HDF5 reader. The
canonical workflow runs from original frames through structure search and
candidate assessment. There is no standalone single-crystal structure-refinement
workflow in this release.

## Scientific scope

Start with the [runnable first calculation](single-crystal-quickstart.md), then
the [complete original-frame configuration/API guide](single-crystal-workflow-api.md).
The shared [geometry](mathematics/single-crystal-geometry.md),
[intensity](mathematics/single-crystal-intensities.md) and
[structure-search](mathematics/structure-search.md) references collect equations,
units, supports and ownership for both Rust and Python users.

Supported stages include deterministic spot detection and linking, reciprocal
mapping, lattice hypotheses and FFT/Row grain discovery, positional detector
calibration, prediction, explicit frame validity/shadow/DAC masks, summation and
learned-profile integration, strong-profile fitting, detector/sensor corrections,
empirical relative scaling and structure search. Inputs, priors, finite supports,
signed intensities, quality flags and termination reasons remain explicit.
Application hosts own scan configuration, loader callbacks and output locations.

Geometry fitting refines indexed spot positions. Internal coordinate/ADP
polishing belongs to structure search. Neither operation supplies a general
single-crystal refinement workflow. Search currently uses non-resonant X-ray
intensities, a supplied cell and non-H element types or optional composition priors. Symmetry may be supplied
or assessed through [native space-group search](space-group-search.md). A low R1 or
correct atom count alone does not establish a correct or unique structure.
The measured search has an existing ADP-bound stagnation limitation; report its
termination and diagnostics rather than claiming every candidate polish converged.
Research strategies remain opt-in and carry their documented qualifications.

The [Rust-only workflow coordinator](single-crystal-native-workflow.md) runs
the complete original-frame path without an interpreter. Its first recipe uses
a supplied primitive-orthorhombic metric, explicit mask assumptions and the
versioned search profile; other low-level methods retain their existing APIs.
The current example uses [automatic effort](space-group-search-speed.md): try
short coarse phasing first, then use finer/broader search only after candidate
acceptance fails. Explicit `fast` and `thorough` modes remain available.

## Rust package boundary

Single-crystal numerical kernels ship in one `phasesmith-single-crystal` crate:
`spots`, `solution` and `row_search` are feature-selectable modules. The usual
facade paths shown above remain available. HDF5 lives in its separate adapter
crate, and the complete application-neutral coordinator lives in
`phasesmith-workflows`. See [migration](migration-0.8.md) for direct imports from
the former unpublished component crates.

## Python API

```python
from phasesmith.single_crystal import (
    frame_validity,
    geometry_refinement,
    grains,
    integration,
    prediction,
    profiles,
    reduction,
    rotation,
    solution,
    spots,
)
from phasesmith.single_crystal.reciprocal import PanelGeometry
from phasesmith.io.hdf5 import Hdf5Frames
from phasesmith.io.hkl import HklData, read_hkl
```

The root loads submodules on demand and avoids flattening hundreds of names.
Each public module has an explicit `__all__`. Shared cells, symmetry, structures,
execution policy and file adapters retain their existing owners; no duplicate
`UnitCell` or observation type is introduced.

| Area | Public submodules |
| --- | --- |
| Original frames, masks and positions | `spots`, `rotation`, `reciprocal`, `frame_validity`, `frame_input` |
| Lattices, cells and grains | `indexing`, `cell_metrics`, `cell_comparison`, `orthorhombic`, `grains`, `rows` |
| Grain selection controls | `ownership`, `retention`, `recovery` |
| Prediction and geometry | `prediction`, `geometry_refinement` |
| Integration and profiles | `integration`, `integration_quality`, `profiles`, `profile_refinement` |
| Background and corrections | `omega_background`, `intensity_response`, `sensor_offsets`, `intensity_uncertainty` |
| Reduction, scaling and persistence | `reduction`, `reduction_io`, `reduction_cache`, `empirical_scaling`, `scan_scaling` |
| Search and assessment | `solution`, `solution_symmetry`, `search` |
| Complete native file workflow | `workflow` (`WorkflowConfig`, `WorkflowResult`, `run_workflow`) |

Native kernels evaluate numerical values/analytical derivatives. The canonical
original-frame workflow's coordination, design construction and file products
also run in Rust. Python's `workflow` module is its optional configuration/array
and notification boundary. Separate lower-level convenience APIs, including
`prediction.refine_crystal`, retain documented NumPy optimization coordinators;
the historical Python raw-workflow runner remains an independent regression path.
Importing or running the supported native path needs no SciPy, h5py, hdf5plugin,
Gemmi, Dioptas, Qt or external solver.
SciPy remains an independent validation/reference dependency, including the
explicit reference backend for omega background; it is not a production backend
requirement. CIF convenience and post-search external assessment are separate.

## Reproducible search configuration

Known full-cell composition uses `solution.solve_structure`. Element-type-only
search uses `solution.solve_structure_elements`, or an explicit named profile:

```python
from phasesmith.crystallography import UnitCell
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.search import ElementSearchConfig, search_structure_elements
from phasesmith.single_crystal.solution import SolutionOptions, write_solution

config = ElementSearchConfig.validated_v1(
    options=SolutionOptions(
        phasing="dual_space",
        completion=False,
        trials=4,
        cycles=800,
        max_seconds=60,
        dense_memory_budget_bytes=128 * 1024**2,
    )
)
result = search_structure_elements(
    read_hkl("/external/data/reflections.hkl"),
    cell=UnitCell(18.2, 8.8, 5.2, 90, 90, 90),
    space_group=space_group_by_symbol("Pbca").space_group,
    elements=["Mg", "Si", "O"],
    configuration=config,
)
write_solution(result, "/external/results/new-search")
print(result.termination, result.target_met, result.to_record()["search_configuration"])
```

The cell above is illustrative, not the MgSiO3 calibration. Supply independently
justified priors. The named profile expands the reviewed frozen combination,
including its arithmetic-changing strategies, into a complete immutable record.
It adds no external program/corpus path. Explicit low-level defaults remain
unchanged. Conflicting algorithm keyword overrides are rejected by the named
entry point. A memory bound is an input-only resource limit, never a reference
atom count; use `estimate_solution_memory` before a large search.

## Native builds and packaging

```toml
[dependencies]
phasesmith = "0.8"
```

The facade defaults to `single-crystal`, which includes `spots`, `solution` and
`row-search`. Use `default-features = false` for a minimal powder build, or select
those capabilities individually. Pre-release experimental Cargo feature names are removed in 0.8; use the
capability names above. HDF5 stays optional for
Rust: add `features = ["hdf5"]` for original detector files. It is included in
normal Python builds through the wheel configuration.

Bundled HDF5 builds need a C compiler, CMake >=3.26 and Git, but an installed
wheel needs no installed HDF5 or Python filter package. See
[native HDF5 input](native-hdf5.md) for supported filters and resource bounds.
The numerical core remains dependency-light; FFT, row discovery, search and
I/O remain separate crates. No interpreter is required for those Rust APIs.

## Migration and saved data

Use `phasesmith.single_crystal.<module>`. Experimental import wrappers are
removed in 0.8 following migration of their Dioptas consumer. Independent
numerical references are retained under `phasesmith.single_crystal.reference`
for validation, outside the supported production exports. Historical Python
pickles containing experimental module paths are not migrated automatically;
portable, pickle-free observation/recipe formats remain supported.

The supported module CLIs are:

```sh
python -m phasesmith.single_crystal.solution --help
python -m phasesmith.single_crystal.reduction --help
```

New search summaries use `phasesmith.single-crystal-solution.v1`.
`upgrade_solution_record` copies legacy v4/v5 metadata without altering candidates,
histories or scientific evidence; `result.to_record(legacy=True)` explicitly
retains the old summary form. Historical profile-case, reduction and cache schema
identifiers remain unchanged and readable. They are data-format identifiers,
not current support-status flags. No archived result is silently rewritten.
Canonical implementation hashes deliberately invalidate old content-addressed
reduction cache keys; teacher/source provenance and original archives remain.

## Measured example and evidence

The [MgSiO3 raw workflow](enstatite-raw-workflow.md) is an exposed development
regression with declared provisional spindle angles and aperture assumptions,
not an untouched holdout or absolute-intensity calibration. All eight source
files (about 470 MB) and large products remain external to Git. The example
reads original frames, masks shadows, indexes, calibrates geometry, learns and
fits profiles, scales signed intensities, searches, then assesses the saved
candidate after the numerical worker exits. Source HKLs, atomic coordinates,
atom counts and archived measurements do not enter that worker.

The [frame-input timing report](frame-input-optimization.md) records every staged
gain and regression. Before namespace promotion, full v23 takes 18.56–18.65 s
on the measured M4 Pro and matches v11 numerical results bitwise. This is not a
cross-platform timing guarantee. Independent transfer coverage remains bounded;
the previously measured 65-case search corpus is now exposed regression data.

The [0.8 release candidate](release-candidate-0.8.md) records the promotion checks,
installed-wheel/source distribution evidence and remaining release work.


The completed 0.8 installed-wheel v24 checks take 19.06/19.39 s and reproduce
all retained v11 numerical arrays, candidate records and histories bitwise, with
R1 .04085182070733226. See the release candidate for actual distribution checks
and timing limits; the earlier v23 pair is retained as historical evidence.

The [Dioptas application integration](dioptas-native-workflow.md) exposes the
complete known-cell HDF5 recipe through the ordinary Python wheel, with native
processing, cooperative cancellation and stage notifications.
