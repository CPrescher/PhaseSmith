# First single-crystal calculation

This tutorial runs with NumPy and the ordinary PhaseSmith wheel. It needs no
private measurements, external solver, SciPy, Gemmi or Python HDF5 package.
It demonstrates detection, HKL input, bounded structure search and saved results.
The separate [original-frame workflow](single-crystal-workflow-api.md) explains
how to configure measured HDF5 data.

## Install the matching version

For the published 0.8 release:

```sh
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install 'phasesmith>=0.8,<0.9'
python -c "import phasesmith; print(phasesmith.__version__, phasesmith.__file__)"
```

For a source checkout, follow the [development installation](https://github.com/CPrescher/PhaseSmith#development)
in the repository README. Source builds need Rust 1.89+, a C compiler, CMake
3.26+ and Git; installed wheels do not.

## Run the self-contained example

For a published 0.8 installation, use the
[versioned script](https://github.com/CPrescher/PhaseSmith/blob/v0.8.0/examples/single_crystal/first_search.py)
from the matching release tag. Save it as
`examples/single_crystal/first_search.py`, or use that path from a source checkout
of the same tag.
Select a new output directory:

```sh
python examples/single_crystal/first_search.py --output /absolute/path/to/new-tutorial
```

The script generates one Gaussian detector spot and 196 ideal HKL intensities
from a synthetic single-Fe-atom P1 cell. It writes the HKL file, reads it through
the public adapter, runs one bounded native search and exports its result.
All atom coordinates remain in the generator. The search receives only the
cell, P1 symmetry and explicit **full-cell composition Fe=1**.

This deliberately simple example checks installation and API use. Its unit
uncertainties are illustrative weights, not experimental error estimates. With
one atom the absolute origin is unidentifiable; a tiny R1 here is not evidence
of general structure recovery. Measured validation is reported separately in
the [65-case comparison](solution-holdout-results.md).

The printed report should contain one spot, 196 input reflections, one candidate
and `target_met: true`. Exact R1 digits and elapsed time can vary by platform.
The output contains:

| Path | Meaning |
| --- | --- |
| `synthetic.hkl` | Generated `h k l I sigma(I)` rows; input provenance is hashed by the reader |
| `tutorial.json` | Declared synthetic assumptions, priors and observed outcome |
| `search/result.json` | Search options, stop reason, candidates and phasing history |
| `search/density-and-observations.npz` | Original observations, derived phasing amplitudes and density; no pickle |
| `search/candidate-p1.cif` | Preferred full-cell candidate, if one was returned |

Existing output directories are rejected. Use another path for a repeat.

## Replace synthetic HKLs with measurements

The essential public API is:

```python
from phasesmith.crystallography import UnitCell
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure, write_solution

data = read_hkl("measured.hkl")
result = solve_structure(
    data,
    cell=UnitCell(6, 7, 8, 90, 90, 90),  # replace with independently justified values
    space_group=space_group_by_symbol("P1").space_group,
    composition={"Fe": 1},  # full-cell non-H counts, not atom fractions
    options=SolutionOptions(max_seconds=60),
)
write_solution(result, "new-search-output")
print(result.termination, result.target_met)
if result.best is not None:
    print(result.best.r1, result.best.species)
```

The illustrative cell/composition are not recommendations for an unknown sample.
Supply intensities, not amplitudes, with positive finite sigma(I). Signed and
negative observations are retained; derived phasing amplitudes use the declared
nonnegative approximation. The strict reader rejects a SHELX instruction/atom
trailer after the reflection terminator. Keep such a trailer separately rather
than passing reference coordinates into a search.

`solve_structure` uses supplied symmetry and composition. For element types
without counts, use the [named element-search configuration](single-crystal.md#reproducible-search-configuration).
The complete native workflow can also [assess unknown symmetry](space-group-search.md).
These are distinct input contracts; changing one into another changes the evidence.

## Read a result before trusting it

`result.best` can be absent or incorrect. `target_met` means the requested
numerical target was met under the supplied priors; it is not the complete
workflow's independent `WorkflowResult.accepted` assessment. Inspect stop reasons,
alternatives, reflection coverage, chemistry and external structural evidence.
Search-internal coordinate/ADP polishing is not a complete structure-refinement
workflow. See [result interpretation and troubleshooting](single-crystal-workflow-api.md#results-and-failures).
