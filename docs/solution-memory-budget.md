# Explicit dense memory budget for structure search

The supported experimental search controls now include
`SolutionOptions.dense_memory_budget_bytes`. The default is **16,000,000 bytes**,
exactly the historical two-million-`f64` limit. An explicit value from 8 through
1,073,741,824 bytes admits larger dense reflection buffers. Equality at the
boundary is accepted; partial trailing bytes cannot admit another `f64`.

This is a limit for controlled dense reflection buffers, not total process RSS.
P1 scattering/derivative admission, constrained-model Jacobians and their linear
operator paths share the same policy. Existing FFT-grid/atom/reflection limits,
the 32 MiB geometry cache, 8 MiB normal cache, 4 MiB repair-column cache and
separate orbit/peak-cache bounds remain independent. Accepted/trial buffers,
observations, work vectors, multiple candidate models and concurrent workers
can coexist. A 64 MiB setting does not promise a 64 MiB process footprint.

## Input-only admission

For `N` full-cell atoms and `M` unique allowed symmetry/Friedel-expanded measured
reciprocal rows, retain the existing conservative P1 envelope:

```
required_dense_bytes = 8 * M * (5*N + 7)
dense_elements = B // 8
memory_atom_cap = max(0, ((dense_elements // M) - 7) // 5)
maximum_atoms = min(128, ceil(cell_volume_angstrom3 / 8), memory_atom_cap)
```

Here `//` denotes integer floor division; negative capacity saturates at zero. Rust uses checked multiplication for allocation
admission and reports both required and configured bytes on failure. Its
preflight uses the same native observation preparation as search, including
absences, duplicates and supplied symmetry. The independent Python reference
enumerates every candidate count 1--128 and tests the inequalities directly.
Neither computation accepts reference coordinates or an expected atom count.

For constrained models, the exact independent parameter count `P` gives the
separate check `8*P*M <= B` on original rows before any optional equivalent-family
reduction. Each prepared model retains the policy through cached, reused,
blocked and matrix-free-normal-operator paths. This last path retains a dense
Jacobian while avoiding a dense normal matrix; it is not a Jacobian-free method.

No profile, structure-factor, phasing, support, derivative, weighting, ranking,
stopping or optimizer equation changes. Changing only a nonbinding budget must
preserve numerical values and deterministic fixed-work search results exactly.

## Python and Rust use

```python
from phasesmith.experimental.solution import (
    SolutionOptions, estimate_solution_memory, solve_structure_elements,
)

budget = 64 * 1024**2
estimate = estimate_solution_memory(
    observations, cell=cell, space_group=group,
    dense_memory_budget_bytes=budget,
)
result = solve_structure_elements(
    observations, cell=cell, space_group=group, elements=("Mg", "Si", "O"),
    options=SolutionOptions(dense_memory_budget_bytes=budget),
)
```

Omitting `maximum_atoms` now selects the input-only cap under the configured
budget. With the default budget, already admissible searches retain their old
volume-derived cap. Inputs whose old automatic volume-only request would exceed
memory now receive an admissible memory-limited cap. Zero capacity fails before
search. An explicit `maximum_atoms` is never silently reduced: exceeding the
budget remains an error. A numerical target or successful process exit still
does not establish correct or complete structure recovery.

The configured byte policy is retained in serialized result options. Older
option records that omit it load with the historical default. Known-composition
search applies the same budget to its P1 model/polishing, element-swap evaluation
and supplied-symmetry candidate refinement. Its composition remains an explicit
input contract, separate from count-free element search.

Rust callers construct `DenseMemoryBudget::new(bytes)` and set
`SolveOptions.dense_memory_budget`. The public input-only
`estimate_solution_memory` reports expanded rows and volume/memory caps.
Existing low-level constructors and model/linear functions retain their default
budget; explicit model variants and prepared-model constructors carry a checked
budget. No global or thread-local allocation policy is introduced.

The known-composition CLI accepts `--dense-memory-budget-bytes`. The external
enstatite probe accepts the same option, uses native preflight, and keeps its
reference count/coordinate grading after the search worker stops:

```sh
python benchmarks/enstatite_structure_search.py \
  --data-root /path/to/enstatite-files \
  --observations /path/to/ours-own-sigma-observations.npz \
  --dense-memory-budget-bytes 67108864 \
  --output runs/enstatite-memory-64mib
```

Raw files remain external and checksum-verified. The earlier 36-atom probe
summaries and frozen 65-case solver evidence are unchanged. This new code is
not substituted for the archived frozen executable, and the exposed 65 cases
are not rerun as a new holdout.

## Verification

Rust tests check inclusive byte boundaries, overflow, unsupported budgets,
large P1 and symmetry Jacobians, and every linear-operator route beyond the old
limit. Python tests compare native preflight against independent exhaustive
admission on deterministic randomized budgets, check absence/duplicate handling,
automatic and explicit caps, serialized policy, invalid inputs and bitwise
fixed-work search invariance. Existing independent values/derivative,
normalization and support tests remain applicable because their equations are
unchanged. GSAS-II's powder oracle does not cover this allocation policy.

The native Criterion benchmark covers fused values/analytical derivatives over
6,400 reflections and 24 or 64 atoms. The 24-atom default/larger-budget pair
measures the nonbinding policy on identical work; the 64-atom case exercises an
envelope beyond the historical bound. It is a kernel workload, not an
end-to-end structure-search speed claim.
