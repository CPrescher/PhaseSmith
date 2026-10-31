# Row search (DAFi-based)

**Supported namespace:** this method is included in the 0.8 single-crystal
capability. Use `phasesmith.single_crystal`; ordinary builds include the native
kernels. The historical filename is retained for existing links. See the
[supported guide](single-crystal.md) for scope and migration. Numerical equations,
finite support and the specific limitations below remain unchanged.

This opt-in experimental alternative groups unindexed reciprocal reflections
using equidistant rows. Algorithm credit belongs to Aslandukov et al. (2022),
[DOI 10.1107/S1600576722008081](https://doi.org/10.1107/S1600576722008081).
The implementation follows the cited publication.

Build with `maturin develop --release --features spots`. The
normal package build remains feature-gated. No external checkout, executable,
file format, SciPy, spglib or process-global thread setting is required.

## API and ownership

```python
from phasesmith.single_crystal.grains import GrainSearchOptions, find_grains
from phasesmith.single_crystal.rows import RowSearchOptions, group_rows

# Exact grouping of every supplied coordinate, without cell inference.
grouping = group_rows(q, options=RowSearchOptions(workers=8))

# Grain workflow: training-only row groups, per-group cell inference,
# training consolidation, then the existing frozen reserved-data validation.
result = find_grains(
    q,
    training=training,
    method="rows",
    options=GrainSearchOptions(grid_size=256),
    row_options=RowSearchOptions(workers=8),
)
training_grouping = result.row_grouping
```

The display name is **Row search (DAFi-based)**; the selector is `rows`.
`method="fft"` remains the default. Existing ensemble search stays FFT.
`row_options` with the FFT selector raises to prevent ignored settings.

`group_rows` accepts finite real `(n,3)` arrays, with at most 100,000 positions
and absolute coordinate components at most 1,000. Inputs are copied before the
GIL is released. Labels and group indices refer to the original input order;
groups retain the exact frozen order. Labels are one-based, zero unassigned.
Mean, scale, labels and group arrays are read-only. Coincident extra observations
are conserved as unassigned occurrences. Work, resolved options, effective
direction method, cache counters and timings remain in `diagnostics`.

In the grain workflow, grouping sees **only `q[training]`**. Its indices address
that array: `np.flatnonzero(result.training)[group]` gives original input rows.
The raw `result.row_grouping.labels` are distinct from `result.labels`, which
represent final competition among inferred lattice models. A raw group is not
automatically an indexed candidate or an established physical grain.

Each sufficiently large rank-three group enters the existing PhaseSmith FFT
**cell-inference** machinery once, with one proposed lattice per group. All
small, failed and unsuccessful groups remain audited. Candidates enter the
unchanged training-only evidence consolidation and joint assignment. Reserved
coordinates are first scored after that pool is frozen. The existing rank,
support, retention and rotated-control acceptance gates remain unchanged;
rejected final models remain visible. The row method replaces whole-cloud
discovery; its later cell-inference step still uses FFTs.

Exact parity here concerns the **row grouping** and its deterministic work.
The research report's independent SFSCXRD cell/position fitter is not imported
as a runtime dependency. PhaseSmith's later fitted cells, accepted models and
ownership need not equal that separate diagnostic pipeline. Neither pipeline
establishes grain truth, chemistry, primitivity or a crystal structure.

## Frozen mathematics and selected settings

The complete imported numerical conventions are in the repository file
`crates/phasesmith-row-search/ALGORITHM.md`.
The crate's `src/lib.rs` is byte-for-byte the selected independent source,
SHA256 `e6ffdeaa3717dd451d1d3d6752aa069484e2e82585c9d0d10d5eacc49c9f1915`.
Source provenance and the original selected settings are recorded in
[`row-search-provenance.json`](row-search-provenance.json).

Canonical unique-point normalization, histogram tie order, projection offsets,
spacing/phase arithmetic, extraction batches and cache representation are
unchanged. No approximate arithmetic or new grouping equation is introduced.
PhaseSmith's `RowSearchOptions()` explicitly enables the selected sparse
configuration, including `naive_max_points=2048`, four projection offsets,
relative gaps, phase minimum three, eight native workers and 512 MiB cache.
The imported Rust `Options::default()` retains its original compatibility
defaults; callers of the crate must enable the selected settings explicitly.

Both naive pair voting and smart row nomination are published methods. The
size crossover is an engineering policy. The fixed histogram and independent
projection/gap interpretations remain documented limitations, not recovered
private DAFi behavior. The independent readable NumPy reference is
`phasesmith.single_crystal.rows_reference`; it never calls Rust.

`cache_megabytes` is the total retained-payload budget in binary MiB, not RSS
and not a per-worker amount. Zero disables it. `cache_unlimited=True` removes
the budget; no arbitrary 1 GiB ceiling is imposed. Workers are 1–8 for grouping;
`GrainSearchOptions.workers` separately controls the existing 1–4-worker fitter.
They execute serially, so those budgets are not multiplied.

`seconds`, `max_centers` and `max_evaluations` bound grouping work. Partial
results retain `budget_exhausted` and `complete=False`. `find_grains` refuses to
validate a partial grouping; use `group_rows` to inspect it. A requested group
cap is recorded separately and does not prove exhaustive recovery. Completed
outputs are worker/cache/permutation deterministic. Wall-deadline partial
outputs are not exact quality comparators. Progress callbacks surround the
grouping call and individual cell fits; the grouping call releases the GIL and
uses its native deadline, with no callback inside its hot loop.

## Validation and performance

Permanent fixtures for the three sparse target training populations contain
only frozen observed coordinates and our independent Rust results, with hashes.
They check ordered groups, labels, normalization bits and all deterministic
work/cache counters. No fixture is silently regenerated. Tests also compare
complete small runs to the NumPy reference, exercise invalid inputs, empty and
zero-spread clouds, duplicate/permutation handling, worker/cache settings,
partial budgets and actual reserved-coordinate perturbation through the full
grain entry point. The original standalone Rust equation and adaptive-policy
tests are retained. No GSAS-II oracle covers this positional grouping algorithm;
source fidelity, independent equations and the frozen standalone implementation
are the relevant comparisons. No profile term or analytical derivative changes.

```sh
python benchmarks/row_search.py --output row-benchmark.json
python benchmarks/row_search.py --artifact /path/to/dafi-sparse-grouping \
    --output frozen-row-parity.json
```

The first benchmark uses a realistic two-lattice/noise cloud at one and eight
workers and verifies exact repeated groups. The second checks all 17 frozen
full inputs and 15 detector training subsets, including both published examples.
It reads saved results and never runs original DAFi or changes earlier outputs.
The whole-package build uses normal release compilation; standalone PGO timings
do not automatically describe its compiler profile or Python boundary costs.

The original development-panel caveats remain: target recovery improved but
full-data conditional coherence fell on 1A/B/C, with a 2B cell-alias warning.
Integration does not retune those cases or promote these already examined
populations into new blinded scientific validation.
