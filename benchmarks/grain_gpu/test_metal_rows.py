"""GPU row preparation checks against independent arrays and frozen Rust data."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from metal_rows import MetalRows, exact_projection
from phasesmith import _core
from phasesmith.single_crystal.rows import RowSearchOptions
from rows_compare import scientific


@pytest.mark.parametrize("n", [3, 17, 128, 1026, 12000])
@pytest.mark.parametrize("keep", [3, 4])
def test_integer_sort_and_segment_compaction(n, keep):
    rng = np.random.default_rng(712 + n)
    keys = rng.integers(0, max(2, n // 2), (8, n), dtype=np.uint32)
    actual = MetalRows().compact(keys, keep)
    expected = []
    for r, row in enumerate(keys):
        values, counts = np.unique(row, return_counts=True)
        allowed = set(values[counts >= keep])
        expected.extend((r, int(k), i) for i, k in enumerate(row) if k in allowed)
    actual = actual[np.lexsort((actual[:, 2], actual[:, 1], actual[:, 0]))]
    expected = np.asarray(sorted(expected), dtype=np.uint32).reshape(-1, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("grids", [2, 4])
def test_screened_projections_preserve_float64_keys_including_boundary_neighbors(grids):
    rng = np.random.default_rng(713)
    p = rng.uniform(-1, 1, (301, 3))
    v = rng.normal(size=(64, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    metal = MetalRows(gpu_projection=True)
    metal.audit_projection = True
    actual, _ = metal.project(p, v, grids, 0.002)
    np.testing.assert_array_equal(np.array(actual), exact_projection(p, v, grids, 0.002))
    x = np.arange(-200, 201, dtype=float) * 0.004
    p = np.column_stack(
        (
            np.r_[x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf)],
            np.zeros(len(x) * 3),
            np.linspace(-1, 1, len(x) * 3),
        )
    )
    v = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    actual, corrected = metal.project(p, v, grids, 0.002)
    assert corrected > 0
    np.testing.assert_array_equal(np.array(actual), exact_projection(p, v, grids, 0.002))
    with pytest.raises(ValueError, match="projection_epsilon"):
        metal.project(p, v, grids, 0.0001)


@pytest.mark.parametrize("fixture", ["dataset-1-C", "dataset-2-A", "dataset-2-C"])
@pytest.mark.parametrize("gpu_projection", [False, True])
def test_frozen_groups_all_fields_and_native_baseline(fixture, gpu_projection):
    path = Path(__file__).resolve().parents[2] / "tests/fixtures/row-search" / f"{fixture}.npz"
    with np.load(path) as z:
        q = z["points"]
        labels = z["labels"]
    options = json.dumps(RowSearchOptions()._native_options())
    expected = json.loads(_core.experimental_row_search(q, options))
    m = MetalRows(gpu_projection=gpu_projection)
    baseline = json.loads(m.run(q, options, gpu=False))
    actual = json.loads(m.run(q, options))
    assert scientific(actual) == scientific(baseline) == scientific(expected)
    np.testing.assert_array_equal(actual["labels"], labels)


@pytest.mark.parametrize("workers,cache", [(1, 0), (8, 0), (1, 512), (8, 512)])
def test_workers_caches_duplicates_and_zero_spread(workers, cache):
    m = MetalRows(gpu_projection=True)
    rng = np.random.default_rng(714)
    q = rng.uniform(-1, 1, (90, 3))
    q[:, 2] = 0
    q = np.concatenate([q, q[:3]])
    opts = replace(RowSearchOptions(), workers=workers, cache_megabytes=cache)._native_options()
    text = json.dumps(opts)
    assert scientific(json.loads(m.run(q, text))) == scientific(
        json.loads(_core.experimental_row_search(q, text))
    )


def test_callback_failure_and_deadline_are_not_silent_fallbacks():
    q = np.random.default_rng(715).uniform(-1, 1, (90, 3))
    m = MetalRows()
    opts = RowSearchOptions()._native_options()
    opts["projection_grids"] = 0
    with pytest.raises(ValueError, match="fixed 2/4"):
        m.run(q, json.dumps(opts))
    opts = RowSearchOptions(seconds=0)._native_options()
    result = json.loads(m.run(q, json.dumps(opts)))
    assert result["budget_exhausted"] and not m.records
    for bad in [np.zeros((2, 2)), np.full((2, 3), np.nan), np.full((2, 3), 1001.0)]:
        with pytest.raises(ValueError):
            m.run(bad, json.dumps(opts))
