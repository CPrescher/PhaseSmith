"""Frozen real grouping, independent equations and training-only integration."""

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import rows_reference
from phasesmith.experimental.grains import GrainSearchOptions, find_grains
from phasesmith.experimental.rows import RowSearchOptions, group_rows

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_row_search"), reason="experimental-spots feature not built"
)
FIXTURES = Path(__file__).parent / "fixtures/row-search"


def small_cloud():
    return np.asarray(
        [
            [i * 0.19, j * 0.23 + ((i * 7 + j * 11 + k * 13) % 5) * 1e-6, k * 0.31]
            for i, j, k in product(range(4), range(3), range(3))
        ]
    )


@pytest.mark.parametrize("record", json.loads((FIXTURES / "manifest.json").read_text()))
def test_frozen_sparse_grouping_is_identical(record):
    path = FIXTURES / f"{record['tag']}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["fixture_sha256"]
    with np.load(path) as z:
        result = group_rows(z["points"])
        np.testing.assert_array_equal(result.labels, z["labels"])
        np.testing.assert_array_equal(result.mean.view(np.uint64), z["mean"].view(np.uint64))
        np.testing.assert_array_equal(result.scale.view(np.uint64), z["scale"].view(np.uint64))
        assert len(result.groups) == len(z["offsets"]) - 1
        for group, lo, hi in zip(result.groups, z["offsets"][:-1], z["offsets"][1:], strict=True):
            np.testing.assert_array_equal(group, z["flat"][lo:hi])
        assert {k: result.diagnostics[k] for k in record["work"]} == record["work"]
        assert result.complete and result.options.naive_max_points == 2048


@pytest.mark.parametrize("smart,crossover", [(False, 0), (True, 0), (True, 2048)])
def test_full_small_run_matches_independent_numpy(smart, crossover):
    q = small_cloud()
    options = RowSearchOptions(directions=512, groups=6, smart=smart, naive_max_points=crossover)
    expected = rows_reference.run(q, options._native_options())
    actual = group_rows(q, options=options)
    np.testing.assert_array_equal(actual.labels, expected)
    assert np.count_nonzero(actual.labels) >= 20


def test_workers_cache_duplicates_strides_and_permutation():
    q = small_cloud()
    options = RowSearchOptions(directions=512, groups=6)
    expected = group_rows(q, options=options)
    for workers in (1, 2, 4, 8):
        for budget, unlimited in ((0, False), (512, False), (0, True)):
            actual = group_rows(
                q,
                options=replace(
                    options, workers=workers, cache_megabytes=budget, cache_unlimited=unlimited
                ),
            )
            np.testing.assert_array_equal(actual.labels, expected.labels)
            for a, b in zip(actual.groups, expected.groups, strict=True):
                np.testing.assert_array_equal(a, b)
    reverse = group_rows(q[::-1], options=options)
    np.testing.assert_array_equal(reverse.labels[::-1], expected.labels)
    padded = group_rows(np.r_[q, q[:1], q[:1]], options=options)
    np.testing.assert_array_equal(padded.labels[: len(q)], expected.labels)
    assert not padded.labels[-2:].any() and padded.diagnostics["duplicate_observations"] == 2
    assert padded.diagnostics["effective_smart"] is False
    for array in (expected.labels, expected.mean, expected.scale, *expected.groups):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 0},
        {"workers": 9},
        {"workers": True},
        {"seconds": np.inf},
        {"seconds": -1},
        {"seconds": True},
        {"direction_epsilon": np.nan},
        {"projection_epsilon": 0},
        {"projection_grids": 3},
        {"min_row": 1},
        {"gap_min_row": 1},
        {"phase_min_row": 1},
        {"relative_epsilon": 0.5},
        {"gap_epsilon": 0},
        {"kmax": 101},
        {"cache_megabytes": -1},
        {"cache_megabytes": 2**63},
        {"smart": 1},
        {"naive_max_points": 1.2},
        {"max_evaluations": -1},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        RowSearchOptions(**kwargs)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 2)),
        [[np.nan, 0, 0]],
        [[np.inf, 0, 0]],
        [[1001, 0, 0]],
        np.zeros((100001, 3)),
        np.zeros((2, 3), complex),
        np.zeros((2, 3), bool),
    ],
)
def test_invalid_points(points):
    with pytest.raises(ValueError):
        group_rows(points)


def test_empty_partial_output_and_private_boundary():
    result = group_rows(np.empty((0, 3)))
    assert result.labels.shape == (0,) and not result.groups and result.complete
    np.testing.assert_array_equal(result.scale, [1.0, 1.0, 1.0])
    partial = group_rows(small_cloud(), options=RowSearchOptions(seconds=0))
    assert not partial.complete and partial.diagnostics["budget_exhausted"]
    with pytest.raises(TypeError):
        group_rows(small_cloud(), options={})
    with pytest.raises(ValueError):
        _core.experimental_row_search(np.zeros((2, 2)), "{}")
    with pytest.raises(ValueError):
        _core.experimental_row_search(np.array([[np.nan, 0.0, 0.0]]), "{}")
    with pytest.raises(ValueError):
        _core.experimental_row_search(small_cloud(), '{"workers":0}')


def grain_cloud():
    rng = np.random.default_rng(49)
    h = np.asarray(list(product(range(-3, 4), repeat=3)))
    h = h[np.any(h != 0, axis=1)]
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    q = h @ np.linalg.inv(np.diag([4.3, 5.1, 7.7]) @ rotation).T
    q += rng.normal(0, 1e-5, h.shape)
    return np.r_[q, rng.uniform(-0.8, 0.8, (40, 3))]


def grain_options():
    return GrainSearchOptions(
        grid_size=64,
        max_length=12,
        fft_peaks=64,
        basis_vectors=32,
        max_pairs=32,
        cell_refinements=16,
        max_grains=3,
        null_rotations=8,
    )


def test_row_grain_path_retains_grouping_and_freezes_before_reserved():
    q = grain_cloud()
    train = np.arange(len(q)) % 5 != 0
    result = find_grains(q, training=train, options=grain_options(), method="rows")
    assert result.method == "row_grains" and sum(c.accepted for c in result.candidates) == 1
    np.testing.assert_allclose(result.candidates[0].volume, 4.3 * 5.1 * 7.7, rtol=2e-4)
    raw = group_rows(q[train])
    np.testing.assert_array_equal(result.row_grouping.labels, raw.labels)
    changed = q.copy()
    changed[~train] = np.random.default_rng(155).uniform(-2, 2, (np.count_nonzero(~train), 3))
    alternate = find_grains(changed, training=train, options=grain_options(), method="rows")
    np.testing.assert_array_equal(alternate.row_grouping.labels, raw.labels)
    np.testing.assert_array_equal(alternate.translation_vectors, result.translation_vectors)
    np.testing.assert_array_equal(alternate.labels[train], result.labels[train])
    for a, b in zip(alternate.candidates, result.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
    assert result.diagnostics[0]["display_name"] == "Row search (DAFi-based)"
    assert any(
        r["status"] == "too_small_or_rank_deficient" for r in result.diagnostics[0]["groups"]
    )
    with pytest.raises(RuntimeError, match="budget"):
        find_grains(
            q, options=grain_options(), method="rows", row_options=RowSearchOptions(seconds=0)
        )


def test_default_fft_path_and_cancellation_remain_explicit():
    q = grain_cloud()
    default = find_grains(q, options=grain_options())
    explicit = find_grains(q, options=grain_options(), method="fft")
    assert default.method == explicit.method == "fft_grains"
    assert default.row_grouping is explicit.row_grouping is None
    np.testing.assert_array_equal(default.labels, explicit.labels)
    for a, b in zip(default.candidates, explicit.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
    with pytest.raises(ValueError, match="method"):
        find_grains(q, method="unknown")
    with pytest.raises(ValueError, match="row_options"):
        find_grains(q, row_options=RowSearchOptions())

    def cancel(*_):
        raise RuntimeError("user cancelled")

    with pytest.raises(RuntimeError, match="user cancelled"):
        find_grains(q, method="rows", progress=cancel)
