"""Research comparison bookkeeping: no DIALS installation or image data needed."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "position_comparison", Path(__file__).parents[1] / "benchmarks/compare_dials_positions.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_common_partition_excludes_scan_edges_crossings_long_and_invalid_tracks():
    data = dict(
        ranges=np.array([[0, 1], [1, 2], [4, 6], [6, 8], [10, 19], [25, 27], [74, 76], [12, 13]]),
        xyz=np.zeros((8, 3)),
    )
    data["xyz"][-1, 0] = np.nan
    keep, training = comparison.selection(data, 76)
    np.testing.assert_array_equal(keep, [False, True, False, True, False, True, False, False])
    assert not training[1] and training[3] and not training[5]


def test_position_match_is_one_to_one_and_maximizes_cardinality_before_distance():
    pytest.importorskip("scipy")
    # Greedily pairing a0-b0 would lose a1, which cannot pair with b1.
    a = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    b = np.array([[-0.1, 0.0, 0.0], [1.9, 0.0, 0.0]])
    pairs, da, db = comparison.match_positions(a, b)
    np.testing.assert_array_equal(pairs, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(da, [2, 1])
    np.testing.assert_array_equal(db, [2, 1])


def test_position_match_uses_separate_inclusive_spatial_and_angular_limits():
    pytest.importorskip("scipy")
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 1.0], [2.0001, 0.0, 0.0], [0.0, 0.0, 1.0001]])
    pairs, da, db = comparison.match_positions(a, b)
    np.testing.assert_array_equal(pairs, [[0, 0]])
    np.testing.assert_array_equal(da, [1])
    np.testing.assert_array_equal(db, [1, 0, 0])
    empty, _, _ = comparison.match_positions(a, b[:0])
    assert empty.shape == (0, 2)
