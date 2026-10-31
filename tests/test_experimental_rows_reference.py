"""Independent NumPy checks of declared boundaries and projection geometry."""

import numpy as np
from phasesmith.experimental import rows_reference as r

OPTIONS = dict(
    min_row=4,
    projection_epsilon=0.002,
    relative_epsilon=0.02,
    gap_epsilon=0.02,
    relative_gap=True,
    kmax=5,
)


def test_antipodal_direction():
    for v in [[1.0, 2.0, 3.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
        assert r.key(np.array(v), 0.0033) == r.key(-np.array(v), 0.0033)


def test_closed_interval():
    assert r.interval_support([3.0, 5.0], dict(kmax=1, relative_epsilon=0.25)) == 2


def test_phase_boundary_and_wrap():
    assert len(r.phase_subset([(0.0, 0), (1.25, 1)], 1.0, 0.25)) == 2
    assert len(r.phase_subset([(0.0, 0), (1.2500001, 1)], 1.0, 0.25)) == 1
    assert len(r.phase_subset([(0.99, 0), (2.01, 1)], 1.0, 0.021)) == 2


def test_mixed_projection_boundary():
    q = np.array(
        [[0.0, 0.0039, 0.0019], [0.2, 0.0041, 0.0021], [0.4, 0.0039, 0.0021], [0.6, 0.0041, 0.0019]]
    )
    two = r.projected_grids(
        q, range(4), np.array([1.0, 0.0, 0.0]), dict(OPTIONS, projection_grids=2)
    )
    four = r.projected_grids(
        q, range(4), np.array([1.0, 0.0, 0.0]), dict(OPTIONS, projection_grids=4)
    )
    assert not any(two)
    assert max(len(row) for grid in four for row in grid) == 4


def test_bounded_rows_do_not_chain():
    q = np.array([[i * 0.05, 0.001 + (i // 2) * 0.002, 0.001] for i in range(6)])
    grids = r.projected_grids(
        q, range(6), np.array([1.0, 0.0, 0.0]), dict(OPTIONS, projection_grids=0)
    )
    assert max(len(row) for grid in grids for row in grid) == 4
