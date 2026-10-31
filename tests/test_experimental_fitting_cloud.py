"""Ordered fitting selections and unchanged least-squares proposal results."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal import grains as G
from phasesmith.single_crystal.reference.grains_reference import fitting_selection

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


@pytest.mark.parametrize("count", [0, 1, 301, 2048, 2049, 5003])
def test_fitting_snapshot_matches_all_original_selected_bits_and_independent_equations(count):
    rng = np.random.default_rng(51283 + count)
    q = rng.uniform(-1, 1, (count, 3))
    if count >= 7:
        q[:7] = [
            [0, -0.0, 0],
            [1, 0.125, 0],
            [1, 0.25, 0],
            [1, 0.5, 0],
            [-1.5, 0, 0],
            [-0.5, 0, 0],
            [0.5, 0, 0],
        ]
    for workers in (1, 4):
        engine = _core.ExperimentalLatticeEngine(workers)
        for source in [q, q[::-1], np.asfortranarray(q)]:
            original = source.copy()
            cloud = engine.prepare_fitting(source)
            for a in [np.eye(3), np.array([[4, 1, 0], [0, 5, 1], [1, 0, 6]], float)]:
                b = np.linalg.inv(a).T
                for tolerance in (0.125, 0.25, 0.5):
                    h, error = engine.score(original, a, b)
                    ids = np.flatnonzero((error < tolerance) & np.any(h != 0, axis=1))
                    expected = h[ids], original[ids], ids, error[ids]
                    actual = cloud.selected(a, b, tolerance)
                    independent = fitting_selection(original, a, b, tolerance)
                    for x, y, z in zip(actual, expected, independent, strict=True):
                        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
                        np.testing.assert_array_equal(x.view(np.uint8), z.view(np.uint8))
    snapshot = engine.prepare_fitting(q)
    expected = snapshot.selected(np.eye(3), np.eye(3), 0.5)
    del engine
    q[:] = np.nan
    for actual, previous in zip(
        snapshot.selected(np.eye(3), np.eye(3), 0.5), expected, strict=True
    ):
        np.testing.assert_array_equal(actual.view(np.uint8), previous.view(np.uint8))


def test_fitting_snapshot_invalid_shapes_values_and_tolerances():
    engine = _core.ExperimentalLatticeEngine(1)
    for q in [np.zeros((1, 2)), np.full((1, 3), np.nan), np.zeros((100001, 3))]:
        with pytest.raises(ValueError):
            engine.prepare_fitting(q)
    cloud = engine.prepare_fitting(np.zeros((3, 3)))
    for a, b, tol in [
        (np.zeros((2, 3)), np.eye(3), 0.1),
        (np.eye(3), np.zeros((2, 3)), 0.1),
        (np.full((3, 3), np.nan), np.eye(3), 0.1),
        (np.eye(3), np.eye(3), 0),
        (np.eye(3), np.eye(3), np.nan),
        (np.eye(3), np.eye(3), np.inf),
    ]:
        with pytest.raises(ValueError):
            cloud.selected(a, b, tol)


def test_fitting_snapshot_keeps_complete_proposal_results_for_signal_aliases_and_failures():
    rng = np.random.default_rng(92834)
    engine = _core.ExperimentalLatticeEngine(4)
    options = G.GrainSearchOptions()
    for _ in range(8):
        truth = np.diag(rng.uniform(3, 8, 3)) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = rng.integers(-4, 5, (100, 3))
        q = h @ np.linalg.inv(truth).T + rng.normal(0, 0.0005, h.shape)
        q = np.concatenate([q, rng.uniform(-1, 1, (201, 3))])
        cloud = engine.prepare_fitting(q)
        for scale in [1.0, 2.0, 0.5]:
            start = scale * truth + rng.normal(0, 0.001, (3, 3))
            for cores in [(), (24, 48, 96)]:
                original = G._refine(engine, q, start, options, integer_cores=cores)
                actual = G._refine(engine, q, start, options, integer_cores=cores, _cloud=cloud)
                if original is None:
                    assert actual is None
                else:
                    np.testing.assert_array_equal(actual.view(np.uint8), original.view(np.uint8))
