"""Retained FFT lines, radial boundaries and prepared fused Fourier moments."""

import numpy as np
import pytest
from phasesmith import _core

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


@pytest.mark.parametrize("n", [16, 32, 64, 128, 256])
def test_windowed_maxima_preserve_full_fft_values_vectors_ties_and_total(n):
    rng = np.random.default_rng(81231 + n)
    q = rng.uniform(-1.5, 1.5, (301, 3))
    step = 3.01 / (n - 4)
    dense = _core.ExperimentalLatticeEngine(1).build_field(q, n, step)
    spacing = 1 / (n * step)
    for minimum, maximum in [(spacing, 15.0), (2 * spacing, 4 * spacing)]:
        expected = dense.maxima(minimum, maximum, 400)
        for workers in (1, 4):
            actual = _core.ExperimentalLatticeEngine(workers).windowed_maxima(
                q, n, step, minimum, maximum, 400
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            assert actual[2] == expected[2]


def test_windowed_sparse_empty_lines_and_all_equal_scores_have_original_index_order():
    engine = _core.ExperimentalLatticeEngine(4)
    for q in [np.zeros((3, 3)), np.array([[0.0, 0.0, 0.0], [0.125, 0.0, 0.125]])]:
        for bounds in [(0.5, 4.0), (1.0, 40.0), (40.0, 41.0), (1.0, 1.0)]:
            old = engine.build_field(q, 16, 0.25).maxima(*bounds, 97)
            new = engine.windowed_maxima(q, 16, 0.25, *bounds, 97)
            np.testing.assert_array_equal(new[0], old[0])
            np.testing.assert_array_equal(new[1], old[1])
            assert new[2] == old[2]


def test_windowed_snapshot_and_boundaries_reject_invalid_inputs():
    engine = _core.ExperimentalLatticeEngine(1)
    q = np.array([[0.1, 0.2, -0.3]])
    for n, step, lo, hi, limit in [
        (15, 0.1, 1.0, 3.0, 80),
        (258, 0.1, 1.0, 3.0, 80),
        (32, 0.0, 1.0, 3.0, 80),
        (32, 0.001, 1.0, 3.0, 80),
        (32, 0.1, 0.0, 3.0, 80),
        (32, 0.1, 3.0, 1.0, 80),
        (32, 0.1, 1.0, np.inf, 80),
        (32, 0.1, 1.0, 3.0, 0),
        (32, 0.1, 1.0, 3.0, 1001),
    ]:
        with pytest.raises(ValueError):
            engine.windowed_maxima(q, n, step, lo, hi, limit)
    for bad in [np.empty((0, 3)), np.full((1, 3), np.nan), np.zeros((1, 2))]:
        with pytest.raises(ValueError):
            engine.windowed_maxima(bad, 32, 0.1, 1.0, 3.0, 80)
    with pytest.raises(ValueError, match="direct-grid"):
        engine.windowed_maxima(np.zeros((1, 3)), 32, np.nextafter(0.0, 1.0), 1.0, 3.0, 80)


def test_prepared_cloud_is_an_owned_snapshot_with_exact_fused_moments_and_derivatives():
    from phasesmith.single_crystal.reciprocal import lattice_fourier
    from phasesmith.single_crystal.reference.reciprocal_reference import (
        lattice_fourier as reference,
    )

    rng = np.random.default_rng(518273)
    for count in (301, 1024, 1025, 2001):
        q, vectors = rng.normal(size=(count, 3)), rng.normal(size=(27, 3))
        expected = lattice_fourier(q, vectors)
        for workers in (1, 4):
            data = q.copy()
            cloud = _core.ExperimentalLatticeEngine(workers).prepare_fourier(data)
            data[:] = np.nan
            actual = cloud.evaluate(vectors)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(actual, reference(q, vectors), atol=1e-13, rtol=1e-12)
            assert cloud.evaluate(np.empty((0, 3))).shape == (0, 13)
            for j in range(3):
                step = np.eye(3)[j] * 2e-6
                plus, minus = cloud.evaluate(vectors + step), cloud.evaluate(vectors - step)
                np.testing.assert_allclose(
                    (plus[:, 0] - minus[:, 0]) / 4e-6,
                    actual[:, 1 + j],
                    atol=2e-9,
                    rtol=1e-7,
                )
                np.testing.assert_allclose(
                    (plus[:, 1:4] - minus[:, 1:4]) / 4e-6,
                    actual[:, 4:].reshape(-1, 3, 3)[:, :, j],
                    atol=1e-8,
                    rtol=1e-6,
                )
            for bad in (np.full((2, 3), np.nan), np.zeros((2, 2))):
                with pytest.raises(ValueError):
                    cloud.evaluate(bad)
    engine = _core.ExperimentalLatticeEngine(1)
    for bad in (np.empty((0, 3)), np.zeros((2, 2)), np.full((2, 3), np.nan)):
        with pytest.raises(ValueError):
            engine.prepare_fourier(bad)
    with pytest.raises(ValueError, match="overflow"):
        engine.prepare_fourier(np.full((1, 3), 1e308)).evaluate(np.ones((1, 3)))
    with pytest.raises(ValueError, match="budget"):
        engine.prepare_fourier(np.zeros((100000, 3))).evaluate(np.zeros((2001, 3)))


def test_hybrid_evaluator_reuses_one_small_batch_snapshot_and_keeps_large_batch_path():
    from phasesmith.single_crystal.grains import _fourier_evaluator

    rng = np.random.default_rng(91182)
    q = rng.normal(size=(301, 3))
    native = _core.ExperimentalLatticeEngine(4)
    calls = dict(ordinary=[], prepared=[], snapshots=0)

    class Cloud:
        def __init__(self, cloud):
            self.cloud = cloud

        def evaluate(self, v):
            calls["prepared"].append(len(v))
            return self.cloud.evaluate(v)

    class Engine:
        def fourier(self, points, v):
            calls["ordinary"].append(len(v))
            return native.fourier(points, v)

        def prepare_fourier(self, points):
            calls["snapshots"] += 1
            return Cloud(native.prepare_fourier(points))

    evaluate = _fourier_evaluator(Engine(), q)
    for count in (400, 33, 32, 4, 1):
        v = rng.normal(size=(count, 3))
        np.testing.assert_array_equal(evaluate(v), native.fourier(q, v))
    assert calls == dict(ordinary=[400, 33], prepared=[32, 4, 1], snapshots=1)


@pytest.mark.parametrize("n", [18, 30, 48, 96])
def test_sparse_columns_plan_reuse_and_spherical_skips_are_bit_exact(n):
    rng = np.random.default_rng(73910 + n)
    engine = _core.ExperimentalLatticeEngine(4)
    # Include empty X slabs, duplicate accumulation, exact grid planes, and
    # non-power-of-two transforms; repeat a plan after another grid size.
    clouds = [
        rng.uniform(-0.3, 0.3, (37, 3)),
        rng.uniform(-1.1, 1.1, (3001, 3)),
        np.repeat([[0.0, -0.0, 0.25], [0.5, 0.0, -0.5]], 201, axis=0),
    ]
    for edge in [n, 16, n]:
        step = 2.21 / (edge - 4)
        spacing = 1 / (edge * step)
        for q in clouds:
            for lo, hi in [(spacing, 3.01 * spacing), (0.5, 15.0)]:
                expected = engine.build_field(q, edge, step).maxima(lo, hi, 73)
                actual = engine.windowed_maxima(q, edge, step, lo, hi, 73)
                for x, y in zip(actual[:2], expected[:2], strict=True):
                    np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
                assert actual[2] == expected[2]
