"""Exact discrete ranking and bounded native Fourier proposal refinement."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import grains_reference as ref
from phasesmith.experimental.grains import GrainSearchOptions, _triplet_keys, _triplets
from phasesmith.experimental.reciprocal import lattice_fourier

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 17, 80])
def test_native_triplet_keys_preserve_strongest_pair_ties_and_lexicographic_set(size):
    support = np.random.default_rng(8229 + size).random((size, 65)) < 0.3
    support[:2] = False
    counts = support.astype(np.int64) @ support.T.astype(np.int64)
    for workers in (1, 4):
        bits = _core.ExperimentalLatticeEngine(workers).pack_support(support)
        for budget in (1, 12, 128):
            actual = bits.candidate_keys(budget)
            expected = np.array(_triplet_keys(counts, size, budget), dtype=np.int64).reshape(-1, 3)
            np.testing.assert_array_equal(actual, expected)


def test_native_count_volume_key_rank_preserves_duplicates_and_ties_and_validates():
    support = np.random.default_rng(734).random((17, 65)) < 0.4
    bits = _core.ExperimentalLatticeEngine(4).pack_support(support)
    keys = np.array([[0, 2, 6], [0, 2, 6], [1, 4, 5], [3, 9, 15], [2, 2, 2]], np.int64)
    volumes = np.array([2.0, 2.0, 1.0, 3.0, 4.0])
    order, counts = bits.rank_triplets(keys, volumes)
    expected = np.array([np.all(support[k], axis=0).sum() for k in keys])
    np.testing.assert_array_equal(counts, expected)
    assert order.tolist() == sorted(
        range(len(keys)), key=lambda i: (-expected[i], volumes[i], tuple(keys[i]), i)
    )
    assert bits.rank_triplets(np.empty((0, 3), np.int64), np.empty(0))[0].size == 0
    for k, v in [(keys[:, :2], volumes), (keys, volumes[:-1]), (keys - 18, volumes),
                 (keys, np.zeros(5)), (keys, np.full(5, np.nan))]:
        with pytest.raises(ValueError):
            bits.rank_triplets(k, v)
    for budget in (0, 257):
        with pytest.raises(ValueError):
            bits.candidate_keys(budget)


def test_native_triplets_keep_numpy_geometry_and_full_or_lazy_reference_prefix():
    rng = np.random.default_rng(38812)
    q = rng.uniform(-1, 1, (301, 3))
    v = rng.uniform(-8, 8, (40, 3))
    v[4] = v[0] + 1e-7 * v[1]
    options = GrainSearchOptions(max_pairs=32, max_condition=5)
    full = ref.phase_triplets(q, v, options)
    engine = _core.ExperimentalLatticeEngine(4)
    assert _triplets(q, v, options, engine) == full
    for count in (1, 8, 32):
        assert _triplets(q, v, options, engine, limit=count) == full[:count]


def test_native_newton_independent_equations_fused_moments_bounds_and_worker_repeat():
    rng = np.random.default_rng(48375)
    q = rng.uniform(-0.8, 0.8, (301, 3))
    engine = _core.ExperimentalLatticeEngine(1)
    n = 64
    step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * 10))
    initial = engine.windowed_maxima(q, n, step, 1.5, 10, 40)[0]
    radius = 1.5 / (n * step)
    reference, rm = ref.fourier_newton(q, initial, radius)
    previous = None
    for workers in (1, 4, 4):
        v, m, stats = _core.ExperimentalLatticeEngine(workers).refine_fourier(q, initial, radius)
        # Closed-form spectral damping and pivoted LU differ from LAPACK in
        # roundoff. These resolved fixture maxima retain their local basin.
        np.testing.assert_allclose(v, reference, atol=3e-6, rtol=0)
        np.testing.assert_allclose(m, rm, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(m, lattice_fourier(q, v))
        assert np.all(abs(v - initial) <= radius)
        assert np.all(m[:, 0] >= lattice_fourier(q, initial)[:, 0])
        assert np.all(stats[:, 0] <= 81) and stats[:, 1].sum() > 0
        if previous is not None:
            for a, b in zip(previous, (v, m, stats), strict=True):
                np.testing.assert_array_equal(a, b)
        previous = v, m, stats
    assert GrainSearchOptions().native_fourier_refinement is False
    with pytest.raises(ValueError):
        GrainSearchOptions(native_fourier_refinement=1)


def test_native_newton_constant_cloud_exact_cache_and_invalid_inputs():
    engine = _core.ExperimentalLatticeEngine(1)
    q = np.zeros((3, 3))
    initial = np.array([[1.0, 2.0, 3.0], [-0.0, 1.0, -2.0]])
    v, m, counts = engine.refine_fourier(q, initial, 0.5)
    np.testing.assert_array_equal(v, initial)
    np.testing.assert_array_equal(m[:, 0], [1, 1])
    assert not m[:, 1:].any()
    # Exact keys distinguish -0.0 from +0.0 even in this constant objective.
    assert counts.tolist() == [[1, 4], [2, 3]]
    assert engine.refine_fourier(q, np.empty((0, 3)), 0.5)[0].shape == (0, 3)
    for points, vectors, radius in [(q[:0], initial, 0.5), (q, initial, 0.0),
                                    (q, initial, np.inf), (q, np.full((2, 3), np.nan), 0.5),
                                    (q, np.zeros((401, 3)), 0.5), (q + 1001, initial, 0.5)]:
        with pytest.raises(ValueError):
            engine.refine_fourier(points, vectors, radius)


def test_batched_first_support_matches_scalar_rounding_zero_exclusion_and_strict_gate():
    from phasesmith.experimental.grains import _refine, _refine_many

    rng = np.random.default_rng(82021)
    bases = np.array([np.eye(3), [[4, 1, 0], [0, 5, 1], [1, 0, 6]]], dtype=float)
    q = rng.uniform(-1, 1, (301, 3))
    q[:7] = [[0, 0, 0], [1, 0, 0], [1, 0.125, 0], [1, 0.25, 0],
             [1, 0.5, 0], [-1.5, 0, 0], [-0.5, 0, 0]]
    reciprocal = np.array([np.linalg.inv(a).T for a in bases])
    for workers in (1, 4):
        engine = _core.ExperimentalLatticeEngine(workers)
        for tolerance in (0.125, 0.25, 0.5):
            expected = []
            for a, b in zip(bases, reciprocal, strict=True):
                h, error = engine.score(q, a, b)
                expected.append(int(((error < tolerance) & np.any(h != 0, axis=1)).sum()))
            np.testing.assert_array_equal(
                engine.initial_support(q, bases, reciprocal, tolerance), expected
            )
        options = GrainSearchOptions()
        result = _refine_many(engine, q, bases, options)
        for a, actual in zip(bases, result, strict=True):
            expected = _refine(engine, q, a, options)
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)
        for rows, b, tol in [(bases[:, :2], reciprocal, 0.1), (bases, reciprocal[:1], 0.1),
                              (bases, reciprocal, 0.0), (bases, reciprocal, np.nan)]:
            with pytest.raises(ValueError):
                engine.initial_support(q, rows, b, tol)


def test_two_lane_fused_moments_keep_scalar_bits_and_bulk_dispatch_for_odd_tails():
    rng = np.random.default_rng(88213)
    q = rng.normal(size=(1001, 3))
    q[:4] = [[0.0, -0.0, 0.0], [1.0, 0.0, -1.0], [-1.0, -0.0, 1.0], [0, 0, 0]]
    for count in (0, 1, 2, 3, 64, 65):
        vectors = rng.normal(size=(count, 3))
        expected = lattice_fourier(q, vectors)
        for workers in (1, 4):
            engine = _core.ExperimentalLatticeEngine(workers)
            np.testing.assert_array_equal(
                engine.fourier_tiled(q, vectors).view(np.uint64), expected.view(np.uint64)
            )
            np.testing.assert_array_equal(
                engine.fourier(q, vectors).view(np.uint64), expected.view(np.uint64)
            )


def test_fourier_reuse_is_scoped_copies_outputs_and_keeps_every_semantic_key():
    from dataclasses import replace

    from phasesmith.experimental import discovery_cache as cache

    calls = []

    @cache.fourier
    def pure(engine, q, options):
        calls.append(1)
        return q[:2].copy(), np.array([0.1, 0.2]), dict(total=17)

    options = GrainSearchOptions()
    q = np.arange(12, dtype=float).reshape(4, 3)
    engine = object()

    @cache.scope
    def run():
        v, s, layout = pure(engine, q, options)
        v[:] = -1
        s[:] = -1
        layout["total"] = -1
        actual = pure(engine, q.copy(), options)
        np.testing.assert_array_equal(actual[0], q[:2])
        np.testing.assert_array_equal(actual[1], [0.1, 0.2])
        assert actual[2] == dict(total=17)
        pure(engine, q[::-1], options)
        pure(engine, q.astype(np.float32), options)
        pure(engine, q, replace(options, fft_peaks=193))
        pure(engine, q, replace(options, native_fourier_refinement=True))
        assert cache.statistics()["hits"] == 1
        assert cache.statistics()["misses"] == 5

    run()
    run()
    assert len(calls) == 10 and cache.statistics() is None

    @cache.scope
    def fail():
        pure(engine, q, options)
        raise RuntimeError("test")

    with pytest.raises(RuntimeError):
        fail()
    assert cache.statistics() is None
    bounded = cache._Cache(max_bytes=1000, max_entries=1)
    key = (b"first",)
    value = (q[:2], np.zeros(2), {})
    bounded.put(key, value)
    bounded.put(key, value)
    assert bounded.bytes <= 1000 and len(bounded.rows) == 1
    bounded.put((b"second",), value)
    assert bounded.get(key) is None and len(bounded.rows) == 1
