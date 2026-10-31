"""Numerical contracts for exact FFT-search acceleration; no dataset labels."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.grains import GrainSearchOptions, _triplets, _vectors
from phasesmith.single_crystal.reciprocal import lattice_fourier
from phasesmith.single_crystal.reference import grains_reference as ref
from phasesmith.single_crystal.reference.reciprocal_reference import (
    lattice_fourier as fourier_reference,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots feature not built"
)


def test_parallel_fourier_preserves_serial_order_and_analytical_derivatives():
    rng = np.random.default_rng(753123)
    q, v = rng.normal(size=(301, 3)), rng.normal(size=(27, 3))
    serial = lattice_fourier(q, v)
    for workers in (1, 4, 4):
        engine = _core.ExperimentalLatticeEngine(workers)
        result = engine.fourier(q, v)
        np.testing.assert_array_equal(result, serial)
        np.testing.assert_allclose(result, fourier_reference(q, v), atol=1e-13, rtol=1e-12)
        for j in range(3):
            delta = np.eye(3)[j] * 2e-6
            plus, minus = engine.fourier(q, v + delta), engine.fourier(q, v - delta)
            np.testing.assert_allclose(
                (plus[:, 0] - minus[:, 0]) / 4e-6, result[:, 1 + j], atol=2e-9, rtol=1e-7
            )
            np.testing.assert_allclose(
                (plus[:, 1:4] - minus[:, 1:4]) / 4e-6,
                result[:, 4:].reshape(-1, 3, 3)[:, :, j],
                atol=1e-8,
                rtol=1e-6,
            )
        np.testing.assert_allclose(engine.fourier(q, np.zeros((1, 3)))[0, :4], [1, 0, 0, 0])
        assert engine.fourier(q, np.empty((0, 3))).shape == (0, 13)
        for bad in (np.empty((0, 3)), np.full((2, 3), np.nan), np.zeros((3, 2))):
            with pytest.raises(ValueError):
                engine.fourier(bad, v)
        with pytest.raises(ValueError, match="overflow"):
            engine.fourier(np.full((1, 3), 1e308), np.ones((1, 3)))
        with pytest.raises(ValueError, match="budget"):
            engine.fourier(np.zeros((100000, 3)), np.zeros((2001, 3)))


@pytest.mark.parametrize("n", [1, 63, 64, 65, 301])
def test_packed_phase_counts_are_exact_with_padding_snapshot_and_empty_queries(n):
    rng = np.random.default_rng(n)
    original = rng.random((11, n)) < 0.3
    original[0] = True
    original[1] = False
    ids = np.array([[0, 0, 0], [0, 1, 2], [2, 4, 8], [10, 6, 9]], dtype=np.int64)
    for workers in (1, 4):
        data = original.copy()
        bits = _core.ExperimentalLatticeEngine(workers).pack_support(data)
        data[:] = False  # The owned snapshot must survive caller mutation.
        expected = original.astype(np.int64) @ original.T.astype(np.int64)
        np.testing.assert_array_equal(bits.pair_counts(), expected)
        np.testing.assert_array_equal(
            bits.triplet_counts(ids), [np.all(original[row], axis=0).sum() for row in ids]
        )
        assert bits.triplet_counts(np.empty((0, 3), np.int64)).shape == (0,)
        for bad in (np.array([[-1, 0, 0]], np.int64), np.array([[11, 0, 0]], np.int64)):
            with pytest.raises(ValueError):
                bits.triplet_counts(bad)
    for shape in [(401, 1), (1, 0), (400, 50001)]:
        with pytest.raises(ValueError):
            _core.ExperimentalLatticeEngine(1).pack_support(np.zeros(shape, bool))


def test_batched_triplets_retain_original_candidate_set_and_tie_order():
    rng = np.random.default_rng(80731)
    q = rng.normal(size=(301, 3))
    v = rng.uniform(-5, 5, (17, 3))
    v[-1] = v[0]  # Singular and repeated directions are excluded.
    o = GrainSearchOptions(min_volume=2, max_volume=120, max_pairs=12)
    expected = ref.phase_triplets(q, v, o)
    for workers in (1, 4):
        actual = _triplets(q, v, o, _core.ExperimentalLatticeEngine(workers))
        assert actual == expected
    assert _triplets(q, np.empty((0, 3)), o) == []


def test_three_cutoff_counts_reference_boundaries_and_worker_repeat():
    rng = np.random.default_rng(45671)
    bases = np.array([np.eye(3), [[4, 1, 0], [0, 5, 1], [1, 0, 6]]], dtype=float)
    q = rng.uniform(-1, 1, (301, 3))
    q[:7] = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0.125, 0],
        [1, 0.25, 0],
        [1, 0.5, 0],
        [-1.5, 0, 0],
        [-0.5, 0, 0],
    ]
    charts = rng.normal(size=(len(q), 3, 3))
    charts[:7] = np.eye(3)
    enabled = rng.random(len(q)) > 0.2
    enabled[:7] = True
    tolerances = np.array([0.125, 0.25, 0.5])
    expected = ref.candidate_counts(q, bases, charts, enabled, tolerances, 0.25, 0.125)
    old = None
    for workers in (1, 4, 4):
        engine = _core.ExperimentalLatticeEngine(workers)
        result = engine.candidate_counts(
            q,
            bases,
            np.linalg.inv(bases).transpose(0, 2, 1),
            charts,
            enabled,
            tolerances,
            0.25,
            0.125,
        )
        np.testing.assert_array_equal(
            result[:, [0, 2, 3, 5, 6, 8]], expected[:, [0, 2, 3, 5, 6, 8]]
        )
        np.testing.assert_allclose(result, expected, atol=2e-14, rtol=2e-14)
        if old is not None:
            np.testing.assert_array_equal(result, old)
        old = result
        for bad in (
            np.array([0.25, 0.125, 0.5]),
            np.array([0.1, 0.1, 0.5]),
            np.array([0.1, np.nan, 0.5]),
        ):
            with pytest.raises(ValueError):
                engine.candidate_counts(
                    q,
                    bases,
                    np.linalg.inv(bases).transpose(0, 2, 1),
                    charts,
                    enabled,
                    bad,
                    0.25,
                    0.125,
                )


def test_cached_newton_matches_original_full_batch_serial_ascent():
    # The serial oracle intentionally repeats all old rejected trials. This
    # directly tests that retiring stationary vectors cannot affect another.
    q = np.random.default_rng(48375).uniform(-0.8, 0.8, (301, 3))
    options = GrainSearchOptions(grid_size=64, max_length=10, fft_peaks=40, basis_vectors=20)
    engine = _core.ExperimentalLatticeEngine(1)
    n = options.grid_size
    step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * options.max_length))
    field = engine.build_field(q, n, step)
    initial, _, _ = field.maxima(options.min_length, options.max_length, options.fft_peaks)
    vectors = initial.copy()
    radius = 1.5 / (n * step)
    for _ in range(20):
        current = lattice_fourier(q, vectors)
        h = -current[:, 4:].reshape(-1, 3, 3)
        h += np.maximum(0, 0.01 - np.linalg.eigvalsh(h)[:, 0])[:, None, None] * np.eye(3)
        delta = np.linalg.solve(h, current[:, 1:4, None])[:, :, 0]
        delta *= np.minimum(1, 0.3 / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        improved = np.zeros(len(vectors), bool)
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(vectors + scale * delta, initial - radius, initial + radius)
            better = (lattice_fourier(q, trial)[:, 0] > current[:, 0]) & ~improved
            vectors[better] = trial[better]
            improved |= better
        if not improved.any():
            break
    from phasesmith.single_crystal.grains import _fourier_peaks

    for workers in (1, 4):
        actual, scores, _ = _fourier_peaks(_core.ExperimentalLatticeEngine(workers), q, options)
        np.testing.assert_array_equal(actual, vectors)
        np.testing.assert_array_equal(scores, lattice_fourier(q, vectors)[:, 0])
    a = _vectors(engine, q, options)
    b = _vectors(_core.ExperimentalLatticeEngine(4), q, options)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_lazy_triplets_equal_full_prefix_including_rejected_geometry():
    rng = np.random.default_rng(56123)
    q = rng.normal(size=(1001, 3))
    v = rng.uniform(-8, 8, (40, 3))
    v[4] = v[0] + v[1] * 1e-7
    options = GrainSearchOptions(max_pairs=32, max_condition=5)
    full = ref.phase_triplets(q, v, options)
    for count in (1, 8, 32, len(full) + 1):
        for workers in (1, 4):
            result = _triplets(q, v, options, _core.ExperimentalLatticeEngine(workers), limit=count)
            assert result == full[:count]
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            _triplets(q, v, options, limit=bad)


def test_conditional_population_rank_and_prepared_direction_reuse():
    from phasesmith.single_crystal.grains import GrainConditioningOptions, _conditioned_proposals

    rng = np.random.default_rng(382913)
    q = rng.uniform(-0.7, 0.7, (301, 3))
    q[:60, 0] = np.rint(q[:60, 0] * 5) / 5
    v = np.array([[0, 7, 0], [5, 0, 0], [0, 0, 6]], dtype=float)
    prepared = (v, np.array([0.1, 0.05, 0.02]), {"prepared": True})
    options = GrainSearchOptions(grid_size=64, max_length=15)
    cond = GrainConditioningOptions(max_seeds=1, rank_by_population=True)
    calls = []
    _, audit = _conditioned_proposals(
        _core.ExperimentalLatticeEngine(1),
        q,
        options,
        cond,
        progress=lambda n, total: calls.append((n, total)),
        prepared=prepared,
    )
    assert calls == [(0, 1)]
    counts = np.array([np.count_nonzero(abs(q @ r - np.rint(q @ r)) < 0.08) for r in v])
    assert audit["phase_counts"] == counts.tolist()
    assert audit["seed_order"] == np.argsort(-counts, kind="stable").tolist()
    assert audit["seeds"][0]["seed"] == int(np.argmax(counts))
    assert audit["rank_by_population"] is True
    with pytest.raises(ValueError):
        GrainConditioningOptions(rank_by_population=1)


def test_small_serial_score_path_matches_parallel_path_at_dispatch_boundary():
    rng = np.random.default_rng(9131)
    basis = np.array([[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 6.0]])
    reciprocal = np.linalg.inv(basis).T
    for n in (200, 2048, 2049):
        q = rng.uniform(-1, 1, (n, 3))
        h, e = _core.ExperimentalLatticeEngine(1).score(q, basis, reciprocal)
        for workers in (2, 4):
            actual, error = _core.ExperimentalLatticeEngine(workers).score(q, basis, reciprocal)
            np.testing.assert_array_equal(actual, h)
            np.testing.assert_array_equal(error, e)
        expected_h = np.rint(q @ basis.T)
        np.testing.assert_array_equal(h, expected_h)
        np.testing.assert_allclose(
            e, np.linalg.norm(q - expected_h @ reciprocal, axis=1), atol=2e-15, rtol=2e-14
        )


@pytest.mark.parametrize("cosine", [np.nextafter(0.9995, 0), 0.9995, np.nextafter(0.9995, 1)])
def test_vector_shortlist_preserves_scalar_collinearity_boundary(monkeypatch, cosine):
    import phasesmith.single_crystal.grains as grains

    v = np.array([[4.0, 0, 0], [4 * cosine, 4 * np.sqrt(1 - cosine**2), 0], [0, 4, 0], [0, 0, 4]])
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    monkeypatch.setattr(grains, "_fourier_peaks", lambda *args: (v, scores, {}))
    chosen = []
    for i, r in enumerate(v):
        length = np.linalg.norm(r)
        if any(abs(np.dot(r, v[j])) / (length * np.linalg.norm(v[j])) > 0.9995 for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == 3:
            break
    options = GrainSearchOptions(fft_peaks=4, basis_vectors=3)
    actual, _, _ = grains._vectors(_core.ExperimentalLatticeEngine(1), np.ones((1, 3)), options)
    np.testing.assert_array_equal(actual, v[chosen])
