"""Independent native checks and blind multigrain recovery, positions only."""

import json
from itertools import product

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import grains_reference as ref
from phasesmith.experimental.grains import GrainSearchOptions, find_grains, integer_lattice_basis

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots feature not built"
)


@pytest.mark.parametrize("n", [32, 64])
def test_deposition_fft_against_numpy_and_exact_worker_repeat(n):
    q = np.random.default_rng(n).uniform(-0.65, 0.65, (301, 3))
    old = None
    for workers in (1, 4, 4):
        result = _core.ExperimentalLatticeEngine(workers).build_field(q, n, 0.05).to_numpy()
        np.testing.assert_allclose(result, ref.fourier_field(q, n, 0.05), rtol=3e-5, atol=1e-5)
        assert result[0, 0, 0] == pytest.approx(len(q), abs=3e-5)
        if old is not None:
            np.testing.assert_array_equal(result, old)
        old = result


def test_maxima_boundaries_ties_and_stable_rank():
    n = 16
    values = np.zeros((n, n, n // 2 + 1), dtype=np.float32)
    values[1, 0, 0] = values[-1, 0, 0] = 10
    values[0, 3, 2] = 8
    engine = _core.ExperimentalLatticeEngine(4)
    field = engine.field_from_array(values, 0.1)
    vectors, scores, total = field.maxima(0.5, 3.0, 40)
    freq = np.fft.fftfreq(n, 0.1)
    positive = np.fft.rfftfreq(n, 0.1)
    xyz = np.stack(np.meshgrid(freq, freq, positive, indexing="ij"), axis=-1)
    length = np.linalg.norm(xyz, axis=-1)
    masked = values.copy()
    masked[(length < 0.5) | (length > 3)] = -np.inf
    keep = np.isfinite(masked)
    for dx, dy, dz in product((-1, 0, 1), repeat=3):
        other = np.roll(masked, (dx, dy), axis=(0, 1))
        other = other[:, :, np.clip(np.arange(n // 2 + 1) + dz, 0, n // 2)]
        keep &= masked >= other
    ids = np.flatnonzero(keep)
    order = np.lexsort((ids, -masked.ravel()[ids]))[:40]
    np.testing.assert_array_equal(vectors, xyz.reshape(-1, 3)[ids[order]])
    np.testing.assert_array_equal(scores, masked.ravel()[ids[order]])
    assert total == len(ids)


def test_score_assignment_independent_and_exact_boundaries():
    rng = np.random.default_rng(823)
    models = np.array(
        [np.diag([4.0, 5.0, 6.0]), [[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 6.0]]]
    )
    q = np.r_[rng.uniform(-1, 1, (500, 3)), [[0.25, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]]]
    engine = _core.ExperimentalLatticeEngine(4)
    reciprocal = np.linalg.inv(models).transpose(0, 2, 1)
    for a, b in zip(models, reciprocal, strict=True):
        h, error = engine.score(q, a, b)
        expected = np.rint(q @ a.T)
        np.testing.assert_array_equal(h, expected)
        np.testing.assert_allclose(error, np.linalg.norm(q - expected @ b, axis=1), atol=2e-16)
    actual = engine.assign(q, models, reciprocal, 0.03, 0.8)
    expected = ref.assign(q, models, 0.03, 0.8)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    identity = np.eye(3)[None]
    labels, amb, multi = engine.assign(
        np.array([[1.0, 0, 0], [1.125, 0, 0], [0.0, 0, 0]]), identity, identity, 0.125, 0.8
    )
    np.testing.assert_array_equal(labels, [1, 0, 0])  # strict boundary; exclude 000
    doubled = np.repeat(identity, 2, axis=0)
    labels, amb, multi = engine.assign(np.array([[1.0, 0, 0]]), doubled, doubled, 0.1, 0.8)
    assert labels[0] == 0 and amb == multi == 1


def mixed_cloud(same_cell=False):
    rng = np.random.default_rng(33)
    parts, truths = [], []
    for edges in ([4.3, 5.1, 7.7], [4.3, 5.1, 7.7] if same_cell else [5.2, 6.2, 8.3]):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        a = np.diag(edges) @ rotation
        h = rng.integers(-5, 6, (500, 3))
        h = np.unique(h[np.any(h != 0, axis=1)], axis=0)
        parts.append(h @ np.linalg.inv(a).T + rng.normal(0, 0.0005, h.shape))
        truths.append(a)
    return np.concatenate([*parts, rng.uniform(-1, 1, (100, 3))]), truths, list(map(len, parts))


@pytest.mark.parametrize("same_cell", [False, True])
def test_blind_two_grains_and_same_cell_orientations(same_cell):
    q, truths, sizes = mixed_cloud(same_cell)
    options = GrainSearchOptions(max_length=15, grid_size=128, max_grains=4)
    result = find_grains(q, options=options)
    assert len(result.candidates) == 2 and all(c.accepted for c in result.candidates)
    for a, c in zip(truths, result.candidates, strict=True):
        assert c.volume == pytest.approx(abs(np.linalg.det(a)), rel=0.002)
        change = c.direct_basis @ np.linalg.inv(a)
        np.testing.assert_allclose(change, np.rint(change), atol=0.015)
    for i, block in enumerate(np.split(result.labels[: sum(sizes)], [sizes[0]]), 1):
        assert np.mean(block == i) > 0.97
    json.dumps(result.diagnostics, allow_nan=False)
    assert not result.labels.flags.writeable


def test_heldout_cannot_change_discovery_pool_or_training_ownership():
    q, _, _ = mixed_cloud()
    training = np.arange(len(q)) % 5 != 0
    options = GrainSearchOptions(max_length=15, grid_size=128, max_grains=3)
    first = find_grains(q, training=training, options=options)
    q[~training] = np.random.default_rng(920).uniform(-1, 1, ((~training).sum(), 3))
    second = find_grains(q, training=training, options=options)
    assert len(first.candidates) == len(second.candidates)
    for a, b in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
        np.testing.assert_array_equal(a.hkl[training], b.hkl[training])
        assert a.accepted and not b.accepted
    np.testing.assert_array_equal(first.labels[training], second.labels[training])
    assert first.diagnostics[:2] == second.diagnostics[:2]


def test_random_cloud_rejected_and_cancellation_propagates():
    q = np.random.default_rng(8221).uniform(-1, 1, (350, 3))
    result = find_grains(q, options=GrainSearchOptions(max_length=15, grid_size=128, max_grains=2))
    assert not any(c.accepted for c in result.candidates)

    def cancel(*args):
        raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        find_grains(q, progress=cancel)


def test_integer_generated_lattice_not_an_assumed_cell():
    h = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6], [2, 4, 6]])
    b = integer_lattice_basis(h)
    assert round(abs(np.linalg.det(b))) == 48
    np.testing.assert_allclose(h @ np.linalg.inv(b), np.rint(h @ np.linalg.inv(b)))
    assert integer_lattice_basis(h[:2]) is None
    assert round(abs(np.linalg.det(integer_lattice_basis(np.r_[h, [[1, 1, 1]]])))) == 4


@pytest.mark.parametrize(
    "kw",
    [
        {"workers": 5},
        {"grid_size": 512},
        {"max_grains": 0},
        {"null_rotations": 2},
        {"seed": -1},
        {"grid_size": True},
        {"max_length": np.nan},
        {"min_volume": -1},
        {"ambiguity_ratio": 1},
    ],
)
def test_options_reject_invalid_budgets(kw):
    with pytest.raises(ValueError):
        GrainSearchOptions(**kw)


def test_native_and_array_boundaries():
    engine = _core.ExperimentalLatticeEngine(1)
    for q, n, step in [
        (np.zeros((0, 3)), 32, 0.1),
        (np.ones((3, 3)), 17, 0.1),
        (np.ones((3, 3)), 512, 0.1),
        (np.full((3, 3), np.nan), 32, 0.1),
        (np.ones((3, 3)), 32, 0),
        (np.ones((3, 3)) * 100, 32, 0.1),
    ]:
        with pytest.raises(ValueError):
            engine.build_field(q, n, step)
    q, _, _ = mixed_cloud()
    for bad in [np.ones(len(q)), np.ones((len(q), 1), bool)]:
        with pytest.raises(ValueError, match="training"):
            find_grains(q, training=bad)
    with pytest.raises(ValueError, match="insufficient"):
        find_grains(q[:5])
    with pytest.raises(ValueError):
        integer_lattice_basis(np.ones((3, 3)))


@pytest.mark.parametrize(
    "basis",
    [
        [[4.1, 0, 0], [1.2, 5.3, 0], [-0.8, 0.5, 6.7]],
        [[5.2, 0, 0], [-2.6, 4.5033321, 0], [0, 0, 8.4]],
        [[0, 3.2, 3.2], [3.2, 0, 3.2], [3.2, 3.2, 0]],
    ],
)
def test_general_metric_without_a_crystal_system_prior(basis):
    from test_experimental_indexing import synthetic

    q, truth = synthetic(basis, outliers=100)
    result = find_grains(q, options=GrainSearchOptions(max_length=15, grid_size=256, max_grains=3))
    best = result.candidates[0]
    assert best.accepted and best.validation_fraction > 0.7
    assert best.volume == pytest.approx(abs(np.linalg.det(truth)), rel=0.01)
    change = best.direct_basis @ np.linalg.inv(truth)
    np.testing.assert_allclose(change, np.rint(change), atol=0.025)
