"""Geometry equations, derivatives and blind lattice recovery, no target fixture."""

from types import SimpleNamespace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import reciprocal_reference as ref
from phasesmith.experimental.indexing import (
    IndexingOptions,
    cell_parameters,
    find_lattices,
    reduce_basis,
)
from phasesmith.experimental.reciprocal import (
    PanelGeometry,
    correct_positions,
    lattice_fourier,
    reciprocal_vectors,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_reciprocal"), reason="experimental-spots feature not built"
)


def panel(axis=(1, 0, 0)):
    return PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.09,
        pixel1=8e-5,
        pixel2=9e-5,
        wavelength=0.4,
        rot1=0.02,
        rot2=-0.01,
        rot3=0.08,
        axis=axis,
    )


def test_geometry_randomized_independent_reference_and_inverse_rotation():
    rng = np.random.default_rng(80723)
    for _ in range(5):
        g = panel(rng.normal(size=3))
        samples = rng.uniform([0, 0, -350], [2047, 2047, 350], (100, 3))
        q = reciprocal_vectors(samples, g)
        np.testing.assert_allclose(q, ref.reciprocal_vectors(samples, g), atol=2e-15, rtol=2e-14)
        fixed = samples.copy()
        fixed[:, 2] = 0
        np.testing.assert_allclose(
            np.linalg.norm(q, axis=1),
            np.linalg.norm(reciprocal_vectors(fixed, g), axis=1),
            atol=2e-15,
        )
    assert reciprocal_vectors(np.empty((0, 3)), g).shape == (0, 3)


def test_direct_beam_pixel_centres_and_no_two_pi():
    g = PanelGeometry.from_poni(
        distance=1, poni1=0.5, poni2=0.5, pixel1=1, pixel2=1, wavelength=2, axis=[0, 1, 0]
    )
    q = reciprocal_vectors(np.array([[0, 0, 0], [1, 0, 0], [1, 0, 90.0]]), g)
    np.testing.assert_allclose(q[0], 0, atol=0)
    np.testing.assert_allclose(q[1], [0, 1 / np.sqrt(8), (1 / np.sqrt(2) - 1) / 2], atol=1e-16)


@pytest.mark.parametrize(
    "vectors,wavelength",
    [
        (np.zeros((5, 3)), 1),
        (np.ones((4, 3)), 1),
        (np.eye(5, 3), -1),
        (np.full((5, 3), np.nan), 1),
    ],
)
def test_invalid_geometry(vectors, wavelength):
    with pytest.raises(ValueError):
        PanelGeometry(vectors, wavelength)


def test_nonfinite_and_bad_shape():
    for samples in [np.zeros((3, 2)), np.full((3, 3), np.nan), np.array([[1j, 2, 3]])]:
        with pytest.raises(ValueError):
            reciprocal_vectors(samples, panel())
    with pytest.raises(ValueError):
        lattice_fourier(np.empty((0, 3)), np.ones((2, 3)))


def test_fourier_fused_derivatives_and_reference():
    rng = np.random.default_rng(7531)
    q, v = rng.normal(size=(110, 3)), rng.normal(size=(8, 3))
    result = lattice_fourier(q, v)
    np.testing.assert_allclose(result, ref.lattice_fourier(q, v), atol=1e-13, rtol=1e-12)
    step = 2e-6
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = step
        plus, minus = lattice_fourier(q, v + delta), lattice_fourier(q, v - delta)
        np.testing.assert_allclose(
            (plus[:, 0] - minus[:, 0]) / (2 * step), result[:, 1 + j], atol=2e-9
        )
        np.testing.assert_allclose(
            (plus[:, 1:4] - minus[:, 1:4]) / (2 * step),
            result[:, 4:].reshape(-1, 3, 3)[:, :, j],
            atol=1e-8,
        )
    assert lattice_fourier(q, np.zeros((1, 3)))[0, 0] == pytest.approx(1, abs=2e-15)


def test_each_observation_corrected_before_combining():
    class Positions(SimpleNamespace):
        def __len__(self):
            return 2

    p = Positions(
        observations=np.array(
            [[0, 0, 1100, 1000, -15], [0, 1, 1101, 1001, -14], [1, 2, 600, 800, 30.0]]
        )
    )
    result = correct_positions(p, panel())
    np.testing.assert_allclose(
        result.points[0],
        ref.reciprocal_vectors(p.observations[:2, 2:], panel()).mean(axis=0),
        atol=1e-15,
    )
    assert result.rms[0] > 0 and result.rms[1] == 0
    assert not result.points.flags.writeable


def synthetic(basis, seed=33, outliers=100):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    basis = np.asarray(basis) @ rotation
    h = rng.integers(-6, 7, (700, 3))
    h = np.unique(h[np.any(h != 0, axis=1)], axis=0)
    q = h @ np.linalg.inv(basis).T + rng.normal(0, 0.001, h.shape)
    q = np.r_[q, rng.uniform(-1, 1, (outliers, 3))]
    return q, basis


@pytest.mark.parametrize(
    "basis",
    [
        np.diag([4.3, 5.1, 7.7]),
        np.array([[4.1, 0, 0], [1.2, 5.3, 0], [-0.8, 0.5, 6.7]]),
        np.array([[5.2, 0, 0], [-2.6, 4.5033321, 0], [0, 0, 8.4]]),
        np.array([[0, 3.2, 3.2], [3.2, 0, 3.2], [3.2, 3.2, 0]]),
    ],
)
def test_blind_cell_with_outliers(basis):
    q, truth = synthetic(basis)
    result = find_lattices(q, options=IndexingOptions(max_length=15, grid_size=128))
    assert result.candidates, result.message
    best = result.candidates[0]
    assert best.accepted
    assert best.validation_fraction > 0.7
    assert best.volume == pytest.approx(abs(np.linalg.det(truth)), rel=0.02)
    change = best.direct_basis @ np.linalg.inv(truth)
    np.testing.assert_allclose(change, np.rint(change), atol=0.025)


def test_random_cloud_does_not_pass():
    rng = np.random.default_rng(831)
    result = find_lattices(
        rng.uniform(-1, 1, (250, 3)), options=IndexingOptions(max_length=15, grid_size=64)
    )
    assert not any(c.accepted for c in result.candidates)


def test_held_out_positions_cannot_change_fitted_basis():
    q, _ = synthetic(np.diag([4.3, 5.1, 7.7]))
    training = np.arange(len(q)) % 5 != 0
    options = IndexingOptions(max_length=15, grid_size=128)
    first = find_lattices(q, training=training, options=options)
    q[~training] = np.random.default_rng(929).uniform(-1, 1, (np.count_nonzero(~training), 3))
    second = find_lattices(q, training=training, options=options)
    np.testing.assert_array_equal(
        first.candidates[0].direct_basis, second.candidates[0].direct_basis
    )
    assert not second.candidates[0].accepted


def test_reduction_preserves_lattice_volume_and_metric():
    a = np.array([[4, 0, 0], [8, 5, 0], [12, 10, 6.0]])
    reduced = reduce_basis(a)
    assert abs(np.linalg.det(reduced)) == pytest.approx(120)
    np.testing.assert_allclose(cell_parameters(reduced), [4, 5, 6, 90, 90, 90], atol=1e-12)


@pytest.mark.parametrize(
    "kw",
    [
        {"grid_size": 1024},
        {"min_length": 0},
        {"max_length": float("nan")},
        {"fractional_tolerance": 0.5},
    ],
)
def test_indexing_limits(kw):
    with pytest.raises(ValueError):
        IndexingOptions(**kw)
