"""Independent forward/inverse geometry, derivatives, refinement and scan support."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.prediction import (
    RefinementOptions,
    orientation_matrix,
    predict_scan,
    predict_spots,
    refine_crystal,
)
from phasesmith.experimental.prediction_reference import predict as reference
from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_predict_spots"), reason="experimental-spots feature not built"
)


def setup(seed=643):
    rng = np.random.default_rng(seed)
    g = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=8e-5,
        pixel2=8e-5,
        rot1=0.01,
        rot2=-0.03,
        rot3=0.05,
        wavelength=0.4,
        axis=[-0.97, 0.12, 0.2],
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    direct = np.array([[4.7, 0, 0], [0.9, 6.3, 0], [-0.4, 0.2, 8.1]]) @ rotation
    b = np.linalg.inv(direct).T
    h = rng.integers(-5, 6, (500, 3), dtype=np.int64)
    hint = rng.uniform(-200, -30, len(h))
    return rng, g, b, h, hint


@pytest.mark.parametrize("other", [False, True])
def test_randomized_reference_and_inverse(other):
    for seed in [643, 231, 112]:
        _, g, b, h, hint = setup(seed)
        p = predict_spots(h, b, g, hint, other=other)
        expected = reference(h, b, g, hint, other=other)
        good = (p.flags == 0) & np.isfinite(expected).all(axis=1)
        assert good.sum() > 80
        np.testing.assert_allclose(p.values[good], expected[good], atol=3e-10, rtol=3e-12)
        q = reciprocal_vectors(p.values[good], g)
        np.testing.assert_allclose(q, h[good] @ b, atol=3e-15, rtol=2e-13)
        assert not p.values.flags.writeable and not p.jacobian.flags.writeable


@pytest.mark.parametrize("other", [False, True])
def test_fused_analytical_jacobian(other):
    _, g, b, h, hint = setup()
    p = predict_spots(h, b, g, hint, other=other)
    good = (p.flags == 0) & (np.max(abs(p.jacobian), axis=(1, 2)) < 1e5)
    h, hint = h[good][:25], hint[good][:25]
    p = predict_spots(h, b, g, hint, other=other)
    step = 2e-8
    for i in range(9):
        delta = np.zeros((3, 3))
        delta.ravel()[i] = step
        plus = predict_spots(h, b + delta, g, hint, other=other)
        minus = predict_spots(h, b - delta, g, hint, other=other)
        np.testing.assert_allclose(
            (plus.values - minus.values) / (2 * step), p.jacobian[:, :, i], atol=3e-4, rtol=3e-7
        )


def test_flags_and_turn_unwrapping():
    _, g, b, h, hint = setup()
    zero = predict_spots(np.zeros((1, 3), np.int64), b, g, [0.0])
    assert zero.flags[0] == 1 and np.isnan(zero.values).all()
    p = predict_spots(h, b, g, hint)
    shifted = predict_spots(h, b, g, hint + 720)
    good = (p.flags == 0) & (shifted.flags == 0)
    np.testing.assert_allclose(shifted.values[good, 2] - p.values[good, 2], 720, atol=2e-13)
    np.testing.assert_allclose(shifted.values[good, :2], p.values[good, :2], atol=3e-10, rtol=2e-12)
    # Explicit tangent to an Ewald sphere for a simple vertical-axis plane.
    plane = PanelGeometry.from_poni(
        distance=1, poni1=0, poni2=0, pixel1=0.01, pixel2=0.01, wavelength=1, axis=[1, 0, 0]
    )
    tangent = predict_spots(np.array([[0, 0, 2]]), np.eye(3), plane, [0])
    assert tangent.flags[0] == 2


def test_synthetic_refinement_and_held_out_independence():
    rng, g, b, h, hint = setup()
    exact = predict_spots(h, b, g, hint)
    good = exact.flags == 0
    h, observed = h[good], exact.values[good].copy()
    observed += rng.normal(size=observed.shape) * [0.12, 0.12, 0.025]
    training = np.arange(len(h)) % 5 != 0
    start = b + rng.normal(0, 1e-4, b.shape)
    result = refine_crystal(observed, h, start, g, training=training)
    assert result.metrics["validation_improved"]
    assert result.metrics["validation"]["after"]["rms_xy"] < 0.3
    assert result.metrics["validation"]["after"]["rms_omega"] < 0.04
    np.testing.assert_allclose(result.reciprocal_basis, b, atol=1e-5)
    assert np.all(np.diff(result.history) < 0)
    shifted = observed.copy()
    shifted[~training] += [1, -1, 0.1]
    second = refine_crystal(shifted, h, start, g, training=training)
    np.testing.assert_array_equal(result.reciprocal_basis, second.reciprocal_basis)
    u = orientation_matrix(result.reciprocal_basis)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=2e-15)
    assert np.linalg.det(u) == pytest.approx(1, abs=2e-15)


def test_scan_prediction_two_roots_and_half_open_boundaries():
    _, g, b, _h, _hint = setup()
    edges = np.arange(-150, -69, 1.0)
    bounds = np.c_[edges[:-1], edges[1:]]
    result = predict_scan(b, g, bounds, (2048, 2048), qmax=1.2)
    assert len(result.values) > 100
    assert np.all(result.values[:, 2] >= bounds[result.frame_indices, 0])
    assert np.all(result.values[:, 2] < bounds[result.frame_indices, 1])
    q = reciprocal_vectors(result.values, g)
    np.testing.assert_allclose(q, result.hkl @ b, atol=3e-15, rtol=2e-13)
    # Closing boundary excludes; opening boundary includes the same crossing.
    theta = result.values[len(result.values) // 2, 2]
    before = predict_scan(b, g, np.array([[theta - 1, theta]]), (2048, 2048), qmax=1.2)
    after = predict_scan(b, g, np.array([[theta, theta + 1]]), (2048, 2048), qmax=1.2)
    assert not np.any(before.values[:, 2] == theta)
    assert np.any(after.values[:, 2] == theta)
    reverse = predict_scan(b, g, bounds[:, ::-1][::-1], (2048, 2048), qmax=1.2)
    assert len(reverse.values) == len(result.values)


def test_bad_inputs_and_budgets():
    _, g, b, h, _hint = setup()
    for bad in [h.astype(float), np.zeros((4, 2), int), np.full((2, 3), 1000001)]:
        with pytest.raises(ValueError):
            predict_spots(bad, b, g, 0)
    with pytest.raises(ValueError):
        predict_spots(h, np.zeros((3, 3)), g, 0)
    with pytest.raises(ValueError):
        predict_spots(h, b, g, np.nan)
    with pytest.raises(ValueError, match="budget"):
        predict_scan(b, g, np.array([[0, 10.0]]), (2048, 2048), qmax=2, max_hkl=10)
    with pytest.raises(ValueError, match="180"):
        predict_scan(b, g, np.array([[0, 200.0]]), (2048, 2048), qmax=1)
    with pytest.raises(ValueError):
        RefinementOptions(xy_scale=0)


def test_invalid_final_validation_predictions_are_not_dropped():
    # A side panel sees both forward and near-backscattered rays. Held-out
    # near-tangent observations cease to have Ewald roots after fitting the
    # training crystal. This must fail validation, not shrink its population.
    g = PanelGeometry(
        np.array([[0.1, 0, 0], [0, 0.001, 0], [0, 0, 0.001], [0, 0, 1], [1, 0, 0]]), 1.0
    )
    rng = np.random.default_rng(372)
    h = rng.integers(-3, 4, (100, 3), dtype=np.int64)
    h[:, 0] = rng.integers(1, 4, 100)
    truth, start = np.eye(3) * 0.101, np.eye(3) * 0.099
    p = predict_spots(h, truth, g, -40.0)
    h, observed = h[p.flags == 0], p.values[p.flags == 0]
    held_h = np.tile([1, 20, 0], (5, 1))
    held = predict_spots(held_h, start, g, 0.0)
    assert not held.flags.any()
    count = len(h)
    result = refine_crystal(
        np.r_[observed, held.values],
        np.r_[h, held_h],
        start,
        g,
        training=np.arange(count + 5) < count,
    )
    assert result.metrics["validation"]["count"] == 5
    assert np.isinf(result.metrics["validation"]["after"]["rms_xy"])
    assert not result.metrics["validation_improved"]


def test_behind_panel_and_rank_deficient_refinement():
    g = PanelGeometry(
        np.array([[0, 0, -1], [0, 0.001, 0], [0.001, 0, 0], [0, 0, 1], [1, 0, 0]]), 1.0
    )
    p = predict_spots(np.array([[0, 1, 0]]), np.eye(3) * 0.1, g, 0.0)
    assert p.flags[0] == 4
    _, g, b, h, hint = setup()
    h[:, 2] = 0
    p = predict_spots(h, b, g, hint)
    h, observed = h[p.flags == 0], p.values[p.flags == 0]
    with pytest.raises(ValueError, match="span"):
        refine_crystal(observed, h, b, g, training=np.arange(len(h)) % 5 != 0)
