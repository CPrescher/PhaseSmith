"""Optional image/plane map: forward q, inverse predictions and analytical chains."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith.experimental.geometry_refinement import parameter_state, predict_with_detector
from phasesmith.experimental.geometry_refinement_reference import detector_partials
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.prediction_reference import predict as reference_prediction
from phasesmith.experimental.reciprocal import reciprocal_vectors
from phasesmith.experimental.reciprocal_reference import reciprocal_vectors as reference_q
from test_experimental_prediction import setup


class ScaledMap:
    def to_plane(self, xy):
        return xy * [1.1, 0.9] + [3.0, -2.0]

    def from_plane(self, xy):
        return (xy - [3.0, -2.0]) / [1.1, 0.9]

    def inverse_jacobian(self, xy):
        return np.broadcast_to(np.diag([1 / 1.1, 1 / 0.9]), (len(xy), 2, 2))


def test_forward_inverse_and_independent_references():
    _, g, b, h, hints = setup()
    mapped = replace(g, pixel_map=ScaledMap())
    p = predict_spots(h, b, mapped, hints)
    baseline = predict_spots(h, b, g, hints)
    valid = p.flags == 0
    np.testing.assert_array_equal(p.flags, baseline.flags)
    np.testing.assert_allclose(
        p.values[valid, :2], mapped.pixel_map.from_plane(baseline.values[valid, :2]), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        p.values[valid], reference_prediction(h, b, mapped, hints)[valid], atol=3e-10, rtol=3e-12
    )
    np.testing.assert_allclose(
        reciprocal_vectors(p.values[valid], mapped), h[valid] @ b, atol=3e-15, rtol=2e-13
    )
    np.testing.assert_allclose(
        reciprocal_vectors(p.values[valid], mapped),
        reference_q(p.values[valid], mapped),
        atol=3e-15,
        rtol=2e-13,
    )
    assert not p.values.flags.writeable


def test_prediction_and_detector_derivatives_with_nontrivial_chain():
    _, g, b, h, hints = setup()
    g = replace(g, pixel_map=ScaledMap())
    p = predict_spots(h, b, g, hints)
    good = (p.flags == 0) & (np.max(abs(p.jacobian), axis=(1, 2)) < 1e5)
    h, hints = h[good][:20], hints[good][:20]
    _, jac, det, _ = predict_with_detector(h, b, g, hints)
    _, reference_detector = detector_partials(h, b, g, hints)
    np.testing.assert_allclose(det, reference_detector, atol=2e-5, rtol=1e-11)
    for k in range(9):
        delta = np.zeros((3, 3))
        delta.ravel()[k] = 2e-8
        num = (
            predict_spots(h, b + delta, g, hints).values
            - predict_spots(h, b - delta, g, hints).values
        ) / (4e-8)
        np.testing.assert_allclose(num, jac[:, :, k], atol=3e-4, rtol=3e-7)
        dd = np.zeros((5, 3))
        step = 1e-9 if k < 3 else 1e-11
        dd.ravel()[k] = step
        gp = replace(g, vectors=g.vectors + dd)
        gm = replace(g, vectors=g.vectors - dd)
        num = (predict_spots(h, b, gp, hints).values - predict_spots(h, b, gm, hints).values) / (
            2 * step
        )
        np.testing.assert_allclose(num, det[:, :, k], atol=0.1, rtol=2e-6)
    _, changed, _, _ = parameter_state(np.zeros(9), b, g, g.vectors[0], False)
    assert changed.pixel_map is g.pixel_map


def test_invalid_map_and_malformed_results_fail_explicitly():
    _, g, b, h, hints = setup()
    with pytest.raises(TypeError, match="pixel_map"):
        replace(g, pixel_map=object())

    class Bad(ScaledMap):
        def to_plane(self, xy):
            return np.zeros((len(xy), 3))

    with pytest.raises(ValueError, match="finite real"):
        reciprocal_vectors(np.zeros((2, 3)), replace(g, pixel_map=Bad()))

    class BadJac(ScaledMap):
        def inverse_jacobian(self, xy):
            return np.full((len(xy), 2, 2), np.nan)

    with pytest.raises(ValueError, match="inverse Jacobian"):
        predict_spots(h, b, replace(g, pixel_map=BadJac()), hints)
