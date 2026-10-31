"""Physical recovery, derivative parity, heldouts, conventions and invalid inputs."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.geometry_refinement import (
    GeometryOptions,
    parameter_state,
    predict_with_detector,
    reference_basis,
    refine_geometry,
)
from phasesmith.experimental.geometry_refinement_reference import detector_partials
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.reciprocal import PanelGeometry
from test_experimental_prediction import setup

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_predict_spots_with_detector"), reason="geometry feature build"
)


def sample(seed=643):
    _, g, b, h, hint = setup(seed)
    p = predict_spots(h, b, g, hint)
    keep = (p.flags == 0) & (np.max(abs(p.values[:, :2]), axis=1) < 6000)
    return g, b, h[keep], hint[keep]


@pytest.mark.parametrize("seed", [643, 231, 112])
def test_native_detector_reference_and_finite_differences(seed):
    g, b, h, hint = sample(seed)
    h = h[:30]
    hint = hint[:30]
    v, j, d, f = predict_with_detector(h, b, g, hint)
    base = predict_spots(h, b, g, hint)
    ref, rd = detector_partials(h, b, g, hint)
    np.testing.assert_array_equal(v, base.values)
    np.testing.assert_array_equal(j, base.jacobian)
    np.testing.assert_array_equal(f, base.flags)
    np.testing.assert_allclose(v, ref, atol=2e-9, rtol=1e-12)
    np.testing.assert_allclose(d, rd, atol=2e-5, rtol=1e-11)
    for k in range(9):
        step = 1e-9 if k < 3 else 1e-11
        delta = np.zeros((5, 3))
        delta.ravel()[k] = step
        vp = predict_spots(h, b, PanelGeometry(g.vectors + delta, g.wavelength), hint).values
        vm = predict_spots(h, b, PanelGeometry(g.vectors - delta, g.wavelength), hint).values
        np.testing.assert_allclose((vp - vm) / (2 * step), d[:, :, k], atol=0.1, rtol=2e-6)


@pytest.mark.parametrize("vary", [False, True])
def test_physical_chain_and_known_pose_recovery(vary):
    g, b, h, hint = sample()
    pivot = g.vectors[0] + 1023.5 * (g.vectors[1] + g.vectors[2])
    n = 12 if vary else 9
    truth = np.array(
        [
            0.0002,
            -0.0001,
            0.0003,
            0.03,
            -0.04,
            0.02,
            0.0008,
            -0.0006,
            0.0004,
            0.0005,
            -0.0003,
            0.0002,
        ]
    )[:n]
    bt, gt, jb, jg = parameter_state(truth, b, g, pivot, vary)
    values, j, d, f = predict_with_detector(h, bt, gt, hint)
    keep = f == 0
    h = h[keep]
    hint = hint[keep]
    values = values[keep]
    j = j[keep]
    d = d[keep]
    chain = j @ jb + d @ jg
    for k in range(n):
        delta = np.eye(n)[k] * 1e-7
        bp, gp, _, _ = parameter_state(truth + delta, b, g, pivot, vary)
        bm, gm, _, _ = parameter_state(truth - delta, b, g, pivot, vary)
        numerical = (
            predict_spots(h, bp, gp, hint).values - predict_spots(h, bm, gm, hint).values
        ) / (2e-7)
        np.testing.assert_allclose(numerical, chain[:, :, k], atol=3e-5, rtol=2e-6)
    training = np.arange(len(h)) % 5 != 0
    fit = refine_geometry(values, h, b, g, (2048, 2048), training=training, refine_lengths=vary)
    assert fit.acceptable and fit.scaled_condition < 1e4
    np.testing.assert_allclose(fit.parameters, truth, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(fit.predictions.values, values, atol=2e-7)
    # Held-out observations can affect reported validation, never fitted parameters.
    perturbed = values.copy()
    perturbed[~training, :2] += 0.1
    again = refine_geometry(
        perturbed, h, b, g, (2048, 2048), training=training, refine_lengths=vary
    )
    np.testing.assert_array_equal(fit.parameters, again.parameters)


def test_reference_metric_and_invalid_boundaries():
    g, b, h, hint = sample()
    cell = np.array([5.0, 7.0, 9.0, 80.0, 95.0, 105.0])
    ref = reference_basis(b, cell)
    from phasesmith.experimental.indexing import cell_parameters

    np.testing.assert_allclose(cell_parameters(np.linalg.inv(ref).T), cell, atol=1e-12)
    for bad in ([0, 7, 9, 90, 90, 90], [5, 7, 9, 5, 5, 175], [5, 7, np.nan, 90, 90, 90]):
        with pytest.raises(ValueError):
            reference_basis(b, bad)
    for bad in [dict(xy_scale=0), dict(rotation_limit_deg=20), dict(max_evaluations=True)]:
        with pytest.raises(ValueError):
            GeometryOptions(**bad)
    p = predict_spots(h, b, g, hint)
    with pytest.raises(ValueError, match="withheld"):
        refine_geometry(p.values, h, b, g, (2048, 2048), training=np.ones(len(h), bool))
    with pytest.raises(ValueError):
        refine_geometry(p.values, h, b, g, (2048, 2048), training=np.ones(len(h), int))
    with pytest.raises(ValueError):
        parameter_state(np.zeros(8), b, g, g.vectors[0], False)


def test_nonconvergence_cannot_be_accepted():
    g, b, h, hint = sample()
    v = g.vectors.copy()
    v[0] += [0.00003, -0.00002, 0.00001]
    observed = predict_spots(h, b, PanelGeometry(v, g.wavelength), hint).values
    fit = refine_geometry(
        observed,
        h,
        b,
        g,
        (2048, 2048),
        training=np.arange(len(h)) % 5 != 0,
        options=GeometryOptions(max_evaluations=1),
    )
    assert not fit.converged and not fit.acceptable
