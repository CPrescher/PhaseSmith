"""Metric settings, exact index transforms and constrained positional fitting."""

import numpy as np
import pytest
from phasesmith.experimental.orthorhombic import (
    find_orthorhombic_settings,
    orthorhombic_chain,
    orthorhombic_step,
    project_orthorhombic,
)
from phasesmith.experimental.prediction import predict_spots, refine_crystal
from phasesmith.experimental.prediction_reference import predict as reference
from test_experimental_prediction import setup


@pytest.mark.parametrize(
    "centring,primitive",
    [
        ("P", np.eye(3)),
        ("C", np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]])),
        ("I", np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])),
        ("F", np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])),
    ],
)
def test_settings_from_sheared_primitive_lattices(centring, primitive):
    rng = np.random.default_rng(431)
    for lengths in ([4.1, 6.7, 9.3], [5.2, 8.8, 12.4]):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        conventional = np.diag(lengths) @ rotation
        a = primitive @ conventional
        a = np.array([[1, 2, 0], [0, 1, -1], [0, 0, 1]]) @ a
        result = find_orthorhombic_settings(a)
        assert result
        candidate = result[0]
        if centring == "C":
            assert candidate.centring in ("A", "B", "C")
        else:
            assert candidate.centring == centring
        np.testing.assert_allclose(candidate.cell[:3], lengths, atol=2e-13)
        np.testing.assert_allclose(candidate.cell[3:], 90, atol=2e-13)
        np.testing.assert_allclose(candidate.direct_basis, candidate.transform @ a, atol=1e-13)
        h = rng.integers(-10, 11, (100, 3))
        transformed = h @ candidate.transform.T
        np.testing.assert_allclose(
            transformed @ np.linalg.inv(candidate.direct_basis).T,
            h @ np.linalg.inv(a).T,
            atol=2e-14,
        )
        assert not candidate.transform.flags.writeable


def test_metric_threshold_and_invalid_inputs():
    a = np.diag([4.0, 7.0, 11.0])
    a[1, 0] = 0.02
    assert find_orthorhombic_settings(a, max_angle=0.3)
    assert not find_orthorhombic_settings(a, max_angle=0.01)
    for kwargs in [dict(max_angle=0), dict(max_angle=np.nan), dict(coefficient_limit=5)]:
        with pytest.raises(ValueError):
            find_orthorhombic_settings(a, **kwargs)
    with pytest.raises(ValueError):
        find_orthorhombic_settings(-a)


def test_orthorhombic_basis_and_prediction_chains():
    _, geometry, b, h, hint = setup()
    b = project_orthorhombic(b)
    p = predict_spots(h, b, geometry, hint)
    good = (p.flags == 0) & (np.max(abs(p.jacobian), axis=(1, 2)) < 1e5)
    h, hint = h[good][:40], hint[good][:40]
    p = predict_spots(h, b, geometry, hint)
    chain = orthorhombic_chain(b)
    step_size = 1e-7
    for k in range(6):
        delta = np.eye(6)[k] * step_size
        plus, minus = orthorhombic_step(b, delta), orthorhombic_step(b, -delta)
        np.testing.assert_allclose(
            (plus - minus) / (2 * step_size), chain[:, k].reshape(3, 3), atol=3e-10
        )
        # Independent forward reference, not the Rust prediction implementation.
        finite_difference = (
            reference(h, plus, geometry, hint) - reference(h, minus, geometry, hint)
        ) / (2 * step_size)
        np.testing.assert_allclose(
            finite_difference, p.jacobian @ chain[:, k], rtol=2e-7, atol=3e-5
        )
    next_basis = orthorhombic_step(b, np.array([0.02, -0.01, 0.03, 0.05, 0.02, -0.03]))
    metric = next_basis @ next_basis.T
    np.testing.assert_allclose(metric, np.diag(np.diag(metric)), atol=3e-17)


def test_orthorhombic_refinement_is_six_parameter_training_only():
    rng, g, b, h, hint = setup()
    b = project_orthorhombic(b)
    exact = predict_spots(h, b, g, hint)
    good = exact.flags == 0
    h, observed = h[good], exact.values[good].copy()
    observed += rng.normal(size=observed.shape) * [0.12, 0.12, 0.02]
    training = np.arange(len(h)) % 5 != 0
    start = b + rng.normal(0, 1e-4, b.shape)
    result = refine_crystal(observed, h, start, g, training=training, symmetry="orthorhombic")
    np.testing.assert_allclose(result.cell[3:], 90, atol=3e-13)
    assert result.metrics["validation_improved"]
    assert result.metrics["validation"]["after"]["rms_xy"] < 0.3
    assert np.all(np.diff(result.history) < 0)
    shifted = observed.copy()
    shifted[~training] += [1, -1, 0.1]
    second = refine_crystal(shifted, h, start, g, training=training, symmetry="orthorhombic")
    np.testing.assert_array_equal(result.reciprocal_basis, second.reciprocal_basis)
    with pytest.raises(ValueError, match="symmetry"):
        refine_crystal(observed, h, start, g, training=training, symmetry="cubic")


def test_centred_setting_predicts_same_physical_reciprocal_lattice():
    from phasesmith.experimental.prediction import predict_scan

    _, g, _, _, _ = setup()
    a = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]]) @ np.diag([4.1, 6.7, 9.3])
    setting = find_orthorhombic_settings(a)[0]
    bounds = np.array([[-130.0, -60.0]])
    primitive = predict_scan(np.linalg.inv(a).T, g, bounds, (2048, 2048), qmax=1)
    conventional = predict_scan(
        np.linalg.inv(setting.direct_basis).T,
        g,
        bounds,
        (2048, 2048),
        qmax=1,
        lattice_transform=setting.transform,
    )
    assert len(primitive.values) == len(conventional.values) > 50
    # Compare by invariant original Miller index, rather than enumerated order.
    original_h = np.rint(conventional.hkl @ np.linalg.inv(setting.transform).T).astype(int)

    def key(h):
        return np.lexsort((h[:, 2], h[:, 1], h[:, 0]))

    np.testing.assert_array_equal(original_h[key(original_h)], primitive.hkl[key(primitive.hkl)])
    np.testing.assert_allclose(
        conventional.values[key(original_h)], primitive.values[key(primitive.hkl)], atol=3e-10
    )
    with pytest.raises(ValueError, match="lattice_transform"):
        predict_scan(
            np.linalg.inv(a).T,
            g,
            bounds,
            (2048, 2048),
            qmax=1,
            lattice_transform=np.zeros((3, 3), int),
        )
