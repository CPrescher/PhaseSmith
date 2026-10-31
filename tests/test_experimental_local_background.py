"""Guarded supports, independent planar equations and known-count injections."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.integration import IntegrationOptions, integrate_series
from phasesmith.experimental.local_background_reference import measure
from phasesmith.experimental.prediction import ScanPredictions

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_local_background"), reason="experimental plane build"
)


def run(data, *, xy=(30.0, 30.0), radius=9, mask=None, variance=None, gap=3, margin=6):
    p = ScanPredictions(np.array([[1, 0, 0]]), np.array([[*xy, 0.5]]), np.array([0]), 1.0)
    return integrate_series(
        lambda _: data,
        [[0, 1]],
        p,
        mask_loader=None if mask is None else lambda _: mask,
        variance_loader=None if variance is None else lambda _: variance,
        options=IntegrationOptions(
            pixel_radius=radius,
            omega_half_width=0.4,
            background_model="plane",
            background_gap=gap,
            background_margin=margin,
        ),
    )


def test_plane_with_asymmetric_mask_and_background_contaminant():
    y, x = np.indices((61, 61))
    data = 40 + 0.17 * (x - 30) - 0.23 * (y - 30)
    data[28:33, 29:32] += 40
    data[16, 30] += 10000
    mask = (x < 20) & (y < 30)
    var = np.full(data.shape, 4.0)
    t = run(data, mask=mask, variance=var)
    ref, diag, _ = measure(data, t.boxes, mask=mask, variance=var)
    np.testing.assert_allclose(t.frame_values, ref, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(t.background_models, diag, rtol=1e-9, atol=2e-10)
    assert t.values[0, 0] == pytest.approx(600, abs=1e-9)
    assert t.background_models[0, 7] == 1
    assert t.flags[0] == 0


def test_randomized_equations_and_repeat_determinism():
    rng = np.random.default_rng(20261026)
    y, x = np.indices((61, 61))
    for _ in range(10):
        data = 25 + 0.1 * x + 0.08 * y + rng.normal(0, 2, x.shape)
        data[28:33, 28:33] += rng.uniform(0, 40)
        var = rng.uniform(3, 7, x.shape)
        mask = rng.random(x.shape) < 0.02
        t = run(data, mask=mask, variance=var)
        ref, diag, _ = measure(data, t.boxes, mask=mask, variance=var)
        np.testing.assert_allclose(t.frame_values, ref, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(t.background_models, diag, rtol=2e-8, atol=2e-9)
        other = run(data, mask=mask, variance=var)
        assert t.values.tobytes() == other.values.tobytes()
        assert t.background_models.tobytes() == other.background_models.tobytes()
        assert not t.background_models.flags.writeable


def test_conditional_pixel_derivatives_and_covariance():
    y, x = np.indices((61, 61))
    data = 35 + 0.2 * x - 0.1 * y
    data[30, 30] += 100
    var = np.full(data.shape, 3.0)
    t = run(data, variance=var)
    _, _, gradients = measure(data, t.boxes, variance=var)
    np.testing.assert_allclose(t.values[0, 1] ** 2, np.sum(gradients[0] ** 2 * var), rtol=1e-13)
    for yy, xx in [(30, 30), (16, 30), (30, 41), (0, 0)]:
        plus, minus = data.copy(), data.copy()
        plus[yy, xx] += 1e-3
        minus[yy, xx] -= 1e-3
        finite = (
            run(plus, variance=var).values[0, 0] - run(minus, variance=var).values[0, 0]
        ) / 2e-3
        assert finite == pytest.approx(gradients[0, yy, xx], abs=2e-8)
    np.testing.assert_allclose(t.background_models[0, 3:6], [-361, 0, 0], atol=1e-12)


def test_known_broad_peak_counts_and_boundary_warning():
    y, x = np.indices((81, 81))
    peak = 1400 * np.exp(-0.5 * (((x - 42) / 3.2) ** 2 + ((y - 39) / 2.7) ** 2))
    data = 50 + 0.25 * (x - 40) - 0.1 * (y - 40) + peak
    small = run(data, xy=(40, 40), radius=5)
    large = run(data, xy=(40, 40), radius=13)
    assert small.flags[0] & 2048
    assert not large.flags[0] & 2048
    assert small.values[0, 0] < 0.9 * peak.sum()
    assert large.values[0, 0] == pytest.approx(peak.sum(), rel=0.001)
    np.testing.assert_allclose(large.frame_values[0, :2], [42, 39], atol=0.01)
    np.testing.assert_allclose(large.frame_values[0, 7:9], [3.2, 2.7], atol=0.03)


def test_weak_signed_measurements_and_reported_errors():
    rng = np.random.default_rng(20261027)
    var = np.full((61, 61), 25.0)
    pulls = []
    for _ in range(120):
        data = 50 + rng.normal(0, 5, var.shape)
        t = run(data, radius=4, variance=var)
        pulls.append(t.values[0, 0] / t.values[0, 1])
    assert abs(np.mean(pulls)) < 0.25
    assert 0.8 < np.std(pulls, ddof=1) < 1.2
    assert np.count_nonzero(np.asarray(pulls) < 0) > 40


def test_bad_background_saturation_and_masks():
    data = np.full((61, 61), 20.0)
    t = run(data, mask=np.ones_like(data, bool))
    assert t.flags[0] & 16
    assert np.isnan(t.values[0, 0])
    # One-dimensional background support cannot constrain a plane.
    mask = np.ones_like(data, bool)
    mask[30, :] = False
    t = run(data, mask=mask)
    assert t.flags[0] & 16
    data[30, 30] = -5
    t = run(data, radius=0)
    assert t.values[0, 0] == pytest.approx(-25, abs=1e-10)
    assert t.flags[0] == 0


def test_guard_boundaries_competing_predictions_and_saturation():
    data = np.full((71, 71), 20.0)
    data[35, 25] += 17
    data[35, 39] += 31
    # For the first box [20,31), a three-pixel guard ends at x=34, exclusive.
    data[35, 33] += 1000
    p = ScanPredictions(
        np.array([[1, 0, 0], [0, 1, 0]]),
        np.array([[25.0, 35.0, 0.5], [39.0, 35.0, 0.5]]),
        np.array([0, 0]),
        1.0,
    )
    options = IntegrationOptions(
        pixel_radius=5,
        omega_half_width=0.4,
        background_model="plane",
        background_gap=3,
        background_margin=6,
    )
    t = integrate_series(lambda _: data, [[0, 1]], p, options=options, saturation=900.0)
    np.testing.assert_allclose(t.values[:, 0], [17, 31], atol=1e-9)
    assert not np.any(t.flags & (4 | 8 | 16))
    data[35, 25] = 1000
    t = integrate_series(lambda _: data, [[0, 1]], p, options=options, saturation=900.0)
    assert t.flags[0] & 4
    p = ScanPredictions(
        np.array([[1, 0, 0], [0, 1, 0]]),
        np.array([[25.0, 35.0, 0.5], [30.0, 35.0, 0.5]]),
        np.array([0, 0]),
        1.0,
    )
    t = integrate_series(lambda _: data, [[0, 1]], p, options=options)
    assert np.all(t.flags & 8)
    edge = run(np.full((61, 61), 20.0), xy=(0.0, 0.0), radius=5)
    assert edge.flags[0] & 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(background_model="typo"),
        dict(background_gap=1),
        dict(background_model="plane", min_background=2),
        dict(background_clip=np.nan),
        dict(background_clip=True),
        dict(background_gap=-1),
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        IntegrationOptions(**kwargs)
