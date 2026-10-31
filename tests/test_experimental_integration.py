"""Fixed-support counts, independent equations and rotation boundary contracts."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.integration import (
    IntegrationFlag,
    IntegrationOptions,
    integrate_series,
)
from phasesmith.single_crystal.prediction import ScanPredictions
from phasesmith.single_crystal.reference.integration_reference import measure_frame

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_integrate_reduce"), reason="experimental build"
)


def predictions(xyz):
    xyz = np.array(xyz, float).reshape(-1, 3)
    return ScanPredictions(
        np.column_stack([np.arange(1, len(xyz) + 1), np.zeros((len(xyz), 2), int)]).astype(
            np.int64
        ),
        xyz,
        np.zeros(len(xyz), np.int64),
        1.0,
    )


def test_injected_rocking_counts_variance_and_moments():
    images = np.full((7, 25, 25), 10.0)
    for i, signal in enumerate([0, 0, 3, 10, 5, 0, 0]):
        images[i, 11:14, 11:14] += signal
    bounds = np.column_stack([np.arange(7) - 3.5, np.arange(7) - 2.5])
    table = integrate_series(
        lambda i: images[i],
        bounds,
        predictions([[12, 12, 0]]),
        options=IntegrationOptions(pixel_radius=1, omega_half_width=1.5),
        variance_loader=lambda _: np.full((25, 25), 4.0),
    )
    np.testing.assert_allclose(table.column("intensity"), [9 * 18], atol=1e-12)
    assert table.flags[0] == 0
    curve = table.rocking_curve(0)
    np.testing.assert_array_equal(curve["frame_indices"], [2, 3, 4])
    np.testing.assert_allclose(curve["intensity"], [27, 90, 45], atol=1e-12)
    np.testing.assert_allclose(np.sum(curve["omega"] * curve["intensity"]) / 162, 1 / 9, atol=1e-15)
    np.testing.assert_allclose(
        table.column("sigma") ** 2, [np.sum(curve["sigma"] ** 2)], rtol=1e-15
    )
    for a in (table.values, table.boxes, curve["intensity"]):
        assert not a.flags.writeable
    again = integrate_series(
        lambda i: images[i],
        bounds,
        predictions([[12, 12, 0]]),
        options=table.options,
        variance_loader=lambda _: np.full((25, 25), 4.0),
    )
    assert table.values.tobytes() == again.values.tobytes()


def test_randomized_independent_pixel_equations():
    rng = np.random.default_rng(20261025)
    images = rng.normal(8, 4, (4, 35, 40))
    masks = rng.random(images.shape) < 0.02
    variance = rng.uniform(1, 8, images.shape)
    bounds = np.column_stack([np.arange(4), np.arange(1, 5)])
    p = predictions([[10.2, 12.4, 1.4], [25, 23, 2.6], [12, 12, 1.5]])
    t = integrate_series(
        lambda i: images[i],
        bounds,
        p,
        mask_loader=lambda i: masks[i],
        variance_loader=lambda i: variance[i],
        options=IntegrationOptions(pixel_radius=2),
    )
    for frame in range(4):
        idx = t.frame_indices == frame
        ref = measure_frame(
            images[frame], t.boxes[idx], mask=masks[frame], variance=variance[frame]
        )
        np.testing.assert_allclose(
            t.frame_values[idx][:, [2, 3, 4, 9, 10]], ref, rtol=3e-14, atol=1e-12
        )
    for i in range(len(t)):
        v = t.frame_values[t.observation_ids == i]
        np.testing.assert_allclose(
            t.values[i, :2], [v[:, 2].sum(), np.sqrt((v[:, 3] ** 2).sum())], rtol=1e-15, atol=1e-14
        )
    assert np.all(t.flags[[0, 2]] & int(IntegrationFlag.OVERLAP))


def test_linear_derivatives_and_signed_weak_measurement():
    image = np.full((21, 21), 10.0)
    image[10, 10] = 7.0
    p = predictions([[10, 10, 0.5]])
    kw = dict(
        options=IntegrationOptions(pixel_radius=0, omega_half_width=0.4),
        variance_loader=lambda _: np.ones_like(image),
    )

    def evaluate(a):
        return integrate_series(lambda _: a, [[0, 1]], p, **kw)

    t = evaluate(image)
    assert t.values[0, 0] == -3
    assert t.flags[0] == 0  # no threshold/centroid flag for negative predictions
    for y, x, expected in [(10, 10, 1), (10, 11, -1 / t.frame_values[0, 10]), (0, 0, 0)]:
        plus, minus = image.copy(), image.copy()
        plus[y, x] += 1e-4
        minus[y, x] -= 1e-4
        actual = (evaluate(plus).values[0, 0] - evaluate(minus).values[0, 0]) / 2e-4
        assert actual == pytest.approx(expected, abs=2e-10)


def test_boundaries_gaps_reverse_no_data():
    data = np.full((20, 20), 10.0)
    bounds = np.array([[0, 1], [1, 2], [3, 4]], float)
    p = predictions([[10, 10, 1], [10, 10, 2.5], [10, 10, 8]])
    kw = dict(options=IntegrationOptions(pixel_radius=0, omega_half_width=0.5))
    t = integrate_series(lambda _: data, bounds, p, **kw)
    assert t.flags[0] == 0
    assert t.flags[1] & int(IntegrationFlag.ANGULAR | IntegrationFlag.NO_DATA)
    assert t.flags[2] & int(IntegrationFlag.NO_DATA)
    assert np.isnan(t.values[2, 0])
    reverse = integrate_series(lambda _: data, bounds[::-1, ::-1], p, **kw)
    np.testing.assert_allclose(t.values, reverse.values, equal_nan=True)
    np.testing.assert_array_equal(t.flags, reverse.flags)
    # Exact support boundary at 2 does not include the [1,2) exposure.
    edge = integrate_series(
        lambda _: data, [[0, 1], [1, 2], [2, 3]], predictions([[10, 10, 2.5]]), **kw
    )
    np.testing.assert_array_equal(edge.frame_indices, [2])


def test_mask_saturation_edge_background_and_competing_grains():
    data = np.full((20, 20), 10.0)
    data[0, 0] = 100
    mask = np.zeros_like(data, bool)
    mask[1, 1] = True
    t = integrate_series(
        lambda _: data,
        [[0, 1]],
        predictions([[0, 0, 0.5], [1, 1, 0.5]]),
        mask_loader=lambda _: mask,
        saturation=50,
        grain_ids=np.array([4, 8]),
        options=IntegrationOptions(pixel_radius=1, omega_half_width=0.4),
    )
    assert t.flags[0] & 1 and t.flags[0] & 2 and t.flags[0] & 4 and t.flags[0] & 8
    np.testing.assert_array_equal(t.grain_ids, [4, 8])
    bad = integrate_series(
        lambda _: data,
        [[0, 1]],
        predictions([[0, 0, 0.5]]),
        mask_loader=lambda _: np.ones_like(mask),
        options=IntegrationOptions(omega_half_width=0.4),
    )
    assert bad.flags[0] & 16 and np.isnan(bad.values[0, 0])


@pytest.mark.parametrize(
    "kw",
    [
        dict(pixel_radius=-1),
        dict(pixel_radius=True),
        dict(omega_half_width=0),
        dict(omega_half_width=np.nan),
        dict(background_margin=0),
        dict(max_observations=0),
    ],
)
def test_invalid_options(kw):
    with pytest.raises(ValueError):
        IntegrationOptions(**kw)


def test_invalid_inputs_and_cancellation():
    p = predictions([[10, 10, 0.5]])
    image = np.ones((20, 20))
    for kw in [
        dict(mask_loader=lambda _: np.ones((2, 2))),
        dict(variance_loader=lambda _: np.zeros_like(image)),
        dict(grain_ids=np.array([-1])),
    ]:
        with pytest.raises(ValueError):
            integrate_series(lambda _: image, [[0, 1]], p, **kw)
    with pytest.raises(ValueError, match="overlapping"):
        integrate_series(lambda _: image, [[0, 1], [0.5, 1.5]], p)
    with pytest.raises(ValueError, match="budget"):
        integrate_series(
            lambda _: image, [[0, 1], [1, 2]], p, options=IntegrationOptions(max_observations=1)
        )
    with pytest.raises(ValueError, match="footprint"):
        integrate_series(lambda _: image, [[0, 1]], predictions([[30, 10, 0.5]]))

    def cancel(*_):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        integrate_series(lambda _: image, [[0, 1]], p, progress=cancel)


def test_empty_predictions_and_owned_reused_loader_buffers():
    images = np.ones((3, 20, 20))
    p = predictions([])
    t = integrate_series(lambda i: images[i], [[0, 1], [1, 2], [2, 3]], p)
    assert t.values.shape == (0, 6) and len(t.frame_hashes) == 3
    buffer = np.ones((20, 20))

    def data(_):
        buffer.fill(5)
        buffer[10, 10] = 9
        return buffer

    def variance(_):
        buffer.fill(2)
        return buffer

    t = integrate_series(
        data,
        [[0, 1]],
        predictions([[10, 10, 0.5]]),
        variance_loader=variance,
        options=IntegrationOptions(pixel_radius=0, omega_half_width=0.4),
    )
    assert t.values[0, 0] == 4
