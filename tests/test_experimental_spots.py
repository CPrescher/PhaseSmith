"""Detector tests use pixels and explicit noise only, never a crystal/source model."""

from __future__ import annotations

import numpy as np
import pytest
from phasesmith.experimental import spots, spots_reference
from phasesmith.experimental.spots import SpotFlag, SpotOptions, detect_spots, extract_spots

native = pytest.mark.skipif(not spots.available(), reason="experimental-spots feature not built")


def scene():
    image = np.full((64, 80), 5.0)
    image[12:15, 17:20] += 100
    image[40:43, 53:56] += 50
    return image


def test_feature_is_explicit_and_not_a_root_export():
    import phasesmith

    assert not hasattr(phasesmith, "detect_spots")
    assert isinstance(spots.available(), bool)


def test_unavailable_feature_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(spots, "available", lambda: False)
    with pytest.raises(RuntimeError, match="experimental-spots"):
        detect_spots(scene())


@native
def test_square_spot_areas_centroids_and_uncertainties():
    result = detect_spots(scene())
    assert len(result) == 2
    np.testing.assert_allclose(result.column("intensity"), [900, 450], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.column("x"), [18, 54], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.column("y"), [13, 41], atol=1e-12, rtol=0)
    assert result.unflagged.all()
    m, n = result.column("n_pixels"), result.column("n_background")
    expected_var = [900, 450] + 5 * m + m * m * 5 / n
    np.testing.assert_allclose(result.column("sigma") ** 2, expected_var, atol=1e-10, rtol=0)
    assert result.rejected_small_components == 0
    assert not result.values.flags.writeable


@native
@pytest.mark.parametrize("supplied_variance", [False, True])
def test_full_native_reference_records_randomized(supplied_variance):
    rng = np.random.default_rng(20261024)
    for _ in range(4):
        image = scene() + rng.normal(0, 0.3, (64, 80))
        mask = rng.random(image.shape) < 0.015
        image[mask] = np.nan
        variance = rng.uniform(1, 5, image.shape) if supplied_variance else None
        options = SpotOptions(threshold_sigma=4.5)
        kw = dict(mask=mask, variance=variance, options=options, gain=1.7, read_noise=0.5)
        actual, expected = detect_spots(image, **kw), spots_reference.detect_spots(image, **kw)
        np.testing.assert_array_equal(actual.boxes, expected.boxes)
        np.testing.assert_array_equal(actual.flags, expected.flags)
        # Direct local sums versus large integral-table differences, small count arrays.
        np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=2e-10)
        assert actual.rejected_small_components == expected.rejected_small_components


@native
def test_complete_records_repeat_exactly_and_inputs_are_preserved():
    image, mask = scene(), np.zeros((64, 80), dtype=bool)
    before = image.copy()
    a, b = detect_spots(image, mask=mask), detect_spots(image, mask=mask)
    np.testing.assert_array_equal(a.boxes, b.boxes)
    np.testing.assert_array_equal(a.flags, b.flags)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(image, before)
    assert not mask.any()


@native
def test_signed_extraction_and_fixed_support_analytical_derivatives():
    image = np.full((25, 25), 10.0)
    image[12, 12] = 7
    variance = np.full(image.shape, 4.0)
    boxes = np.array([[12, 12, 13, 13]])
    result = extract_spots(image, boxes, variance=variance)
    assert result.column("intensity")[0] == -3
    assert result.flags[0] == SpotFlag.NO_CENTROID
    np.testing.assert_allclose(result.column("sigma") ** 2, [4 + 4 / 80], rtol=0, atol=1e-13)
    for index, expected in [((12, 12), 1), ((8, 8), -1 / 80), ((0, 0), 0)]:
        hi, lo = image.copy(), image.copy()
        hi[index] += 1e-4
        lo[index] -= 1e-4
        derivative = (
            extract_spots(hi, boxes, variance=variance).column("intensity")
            - extract_spots(lo, boxes, variance=variance).column("intensity")
        ) / 2e-4
        np.testing.assert_allclose(derivative, expected, rtol=0, atol=3e-10)


@native
def test_conditional_uncertainty_matches_independent_noise_ensemble():
    rng = np.random.default_rng(83912)
    image = np.full((24, 24), 10.0)
    image[10:13, 10:13] += 3
    boxes = np.array([[10, 10, 13, 13]])
    variance = np.full(image.shape, 4.0)
    z = []
    for _ in range(400):
        result = extract_spots(image + rng.normal(0, 2, image.shape), boxes, variance=variance)
        z.append((result.column("intensity")[0] - 27) / result.column("sigma")[0])
    assert abs(np.mean(z)) < 0.15
    assert 0.85 < np.std(z, ddof=1) < 1.15


@native
def test_gaussian_finite_aperture_area_and_moments():
    y, x = np.mgrid[:61, :61]
    signal = 1000 * np.exp(-((x - 30) ** 2 + (y - 29) ** 2) / 2) / (2 * np.pi)
    result = extract_spots(signal + 2, np.array([[24, 23, 37, 36]]))
    np.testing.assert_allclose(result.column("intensity"), 1000, rtol=0, atol=0.001)
    np.testing.assert_allclose(result.column("x"), 30, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.column("y"), 29, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.column("width_x"), 1, rtol=0, atol=2e-6)


@native
def test_quality_flags_are_retained():
    image = scene()
    mask = np.zeros(image.shape, dtype=bool)
    mask[11, 16] = True  # Inside the padded box, outside the detected component.
    result = detect_spots(image, mask=mask, saturation=90)
    assert len(result) == 2
    assert result.flags[0] & SpotFlag.MASKED
    assert result.flags[0] & SpotFlag.SATURATED
    large = detect_spots(image, options=SpotOptions(max_pixels=5))
    assert np.all(large.flags & SpotFlag.LARGE)
    edge = extract_spots(image, np.array([[0, 0, 3, 3]]))
    assert edge.flags[0] & SpotFlag.EDGE
    missing = extract_spots(image, np.array([[0, 0, 80, 64]]))
    assert missing.flags[0] & SpotFlag.BACKGROUND
    assert np.isnan(missing.column("intensity")[0])
    assert np.isnan(missing.column("sigma")[0])


@native
def test_overlap_excludes_neighbor_boxes_from_background():
    boxes = np.array([[15, 10, 23, 17], [20, 10, 28, 17]])
    actual = extract_spots(scene(), boxes)
    expected = spots_reference.extract_spots(scene(), boxes)
    assert np.all(actual.flags & SpotFlag.OVERLAP)
    np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=1e-12)


@native
def test_hot_pixels_minimum_size_and_budget_failure():
    image = np.full((64, 64), 4.0)
    image[10, 10] = image[30, 30] = 100
    result = detect_spots(image)
    assert len(result) == 0
    assert result.rejected_small_components == 2
    with pytest.raises(ValueError, match="max_spots"):
        detect_spots(image, options=SpotOptions(min_pixels=1, max_spots=1))


@native
def test_absolute_threshold_is_strict_and_diagonal_connectivity_is_eight():
    image = np.zeros((32, 32))
    image[12, 12] = image[13, 13] = image[14, 14] = 10
    variance = np.ones(image.shape)
    assert (
        len(detect_spots(image, variance=variance, options=SpotOptions(threshold_absolute=10))) == 0
    )
    result = detect_spots(image, variance=variance, options=SpotOptions(threshold_absolute=9))
    assert len(result) == 1


@native
def test_empty_masked_and_strided_images():
    assert len(detect_spots(np.ones((3, 4)))) == 0
    image = scene()
    assert len(detect_spots(image, mask=np.ones(image.shape, dtype=bool))) == 0
    assert len(extract_spots(image, np.empty((0, 4), dtype=int))) == 0
    masked = np.zeros(image.shape, dtype=bool)
    masked[10:13, 10:13] = True
    measurement = extract_spots(image, np.array([[10, 10, 13, 13]]), mask=masked)
    assert np.isnan(measurement.column("intensity")[0])
    assert np.isnan(measurement.column("sigma")[0])
    expected = spots_reference.detect_spots(image[:, ::-1])
    actual = detect_spots(image[:, ::-1])
    np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=1e-10)


@native
def test_finite_inputs_that_overflow_sums_fail_explicitly():
    with pytest.raises(ValueError, match="overflow"):
        detect_spots(np.full((32, 32), 1e308))


@pytest.mark.parametrize(
    "kw",
    [
        dict(threshold_sigma=0),
        dict(threshold_sigma=np.nan),
        dict(outer_radius=2),
        dict(inner_radius=-1),
        dict(padding=129),
        dict(min_pixels=True),
        dict(min_pixels=1.5),
        dict(max_pixels=2),
        dict(background_margin=0),
        dict(max_spots=0),
    ],
)
def test_invalid_options(kw):
    with pytest.raises((TypeError, ValueError)):
        SpotOptions(**kw)


@native
@pytest.mark.parametrize(
    "image,kw",
    [
        (np.zeros(3), {}),
        (np.zeros((0, 3)), {}),
        (np.ones((5, 5), dtype=complex), {}),
        (np.full((5, 5), np.nan), {}),
        (np.ones((5, 5)), dict(mask=np.zeros((5, 5)))),
        (np.ones((5, 5)), dict(mask=np.zeros((4, 5), dtype=bool))),
        (np.ones((5, 5)), dict(variance=np.zeros((5, 5)))),
        (np.ones((5, 5)), dict(gain=0)),
        (np.ones((5, 5)), dict(read_noise=-1)),
        (np.ones((5, 5)), dict(saturation=np.inf)),
    ],
)
def test_invalid_images_and_noise(image, kw):
    with pytest.raises((TypeError, ValueError)):
        detect_spots(image, **kw)


@native
@pytest.mark.parametrize(
    "boxes",
    [
        [[1.0, 2.0, 3.0, 4.0]],
        [[-1, 2, 3, 4]],
        [[2, 2, 1, 3]],
        [[1, 2, 1000, 4]],
        [[1, 2, 3]],
        [[0, 0, 2**64 - 1, 2]],
    ],
)
def test_invalid_boxes(boxes):
    with pytest.raises((TypeError, ValueError)):
        extract_spots(scene(), boxes)
