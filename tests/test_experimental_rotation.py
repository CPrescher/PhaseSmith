"""Rotation peak hunting is position-first; no cell, HKL or integrated I input."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import rotation as r
from phasesmith.experimental import rotation_reference as reference
from phasesmith.experimental.spots import detect_spots

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_rotation_link"), reason="experimental rotation build needed"
)


def scene():
    images = np.full((7, 48, 60), 5.0)
    for f, excess in enumerate([0, 0, 50, 100, 50, 0, 0]):
        images[f, 20:23, 25:28] += excess
    bounds = np.column_stack([np.arange(7) * 0.5, (np.arange(7) + 1) * 0.5])
    return images, bounds


def test_position_hunt_reads_once_and_keeps_per_frame_observations():
    images, bounds = scene()
    calls = []

    def loader(i):
        calls.append(i)
        return images[i]

    actual = r.locate_series(loader, bounds)
    assert calls == list(range(7))
    assert len(actual) == 1
    assert "intensity" not in actual.columns
    np.testing.assert_array_equal(actual.observations[:, 1], [2, 3, 4])
    np.testing.assert_allclose(actual.values[0, :3], [26, 21, 1.75], atol=1e-12, rtol=0)
    assert actual.values[0, -1] == 3
    assert actual.flags[0] == 0
    assert not actual.observations.flags.writeable
    repeat = r.locate_series(images.__getitem__, bounds)
    np.testing.assert_array_equal(actual.values, repeat.values)
    np.testing.assert_array_equal(actual.observations, repeat.observations)


def test_link_reference_randomized_and_centroid_reference():
    rng = np.random.default_rng(42)
    for _ in range(6):
        images = rng.normal(5, 0.1, (5, 48, 60))
        for f in range(5):
            for x, y in [(14, 12), (32, 30)]:
                dx = rng.integers(-1, 2)
                images[f, y : y + 3, x + dx : x + dx + 3] += 100
        tables = [detect_spots(image) for image in images]
        frames = np.repeat(np.arange(5), [len(t) for t in tables])
        xy = np.concatenate([t.values[:, :2] for t in tables])
        bounds = np.column_stack([np.arange(5), np.arange(1, 6)])
        labels = r.link_spots(tables, bounds)
        np.testing.assert_array_equal(labels, reference.link_spots(frames, xy, np.zeros(5, int), 5))
        actual = r.locate_series(images.__getitem__, bounds)
        rows = np.concatenate([t.values for t in tables])
        for k in range(len(actual)):
            chosen = labels == k
            samples = np.column_stack(
                [rows[chosen, :2], bounds[frames[chosen]].mean(axis=1), rows[chosen, 2]]
            )
            expected = reference.summarize(samples)
            np.testing.assert_allclose(actual.values[k], expected, atol=2e-12, rtol=0)


def test_no_link_across_angular_gap_and_reverse_scan():
    images, bounds = scene()
    bounds[4:] += 5
    actual = r.locate_series(images.__getitem__, bounds)
    assert len(actual) == 2
    assert np.all(actual.flags & r.SCAN_BOUNDARY)
    _, original = scene()
    reverse = r.locate_series(images[::-1].__getitem__, original[::-1, ::-1])
    np.testing.assert_allclose(reverse.values[0, :3], [26, 21, 1.75], atol=1e-12, rtol=0)


def test_ambiguous_same_frame_detections_flagged():
    images = np.full((3, 48, 60), 5.0)
    images[0, 20:23, 20:23] += 100
    images[1, 20:23, 18:21] += 100
    images[1, 20:23, 25:28] += 100
    images[2, 20:23, 20:23] += 100
    bounds = np.column_stack([np.arange(3), np.arange(1, 4)])
    result = r.locate_series(
        images.__getitem__, bounds, rotation_options=r.RotationOptions(link_radius=7)
    )
    assert len(result) == 1
    assert result.flags[0] & r.AMBIGUOUS
    assert result.flags[0] & r.SCAN_BOUNDARY


def test_masks_saturation_and_empty_scan():
    images, bounds = scene()
    mask = np.zeros(images.shape[1:], bool)
    mask[21, 26] = True
    result = r.locate_series(images.__getitem__, bounds, mask_loader=lambda _: mask, saturation=90)
    assert result.flags[0] & 2
    assert result.flags[0] & 4
    result = r.locate_series(lambda i: np.full((48, 60), 5.0), bounds)
    assert len(result) == 0 and result.observations.shape == (0, 5)


@pytest.mark.parametrize(
    "bounds", [[[0, 0]], [[0, 1], [0, 1]], [[0, 1], [2, 1]], [[0, np.nan]], [[0, 1], [-2, -1]]]
)
def test_invalid_angles(bounds):
    with pytest.raises(ValueError):
        r.angle_segments(bounds)


def test_budget_shape_and_options_validation():
    images, bounds = scene()
    with pytest.raises(ValueError, match="budget"):
        r.locate_series(
            images.__getitem__, bounds, rotation_options=r.RotationOptions(max_observations=1)
        )
    with pytest.raises(ValueError, match="same detector shape"):
        r.locate_series(lambda i: images[i] if i != 2 else images[i, :5], bounds)
    with pytest.raises(ValueError):
        r.RotationOptions(link_radius=-1)
    with pytest.raises(ValueError):
        r.RotationOptions(max_candidates=0.5)


def test_native_link_exact_boundary_empty_frame_and_invalid_inputs():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    frames = np.array([0, 1, 3], np.uint32)
    seg = np.zeros(4, np.uint32)
    np.testing.assert_array_equal(
        _core.experimental_rotation_link(frames, xy, seg, 5.0, 100), [0, 0, 1]
    )
    np.testing.assert_array_equal(
        _core.experimental_rotation_link(frames, xy, seg, 4.999, 100), [0, 1, 2]
    )
    with pytest.raises(ValueError, match="budget"):
        _core.experimental_rotation_link(frames, xy, seg, 5.0, 0)
    with pytest.raises(ValueError):
        _core.experimental_rotation_link(frames[::-1].copy(), xy, seg, 5.0, 100)
    with pytest.raises(ValueError):
        _core.experimental_rotation_link(frames, xy[:, :1].copy(), seg, 5.0, 100)


def test_half_degree_step_and_distinct_one_degree_width():
    bounds, segments = r.angle_segments([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    np.testing.assert_array_equal(segments, [0, 0, 0])
    np.testing.assert_allclose(bounds.mean(axis=1), [0.5, 1.0, 1.5], atol=0, rtol=0)


def test_centroid_translation_and_fixed_weight_derivatives():
    samples = np.array([[2.0, 3.0, -128.0, 10.0], [4.0, 5.0, -127.0, 30.0]])
    result = _core.experimental_rotation_centroid(samples)
    np.testing.assert_allclose(result[:3], [3.5, 4.5, -127.25], atol=1e-12, rtol=0)
    for column in range(3):
        hi, lo = samples.copy(), samples.copy()
        hi[0, column] += 1e-4
        lo[0, column] -= 1e-4
        derivative = (
            _core.experimental_rotation_centroid(hi)[column]
            - _core.experimental_rotation_centroid(lo)[column]
        ) / 2e-4
        np.testing.assert_allclose(derivative, 0.25, atol=1e-10, rtol=0)
    translated = samples.copy()
    translated[:, :3] += [1000.0, -2000.0, 1e6]
    shifted = _core.experimental_rotation_centroid(translated)
    np.testing.assert_allclose(shifted[:3] - result[:3], [1000.0, -2000.0, 1e6], atol=1e-10, rtol=0)
    np.testing.assert_allclose(shifted[3], result[3], atol=1e-12, rtol=0)
