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


@pytest.mark.parametrize("workers", [1, 3])
def test_supplied_variance_survives_background_subtraction_and_buffer_reuse(workers):
    images, bounds = scene()
    residuals = images - 5
    variances = np.maximum(images, 1)
    image_buffer = np.empty_like(images[0])
    variance_buffer = np.empty_like(variances[0])
    loaded = []

    def image_loader(i):
        image_buffer[:] = residuals[i]
        return image_buffer

    def variance_loader(i):
        loaded.append(i)
        variance_buffer[:] = variances[i]
        return variance_buffer

    actual = r.locate_series(image_loader, bounds, variance_loader=variance_loader, workers=workers)
    expected = r.locate_series(residuals.__getitem__, bounds, variance_loader=variances.__getitem__)
    assert loaded == list(range(len(images)))
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.observations, expected.observations)
    assert actual.frame_hashes == expected.frame_hashes
    tables = [detect_spots(a, variance=v) for a, v in zip(residuals, variances, strict=True)]
    np.testing.assert_array_equal(
        actual.observations[:, 2:4], np.concatenate([t.values[:, :2] for t in tables])
    )
    # Hashes include uncertainty inputs, even if centroids/detections are unchanged.
    changed = r.locate_series(
        residuals.__getitem__, bounds, variance_loader=lambda i: variances[i] * 1.01
    )
    assert actual.frame_hashes != changed.frame_hashes


@pytest.mark.parametrize(
    "variance", [np.ones((2, 3)), np.zeros((48, 60)), np.full((48, 60), np.nan)]
)
def test_invalid_supplied_frame_variances_fail(variance):
    images, bounds = scene()
    with pytest.raises(ValueError, match="variance"):
        r.locate_series(images.__getitem__, bounds, variance_loader=lambda _: variance)


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


def test_finished_frame_observer_is_ordered_and_does_not_change_results():
    images, bounds = scene()
    events, snapshots = [], []

    def finished(index, table):
        events.append(("frame", index))
        snapshots.append(table)
        assert not table.values.flags.writeable
        assert not table.flags.flags.writeable

    observed = r.locate_series(
        images.__getitem__,
        bounds,
        frame_finished=finished,
        progress=lambda done, total: events.append(("progress", done)),
    )
    original = r.locate_series(images.__getitem__, bounds)
    assert events == [
        event for i in range(len(images)) for event in (("frame", i), ("progress", i + 1))
    ]
    assert [len(t) for t in snapshots] == [0, 0, 1, 1, 1, 0, 0]
    for name in ("values", "flags", "observations", "boxes", "frame_ranges"):
        np.testing.assert_array_equal(getattr(observed, name), getattr(original, name))
    assert observed.frame_hashes == original.frame_hashes
    calls = []

    def cancel(index, table):
        calls.append(index)
        raise RuntimeError("cancel observer")

    with pytest.raises(RuntimeError, match="cancel observer"):
        r.locate_series(images.__getitem__, bounds, frame_finished=cancel)
    assert calls == [0]


@pytest.mark.parametrize("workers", [1, 2, 4, 6, 8, 10, 32])
def test_parallel_complete_records_and_loader_reuse_are_byte_identical(workers):
    from threading import get_ident

    images, bounds = scene()
    rng = np.random.default_rng(9821)
    images += rng.normal(0, 0.2, images.shape)
    masks = rng.random(images.shape) < 0.01
    images[masks] = np.nan
    expected = r.locate_series(images.__getitem__, bounds, mask_loader=masks.__getitem__)
    buffer, mask_buffer = np.empty_like(images[0]), np.empty_like(masks[0])
    calls, events, snapshots = [], [], []
    caller = get_ident()

    def load(i):
        assert get_ident() == caller
        calls.append(i)
        buffer[:] = images[i]
        return buffer

    def mask(i):
        assert get_ident() == caller
        mask_buffer[:] = masks[i]
        return mask_buffer

    def finished(i, table):
        assert get_ident() == caller
        events.append(("frame", i))
        snapshots.append(table)

    result = r.locate_series(
        load,
        bounds,
        mask_loader=mask,
        workers=workers,
        frame_finished=finished,
        progress=lambda done, total: events.append(("progress", done)),
    )
    assert calls == list(range(len(images)))
    assert events == [event for i in calls for event in (("frame", i), ("progress", i + 1))]
    for name in ("values", "flags", "observations", "boxes", "frame_ranges", "omega_bounds"):
        assert getattr(result, name).tobytes() == getattr(expected, name).tobytes(), name
    assert result.frame_hashes == expected.frame_hashes
    for i, table in enumerate(snapshots):
        original = detect_spots(images[i], mask=masks[i])
        for name in ("values", "boxes", "flags"):
            assert getattr(table, name).tobytes() == getattr(original, name).tobytes()


def test_parallel_detectors_overlap_but_callbacks_remain_ordered(monkeypatch):
    from threading import Barrier, get_ident

    images, bounds = scene()
    barrier = Barrier(2, timeout=10)
    original = r._measure_frame
    threads = set()
    caller = get_ident()

    def measure(*args):
        threads.add(get_ident())
        barrier.wait()
        return original(*args)

    monkeypatch.setattr(r, "_measure_frame", measure)
    completed = []
    # Four frames form two batches; a serial implementation would time out.
    r.locate_series(
        images.__getitem__,
        bounds[:4],
        workers=2,
        frame_finished=lambda i, table: completed.append(i),
    )
    assert len(threads) == 2 and caller not in threads
    assert completed == list(range(4))


@pytest.mark.parametrize("stop", ["observer", "progress", "budget", "loader", "detector"])
def test_parallel_failure_is_bounded_ordered_and_joins_workers(monkeypatch, stop):
    from threading import enumerate as thread_list

    images, bounds = scene()
    loaded, completed = [], []
    original = r._measure_frame

    def load(i):
        loaded.append(i)
        if stop == "loader" and i == 1:
            raise RuntimeError("loader stop")
        return images[2] if stop == "budget" else images[i]

    def finished(i, table):
        completed.append(i)
        if stop == "observer":
            raise RuntimeError("observer stop")

    def progress(done, total):
        if stop == "progress":
            raise RuntimeError("progress stop")

    def measure(*args):
        if stop == "detector":
            raise RuntimeError("detector stop")
        return original(*args)

    monkeypatch.setattr(r, "_measure_frame", measure)
    with pytest.raises((RuntimeError, ValueError), match=r"stop|budget"):
        r.locate_series(
            load,
            bounds,
            workers=2,
            frame_finished=finished,
            progress=progress,
            rotation_options=r.RotationOptions(max_observations=1 if stop == "budget" else 100),
        )
    assert loaded == ([0, 1, 2] if stop == "budget" else [0, 1])
    assert completed == ([] if stop == "detector" else [0])
    assert not any(t.name.startswith("phasesmith-spots") for t in thread_list())


@pytest.mark.parametrize("workers", [True, np.bool_(False), 1.5, "2", 0, -1, 33])
def test_invalid_worker_count_does_not_read_frames(workers):
    _, bounds = scene()
    with pytest.raises((ValueError, TypeError), match="workers"):
        r.locate_series(
            lambda _: pytest.fail("must validate before loading"), bounds, workers=workers
        )
