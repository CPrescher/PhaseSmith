import threading

import numpy as np
import pytest
from phasesmith.single_crystal.integration import (
    FrameExecution,
    IntegrationOptions,
    integrate_series,
)
from phasesmith.single_crystal.profiles import prepare_profiles
from test_experimental_integration import predictions


@pytest.mark.parametrize("plane", [False, True])
@pytest.mark.parametrize("variance", [False, True])
def test_ordered_frame_workers_preserve_arrays_snapshots_and_budget(plane, variance):
    rng = np.random.default_rng(683102)
    images = rng.normal(20, 3, (7, 45, 45))
    masks = rng.random(images.shape) < 0.01
    bounds = np.column_stack([np.arange(7), np.arange(1, 8)])[::-1, ::-1]
    p = predictions([[12.2, 14.4, 2.7], [28, 27, 3.1]])
    owner = threading.get_ident()

    def run(workers, budget=1024**2):
        buffer = images[0].copy()
        loaded, events = [], []

        def load(i):
            assert threading.get_ident() == owner
            loaded.append(i)
            buffer[:] = images[i]
            return buffer

        def progress(n, total):
            assert threading.get_ident() == owner
            events.append(n)

        result = (prepare_profiles if plane else integrate_series)(
            load,
            bounds,
            p,
            mask_loader=lambda i: masks[i],
            variance_loader=(lambda _: np.full((45, 45), 3.0)) if variance else None,
            options=IntegrationOptions(
                pixel_radius=2, background_model="plane" if plane else "mean"
            ),
            execution=FrameExecution(workers, budget),
            progress=progress,
        )
        assert loaded == list(range(7))
        assert events == list(range(8))
        table = result.summation if plane else result
        assert table.execution["peak_payload_bytes"] <= budget
        return result, table

    base, baseline = run(1)
    for workers in [2, 4, 10]:
        result, table = run(workers)
        assert table.frame_hashes == baseline.frame_hashes
        for name in [
            "values",
            "flags",
            "boxes",
            "frame_values",
            "frame_flags",
            "background_models",
        ]:
            if getattr(table, name) is not None:
                assert getattr(table, name).tobytes() == getattr(baseline, name).tobytes()
        if plane:
            for name in ["samples", "valid", "covariance", "offsets"]:
                assert getattr(result, name).tobytes() == getattr(base, name).tobytes()
    # Enough for one frame only: memory admission limits concurrency independently of workers.
    _, table = run(10, 105_000 if variance else 60_000)
    assert table.execution["maximum_in_flight_frames"] == 1


def test_frame_worker_failure_and_cancellation_are_ordered():
    image = np.ones((21, 21))
    bounds = np.column_stack([np.arange(8), np.arange(1, 9)])
    loaded, events = [], []

    def load(i):
        loaded.append(i)
        if i == 2:
            raise RuntimeError("loader failed")
        return image

    with pytest.raises(RuntimeError, match="loader failed"):
        integrate_series(
            load,
            bounds,
            predictions([[10, 10, 2.5]]),
            execution=FrameExecution(4),
            progress=lambda n, _: events.append(n),
        )
    assert loaded == [0, 1, 2]
    assert events == [0, 1, 2]

    def cancel(n, total):
        if n == 1:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        integrate_series(
            lambda _: image,
            bounds,
            predictions([[10, 10, 2.5]]),
            execution=FrameExecution(4),
            progress=cancel,
        )
    with pytest.raises(ValueError, match="payload budget"):
        integrate_series(
            lambda _: image, bounds, predictions([[10, 10, 2.5]]), execution=FrameExecution(4, 1)
        )
    for workers, budget in [(0, 1), (33, 1), (1, 0), (1, 2**30 + 1)]:
        with pytest.raises(ValueError):
            FrameExecution(workers, budget)
