"""Single-crystal streaming rotation-series position hunting.

File formats and angle interpretation belong to the caller. Angles are degrees;
frame bounds are exposure start/end, not an inferred step or motor readback.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .. import _core
from .spots import SpotOptions, SpotTable, _readonly, available, detect_spots

__all__ = ["PositionTable", "RotationOptions", "angle_segments", "link_spots", "locate_series"]

AMBIGUOUS = 128
SCAN_BOUNDARY = 256


@dataclass(frozen=True, slots=True)
class RotationOptions:
    """Distance per adjacent frame (pixels), explicit safety budgets."""

    link_radius: float = 5.0
    max_candidates: int = 10000000
    max_observations: int = 1000000

    def __post_init__(self):
        if (
            isinstance(self.link_radius, (bool, np.bool_, complex, str))
            or not np.isscalar(self.link_radius)
            or not np.isfinite(self.link_radius)
        ):
            raise ValueError("link_radius must be finite")
        if not 0 < self.link_radius <= 1e6:
            raise ValueError("link_radius must be positive and <= 1e6")
        for name in ("max_candidates", "max_observations"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not 0 <= v <= np.iinfo(np.uint32).max:
                raise ValueError(f"{name} out of range")
        if self.max_observations == 0:
            raise ValueError("invalid rotation observation budget")


def angle_segments(omega_bounds) -> tuple[NDArray, NDArray]:
    """Validate ordered exposures; gaps delimit separate segments.

    Forward and reverse scans may have overlapping angular exposures.
    Direction reversals are rejected. Angular adjacency tolerance is 1e-8 degrees.
    """
    raw = np.asarray(omega_bounds)
    if (
        raw.dtype.kind not in "fiu"
        or raw.ndim != 2
        or raw.shape[1] != 2
        or not 0 < len(raw) <= np.iinfo(np.uint32).max
    ):
        raise ValueError("omega_bounds must be a nonempty real (n_frames,2) array")
    bounds = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(bounds).all() or np.any(np.abs(bounds) > 1e8):
        raise ValueError("omega bounds must be finite and within +/-1e8 degrees")
    width = bounds[:, 1] - bounds[:, 0]
    direction = np.sign(width[0])
    if direction == 0 or np.any(direction * width <= 0):
        raise ValueError("frame widths must be nonzero and have one scan direction")
    gap = direction * (bounds[1:, 0] - bounds[:-1, 1])
    if np.any(direction * np.diff(bounds[:, 0]) <= 0):
        raise ValueError("omega exposure starts must be strictly ordered")
    segments = np.r_[0, np.cumsum(gap > 1e-8)].astype(np.uint32)
    return bounds, segments


_DEFAULT_ROTATION = RotationOptions()
_DEFAULT_SPOTS = SpotOptions()


def link_spots(tables: Sequence[SpotTable], omega_bounds, *, options=_DEFAULT_ROTATION):
    """Return contiguous track labels in frame/spot order; no gaps are bridged."""
    if not available() or not hasattr(_core, "experimental_rotation_link"):
        raise RuntimeError("build PhaseSmith with --features spots  for rotation support")
    if not isinstance(options, RotationOptions):
        raise TypeError("options must be RotationOptions")
    bounds, segments = angle_segments(omega_bounds)
    if len(tables) != len(bounds):
        raise ValueError("one detection table is required per omega exposure")
    n = sum(len(t) for t in tables)
    if n > options.max_observations:
        raise ValueError("rotation observation budget exceeded")
    frames = np.repeat(np.arange(len(tables), dtype=np.uint32), [len(t) for t in tables])
    xy = np.concatenate([t.values[:, :2] for t in tables], axis=0)
    labels = _core.experimental_rotation_link(
        frames, np.ascontiguousarray(xy), segments, options.link_radius, options.max_candidates
    )
    return _readonly(labels, np.uint32)


@dataclass(frozen=True, slots=True)
class PositionTable:
    """Unindexed position candidates; detected weights are not integrated intensities.

    Observations contain [track, frame, x, y, omega_midpoint]. Frame-level
    positions are retained because detector x/y can move during rotation.
    Track means summarize detected samples only, so angular tails below the
    detection threshold do not contribute. No angular precision is inferred.
    """

    boxes: NDArray
    frame_ranges: NDArray
    values: NDArray
    flags: NDArray
    observations: NDArray
    omega_bounds: NDArray
    frame_hashes: tuple[str, ...]
    columns: ClassVar = ("x", "y", "omega", "omega_rms", "n_observations")

    def __post_init__(self):
        n = len(self.values)
        if (
            np.shape(self.values) != (n, 5)
            or np.shape(self.boxes) != (n, 4)
            or np.shape(self.frame_ranges) != (n, 2)
            or np.shape(self.flags) != (n,)
        ):
            raise ValueError("inconsistent position table shapes")
        if np.ndim(self.observations) != 2 or np.shape(self.observations)[1] != 5:
            raise ValueError("observations must have shape (n,5)")
        for name, dtype in (
            ("boxes", np.int64),
            ("frame_ranges", np.int64),
            ("values", np.float64),
            ("flags", np.uint32),
            ("observations", np.float64),
            ("omega_bounds", np.float64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    def __len__(self):
        return len(self.values)

    def column(self, name):
        return self.values[:, self.columns.index(name)]


def _measure_frame(image, mask, spot_options, gain, read_noise, saturation, variance=None):
    table = detect_spots(
        image,
        mask=mask,
        options=spot_options,
        gain=gain,
        read_noise=read_noise,
        saturation=saturation,
        variance=variance,
    )
    fingerprint = hashlib.sha256()
    for a in (image, mask):
        fingerprint.update(str((a.shape, a.dtype)).encode())
        fingerprint.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    if variance is not None:
        fingerprint.update(str((variance.shape, variance.dtype)).encode())
        fingerprint.update(memoryview(np.ascontiguousarray(variance)).cast("B"))
    return table, fingerprint.hexdigest()


def _detect_frames(
    frame_loader,
    mask_loader,
    count,
    workers,
    options,
    gain,
    read_noise,
    saturation,
    variance_loader=None,
):
    """Bounded detection; loaders and ordered callbacks stay on the calling thread."""
    shape = None

    def load(i):
        nonlocal shape
        image = np.asarray(frame_loader(i))
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise ValueError("all series frames must have the same detector shape")
        mask = np.zeros(shape, bool) if mask_loader is None else np.asarray(mask_loader(i))
        variance = None if variance_loader is None else np.asarray(variance_loader(i))
        if workers > 1:
            # A caller may reuse a loading buffer. Snapshot before requesting
            # another frame, not later when a detector thread happens to start.
            image, mask = image.copy(), mask.copy()
            if variance is not None:
                variance = variance.copy()
        return image, mask, variance

    if workers == 1:
        for i in range(count):
            image, mask, variance = load(i)
            yield i, _measure_frame(image, mask, options, gain, read_noise, saturation, variance)
        return

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phasesmith-spots")
    pending = deque()
    next_frame, failed = 0, False

    def submit():
        nonlocal next_frame, failed
        i = next_frame
        next_frame += 1
        try:
            image, mask, variance = load(i)
            future = pool.submit(
                _measure_frame, image, mask, options, gain, read_noise, saturation, variance
            )
        except Exception as exc:
            # Surface loader/shape errors in source order, after earlier callbacks.
            future = Future()
            future.set_exception(exc)
            failed = True
        pending.append((i, future))

    try:
        while next_frame < min(count, workers) and not failed:
            submit()
        while pending:
            i, future = pending.popleft()
            yield i, future.result()
            if next_frame < count and not failed:
                submit()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def locate_series(
    frame_loader: Callable[[int], NDArray],
    omega_bounds,
    *,
    mask_loader: Callable[[int], NDArray] | None = None,
    variance_loader: Callable[[int], NDArray] | None = None,
    spot_options=_DEFAULT_SPOTS,
    rotation_options=_DEFAULT_ROTATION,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
    progress: Callable[[int, int], None] | None = None,
    frame_finished: Callable[[int, SpotTable], None] | None = None,
    workers: int = 1,
) -> PositionTable:
    """First-stage peak hunt: read a series once and return x/y/omega candidates.

    Existing detector sums serve only as positive centroid weights and noise
    thresholds. This performs no forced-aperture rocking integration, profile
    fit, reciprocal-space transform, unit-cell solution or symmetry assignment.
    frame_finished receives a zero-based index and immutable per-frame detections
    after fingerprinting, before progress/linking. It runs on the caller's thread;
    exceptions abort processing. It must not mutate or retain mutable loader data.

    workers=1 retains streaming serial execution. With more workers, at most
    workers frames are in flight, with owned image/mask snapshots. Loaders remain
    serial on the caller's thread but can run ahead of completed-frame observers.
    Observers, progress, budgets and linking consume results in source order.
    A callback error stops loading and joins/cancels the bounded pending work.
    variance_loader optionally supplies independent per-pixel variances in
    squared count units. It shares the image shape/mask and must be positive
    and finite at unmasked pixels. Variances are fingerprinted with each frame.
    """
    bounds, segments = angle_segments(omega_bounds)
    if not isinstance(spot_options, SpotOptions) or not isinstance(
        rotation_options, RotationOptions
    ):
        raise TypeError("expected SpotOptions and RotationOptions")
    if not available() or not hasattr(_core, "experimental_rotation_link"):
        raise RuntimeError("build PhaseSmith with --features spots  for rotation support")
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)):
        raise TypeError("workers must be an integer from 1 to 32")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be an integer from 1 to 32")
    workers = min(int(workers), len(bounds))
    tables, hashes = [], []
    count = 0
    detections = _detect_frames(
        frame_loader,
        mask_loader,
        len(bounds),
        workers,
        spot_options,
        gain,
        read_noise,
        saturation,
        variance_loader,
    )
    try:
        for i, (table, fingerprint) in detections:
            tables.append(table)
            hashes.append(fingerprint)
            count += len(table)
            if count > rotation_options.max_observations:
                raise ValueError("rotation observation budget exceeded")
            if frame_finished:
                frame_finished(i, table)
            if progress:
                progress(i + 1, len(bounds))
    finally:
        detections.close()
    labels = link_spots(tables, bounds, options=rotation_options)
    frames = np.repeat(np.arange(len(bounds)), [len(t) for t in tables])
    detected = np.concatenate([t.values for t in tables])
    detected_boxes = np.concatenate([t.boxes for t in tables])
    detected_flags = np.concatenate([t.flags for t in tables])
    n = int(labels.max()) + 1 if len(labels) else 0
    boxes = np.empty((n, 4), np.int64)
    ranges = np.empty((n, 2), np.int64)
    values = np.empty((n, 5))
    flags = np.zeros(n, np.uint32)
    groups = np.split(np.argsort(labels, kind="stable"), np.cumsum(np.bincount(labels))[:-1])
    for track, indices in enumerate(groups if n else []):
        fs = frames[indices]
        bs = detected_boxes[indices]
        boxes[track] = [*bs[:, :2].min(axis=0), *bs[:, 2:].max(axis=0)]
        ranges[track] = [fs.min(), fs.max() + 1]
        samples = np.column_stack(
            [detected[indices, :2], bounds[fs].mean(axis=1), detected[indices, 2]]
        )
        summary = _core.experimental_rotation_centroid(np.ascontiguousarray(samples))
        values[track] = summary
        flags[track] = np.bitwise_or.reduce(detected_flags[indices])
        if not np.isfinite(summary[:3]).all():
            flags[track] |= 64
        if len(np.unique(fs)) != len(fs):
            flags[track] |= AMBIGUOUS
        if fs.min() == 0 or fs.max() == len(bounds) - 1:
            flags[track] |= SCAN_BOUNDARY
        if fs.min() > 0 and segments[fs.min() - 1] != segments[fs.min()]:
            flags[track] |= SCAN_BOUNDARY
        if fs.max() + 1 < len(bounds) and segments[fs.max() + 1] != segments[fs.max()]:
            flags[track] |= SCAN_BOUNDARY
    observations = np.column_stack([labels, frames, detected[:, :2], bounds[frames].mean(axis=1)])
    return PositionTable(boxes, ranges, values, flags, observations, bounds, tuple(hashes))
