"""Prediction-driven finite-aperture rotation integration, independent of any GUI.

See docs/experimental-reflection-integration.md for equations and finite support.
The frame loader is called once per exposure, in source order. All measurement
and cross-frame accumulation runs in Rust; Python orchestrates frame arrays.
"""

import hashlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntFlag
from typing import ClassVar

import numpy as np

from phasesmith import _core

from .prediction import ScanPredictions, _hkl, _readonly
from .rotation import angle_segments
from .spots import SpotOptions, _inputs, extract_spots


@dataclass(frozen=True)
class FrameExecution:
    """Bounded owned frame payloads; ordinary output/workspace is separate."""

    workers: int = 1
    max_in_flight_bytes: int = 512 * 1024**2

    def __post_init__(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("frame workers must be in 1..32")
        if type(self.max_in_flight_bytes) is not int or not 1 <= self.max_in_flight_bytes <= 2**30:
            raise ValueError("frame payload budget must be in 1 byte..1 GiB")


def _ordered_frames(load, measure, count, execution, stats):
    """Caller-owned loaders; bounded native tasks and source-order results/errors."""
    stats.update(
        requested_workers=execution.workers,
        maximum_in_flight_frames=0,
        payload_budget_bytes=execution.max_in_flight_bytes,
        peak_payload_bytes=0,
    )

    def prepare(frame):
        payload = load(frame)
        image, _mask, variance = payload
        size = sum(a.nbytes for a in payload if a is not None)
        size += 2 * image.size * (9 + (8 if variance is not None else 0))
        if size > execution.max_in_flight_bytes:
            raise ValueError("one frame exceeds the declared frame payload budget")
        return payload, size

    if execution.workers == 1:
        for frame in range(count):
            payload, size = prepare(frame)
            stats["maximum_in_flight_frames"] = 1
            stats["peak_payload_bytes"] = max(stats["peak_payload_bytes"], size)
            yield measure(frame, payload)
        return
    pool = ThreadPoolExecutor(max_workers=execution.workers, thread_name_prefix="phasesmith-frames")
    pending = deque()
    charged = 0
    largest = 0
    try:
        for frame in range(count):
            # Release completed work before requesting another loader snapshot.
            while pending and (
                len(pending) >= execution.workers
                or charged + largest > execution.max_in_flight_bytes
            ):
                old_size, future = pending.popleft()
                yield future.result()
                charged -= old_size
            try:
                payload, size = prepare(frame)
                largest = max(largest, size)
            except Exception as error:
                future = Future()
                future.set_exception(error)
                pending.append((0, future))
                break
            while pending and charged + size > execution.max_in_flight_bytes:
                old_size, future = pending.popleft()
                yield future.result()
                charged -= old_size
            pending.append((size, pool.submit(measure, frame, payload)))
            charged += size
            stats["maximum_in_flight_frames"] = max(stats["maximum_in_flight_frames"], len(pending))
            stats["peak_payload_bytes"] = max(stats["peak_payload_bytes"], charged)
        while pending:
            _, future = pending.popleft()
            yield future.result()
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


class IntegrationFlag(IntFlag):
    EDGE = 1
    MASKED = 2
    SATURATED = 4
    OVERLAP = 8
    BACKGROUND = 16
    ANGULAR = 256
    NO_DATA = 1024
    BOUNDARY = 2048
    BACKGROUND_MODEL = 4096


@dataclass(frozen=True)
class IntegrationOptions:
    pixel_radius: int = 5
    omega_half_width: float = 1.0
    background_margin: int = 4
    min_background: int = 12
    max_observations: int = 1_000_000
    background_model: str = "mean"
    background_gap: int = 0
    background_clip: float = 4.0

    def __post_init__(self):
        for name, low, high in (
            ("pixel_radius", 0, 128),
            ("background_margin", 1, 128),
            ("min_background", 1, 1_000_000),
            ("max_observations", 1, 1_000_000),
            ("background_gap", 0, 128),
        ):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer))
                or not low <= v <= high
            ):
                raise ValueError(f"{name} must be an integer in [{low},{high}]")
            object.__setattr__(self, name, int(v))
        if (
            isinstance(self.omega_half_width, (bool, complex))
            or not np.isscalar(self.omega_half_width)
            or not np.isfinite(self.omega_half_width)
            or not 0 < self.omega_half_width <= 90
        ):
            raise ValueError("omega_half_width must be finite in (0,90] degrees")
        object.__setattr__(self, "omega_half_width", float(self.omega_half_width))
        if self.background_model not in ("mean", "plane"):
            raise ValueError("background_model must be mean or plane")
        if self.background_model == "mean" and self.background_gap:
            raise ValueError("background_gap requires the plane background model")
        if self.background_model == "plane" and self.min_background < 3:
            raise ValueError("plane background requires at least three pixels")
        if (
            isinstance(self.background_clip, (bool, complex))
            or not np.isscalar(self.background_clip)
            or not np.isfinite(self.background_clip)
            or not 2 <= self.background_clip <= 10
        ):
            raise ValueError("background_clip must be finite in [2,10]")
        object.__setattr__(self, "background_clip", float(self.background_clip))


@dataclass(frozen=True)
class ReflectionTable:
    background_columns: ClassVar = (
        "plane_offset",
        "plane_x",
        "plane_y",
        "dI_doffset",
        "dI_dx",
        "dI_dy",
        "candidate_background",
        "rejected_background",
        "background_residual_rms",
        "boundary_intensity",
        "boundary_sigma",
        "boundary_fraction",
    )
    hkl: np.ndarray
    predicted: np.ndarray
    grain_ids: np.ndarray
    values: np.ndarray
    flags: np.ndarray
    observation_ids: np.ndarray
    frame_indices: np.ndarray
    boxes: np.ndarray
    frame_values: np.ndarray
    frame_flags: np.ndarray
    omega_bounds: np.ndarray
    frame_hashes: tuple
    options: IntegrationOptions
    noise_model: str
    noise_parameters: tuple | None = None
    background_models: np.ndarray | None = None
    angular_half_widths: np.ndarray | None = None
    execution: dict | None = None
    columns: ClassVar = (
        "intensity",
        "sigma",
        "background_counts",
        "n_pixels",
        "n_frames",
        "n_background",
    )

    def __post_init__(self):
        for name in (
            "hkl",
            "predicted",
            "grain_ids",
            "values",
            "flags",
            "observation_ids",
            "frame_indices",
            "boxes",
            "frame_values",
            "frame_flags",
            "omega_bounds",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.background_models is not None:
            if np.shape(self.background_models) != (len(self.frame_indices), 12):
                raise ValueError("background_models must have shape (observations,12)")
            object.__setattr__(self, "background_models", _readonly(self.background_models))

        if self.angular_half_widths is not None:
            widths = np.asarray(self.angular_half_widths)
            if (
                widths.shape != (len(self.hkl),)
                or not np.isfinite(widths).all()
                or (widths <= 0).any()
                or (widths > 90).any()
            ):
                raise ValueError("invalid per-reflection angular half widths")
            object.__setattr__(self, "angular_half_widths", _readonly(widths))

    def __len__(self):
        return len(self.hkl)

    def column(self, name):
        return self.values[:, self.columns.index(name)]

    def rocking_curve(self, reflection):
        """Read-only arrays for one crossing, preserving source-frame order and widths."""
        if type(reflection) is not int or not 0 <= reflection < len(self):
            raise IndexError("reflection index outside table")
        selected = self.observation_ids == reflection
        frames = self.frame_indices[selected]
        bounds = self.omega_bounds[frames]
        v = self.frame_values[selected]
        return {
            name: _readonly(value)
            for name, value in dict(
                frame_indices=frames,
                omega_bounds=bounds,
                omega=bounds.mean(axis=1),
                intensity=v[:, 2],
                sigma=v[:, 3],
                background=v[:, 4],
                n_pixels=v[:, 9],
                n_background=v[:, 10],
                flags=self.frame_flags[selected],
            ).items()
        }


def integrate_series(
    frame_loader,
    omega_bounds,
    predictions,
    *,
    mask_loader=None,
    variance_loader=None,
    grain_ids=None,
    options=None,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
    progress=None,
    execution=None,
    _profile_sink=None,
    _profile_half_widths=None,
):
    """Measure every predicted crossing, including weak and negative observations.

    predictions is ScanPredictions (typically from predict_scan). Concatenate
    competing grains and supply grain_ids to flag their shared apertures.
    Callbacks return image-shaped NumPy arrays. Optional progress(done,total)
    may raise to cancel; no partial result is returned. Loaders may reuse buffers.
    Bounds may run forwards or backwards; overlapping exposures are rejected.
    """
    if not hasattr(_core, "experimental_integrate_reduce"):
        raise RuntimeError("build PhaseSmith with --features experimental-spots for integration")
    if not isinstance(predictions, ScanPredictions):
        raise TypeError("predictions must be ScanPredictions")
    options = options or IntegrationOptions()
    if not isinstance(options, IntegrationOptions):
        raise TypeError("options must be IntegrationOptions")
    execution = FrameExecution() if execution is None else execution
    if not isinstance(execution, FrameExecution):
        raise TypeError("execution must be FrameExecution")
    hkl = _hkl(predictions.hkl)
    xyz = np.asarray(predictions.values)
    if xyz.dtype.kind not in "fiu" or xyz.shape != (len(hkl), 3) or not np.isfinite(xyz).all():
        raise ValueError("predicted x,y,omega must be finite real (n,3)")
    xyz = np.array(xyz, dtype=float, order="C", copy=True)
    if (abs(xyz) > 1e8).any() or (np.all(hkl == 0, axis=1)).any():
        raise ValueError("prediction coordinates exceed budget or HKL is zero")
    grains = np.zeros(len(hkl), np.int64) if grain_ids is None else np.asarray(grain_ids)
    if (
        grains.shape != (len(hkl),)
        or grains.dtype.kind not in "iu"
        or (grains < 0).any()
        or (grains > np.iinfo(np.int64).max).any()
    ):
        raise ValueError("grain_ids must be nonnegative signed-range integers, one per crossing")
    grains = np.array(grains, dtype=np.int64, copy=True)
    identity = np.column_stack([hkl, xyz, grains])
    if len(np.unique(identity, axis=0)) != len(hkl):
        raise ValueError("duplicate predicted crossing for one grain")
    bounds, _ = angle_segments(omega_bounds)
    low, high = np.min(bounds, axis=1), np.max(bounds, axis=1)
    order = np.argsort(low)
    lows, highs = low[order], high[order]
    if (lows[1:] < highs[:-1] - 1e-8).any():
        raise ValueError("overlapping angular exposures require a separate normalization model")
    widths = (
        np.full(len(xyz), options.omega_half_width)
        if _profile_half_widths is None
        else np.asarray(_profile_half_widths)
    )
    if (
        widths.shape != (len(xyz),)
        or widths.dtype.kind not in "fiu"
        or not np.isfinite(widths).all()
        or (widths <= 0).any()
        or (widths > 90).any()
    ):
        raise ValueError("invalid per-reflection angular half widths")
    start, end = xyz[:, 2] - widths, xyz[:, 2] + widths
    incomplete = (start < lows[0] - 1e-8) | (end > highs[-1] + 1e-8)
    for left, right in zip(highs[:-1], lows[1:], strict=True):
        if right > left + 1e-8:
            incomplete |= (start < right) & (end > left)
    extraction = SpotOptions(
        background_margin=options.background_margin,
        min_background=options.min_background,
        max_spots=1_000_000,
    )
    ids_parts, frames_parts, boxes_parts, values_parts, flags_parts = [], [], [], [], []
    hashes, shape, observations = [], None, 0
    model_parts = []

    def load(frame):
        nonlocal shape, observations
        image = np.array(frame_loader(frame), copy=True)
        mask = None if mask_loader is None else np.array(mask_loader(frame), copy=True)
        variance = None if variance_loader is None else np.array(variance_loader(frame), copy=True)
        if shape is None:
            shape = image.shape
            if (
                (xyz[:, 0] < -0.5)
                | (xyz[:, 0] >= shape[1] - 0.5)
                | (xyz[:, 1] < -0.5)
                | (xyz[:, 1] >= shape[0] - 0.5)
            ).any():
                raise ValueError("predicted centres must lie on the detector footprint")
        if image.shape != shape:
            raise ValueError("all frames must have the same detector shape")
        observations += int(np.count_nonzero((start < high[frame]) & (end > low[frame])))
        if observations > options.max_observations:
            raise ValueError("integration observation budget exceeded")
        return image, mask, variance

    def measure(frame, payload):
        image, mask, variance = payload
        data, excluded, var, _, _, _ = _inputs(image, mask, variance, gain, read_noise, saturation)
        digest = hashlib.sha256()
        for a in (data, excluded, var):
            if a is not None:
                digest.update(str((a.shape, a.dtype.str)).encode())
                digest.update(a.tobytes(order="C"))
        ids = np.flatnonzero((start < high[frame]) & (end > low[frame]))
        table, model, extracted = None, None, None
        if len(ids):
            centres = np.floor(xyz[ids, :2] + 0.5).astype(np.int64)
            r = options.pixel_radius
            boxes = np.column_stack(
                [np.maximum(centres - r, 0), np.minimum(centres + r + 1, [shape[1], shape[0]])]
            )
            if options.background_model == "plane":
                if not hasattr(_core, "experimental_local_background"):
                    raise RuntimeError("rebuild PhaseSmith with the local-background kernel")
                from .spots import SpotTable

                backend = (
                    _core.experimental_profile_frame
                    if _profile_sink is not None
                    else _core.experimental_local_background
                )
                extracted = backend(
                    data.ravel(),
                    shape[0],
                    shape[1],
                    excluded.ravel(),
                    boxes,
                    options.background_gap,
                    options.background_margin,
                    options.min_background,
                    options.background_clip,
                    gain,
                    read_noise,
                    saturation,
                    None if var is None else var.ravel(),
                )
                b, v, f, model = extracted[:4]
                table = SpotTable(b, v, f)
            else:
                table = extract_spots(
                    data,
                    boxes,
                    mask=excluded,
                    variance=var,
                    options=extraction,
                    gain=gain,
                    read_noise=read_noise,
                    saturation=saturation,
                )
        return frame, digest.hexdigest(), ids, table, model, extracted

    stats = {}
    if progress:
        progress(0, len(bounds))
    results = _ordered_frames(load, measure, len(bounds), execution, stats)
    try:
        for frame, digest, ids, table, model, extracted in results:
            hashes.append(digest)
            if len(ids):
                if model is not None:
                    model_parts.append(model)
                if _profile_sink is not None:
                    _profile_sink(ids, frame, table.boxes, extracted[4], extracted[5], extracted[6])
                ids_parts.append(ids)
                frames_parts.append(np.full(len(ids), frame, np.int64))
                boxes_parts.append(table.boxes)
                values_parts.append(table.values)
                flags_parts.append(table.flags & np.uint32(~64 & 0xFFFFFFFF))
            if progress:
                progress(frame + 1, len(bounds))
    finally:
        results.close()
    ids = np.concatenate(ids_parts) if ids_parts else np.empty(0, np.int64)
    frames = np.concatenate(frames_parts) if frames_parts else np.empty(0, np.int64)
    boxes = np.concatenate(boxes_parts) if boxes_parts else np.empty((0, 4), np.int64)
    values = np.concatenate(values_parts) if values_parts else np.empty((0, 11))
    flags = np.concatenate(flags_parts) if flags_parts else np.empty(0, np.uint32)
    sums, quality = _core.experimental_integrate_reduce(len(hkl), ids, values, flags)
    quality[incomplete] |= int(IntegrationFlag.ANGULAR)
    return ReflectionTable(
        hkl,
        xyz,
        grains,
        sums,
        quality,
        ids,
        frames,
        boxes,
        values,
        flags,
        bounds,
        tuple(hashes),
        options,
        "supplied_independent_pixel_variance"
        if variance_loader
        else "poisson_plus_read_noise_with_variance_floor",
        (float(gain), float(read_noise), None if saturation is None else float(saturation)),
        (np.concatenate(model_parts) if model_parts else np.empty((0, 12)))
        if options.background_model == "plane"
        else None,
        widths,
        stats,
    )
