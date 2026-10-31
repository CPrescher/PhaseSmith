# SPDX-License-Identifier: MIT
"""Portable original-frame validity; True means excluded, never zero intensity.

The illumination estimator adapts Dioptas's MIT-licensed frame_validity.py at
651c27f98108fb19f60db9a8cb227755b2f9d6a2. See THIRD_PARTY_NOTICES.md. Readers,
display transformations, application caches and resource policy remain callers'
responsibility. SciPy is imported only by the illumination estimator.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .rotation import angle_segments

__all__ = [
    "CombinedValidity",
    "DacApertureMask",
    "FrameValidity",
    "SeatShadowOptions",
    "SeatShadowUnavailable",
    "estimate_seat_shadows",
    "flag_invalid_tracks",
]

_MAX_CELLS = 50_000_000
_SCHEMA = "phasesmith.frame-validity.v1"


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    if not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(value)


def _shape(shape):
    if len(shape) != 2:
        raise ValueError("detector shape must have two dimensions")
    result = tuple(_integer(v, "detector dimension", 1, 100_000) for v in shape)
    if np.prod(result) > 100_000_000:
        raise ValueError("detector pixel budget exceeded")
    return result


def _ids(frame_ids, count):
    result = tuple(str(i) for i in range(count)) if frame_ids is None else tuple(frame_ids)
    if (
        len(result) != count
        or any(not isinstance(v, str) or not v for v in result)
        or len(set(result)) != count
    ):
        raise ValueError("frame_ids must be unique nonempty strings in source order")
    return result


def _readonly(array, dtype=None):
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _box_counts(mask, boxes):
    """Vectorized rectangular support checks; x1/y1 are exclusive."""
    b = np.asarray(boxes)
    if b.ndim != 2 or b.shape[1] != 4 or b.dtype.kind not in "iu":
        raise ValueError("boxes must be integer (n,4) arrays")
    h, w = mask.shape
    inside = (b[:, :2] >= 0).all(axis=1) & (b[:, 2:] <= [w, h]).all(axis=1)
    inside &= (b[:, 2:] > b[:, :2]).all(axis=1)
    out = np.ones(len(b), np.int64)
    if inside.any():
        x0, y0, x1, y1 = b[inside].astype(np.int64).T
        if mask.any():
            integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
            out[inside] = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        else:
            out[inside] = 0
    return out


def _coarse_boxes_valid(mask, boxes, shape, block, fixed):
    b = np.asarray(boxes)
    if b.ndim != 2 or b.shape[1] != 4 or b.dtype.kind not in "iu":
        raise ValueError("boxes must be integer (n,4) arrays")
    inside = (b[:, :2] >= 0).all(axis=1) & (b[:, 2:] <= shape[::-1]).all(axis=1)
    inside &= (b[:, 2:] > b[:, :2]).all(axis=1)
    coarse = np.c_[b[:, :2] // block, (b[:, 2:] + block - 1) // block]
    result = inside & (_box_counts(mask, coarse) == 0)
    if fixed is not None:
        result &= _box_counts(fixed, b) == 0
    return _readonly(result)


class _MaskAccess:
    def validate(self, frame_ids, shape=None, omega_bounds=None):
        if tuple(frame_ids) != self.frame_ids or (shape is not None and tuple(shape) != self.shape):
            raise ValueError("validity belongs to different ordered frames or detector shape")
        if (
            omega_bounds is not None
            and self.omega_bounds is not None
            and not np.array_equal(omega_bounds, self.omega_bounds)
        ):
            raise ValueError("validity exposure angles differ")

    def verify_frame(self, frame, image):
        """Check layout and, for inferred masks, original-count source hash."""
        frame = _integer(frame, "frame", 0, len(self.frame_ids) - 1)
        array = np.asarray(image)
        if array.shape != self.shape or array.dtype.kind not in "fiu":
            raise ValueError("source frame differs from validity detector layout")
        hashes = getattr(self, "provenance", {}).get("image_sha256")
        if hashes is not None:
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if len(hashes) != len(self.frame_ids) or digest != hashes[frame]:
                raise ValueError("source frame hash differs from validity snapshot")

    def valid_points(self, frames, xy):
        """Centre eligibility only; integration must still consume full masks."""
        f, p = np.asarray(frames), np.asarray(xy)
        if f.ndim != 1 or f.dtype.kind not in "iu" or p.shape != (len(f), 2):
            raise ValueError("expected integer frames and real (n,2) points")
        if p.dtype.kind not in "fiu":
            raise ValueError("points must be real")
        result = np.zeros(len(f), bool)
        for frame in np.unique(f):
            if frame < 0 or frame >= len(self.frame_ids):
                continue
            sel = np.flatnonzero((f == frame) & np.isfinite(p).all(axis=1))
            good = (p[sel] >= -0.5).all(axis=1) & (p[sel] < np.asarray(self.shape)[::-1] - 0.5).all(
                axis=1
            )
            sel = sel[good]
            xy_int = np.floor(p[sel] + 0.5).astype(np.int64)
            result[sel] = ~self.mask(int(frame))[xy_int[:, 1], xy_int[:, 0]]
        return _readonly(result)

    def boxes_valid(self, frame, boxes):
        return _readonly(_box_counts(self.mask(frame), boxes) == 0)


@dataclass(frozen=True, init=False)
class FrameValidity(_MaskAccess):
    """Owned coarse exclusions plus an optional precise fixed pixel mask.

    Every coarse True cell excludes its entire detector block, including a
    partial edge block. Arrays use exactly the orientation supplied by callers.
    Provenance is copied on access. No reader or GUI dependency is retained.
    """

    frame_ids: tuple[str, ...]
    shape: tuple[int, int]
    excluded: np.ndarray
    block: int
    fixed_mask: np.ndarray | None
    omega_bounds: np.ndarray | None
    _provenance: str
    _fingerprint: str

    def __init__(
        self,
        frame_ids,
        shape,
        excluded,
        block=1,
        *,
        fixed_mask=None,
        omega_bounds=None,
        provenance=None,
    ):
        shape = _shape(shape)
        block = _integer(block, "block", 1, 1024)
        a = np.asarray(excluded)
        if a.ndim != 3 or not 1 <= a.shape[0] <= 1_000_000:
            raise ValueError("one boolean coarse mask is required per frame")
        frame_ids = _ids(frame_ids, len(a))
        expected = (len(a), *((v + block - 1) // block for v in shape))
        if a.dtype != bool or a.shape != expected or a.size > _MAX_CELLS:
            raise ValueError("invalid coarse mask shape/dtype or mask cell budget")
        fixed = None if fixed_mask is None else np.asarray(fixed_mask)
        if fixed is not None and (fixed.shape != shape or fixed.dtype != bool):
            raise ValueError("fixed mask must be boolean with detector shape")
        bounds = None if omega_bounds is None else angle_segments(omega_bounds)[0]
        if bounds is not None and len(bounds) != len(a):
            raise ValueError("one exposure is required per mask frame")
        for name, value in (
            ("frame_ids", frame_ids),
            ("shape", shape),
            ("block", block),
            ("excluded", _readonly(a)),
            ("fixed_mask", None if fixed is None else _readonly(fixed)),
            ("omega_bounds", None if bounds is None else _readonly(bounds)),
            ("_provenance", _json({} if provenance is None else provenance)),
        ):
            object.__setattr__(self, name, value)
        digest = hashlib.sha256(_json(self._header()).encode())
        for array in (self.excluded, self.fixed_mask, self.omega_bounds):
            if array is not None:
                digest.update(memoryview(array).cast("B"))
        object.__setattr__(self, "_fingerprint", digest.hexdigest())

    @property
    def provenance(self):
        return json.loads(self._provenance)

    def fingerprint(self):
        return self._fingerprint

    def _header(self):
        return dict(
            schema=_SCHEMA,
            frame_ids=self.frame_ids,
            shape=self.shape,
            block=self.block,
            fixed_mask=self.fixed_mask is not None,
            omega_bounds=self.omega_bounds is not None,
            coordinates="caller-supplied detector array; zero-based pixel centres",
            semantics="True=excluded; never zero/interpolated intensity",
            provenance=self.provenance,
        )

    def metadata(self):
        return dict(self._header(), fingerprint=self.fingerprint())

    def mask(self, frame):
        frame = _integer(frame, "frame", 0, len(self.frame_ids) - 1)
        a = np.repeat(np.repeat(self.excluded[frame], self.block, 0), self.block, 1)
        a = a[: self.shape[0], : self.shape[1]].copy()
        if self.fixed_mask is not None:
            a |= self.fixed_mask
        return a

    native_mask = mask

    def rows(self, start, end):
        """Bounded flattened-pixel by frame tile; never expand the mask cube."""
        start = _integer(start, "start", 0, int(np.prod(self.shape)))
        end = _integer(end, "end", start, int(np.prod(self.shape)))
        if (end - start) * len(self.frame_ids) > _MAX_CELLS:
            raise ValueError("mask tile cell budget exceeded")
        y, x = np.divmod(np.arange(start, end), self.shape[1])
        a = self.excluded[:, y // self.block, x // self.block].T.copy()
        if self.fixed_mask is not None:
            a |= self.fixed_mask.ravel()[start:end, None]
        return a

    def boxes_valid(self, frame, boxes):
        frame = _integer(frame, "frame", 0, len(self.frame_ids) - 1)
        return _coarse_boxes_valid(
            self.excluded[frame], boxes, self.shape, self.block, self.fixed_mask
        )

    def save(self, path):
        arrays = dict(excluded=self.excluded)
        for name in ("fixed_mask", "omega_bounds"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        arrays["metadata"] = np.frombuffer(_json(self.metadata()).encode(), np.uint8)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path, *, frame_ids=None, shape=None, omega_bounds=None):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].tobytes())
            if meta.get("schema") != _SCHEMA:
                raise ValueError("unsupported frame-validity archive schema")
            expected = {"excluded", "metadata"} | {
                n for n in ("fixed_mask", "omega_bounds") if meta[n]
            }
            if set(archive.files) != expected:
                raise ValueError("frame-validity archive fields disagree with metadata")
            result = cls(
                meta["frame_ids"],
                meta["shape"],
                archive["excluded"],
                meta["block"],
                fixed_mask=archive["fixed_mask"] if meta["fixed_mask"] else None,
                omega_bounds=archive["omega_bounds"] if meta["omega_bounds"] else None,
                provenance=meta["provenance"],
            )
        if result.fingerprint() != meta["fingerprint"]:
            raise ValueError("frame-validity archive fingerprint mismatch")
        if _json(result.metadata()) != _json(meta):
            raise ValueError("frame-validity archive metadata mismatch")
        result.validate(result.frame_ids if frame_ids is None else frame_ids, shape, omega_bounds)
        return result


@dataclass(frozen=True)
class SeatShadowOptions:
    reference_first: int | None = None
    reference_last: int | None = None  # inclusive
    threshold: float = 0.85
    guard_pixels: int = 32
    block: int = 16

    def __post_init__(self):
        if (
            isinstance(self.threshold, (bool, np.bool_, str, complex))
            or not np.isscalar(self.threshold)
            or not np.isfinite(self.threshold)
            or not 0 < self.threshold < 1
        ):
            raise ValueError("seat-shadow threshold must lie strictly between zero and one")
        _integer(self.guard_pixels, "guard_pixels", 0, 4096)
        _integer(self.block, "block", 1, 1024)
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "guard_pixels", int(self.guard_pixels))
        object.__setattr__(self, "block", int(self.block))
        for name in ("reference_first", "reference_last"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, "reference frame", 0, 1_000_000))


class SeatShadowUnavailable(ValueError):
    """Original frames do not provide enough illuminated reference information."""


def _coarse_frame(image, mask, block):
    digest = hashlib.sha256(memoryview(image).cast("B")).hexdigest()
    complete = not (image.shape[0] % block or image.shape[1] % block)
    valid = np.isfinite(image) & ~mask
    if np.any(image[valid] < 0):
        raise ValueError(
            "seat-shadow estimation requires original nonnegative counts, not residuals"
        )
    if np.any(image[valid] > 1e30):
        raise ValueError("original counts exceed the 1e30 coarse-estimator range")
    if complete and valid.all():
        c = np.median(
            image.reshape(image.shape[0] // block, block, image.shape[1] // block, block),
            axis=(1, 3),
        )
    else:
        a = np.where(valid, image, np.nan)
        a = np.pad(
            a, ((0, (-a.shape[0]) % block), (0, (-a.shape[1]) % block)), constant_values=np.nan
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(
                a.reshape(a.shape[0] // block, block, a.shape[1] // block, block), axis=(1, 3)
            )
    return np.nan_to_num(c), digest, hashlib.sha256(mask.tobytes()).hexdigest(), bool(valid.any())


def estimate_seat_shadows(
    frame_loader,
    frame_count,
    *,
    frame_ids=None,
    mask_loader=None,
    fixed_mask=None,
    omega_bounds=None,
    options=None,
    workers=1,
    progress=None,
):
    """Estimate broad side-connected attenuation from original count arrays.

    Read each source frame once, in order on the calling thread. At most workers
    owned frame snapshots are in flight; retain only coarse frames. Detection
    normalization never changes counts or infers a wholly blocked frame from
    uniform dimming. Caller-owned orientation is used without any implicit flip.
    """
    try:
        from scipy import ndimage
    except ImportError as error:
        raise ImportError("seat-shadow estimation requires phasesmith[masking] (SciPy)") from error
    count = _integer(frame_count, "frame_count", 1, 1_000_000)
    if count < 5:
        raise SeatShadowUnavailable("seat-shadow estimation needs at least five frames")
    ids = _ids(frame_ids, count)
    bounds = None if omega_bounds is None else angle_segments(omega_bounds)[0]
    if bounds is not None and len(bounds) != count:
        raise ValueError("one exposure is required per source frame")
    workers = _integer(workers, "workers", 1, 32)
    opts = SeatShadowOptions() if options is None else options
    if not isinstance(opts, SeatShadowOptions):
        raise TypeError("options must be SeatShadowOptions")
    first = round(0.37 * (count - 1)) if opts.reference_first is None else opts.reference_first
    last = round(0.59 * (count - 1)) if opts.reference_last is None else opts.reference_last
    if not 0 <= first < last < count:
        raise ValueError("choose a valid inclusive illuminated reference-frame range")
    shape = None

    def load(index):
        nonlocal shape
        image = np.asarray(frame_loader(index))
        if image.ndim != 2 or image.dtype.kind not in "fiu":
            raise ValueError("original frames must be real two-dimensional count arrays")
        if shape is None:
            shape = _shape(image.shape)
            cells = count * np.prod([(v + opts.block - 1) // opts.block for v in shape])
            if cells > _MAX_CELLS:
                raise ValueError("coarse frame cell budget exceeded")
        if image.shape != shape:
            raise ValueError("all source frames must have the same detector shape")
        mask = np.zeros(shape, bool) if mask_loader is None else np.asarray(mask_loader(index))
        if mask.shape != shape or mask.dtype != bool:
            raise ValueError("source mask must be boolean with detector shape")
        if fixed_mask is not None:
            fixed = np.asarray(fixed_mask)
            if fixed.shape != shape or fixed.dtype != bool:
                raise ValueError("fixed mask must be boolean with detector shape")
            mask = mask | fixed
        return np.array(image, order="C", copy=True), np.array(mask, order="C", copy=True)

    coarse, hashes, mask_hashes, available = [], [], [], []
    if progress:
        progress(0, count)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phasesmith-shadow") as pool:
        pending = deque()
        index = 0
        while index < count or pending:
            while index < count and len(pending) < workers:
                image, mask = load(index)
                pending.append(pool.submit(_coarse_frame, image, mask, opts.block))
                index += 1
            c, digest, mask_digest, has_valid_pixels = pending.popleft().result()
            coarse.append(c)
            hashes.append(digest)
            mask_hashes.append(mask_digest)
            available.append(has_valid_pixels)
            if progress:
                progress(len(coarse), count)
    smooth = ndimage.gaussian_filter(
        ndimage.median_filter(np.asarray(coarse, dtype=np.float32), size=(1, 3, 3)), sigma=(0, 1, 1)
    )
    references = np.linspace(first, last, min(5, last - first + 1)).astype(int)
    reference = np.median(smooth[references], axis=0)
    if not np.any(reference > 40):
        raise SeatShadowUnavailable("illuminated reference background must exceed 40 counts")
    ratio = smooth / np.maximum(reference, 1)
    scales = np.quantile(ratio[:, reference > 40], 0.9, axis=1)
    if np.any((scales <= 0) & available):
        raise SeatShadowUnavailable(
            "wholly dark unmasked frames have unknown visibility; supply reviewed masks or geometry"
        )
    ratio /= np.maximum(scales[:, None, None], 1e-6)
    support = reference > 25
    _, xx = np.indices(reference.shape)
    sides = (xx * opts.block < shape[1] * 350 / 2048) | (xx * opts.block > shape[1] * 1700 / 2048)
    excluded = []
    for index, frame in enumerate(ratio):
        if not available[index]:
            excluded.append(np.ones(reference.shape, bool))
            continue
        candidate = ndimage.binary_fill_holes(
            ndimage.binary_closing((frame < opts.threshold) & support)
        )
        labels, _ = ndimage.label(candidate)
        sizes = np.bincount(labels.ravel())
        connected = np.unique(labels[sides])
        connected = connected[(connected != 0) & (sizes[connected] >= 20)]
        keep = np.isin(labels, connected)
        margin = int(np.ceil(opts.guard_pixels / opts.block))
        excluded.append(ndimage.binary_dilation(keep, iterations=margin) if margin else keep)
    return FrameValidity(
        ids,
        shape,
        np.asarray(excluded),
        opts.block,
        fixed_mask=fixed_mask,
        omega_bounds=bounds,
        provenance=dict(
            algorithm="seat-illumination-ratio-v2",
            options=asdict(opts),
            reference_indices=references.tolist(),
            image_sha256=hashes,
            input_mask_sha256=mask_hashes,
            detection_scales=scales.tolist(),
            globally_low_signal_frames=np.flatnonzero(scales < 0.5).tolist(),
            known_unavailable_frames=np.flatnonzero(~np.asarray(available)).tolist(),
            low_signal_policy="diagnostic only; exclusions follow spatial shadow components",
            input_kind="original nonnegative counts; no normalization applied to measurements",
            upstream="Dioptas 651c27f98108fb19f60db9a8cb227755b2f9d6a2; MIT",
        ),
    )


def flag_invalid_tracks(table, validity):
    """Preserve rows/values and flag a masked support or adjacent exposure.

    Track boxes and ranges are half-open. Check one neighboring exposure on
    either side within the same contiguous angular segment. Never bridge a gap.
    Integration/profile callers must also consume the original per-frame masks.
    """
    from .rotation import PositionTable

    if not isinstance(table, PositionTable):
        raise TypeError("table must be PositionTable")
    if len(validity.frame_ids) != len(table.omega_bounds):
        raise ValueError("validity frame count differs from position series")
    validity.validate(validity.frame_ids, omega_bounds=table.omega_bounds)
    _, segments = angle_segments(table.omega_bounds)
    first, last = table.frame_ranges.T
    if np.any(first < 0) or np.any(last > len(segments)) or np.any(last <= first):
        raise ValueError("track frame ranges are invalid")
    bad = np.zeros(len(table), bool)
    for frame in range(len(segments)):
        same = (frame >= first) & (frame < last)
        before = (frame == first - 1) & (segments[frame] == segments[first])
        after = (frame == last) & (segments[frame] == segments[last - 1])
        ids = np.flatnonzero(same | before | after)
        if len(ids):
            bad[ids] |= ~validity.boxes_valid(frame, table.boxes[ids])
    flags = table.flags.copy()
    flags[bad] |= 2  # MASKED, same bit used by native detector/integration tables.
    return replace(table, flags=flags)


def _dot_range(rays, axis, normal, low, high):
    """Exact extrema of ray dot R(axis,omega)normal on a closed interval."""
    parallel = (axis @ normal) * axis
    c = rays @ parallel
    a, b = rays @ (normal - parallel), rays @ np.cross(axis, normal)
    ends = (c + a * np.cos(low) + b * np.sin(low), c + a * np.cos(high) + b * np.sin(high))
    minimum, maximum = np.minimum(*ends), np.maximum(*ends)
    phase, radius = np.arctan2(b, a), np.hypot(a, b)
    period = 2 * np.pi
    for point, value, maximum_point in (
        (phase, c + radius, True),
        (phase + np.pi, c - radius, False),
    ):
        occurs = np.ceil((low - point) / period) <= np.floor((high - point) / period)
        if maximum_point:
            maximum = np.where(occurs, np.maximum(maximum, value), maximum)
        else:
            minimum = np.where(occurs, np.minimum(minimum, value), minimum)
    return minimum, maximum


@dataclass(frozen=True, init=False)
class DacApertureMask(_MaskAccess):
    """Conservative pixel/exposure masks for a declared symmetric conical DAC.

    Four pixel corners and the complete closed angular interval must stay in
    the same cone. Convexity of each cone covers the affine pixel interior.
    Non-affine pixel maps require caller-supplied mapped masks instead. At most
    tile_rows detector rows are evaluated at once; no full mask cube is cached.
    """

    geometry: object
    frame_ids: tuple[str, ...]
    shape: tuple[int, int]
    omega_bounds: np.ndarray
    axis_at_omega_zero: np.ndarray
    half_opening_degrees: float
    fixed_mask: np.ndarray | None
    tile_rows: int
    pixel_block: int

    def __init__(
        self,
        geometry,
        omega_bounds,
        shape,
        *,
        axis_at_omega_zero,
        half_opening_degrees,
        frame_ids=None,
        fixed_mask=None,
        tile_rows=64,
        pixel_block=1,
    ):
        from .reciprocal import PanelGeometry

        if not isinstance(geometry, PanelGeometry):
            raise TypeError("geometry must be PanelGeometry")
        if geometry.pixel_map is not None:
            raise ValueError("non-affine pixel maps require explicitly mapped caller masks")
        bounds = angle_segments(omega_bounds)[0]
        shape = _shape(shape)
        normal = np.asarray(axis_at_omega_zero)
        if (
            normal.shape != (3,)
            or normal.dtype.kind not in "fiu"
            or not np.isfinite(normal).all()
            or not np.any(normal)
        ):
            raise ValueError("DAC axis must be a finite nonzero real triple")
        normal = normal.astype(np.float64)
        if not np.isfinite(normal).all():
            raise ValueError("DAC axis must be representable as finite float64")
        normal = normal / np.max(abs(normal))
        normal = normal / np.linalg.norm(normal)
        opening = half_opening_degrees
        if (
            isinstance(opening, (bool, np.bool_, str, complex))
            or not np.isscalar(opening)
            or not np.isfinite(opening)
            or not 0 < opening < 90
        ):
            raise ValueError("DAC half-opening must lie strictly between zero and 90 degrees")
        fixed = None if fixed_mask is None else np.asarray(fixed_mask)
        if fixed is not None and (fixed.shape != shape or fixed.dtype != bool):
            raise ValueError("fixed mask must be boolean with detector shape")
        for name, value in (
            ("geometry", geometry),
            ("omega_bounds", _readonly(bounds)),
            ("shape", shape),
            ("frame_ids", _ids(frame_ids, len(bounds))),
            ("axis_at_omega_zero", _readonly(normal)),
            ("half_opening_degrees", float(opening)),
            ("fixed_mask", None if fixed is None else _readonly(fixed)),
            ("tile_rows", _integer(tile_rows, "tile_rows", 1, 1024)),
            ("pixel_block", _integer(pixel_block, "pixel_block", 1, 1024)),
        ):
            object.__setattr__(self, name, value)

    def metadata(self):
        return dict(
            schema="phasesmith.dac-aperture-mask.v1",
            frame_ids=self.frame_ids,
            shape=self.shape,
            geometry_vectors=self.geometry.vectors.tolist(),
            wavelength=self.geometry.wavelength,
            omega_bounds=self.omega_bounds.tolist(),
            axis_at_omega_zero=self.axis_at_omega_zero.tolist(),
            half_opening_degrees=self.half_opening_degrees,
            pixel_block=self.pixel_block,
            support=(
                "closed full exposure and four affine pixel corners; same cone; cosine margin 8eps"
            ),
            fixed_mask_sha256=None
            if self.fixed_mask is None
            else hashlib.sha256(self.fixed_mask.tobytes()).hexdigest(),
        )

    def fingerprint(self):
        # Tiling is an implementation detail; it cannot alter the mask identity.
        return hashlib.sha256(_json(self.metadata()).encode()).hexdigest()

    def _coarse_mask(self, frame):
        frame = _integer(frame, "frame", 0, len(self.frame_ids) - 1)
        low, high = np.deg2rad(sorted(self.omega_bounds[frame]))
        origin, fast, slow, beam, axis = self.geometry.vectors
        axis, beam = axis / np.linalg.norm(axis), beam / np.linalg.norm(beam)
        limit = np.cos(np.deg2rad(self.half_opening_degrees)) - 8 * np.finfo(float).eps
        low_in, high_in = _dot_range(beam[None], axis, self.axis_at_omega_zero, low, high)
        h, w = self.shape
        block = self.pixel_block
        ch, cw = (h + block - 1) // block, (w + block - 1) // block
        if not ((low_in[0] >= limit) or (high_in[0] <= -limit)):
            return np.ones((ch, cw), bool)
        result = np.empty((ch, cw), bool)
        rows_per_tile = max(1, self.tile_rows // block)
        for y0 in range(0, ch, rows_per_tile):
            y1 = min(ch, y0 + rows_per_tile)
            yy, xx = np.mgrid[y0:y1, :cw]
            minimum = np.full(xx.size, np.inf)
            maximum = np.full(xx.size, -np.inf)
            for right, bottom in ((False, False), (False, True), (True, False), (True, True)):
                x = np.minimum((xx.ravel() + int(right)) * block, w) - 0.5
                y = np.minimum((yy.ravel() + int(bottom)) * block, h) - 0.5
                rays = origin + x[:, None] * fast + y[:, None] * slow
                lengths = np.linalg.norm(rays, axis=1)
                if np.any(lengths == 0):
                    raise ValueError("pixel corner coincides with sample origin")
                rays /= lengths[:, None]
                lo, hi = _dot_range(rays, axis, self.axis_at_omega_zero, low, high)
                minimum = np.minimum(minimum, lo)
                maximum = np.maximum(maximum, hi)
            result[y0:y1] = ~((minimum >= limit) | (maximum <= -limit)).reshape(y1 - y0, cw)
        return result

    def mask(self, frame):
        coarse = self._coarse_mask(frame)
        result = np.repeat(np.repeat(coarse, self.pixel_block, 0), self.pixel_block, 1)
        result = result[: self.shape[0], : self.shape[1]].copy()
        if self.fixed_mask is not None:
            result |= self.fixed_mask
        return result

    def boxes_valid(self, frame, boxes):
        return _coarse_boxes_valid(
            self._coarse_mask(frame), boxes, self.shape, self.pixel_block, self.fixed_mask
        )


@dataclass(frozen=True)
class CombinedValidity(_MaskAccess):
    """OR-composition of shared providers, evaluated one frame at a time."""

    providers: tuple

    def __post_init__(self):
        providers = tuple(self.providers)
        if not providers or any(not isinstance(p, _MaskAccess) for p in providers):
            raise ValueError("at least one frame-validity provider is required")
        first = providers[0]
        bounds = next((p.omega_bounds for p in providers if p.omega_bounds is not None), None)
        for provider in providers:
            provider.validate(first.frame_ids, first.shape, bounds)
        object.__setattr__(self, "providers", providers)

    @property
    def frame_ids(self):
        return self.providers[0].frame_ids

    @property
    def shape(self):
        return self.providers[0].shape

    @property
    def omega_bounds(self):
        return next((p.omega_bounds for p in self.providers if p.omega_bounds is not None), None)

    def mask(self, frame):
        result = self.providers[0].mask(frame)
        for provider in self.providers[1:]:
            result |= provider.mask(frame)
        return result

    def verify_frame(self, frame, image):
        for provider in self.providers:
            provider.verify_frame(frame, image)

    def boxes_valid(self, frame, boxes):
        result = np.ones(len(boxes), bool)
        for provider in self.providers:
            result &= provider.boxes_valid(frame, boxes)
        return _readonly(result)

    def fingerprint(self):
        return hashlib.sha256(_json([p.fingerprint() for p in self.providers]).encode()).hexdigest()
