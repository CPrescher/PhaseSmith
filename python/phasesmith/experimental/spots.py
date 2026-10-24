"""Experimental detector spot detection and signed aperture intensity extraction.

Requires a source build with ``--features experimental-spots``. Default builds
contain no detector-processing kernel. No indexing or atomic model is involved.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import IntFlag
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _core


class SpotFlag(IntFlag):
    """Measurement limitations, retained individually rather than silently rejected."""

    EDGE = 1
    MASKED = 2
    SATURATED = 4
    OVERLAP = 8
    BACKGROUND = 16
    LARGE = 32
    NO_CENTROID = 64


@dataclass(frozen=True, slots=True)
class SpotOptions:
    """Explicit detection thresholds, finite apertures and safety budgets, in pixels."""

    threshold_sigma: float = 5.0
    threshold_absolute: float = 0.0
    inner_radius: int = 2
    outer_radius: int = 8
    min_pixels: int = 3
    max_pixels: int = 4096
    padding: int = 2
    background_margin: int = 4
    min_background: int = 12
    max_spots: int = 10000

    def __post_init__(self) -> None:
        for name in ("threshold_sigma", "threshold_absolute"):
            value = getattr(self, name)
            if isinstance(value, (bool, complex)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in tuple(self.__dataclass_fields__)[2:]:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < 0 or value > np.iinfo(np.uint32).max:
                raise ValueError(f"{name} outside supported integer range")
            object.__setattr__(self, name, int(value))
        if (
            self.threshold_sigma <= 0
            or self.threshold_absolute < 0
            or not 0 <= self.inner_radius < self.outer_radius <= 128
            or self.padding > 128
            or not 1 <= self.background_margin <= 128
            or not 1 <= self.min_pixels <= self.max_pixels
            or self.min_background < 1
            or not 1 <= self.max_spots <= 1000000
        ):
            raise ValueError("invalid spot support, thresholds or component budgets")


_DEFAULT_OPTIONS = SpotOptions()


def _readonly(values: ArrayLike, dtype: object) -> NDArray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class SpotTable:
    """Immutable measurements. Coordinates are zero-based pixel-center indices.

    ``boxes`` are half-open [x0,y0,x1,y1]. Intensity is the signed sum within
    each finite box, not a total-profile extrapolation or corrected I(hkl).
    Widths are positive-excess moments, not fitted widths or centroid errors.
    Detection-selected intensities have selection bias near the threshold.
    """

    boxes: NDArray[np.int64]
    values: NDArray[np.float64]
    flags: NDArray[np.uint32]
    rejected_small_components: int = 0
    noise_model: str = "poisson_plus_read_noise_with_variance_floor"
    columns: ClassVar[tuple[str, ...]] = (
        "x",
        "y",
        "intensity",
        "sigma",
        "background",
        "background_sigma",
        "peak_value",
        "width_x",
        "width_y",
        "n_pixels",
        "n_background",
    )

    def __post_init__(self) -> None:
        n = len(self.boxes)
        if np.shape(self.boxes) != (n, 4) or np.shape(self.values) != (n, 11):
            raise ValueError("inconsistent spot table dimensions")
        if np.shape(self.flags) != (n,):
            raise ValueError("flags must have one entry per spot")
        object.__setattr__(self, "boxes", _readonly(self.boxes, np.int64))
        object.__setattr__(self, "values", _readonly(self.values, np.float64))
        object.__setattr__(self, "flags", _readonly(self.flags, np.uint32))

    def __len__(self) -> int:
        return len(self.boxes)

    def column(self, name: str) -> NDArray[np.float64]:
        """Return one read-only numerical column."""
        return self.values[:, self.columns.index(name)]

    @property
    def unflagged(self) -> NDArray[np.bool_]:
        """Records without known quality flags; not a calibrated reliability claim."""
        return self.flags == 0


def available() -> bool:
    """Whether this installed extension explicitly enabled experimental spot kernels."""
    return hasattr(_core, "experimental_spots")


def _inputs(
    image: ArrayLike,
    mask: ArrayLike | None,
    variance: ArrayLike | None,
    gain: float,
    read_noise: float,
    saturation: float | None,
) -> tuple:
    raw = np.asarray(image)
    if raw.ndim != 2 or raw.size == 0 or raw.size > np.iinfo(np.uint32).max:
        raise ValueError("image must be nonempty and two-dimensional with <= 2^32-1 pixels")
    if raw.dtype.kind not in "fiu":
        raise ValueError("image must have a real numeric dtype")
    data = np.array(raw, dtype=np.float64, order="C", copy=True)
    excluded = np.zeros(data.shape, dtype=bool) if mask is None else np.asarray(mask)
    if excluded.dtype != np.bool_ or excluded.shape != data.shape:
        raise ValueError("mask must be boolean and match the image shape")
    excluded = np.array(excluded, order="C", copy=True)
    if not np.isfinite(data[~excluded]).all():
        raise ValueError("unmasked image values must be finite")
    gain, read_noise = float(gain), float(read_noise)
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError("gain must be positive and read_noise nonnegative, both finite")
    if not np.isfinite(read_noise * read_noise):
        raise ValueError("read_noise variance overflow")
    if saturation is not None:
        saturation = float(saturation)
        if not np.isfinite(saturation):
            raise ValueError("saturation must be finite")
    var = None
    if variance is not None:
        vr = np.asarray(variance)
        if vr.dtype.kind not in "fiu" or vr.shape != data.shape:
            raise ValueError("variance must be a real array matching the image shape")
        var = np.array(vr, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(var[~excluded]).all() or np.any(var[~excluded] <= 0):
            raise ValueError("unmasked variances must be finite and positive")
    return data, excluded, var, gain, read_noise, saturation


def _boxes(boxes: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(boxes)
    if raw.ndim != 2 or raw.shape[1] != 4 or raw.dtype.kind not in "iu":
        raise ValueError("boxes must be an integer array with shape (n,4)")
    if np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("box coordinates outside signed integer range")
    return np.array(raw, dtype=np.int64, copy=True, order="C")


def _run(
    image: ArrayLike,
    boxes: ArrayLike | None,
    *,
    mask: ArrayLike | None,
    variance: ArrayLike | None,
    options: SpotOptions,
    gain: float,
    read_noise: float,
    saturation: float | None,
) -> SpotTable:
    if not available():
        raise RuntimeError(
            "This build has no spot engine; build PhaseSmith with "
            "maturin develop --release --features experimental-spots"
        )
    if not isinstance(options, SpotOptions):
        raise TypeError("options must be SpotOptions")
    data, excluded, var, gain, read_noise, saturation = _inputs(
        image, mask, variance, gain, read_noise, saturation
    )
    result = _core.experimental_spots(
        data.ravel(),
        *data.shape,
        excluded.ravel(),
        astuple(options),
        gain,
        read_noise,
        saturation,
        None if var is None else var.ravel(),
        None if boxes is None else _boxes(boxes),
    )
    return SpotTable(
        *result,
        noise_model="supplied_independent_variance"
        if var is not None
        else "poisson_plus_read_noise_with_variance_floor",
    )


def detect_spots(
    image: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    options: SpotOptions = _DEFAULT_OPTIONS,
    gain: float = 1.0,
    read_noise: float = 0.0,
    saturation: float | None = None,
) -> SpotTable:
    """Find spots and sum padded signal boxes; all retained quality flags are returned.

    Thresholding uses two annular-mean passes and 8-connectivity. No crystal,
    calibration, element list, known cell, profile or atomic structure is required.
    """
    return _run(
        image,
        None,
        mask=mask,
        variance=variance,
        options=options,
        gain=gain,
        read_noise=read_noise,
        saturation=saturation,
    )


def extract_spots(
    image: ArrayLike,
    boxes: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    options: SpotOptions = _DEFAULT_OPTIONS,
    gain: float = 1.0,
    read_noise: float = 0.0,
    saturation: float | None = None,
) -> SpotTable:
    """Measure caller-supplied half-open boxes, retaining weak and negative intensities.

    All supplied signal boxes are excluded from every background ring. Masked
    signal pixels are omitted and flagged; no missing-signal correction is applied.
    """
    return _run(
        image,
        boxes,
        mask=mask,
        variance=variance,
        options=options,
        gain=gain,
        read_noise=read_noise,
        saturation=saturation,
    )
