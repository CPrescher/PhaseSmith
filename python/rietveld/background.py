"""Model-independent background estimation and subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

XYPATTERN_REVISION = "6e4574d75d2d6fcefc633f9fbecc27b8f1bcd817"


def _float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have a real floating-point or integer dtype")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result


def smooth_bruckner(
    y: ArrayLike,
    smooth_points: int,
    iterations: int = 50,
) -> NDArray[np.float64]:
    """Return the pinned xypattern-compatible Bruckner smoothed envelope.

    ``smooth_points`` is the half-window in samples. The complete moving mean
    therefore contains ``2 * smooth_points + 1`` samples.
    """

    samples = _float_vector(y, "y")
    if samples.size == 0:
        raise ValueError("y must not be empty")
    points = _non_negative_integer(smooth_points, "smooth_points")
    count = _non_negative_integer(iterations, "iterations")
    return _readonly(_core.smooth_bruckner(samples, points, count))


@dataclass(frozen=True, slots=True)
class BackgroundSubtractionResult:
    """Plain arrays produced by one background-estimation pass."""

    background: NDArray[np.float64]
    corrected_y: NDArray[np.float64]
    smoothed_y: NDArray[np.float64]
    smooth_points: int

    def __post_init__(self) -> None:
        arrays = {
            "background": _float_vector(self.background, "background"),
            "corrected_y": _float_vector(self.corrected_y, "corrected_y"),
            "smoothed_y": _float_vector(self.smoothed_y, "smoothed_y"),
        }
        shape = arrays["background"].shape
        if arrays["corrected_y"].shape != shape or arrays["smoothed_y"].shape != shape:
            raise ValueError("background subtraction arrays must have the same shape")
        for name, values in arrays.items():
            object.__setattr__(self, name, _readonly(values))
        object.__setattr__(
            self,
            "smooth_points",
            _non_negative_integer(self.smooth_points, "smooth_points"),
        )


@dataclass(frozen=True, slots=True)
class SmoothBrucknerBackground:
    """Physical-width Bruckner estimator with optional Chebyshev compression.

    The defaults match xypattern's ``SmoothBrucknerBackground``. Set
    ``chebyshev_order=None`` to use the native smoothed envelope directly.
    """

    smooth_width: float = 0.1
    iterations: int = 50
    chebyshev_order: int | None = 50

    def __post_init__(self) -> None:
        width = float(self.smooth_width)
        if not np.isfinite(width) or width < 0.0:
            raise ValueError("smooth_width must be finite and non-negative")
        object.__setattr__(self, "smooth_width", width)
        object.__setattr__(
            self,
            "iterations",
            _non_negative_integer(self.iterations, "iterations"),
        )
        if self.chebyshev_order is not None:
            object.__setattr__(
                self,
                "chebyshev_order",
                _non_negative_integer(self.chebyshev_order, "chebyshev_order"),
            )

    def _prepare(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        grid = _float_vector(x, "x")
        samples = _float_vector(y, "y")
        if grid.size < 2:
            raise ValueError("x must contain at least two samples")
        if samples.shape != grid.shape:
            raise ValueError("y must have the same shape as x")
        spacing = np.diff(grid)
        if np.any(spacing <= 0.0):
            raise ValueError("x must be strictly increasing")
        if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=0.0):
            raise ValueError("x must be uniformly spaced for a physical smoothing width")
        points = int(self.smooth_width / float(spacing[0]))
        if self.chebyshev_order is not None and self.chebyshev_order >= grid.size:
            raise ValueError("chebyshev_order must be smaller than the number of samples")
        return grid, samples, points

    def _components(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
        grid, samples, points = self._prepare(x, y)
        smoothed = smooth_bruckner(samples, points, self.iterations)
        if self.chebyshev_order is None:
            background = smoothed
        else:
            normalized_x = 2.0 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1.0
            coefficients = np.polynomial.chebyshev.chebfit(
                normalized_x,
                smoothed,
                self.chebyshev_order,
            )
            background = _readonly(np.polynomial.chebyshev.chebval(normalized_x, coefficients))
        return samples, background, smoothed, points

    def estimate(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Estimate and return the background on a uniform increasing grid."""

        _, background, _, _ = self._components(x, y)
        return background

    def subtract(self, x: ArrayLike, y: ArrayLike) -> BackgroundSubtractionResult:
        """Estimate the background and return it alongside ``y - background``."""

        samples, background, smoothed, points = self._components(x, y)
        return BackgroundSubtractionResult(
            background,
            _readonly(samples - background),
            smoothed,
            points,
        )
