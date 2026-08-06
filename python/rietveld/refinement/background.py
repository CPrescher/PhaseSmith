"""Typed, grid-normalized background models for refinement workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class PolynomialBackground:
    """Additive polynomial on the normalized pattern coordinate ``[-1, 1]``.

    Coefficients are ordered by increasing power and have intensity units.
    The supplied :class:`~rietveld.pattern.PowderPattern` background remains a
    separate fixed contribution; this model is added to it.
    """

    background_id: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.background_id, str)
            or not self.background_id
            or self.background_id != self.background_id.strip()
        ):
            raise ValueError("background_id must be a non-empty trimmed string")
        values = tuple(float(value) for value in self.coefficients)
        if not values or not np.isfinite(values).all():
            raise ValueError("background coefficients must be a non-empty finite tuple")
        object.__setattr__(self, "coefficients", values)

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return sample-major analytical coefficient derivatives."""

        grid = np.asarray(x, dtype=np.float64)
        if grid.ndim != 1 or not np.isfinite(grid).all():
            raise ValueError("background x must be a finite vector")
        if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
            raise ValueError("background x must be strictly increasing")
        if grid.size <= 1:
            normalized = np.zeros(grid.size, dtype=np.float64)
        else:
            normalized = 2.0 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1.0
        result = np.ascontiguousarray(
            np.column_stack([normalized**power for power in range(len(self.coefficients))])
        )
        result.flags.writeable = False
        return result

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the additive background on a validated grid."""

        result = np.ascontiguousarray(
            self.basis(x) @ np.asarray(self.coefficients, dtype=np.float64)
        )
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> PolynomialBackground:
        """Return an immutable model with replacement coefficients."""

        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.coefficients),) or not np.isfinite(values).all():
            raise ValueError("replacement background coefficients have an invalid shape")
        return replace(self, coefficients=tuple(map(float, values)))
