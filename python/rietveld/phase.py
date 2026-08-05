"""Typed phase and reflection geometry models.

This module intentionally contains no profile evaluation or refinement state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly_float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, init=False)
class ReflectionGeometryBatch:
    """Plain contiguous geometry and base intensities for reflections."""

    hkl: NDArray[np.int64]
    d_spacing_angstrom: NDArray[np.float64]
    two_theta_deg: NDArray[np.float64]
    base_integrated_intensity: NDArray[np.float64]

    def __init__(
        self,
        hkl: ArrayLike,
        d_spacing_angstrom: ArrayLike,
        two_theta_deg: ArrayLike,
        base_integrated_intensity: ArrayLike,
    ) -> None:
        """Copy and validate a reflection structure-of-arrays batch."""

        raw_hkl = np.asarray(hkl)
        if raw_hkl.ndim != 2 or raw_hkl.shape[1] != 3:
            raise ValueError("hkl must have shape (reflection_count, 3)")
        if not np.issubdtype(raw_hkl.dtype, np.integer):
            raise ValueError("hkl must contain integer indices")
        hkl_array = np.array(raw_hkl, dtype=np.int64, copy=True, order="C")
        if not np.array_equal(raw_hkl, hkl_array):
            raise ValueError("hkl indices must fit in signed 64-bit integers")
        d_spacing = _readonly_float_vector(d_spacing_angstrom, "d_spacing_angstrom")
        positions = _readonly_float_vector(two_theta_deg, "two_theta_deg")
        intensities = _readonly_float_vector(base_integrated_intensity, "base_integrated_intensity")
        count = hkl_array.shape[0]
        if d_spacing.size != count or positions.size != count or intensities.size != count:
            raise ValueError("all reflection arrays must have the same reflection count")
        if np.any(d_spacing <= 0.0):
            raise ValueError("d_spacing_angstrom must be positive")
        if np.any((positions <= 0.0) | (positions >= 180.0)):
            raise ValueError("two_theta_deg must lie strictly within (0, 180)")
        hkl_array.flags.writeable = False
        object.__setattr__(self, "hkl", hkl_array)
        object.__setattr__(self, "d_spacing_angstrom", d_spacing)
        object.__setattr__(self, "two_theta_deg", positions)
        object.__setattr__(self, "base_integrated_intensity", intensities)

    @property
    def reflection_count(self) -> int:
        """Return the number of reflections."""

        return int(self.two_theta_deg.size)


@dataclass(frozen=True, slots=True, init=False)
class ReciprocalMetric:
    """Symmetric positive-definite reciprocal metric tensor."""

    matrix: NDArray[np.float64]

    def __init__(self, matrix: ArrayLike) -> None:
        """Copy and validate a 3-by-3 reciprocal metric."""

        value = np.array(matrix, dtype=np.float64, copy=True, order="C")
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ValueError("reciprocal metric must be a finite 3-by-3 matrix")
        if not np.allclose(value, value.T, rtol=0.0, atol=1e-13):
            raise ValueError("reciprocal metric must be symmetric")
        if np.any(np.linalg.eigvalsh(value) <= 0.0):
            raise ValueError("reciprocal metric must be positive definite")
        value.flags.writeable = False
        object.__setattr__(self, "matrix", value)

    @classmethod
    def orthogonal(
        cls, a_angstrom: float, b_angstrom: float, c_angstrom: float
    ) -> ReciprocalMetric:
        """Construct the reciprocal metric for an orthogonal direct cell."""

        lengths = np.asarray([a_angstrom, b_angstrom, c_angstrom], dtype=np.float64)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("orthogonal cell lengths must be positive and finite")
        return cls(np.diag(1.0 / lengths**2))
