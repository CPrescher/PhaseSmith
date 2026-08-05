"""Typed powder-pattern inputs and high-level calculation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .results import AccumulationResult, PatternDerivatives, SupportJacobian


def _optional_vector(
    values: ArrayLike | None,
    name: str,
    count: int,
    *,
    positive: bool = False,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, init=False)
class PowderPattern:
    """Observed-grid data and an optional supplied background."""

    x: NDArray[np.float64]
    observed_y: NDArray[np.float64] | None
    uncertainty: NDArray[np.float64] | None
    mask: NDArray[np.bool_] | None
    background: NDArray[np.float64]

    def __init__(
        self,
        x: ArrayLike,
        *,
        observed_y: ArrayLike | None = None,
        uncertainty: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        background: ArrayLike | None = None,
    ) -> None:
        """Copy and validate script-facing pattern arrays."""

        grid = np.array(x, dtype=np.float64, copy=True, order="C")
        if grid.ndim != 1 or not np.isfinite(grid).all():
            raise ValueError("x must be a finite one-dimensional array")
        if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
            raise ValueError("x must be strictly increasing")
        count = int(grid.size)
        observed = _optional_vector(observed_y, "observed_y", count)
        sigma = _optional_vector(uncertainty, "uncertainty", count, positive=True)
        if mask is None:
            mask_array = None
        else:
            raw_mask = np.asarray(mask)
            if raw_mask.dtype != np.bool_:
                raise ValueError("mask must contain boolean values")
            mask_array = np.array(raw_mask, dtype=np.bool_, copy=True, order="C")
            if mask_array.shape != (count,):
                raise ValueError(f"mask must have shape ({count},)")
            mask_array.flags.writeable = False
        if background is None:
            background_array = np.zeros(count, dtype=np.float64)
        else:
            background_array = _optional_vector(background, "background", count)
            if background_array is None:  # pragma: no cover - narrowed above
                raise RuntimeError("background validation failed")
        grid.flags.writeable = False
        background_array.flags.writeable = False
        object.__setattr__(self, "x", grid)
        object.__setattr__(self, "observed_y", observed)
        object.__setattr__(self, "uncertainty", sigma)
        object.__setattr__(self, "mask", mask_array)
        object.__setattr__(self, "background", background_array)


@dataclass(frozen=True, slots=True)
class PhasePatternComponent:
    """Diagnostic calculated profile for one identified phase."""

    phase_id: str
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the diagnostic label and one-dimensional finite curve."""

        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("phase component ID must be a non-empty string")
        if self.y.dtype != np.float64 or self.y.ndim != 1 or not np.isfinite(self.y).all():
            raise ValueError("phase component y must be a finite float64 vector")


@dataclass(frozen=True, slots=True)
class PatternCalculationResult:
    """Total pattern, fused profile result, and durable reflection labels."""

    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background: NDArray[np.float64]
    accumulation: AccumulationResult
    reflection_keys: tuple[tuple[str, str], ...]
    phase_offsets: NDArray[np.int64]
    phase_components: tuple[PhasePatternComponent, ...] = ()

    def __post_init__(self) -> None:
        """Validate high-level array dimensions and durable label mappings."""

        sample_count = self.accumulation.y.size
        for name, array in (
            ("y", self.y),
            ("profile_y", self.profile_y),
            ("background", self.background),
        ):
            if array.dtype != np.float64 or array.shape != (sample_count,):
                raise ValueError(f"{name} must be a float64 vector matching the sample count")
        if self.phase_offsets.dtype != np.int64 or self.phase_offsets.ndim != 1:
            raise ValueError("phase_offsets must be a one-dimensional int64 array")
        if self.phase_offsets.size == 0 or self.phase_offsets[0] != 0:
            raise ValueError("phase_offsets must begin at zero")
        if np.any(np.diff(self.phase_offsets) < 0):
            raise ValueError("phase_offsets must be nondecreasing")
        if int(self.phase_offsets[-1]) != len(self.reflection_keys):
            raise ValueError("phase_offsets must span every reflection key")
        if len(self.reflection_keys) != self.derivatives.local.peak_count:
            raise ValueError("reflection keys must match local support blocks")
        if self.phase_components and len(self.phase_components) != self.phase_offsets.size - 1:
            raise ValueError("phase components must match phase offsets")
        if any(component.y.shape != (sample_count,) for component in self.phase_components):
            raise ValueError("phase component sample counts must match the total pattern")

    @property
    def derivatives(self) -> PatternDerivatives:
        """Return the fused profile derivatives."""

        return self.accumulation.derivatives

    @property
    def jacobian(self) -> SupportJacobian | NDArray[np.float64]:
        """Return the selected local Jacobian layout."""

        return self.accumulation.jacobian

    def phase_y(self, phase_id: str) -> NDArray[np.float64]:
        """Return one requested diagnostic phase curve."""

        for component in self.phase_components:
            if component.phase_id == phase_id:
                return component.y
        raise KeyError(f"no diagnostic phase component for {phase_id!r}")
