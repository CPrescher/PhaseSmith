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


@dataclass(frozen=True, slots=True, init=False)
class TofPowderPattern:
    """Observed TOF-grid data in microseconds and an optional supplied background."""

    tof_us: NDArray[np.float64]
    observed_y: NDArray[np.float64] | None
    uncertainty: NDArray[np.float64] | None
    mask: NDArray[np.bool_] | None
    background: NDArray[np.float64]

    def __init__(
        self,
        tof_us: ArrayLike,
        *,
        observed_y: ArrayLike | None = None,
        uncertainty: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        background: ArrayLike | None = None,
    ) -> None:
        """Copy and validate microsecond-domain pattern arrays."""

        validated = PowderPattern(
            tof_us,
            observed_y=observed_y,
            uncertainty=uncertainty,
            mask=mask,
            background=background,
        )
        object.__setattr__(self, "tof_us", validated.x)
        object.__setattr__(self, "observed_y", validated.observed_y)
        object.__setattr__(self, "uncertainty", validated.uncertainty)
        object.__setattr__(self, "mask", validated.mask)
        object.__setattr__(self, "background", validated.background)


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


@dataclass(frozen=True, slots=True)
class StructuralReflectionResult:
    """Structure-factor and geometry diagnostics in stable reflection order.

    ``f`` is the representative complex amplitude; ``f_squared`` and
    ``integrated_intensity`` use the Friedel-pair mean for powder calculations.
    Consequently ``f_squared`` need not equal ``abs(f)**2`` with dispersion.
    """

    reflection_ids: tuple[str, ...]
    f: NDArray[np.complex128]
    f_squared: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]
    q_squared_inverse_angstrom2: NDArray[np.float64]
    s_inverse_angstrom: NDArray[np.float64]
    d_spacing_angstrom: NDArray[np.float64]
    two_theta_deg: NDArray[np.float64]
    component_index: NDArray[np.int64] | None = None
    base_reflection_index: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        """Validate every reflection-major diagnostic array."""

        ids = tuple(self.reflection_ids)
        count = len(ids)
        if len(set(ids)) != count or any(not value for value in ids):
            raise ValueError("reflection_ids must be non-empty and unique")
        if self.f.dtype != np.complex128 or self.f.shape != (count,):
            raise ValueError("f must be a complex128 vector matching reflection IDs")
        if not np.isfinite(self.f.real).all() or not np.isfinite(self.f.imag).all():
            raise ValueError("f must contain only finite values")
        for name, array in (
            ("f_squared", self.f_squared),
            ("integrated_intensity", self.integrated_intensity),
            ("q_squared_inverse_angstrom2", self.q_squared_inverse_angstrom2),
            ("s_inverse_angstrom", self.s_inverse_angstrom),
            ("d_spacing_angstrom", self.d_spacing_angstrom),
            ("two_theta_deg", self.two_theta_deg),
        ):
            if array.dtype != np.float64 or array.shape != (count,):
                raise ValueError(f"{name} must be a float64 vector matching reflection IDs")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        for array in (
            self.f,
            self.f_squared,
            self.integrated_intensity,
            self.q_squared_inverse_angstrom2,
            self.s_inverse_angstrom,
            self.d_spacing_angstrom,
            self.two_theta_deg,
        ):
            array.flags.writeable = False
        component_index = (
            np.zeros(count, dtype=np.int64)
            if self.component_index is None
            else np.array(self.component_index, dtype=np.int64, copy=True, order="C")
        )
        base_reflection_index = (
            np.arange(count, dtype=np.int64)
            if self.base_reflection_index is None
            else np.array(self.base_reflection_index, dtype=np.int64, copy=True, order="C")
        )
        for name, array in (
            ("component_index", component_index),
            ("base_reflection_index", base_reflection_index),
        ):
            if array.shape != (count,) or np.any(array < 0):
                raise ValueError(f"{name} must be a nonnegative int64 vector matching reflections")
            array.flags.writeable = False
        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "base_reflection_index", base_reflection_index)
        object.__setattr__(self, "reflection_ids", ids)


@dataclass(frozen=True, slots=True)
class StructuralPatternCalculationResult:
    """One structural phase pattern plus reflection-level diagnostics."""

    phase_id: str
    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background: NDArray[np.float64]
    accumulation: AccumulationResult
    reflections: StructuralReflectionResult

    def __post_init__(self) -> None:
        """Validate pattern arrays and reflection-support alignment."""

        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("phase_id must be a non-empty string")
        count = self.accumulation.y.size
        for name, array in (
            ("y", self.y),
            ("profile_y", self.profile_y),
            ("background", self.background),
        ):
            if array.dtype != np.float64 or array.shape != (count,):
                raise ValueError(f"{name} must be a float64 vector matching the sample count")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            array.flags.writeable = False
        if self.reflections.integrated_intensity.size != self.derivatives.local.peak_count:
            raise ValueError("reflection diagnostics must match local support blocks")

    @property
    def derivatives(self) -> PatternDerivatives:
        """Return the fused analytical profile derivatives."""

        return self.accumulation.derivatives

    @property
    def jacobian(self) -> SupportJacobian | NDArray[np.float64]:
        """Return the selected support-sparse or dense local layout."""

        return self.accumulation.jacobian


@dataclass(frozen=True, slots=True)
class StructuralPatternJvpResult:
    """Structural pattern values and one structural forward derivative product."""

    result: StructuralPatternCalculationResult
    parameter_names: tuple[str, ...]
    d_y: NDArray[np.float64]
    d_integrated_intensity: NDArray[np.float64]
    d_two_theta_deg: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate structural parameter and derivative-vector dimensions."""

        names = tuple(self.parameter_names)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("parameter_names must be non-empty and unique")
        reflection_count = self.result.reflections.integrated_intensity.size
        for name, array, count in (
            ("d_y", self.d_y, self.result.y.size),
            ("d_integrated_intensity", self.d_integrated_intensity, reflection_count),
            ("d_two_theta_deg", self.d_two_theta_deg, reflection_count),
        ):
            if array.dtype != np.float64 or array.shape != (count,):
                raise ValueError(f"{name} must be a float64 vector with the expected length")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            array.flags.writeable = False
        object.__setattr__(self, "parameter_names", names)


@dataclass(frozen=True, slots=True)
class StructuralPatternLinearizationResult:
    """Structural pattern values and a reusable parameter-major Jacobian."""

    result: StructuralPatternCalculationResult
    parameter_names: tuple[str, ...]
    jacobian: NDArray[np.float64]

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        expected = (len(names), self.result.y.size)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("parameter_names must be non-empty and unique")
        if self.jacobian.dtype != np.float64 or self.jacobian.shape != expected:
            raise ValueError("jacobian must be parameter-major and match pattern samples")
        if not np.isfinite(self.jacobian).all():
            raise ValueError("jacobian must contain only finite values")
        self.jacobian.flags.writeable = False
        object.__setattr__(self, "parameter_names", names)


@dataclass(frozen=True, slots=True)
class StructuralPatternVjpResult:
    """Structural pattern values and one reverse product from sample weights."""

    result: StructuralPatternCalculationResult
    parameter_names: tuple[str, ...]
    gradient: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the stable structural-gradient layout."""

        names = tuple(self.parameter_names)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("parameter_names must be non-empty and unique")
        if self.gradient.dtype != np.float64 or self.gradient.shape != (len(names),):
            raise ValueError("gradient must be a float64 vector matching parameter_names")
        if not np.isfinite(self.gradient).all():
            raise ValueError("gradient must contain only finite values")
        self.gradient.flags.writeable = False
        object.__setattr__(self, "parameter_names", names)
