"""Typed analytical background models for refinement workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _grid(x: ArrayLike) -> NDArray[np.float64]:
    grid = np.asarray(x, dtype=np.float64)
    if grid.ndim != 1 or not np.isfinite(grid).all():
        raise ValueError("background x must be a finite vector")
    if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
        raise ValueError("background x must be strictly increasing")
    return grid


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("background_id must be a non-empty trimmed string")


def _parameters(values: tuple[float, ...]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    result.flags.writeable = False
    return result


@runtime_checkable
class DifferentiableBackground(Protocol):
    """Immutable analytical background accepted by refinement."""

    background_id: str

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def coefficients(self) -> tuple[float, ...]: ...

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]: ...

    def basis(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def replace_coefficients(self, coefficients: ArrayLike) -> DifferentiableBackground: ...


@dataclass(frozen=True, slots=True)
class PolynomialBackground:
    """Power series on the normalized pattern coordinate ``[-1, 1]``."""

    background_id: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_id(self.background_id)
        values = tuple(float(value) for value in self.coefficients)
        if not values or not np.isfinite(values).all():
            raise ValueError("background coefficients must be a non-empty finite tuple")
        object.__setattr__(self, "coefficients", values)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"coefficient_{index}" for index in range(len(self.coefficients)))

    @property
    def parameters(self) -> NDArray[np.float64]:
        return _parameters(self.coefficients)

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return ((-np.inf, np.inf),) * len(self.coefficients)

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        grid = _grid(x)
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
        result = np.ascontiguousarray(self.basis(x) @ self.parameters)
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> PolynomialBackground:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.coefficients),) or not np.isfinite(values).all():
            raise ValueError("replacement background coefficients have an invalid shape")
        return replace(self, coefficients=tuple(map(float, values)))

    def with_parameters(self, parameters: ArrayLike) -> PolynomialBackground:
        return self.replace_coefficients(parameters)


@dataclass(frozen=True, slots=True)
class ChebyshevBackground:
    """Chebyshev series on one explicit closed coordinate domain."""

    background_id: str
    coefficients: tuple[float, ...]
    domain_deg: tuple[float, float]

    def __post_init__(self) -> None:
        _validate_id(self.background_id)
        coefficients = tuple(float(value) for value in self.coefficients)
        domain = tuple(float(value) for value in self.domain_deg)
        if not coefficients or not np.isfinite(coefficients).all():
            raise ValueError("background coefficients must be a non-empty finite tuple")
        if len(domain) != 2 or not np.isfinite(domain).all() or domain[0] >= domain[1]:
            raise ValueError("domain_deg must contain two increasing finite values")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "domain_deg", domain)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"coefficient_{index}" for index in range(len(self.coefficients)))

    @property
    def parameters(self) -> NDArray[np.float64]:
        return _parameters(self.coefficients)

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return ((-np.inf, np.inf),) * len(self.coefficients)

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        grid = _grid(x)
        lower, upper = self.domain_deg
        tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(lower), abs(upper), 1.0)
        if np.any((grid < lower - tolerance) | (grid > upper + tolerance)):
            raise ValueError("background x lies outside domain_deg")
        normalized = 2.0 * (grid - lower) / (upper - lower) - 1.0
        result = np.ascontiguousarray(
            np.polynomial.chebyshev.chebvander(normalized, len(self.coefficients) - 1)
        )
        result.flags.writeable = False
        return result

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]:
        result = np.ascontiguousarray(self.basis(x) @ self.parameters)
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> ChebyshevBackground:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.coefficients),) or not np.isfinite(values).all():
            raise ValueError("replacement background coefficients have an invalid shape")
        return replace(self, coefficients=tuple(map(float, values)))

    def with_parameters(self, parameters: ArrayLike) -> ChebyshevBackground:
        return self.replace_coefficients(parameters)


@dataclass(frozen=True, slots=True)
class PointBackground:
    """Linear interpolation through fixed knots with refinable values."""

    background_id: str
    knot_x: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_id(self.background_id)
        knots = tuple(float(value) for value in self.knot_x)
        values = tuple(float(value) for value in self.values)
        if len(knots) < 2 or len(values) != len(knots):
            raise ValueError("point background requires at least two matching knots and values")
        if not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0.0):
            raise ValueError("point background knots must be finite and strictly increasing")
        if not np.isfinite(values).all():
            raise ValueError("point background values must be finite")
        object.__setattr__(self, "knot_x", knots)
        object.__setattr__(self, "values", values)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return self.values

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"value_{index}" for index in range(len(self.values)))

    @property
    def parameters(self) -> NDArray[np.float64]:
        return _parameters(self.values)

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return ((-np.inf, np.inf),) * len(self.values)

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        grid = _grid(x)
        knots = np.asarray(self.knot_x)
        result = np.zeros((grid.size, knots.size), dtype=np.float64)
        right = np.searchsorted(knots, grid, side="right")
        result[right == 0, 0] = 1.0
        result[right == knots.size, -1] = 1.0
        interior = (right > 0) & (right < knots.size)
        rows = np.flatnonzero(interior)
        upper = right[interior]
        lower = upper - 1
        fraction = (grid[interior] - knots[lower]) / (knots[upper] - knots[lower])
        result[rows, lower] = 1.0 - fraction
        result[rows, upper] = fraction
        result = np.ascontiguousarray(result)
        result.flags.writeable = False
        return result

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]:
        result = np.ascontiguousarray(self.basis(x) @ self.parameters)
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> PointBackground:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.values),) or not np.isfinite(values).all():
            raise ValueError("replacement point values have an invalid shape")
        return replace(self, values=tuple(map(float, values)))

    def with_parameters(self, parameters: ArrayLike) -> PointBackground:
        return self.replace_coefficients(parameters)


@dataclass(frozen=True, slots=True)
class AmorphousPeak:
    """One normalized Gaussian broad component in degrees `2theta`."""

    area: float
    center_deg: float
    fwhm_deg: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.area, self.center_deg, self.fwhm_deg)).all():
            raise ValueError("amorphous peak parameters must be finite")
        if self.area < 0.0 or self.fwhm_deg <= 0.0:
            raise ValueError("amorphous peak area must be non-negative and FWHM positive")


@dataclass(frozen=True, slots=True)
class AmorphousBackground:
    """Sum of broad area-normalized Gaussian amorphous components."""

    background_id: str
    peaks: tuple[AmorphousPeak, ...]

    def __post_init__(self) -> None:
        _validate_id(self.background_id)
        peaks = tuple(self.peaks)
        if not peaks or any(not isinstance(peak, AmorphousPeak) for peak in peaks):
            raise TypeError("peaks must be a non-empty tuple of AmorphousPeak objects")
        object.__setattr__(self, "peaks", peaks)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return tuple(
            value for peak in self.peaks for value in (peak.area, peak.center_deg, peak.fwhm_deg)
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"peak_{index}.{name}"
            for index in range(len(self.peaks))
            for name in ("area", "center_deg", "fwhm_deg")
        )

    @property
    def parameters(self) -> NDArray[np.float64]:
        return _parameters(self.coefficients)

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            bound
            for _peak in self.peaks
            for bound in (
                (0.0, np.inf),
                (-np.inf, np.inf),
                (np.finfo(np.float64).tiny, np.inf),
            )
        )

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        grid = _grid(x)
        columns = []
        factor = 4.0 * np.log(2.0)
        normalization = np.sqrt(factor / np.pi)
        for peak in self.peaks:
            delta = grid - peak.center_deg
            gaussian = (
                normalization / peak.fwhm_deg * np.exp(-factor * (delta / peak.fwhm_deg) ** 2)
            )
            value = peak.area * gaussian
            columns.extend(
                (
                    gaussian,
                    value * 2.0 * factor * delta / peak.fwhm_deg**2,
                    value * (-1.0 / peak.fwhm_deg + 2.0 * factor * delta**2 / peak.fwhm_deg**3),
                )
            )
        result = np.ascontiguousarray(np.column_stack(columns))
        result.flags.writeable = False
        return result

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]:
        grid = _grid(x)
        result = np.zeros(grid.size, dtype=np.float64)
        factor = 4.0 * np.log(2.0)
        normalization = np.sqrt(factor / np.pi)
        for peak in self.peaks:
            result += (
                peak.area
                * normalization
                / peak.fwhm_deg
                * np.exp(-factor * ((grid - peak.center_deg) / peak.fwhm_deg) ** 2)
            )
        result = np.ascontiguousarray(result)
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> AmorphousBackground:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (3 * len(self.peaks),) or not np.isfinite(values).all():
            raise ValueError("replacement amorphous parameters have an invalid shape")
        peaks = tuple(
            AmorphousPeak(*map(float, values[index : index + 3]))
            for index in range(0, values.size, 3)
        )
        return replace(self, peaks=peaks)

    def with_parameters(self, parameters: ArrayLike) -> AmorphousBackground:
        return self.replace_coefficients(parameters)


@dataclass(frozen=True, slots=True)
class CompositeBackground:
    """Ordered additive composition of analytical background models."""

    background_id: str
    components: tuple[DifferentiableBackground, ...]

    def __post_init__(self) -> None:
        _validate_id(self.background_id)
        components = tuple(self.components)
        if not components or any(
            not isinstance(item, DifferentiableBackground) for item in components
        ):
            raise TypeError("components must be a non-empty tuple of differentiable backgrounds")
        if len({item.background_id for item in components}) != len(components):
            raise ValueError("composite background component IDs must be unique")
        object.__setattr__(self, "components", components)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return tuple(value for item in self.components for value in item.coefficients)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"{item.background_id}.{name}"
            for item in self.components
            for name in item.parameter_names
        )

    @property
    def parameters(self) -> NDArray[np.float64]:
        return _parameters(self.coefficients)

    @property
    def parameter_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(bound for item in self.components for bound in item.parameter_bounds)

    def basis(self, x: ArrayLike) -> NDArray[np.float64]:
        result = np.ascontiguousarray(np.column_stack([item.basis(x) for item in self.components]))
        result.flags.writeable = False
        return result

    def calculate(self, x: ArrayLike) -> NDArray[np.float64]:
        result = np.ascontiguousarray(
            sum((item.calculate(x) for item in self.components), np.zeros(_grid(x).size))
        )
        result.flags.writeable = False
        return result

    def replace_coefficients(self, coefficients: ArrayLike) -> CompositeBackground:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.coefficients),) or not np.isfinite(values).all():
            raise ValueError("replacement composite parameters have an invalid shape")
        components = []
        offset = 0
        for component in self.components:
            end = offset + len(component.coefficients)
            components.append(component.replace_coefficients(values[offset:end]))
            offset = end
        return replace(self, components=tuple(components))

    def with_parameters(self, parameters: ArrayLike) -> CompositeBackground:
        return self.replace_coefficients(parameters)
