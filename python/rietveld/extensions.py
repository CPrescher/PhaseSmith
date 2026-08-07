"""Versioned, batch-oriented extension contracts for reflection physics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystallography import UnitCell
from .instrument import ConstantWavelengthInstrument
from .phase import ReflectionGeometryBatch

PHYSICS_PROVIDER_API_VERSION = 1
_PROVIDER_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


def _vector(values: ArrayLike, name: str, count: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _matrix(values: ArrayLike, name: str, rows: int, columns: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Stable identity and compatibility metadata for one provider."""

    provider_id: str
    provider_version: str
    api_version: int = PHYSICS_PROVIDER_API_VERSION

    def __post_init__(self) -> None:
        """Validate persistence-safe provider metadata."""

        if not isinstance(self.provider_id, str) or not _PROVIDER_ID.fullmatch(self.provider_id):
            raise ValueError("provider_id must be a lowercase dotted identifier")
        if (
            not isinstance(self.provider_version, str)
            or not self.provider_version
            or any(character.isspace() for character in self.provider_version)
        ):
            raise ValueError("provider_version must be a non-empty token")
        if self.api_version <= 0:
            raise ValueError("api_version must be positive")


@dataclass(frozen=True, slots=True)
class PhysicsContext:
    """Typed immutable inputs supplied once to a physics provider."""

    reflections: ReflectionGeometryBatch
    instrument: ConstantWavelengthInstrument
    unit_cell: UnitCell | None = None


@dataclass(frozen=True, slots=True, init=False)
class PhysicsContribution:
    """Contiguous reflection contributions and analytical derivative chains."""

    gaussian_variance_deg2: NDArray[np.float64]
    lorentzian_fwhm_deg: NDArray[np.float64]
    intensity_multiplier: NDArray[np.float64]
    d_gaussian_variance_d_position: NDArray[np.float64]
    d_lorentzian_fwhm_d_position: NDArray[np.float64]
    d_intensity_multiplier_d_position: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    d_gaussian_variance_d_parameters: NDArray[np.float64]
    d_lorentzian_fwhm_d_parameters: NDArray[np.float64]
    d_intensity_multiplier_d_parameters: NDArray[np.float64]

    def __init__(
        self,
        *,
        gaussian_variance_deg2: ArrayLike,
        lorentzian_fwhm_deg: ArrayLike,
        intensity_multiplier: ArrayLike,
        d_gaussian_variance_d_position: ArrayLike,
        d_lorentzian_fwhm_d_position: ArrayLike,
        d_intensity_multiplier_d_position: ArrayLike,
        parameter_names: tuple[str, ...] | list[str],
        d_gaussian_variance_d_parameters: ArrayLike,
        d_lorentzian_fwhm_d_parameters: ArrayLike,
        d_intensity_multiplier_d_parameters: ArrayLike,
    ) -> None:
        """Copy, validate, and freeze a provider result."""

        names = tuple(parameter_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("provider parameter names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("provider parameter names must be unique")
        gaussian_raw = np.asarray(gaussian_variance_deg2)
        if gaussian_raw.ndim != 1:
            raise ValueError("gaussian_variance_deg2 must be one-dimensional")
        count = int(gaussian_raw.size)
        fields = {
            "gaussian_variance_deg2": _vector(gaussian_raw, "gaussian_variance_deg2", count),
            "lorentzian_fwhm_deg": _vector(lorentzian_fwhm_deg, "lorentzian_fwhm_deg", count),
            "intensity_multiplier": _vector(intensity_multiplier, "intensity_multiplier", count),
            "d_gaussian_variance_d_position": _vector(
                d_gaussian_variance_d_position, "d_gaussian_variance_d_position", count
            ),
            "d_lorentzian_fwhm_d_position": _vector(
                d_lorentzian_fwhm_d_position, "d_lorentzian_fwhm_d_position", count
            ),
            "d_intensity_multiplier_d_position": _vector(
                d_intensity_multiplier_d_position,
                "d_intensity_multiplier_d_position",
                count,
            ),
            "d_gaussian_variance_d_parameters": _matrix(
                d_gaussian_variance_d_parameters,
                "d_gaussian_variance_d_parameters",
                len(names),
                count,
            ),
            "d_lorentzian_fwhm_d_parameters": _matrix(
                d_lorentzian_fwhm_d_parameters,
                "d_lorentzian_fwhm_d_parameters",
                len(names),
                count,
            ),
            "d_intensity_multiplier_d_parameters": _matrix(
                d_intensity_multiplier_d_parameters,
                "d_intensity_multiplier_d_parameters",
                len(names),
                count,
            ),
        }
        if np.any(fields["gaussian_variance_deg2"] < 0.0):
            raise ValueError("gaussian_variance_deg2 must be non-negative")
        if np.any(fields["lorentzian_fwhm_deg"] < 0.0):
            raise ValueError("lorentzian_fwhm_deg must be non-negative")
        if np.any(fields["intensity_multiplier"] < 0.0):
            raise ValueError("intensity_multiplier must be non-negative")
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "parameter_names", names)

    @property
    def reflection_count(self) -> int:
        """Return the number of represented reflections."""

        return int(self.gaussian_variance_deg2.size)

    @classmethod
    def neutral(cls, reflection_count: int) -> PhysicsContribution:
        """Create an exact no-broadening, unit-intensity contribution."""

        if reflection_count < 0:
            raise ValueError("reflection_count must be non-negative")
        zeros = np.zeros(reflection_count, dtype=np.float64)
        empty = np.zeros((0, reflection_count), dtype=np.float64)
        return cls(
            gaussian_variance_deg2=zeros,
            lorentzian_fwhm_deg=zeros,
            intensity_multiplier=np.ones(reflection_count, dtype=np.float64),
            d_gaussian_variance_d_position=zeros,
            d_lorentzian_fwhm_d_position=zeros,
            d_intensity_multiplier_d_position=zeros,
            parameter_names=(),
            d_gaussian_variance_d_parameters=empty,
            d_lorentzian_fwhm_d_parameters=empty,
            d_intensity_multiplier_d_parameters=empty,
        )


@runtime_checkable
class ReflectionPhysicsProvider(Protocol):
    """Protocol implemented by built-in and third-party batch providers."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return stable identity and API compatibility metadata."""

    def evaluate(self, context: PhysicsContext) -> PhysicsContribution:
        """Evaluate all reflections in one vectorized provider call."""


def evaluate_provider(
    provider: ReflectionPhysicsProvider, context: PhysicsContext
) -> PhysicsContribution:
    """Validate provider compatibility and its plain-array result."""

    descriptor = provider.descriptor
    if descriptor.api_version != PHYSICS_PROVIDER_API_VERSION:
        raise ValueError(
            f"provider API {descriptor.api_version} is incompatible with "
            f"API {PHYSICS_PROVIDER_API_VERSION}"
        )
    contribution = provider.evaluate(context)
    if not isinstance(contribution, PhysicsContribution):
        raise TypeError("provider.evaluate() must return PhysicsContribution")
    if contribution.reflection_count != context.reflections.reflection_count:
        raise ValueError("provider result reflection count does not match its context")
    return contribution


def compose_contributions(contributions: tuple[PhysicsContribution, ...]) -> PhysicsContribution:
    """Compose widths additively and intensity modifiers by the product rule."""

    if not contributions:
        raise ValueError("at least one contribution is required for composition")
    count = contributions[0].reflection_count
    if any(item.reflection_count != count for item in contributions):
        raise ValueError("all contributions must have the same reflection count")
    names = tuple(name for item in contributions for name in item.parameter_names)
    if len(names) != len(set(names)):
        raise ValueError("composed provider parameter names must be globally unique")
    gaussian = sum((item.gaussian_variance_deg2 for item in contributions), np.zeros(count))
    lorentzian = sum((item.lorentzian_fwhm_deg for item in contributions), np.zeros(count))
    d_gaussian_position = sum(
        (item.d_gaussian_variance_d_position for item in contributions), np.zeros(count)
    )
    d_lorentzian_position = sum(
        (item.d_lorentzian_fwhm_d_position for item in contributions), np.zeros(count)
    )
    prefix = [np.ones(count)]
    for item in contributions:
        prefix.append(prefix[-1] * item.intensity_multiplier)
    suffix = [np.ones(count) for _item in range(len(contributions) + 1)]
    for index in range(len(contributions) - 1, -1, -1):
        suffix[index] = suffix[index + 1] * contributions[index].intensity_multiplier
    multiplier = prefix[-1]
    d_multiplier_position = np.zeros(count)
    d_gaussian_parameters = []
    d_lorentzian_parameters = []
    d_multiplier_parameters = []
    for index, item in enumerate(contributions):
        other_multiplier = prefix[index] * suffix[index + 1]
        d_multiplier_position += other_multiplier * item.d_intensity_multiplier_d_position
        d_gaussian_parameters.append(item.d_gaussian_variance_d_parameters)
        d_lorentzian_parameters.append(item.d_lorentzian_fwhm_d_parameters)
        d_multiplier_parameters.append(
            item.d_intensity_multiplier_d_parameters * other_multiplier[None, :]
        )
    return PhysicsContribution(
        gaussian_variance_deg2=gaussian,
        lorentzian_fwhm_deg=lorentzian,
        intensity_multiplier=multiplier,
        d_gaussian_variance_d_position=d_gaussian_position,
        d_lorentzian_fwhm_d_position=d_lorentzian_position,
        d_intensity_multiplier_d_position=d_multiplier_position,
        parameter_names=names,
        d_gaussian_variance_d_parameters=np.concatenate(d_gaussian_parameters, axis=0),
        d_lorentzian_fwhm_d_parameters=np.concatenate(d_lorentzian_parameters, axis=0),
        d_intensity_multiplier_d_parameters=np.concatenate(d_multiplier_parameters, axis=0),
    )


@dataclass(frozen=True, slots=True)
class CompositePhysicsProvider:
    """Explicit ordered composition of independent physics providers."""

    providers: tuple[ReflectionPhysicsProvider, ...]
    descriptor: ClassVar[ProviderDescriptor] = ProviderDescriptor("rietveld.composite", "1")

    def __post_init__(self) -> None:
        """Require a non-empty immutable provider sequence."""

        object.__setattr__(self, "providers", tuple(self.providers))
        if not self.providers:
            raise ValueError("CompositePhysicsProvider requires at least one provider")

    def evaluate(self, context: PhysicsContext) -> PhysicsContribution:
        """Evaluate every child once, then compose their plain arrays."""

        return compose_contributions(
            tuple(evaluate_provider(provider, context) for provider in self.providers)
        )
