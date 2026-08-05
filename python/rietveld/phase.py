"""Typed phase and reflection geometry models.

This module intentionally contains no profile evaluation or refinement state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .extensions import ReflectionPhysicsProvider
    from .intensity_corrections import IntegratedIntensityCorrectionProvider
    from .scattering import ScatteringFactorProvider
    from .structure import CrystalStructure
    from .symmetry import GeneratedReflectionBatch


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


def _stable_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True, slots=True, init=False)
class ReflectionBatch:
    """Durably identified reflections owned by one phase."""

    reflection_ids: tuple[str, ...]
    geometry: ReflectionGeometryBatch

    def __init__(
        self,
        reflection_ids: tuple[str, ...] | list[str],
        hkl: ArrayLike,
        d_spacing_angstrom: ArrayLike,
        two_theta_deg: ArrayLike,
        integrated_intensity: ArrayLike,
    ) -> None:
        """Validate IDs and construct the immutable numerical geometry batch."""

        ids = tuple(_stable_id(value, "reflection_id") for value in reflection_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("reflection_ids must be unique within a phase")
        geometry = ReflectionGeometryBatch(
            hkl,
            d_spacing_angstrom,
            two_theta_deg,
            integrated_intensity,
        )
        if len(ids) != geometry.reflection_count:
            raise ValueError("reflection_ids must match the reflection count")
        object.__setattr__(self, "reflection_ids", ids)
        object.__setattr__(self, "geometry", geometry)

    @property
    def reflection_count(self) -> int:
        """Return the number of reflections."""

        return self.geometry.reflection_count

    @property
    def integrated_intensity(self) -> NDArray[np.float64]:
        """Return the base integrated-intensity array."""

        return self.geometry.base_integrated_intensity

    @property
    def hkl(self) -> NDArray[np.int64]:
        """Return Miller indices in reflection order."""

        return self.geometry.hkl

    @property
    def d_spacing_angstrom(self) -> NDArray[np.float64]:
        """Return d-spacings in ångströms."""

        return self.geometry.d_spacing_angstrom

    @property
    def two_theta_deg(self) -> NDArray[np.float64]:
        """Return reflection positions in degrees two-theta."""

        return self.geometry.two_theta_deg


@dataclass(frozen=True, slots=True)
class Phase:
    """One identified powder phase and its reflection-physics configuration."""

    phase_id: str
    name: str
    reflections: ReflectionBatch
    scale: float = 1.0
    physics: ReflectionPhysicsProvider | None = None

    def __post_init__(self) -> None:
        """Validate stable identity, scale, and typed reflection ownership."""

        _stable_id(self.phase_id, "phase_id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("phase name must be a non-empty string")
        if not isinstance(self.reflections, ReflectionBatch):
            raise TypeError("reflections must be a ReflectionBatch")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("phase scale must be non-negative and finite")


@dataclass(frozen=True, slots=True, init=False)
class StructuralReflectionBatch:
    """Stable reflection families whose geometry follows the structure cell."""

    reflection_ids: tuple[str, ...]
    hkl: NDArray[np.int64]
    multiplicity: NDArray[np.int64]

    def __init__(
        self,
        reflection_ids: tuple[str, ...] | list[str],
        hkl: ArrayLike,
        multiplicity: ArrayLike,
    ) -> None:
        """Copy canonical indices and positive powder multiplicities."""

        ids = tuple(_stable_id(value, "reflection_id") for value in reflection_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("reflection_ids must be unique within a phase")
        raw_hkl = np.asarray(hkl)
        if (
            raw_hkl.ndim != 2
            or raw_hkl.shape[1] != 3
            or not np.issubdtype(raw_hkl.dtype, np.integer)
        ):
            raise ValueError("hkl must have shape (reflection_count, 3) and integer dtype")
        indices = np.array(raw_hkl, dtype=np.int64, copy=True, order="C")
        if not np.array_equal(raw_hkl, indices):
            raise ValueError("hkl indices must fit in signed 64-bit integers")
        raw_multiplicity = np.asarray(multiplicity)
        if raw_multiplicity.ndim != 1 or not np.issubdtype(raw_multiplicity.dtype, np.integer):
            raise ValueError("multiplicity must be a one-dimensional integer array")
        multiplicities = np.array(raw_multiplicity, dtype=np.int64, copy=True, order="C")
        if not np.array_equal(raw_multiplicity, multiplicities) or np.any(multiplicities <= 0):
            raise ValueError("multiplicity values must be positive signed 64-bit integers")
        count = indices.shape[0]
        if count == 0:
            raise ValueError("at least one structural reflection is required")
        if len(ids) != count or multiplicities.shape != (count,):
            raise ValueError("reflection IDs, hkl, and multiplicity must have equal lengths")
        indices.flags.writeable = False
        multiplicities.flags.writeable = False
        object.__setattr__(self, "reflection_ids", ids)
        object.__setattr__(self, "hkl", indices)
        object.__setattr__(self, "multiplicity", multiplicities)

    @classmethod
    def from_generated(cls, reflections: GeneratedReflectionBatch) -> StructuralReflectionBatch:
        """Drop cached metric values while retaining stable generated families."""

        from .symmetry import GeneratedReflectionBatch

        if not isinstance(reflections, GeneratedReflectionBatch):
            raise TypeError("reflections must be a GeneratedReflectionBatch")
        return cls(reflections.reflection_ids, reflections.hkl, reflections.multiplicity)

    @property
    def reflection_count(self) -> int:
        """Return the number of reflection families."""

        return int(self.hkl.shape[0])


@dataclass(frozen=True, slots=True)
class RietveldPhase:
    """One structural phase with explicit scattering and correction models.

    Reflection geometry and integrated intensities are calculated from the
    structure for every evaluation. This model is separate from :class:`Phase`,
    whose intensities are independent variables for Le Bail extraction.
    """

    phase_id: str
    name: str
    structure: CrystalStructure
    reflections: StructuralReflectionBatch
    scattering: ScatteringFactorProvider
    intensity_correction: IntegratedIntensityCorrectionProvider
    scale: float = 1.0
    physics: ReflectionPhysicsProvider | None = None
    coordinate_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        """Validate stable identity and explicitly supplied physical models."""

        from .intensity_corrections import IntegratedIntensityCorrectionProvider
        from .scattering import ScatteringFactorProvider, ScatteringProviderDescriptor
        from .structure import CrystalStructure

        _stable_id(self.phase_id, "phase_id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("phase name must be a non-empty string")
        if not isinstance(self.structure, CrystalStructure):
            raise TypeError("structure must be a CrystalStructure")
        if not isinstance(self.reflections, StructuralReflectionBatch):
            raise TypeError("reflections must be a StructuralReflectionBatch")
        if not isinstance(self.scattering, ScatteringFactorProvider):
            raise TypeError("scattering must implement ScatteringFactorProvider")
        if not isinstance(self.scattering.descriptor, ScatteringProviderDescriptor):
            raise TypeError("scattering descriptor must be a ScatteringProviderDescriptor")
        if not isinstance(self.intensity_correction, IntegratedIntensityCorrectionProvider):
            raise TypeError(
                "intensity_correction must implement IntegratedIntensityCorrectionProvider"
            )
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("phase scale must be non-negative and finite")
        if not np.isfinite(self.coordinate_tolerance) or self.coordinate_tolerance <= 0.0:
            raise ValueError("coordinate_tolerance must be positive and finite")


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
