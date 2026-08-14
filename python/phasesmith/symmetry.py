"""Typed symmetry operations and prepared reciprocal-family generation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from .crystallography import UnitCell

DEFAULT_MAX_REFLECTION_CANDIDATES = 50_000_000


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _integer_matrix(values: ArrayLike, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.shape != (3, 3) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a 3 by 3 integer matrix")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    if not np.array_equal(raw, result):
        raise ValueError(f"{name} entries must fit signed 64-bit integers")
    _freeze(result)
    return result


def _determinant_3x3(matrix: NDArray[np.int64]) -> int:
    values = [[int(value) for value in row] for row in matrix]
    return (
        values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )


def _fraction(value: Fraction | int | str) -> Fraction:
    if isinstance(value, float):
        raise TypeError("symmetry translations must use Fraction, integer, or rational string")
    try:
        result = Fraction(value) % 1
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid exact symmetry translation {value!r}") from error
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SymmetryOperation:
    """Exact fractional operation ``x' = rotation @ x + translation``."""

    rotation: NDArray[np.int64]
    translation: tuple[Fraction, Fraction, Fraction]

    def __init__(
        self,
        rotation: ArrayLike,
        translation: tuple[Fraction | int | str, Fraction | int | str, Fraction | int | str]
        | list[Fraction | int | str],
    ) -> None:
        """Copy and validate one unimodular operation."""

        matrix = _integer_matrix(rotation, "rotation")
        if abs(_determinant_3x3(matrix)) != 1:
            raise ValueError("symmetry rotation determinant must be exactly +1 or -1")
        if len(translation) != 3:
            raise ValueError("symmetry translation must contain three exact fractions")
        fractions = tuple(_fraction(value) for value in translation)
        object.__setattr__(self, "rotation", matrix)
        object.__setattr__(self, "translation", fractions)

    @classmethod
    def identity(cls) -> SymmetryOperation:
        """Construct the exact identity operation."""

        return cls(np.eye(3, dtype=np.int64), (0, 0, 0))

    def apply_fractional(self, xyz: ArrayLike) -> NDArray[np.float64]:
        """Apply the operation and wrap one coordinate into ``[0, 1)``."""

        coordinate = np.asarray(xyz, dtype=np.float64)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ValueError("xyz must contain three finite fractional coordinates")
        translation = np.array([float(value) for value in self.translation])
        result = np.mod(self.rotation @ coordinate + translation, 1.0)
        _freeze(result)
        return result

    def __eq__(self, other: object) -> bool:
        """Compare exact operation values rather than NumPy object identity."""

        return (
            isinstance(other, SymmetryOperation)
            and np.array_equal(self.rotation, other.rotation)
            and self.translation == other.translation
        )

    def __hash__(self) -> int:
        """Hash the exact immutable operation values."""

        return hash((tuple(int(value) for value in self.rotation.flat), self.translation))


@dataclass(frozen=True, slots=True)
class MetricConstraints:
    """Exact equations for direct metric order ``g11,g22,g33,g23,g13,g12``."""

    equations: NDArray[np.int64]
    parameterization_basis: NDArray[np.int64]
    independent_parameter_count: int


@dataclass(frozen=True, slots=True)
class ExpandedSites:
    """Unique symmetry-expanded fractional sites and source indices."""

    fractional_xyz: NDArray[np.float64]
    source_site: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ReflectionFamilies:
    """Canonical identity and conventional display indices for reflection families."""

    reflection_ids: tuple[str, ...]
    canonical_hkl: NDArray[np.int64]
    conventional_hkl: NDArray[np.int64]
    multiplicity: NDArray[np.int64]


def _operation_arrays(
    operations: tuple[SymmetryOperation, ...] | list[SymmetryOperation],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    values = tuple(operations)
    if not values or any(not isinstance(operation, SymmetryOperation) for operation in values):
        raise ValueError("operations must contain at least one SymmetryOperation")
    rotations = np.ascontiguousarray(np.stack([operation.rotation for operation in values]))
    numerators = np.array(
        [[value.numerator for value in operation.translation] for operation in values],
        dtype=np.int64,
        order="C",
    )
    denominators = np.array(
        [[value.denominator for value in operation.translation] for operation in values],
        dtype=np.int64,
        order="C",
    )
    return rotations, numerators, denominators


def _hkl_array(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("hkl must be a two-dimensional integer array with three columns")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    if not np.array_equal(raw, result):
        raise ValueError("hkl entries must fit signed 64-bit integers")
    _freeze(result)
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SpaceGroup:
    """Validated, closed exact operation set with canonical operation order."""

    operations: tuple[SymmetryOperation, ...]
    crystal_system: str
    metric_constraints: MetricConstraints
    _native: object

    def __init__(self, operations: tuple[SymmetryOperation, ...] | list[SymmetryOperation]) -> None:
        """Validate closure natively and cache canonical group topology."""

        rotations, numerators, denominators = _operation_arrays(operations)
        native = _core._PreparedReflectionGenerator(
            rotations.reshape(-1),
            numerators.reshape(-1),
            denominators.reshape(-1),
            False,
            1,
        )
        topology = native.topology()
        native_rotations = np.asarray(topology[0], dtype=np.int64).reshape(-1, 3, 3)
        native_numerators = np.asarray(topology[1], dtype=np.int64).reshape(-1, 3)
        native_denominators = np.asarray(topology[2], dtype=np.int64).reshape(-1, 3)
        canonical = tuple(
            SymmetryOperation(
                rotation,
                tuple(
                    Fraction(int(numerator), int(denominator))
                    for numerator, denominator in zip(nums, dens, strict=True)
                ),
            )
            for rotation, nums, dens in zip(
                native_rotations, native_numerators, native_denominators, strict=True
            )
        )
        equations = np.asarray(topology[4], dtype=np.int64).reshape(-1, 6)
        parameterization_basis = np.asarray(topology[5], dtype=np.int64).reshape(-1, 6)
        _freeze(equations)
        _freeze(parameterization_basis)
        object.__setattr__(self, "operations", canonical)
        object.__setattr__(self, "crystal_system", str(topology[3]))
        object.__setattr__(
            self,
            "metric_constraints",
            MetricConstraints(equations, parameterization_basis, int(topology[6])),
        )
        object.__setattr__(self, "_native", native)

    @classmethod
    def p1(cls) -> SpaceGroup:
        """Construct P1 without relying on a space-group database."""

        return cls([SymmetryOperation.identity()])

    def expand_sites(self, fractional_xyz: ArrayLike, *, tolerance: float = 1e-10) -> ExpandedSites:
        """Expand asymmetric sites and deduplicate special positions."""

        xyz = np.array(fractional_xyz, dtype=np.float64, copy=True, order="C")
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("fractional_xyz must have finite shape (site_count, 3)")
        positions, source = self._native.expand_sites(xyz.reshape(-1), float(tolerance))
        positions = np.asarray(positions)
        source = np.asarray(source)
        _freeze(positions)
        _freeze(source)
        return ExpandedSites(positions, source)

    def systematic_absences(self, hkl: ArrayLike) -> NDArray[np.bool_]:
        """Test exact general-position systematic absences for a batch."""

        indices = _hkl_array(hkl)
        result = np.asarray(self._native.systematic_absences(indices.reshape(-1)))
        _freeze(result)
        return result

    def reflection_families(
        self, hkl: ArrayLike, *, merge_friedel: bool = True
    ) -> ReflectionFamilies:
        """Return stable canonical indices, display indices, and multiplicities."""

        indices = _hkl_array(hkl)
        native = _native_generator(self, merge_friedel, 1)
        ids, canonical, conventional, multiplicity = native.reflection_families(
            indices.reshape(-1)
        )
        canonical = np.asarray(canonical)
        conventional = np.asarray(conventional)
        multiplicity = np.asarray(multiplicity)
        _freeze(canonical)
        _freeze(conventional)
        _freeze(multiplicity)
        return ReflectionFamilies(tuple(ids), canonical, conventional, multiplicity)

    def __eq__(self, other: object) -> bool:
        """Compare canonical exact operation sets."""

        return isinstance(other, SpaceGroup) and self.operations == other.operations

    def __hash__(self) -> int:
        """Hash the canonical exact operation set."""

        return hash(self.operations)


@runtime_checkable
class ReflectionRange(Protocol):
    """Protocol for typed inclusive reflection-selection ranges."""

    def _native_range(self) -> tuple[str, NDArray[np.float64]]:
        """Return the internal range discriminator and contiguous parameters."""


@dataclass(frozen=True, slots=True)
class DSpacingRange:
    """Inclusive d-spacing range in ångströms."""

    min_angstrom: float
    max_angstrom: float

    def _native_range(self) -> tuple[str, NDArray[np.float64]]:
        return "d", np.array([self.min_angstrom, self.max_angstrom], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ScatteringVectorRange:
    """Inclusive ``Q = 2 pi / d`` range in inverse ångströms."""

    min_inverse_angstrom: float
    max_inverse_angstrom: float

    def _native_range(self) -> tuple[str, NDArray[np.float64]]:
        return "q", np.array(
            [self.min_inverse_angstrom, self.max_inverse_angstrom], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class CwTwoThetaRange:
    """Inclusive monochromatic constant-wavelength ``2 theta`` range."""

    min_deg: float
    max_deg: float
    wavelength_angstrom: float

    def _native_range(self) -> tuple[str, NDArray[np.float64]]:
        return "cw", np.array(
            [self.min_deg, self.max_deg, self.wavelength_angstrom], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class TofRange:
    """Inclusive TOF range with an explicit safe d-spacing search interval."""

    min_us: float
    max_us: float
    search_min_d_angstrom: float
    search_max_d_angstrom: float
    zero_us: float
    difc_us_per_angstrom: float
    difa_us_per_angstrom2: float = 0.0
    difb_us_angstrom: float = 0.0

    def _native_range(self) -> tuple[str, NDArray[np.float64]]:
        return "tof", np.array(
            [
                self.min_us,
                self.max_us,
                self.search_min_d_angstrom,
                self.search_max_d_angstrom,
                self.zero_us,
                self.difc_us_per_angstrom,
                self.difa_us_per_angstrom2,
                self.difb_us_angstrom,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class GeneratedReflectionBatch:
    """Generated families with stable calculation indices and conventional labels."""

    reflection_ids: tuple[str, ...]
    hkl: NDArray[np.int64]
    conventional_hkl: NDArray[np.int64]
    multiplicity: NDArray[np.int64]
    d_spacing_angstrom: NDArray[np.float64]
    reciprocal_length_inverse_angstrom: NDArray[np.float64]
    d_spacing_derivatives: NDArray[np.float64]


def _native_generator(space_group: SpaceGroup, merge_friedel: bool, max_candidates: int) -> object:
    rotations, numerators, denominators = _operation_arrays(space_group.operations)
    return _core._PreparedReflectionGenerator(
        rotations.reshape(-1),
        numerators.reshape(-1),
        denominators.reshape(-1),
        bool(merge_friedel),
        int(max_candidates),
    )


@dataclass(frozen=True, slots=True, init=False)
class PreparedReflectionGenerator:
    """Cached native group topology for repeated cell/range generation."""

    space_group: SpaceGroup
    merge_friedel: bool
    max_candidates: int
    _native: object

    def __init__(
        self,
        space_group: SpaceGroup,
        *,
        merge_friedel: bool = True,
        max_candidates: int = DEFAULT_MAX_REFLECTION_CANDIDATES,
    ) -> None:
        """Prepare exact group topology and an explicit enumeration safety limit."""

        if not isinstance(space_group, SpaceGroup):
            raise TypeError("space_group must be a SpaceGroup")
        candidate_limit = int(max_candidates)
        if candidate_limit <= 0:
            raise ValueError("max_candidates must be positive")
        native = _native_generator(space_group, merge_friedel, candidate_limit)
        object.__setattr__(self, "space_group", space_group)
        object.__setattr__(self, "merge_friedel", bool(merge_friedel))
        object.__setattr__(self, "max_candidates", candidate_limit)
        object.__setattr__(self, "_native", native)

    def generate(self, cell: UnitCell, range_: ReflectionRange) -> GeneratedReflectionBatch:
        """Generate unique allowed families for one cell and inclusive range."""

        if not isinstance(cell, UnitCell):
            raise TypeError("cell must be a UnitCell")
        if not isinstance(range_, ReflectionRange):
            raise TypeError("range_ must implement the ReflectionRange protocol")
        kind, parameters = range_._native_range()
        arrays = self._native.generate(*cell.as_tuple(), kind, parameters)
        result = GeneratedReflectionBatch(
            reflection_ids=tuple(arrays[0]),
            hkl=np.asarray(arrays[1]),
            conventional_hkl=np.asarray(arrays[2]),
            multiplicity=np.asarray(arrays[3]),
            d_spacing_angstrom=np.asarray(arrays[4]),
            reciprocal_length_inverse_angstrom=np.asarray(arrays[5]),
            d_spacing_derivatives=np.asarray(arrays[6]),
        )
        for array in (
            result.hkl,
            result.conventional_hkl,
            result.multiplicity,
            result.d_spacing_angstrom,
            result.reciprocal_length_inverse_angstrom,
            result.d_spacing_derivatives,
        ):
            _freeze(array)
        return result
