"""Symmetry-aware lattice parameters and guarded CW reflection domains."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..crystallography import UnitCell
from ..instrument import TofInstrument
from ..phase import ReflectionBatch
from ..symmetry import DSpacingRange, PreparedReflectionGenerator, SpaceGroup

_CELL_ATTRIBUTES = (
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
)
_LENGTH_NAMES = _CELL_ATTRIBUTES[:3]
_ANGLE_NAMES = _CELL_ATTRIBUTES[3:]


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _cell_values(cell: UnitCell) -> NDArray[np.float64]:
    return np.asarray(cell.as_tuple(), dtype=np.float64)


def _equal_metric_components(basis: NDArray[np.int64], left: int, right: int) -> bool:
    return bool(np.array_equal(basis[:, left], basis[:, right]))


def _single_equal_diagonal_pair(basis: NDArray[np.int64]) -> tuple[int, int, int]:
    pairs = [
        (left, right)
        for left in range(3)
        for right in range(left + 1, 3)
        if _equal_metric_components(basis, left, right)
    ]
    if len(pairs) != 1:
        raise NotImplementedError("the lattice setting has no unique equal-axis plane")
    plane = pairs[0]
    unique = next(axis for axis in range(3) if axis not in plane)
    return plane[0], plane[1], unique


@dataclass(frozen=True, slots=True, init=False)
class LatticeParameterization:
    """Independent physical lattice variables for one crystallographic setting.

    The mapping is setting-aware. For example, a monoclinic group with its
    two-fold direction along ``a`` exposes ``a, b, c, alpha`` rather than
    assuming the conventional unique-``b`` setting.
    """

    space_group: SpaceGroup
    crystal_system: str
    parameter_names: tuple[str, ...]
    reference_values: NDArray[np.float64]
    _kind: str
    _axes: tuple[int, ...]
    _fixed_cell_values: NDArray[np.float64]

    def __init__(self, space_group: SpaceGroup, cell: UnitCell) -> None:
        if not isinstance(space_group, SpaceGroup):
            raise TypeError("space_group must be a SpaceGroup")
        if not isinstance(cell, UnitCell):
            raise TypeError("cell must be a UnitCell")
        basis = space_group.metric_constraints.parameterization_basis
        system = space_group.crystal_system
        cell_values = _cell_values(cell)
        kind: str
        axes: tuple[int, ...]
        names: tuple[str, ...]
        values: NDArray[np.float64]

        if system == "triclinic":
            kind = "triclinic"
            axes = tuple(range(6))
            names = _CELL_ATTRIBUTES
            values = cell_values.copy()
        elif system == "monoclinic":
            free_angles = tuple(index for index in range(3) if np.any(basis[:, 3 + index]))
            if len(free_angles) != 1:
                raise NotImplementedError("monoclinic setting must expose exactly one free angle")
            angle = free_angles[0]
            kind = "monoclinic"
            axes = (0, 1, 2, 3 + angle)
            names = (*_LENGTH_NAMES, _ANGLE_NAMES[angle])
            values = cell_values[list(axes)].copy()
        elif system == "orthorhombic":
            kind = "orthorhombic"
            axes = (0, 1, 2)
            names = _LENGTH_NAMES
            values = cell_values[:3].copy()
        elif system in ("tetragonal", "hexagonal"):
            first, second, unique = _single_equal_diagonal_pair(basis)
            representatives = tuple(sorted((first, unique)))
            kind = system
            axes = (first, second, unique, *representatives)
            names = tuple(_LENGTH_NAMES[index] for index in representatives)
            values = cell_values[list(representatives)].copy()
        elif system == "trigonal":
            diagonals_equal = all(_equal_metric_components(basis, 0, index) for index in (1, 2))
            off_diagonals_equal = all(_equal_metric_components(basis, 3, index) for index in (4, 5))
            if diagonals_equal and off_diagonals_equal and np.any(basis[:, 3]):
                kind = "rhombohedral"
                axes = (0,)
                names = ("a_angstrom", "alpha_deg")
                values = np.array([cell_values[0], cell_values[3]])
            else:
                first, second, unique = _single_equal_diagonal_pair(basis)
                representatives = tuple(sorted((first, unique)))
                kind = "hexagonal"
                axes = (first, second, unique, *representatives)
                names = tuple(_LENGTH_NAMES[index] for index in representatives)
                values = cell_values[list(representatives)].copy()
        elif system == "cubic":
            kind = "cubic"
            axes = (0, 1, 2)
            names = ("a_angstrom",)
            values = np.array([cell_values[0]])
        else:  # pragma: no cover - native crystal-system enum is closed
            raise NotImplementedError(f"unsupported crystal system {system!r}")

        _freeze(values)
        _freeze(cell_values)
        object.__setattr__(self, "space_group", space_group)
        object.__setattr__(self, "crystal_system", system)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "reference_values", values)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_axes", axes)
        object.__setattr__(self, "_fixed_cell_values", cell_values)
        reconstructed = self.to_cell(values)
        if not np.allclose(reconstructed.as_tuple(), cell.as_tuple(), rtol=0.0, atol=2.0e-10):
            raise NotImplementedError(
                "the compatible unit cell uses a non-conventional lattice setting"
            )

    @property
    def parameter_count(self) -> int:
        """Return the number of symmetry-independent lattice variables."""

        return len(self.parameter_names)

    def values_from_cell(self, cell: UnitCell) -> NDArray[np.float64]:
        """Extract independent variables after checking the setting mapping."""

        if not isinstance(cell, UnitCell):
            raise TypeError("cell must be a UnitCell")
        values = _cell_values(cell)
        if self._kind == "triclinic":
            result = values
        elif self._kind == "monoclinic":
            result = values[list(self._axes)]
        elif self._kind in ("orthorhombic",):
            result = values[:3]
        elif self._kind in ("tetragonal", "hexagonal"):
            result = values[list(self._axes[3:])]
        elif self._kind == "rhombohedral":
            result = np.array([values[0], values[3]])
        else:
            result = np.array([values[0]])
        rebuilt = self.to_cell(result)
        if not np.allclose(rebuilt.as_tuple(), cell.as_tuple(), rtol=0.0, atol=2.0e-9):
            raise ValueError("cell is incompatible with this lattice parameterization")
        result = np.ascontiguousarray(result)
        _freeze(result)
        return result

    def to_cell(self, values: ArrayLike) -> UnitCell:
        """Expand independent variables into all six physical cell parameters."""

        independent = np.asarray(values, dtype=np.float64)
        if independent.shape != (self.parameter_count,) or not np.isfinite(independent).all():
            raise ValueError("lattice values must be one finite value per independent parameter")
        full = np.array(self._fixed_cell_values, copy=True)
        if self._kind == "triclinic":
            full[:] = independent
        elif self._kind == "monoclinic":
            full[list(self._axes)] = independent
        elif self._kind == "orthorhombic":
            full[:3] = independent
        elif self._kind in ("tetragonal", "hexagonal"):
            first, second, unique, *representatives = self._axes
            value_by_axis = dict(zip(representatives, independent, strict=True))
            plane_value = value_by_axis[first] if first in value_by_axis else value_by_axis[second]
            full[first] = plane_value
            full[second] = plane_value
            full[unique] = value_by_axis[unique]
        elif self._kind == "rhombohedral":
            full[:3] = independent[0]
            full[3:] = independent[1]
        else:
            full[:3] = independent[0]
        return UnitCell(*map(float, full))

    def cell_parameter_jacobian(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return analytical ``d(a,b,c,alpha,beta,gamma)/d(independent)``."""

        self.to_cell(values)  # central validation, including physical cell validity
        jacobian = np.zeros((6, self.parameter_count), dtype=np.float64)
        if self._kind == "triclinic":
            jacobian[:] = np.eye(6)
        elif self._kind == "monoclinic":
            for column, row in enumerate(self._axes):
                jacobian[row, column] = 1.0
        elif self._kind == "orthorhombic":
            jacobian[:3] = np.eye(3)
        elif self._kind in ("tetragonal", "hexagonal"):
            first, second, unique, *representatives = self._axes
            for column, representative in enumerate(representatives):
                if representative == unique:
                    jacobian[unique, column] = 1.0
                else:
                    jacobian[first, column] = 1.0
                    jacobian[second, column] = 1.0
        elif self._kind == "rhombohedral":
            jacobian[:3, 0] = 1.0
            jacobian[3:, 1] = 1.0
        else:
            jacobian[:3, 0] = 1.0
        _freeze(jacobian)
        return jacobian


@dataclass(frozen=True, slots=True, init=False)
class LatticeParameterBounds:
    """Finite bounds in one lattice parameterization's stable order."""

    parameter_names: tuple[str, ...]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __init__(
        self,
        parameterization: LatticeParameterization,
        lower: ArrayLike,
        upper: ArrayLike,
    ) -> None:
        if not isinstance(parameterization, LatticeParameterization):
            raise TypeError("parameterization must be a LatticeParameterization")
        low = np.array(lower, dtype=np.float64, copy=True, order="C")
        high = np.array(upper, dtype=np.float64, copy=True, order="C")
        shape = (parameterization.parameter_count,)
        if low.shape != shape or high.shape != shape or not np.isfinite(low).all():
            raise ValueError("lattice lower/upper bounds must be finite matching vectors")
        if not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("every lattice lower bound must be below its finite upper bound")
        if np.any(parameterization.reference_values < low) or np.any(
            parameterization.reference_values > high
        ):
            raise ValueError("reference lattice values must lie within their bounds")
        for corner in product(*zip(low, high, strict=True)):
            parameterization.to_cell(corner)
        _freeze(low)
        _freeze(high)
        object.__setattr__(self, "parameter_names", parameterization.parameter_names)
        object.__setattr__(self, "lower", low)
        object.__setattr__(self, "upper", high)

    @classmethod
    def around(
        cls,
        parameterization: LatticeParameterization,
        *,
        relative_length: float = 0.05,
        angle_delta_deg: float = 5.0,
    ) -> LatticeParameterBounds:
        """Construct explicit finite bounds around the reference cell."""

        if not np.isfinite(relative_length) or not 0.0 < relative_length < 1.0:
            raise ValueError("relative_length must lie in (0, 1)")
        if not np.isfinite(angle_delta_deg) or angle_delta_deg <= 0.0:
            raise ValueError("angle_delta_deg must be positive and finite")
        values = parameterization.reference_values
        lower = np.empty_like(values)
        upper = np.empty_like(values)
        for index, (name, value) in enumerate(
            zip(parameterization.parameter_names, values, strict=True)
        ):
            if name.endswith("_angstrom"):
                lower[index] = value * (1.0 - relative_length)
                upper[index] = value * (1.0 + relative_length)
            else:
                lower[index] = max(np.nextafter(0.0, 1.0), value - angle_delta_deg)
                upper[index] = min(np.nextafter(180.0, 0.0), value + angle_delta_deg)
        return cls(parameterization, lower, upper)

    def corner_values(self) -> tuple[NDArray[np.float64], ...]:
        """Return deterministic corners of the independent-parameter box."""

        corners = []
        for values in product(*zip(self.lower, self.upper, strict=True)):
            corner = np.asarray(values, dtype=np.float64)
            _freeze(corner)
            corners.append(corner)
        return tuple(corners)


@dataclass(frozen=True, slots=True)
class LatticeReflectionGeometry:
    """Reflection coordinates and derivatives in independent lattice order."""

    d_spacing_angstrom: NDArray[np.float64]
    coordinate: NDArray[np.float64]
    d_d_spacing_d_parameters: NDArray[np.float64]
    d_coordinate_d_parameters: NDArray[np.float64]
    parameter_names: tuple[str, ...]


def _lattice_spacing_geometry(
    parameterization: LatticeParameterization,
    cell: UnitCell,
    hkl: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = parameterization.values_from_cell(cell)
    spacing = cell.d_spacings(hkl)
    cell_chain = parameterization.cell_parameter_jacobian(values)
    d_spacing = np.ascontiguousarray(spacing.derivatives @ cell_chain)
    _freeze(d_spacing)
    return spacing.d_spacing_angstrom, d_spacing


def cw_lattice_geometry(
    parameterization: LatticeParameterization,
    cell: UnitCell,
    hkl: ArrayLike,
    wavelength_angstrom: float,
) -> LatticeReflectionGeometry:
    """Calculate CW two-theta and its analytical lattice derivative chain."""

    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.0:
        raise ValueError("wavelength_angstrom must be positive and finite")
    spacing, d_spacing = _lattice_spacing_geometry(parameterization, cell, hkl)
    argument = wavelength_angstrom / (2.0 * spacing)
    if np.any(argument >= 1.0):
        raise ValueError("a reflection lies outside the physical monochromatic Bragg domain")
    coordinate = np.ascontiguousarray(2.0 * np.degrees(np.arcsin(argument)))
    derivative_per_d = (
        -180.0
        / np.pi
        * wavelength_angstrom
        / (np.square(spacing) * np.sqrt(1.0 - np.square(argument)))
    )
    d_coordinate = np.ascontiguousarray(derivative_per_d[:, None] * d_spacing)
    _freeze(coordinate)
    _freeze(d_coordinate)
    return LatticeReflectionGeometry(
        spacing,
        coordinate,
        d_spacing,
        d_coordinate,
        parameterization.parameter_names,
    )


def tof_lattice_geometry(
    parameterization: LatticeParameterization,
    cell: UnitCell,
    hkl: ArrayLike,
    instrument: TofInstrument,
) -> LatticeReflectionGeometry:
    """Calculate TOF coordinates and their analytical lattice derivative chain."""

    if not isinstance(instrument, TofInstrument):
        raise TypeError("instrument must be a TofInstrument")
    spacing, d_spacing = _lattice_spacing_geometry(parameterization, cell, hkl)
    coordinate = np.ascontiguousarray(
        instrument.zero_us
        + instrument.difc_us_per_angstrom * spacing
        + instrument.difa_us_per_angstrom2 * np.square(spacing)
        + instrument.difb_us_angstrom / spacing
    )
    derivative_per_d = (
        instrument.difc_us_per_angstrom
        + 2.0 * instrument.difa_us_per_angstrom2 * spacing
        - instrument.difb_us_angstrom / np.square(spacing)
    )
    d_coordinate = np.ascontiguousarray(derivative_per_d[:, None] * d_spacing)
    _freeze(coordinate)
    _freeze(d_coordinate)
    return LatticeReflectionGeometry(
        spacing,
        coordinate,
        d_spacing,
        d_coordinate,
        parameterization.parameter_names,
    )


@dataclass(frozen=True, slots=True)
class GeneratedReflectionDomainResult:
    """One generated topology plus stable-ID transfer diagnostics."""

    reflections: ReflectionBatch
    visible: NDArray[np.bool_]
    guarded_d_min_angstrom: float
    guarded_d_max_angstrom: float
    added_reflection_ids: tuple[str, ...]
    removed_reflection_ids: tuple[str, ...]
    preserved_reflection_count: int


@dataclass(frozen=True, slots=True)
class CwLatticeReflectionDomain:
    """Bounded monochromatic reflection topology with stable intensity transfer."""

    space_group: SpaceGroup
    parameterization: LatticeParameterization
    bounds: LatticeParameterBounds
    wavelength_angstrom: float
    visible_two_theta_min_deg: float
    visible_two_theta_max_deg: float
    initial_intensity: float = 1.0
    merge_friedel: bool = True
    max_candidates: int = 50_000_000
    guard_scale: float = 1.001

    def __post_init__(self) -> None:
        if not isinstance(self.space_group, SpaceGroup):
            raise TypeError("space_group must be a SpaceGroup")
        if not isinstance(self.parameterization, LatticeParameterization):
            raise TypeError("parameterization must be a LatticeParameterization")
        if not isinstance(self.bounds, LatticeParameterBounds):
            raise TypeError("bounds must be LatticeParameterBounds")
        if self.bounds.parameter_names != self.parameterization.parameter_names:
            raise ValueError("lattice bounds must match the parameterization")
        if self.space_group != self.parameterization.space_group:
            raise ValueError("reflection-domain space group must match the parameterization")
        scalars = (
            self.wavelength_angstrom,
            self.visible_two_theta_min_deg,
            self.visible_two_theta_max_deg,
            self.initial_intensity,
            self.guard_scale,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("reflection-domain scalars must be finite")
        if self.wavelength_angstrom <= 0.0 or self.initial_intensity < 0.0:
            raise ValueError("wavelength must be positive and initial intensity non-negative")
        if not 0.0 < self.visible_two_theta_min_deg < self.visible_two_theta_max_deg < 180.0:
            raise ValueError("visible two-theta bounds must lie strictly inside (0, 180)")
        if self.guard_scale < 1.0:
            raise ValueError("guard_scale must be at least one")
        if not isinstance(self.merge_friedel, bool):
            raise TypeError("merge_friedel must be boolean")
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool):
            raise TypeError("max_candidates must be an integer")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

    def _guarded_d_range(self, reference_cell: UnitCell) -> tuple[float, float]:
        corner_cells = np.asarray(
            [
                self.parameterization.to_cell(values).as_tuple()
                for values in self.bounds.corner_values()
            ],
            dtype=np.float64,
        )
        physical_lower = np.min(corner_cells, axis=0)
        physical_upper = np.max(corner_cells, axis=0)
        cos_lower = np.cos(np.deg2rad(physical_upper[3:]))
        cos_upper = np.cos(np.deg2rad(physical_lower[3:]))
        product_lower = min(
            float(alpha * beta * gamma)
            for alpha, beta, gamma in product(*zip(cos_lower, cos_upper, strict=True))
        )
        maximum_squares = np.maximum(np.square(cos_lower), np.square(cos_upper))
        angular_determinant_lower = 1.0 + 2.0 * product_lower - float(np.sum(maximum_squares))
        if angular_determinant_lower <= 0.0:
            raise ValueError(
                "lattice angle bounds are too broad for a finite guarded reflection domain"
            )
        determinant_lower = float(np.prod(physical_lower[:3])) ** 2 * angular_determinant_lower
        trace_upper = float(np.sum(np.square(physical_upper[:3])))
        direct_eigenvalue_lower = 4.0 * determinant_lower / trace_upper**2
        reciprocal_eigenvalue_lower = 1.0 / trace_upper
        reciprocal_eigenvalue_upper = 1.0 / direct_eigenvalue_lower
        reference_eigenvalues = np.linalg.eigvalsh(reference_cell.geometry().reciprocal_metric)
        minimum_ratio = (
            np.sqrt(float(np.min(reference_eigenvalues)) / reciprocal_eigenvalue_upper)
            / self.guard_scale
        )
        maximum_ratio = (
            np.sqrt(float(np.max(reference_eigenvalues)) / reciprocal_eigenvalue_lower)
            * self.guard_scale
        )
        theta_min = np.deg2rad(self.visible_two_theta_min_deg / 2.0)
        theta_max = np.deg2rad(self.visible_two_theta_max_deg / 2.0)
        visible_d_max = self.wavelength_angstrom / (2.0 * np.sin(theta_min))
        visible_d_min = self.wavelength_angstrom / (2.0 * np.sin(theta_max))
        return visible_d_min / maximum_ratio, visible_d_max / minimum_ratio

    def generate(
        self,
        cell: UnitCell,
        previous: ReflectionBatch | None = None,
    ) -> GeneratedReflectionDomainResult:
        """Generate at an accepted cell and transfer intensities by family ID."""

        self.parameterization.values_from_cell(cell)
        d_min, d_max = self._guarded_d_range(cell)
        generated = PreparedReflectionGenerator(
            self.space_group,
            merge_friedel=self.merge_friedel,
            max_candidates=self.max_candidates,
        ).generate(cell, DSpacingRange(d_min, d_max))
        argument = self.wavelength_angstrom / (2.0 * generated.d_spacing_angstrom)
        physical = argument < 1.0
        if not np.any(physical):
            raise ValueError("no physical reflections lie in the guarded CW domain")
        ids = tuple(
            reflection_id
            for reflection_id, selected in zip(generated.reflection_ids, physical, strict=True)
            if selected
        )
        hkl = generated.hkl[physical]
        spacing = generated.d_spacing_angstrom[physical]
        position = 2.0 * np.degrees(np.arcsin(argument[physical]))
        previous_values = (
            {}
            if previous is None
            else dict(
                zip(
                    previous.reflection_ids,
                    previous.integrated_intensity,
                    strict=True,
                )
            )
        )
        intensities = np.asarray(
            [previous_values.get(reflection_id, self.initial_intensity) for reflection_id in ids],
            dtype=np.float64,
        )
        reflections = ReflectionBatch(ids, hkl, spacing, position, intensities)
        visible = (position >= self.visible_two_theta_min_deg) & (
            position <= self.visible_two_theta_max_deg
        )
        visible = np.ascontiguousarray(visible)
        _freeze(visible)
        old_ids = set(previous_values)
        new_ids = set(ids)
        return GeneratedReflectionDomainResult(
            reflections=reflections,
            visible=visible,
            guarded_d_min_angstrom=float(d_min),
            guarded_d_max_angstrom=float(d_max),
            added_reflection_ids=tuple(
                reflection_id for reflection_id in ids if reflection_id not in old_ids
            ),
            removed_reflection_ids=(
                ()
                if previous is None
                else tuple(
                    reflection_id
                    for reflection_id in previous.reflection_ids
                    if reflection_id not in new_ids
                )
            ),
            preserved_reflection_count=sum(reflection_id in old_ids for reflection_id in ids),
        )
