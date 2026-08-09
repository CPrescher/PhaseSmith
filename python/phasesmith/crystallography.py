"""Typed crystallographic models and native P1 numerical interfaces.

This module owns no file parsing and performs no Python loop over atoms or
reflections in production calculation. CIF import is added through a separate
adapter in a later implementation unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

if TYPE_CHECKING:
    from .intensity_corrections import IntegratedIntensityCorrectionProvider
    from .scattering import ScatteringFactorProvider
    from .structure import CrystalStructure

CELL_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "cell.a",
    "cell.b",
    "cell.c",
    "cell.alpha",
    "cell.beta",
    "cell.gamma",
)
MAX_DENSE_DERIVATIVE_ELEMENTS: Final[int] = 50_000_000


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _float_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    _freeze(array)
    return array


def _hkl_array(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("hkl must have shape (reflection_count, 3)")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("hkl must contain integer indices")
    array = np.array(raw, dtype=np.int64, copy=True, order="C")
    if not np.array_equal(raw, array):
        raise ValueError("hkl indices must fit in signed 64-bit integers")
    _freeze(array)
    return array


def _stable_ids(values: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
        for value in ids
    ):
        raise ValueError(f"{name} values must be non-empty strings without surrounding whitespace")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} values must be unique")
    return ids


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """Validated direct and reciprocal geometry for one unit cell."""

    direct_basis: NDArray[np.float64]
    reciprocal_basis: NDArray[np.float64]
    direct_metric: NDArray[np.float64]
    reciprocal_metric: NDArray[np.float64]
    volume_angstrom3: float
    d_volume_d_cell_parameters: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DSpacingResult:
    """D-spacings and derivatives in ``CELL_PARAMETER_NAMES`` order."""

    d_spacing_angstrom: NDArray[np.float64]
    derivatives: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class UnitCell:
    """General direct unit cell in ångströms and degrees."""

    a_angstrom: float
    b_angstrom: float
    c_angstrom: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    def __post_init__(self) -> None:
        """Validate through the same native geometry constructor used in calculations."""

        _core.unit_cell_geometry(*self.as_tuple())

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """Return values in the native/public parameter order."""

        return (
            float(self.a_angstrom),
            float(self.b_angstrom),
            float(self.c_angstrom),
            float(self.alpha_deg),
            float(self.beta_deg),
            float(self.gamma_deg),
        )

    def geometry(self) -> CellGeometry:
        """Calculate direct/reciprocal matrices, volume, and volume derivatives."""

        values = _core.unit_cell_geometry(*self.as_tuple())
        result = CellGeometry(*values)
        for array in (
            result.direct_basis,
            result.reciprocal_basis,
            result.direct_metric,
            result.reciprocal_metric,
            result.d_volume_d_cell_parameters,
        ):
            _freeze(array)
        return result

    def d_spacings(self, hkl: ArrayLike) -> DSpacingResult:
        """Calculate d-spacings and six analytical cell derivatives."""

        indices = _hkl_array(hkl)
        spacing, derivatives = _core.unit_cell_d_spacings(
            np.ascontiguousarray(indices.reshape(-1)), *self.as_tuple()
        )
        _freeze(spacing)
        _freeze(derivatives)
        return DSpacingResult(spacing, derivatives)


@dataclass(frozen=True, slots=True, init=False)
class AtomSiteBatch:
    """Immutable asymmetric-site arrays with isotropic or CIF U displacement."""

    site_ids: tuple[str, ...]
    species: tuple[str, ...]
    fractional_xyz: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    u_iso_angstrom2: NDArray[np.float64]
    anisotropic_mask: NDArray[np.bool_]
    u_aniso_cif_angstrom2: NDArray[np.float64]

    def __init__(
        self,
        site_ids: tuple[str, ...] | list[str],
        species: tuple[str, ...] | list[str],
        fractional_xyz: ArrayLike,
        occupancy: ArrayLike,
        u_iso_angstrom2: ArrayLike,
        *,
        anisotropic_mask: ArrayLike | None = None,
        u_aniso_cif_angstrom2: ArrayLike | None = None,
    ) -> None:
        """Copy and validate one row per independent atom site."""

        ids = _stable_ids(site_ids, "site_ids")
        species_values = tuple(species)
        if len(species_values) != len(ids) or any(
            not isinstance(value, str) or not value.strip() for value in species_values
        ):
            raise ValueError("species must contain one non-empty string per site")
        xyz = np.array(fractional_xyz, dtype=np.float64, copy=True, order="C")
        if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
            raise ValueError("fractional_xyz must have finite shape (site_count, 3)")
        occupancies = _float_vector(occupancy, "occupancy")
        displacement = _float_vector(u_iso_angstrom2, "u_iso_angstrom2")
        if occupancies.shape != (len(ids),) or displacement.shape != (len(ids),):
            raise ValueError("occupancy and u_iso_angstrom2 must match the site count")
        if np.any(occupancies < 0.0) or np.any(displacement < 0.0):
            raise ValueError("occupancy and u_iso_angstrom2 must be non-negative")
        if anisotropic_mask is None:
            mask = np.zeros(len(ids), dtype=np.bool_)
        else:
            raw_mask = np.asarray(anisotropic_mask)
            if raw_mask.dtype != np.bool_ or raw_mask.shape != (len(ids),):
                raise ValueError("anisotropic_mask must be boolean with shape (site_count,)")
            mask = np.array(raw_mask, dtype=np.bool_, copy=True, order="C")
        if u_aniso_cif_angstrom2 is None:
            tensors = np.zeros((len(ids), 6), dtype=np.float64)
        else:
            tensors = np.array(u_aniso_cif_angstrom2, dtype=np.float64, copy=True, order="C")
            if tensors.shape != (len(ids), 6) or not np.isfinite(tensors).all():
                raise ValueError("u_aniso_cif_angstrom2 must have finite shape (site_count, 6)")
        _freeze(xyz)
        _freeze(mask)
        _freeze(tensors)
        object.__setattr__(self, "site_ids", ids)
        object.__setattr__(self, "species", species_values)
        object.__setattr__(self, "fractional_xyz", xyz)
        object.__setattr__(self, "occupancy", occupancies)
        object.__setattr__(self, "u_iso_angstrom2", displacement)
        object.__setattr__(self, "anisotropic_mask", mask)
        object.__setattr__(self, "u_aniso_cif_angstrom2", tensors)

    @property
    def site_count(self) -> int:
        """Return the number of independent sites."""

        return len(self.site_ids)


@dataclass(frozen=True, slots=True)
class P1StructureFactorResult:
    """P1 values and parameter-major dense analytical derivatives."""

    f: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    d_f_d_parameters: NDArray[np.complex128]
    d_intensity_d_parameters: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class P1JvpResult:
    """P1 values and one forward derivative product."""

    f: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    d_f: NDArray[np.complex128]
    d_intensity: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class P1VjpResult:
    """P1 values and one reverse product for intensity weights."""

    f: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    gradient: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StructureFactorResult:
    """General-symmetry values and bounded dense analytical derivatives."""

    f: NDArray[np.complex128]
    f_squared: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]
    q_squared_inverse_angstrom2: NDArray[np.float64]
    s_inverse_angstrom: NDArray[np.float64]
    correction: NDArray[np.float64]
    correction_model_id: str
    parameter_names: tuple[str, ...]
    d_f_d_parameters: NDArray[np.complex128]
    d_integrated_intensity_d_parameters: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StructureFactorValuesResult:
    """General-symmetry values without a materialized structural Jacobian."""

    f: NDArray[np.complex128]
    f_squared: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]
    q_squared_inverse_angstrom2: NDArray[np.float64]
    s_inverse_angstrom: NDArray[np.float64]
    correction: NDArray[np.float64]
    correction_model_id: str


def p1_parameter_names(sites: AtomSiteBatch) -> tuple[str, ...]:
    """Return the stable native P1 parameter order with durable site IDs."""

    coordinates = tuple(
        f"site.{site_id}.{component}" for site_id in sites.site_ids for component in ("x", "y", "z")
    )
    occupancy = tuple(f"site.{site_id}.occupancy" for site_id in sites.site_ids)
    displacement = tuple(f"site.{site_id}.u_iso" for site_id in sites.site_ids)
    return (*CELL_PARAMETER_NAMES, *coordinates, *occupancy, *displacement, "phase.scale")


def _scattering_array(
    scattering_amplitudes: ArrayLike,
    reflection_count: int,
    site_count: int,
) -> NDArray[np.complex128]:
    values = np.array(scattering_amplitudes, dtype=np.complex128, copy=True, order="C")
    if values.shape != (reflection_count, site_count):
        raise ValueError("scattering_amplitudes must have shape (reflection_count, site_count)")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("scattering_amplitudes must contain only finite values")
    _freeze(values)
    return values


def _native_p1_arguments(
    cell: UnitCell,
    hkl: ArrayLike,
    sites: AtomSiteBatch,
    scattering_amplitudes: ArrayLike,
    scale: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128], float]:
    if not isinstance(cell, UnitCell):
        raise TypeError("cell must be a UnitCell")
    if not isinstance(sites, AtomSiteBatch):
        raise TypeError("sites must be an AtomSiteBatch")
    indices = _hkl_array(hkl)
    amplitudes = _scattering_array(scattering_amplitudes, int(indices.shape[0]), sites.site_count)
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value < 0.0:
        raise ValueError("scale must be non-negative and finite")
    return indices, sites.fractional_xyz, amplitudes, scale_value


def _common_native_p1(
    cell: UnitCell,
    indices: NDArray[np.int64],
    sites: AtomSiteBatch,
    amplitudes: NDArray[np.complex128],
    scale: float,
) -> tuple[object, ...]:
    return (
        np.ascontiguousarray(indices.reshape(-1)),
        np.ascontiguousarray(sites.fractional_xyz.reshape(-1)),
        sites.occupancy,
        sites.u_iso_angstrom2,
        np.ascontiguousarray(amplitudes.real.reshape(-1)),
        np.ascontiguousarray(amplitudes.imag.reshape(-1)),
        *cell.as_tuple(),
        scale,
    )


def calculate_p1_structure_factors(
    cell: UnitCell,
    hkl: ArrayLike,
    sites: AtomSiteBatch,
    scattering_amplitudes: ArrayLike,
    *,
    scale: float = 1.0,
    max_dense_derivative_elements: int = MAX_DENSE_DERIVATIVE_ELEMENTS,
) -> P1StructureFactorResult:
    """Calculate complex P1 structure factors, intensities, and dense derivatives.

    Scattering amplitudes are caller supplied and held fixed with respect to
    cell parameters in this foundation slice. Later scattering models provide
    the additional ``df/ds`` chain.
    """

    indices, _, amplitudes, scale_value = _native_p1_arguments(
        cell, hkl, sites, scattering_amplitudes, scale
    )
    parameter_names = p1_parameter_names(sites)
    derivative_elements = 3 * len(parameter_names) * int(indices.shape[0])
    if max_dense_derivative_elements < 0 or derivative_elements > max_dense_derivative_elements:
        raise MemoryError(
            f"dense P1 result requires {derivative_elements} derivative elements; "
            f"limit is {max_dense_derivative_elements}"
        )
    arrays = _core.p1_structure_factors_dense(
        *_common_native_p1(cell, indices, sites, amplitudes, scale_value)
    )
    f = np.asarray(arrays[0]) + 1j * np.asarray(arrays[1])
    d_f = np.asarray(arrays[3]) + 1j * np.asarray(arrays[4])
    intensity = np.asarray(arrays[2])
    d_intensity = np.asarray(arrays[5])
    for array in (f, d_f, intensity, d_intensity):
        _freeze(array)
    return P1StructureFactorResult(
        f=f,
        intensity=intensity,
        parameter_names=parameter_names,
        d_f_d_parameters=d_f,
        d_intensity_d_parameters=d_intensity,
    )


def p1_jacobian_vector_product(
    cell: UnitCell,
    hkl: ArrayLike,
    sites: AtomSiteBatch,
    scattering_amplitudes: ArrayLike,
    tangent: ArrayLike,
    *,
    scale: float = 1.0,
) -> P1JvpResult:
    """Calculate a structural JVP without materializing the dense Jacobian."""

    indices, _, amplitudes, scale_value = _native_p1_arguments(
        cell, hkl, sites, scattering_amplitudes, scale
    )
    direction = _float_vector(tangent, "tangent")
    if direction.shape != (len(p1_parameter_names(sites)),):
        raise ValueError("tangent must match the P1 parameter count")
    arrays = _core.p1_structure_factors_jvp(
        *_common_native_p1(cell, indices, sites, amplitudes, scale_value), direction
    )
    f = np.asarray(arrays[0]) + 1j * np.asarray(arrays[1])
    d_f = np.asarray(arrays[3]) + 1j * np.asarray(arrays[4])
    intensity = np.asarray(arrays[2])
    d_intensity = np.asarray(arrays[5])
    for array in (f, d_f, intensity, d_intensity):
        _freeze(array)
    return P1JvpResult(f, intensity, d_f, d_intensity)


def p1_intensity_transpose_jacobian_vector_product(
    cell: UnitCell,
    hkl: ArrayLike,
    sites: AtomSiteBatch,
    scattering_amplitudes: ArrayLike,
    weights: ArrayLike,
    *,
    scale: float = 1.0,
) -> P1VjpResult:
    """Calculate ``J_intensity.T @ weights`` without a dense Jacobian."""

    indices, _, amplitudes, scale_value = _native_p1_arguments(
        cell, hkl, sites, scattering_amplitudes, scale
    )
    vector = _float_vector(weights, "weights")
    if vector.shape != (indices.shape[0],):
        raise ValueError("weights must match the reflection count")
    arrays = _core.p1_structure_factors_vjp(
        *_common_native_p1(cell, indices, sites, amplitudes, scale_value), vector
    )
    f = np.asarray(arrays[0]) + 1j * np.asarray(arrays[1])
    intensity = np.asarray(arrays[2])
    gradient = np.asarray(arrays[3])
    for array in (f, intensity, gradient):
        _freeze(array)
    return P1VjpResult(f, intensity, p1_parameter_names(sites), gradient)


def _prepare_structure_factor_inputs(
    structure: CrystalStructure,
    hkl: ArrayLike,
    multiplicity: ArrayLike,
    scattering: ScatteringFactorProvider,
    *,
    correction: IntegratedIntensityCorrectionProvider | None = None,
    scale: float = 1.0,
    coordinate_tolerance: float = 1.0e-10,
) -> tuple[tuple[object, ...], NDArray[np.float64], str, tuple[str, ...], int]:
    """Validate and prepare one native general-symmetry reflection batch."""

    from .intensity_corrections import (  # avoid a structure-model import cycle
        NeutralIntegratedIntensityCorrection,
        evaluate_intensity_correction,
    )
    from .scattering import (
        ScatteringContext,
        evaluate_scattering_provider,
        species_from_structure,
    )
    from .structure import CrystalStructure

    if not isinstance(structure, CrystalStructure):
        raise TypeError("structure must be a CrystalStructure")
    indices = _hkl_array(hkl)
    raw_multiplicity = np.asarray(multiplicity)
    if (
        raw_multiplicity.ndim != 1
        or raw_multiplicity.shape != (indices.shape[0],)
        or not np.issubdtype(raw_multiplicity.dtype, np.integer)
    ):
        raise ValueError("multiplicity must be a one-dimensional integer reflection vector")
    multiplicities = np.array(raw_multiplicity, dtype=np.int64, copy=True, order="C")
    if not np.array_equal(raw_multiplicity, multiplicities) or np.any(multiplicities <= 0):
        raise ValueError("multiplicity values must be positive signed 64-bit integers")
    sites = structure.to_site_batch()
    spacing = structure.cell.d_spacings(indices)
    q_squared = np.ascontiguousarray(1.0 / spacing.d_spacing_angstrom**2)
    s = np.ascontiguousarray(0.5 * np.sqrt(q_squared))
    scattering_batch = evaluate_scattering_provider(
        scattering,
        ScatteringContext(species_from_structure(structure), s),
    )
    selected_correction = (
        NeutralIntegratedIntensityCorrection() if correction is None else correction
    )
    correction_batch = evaluate_intensity_correction(selected_correction, q_squared)
    parameter_names = p1_parameter_names(sites)
    native_arguments = (
        np.ascontiguousarray(indices.reshape(-1)),
        multiplicities,
        np.ascontiguousarray(sites.fractional_xyz.reshape(-1)),
        sites.occupancy,
        sites.u_iso_angstrom2,
        sites.anisotropic_mask,
        np.ascontiguousarray(sites.u_aniso_cif_angstrom2.reshape(-1)),
        np.ascontiguousarray(scattering_batch.amplitudes.real.reshape(-1)),
        np.ascontiguousarray(scattering_batch.amplitudes.imag.reshape(-1)),
        np.ascontiguousarray(scattering_batch.d_amplitudes_d_s.real.reshape(-1)),
        np.ascontiguousarray(scattering_batch.d_amplitudes_d_s.imag.reshape(-1)),
        correction_batch.values,
        correction_batch.d_values_d_q_squared,
        *structure.cell.as_tuple(),
        float(scale),
        float(coordinate_tolerance),
    )
    return (
        native_arguments,
        correction_batch.values,
        correction_batch.model_id,
        parameter_names,
        int(indices.shape[0]),
    )


def calculate_structure_factor_values(
    structure: CrystalStructure,
    hkl: ArrayLike,
    multiplicity: ArrayLike,
    scattering: ScatteringFactorProvider,
    *,
    correction: IntegratedIntensityCorrectionProvider | None = None,
    scale: float = 1.0,
    coordinate_tolerance: float = 1.0e-10,
) -> StructureFactorValuesResult:
    """Calculate values and integrated intensities without a dense Jacobian.

    Providers are evaluated once for the complete reflection batch. Symmetry
    expansion and atom/reflection accumulation remain in the native kernel.
    """

    native, correction_values, correction_model_id, _, _ = _prepare_structure_factor_inputs(
        structure,
        hkl,
        multiplicity,
        scattering,
        correction=correction,
        scale=scale,
        coordinate_tolerance=coordinate_tolerance,
    )
    arrays = structure.space_group._native.structure_factor_values(*native)
    f = np.asarray(arrays[0]) + 1j * np.asarray(arrays[1])
    output = tuple(np.asarray(value) for value in arrays[2:6])
    for array in (f, *output):
        _freeze(array)
    return StructureFactorValuesResult(
        f=f,
        f_squared=output[0],
        integrated_intensity=output[1],
        q_squared_inverse_angstrom2=output[2],
        s_inverse_angstrom=output[3],
        correction=correction_values,
        correction_model_id=correction_model_id,
    )


def calculate_structure_factors(
    structure: CrystalStructure,
    hkl: ArrayLike,
    multiplicity: ArrayLike,
    scattering: ScatteringFactorProvider,
    *,
    correction: IntegratedIntensityCorrectionProvider | None = None,
    scale: float = 1.0,
    coordinate_tolerance: float = 1.0e-10,
    max_dense_derivative_elements: int = MAX_DENSE_DERIVATIVE_ELEMENTS,
) -> StructureFactorResult:
    """Calculate general-symmetry values and bounded analytical derivatives.

    A custom scattering provider and correction provider are each called once
    for the complete reflection batch. Symmetry expansion, atom summation, and
    all structural derivatives remain native.
    """

    (
        native,
        correction_values,
        correction_model_id,
        parameter_names,
        reflection_count,
    ) = _prepare_structure_factor_inputs(
        structure,
        hkl,
        multiplicity,
        scattering,
        correction=correction,
        scale=scale,
        coordinate_tolerance=coordinate_tolerance,
    )
    derivative_elements = 3 * len(parameter_names) * reflection_count
    if max_dense_derivative_elements < 0 or derivative_elements > max_dense_derivative_elements:
        raise MemoryError(
            f"dense structure-factor result requires {derivative_elements} derivative elements; "
            f"limit is {max_dense_derivative_elements}"
        )
    arrays = structure.space_group._native.structure_factor_dense(*native)
    f = np.asarray(arrays[0]) + 1j * np.asarray(arrays[1])
    d_f = np.asarray(arrays[6]) + 1j * np.asarray(arrays[7])
    output = tuple(np.asarray(value) for value in arrays[2:6])
    d_intensity = np.asarray(arrays[8])
    for array in (f, d_f, *output, d_intensity):
        _freeze(array)
    return StructureFactorResult(
        f=f,
        f_squared=output[0],
        integrated_intensity=output[1],
        q_squared_inverse_angstrom2=output[2],
        s_inverse_angstrom=output[3],
        correction=correction_values,
        correction_model_id=correction_model_id,
        parameter_names=parameter_names,
        d_f_d_parameters=d_f,
        d_integrated_intensity_d_parameters=d_intensity,
    )
