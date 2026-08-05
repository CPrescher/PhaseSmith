"""Independent NumPy reference for the crystallography foundation slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystallography import AtomSiteBatch, UnitCell, p1_parameter_names


@dataclass(frozen=True, slots=True)
class ReferenceCellGeometry:
    """NumPy-derived metric tensors, volume, and analytical derivatives."""

    direct_metric: NDArray[np.float64]
    reciprocal_metric: NDArray[np.float64]
    volume_angstrom3: float
    d_volume_d_cell_parameters: NDArray[np.float64]
    d_reciprocal_metric_d_cell_parameters: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceP1Result:
    """Independent P1 values and parameter-major derivatives."""

    f: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    d_f_d_parameters: NDArray[np.complex128]
    d_intensity_d_parameters: NDArray[np.float64]


def reference_cell_geometry(cell: UnitCell) -> ReferenceCellGeometry:
    """Build the direct metric and its derivatives from scalar definitions."""

    a, b, c, alpha_deg, beta_deg, gamma_deg = cell.as_tuple()
    alpha, beta, gamma = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    metric = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=np.float64,
    )
    reciprocal = np.linalg.inv(metric)
    volume = float(np.sqrt(np.linalg.det(metric)))
    d_metric = np.zeros((6, 3, 3), dtype=np.float64)
    d_metric[0] = [[2 * a, b * cg, c * cb], [b * cg, 0, 0], [c * cb, 0, 0]]
    d_metric[1] = [[0, a * cg, 0], [a * cg, 2 * b, c * ca], [0, c * ca, 0]]
    d_metric[2] = [[0, 0, a * cb], [0, 0, b * ca], [a * cb, b * ca, 2 * c]]
    radians_per_degree = np.pi / 180.0
    d_metric[3, 1, 2] = d_metric[3, 2, 1] = -b * c * np.sin(alpha) * radians_per_degree
    d_metric[4, 0, 2] = d_metric[4, 2, 0] = -a * c * np.sin(beta) * radians_per_degree
    d_metric[5, 0, 1] = d_metric[5, 1, 0] = -a * b * np.sin(gamma) * radians_per_degree
    d_reciprocal = np.array([-reciprocal @ derivative @ reciprocal for derivative in d_metric])
    d_volume = np.array(
        [0.5 * volume * np.trace(reciprocal @ derivative) for derivative in d_metric]
    )
    return ReferenceCellGeometry(metric, reciprocal, volume, d_volume, d_reciprocal)


def reference_p1_structure_factors(
    cell: UnitCell,
    hkl: ArrayLike,
    sites: AtomSiteBatch,
    scattering_amplitudes: ArrayLike,
    *,
    scale: float = 1.0,
) -> ReferenceP1Result:
    """Evaluate the P1 equations without calling the Rust extension."""

    indices = np.asarray(hkl, dtype=np.int64)
    amplitudes = np.asarray(scattering_amplitudes, dtype=np.complex128)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("hkl must have shape (reflection_count, 3)")
    if amplitudes.shape != (indices.shape[0], sites.site_count):
        raise ValueError("scattering amplitude shape mismatch")
    geometry = reference_cell_geometry(cell)
    h_float = indices.astype(np.float64)
    q_squared = np.einsum("ri,ij,rj->r", h_float, geometry.reciprocal_metric, h_float)
    d_q_squared = np.stack(
        [
            np.einsum("ri,ij,rj->r", h_float, derivative, h_float)
            for derivative in geometry.d_reciprocal_metric_d_cell_parameters
        ]
    )
    phase = np.exp(2j * np.pi * (h_float @ sites.fractional_xyz.T))
    displacement = np.exp(-2.0 * np.pi**2 * q_squared[:, None] * sites.u_iso_angstrom2[None, :])
    base = amplitudes * phase * displacement
    contributions = base * sites.occupancy[None, :]
    f = np.sum(contributions, axis=1)
    names = p1_parameter_names(sites)
    d_f = np.zeros((len(names), indices.shape[0]), dtype=np.complex128)
    for parameter in range(6):
        d_f[parameter] = np.sum(
            contributions
            * (-2.0 * np.pi**2 * d_q_squared[parameter, :, None] * sites.u_iso_angstrom2[None, :]),
            axis=1,
        )
    offset = 6
    for site in range(sites.site_count):
        for component in range(3):
            d_f[offset + 3 * site + component] = (
                2j * np.pi * h_float[:, component] * contributions[:, site]
            )
    offset += 3 * sites.site_count
    for site in range(sites.site_count):
        d_f[offset + site] = base[:, site]
    offset += sites.site_count
    for site in range(sites.site_count):
        d_f[offset + site] = -2.0 * np.pi**2 * q_squared * contributions[:, site]
    intensity = float(scale) * np.abs(f) ** 2
    d_intensity = 2.0 * float(scale) * np.real(np.conjugate(f)[None, :] * d_f)
    d_intensity[-1] = np.abs(f) ** 2
    return ReferenceP1Result(f, intensity, names, d_f, d_intensity)
