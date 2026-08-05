from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld.crystallography_reference import (
    reference_cell_geometry,
    reference_p1_structure_factors,
)


def triclinic_cell() -> rietveld.UnitCell:
    return rietveld.UnitCell(4.3, 5.1, 6.2, 78.0, 83.0, 71.0)


def two_sites() -> rietveld.AtomSiteBatch:
    return rietveld.AtomSiteBatch(
        ["si1", "o1"],
        ["Si", "O"],
        [[0.17, 0.29, 0.43], [0.61, 0.11, 0.37]],
        [0.8, 0.6],
        [0.01, 0.02],
    )


def reflection_case() -> tuple[np.ndarray, np.ndarray]:
    hkl = np.array([[1, 0, 1], [2, -1, 3], [-1, 2, 2]], dtype=np.int64)
    scattering = np.array(
        [[2.0 + 0.1j, 1.0 - 0.2j], [1.5 + 0.05j, 0.7 + 0.15j], [0.9 - 0.1j, 1.2 + 0.2j]],
        dtype=np.complex128,
    )
    return hkl, scattering


def test_unit_cell_geometry_and_d_spacings_match_independent_numpy() -> None:
    cell = triclinic_cell()
    native = cell.geometry()
    reference = reference_cell_geometry(cell)
    np.testing.assert_allclose(native.direct_metric, reference.direct_metric, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        native.reciprocal_metric, reference.reciprocal_metric, rtol=2e-15, atol=2e-17
    )
    assert native.volume_angstrom3 == pytest.approx(reference.volume_angstrom3, rel=2e-15)
    np.testing.assert_allclose(
        native.d_volume_d_cell_parameters,
        reference.d_volume_d_cell_parameters,
        rtol=2e-14,
        atol=2e-14,
    )
    hkl, _ = reflection_case()
    spacing = cell.d_spacings(hkl)
    q_squared = np.einsum(
        "ri,ij,rj->r", hkl.astype(float), reference.reciprocal_metric, hkl.astype(float)
    )
    np.testing.assert_allclose(spacing.d_spacing_angstrom, 1.0 / np.sqrt(q_squared), rtol=2e-15)
    assert spacing.derivatives.shape == (3, 6)


def test_p1_values_and_analytical_derivatives_match_independent_reference() -> None:
    cell = triclinic_cell()
    sites = two_sites()
    hkl, scattering = reflection_case()
    native = rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering, scale=1.3)
    reference = reference_p1_structure_factors(cell, hkl, sites, scattering, scale=1.3)
    assert native.parameter_names == reference.parameter_names
    np.testing.assert_allclose(native.f, reference.f, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(native.intensity, reference.intensity, rtol=3e-15, atol=2e-15)
    np.testing.assert_allclose(
        native.d_f_d_parameters, reference.d_f_d_parameters, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        native.d_intensity_d_parameters,
        reference.d_intensity_d_parameters,
        rtol=3e-14,
        atol=3e-14,
    )


def _perturbed_case(
    cell: rietveld.UnitCell,
    sites: rietveld.AtomSiteBatch,
    scale: float,
    parameter: int,
    delta: float,
) -> tuple[rietveld.UnitCell, rietveld.AtomSiteBatch, float]:
    if parameter < 6:
        names = (
            "a_angstrom",
            "b_angstrom",
            "c_angstrom",
            "alpha_deg",
            "beta_deg",
            "gamma_deg",
        )
        cell = replace(cell, **{names[parameter]: getattr(cell, names[parameter]) + delta})
        return cell, sites, scale
    site_count = sites.site_count
    coordinates = np.array(sites.fractional_xyz, copy=True)
    occupancy = np.array(sites.occupancy, copy=True)
    u_iso = np.array(sites.u_iso_angstrom2, copy=True)
    local = parameter - 6
    if local < 3 * site_count:
        coordinates.reshape(-1)[local] += delta
    elif local < 4 * site_count:
        occupancy[local - 3 * site_count] += delta
    elif local < 5 * site_count:
        u_iso[local - 4 * site_count] += delta
    else:
        scale += delta
    return (
        cell,
        rietveld.AtomSiteBatch(sites.site_ids, sites.species, coordinates, occupancy, u_iso),
        scale,
    )


def test_every_p1_parameter_derivative_matches_centered_difference() -> None:
    cell = triclinic_cell()
    sites = two_sites()
    hkl, scattering = reflection_case()
    scale = 1.3
    result = rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering, scale=scale)
    for parameter in range(len(result.parameter_names)):
        step = 1e-6 if parameter < 3 else 1e-7
        plus = _perturbed_case(cell, sites, scale, parameter, step)
        minus = _perturbed_case(cell, sites, scale, parameter, -step)
        plus_result = rietveld.calculate_p1_structure_factors(
            plus[0], hkl, plus[1], scattering, scale=plus[2]
        )
        minus_result = rietveld.calculate_p1_structure_factors(
            minus[0], hkl, minus[1], scattering, scale=minus[2]
        )
        finite_f = (plus_result.f - minus_result.f) / (2 * step)
        finite_intensity = (plus_result.intensity - minus_result.intensity) / (2 * step)
        np.testing.assert_allclose(
            result.d_f_d_parameters[parameter], finite_f, rtol=2e-7, atol=2e-8
        )
        np.testing.assert_allclose(
            result.d_intensity_d_parameters[parameter],
            finite_intensity,
            rtol=3e-7,
            atol=3e-8,
        )


def test_native_jvp_vjp_match_dense_and_are_adjoint() -> None:
    cell = triclinic_cell()
    sites = two_sites()
    hkl, scattering = reflection_case()
    dense = rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering, scale=1.3)
    tangent = np.linspace(1e-4, 2e-3, len(dense.parameter_names))
    weights = np.array([0.3, -0.7, 1.1])
    jvp = rietveld.p1_jacobian_vector_product(cell, hkl, sites, scattering, tangent, scale=1.3)
    vjp = rietveld.p1_intensity_transpose_jacobian_vector_product(
        cell, hkl, sites, scattering, weights, scale=1.3
    )
    np.testing.assert_allclose(jvp.f, dense.f, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jvp.d_f, tangent @ dense.d_f_d_parameters, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(
        jvp.d_intensity, tangent @ dense.d_intensity_d_parameters, rtol=3e-15, atol=3e-15
    )
    np.testing.assert_allclose(
        vjp.gradient, dense.d_intensity_d_parameters @ weights, rtol=3e-15, atol=3e-15
    )
    assert np.dot(jvp.d_intensity, weights) == pytest.approx(
        np.dot(tangent, vjp.gradient), rel=3e-15, abs=3e-15
    )


def test_integer_translation_and_common_origin_shift_preserve_intensity() -> None:
    cell = triclinic_cell()
    sites = two_sites()
    hkl, scattering = reflection_case()
    original = rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering)
    shift = np.array([1.2, -0.3, 2.4])
    shifted = rietveld.AtomSiteBatch(
        sites.site_ids,
        sites.species,
        sites.fractional_xyz + shift,
        sites.occupancy,
        sites.u_iso_angstrom2,
    )
    translated = rietveld.calculate_p1_structure_factors(cell, hkl, shifted, scattering)
    np.testing.assert_allclose(translated.intensity, original.intensity, rtol=2e-14, atol=2e-14)


def test_empty_sites_zero_occupancy_and_zero_scale_have_defined_derivatives() -> None:
    cell = triclinic_cell()
    hkl = np.array([[1, 0, 0], [0, 0, 0]])
    empty_sites = rietveld.AtomSiteBatch([], [], np.empty((0, 3)), [], [])
    empty = rietveld.calculate_p1_structure_factors(cell, hkl, empty_sites, np.empty((2, 0)))
    np.testing.assert_array_equal(empty.f, 0.0)
    np.testing.assert_array_equal(empty.intensity, 0.0)
    assert empty.d_intensity_d_parameters.shape == (7, 2)

    zero_occupancy_site = rietveld.AtomSiteBatch(["x"], ["X"], [[0.2, 0.3, 0.4]], [0.0], [0.01])
    zero_occupancy = rietveld.calculate_p1_structure_factors(
        cell, hkl[:1], zero_occupancy_site, [[2.0 + 0.5j]]
    )
    assert abs(zero_occupancy.d_f_d_parameters[9, 0]) > 0.0

    occupied_site = rietveld.AtomSiteBatch(["x"], ["X"], [[0.2, 0.3, 0.4]], [1.0], [0.01])
    zero_scale = rietveld.calculate_p1_structure_factors(
        cell, hkl[:1], occupied_site, [[2.0 + 0.5j]], scale=0.0
    )
    np.testing.assert_array_equal(zero_scale.intensity, 0.0)
    np.testing.assert_array_equal(zero_scale.d_intensity_d_parameters[:-1], 0.0)
    assert zero_scale.d_intensity_d_parameters[-1, 0] > 0.0


def test_invalid_crystallographic_inputs_and_dense_limit_are_explicit() -> None:
    with pytest.raises(ValueError, match="positive"):
        rietveld.UnitCell(0.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    cell = triclinic_cell()
    with pytest.raises(ValueError, match="finite d-spacing"):
        cell.d_spacings([[0, 0, 0]])
    sites = two_sites()
    hkl, scattering = reflection_case()
    with pytest.raises(ValueError, match="shape"):
        rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering[:, :1])
    with pytest.raises(MemoryError, match="dense P1"):
        rietveld.calculate_p1_structure_factors(
            cell, hkl, sites, scattering, max_dense_derivative_elements=1
        )


def test_public_crystallographic_arrays_are_read_only() -> None:
    cell = triclinic_cell()
    sites = two_sites()
    hkl, scattering = reflection_case()
    result = rietveld.calculate_p1_structure_factors(cell, hkl, sites, scattering)
    assert not sites.fractional_xyz.flags.writeable
    assert not cell.geometry().reciprocal_metric.flags.writeable
    assert not result.f.flags.writeable
    assert not result.d_intensity_d_parameters.flags.writeable
