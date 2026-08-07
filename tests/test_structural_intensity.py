from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import scattering_reference
from phasesmith.symmetry_reference import reference_expand_sites


def inversion_group() -> phasesmith.SpaceGroup:
    return phasesmith.SpaceGroup(
        [
            phasesmith.SymmetryOperation.identity(),
            phasesmith.SymmetryOperation(
                [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                (0, 0, 0),
            ),
        ]
    )


def structure() -> phasesmith.CrystalStructure:
    return phasesmith.CrystalStructure(
        "phase",
        "General symmetry",
        phasesmith.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0),
        inversion_group(),
        (
            phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
            phasesmith.AtomSite("o", "O1", "O", "O", (0.0, 0.0, 0.0), 0.55, 0.018),
        ),
    )


def reflections() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[1, 0, 1], [2, 1, 1], [1, 2, 3]]), np.array([2, 4, 2])


def reference_values(
    model: phasesmith.CrystalStructure,
    hkl: np.ndarray,
    multiplicity: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = model.cell.geometry()
    h = hkl.astype(float)
    q_squared = np.einsum("ri,ij,rj->r", h, geometry.reciprocal_metric, h)
    s = 0.5 * np.sqrt(q_squared)
    factors, _ = scattering_reference.xray_non_resonant(
        [site.type_symbol for site in model.sites], s
    )
    expanded, source = reference_expand_sites(
        model.space_group,
        np.asarray([site.fractional_xyz for site in model.sites]),
        tolerance=1e-10,
    )
    f = np.zeros(hkl.shape[0], dtype=np.complex128)
    for site_index, site in enumerate(model.sites):
        positions = expanded[source == site_index]
        symmetry_sum = np.exp(2j * np.pi * (h @ positions.T)).sum(axis=1)
        displacement = np.exp(-2.0 * np.pi**2 * site.u_iso_angstrom2 * q_squared)
        f += site.occupancy * factors[:, site_index] * displacement * symmetry_sum
    return f, scale * multiplicity * np.abs(f) ** 2


def test_general_symmetry_values_match_independent_numpy_reference() -> None:
    model = structure()
    hkl, multiplicity = reflections()
    scale = 1.4
    actual = phasesmith.calculate_structure_factors(
        model,
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        scale=scale,
    )
    expected_f, expected_intensity = reference_values(model, hkl, multiplicity, scale)
    np.testing.assert_allclose(actual.f, expected_f, rtol=3e-15, atol=3e-14)
    np.testing.assert_allclose(
        actual.integrated_intensity, expected_intensity, rtol=5e-15, atol=5e-13
    )
    np.testing.assert_allclose(actual.f_squared, np.abs(actual.f) ** 2, rtol=2e-15)
    assert actual.correction_model_id == "neutral"
    np.testing.assert_array_equal(actual.correction, 1.0)
    assert not actual.f.flags.writeable
    assert not actual.d_integrated_intensity_d_parameters.flags.writeable


def test_values_only_api_matches_dense_result_without_derivative_outputs() -> None:
    model = structure()
    hkl, multiplicity = reflections()
    values = phasesmith.calculate_structure_factor_values(
        model,
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        scale=1.4,
    )
    dense = phasesmith.calculate_structure_factors(
        model,
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        scale=1.4,
    )

    np.testing.assert_array_equal(values.f, dense.f)
    np.testing.assert_array_equal(values.f_squared, dense.f_squared)
    np.testing.assert_array_equal(values.integrated_intensity, dense.integrated_intensity)
    np.testing.assert_array_equal(
        values.q_squared_inverse_angstrom2, dense.q_squared_inverse_angstrom2
    )
    np.testing.assert_array_equal(values.s_inverse_angstrom, dense.s_inverse_angstrom)
    np.testing.assert_array_equal(values.correction, dense.correction)
    assert values.correction_model_id == dense.correction_model_id
    assert not values.integrated_intensity.flags.writeable


def test_general_symmetry_selected_derivatives_match_centered_differences() -> None:
    baseline = structure()
    hkl, multiplicity = reflections()
    scale = 1.4

    def evaluate(model: phasesmith.CrystalStructure, phase_scale: float):
        return phasesmith.calculate_structure_factors(
            model,
            hkl,
            multiplicity,
            phasesmith.XrayNonResonant(),
            scale=phase_scale,
        )

    actual = evaluate(baseline, scale)
    selected = (0, 6, 12, 14, len(actual.parameter_names) - 1)
    for parameter in selected:
        step = 1e-6
        plus = baseline
        minus = baseline
        plus_scale = scale
        minus_scale = scale
        if parameter == 0:
            plus = replace(baseline, cell=replace(baseline.cell, a_angstrom=4.7 + step))
            minus = replace(baseline, cell=replace(baseline.cell, a_angstrom=4.7 - step))
        elif parameter == 6:
            plus_site = replace(
                baseline.sites[0],
                fractional_xyz=(baseline.sites[0].fractional_xyz[0] + step, 0.23, 0.31),
            )
            minus_site = replace(
                baseline.sites[0],
                fractional_xyz=(baseline.sites[0].fractional_xyz[0] - step, 0.23, 0.31),
            )
            plus = replace(baseline, sites=(plus_site, baseline.sites[1]))
            minus = replace(baseline, sites=(minus_site, baseline.sites[1]))
        elif parameter == 12:
            plus = replace(
                baseline,
                sites=(replace(baseline.sites[0], occupancy=0.82 + step), baseline.sites[1]),
            )
            minus = replace(
                baseline,
                sites=(replace(baseline.sites[0], occupancy=0.82 - step), baseline.sites[1]),
            )
        elif parameter == 14:
            plus = replace(
                baseline,
                sites=(
                    replace(baseline.sites[0], u_iso_angstrom2=0.012 + step),
                    baseline.sites[1],
                ),
            )
            minus = replace(
                baseline,
                sites=(
                    replace(baseline.sites[0], u_iso_angstrom2=0.012 - step),
                    baseline.sites[1],
                ),
            )
        else:
            plus_scale += step
            minus_scale -= step
        plus_result = evaluate(plus, plus_scale)
        minus_result = evaluate(minus, minus_scale)
        finite_f = (plus_result.f - minus_result.f) / (2.0 * step)
        finite_intensity = (
            plus_result.integrated_intensity - minus_result.integrated_intensity
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual.d_f_d_parameters[parameter], finite_f, rtol=3e-7, atol=3e-7
        )
        np.testing.assert_allclose(
            actual.d_integrated_intensity_d_parameters[parameter],
            finite_intensity,
            rtol=4e-7,
            atol=4e-6,
        )


def test_lp_and_custom_correction_contracts_are_explicit_and_vectorized() -> None:
    model = structure()
    hkl, multiplicity = reflections()
    wavelength = 1.5406
    lp = phasesmith.BraggBrentanoUnpolarizedLp(wavelength)
    actual = phasesmith.calculate_structure_factors(
        model,
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        correction=lp,
    )
    theta = np.arcsin(0.5 * wavelength * np.sqrt(actual.q_squared_inverse_angstrom2))
    expected = (1.0 + np.cos(2.0 * theta) ** 2) / (2.0 * np.sin(theta) ** 2 * np.cos(theta))
    np.testing.assert_allclose(actual.correction, expected, rtol=3e-15)
    step = 1.0e-6
    plus = phasesmith.calculate_structure_factors(
        replace(model, cell=replace(model.cell, a_angstrom=model.cell.a_angstrom + step)),
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        correction=lp,
    )
    minus = phasesmith.calculate_structure_factors(
        replace(model, cell=replace(model.cell, a_angstrom=model.cell.a_angstrom - step)),
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        correction=lp,
    )
    finite_difference = (plus.integrated_intensity - minus.integrated_intensity) / (2.0 * step)
    np.testing.assert_allclose(
        actual.d_integrated_intensity_d_parameters[0],
        finite_difference,
        rtol=4e-7,
        atol=5e-6,
    )

    class CustomCorrection:
        calls = 0

        def evaluate(self, q_squared):
            self.calls += 1
            return phasesmith.IntegratedIntensityCorrection(
                1.0 + 0.1 * q_squared,
                np.full(q_squared.shape, 0.1),
                "example.linear_q_squared",
            )

    custom = CustomCorrection()
    custom_result = phasesmith.calculate_structure_factors(
        model,
        hkl,
        multiplicity,
        phasesmith.XrayNonResonant(),
        correction=custom,
    )
    assert custom.calls == 1
    assert custom_result.correction_model_id == "example.linear_q_squared"

    class WrongShapeCorrection:
        def evaluate(self, q_squared):
            return phasesmith.IntegratedIntensityCorrection([1.0], [0.0], "example.wrong")

    with pytest.raises(ValueError, match="shape"):
        phasesmith.calculate_structure_factors(
            model,
            hkl,
            multiplicity,
            phasesmith.XrayNonResonant(),
            correction=WrongShapeCorrection(),
        )


def test_structural_boundaries_reject_invalid_multiplicity_anisotropy_and_dense_size() -> None:
    model = structure()
    hkl, multiplicity = reflections()
    with pytest.raises(ValueError, match="positive"):
        phasesmith.calculate_structure_factors(model, hkl, [2, 0, 2], phasesmith.XrayNonResonant())
    with pytest.raises(MemoryError, match="dense structure-factor"):
        phasesmith.calculate_structure_factors(
            model,
            hkl,
            multiplicity,
            phasesmith.XrayNonResonant(),
            max_dense_derivative_elements=1,
        )
    anisotropic = replace(
        model.sites[0],
        anisotropic_displacement=phasesmith.AnisotropicDisplacement(
            (0.01, 0.01, 0.01, 0.0, 0.0, 0.0), "U_cif"
        ),
    )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        phasesmith.calculate_structure_factors(
            replace(model, sites=(anisotropic, model.sites[1])),
            hkl,
            multiplicity,
            phasesmith.XrayNonResonant(),
        )
