from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from numpy.typing import ArrayLike
from phasesmith import reference, scattering_reference
from phasesmith.crystallography_reference import (
    reference_cell_geometry,
    reference_structure_factor_values,
)


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
        "structure",
        "General symmetry",
        phasesmith.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0),
        inversion_group(),
        (
            phasesmith.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
            phasesmith.AtomSite("o", "O1", "O", "O", (0.0, 0.0, 0.0), 0.55, 0.018),
        ),
    )


def reflections() -> phasesmith.StructuralReflectionBatch:
    return phasesmith.StructuralReflectionBatch(
        ("1,0,1", "2,1,1", "1,2,3"),
        [[1, 0, 1], [2, 1, 1], [1, 2, 3]],
        [2, 4, 2],
    )


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        1.2e-4,
        1.5e-3,
        3.0e-3,
    )


def xray_phase(*, scale: float = 1.4) -> phasesmith.RietveldPhase:
    return phasesmith.RietveldPhase(
        "phase",
        "Structural phase",
        structure(),
        reflections(),
        phasesmith.XrayNonResonant(),
        phasesmith.NeutralIntegratedIntensityCorrection(),
        scale,
    )


def pattern() -> phasesmith.PowderPattern:
    return phasesmith.PowderPattern(
        np.linspace(10.0, 100.0, 9_001),
        background=np.linspace(0.1, 0.2, 9_001),
    )


def test_fused_structural_pattern_matches_separate_vectorized_layers() -> None:
    phase = xray_phase()
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path

    actual = prepared.calculate()
    structural = phasesmith.calculate_structure_factors(
        phase.structure,
        phase.reflections.hkl,
        phase.reflections.multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
        scale=phase.scale,
    )
    spacing = phase.structure.cell.d_spacings(phase.reflections.hkl).d_spacing_angstrom
    position = 2.0 * np.degrees(np.arcsin(instrument().wavelength_angstrom / (2.0 * spacing)))
    separate = phasesmith.accumulate_cw(
        pattern().x,
        position,
        structural.integrated_intensity,
        instrument(),
    )

    np.testing.assert_allclose(actual.profile_y, separate.y, rtol=2e-15, atol=2e-12)
    np.testing.assert_allclose(actual.reflections.f, structural.f, rtol=2e-15, atol=2e-14)
    np.testing.assert_allclose(
        actual.reflections.integrated_intensity,
        structural.integrated_intensity,
        rtol=3e-15,
        atol=3e-12,
    )
    np.testing.assert_allclose(actual.reflections.two_theta_deg, position, rtol=2e-15)
    np.testing.assert_array_equal(actual.y, actual.profile_y + actual.background)
    assert actual.reflections.reflection_ids == phase.reflections.reflection_ids
    assert not actual.y.flags.writeable


def test_native_fixed_dispersion_and_polarized_lp_match_vectorized_layers() -> None:
    phase = replace(
        xray_phase(),
        scattering=phasesmith.XrayFixedDispersion({"Si": 0.21 + 0.25j, "O": 0.05 + 0.03j}),
        intensity_correction=phasesmith.BraggBrentanoPolarizedLp(
            instrument().wavelength_angstrom,
            0.7,
        ),
    )
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    actual = prepared.calculate()
    expected = phasesmith.calculate_structure_factors(
        phase.structure,
        phase.reflections.hkl,
        phase.reflections.multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
        scale=phase.scale,
    )
    assert prepared.uses_native_fused_path
    assert prepared.leaf_count == 1
    np.testing.assert_allclose(actual.reflections.f, expected.f, rtol=3e-15, atol=3e-14)
    np.testing.assert_allclose(
        actual.reflections.integrated_intensity,
        expected.integrated_intensity,
        rtol=4e-15,
        atol=4e-12,
    )

    direction = np.zeros(
        len(phasesmith.p1_parameter_names(phase.structure.to_isotropic_site_batch()))
    )
    direction[[0, 6, 12, -1]] = [0.04, 0.015, -0.03, 0.05]
    forward = prepared.jvp(direction)
    step = 1.0e-6
    plus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, step)
    )
    minus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, -step)
    )
    np.testing.assert_allclose(
        forward.d_y,
        (plus.profile_y - minus.profile_y) / (2.0 * step),
        rtol=4e-6,
        atol=3e-7,
    )
    weights = np.sin(np.linspace(0.0, 3.0, pattern().x.size))
    reverse = prepared.vjp(weights)
    np.testing.assert_allclose(
        forward.d_y @ weights,
        direction @ reverse.gradient,
        rtol=6e-13,
        atol=3e-10,
    )


def test_fused_structural_pattern_matches_independent_numpy_equations() -> None:
    phase = xray_phase()
    actual = phasesmith.calculate_structural_pattern(
        pattern(),
        phasesmith.ConstantWavelengthExperiment.x_ray(instrument()),
        phase,
    )
    geometry = reference_cell_geometry(phase.structure.cell)
    h = phase.reflections.hkl.astype(np.float64)
    q_squared = np.einsum("ri,ij,rj->r", h, geometry.reciprocal_metric, h)
    s = 0.5 * np.sqrt(q_squared)
    scattering, _ = scattering_reference.xray_non_resonant(
        [site.type_symbol for site in phase.structure.sites],
        s,
    )
    structural = reference_structure_factor_values(
        phase.structure,
        phase.reflections.hkl,
        phase.reflections.multiplicity,
        scattering,
        np.ones(phase.reflections.reflection_count),
        scale=phase.scale,
    )
    positions = 2.0 * np.degrees(
        np.arcsin(0.5 * instrument().wavelength_angstrom * np.sqrt(q_squared))
    )
    expected_y, _, _ = reference.accumulate_cw(
        pattern().x,
        positions,
        structural.integrated_intensity,
        u_deg2=instrument().u_deg2,
        v_deg2=instrument().v_deg2,
        w_deg2=instrument().w_deg2,
        x_deg=instrument().x_deg,
        y_deg=instrument().y_deg,
    )
    np.testing.assert_allclose(actual.reflections.f, structural.f, rtol=3e-15, atol=3e-14)
    np.testing.assert_allclose(
        actual.reflections.integrated_intensity,
        structural.integrated_intensity,
        rtol=4e-15,
        atol=4e-12,
    )
    np.testing.assert_allclose(actual.profile_y, expected_y, rtol=2e-12, atol=5e-9)


def _perturb_phase(
    phase: phasesmith.RietveldPhase,
    direction: np.ndarray,
    step: float,
) -> phasesmith.RietveldPhase:
    cell_values = np.asarray(phase.structure.cell.as_tuple()) + step * direction[:6]
    sites = list(phase.structure.sites)
    site_count = len(sites)
    for site_index, site in enumerate(sites):
        xyz = (
            np.asarray(site.fractional_xyz)
            + step * direction[6 + 3 * site_index : 6 + 3 * site_index + 3]
        )
        occupancy = site.occupancy + step * direction[6 + 3 * site_count + site_index]
        u_iso = (site.u_iso_angstrom2 or 0.0) + step * direction[6 + 4 * site_count + site_index]
        sites[site_index] = replace(
            site,
            fractional_xyz=tuple(xyz),
            occupancy=occupancy,
            u_iso_angstrom2=u_iso,
        )
    model = replace(
        phase.structure,
        cell=phasesmith.UnitCell(*cell_values),
        sites=tuple(sites),
    )
    return replace(phase, structure=model, scale=phase.scale + step * direction[-1])


def test_structural_pattern_jvp_matches_finite_difference_and_vjp_adjoint() -> None:
    phase = xray_phase()
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    names = phasesmith.p1_parameter_names(phase.structure.to_isotropic_site_batch())
    direction = np.zeros(len(names))
    direction[[0, 6, 12, 14, len(names) - 1]] = [0.04, 0.015, -0.03, 0.002, 0.05]

    actual = prepared.jvp(direction)
    step = 1.0e-6
    plus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, step)
    )
    minus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, -step)
    )
    finite_difference = (plus.profile_y - minus.profile_y) / (2.0 * step)
    np.testing.assert_allclose(actual.d_y, finite_difference, rtol=3e-6, atol=2e-7)

    weights = np.sin(np.linspace(0.0, 3.0, pattern().x.size))
    reverse = prepared.vjp(weights)
    assert reverse.parameter_names == names
    np.testing.assert_allclose(
        actual.d_y @ weights,
        direction @ reverse.gradient,
        rtol=5e-13,
        atol=2e-10,
    )


def test_anisotropic_structural_pattern_native_products_match_finite_difference() -> None:
    base = xray_phase()
    aniso_site = replace(
        base.structure.sites[0],
        u_iso_angstrom2=None,
        anisotropic_displacement=phasesmith.AnisotropicDisplacement(
            (0.020, 0.013, 0.027, 0.002, 0.001, 0.003), "U_cif"
        ),
    )
    phase = replace(
        base,
        structure=replace(base.structure, sites=(aniso_site, base.structure.sites[1])),
    )
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path
    names = phasesmith.p1_parameter_names(phase.structure.to_site_batch())
    direction = np.zeros(len(names))
    direction[[0, 6, 12, len(names) - 1]] = [0.04, 0.015, -0.03, 0.05]
    actual = prepared.jvp(direction)
    step = 1e-6
    plus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, step)
    )
    minus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, -step)
    )
    finite = (plus.profile_y - minus.profile_y) / (2.0 * step)
    np.testing.assert_allclose(actual.d_y, finite, rtol=4e-6, atol=3e-7)
    weights = np.sin(np.linspace(0.0, 3.0, pattern().x.size))
    reverse = prepared.vjp(weights)
    np.testing.assert_allclose(
        actual.d_y @ weights,
        direction @ reverse.gradient,
        rtol=7e-13,
        atol=3e-10,
    )
    assert reverse.gradient[names.index("site.si.u_iso")] == 0.0


def test_monochromatic_neutron_structural_pattern_uses_native_path() -> None:
    phase = replace(
        xray_phase(),
        scattering=phasesmith.NeutronNuclear(),
        intensity_correction=phasesmith.NeutralIntegratedIntensityCorrection(),
    )
    experiment = phasesmith.ConstantWavelengthExperiment.neutron(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    result = prepared.calculate()
    assert prepared.uses_native_fused_path
    assert np.all(np.isfinite(result.profile_y))
    assert np.all(result.reflections.integrated_intensity >= 0.0)


def component_experiment() -> phasesmith.ConstantWavelengthExperiment:
    return phasesmith.ConstantWavelengthExperiment.x_ray_components(
        instrument(),
        phasesmith.WavelengthComponents.doublet(1.5406, 1.54439, 0.5),
    )


def component_phase(*, scale: float = 1.4) -> phasesmith.RietveldPhase:
    return replace(
        xray_phase(scale=scale),
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(1.5406),
    )


def test_structural_doublet_matches_sum_of_component_native_batches() -> None:
    phase = replace(
        component_phase(),
        scattering=phasesmith.XrayFixedDispersion({"Si": 0.21 + 0.25j, "O": 0.05 + 0.03j}),
        intensity_correction=phasesmith.BraggBrentanoPolarizedLp(1.5406, 0.7),
    )
    experiment = component_experiment()
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)

    actual = prepared.calculate()

    expected = np.zeros_like(pattern().x)
    weights = experiment.radiation.components.normalized_intensities
    for wavelength, weight in zip(
        experiment.radiation.components.wavelengths_angstrom, weights, strict=True
    ):
        selected_instrument = replace(instrument(), wavelength_angstrom=float(wavelength))
        selected_experiment = phasesmith.ConstantWavelengthExperiment.x_ray(selected_instrument)
        selected_phase = replace(
            phase,
            scale=phase.scale * float(weight),
            intensity_correction=phasesmith.BraggBrentanoPolarizedLp(float(wavelength), 0.7),
        )
        expected += phasesmith.calculate_structural_pattern(
            pattern(), selected_experiment, selected_phase
        ).profile_y

    assert prepared.uses_native_fused_path
    assert prepared.leaf_count == 1
    np.testing.assert_allclose(actual.profile_y, expected, rtol=3e-15, atol=3e-11)
    np.testing.assert_array_equal(actual.reflections.component_index, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(actual.reflections.base_reflection_index, [0, 1, 2, 0, 1, 2])
    assert actual.reflections.reflection_ids[3] == "1,0,1@component[1]"
    assert "wavelength_angstrom" not in actual.derivatives.global_parameter_names


def test_one_structural_component_matches_monochromatic_values_exactly() -> None:
    phase = component_phase()
    monochromatic = phasesmith.calculate_structural_pattern(
        pattern(), phasesmith.ConstantWavelengthExperiment.x_ray(instrument()), phase
    )
    components = phasesmith.calculate_structural_pattern(
        pattern(),
        phasesmith.ConstantWavelengthExperiment.x_ray_components(
            instrument(), phasesmith.WavelengthComponents.monochromatic(1.5406)
        ),
        phase,
    )

    np.testing.assert_array_equal(components.profile_y, monochromatic.profile_y)
    np.testing.assert_array_equal(
        components.reflections.integrated_intensity,
        monochromatic.reflections.integrated_intensity,
    )
    assert components.reflections.reflection_ids[0] == "1,0,1@component[0]"


def test_structural_component_jvp_finite_difference_and_vjp_are_adjoint() -> None:
    phase = component_phase()
    experiment = component_experiment()
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    names = phasesmith.p1_parameter_names(phase.structure.to_isotropic_site_batch())
    direction = np.zeros(len(names))
    direction[[0, 6, 12, 14, len(names) - 1]] = [0.04, 0.015, -0.03, 0.002, 0.05]

    product = prepared.jvp(direction)
    linearization = prepared.linearize()
    assert linearization.parameter_names == names
    np.testing.assert_array_equal(linearization.result.profile_y, product.result.profile_y)
    np.testing.assert_allclose(
        direction @ linearization.jacobian,
        product.d_y,
        rtol=4e-14,
        atol=3e-10,
    )
    step = 1.0e-6
    plus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, step)
    )
    minus = phasesmith.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, -step)
    )
    finite_difference = (plus.profile_y - minus.profile_y) / (2.0 * step)
    np.testing.assert_allclose(product.d_y, finite_difference, rtol=4e-6, atol=3e-7)

    weights = np.sin(np.linspace(0.0, 3.0, pattern().x.size))
    reverse = prepared.vjp(weights)
    np.testing.assert_allclose(
        linearization.jacobian @ weights,
        reverse.gradient,
        rtol=8e-13,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        product.d_y @ weights,
        direction @ reverse.gradient,
        rtol=8e-13,
        atol=3e-10,
    )


def test_structural_component_shared_instrument_derivative_matches_difference() -> None:
    phase = component_phase()
    experiment = component_experiment()
    result = phasesmith.calculate_structural_pattern(pattern(), experiment, phase)
    row = result.derivatives.global_parameter_names.index("u")
    step = 1.0e-8

    def evaluate(delta: float) -> np.ndarray:
        moved_instrument = replace(instrument(), u_deg2=instrument().u_deg2 + delta)
        moved = replace(experiment, instrument=moved_instrument)
        return phasesmith.calculate_structural_pattern(pattern(), moved, phase).profile_y

    finite_difference = (evaluate(step) - evaluate(-step)) / (2.0 * step)
    np.testing.assert_allclose(
        result.derivatives.global_jacobian[row],
        finite_difference,
        rtol=2e-5,
        atol=2e-5,
    )


def test_bragg_brentano_position_corrections_match_equation() -> None:
    geometry = phasesmith.BraggBrentanoGeometry(200.0, 0.35)
    experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.x_ray(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=0.075,
        geometry=geometry,
    )
    actual = phasesmith.calculate_structural_pattern(pattern(), experiment, xray_phase())
    spacing = xray_phase().structure.cell.d_spacings(reflections().hkl).d_spacing_angstrom
    beta = 2.0 * np.arcsin(instrument().wavelength_angstrom / (2.0 * spacing))
    expected = (
        np.degrees(beta)
        + 0.075
        - np.degrees(
            2.0
            * geometry.sample_displacement_mm
            / geometry.goniometer_radius_mm
            * np.cos(beta / 2.0)
        )
    )
    np.testing.assert_allclose(actual.reflections.two_theta_deg, expected, rtol=2e-15)


def test_debye_scherrer_position_corrections_match_equation() -> None:
    geometry = phasesmith.DebyeScherrerGeometry(650.0, 1578.8, 49.9)
    phase = replace(xray_phase(), scattering=phasesmith.NeutronNuclear())
    experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.neutron(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=-0.1,
        geometry=geometry,
    )
    actual = phasesmith.calculate_structural_pattern(pattern(), experiment, phase)
    spacing = phase.structure.cell.d_spacings(reflections().hkl).d_spacing_angstrom
    beta = 2.0 * np.arcsin(instrument().wavelength_angstrom / (2.0 * spacing))
    expected = (
        np.degrees(beta)
        - 0.1
        - 0.18
        / (np.pi * geometry.goniometer_radius_mm)
        * (
            geometry.displace_x_micrometre * np.cos(beta)
            + geometry.displace_y_micrometre * np.sin(beta)
        )
    )
    np.testing.assert_allclose(actual.reflections.two_theta_deg, expected, rtol=2e-15)


@pytest.mark.parametrize(
    "values",
    (
        (0.0, 0.0, 0.0),
        (650.0, np.nan, 0.0),
        (650.0, 0.0, np.inf),
    ),
)
def test_debye_scherrer_geometry_rejects_invalid_values(
    values: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match=r"positive and finite|displacements must be finite"):
        phasesmith.DebyeScherrerGeometry(*values)


@pytest.mark.parametrize(
    ("parameter", "step"),
    (
        ("wavelength_angstrom", 1.0e-6),
        ("zero_shift_deg", 1.0e-6),
        ("sample_displacement_mm", 1.0e-5),
    ),
)
def test_instrument_correction_derivatives_match_finite_difference(
    parameter: str,
    step: float,
) -> None:
    geometry = phasesmith.BraggBrentanoGeometry(240.0, 0.21)
    base_experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.x_ray(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=-0.017,
        geometry=geometry,
    )
    base_phase = replace(
        xray_phase(),
        intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(
            instrument().wavelength_angstrom
        ),
    )
    actual = phasesmith.calculate_structural_pattern(pattern(), base_experiment, base_phase)
    row = actual.derivatives.global_parameter_names.index(parameter)

    def evaluate(delta: float) -> np.ndarray:
        wavelength = instrument().wavelength_angstrom
        zero = base_experiment.zero_shift_deg
        displacement = geometry.sample_displacement_mm
        if parameter == "wavelength_angstrom":
            wavelength += delta
        elif parameter == "zero_shift_deg":
            zero += delta
        else:
            displacement += delta
        selected_instrument = replace(instrument(), wavelength_angstrom=wavelength)
        selected_experiment = phasesmith.ConstantWavelengthExperiment(
            phasesmith.MonochromaticRadiation.x_ray(wavelength),
            selected_instrument,
            zero_shift_deg=zero,
            geometry=phasesmith.BraggBrentanoGeometry(
                geometry.goniometer_radius_mm,
                displacement,
            ),
        )
        selected_phase = replace(
            base_phase,
            intensity_correction=phasesmith.BraggBrentanoUnpolarizedLp(wavelength),
        )
        return phasesmith.calculate_structural_pattern(
            pattern(), selected_experiment, selected_phase
        ).profile_y

    finite_difference = (evaluate(step) - evaluate(-step)) / (2.0 * step)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian[row],
        finite_difference,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("parameter", ("displace_x_micrometre", "displace_y_micrometre"))
def test_debye_scherrer_derivatives_match_finite_difference(parameter: str) -> None:
    geometry = phasesmith.DebyeScherrerGeometry(650.0, 1200.0, -80.0)
    phase = replace(xray_phase(), scattering=phasesmith.NeutronNuclear())
    base_experiment = phasesmith.ConstantWavelengthExperiment(
        phasesmith.MonochromaticRadiation.neutron(instrument().wavelength_angstrom),
        instrument(),
        zero_shift_deg=-0.1,
        geometry=geometry,
    )
    actual = phasesmith.calculate_structural_pattern(pattern(), base_experiment, phase)
    row = actual.derivatives.global_parameter_names.index(parameter)
    step = 1.0e-2

    def evaluate(delta: float) -> np.ndarray:
        moved = phasesmith.DebyeScherrerGeometry(
            geometry.goniometer_radius_mm,
            geometry.displace_x_micrometre
            + (delta if parameter.endswith("_x_micrometre") else 0.0),
            geometry.displace_y_micrometre
            + (delta if parameter.endswith("_y_micrometre") else 0.0),
        )
        experiment = replace(base_experiment, geometry=moved)
        return phasesmith.calculate_structural_pattern(pattern(), experiment, phase).profile_y

    finite_difference = (evaluate(step) - evaluate(-step)) / (2.0 * step)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian[row],
        finite_difference,
        rtol=3e-5,
        atol=2e-5,
    )


def test_builtin_size_broadening_stays_on_fused_structural_path() -> None:
    phase = replace(xray_phase(), physics=phasesmith.IsotropicSizeBroadening(70.0))
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path
    actual = prepared.calculate()

    geometry = phasesmith.ReflectionGeometryBatch(
        phase.reflections.hkl,
        actual.reflections.d_spacing_angstrom,
        actual.reflections.two_theta_deg,
        actual.reflections.integrated_intensity,
    )
    contribution = phase.physics.evaluate(phasesmith.PhysicsContext(geometry, instrument()))
    separate = phasesmith.accumulate_cw_contributions(
        pattern().x,
        geometry.two_theta_deg,
        geometry.base_integrated_intensity,
        instrument(),
        contribution,
    )
    np.testing.assert_allclose(actual.profile_y, separate.y, rtol=3e-15, atol=2e-11)
    assert actual.derivatives.global_parameter_names[-1] == ("isotropic_size.crystallite_size_nm")


def test_fcj_and_sample_physics_compose_on_structural_paths() -> None:
    axial = phasesmith.FcjGeometry(0.013, 0.009)
    phase = replace(xray_phase(), physics=phasesmith.IsotropicSizeBroadening(70.0))
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(
        instrument(),
        axial_geometry=axial,
    )
    prepared = phasesmith.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path
    actual = prepared.calculate()
    geometry = phasesmith.ReflectionGeometryBatch(
        phase.reflections.hkl,
        actual.reflections.d_spacing_angstrom,
        actual.reflections.two_theta_deg,
        actual.reflections.integrated_intensity,
    )
    contribution = phase.physics.evaluate(phasesmith.PhysicsContext(geometry, instrument()))
    separate = phasesmith.accumulate_cw_contributions(
        pattern().x,
        geometry.two_theta_deg,
        geometry.base_integrated_intensity,
        instrument(),
        contribution,
        geometry=axial,
    )
    np.testing.assert_allclose(actual.profile_y, separate.y, rtol=3e-15, atol=2e-11)
    assert actual.derivatives.global_parameter_names[-3:] == (
        "sample_over_radius",
        "detector_over_radius",
        "isotropic_size.crystallite_size_nm",
    )
    for name, step in (("sample_over_radius", 1.0e-7), ("detector_over_radius", 1.0e-7)):
        row = actual.derivatives.global_parameter_names.index(name)
        plus_geometry = replace(axial, **{name: getattr(axial, name) + step})
        minus_geometry = replace(axial, **{name: getattr(axial, name) - step})
        plus = phasesmith.calculate_structural_pattern(
            pattern(), replace(experiment, axial_geometry=plus_geometry), phase
        )
        minus = phasesmith.calculate_structural_pattern(
            pattern(), replace(experiment, axial_geometry=minus_geometry), phase
        )
        np.testing.assert_allclose(
            actual.derivatives.global_jacobian[row],
            (plus.profile_y - minus.profile_y) / (2.0 * step),
            rtol=2e-4,
            atol=2e-4,
        )

    names = phasesmith.p1_parameter_names(phase.structure.to_isotropic_site_batch())
    direction = np.zeros(len(names))
    direction[[0, 6, -1]] = (0.03, -0.02, 0.04)
    forward = prepared.jvp(direction)
    linearization = prepared.linearize()
    np.testing.assert_allclose(direction @ linearization.jacobian, forward.d_y, rtol=2e-13)
    structural_step = 1.0e-5
    plus_phase = _perturb_phase(phase, direction, structural_step)
    minus_phase = _perturb_phase(phase, direction, -structural_step)
    plus_y = phasesmith.calculate_structural_pattern(pattern(), experiment, plus_phase).profile_y
    minus_y = phasesmith.calculate_structural_pattern(pattern(), experiment, minus_phase).profile_y
    np.testing.assert_allclose(
        forward.d_y,
        (plus_y - minus_y) / (2.0 * structural_step),
        rtol=6e-6,
        atol=4e-7,
    )
    weights = np.sin(np.linspace(0.0, 3.0, pattern().x.size))
    reverse = prepared.vjp(weights)
    np.testing.assert_allclose(forward.d_y @ weights, direction @ reverse.gradient, rtol=8e-13)

    custom_phase = replace(
        phase,
        scattering=CountingScattering(),
        intensity_correction=CountingCorrection(),
    )
    fallback = phasesmith.PreparedStructuralPattern(pattern(), experiment, custom_phase)
    assert not fallback.uses_native_fused_path
    np.testing.assert_allclose(
        fallback.calculate().profile_y,
        actual.profile_y,
        rtol=3e-15,
        atol=2e-11,
    )


def test_probe_mismatch_is_rejected_before_calculation() -> None:
    experiment = phasesmith.ConstantWavelengthExperiment.neutron(instrument())
    with pytest.raises(ValueError, match="incompatible"):
        phasesmith.PreparedStructuralPattern(pattern(), experiment, xray_phase())


def test_structural_reflections_can_be_created_from_generated_families() -> None:
    model = structure()
    generated = phasesmith.PreparedReflectionGenerator(model.space_group).generate(
        model.cell,
        phasesmith.DSpacingRange(1.5, 4.0),
    )
    batch = phasesmith.StructuralReflectionBatch.from_generated(generated)
    np.testing.assert_array_equal(batch.hkl, generated.hkl)
    np.testing.assert_array_equal(batch.multiplicity, generated.multiplicity)
    assert batch.reflection_ids == generated.reflection_ids
    assert not batch.hkl.flags.writeable


class CountingScattering:
    descriptor = phasesmith.XRAY_NON_RESONANT_DESCRIPTOR

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: phasesmith.ScatteringContext) -> phasesmith.ScatteringFactorBatch:
        self.calls += 1
        return phasesmith.XrayNonResonant().evaluate(context)


class CountingCorrection:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike):
        self.calls += 1
        return phasesmith.NeutralIntegratedIntensityCorrection().evaluate(
            q_squared_inverse_angstrom2
        )


def test_custom_scattering_and_correction_use_vectorized_fallback_once() -> None:
    scattering = CountingScattering()
    correction = CountingCorrection()
    phase = replace(xray_phase(), scattering=scattering, intensity_correction=correction)
    prepared = phasesmith.PreparedStructuralPattern(
        pattern(),
        phasesmith.ConstantWavelengthExperiment.x_ray(instrument()),
        phase,
    )
    assert not prepared.uses_native_fused_path
    result = prepared.calculate()
    assert scattering.calls == 1
    assert correction.calls == 1
    assert np.all(np.isfinite(result.profile_y))
    with pytest.raises(NotImplementedError, match="built-in fused path"):
        prepared.jvp(np.zeros(len(result.reflections.reflection_ids)))
