from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rietveld
from numpy.typing import ArrayLike
from rietveld import reference, scattering_reference
from rietveld.crystallography_reference import (
    reference_cell_geometry,
    reference_structure_factor_values,
)


def inversion_group() -> rietveld.SpaceGroup:
    return rietveld.SpaceGroup(
        [
            rietveld.SymmetryOperation.identity(),
            rietveld.SymmetryOperation(
                [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                (0, 0, 0),
            ),
        ]
    )


def structure() -> rietveld.CrystalStructure:
    return rietveld.CrystalStructure(
        "structure",
        "General symmetry",
        rietveld.UnitCell(4.7, 5.1, 6.2, 82.0, 87.0, 74.0),
        inversion_group(),
        (
            rietveld.AtomSite("si", "Si1", "Si", "Si", (0.17, 0.23, 0.31), 0.82, 0.012),
            rietveld.AtomSite("o", "O1", "O", "O", (0.0, 0.0, 0.0), 0.55, 0.018),
        ),
    )


def reflections() -> rietveld.StructuralReflectionBatch:
    return rietveld.StructuralReflectionBatch(
        ("1,0,1", "2,1,1", "1,2,3"),
        [[1, 0, 1], [2, 1, 1], [1, 2, 3]],
        [2, 4, 2],
    )


def instrument() -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        1.2e-4,
        1.5e-3,
        3.0e-3,
    )


def xray_phase(*, scale: float = 1.4) -> rietveld.RietveldPhase:
    return rietveld.RietveldPhase(
        "phase",
        "Structural phase",
        structure(),
        reflections(),
        rietveld.XrayNonResonant(),
        rietveld.NeutralIntegratedIntensityCorrection(),
        scale,
    )


def pattern() -> rietveld.PowderPattern:
    return rietveld.PowderPattern(
        np.linspace(10.0, 100.0, 9_001),
        background=np.linspace(0.1, 0.2, 9_001),
    )


def test_fused_structural_pattern_matches_separate_vectorized_layers() -> None:
    phase = xray_phase()
    experiment = rietveld.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = rietveld.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path

    actual = prepared.calculate()
    structural = rietveld.calculate_structure_factors(
        phase.structure,
        phase.reflections.hkl,
        phase.reflections.multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
        scale=phase.scale,
    )
    spacing = phase.structure.cell.d_spacings(phase.reflections.hkl).d_spacing_angstrom
    position = 2.0 * np.degrees(np.arcsin(instrument().wavelength_angstrom / (2.0 * spacing)))
    separate = rietveld.accumulate_cw(
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


def test_fused_structural_pattern_matches_independent_numpy_equations() -> None:
    phase = xray_phase()
    actual = rietveld.calculate_structural_pattern(
        pattern(),
        rietveld.ConstantWavelengthExperiment.x_ray(instrument()),
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
    phase: rietveld.RietveldPhase,
    direction: np.ndarray,
    step: float,
) -> rietveld.RietveldPhase:
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
        cell=rietveld.UnitCell(*cell_values),
        sites=tuple(sites),
    )
    return replace(phase, structure=model, scale=phase.scale + step * direction[-1])


def test_structural_pattern_jvp_matches_finite_difference_and_vjp_adjoint() -> None:
    phase = xray_phase()
    experiment = rietveld.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = rietveld.PreparedStructuralPattern(pattern(), experiment, phase)
    names = rietveld.p1_parameter_names(phase.structure.to_isotropic_site_batch())
    direction = np.zeros(len(names))
    direction[[0, 6, 12, 14, len(names) - 1]] = [0.04, 0.015, -0.03, 0.002, 0.05]

    actual = prepared.jvp(direction)
    step = 1.0e-6
    plus = rietveld.calculate_structural_pattern(
        pattern(), experiment, _perturb_phase(phase, direction, step)
    )
    minus = rietveld.calculate_structural_pattern(
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


def test_monochromatic_neutron_structural_pattern_uses_native_path() -> None:
    phase = replace(
        xray_phase(),
        scattering=rietveld.NeutronNuclear(),
        intensity_correction=rietveld.NeutralIntegratedIntensityCorrection(),
    )
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    prepared = rietveld.PreparedStructuralPattern(pattern(), experiment, phase)
    result = prepared.calculate()
    assert prepared.uses_native_fused_path
    assert np.all(np.isfinite(result.profile_y))
    assert np.all(result.reflections.integrated_intensity >= 0.0)


def test_builtin_size_broadening_stays_on_fused_structural_path() -> None:
    phase = replace(xray_phase(), physics=rietveld.IsotropicSizeBroadening(70.0))
    experiment = rietveld.ConstantWavelengthExperiment.x_ray(instrument())
    prepared = rietveld.PreparedStructuralPattern(pattern(), experiment, phase)
    assert prepared.uses_native_fused_path
    actual = prepared.calculate()

    geometry = rietveld.ReflectionGeometryBatch(
        phase.reflections.hkl,
        actual.reflections.d_spacing_angstrom,
        actual.reflections.two_theta_deg,
        actual.reflections.integrated_intensity,
    )
    contribution = phase.physics.evaluate(rietveld.PhysicsContext(geometry, instrument()))
    separate = rietveld.accumulate_cw_contributions(
        pattern().x,
        geometry.two_theta_deg,
        geometry.base_integrated_intensity,
        instrument(),
        contribution,
    )
    np.testing.assert_allclose(actual.profile_y, separate.y, rtol=3e-15, atol=2e-11)
    assert actual.derivatives.global_parameter_names[-1] == ("isotropic_size.crystallite_size_nm")


def test_probe_mismatch_is_rejected_before_calculation() -> None:
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    with pytest.raises(ValueError, match="incompatible"):
        rietveld.PreparedStructuralPattern(pattern(), experiment, xray_phase())


def test_structural_reflections_can_be_created_from_generated_families() -> None:
    model = structure()
    generated = rietveld.PreparedReflectionGenerator(model.space_group).generate(
        model.cell,
        rietveld.DSpacingRange(1.5, 4.0),
    )
    batch = rietveld.StructuralReflectionBatch.from_generated(generated)
    np.testing.assert_array_equal(batch.hkl, generated.hkl)
    np.testing.assert_array_equal(batch.multiplicity, generated.multiplicity)
    assert batch.reflection_ids == generated.reflection_ids
    assert not batch.hkl.flags.writeable


class CountingScattering:
    descriptor = rietveld.XRAY_NON_RESONANT_DESCRIPTOR

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: rietveld.ScatteringContext) -> rietveld.ScatteringFactorBatch:
        self.calls += 1
        return rietveld.XrayNonResonant().evaluate(context)


class CountingCorrection:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike):
        self.calls += 1
        return rietveld.NeutralIntegratedIntensityCorrection().evaluate(q_squared_inverse_angstrom2)


def test_custom_scattering_and_correction_use_vectorized_fallback_once() -> None:
    scattering = CountingScattering()
    correction = CountingCorrection()
    phase = replace(xray_phase(), scattering=scattering, intensity_correction=correction)
    prepared = rietveld.PreparedStructuralPattern(
        pattern(),
        rietveld.ConstantWavelengthExperiment.x_ray(instrument()),
        phase,
    )
    assert not prepared.uses_native_fused_path
    result = prepared.calculate()
    assert scattering.calls == 1
    assert correction.calls == 1
    assert np.all(np.isfinite(result.profile_y))
    with pytest.raises(NotImplementedError, match="built-in fused path"):
        prepared.jvp(np.zeros(len(result.reflections.reflection_ids)))
