from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import rietveld


def instrument() -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.909,
        u_deg2=257.182710995e-4,
        v_deg2=-640.525145369e-4,
        w_deg2=569.378664828e-4,
        x_deg=0.0,
        y_deg=0.0,
    )


def geometry() -> rietveld.ReflectionGeometryBatch:
    positions = np.array([28.0, 64.0, 118.0])
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    return rietveld.ReflectionGeometryBatch(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        d_spacing,
        positions,
        [9.0, 6.0, 3.0],
    )


def phase() -> rietveld.Phase:
    batch = geometry()
    return rietveld.Phase(
        "neutron-phase",
        "Neutron phase",
        rietveld.ReflectionBatch(
            ["n-100", "n-110", "n-111"],
            batch.hkl,
            batch.d_spacing_angstrom,
            batch.two_theta_deg,
            batch.base_integrated_intensity,
        ),
    )


def test_neutron_experiment_is_explicit_monochromatic_plain_data() -> None:
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    assert experiment.radiation.probe is rietveld.RadiationProbe.NEUTRON
    assert experiment.radiation.probe.value == "neutron"
    assert str(experiment.radiation.probe) == "neutron"
    assert experiment.radiation.wavelength_angstrom == instrument().wavelength_angstrom
    assert {field.name for field in fields(experiment.radiation)} == {
        "probe",
        "wavelength_angstrom",
    }


def test_identical_physical_widths_produce_identical_xray_and_neutron_profiles() -> None:
    x = np.linspace(20.0, 125.0, 10_501)
    neutron = rietveld.calculate_monochromatic_cw_pattern(
        x,
        geometry(),
        rietveld.ConstantWavelengthExperiment.neutron(instrument()),
        jacobian_layout="dense",
    )
    x_ray = rietveld.calculate_monochromatic_cw_pattern(
        x,
        geometry(),
        rietveld.ConstantWavelengthExperiment.x_ray(instrument()),
        jacobian_layout="dense",
    )
    np.testing.assert_array_equal(neutron.y, x_ray.y)
    np.testing.assert_array_equal(neutron.jacobian, x_ray.jacobian)
    np.testing.assert_array_equal(
        neutron.derivatives.global_jacobian, x_ray.derivatives.global_jacobian
    )


def test_neutron_high_level_path_reuses_multiphase_calculation_exactly() -> None:
    x = np.linspace(20.0, 125.0, 10_501)
    pattern = rietveld.PowderPattern(x, background=np.full(x.size, 0.2))
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    neutron = rietveld.calculate_neutron_pattern(pattern, experiment, [phase()])
    shared = rietveld.calculate_pattern(pattern, instrument(), [phase()])
    np.testing.assert_array_equal(neutron.y, shared.y)
    np.testing.assert_array_equal(neutron.profile_y, shared.profile_y)
    np.testing.assert_array_equal(
        neutron.derivatives.global_jacobian, shared.derivatives.global_jacobian
    )


def test_neutron_instrument_derivatives_match_centered_differences() -> None:
    x = np.linspace(62.0, 66.0, 4_001)
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    baseline = rietveld.calculate_monochromatic_cw_pattern(x, geometry(), experiment)
    parameter = "u_deg2"
    step = 1.0e-7
    plus_instrument = replace(instrument(), **{parameter: getattr(instrument(), parameter) + step})
    minus_instrument = replace(instrument(), **{parameter: getattr(instrument(), parameter) - step})
    plus = rietveld.calculate_monochromatic_cw_pattern(
        x,
        geometry(),
        rietveld.ConstantWavelengthExperiment.neutron(plus_instrument),
    ).y
    minus = rietveld.calculate_monochromatic_cw_pattern(
        x,
        geometry(),
        rietveld.ConstantWavelengthExperiment.neutron(minus_instrument),
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    row = baseline.derivatives.global_parameter_names.index("u")
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row], finite_difference, rtol=2e-7, atol=2e-8
    )


def test_neutron_profile_retains_integrated_intensity_and_centroid() -> None:
    position = 64.0
    intensity = 6.0
    x = np.linspace(40.0, 88.0, 240_001)
    batch = geometry()
    one = rietveld.ReflectionGeometryBatch(
        [batch.hkl[1]],
        [batch.d_spacing_angstrom[1]],
        [position],
        [intensity],
    )
    actual = rietveld.calculate_monochromatic_cw_pattern(
        x,
        one,
        rietveld.ConstantWavelengthExperiment.neutron(instrument()),
        support_fwhm=100.0,
    ).y
    area = np.trapezoid(actual, x)
    centroid = np.trapezoid(x * actual, x) / area
    assert area == pytest.approx(intensity, rel=7e-4)
    assert centroid == pytest.approx(position, abs=3e-12)


def test_neutron_path_accepts_same_batch_provider_contract() -> None:
    x = np.linspace(20.0, 125.0, 10_501)
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument())
    provider = rietveld.IsotropicSizeBroadening(80.0)
    actual = rietveld.calculate_monochromatic_cw_pattern(
        x, geometry(), experiment, physics=provider
    )
    shared = rietveld.calculate_cw_pattern(x, geometry(), instrument(), physics=provider)
    np.testing.assert_array_equal(actual.y, shared.y)
    np.testing.assert_array_equal(
        actual.derivatives.global_jacobian, shared.derivatives.global_jacobian
    )


def test_neutron_fcj_path_reuses_shared_asymmetric_kernel_exactly() -> None:
    x = np.linspace(25.0, 31.0, 6_001)
    batch = geometry()
    axial = rietveld.FcjGeometry(0.001, 0.001)
    actual = rietveld.calculate_neutron_fcj_pattern(
        x,
        batch,
        rietveld.ConstantWavelengthExperiment.neutron(instrument()),
        axial,
    )
    shared = rietveld.accumulate_cw_fcj(
        x,
        batch.two_theta_deg,
        batch.base_integrated_intensity,
        instrument(),
        axial,
    )
    np.testing.assert_array_equal(actual.y, shared.y)
    np.testing.assert_array_equal(
        actual.derivatives.global_jacobian, shared.derivatives.global_jacobian
    )


def test_neutron_type_rejects_xray_and_wavelength_mismatch() -> None:
    x = np.linspace(20.0, 125.0, 1_001)
    with pytest.raises(ValueError, match="requires neutron"):
        rietveld.calculate_neutron_pattern(
            rietveld.PowderPattern(x),
            rietveld.ConstantWavelengthExperiment.x_ray(instrument()),
            [phase()],
        )
    mismatched = rietveld.MonochromaticRadiation.neutron(1.8)
    with pytest.raises(ValueError, match="must match exactly"):
        rietveld.ConstantWavelengthExperiment(mismatched, instrument())
    with pytest.raises(TypeError, match="RadiationProbe"):
        rietveld.MonochromaticRadiation("neutron", 1.909)  # type: ignore[arg-type]
