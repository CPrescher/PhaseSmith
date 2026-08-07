from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import reference


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.54056,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def geometry() -> phasesmith.FcjGeometry:
    return phasesmith.FcjGeometry(0.013, 0.009)


def doublet() -> phasesmith.WavelengthComponents:
    return phasesmith.WavelengthComponents.doublet(1.54056, 1.54439, 0.5)


def test_component_model_is_immutable_normalized_plain_data() -> None:
    wavelengths = np.array([1.54056, 1.54439])
    intensities = np.array([2.0, 1.0])
    components = phasesmith.WavelengthComponents(wavelengths, intensities)
    wavelengths[0] = 99.0
    intensities[0] = 99.0

    np.testing.assert_array_equal(components.wavelengths_angstrom, [1.54056, 1.54439])
    np.testing.assert_allclose(components.normalized_intensities, [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(components.wavelength_ratios, [1.0, 1.54439 / 1.54056])
    np.testing.assert_allclose(components.intensity_ratios, [1.0, 0.5])
    assert not components.wavelengths_angstrom.flags.writeable
    assert not components.relative_intensities.flags.writeable


def test_one_component_is_exactly_monochromatic_with_and_without_fcj() -> None:
    x = np.linspace(39.0, 41.0, 2_001)
    positions = np.array([39.8, 40.2])
    intensities = np.array([12.0, 7.0])
    components = phasesmith.WavelengthComponents.monochromatic(instrument().wavelength_angstrom)
    monochromatic = phasesmith.accumulate_cw(
        x, positions, intensities, instrument(), jacobian_layout="dense"
    )
    component_result = phasesmith.accumulate_cw_components(
        x, positions, intensities, instrument(), components, jacobian_layout="dense"
    )
    np.testing.assert_array_equal(component_result.y, monochromatic.y)
    np.testing.assert_array_equal(component_result.jacobian, monochromatic.jacobian)
    np.testing.assert_array_equal(
        component_result.derivatives.global_jacobian,
        monochromatic.derivatives.global_jacobian,
    )

    monochromatic_fcj = phasesmith.accumulate_cw_fcj(
        x,
        positions,
        intensities,
        instrument(),
        geometry(),
        jacobian_layout="dense",
    )
    component_fcj = phasesmith.accumulate_cw_fcj_components(
        x,
        positions,
        intensities,
        instrument(),
        components,
        geometry(),
        jacobian_layout="dense",
    )
    np.testing.assert_array_equal(component_fcj.y, monochromatic_fcj.y)
    np.testing.assert_array_equal(component_fcj.jacobian, monochromatic_fcj.jacobian)
    np.testing.assert_array_equal(
        component_fcj.derivatives.global_jacobian,
        monochromatic_fcj.derivatives.global_jacobian,
    )


def test_fused_fcj_doublet_matches_independent_reference() -> None:
    x = np.linspace(49.0, 51.0, 2_001)
    positions = np.array([49.85, 50.15])
    intensities = np.array([12.0, 7.0])
    actual = phasesmith.accumulate_cw_fcj_components(
        x,
        positions,
        intensities,
        instrument(),
        doublet(),
        geometry(),
        support_fwhm=20.0,
        jacobian_layout="dense",
    )
    expected_y, expected_local, expected_global = reference.accumulate_cw_components(
        x,
        positions,
        intensities,
        reference_wavelength_angstrom=instrument().wavelength_angstrom,
        wavelengths_angstrom=doublet().wavelengths_angstrom,
        relative_component_intensities=doublet().relative_intensities,
        u_deg2=instrument().u_deg2,
        v_deg2=instrument().v_deg2,
        w_deg2=instrument().w_deg2,
        x_deg=instrument().x_deg,
        y_deg=instrument().y_deg,
        sample_over_radius=geometry().sample_over_radius,
        detector_over_radius=geometry().detector_over_radius,
        support_fwhm=20.0,
    )
    np.testing.assert_allclose(actual.y, expected_y, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual.jacobian, expected_local, rtol=6e-11, atol=3e-10)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian,
        expected_global,
        rtol=8e-11,
        atol=8e-9,
    )


@pytest.mark.parametrize(
    "parameter",
    (
        "u",
        "v",
        "w",
        "x",
        "y",
        "sample_over_radius",
        "detector_over_radius",
        "wavelength_ratio[1]",
        "intensity_ratio[1]",
    ),
)
def test_all_shared_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(49.7, 50.5, 1_601)
    positions = np.array([50.0])
    intensities = np.array([8.0])
    model = instrument()
    axial = geometry()
    components = doublet()
    baseline = phasesmith.accumulate_cw_fcj_components(
        x, positions, intensities, model, components, axial, support_fwhm=100.0
    )
    plus_model = minus_model = model
    plus_axial = minus_axial = axial
    plus_components = minus_components = components
    if parameter in {"u", "v", "w"}:
        field = f"{parameter}_deg2"
        step = 1e-8
        plus_model = replace(model, **{field: getattr(model, field) + step})
        minus_model = replace(model, **{field: getattr(model, field) - step})
    elif parameter in {"x", "y"}:
        field = f"{parameter}_deg"
        step = 1e-7
        plus_model = replace(model, **{field: getattr(model, field) + step})
        minus_model = replace(model, **{field: getattr(model, field) - step})
    elif parameter in {"sample_over_radius", "detector_over_radius"}:
        step = 1e-7
        plus_axial = replace(axial, **{parameter: getattr(axial, parameter) + step})
        minus_axial = replace(axial, **{parameter: getattr(axial, parameter) - step})
    elif parameter == "wavelength_ratio[1]":
        step = 1e-7
        ratio = components.wavelength_ratios[1]
        plus_components = phasesmith.WavelengthComponents(
            [model.wavelength_angstrom, model.wavelength_angstrom * (ratio + step)],
            components.relative_intensities,
        )
        minus_components = phasesmith.WavelengthComponents(
            [model.wavelength_angstrom, model.wavelength_angstrom * (ratio - step)],
            components.relative_intensities,
        )
    else:
        step = 1e-7
        ratio = components.intensity_ratios[1]
        plus_components = phasesmith.WavelengthComponents(
            components.wavelengths_angstrom, [1.0, ratio + step]
        )
        minus_components = phasesmith.WavelengthComponents(
            components.wavelengths_angstrom, [1.0, ratio - step]
        )
    plus = phasesmith.accumulate_cw_fcj_components(
        x,
        positions,
        intensities,
        plus_model,
        plus_components,
        plus_axial,
        support_fwhm=100.0,
    ).y
    minus = phasesmith.accumulate_cw_fcj_components(
        x,
        positions,
        intensities,
        minus_model,
        minus_components,
        minus_axial,
        support_fwhm=100.0,
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    row = baseline.derivatives.global_parameter_names.index(parameter)
    scale = max(1.0, float(np.max(np.abs(finite_difference))))
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row],
        finite_difference,
        rtol=1.2e-5,
        atol=5e-7 * scale,
    )


@pytest.mark.parametrize("parameter", ("intensity", "position"))
def test_logical_reflection_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(49.7, 50.5, 1_601)
    positions = np.array([50.0])
    intensities = np.array([8.0])
    baseline = phasesmith.accumulate_cw_fcj_components(
        x,
        positions,
        intensities,
        instrument(),
        doublet(),
        geometry(),
        support_fwhm=100.0,
        jacobian_layout="dense",
    )
    step = 1e-6
    plus_positions = positions.copy()
    minus_positions = positions.copy()
    plus_intensities = intensities.copy()
    minus_intensities = intensities.copy()
    if parameter == "position":
        plus_positions[0] += step
        minus_positions[0] -= step
    else:
        plus_intensities[0] += step
        minus_intensities[0] -= step
    plus = phasesmith.accumulate_cw_fcj_components(
        x,
        plus_positions,
        plus_intensities,
        instrument(),
        doublet(),
        geometry(),
        support_fwhm=100.0,
    ).y
    minus = phasesmith.accumulate_cw_fcj_components(
        x,
        minus_positions,
        minus_intensities,
        instrument(),
        doublet(),
        geometry(),
        support_fwhm=100.0,
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    column = baseline.derivatives.local_parameter_names.index(parameter)
    np.testing.assert_allclose(
        baseline.jacobian[0, column], finite_difference, rtol=4e-6, atol=3e-6
    )


def test_resolved_doublet_conserves_area_and_intensity_ratio() -> None:
    narrow = replace(
        instrument(),
        u_deg2=0.0,
        v_deg2=0.0,
        w_deg2=1.0e-8,
        x_deg=0.0,
        y_deg=0.0,
    )
    base_position = 100.0
    positions, _d_base, _d_ratio = reference.wavelength_component_positions(
        [base_position], narrow.wavelength_angstrom, doublet().wavelengths_angstrom
    )
    x = np.linspace(99.8, positions[0, 1] + 0.2, 40_001)
    result = phasesmith.accumulate_cw_components(
        x, [base_position], [9.0], narrow, doublet(), support_fwhm=100.0
    )
    midpoint = float(np.mean(positions[0]))
    first = np.trapezoid(result.y[x < midpoint], x[x < midpoint])
    second = np.trapezoid(result.y[x > midpoint], x[x > midpoint])
    total = np.trapezoid(result.y, x)

    assert total == pytest.approx(9.0, rel=2e-6)
    assert second / first == pytest.approx(0.5, rel=3e-6)


def test_unresolved_equal_wavelength_limit_matches_monochromatic_values() -> None:
    x = np.linspace(49.0, 51.0, 2_001)
    components = phasesmith.WavelengthComponents(
        [instrument().wavelength_angstrom, instrument().wavelength_angstrom],
        [1.0, 0.5],
    )
    actual = phasesmith.accumulate_cw_components(
        x, [50.0], [8.0], instrument(), components, support_fwhm=20.0
    ).y
    expected = phasesmith.accumulate_cw(x, [50.0], [8.0], instrument(), support_fwhm=20.0).y
    np.testing.assert_allclose(actual, expected, rtol=3e-16, atol=6e-14)


def test_component_boundary_validation_is_clear() -> None:
    with pytest.raises(ValueError, match="equal length"):
        phasesmith.WavelengthComponents([1.0], [1.0, 0.5])
    with pytest.raises(ValueError, match="reference component"):
        phasesmith.WavelengthComponents([1.0, 1.1], [0.0, 1.0])
    with pytest.raises(ValueError, match="match the instrument"):
        phasesmith.accumulate_cw_components(
            [49.0, 50.0],
            [49.5],
            [1.0],
            instrument(),
            phasesmith.WavelengthComponents.monochromatic(1.0),
        )
    with pytest.raises(ValueError, match="Bragg domain"):
        phasesmith.accumulate_cw_components(
            [169.0, 171.0],
            [170.0],
            [1.0],
            instrument(),
            phasesmith.WavelengthComponents(
                [instrument().wavelength_angstrom, 2.0 * instrument().wavelength_angstrom],
                [1.0, 0.5],
            ),
        )
