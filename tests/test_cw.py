from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import reference
from phasesmith._numpy_compat import trapezoid
from phasesmith.oracle import cw_instrument_from_gsas_centidegrees


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.0e-4,
        x_deg=1.0e-3,
        y_deg=2.0e-3,
    )


def reference_parameters(
    positions: np.ndarray, model: phasesmith.ConstantWavelengthInstrument
) -> reference.ReferenceCwProfileParameters:
    return reference.cw_profile_parameters(
        positions,
        u_deg2=model.u_deg2,
        v_deg2=model.v_deg2,
        w_deg2=model.w_deg2,
        x_deg=model.x_deg,
        y_deg=model.y_deg,
    )


def reference_accumulation(
    x: np.ndarray,
    positions: np.ndarray,
    intensities: np.ndarray,
    model: phasesmith.ConstantWavelengthInstrument,
    support_fwhm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return reference.accumulate_cw(
        x,
        positions,
        intensities,
        u_deg2=model.u_deg2,
        v_deg2=model.v_deg2,
        w_deg2=model.w_deg2,
        x_deg=model.x_deg,
        y_deg=model.y_deg,
        support_fwhm=support_fwhm,
    )


def test_gsas_unit_conversion_and_reflection_widths() -> None:
    model = cw_instrument_from_gsas_centidegrees(
        wavelength_angstrom=1.5406,
        u=2.0,
        v=-1.0,
        w=1.0,
        x=0.1,
        y=0.2,
    )
    position = np.array([19.7126092])
    actual = phasesmith.cw_profile_parameters(position, model)

    np.testing.assert_allclose(actual.gaussian_variance_deg2 * 1.0e4, [0.88663051], atol=5e-9)
    theta = np.deg2rad(position / 2.0)
    expected_lorentzian_centideg = 0.1 / np.cos(theta) + 0.2 * np.tan(theta)
    np.testing.assert_allclose(
        actual.lorentzian_fwhm_deg * 100.0,
        expected_lorentzian_centideg,
        rtol=2e-15,
        atol=0.0,
    )


def test_profile_parameters_match_independent_reference() -> None:
    positions = np.array([5.0, 25.0, 63.2, 120.0, 170.0])
    actual = phasesmith.cw_profile_parameters(positions, instrument())
    expected = reference_parameters(positions, instrument())

    for field in (
        "gaussian_variance_deg2",
        "gaussian_fwhm_deg",
        "lorentzian_fwhm_deg",
        "total_fwhm_deg",
        "eta",
        "d_gaussian_fwhm_d_instrument",
        "d_lorentzian_fwhm_d_instrument",
        "d_component_fwhm_d_two_theta",
    ):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=2e-14, atol=2e-15
        )


def test_fused_accumulation_matches_reference_with_overlap() -> None:
    rng = np.random.default_rng(20260805)
    x = np.linspace(18.0, 122.0, 10_401)
    positions = np.sort(rng.uniform(20.0, 120.0, 24))
    intensities = rng.uniform(0.1, 20.0, positions.size)
    support_fwhm = 12.0

    actual = phasesmith.accumulate_cw(
        x,
        positions,
        intensities,
        instrument(),
        support_fwhm=support_fwhm,
        jacobian_layout="dense",
    )
    expected_y, expected_local, expected_global = reference_accumulation(
        x, positions, intensities, instrument(), support_fwhm
    )

    assert actual.derivatives.local_parameter_names == phasesmith.CW_LOCAL_PARAMETER_ORDER
    assert actual.derivatives.global_parameter_names == phasesmith.CW_GLOBAL_PARAMETER_ORDER
    np.testing.assert_allclose(actual.y, expected_y, rtol=4e-14, atol=2e-14)
    np.testing.assert_allclose(actual.jacobian, expected_local, rtol=5e-13, atol=3e-13)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian, expected_global, rtol=5e-13, atol=3e-11
    )


@pytest.mark.parametrize("parameter", phasesmith.CW_GLOBAL_PARAMETER_ORDER)
def test_global_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(49.0, 51.0, 2_001)
    positions = np.array([49.65, 50.35])
    intensities = np.array([12.0, 7.0])
    baseline_model = instrument()
    baseline = phasesmith.accumulate_cw(
        x,
        positions,
        intensities,
        baseline_model,
        support_fwhm=100.0,
    )
    field = f"{parameter}_deg2" if parameter in {"u", "v", "w"} else f"{parameter}_deg"
    step = 1e-8 if parameter in {"u", "v", "w"} else 1e-7
    plus = replace(baseline_model, **{field: getattr(baseline_model, field) + step})
    minus = replace(baseline_model, **{field: getattr(baseline_model, field) - step})
    finite_difference = (
        phasesmith.accumulate_cw(x, positions, intensities, plus, support_fwhm=100.0).y
        - phasesmith.accumulate_cw(x, positions, intensities, minus, support_fwhm=100.0).y
    ) / (2.0 * step)
    row = phasesmith.CW_GLOBAL_PARAMETER_ORDER.index(parameter)

    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row],
        finite_difference,
        rtol=4e-6,
        atol=2e-7 * max(1.0, float(np.max(np.abs(finite_difference)))),
    )


@pytest.mark.parametrize("parameter", phasesmith.CW_LOCAL_PARAMETER_ORDER)
def test_local_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(49.0, 51.0, 2_001)
    positions = np.array([50.0])
    intensities = np.array([12.0])
    step = 1e-6
    baseline = phasesmith.accumulate_cw(
        x,
        positions,
        intensities,
        instrument(),
        support_fwhm=100.0,
        jacobian_layout="dense",
    )
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
    plus = phasesmith.accumulate_cw(
        x, plus_positions, plus_intensities, instrument(), support_fwhm=100.0
    ).y
    minus = phasesmith.accumulate_cw(
        x, minus_positions, minus_intensities, instrument(), support_fwhm=100.0
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    column = phasesmith.CW_LOCAL_PARAMETER_ORDER.index(parameter)

    np.testing.assert_allclose(
        baseline.jacobian[0, column], finite_difference, rtol=3e-7, atol=3e-7
    )


def test_area_and_centroid_for_isolated_reflection() -> None:
    position = 60.0
    intensity = 17.5
    x = np.linspace(58.0, 62.0, 80_001)
    result = phasesmith.accumulate_cw(
        x,
        [position],
        [intensity],
        instrument(),
        support_fwhm=80.0,
    )
    area = trapezoid(result.y, x)
    centroid = trapezoid(x * result.y, x) / area

    assert area == pytest.approx(intensity, rel=1.2e-3)
    assert centroid == pytest.approx(position, abs=2e-12)


@pytest.mark.parametrize(
    ("positions", "model", "match"),
    [
        ([0.0], instrument(), "two_theta"),
        ([180.0], instrument(), "two_theta"),
        ([45.0], replace(instrument(), w_deg2=-1.0), "Gaussian variance"),
        ([45.0], replace(instrument(), x_deg=-1.0), "Lorentzian FWHM"),
    ],
)
def test_invalid_reflection_domains_are_clear(
    positions: list[float],
    model: phasesmith.ConstantWavelengthInstrument,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        phasesmith.cw_profile_parameters(positions, model)


def test_instrument_and_batch_boundary_validation() -> None:
    with pytest.raises(ValueError, match="wavelength_angstrom"):
        replace(instrument(), wavelength_angstrom=0.0)
    with pytest.raises(ValueError, match="finite"):
        replace(instrument(), u_deg2=np.nan)
    with pytest.raises(ValueError, match="equal length"):
        phasesmith.accumulate_cw([1.0, 2.0], [1.5], [], instrument())
    with pytest.raises(ValueError, match="strictly increasing"):
        phasesmith.accumulate_cw([1.0, 1.0], [1.5], [1.0], instrument())
