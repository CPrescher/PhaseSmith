from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld import reference


def instrument() -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.0e-4,
        x_deg=1.0e-3,
        y_deg=2.0e-3,
    )


def geometry() -> rietveld.FcjGeometry:
    return rietveld.FcjGeometry(
        sample_over_radius=0.013,
        detector_over_radius=0.009,
    )


def reference_accumulation(
    x: np.ndarray,
    positions: np.ndarray,
    intensities: np.ndarray,
    support_fwhm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = instrument()
    axial = geometry()
    return reference.accumulate_cw_fcj(
        x,
        positions,
        intensities,
        u_deg2=model.u_deg2,
        v_deg2=model.v_deg2,
        w_deg2=model.w_deg2,
        x_deg=model.x_deg,
        y_deg=model.y_deg,
        sample_over_radius=axial.sample_over_radius,
        detector_over_radius=axial.detector_over_radius,
        support_fwhm=support_fwhm,
    )


@pytest.mark.parametrize(
    ("position", "gaussian", "lorentzian", "sample", "detector"),
    [
        (12.0, 0.018, 0.006, 0.012, 0.012),
        (70.0, 0.035, 0.012, 0.016, 0.009),
        (138.0, 0.050, 0.020, 0.014, 0.014),
    ],
)
def test_native_profile_matches_independent_quadrature(
    position: float,
    gaussian: float,
    lorentzian: float,
    sample: float,
    detector: float,
) -> None:
    x = np.linspace(position - 0.4, position + 0.4, 801)
    actual = rietveld.profile_fcj(
        x,
        position,
        gaussian,
        lorentzian,
        rietveld.FcjGeometry(sample, detector),
    )
    expected = reference.profile_fcj(
        x,
        position,
        gaussian,
        lorentzian,
        sample,
        detector,
        quadrature_order=48,
    )
    for field in actual.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=2e-12, atol=2e-12
        )


def test_fused_cw_fcj_batch_matches_independent_reference() -> None:
    x = np.linspace(10.0, 140.0, 13_001)
    positions = np.array([12.0, 69.95, 70.05, 138.0])
    intensities = np.array([5.0, 11.0, 7.0, 4.0])
    support_fwhm = 18.0
    actual = rietveld.accumulate_cw_fcj(
        x,
        positions,
        intensities,
        instrument(),
        geometry(),
        support_fwhm=support_fwhm,
        jacobian_layout="dense",
    )
    expected_y, expected_local, expected_global = reference_accumulation(
        x, positions, intensities, support_fwhm
    )

    assert actual.derivatives.local_parameter_names == rietveld.CW_FCJ_LOCAL_PARAMETER_ORDER
    assert actual.derivatives.global_parameter_names == rietveld.CW_FCJ_GLOBAL_PARAMETER_ORDER
    np.testing.assert_allclose(actual.y, expected_y, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual.jacobian, expected_local, rtol=5e-11, atol=2e-10)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian,
        expected_global,
        rtol=5e-11,
        atol=5e-9,
    )


@pytest.mark.parametrize("parameter", rietveld.CW_FCJ_GLOBAL_PARAMETER_ORDER)
def test_global_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(11.7, 12.2, 1_001)
    positions = np.array([12.0])
    intensities = np.array([8.0])
    model = instrument()
    axial = geometry()
    baseline = rietveld.accumulate_cw_fcj(
        x, positions, intensities, model, axial, support_fwhm=100.0
    )
    if parameter in {"u", "v", "w"}:
        field = f"{parameter}_deg2"
        step = 1e-8
        plus_model = replace(model, **{field: getattr(model, field) + step})
        minus_model = replace(model, **{field: getattr(model, field) - step})
        plus_axial = minus_axial = axial
    elif parameter in {"x", "y"}:
        field = f"{parameter}_deg"
        step = 1e-7
        plus_model = replace(model, **{field: getattr(model, field) + step})
        minus_model = replace(model, **{field: getattr(model, field) - step})
        plus_axial = minus_axial = axial
    else:
        step = 1e-7
        plus_model = minus_model = model
        plus_axial = replace(axial, **{parameter: getattr(axial, parameter) + step})
        minus_axial = replace(axial, **{parameter: getattr(axial, parameter) - step})
    plus = rietveld.accumulate_cw_fcj(
        x, positions, intensities, plus_model, plus_axial, support_fwhm=100.0
    ).y
    minus = rietveld.accumulate_cw_fcj(
        x, positions, intensities, minus_model, minus_axial, support_fwhm=100.0
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    row = rietveld.CW_FCJ_GLOBAL_PARAMETER_ORDER.index(parameter)
    scale = max(1.0, float(np.max(np.abs(finite_difference))))
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row],
        finite_difference,
        rtol=8e-6,
        atol=4e-7 * scale,
    )


@pytest.mark.parametrize("parameter", rietveld.CW_FCJ_LOCAL_PARAMETER_ORDER)
def test_local_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(11.7, 12.2, 1_001)
    positions = np.array([12.0])
    intensities = np.array([8.0])
    baseline = rietveld.accumulate_cw_fcj(
        x,
        positions,
        intensities,
        instrument(),
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
    plus = rietveld.accumulate_cw_fcj(
        x,
        plus_positions,
        plus_intensities,
        instrument(),
        geometry(),
        support_fwhm=100.0,
    ).y
    minus = rietveld.accumulate_cw_fcj(
        x,
        minus_positions,
        minus_intensities,
        instrument(),
        geometry(),
        support_fwhm=100.0,
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    column = rietveld.CW_FCJ_LOCAL_PARAMETER_ORDER.index(parameter)
    np.testing.assert_allclose(
        baseline.jacobian[0, column], finite_difference, rtol=3e-6, atol=3e-6
    )


def test_zero_geometry_recovers_symmetric_cw_batch() -> None:
    x = np.linspace(39.0, 41.0, 2_001)
    positions = np.array([39.8, 40.2])
    intensities = np.array([12.0, 7.0])
    symmetric = rietveld.accumulate_cw(
        x, positions, intensities, instrument(), support_fwhm=20.0, jacobian_layout="dense"
    )
    fcj = rietveld.accumulate_cw_fcj(
        x,
        positions,
        intensities,
        instrument(),
        rietveld.FcjGeometry(0.0, 0.0),
        support_fwhm=20.0,
        jacobian_layout="dense",
    )
    np.testing.assert_array_equal(fcj.y, symmetric.y)
    np.testing.assert_array_equal(fcj.jacobian, symmetric.jacobian)
    np.testing.assert_array_equal(
        fcj.derivatives.global_jacobian[:5], symmetric.derivatives.global_jacobian
    )
    np.testing.assert_array_equal(fcj.derivatives.global_jacobian[5:], 0.0)


def test_area_centroid_and_skew_reverse_above_ninety_degrees() -> None:
    moments = []
    for position in (20.0, 140.0):
        x = np.linspace(position - 2.0, position + 2.0, 20_001)
        values = rietveld.profile_fcj(
            x, position, 0.05, 0.0, rietveld.FcjGeometry(0.02, 0.01)
        ).value
        area = np.trapezoid(values, x)
        centroid = np.trapezoid(x * values, x) / area
        skew_moment = np.trapezoid((x - centroid) ** 3 * values, x) / area
        moments.append((area, centroid - position, skew_moment))

    for area, _shift, _skew in moments:
        assert area == pytest.approx(1.0, rel=2e-13)
    assert moments[0][1] < 0.0 < moments[1][1]
    assert moments[0][2] < 0.0 < moments[1][2]


def test_public_geometry_validation_is_clear() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rietveld.FcjGeometry(-0.1, 0.1)
    with pytest.raises(ValueError, match="finite"):
        rietveld.FcjGeometry(np.nan, 0.1)
