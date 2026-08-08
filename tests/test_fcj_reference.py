from __future__ import annotations

import numpy as np
import pytest
from phasesmith import reference


def arguments() -> tuple[float, float, float, float, float]:
    return 12.0, 0.018, 0.006, 0.013, 0.009


def test_zero_axial_limit_is_exactly_symmetric() -> None:
    x = np.linspace(11.0, 13.0, 2_001)
    position, gaussian, lorentzian, _sample, _detector = arguments()
    actual = reference.profile_fcj(x, position, gaussian, lorentzian, 0.0, 0.0)
    expected = reference.profile_tch(x - position, gaussian, lorentzian)

    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.d_position, -expected.d_delta)
    np.testing.assert_array_equal(actual.d_gaussian_fwhm, expected.d_gaussian_fwhm)
    np.testing.assert_array_equal(actual.d_lorentzian_fwhm, expected.d_lorentzian_fwhm)
    np.testing.assert_array_equal(actual.d_sample_over_radius, 0.0)
    np.testing.assert_array_equal(actual.d_detector_over_radius, 0.0)


@pytest.mark.parametrize(
    ("position", "gaussian", "lorentzian", "sample", "detector"),
    [
        (12.0, 0.018, 0.006, 0.012, 0.012),
        (18.0, 0.080, 0.035, 0.020, 0.006),
        (70.0, 0.035, 0.012, 0.016, 0.009),
        (138.0, 0.050, 0.020, 0.014, 0.014),
    ],
)
def test_48_point_quadrature_meets_recorded_reference_error(
    position: float,
    gaussian: float,
    lorentzian: float,
    sample: float,
    detector: float,
) -> None:
    x = np.linspace(position - 1.0, position + 1.0, 2_001)
    actual = reference.profile_fcj(
        x, position, gaussian, lorentzian, sample, detector, quadrature_order=48
    )
    expected = reference.profile_fcj(
        x, position, gaussian, lorentzian, sample, detector, quadrature_order=256
    )
    for field in (
        "value",
        "d_position",
        "d_gaussian_fwhm",
        "d_lorentzian_fwhm",
        "d_sample_over_radius",
        "d_detector_over_radius",
    ):
        oracle = getattr(expected, field)
        scale = max(float(np.max(np.abs(oracle))), 1.0)
        assert float(np.max(np.abs(getattr(actual, field) - oracle))) / scale < 5.1e-8


@pytest.mark.parametrize(
    ("position", "gaussian", "lorentzian", "sample", "detector"),
    [
        (5.0, 0.010, 0.002, 0.001, 0.001),
        (20.0, 0.030, 0.010, 0.001, 0.001),
        (70.0, 0.035, 0.012, 0.016, 0.009),
        (150.0, 0.080, 0.030, 0.001, 0.001),
    ],
)
def test_eight_point_small_span_rule_meets_recorded_reference_error(
    position: float,
    gaussian: float,
    lorentzian: float,
    sample: float,
    detector: float,
) -> None:
    x = np.linspace(position - 1.0, position + 1.0, 2_001)
    actual = reference.profile_fcj(
        x, position, gaussian, lorentzian, sample, detector, quadrature_order=8
    )
    expected = reference.profile_fcj(
        x, position, gaussian, lorentzian, sample, detector, quadrature_order=256
    )
    for field in actual.__dataclass_fields__:
        oracle = getattr(expected, field)
        scale = max(float(np.max(np.abs(oracle))), 1.0)
        error = float(np.max(np.abs(getattr(actual, field) - oracle))) / scale
        assert error < 1.5e-8, (field, error)


@pytest.mark.parametrize(
    ("parameter", "derivative", "step"),
    [
        (0, "d_position", 1e-6),
        (1, "d_gaussian_fwhm", 1e-7),
        (2, "d_lorentzian_fwhm", 1e-7),
        (3, "d_sample_over_radius", 1e-7),
        (4, "d_detector_over_radius", 1e-7),
    ],
)
def test_direct_derivatives_match_centered_differences(
    parameter: int, derivative: str, step: float
) -> None:
    x = np.linspace(11.0, 13.0, 2_001)
    baseline_arguments = arguments()
    actual = reference.profile_fcj(x, *baseline_arguments, quadrature_order=64)
    plus = list(baseline_arguments)
    minus = list(baseline_arguments)
    plus[parameter] += step
    minus[parameter] -= step
    finite_difference = (
        reference.profile_fcj(x, *plus, quadrature_order=64).value
        - reference.profile_fcj(x, *minus, quadrature_order=64).value
    ) / (2.0 * step)

    np.testing.assert_allclose(
        getattr(actual, derivative),
        finite_difference,
        rtol=8e-7,
        atol=2e-7 * max(1.0, float(np.max(np.abs(finite_difference)))),
    )


def test_equal_height_derivatives_respect_sample_detector_symmetry() -> None:
    x = np.linspace(11.0, 13.0, 2_001)
    actual = reference.profile_fcj(x, 12.0, 0.018, 0.006, 0.012, 0.012, quadrature_order=64)
    np.testing.assert_array_equal(actual.d_sample_over_radius, actual.d_detector_over_radius)


def test_normalization_centroid_shift_and_skew_reverse_above_ninety() -> None:
    results = []
    for position in (20.0, 140.0):
        x = np.linspace(position - 2.0, position + 2.0, 20_001)
        profile = reference.profile_fcj(
            x, position, 0.05, 0.0, 0.02, 0.01, quadrature_order=64
        ).value
        area = np.trapezoid(profile, x)
        centroid = np.trapezoid(x * profile, x) / area
        third_moment = np.trapezoid((x - centroid) ** 3 * profile, x) / area
        results.append((area, centroid - position, third_moment))

    for area, _shift, _third_moment in results:
        assert area == pytest.approx(1.0, rel=2e-13)
    assert results[0][1] < 0.0 < results[1][1]
    assert results[0][2] < 0.0 < results[1][2]


@pytest.mark.parametrize(
    ("position", "sample", "detector", "message"),
    [
        (0.0, 0.01, 0.01, "position_deg"),
        (180.0, 0.01, 0.01, "position_deg"),
        (20.0, -0.01, 0.01, "non-negative"),
        (1.0, 1.0, 1.0, "angular domain"),
    ],
)
def test_invalid_geometry_is_rejected(
    position: float, sample: float, detector: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reference.profile_fcj([position], position, 0.02, 0.01, sample, detector)
