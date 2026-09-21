from __future__ import annotations

import math

import numpy as np
import phasesmith
import pytest
from phasesmith import reference
from phasesmith._numpy_compat import trapezoid


def test_tch_transform_matches_independent_reference() -> None:
    rng = np.random.default_rng(20260805)
    for gaussian, lorentzian in zip(
        rng.uniform(0.001, 0.4, 100),
        rng.uniform(0.001, 0.4, 100),
        strict=True,
    ):
        actual = phasesmith.tch_shape_from_fwhm(gaussian, lorentzian)
        expected = reference.tch_shape_from_fwhm(gaussian, lorentzian)
        for field in actual.__dataclass_fields__:
            assert getattr(actual, field) == pytest.approx(
                getattr(expected, field), rel=4e-15, abs=2e-14
            )


def test_tch_pure_component_limits() -> None:
    gaussian = phasesmith.tch_shape_from_fwhm(0.2, 0.0)
    assert gaussian.total_fwhm == pytest.approx(0.2, rel=1e-15)
    assert gaussian.eta == 0.0

    lorentzian = phasesmith.tch_shape_from_fwhm(0.0, 0.3)
    assert lorentzian.total_fwhm == pytest.approx(0.3, rel=1e-15)
    assert lorentzian.eta == pytest.approx(1.0, abs=2e-16)


def test_tch_transform_is_stable_for_extreme_width_scales() -> None:
    ordinary = phasesmith.tch_shape_from_fwhm(0.7, 0.3)
    tiny = phasesmith.tch_shape_from_fwhm(0.7e-250, 0.3e-250)
    assert tiny.total_fwhm / 1e-250 == pytest.approx(ordinary.total_fwhm, rel=2e-15)
    assert tiny.eta == pytest.approx(ordinary.eta, rel=2e-15)
    assert tiny.d_total_fwhm_d_gaussian_fwhm == pytest.approx(
        ordinary.d_total_fwhm_d_gaussian_fwhm, rel=2e-15
    )
    with pytest.raises(ValueError, match="outside finite range"):
        phasesmith.tch_shape_from_fwhm(np.finfo(np.float64).max, np.finfo(np.float64).max)


def test_explicit_gaussian_sigma_helpers_apply_chain_rule() -> None:
    sigma = 0.031
    lorentzian = 0.023
    sigma_shape = phasesmith.tch_shape_from_gaussian_sigma(sigma, lorentzian)
    fwhm_shape = phasesmith.tch_shape_from_fwhm(
        reference.GAUSSIAN_FWHM_PER_SIGMA * sigma, lorentzian
    )
    assert sigma_shape.total_fwhm == fwhm_shape.total_fwhm
    assert sigma_shape.eta == fwhm_shape.eta
    assert sigma_shape.d_total_fwhm_d_gaussian_sigma == pytest.approx(
        fwhm_shape.d_total_fwhm_d_gaussian_fwhm * reference.GAUSSIAN_FWHM_PER_SIGMA
    )
    assert sigma_shape.d_eta_d_gaussian_sigma == pytest.approx(
        fwhm_shape.d_eta_d_gaussian_fwhm * reference.GAUSSIAN_FWHM_PER_SIGMA
    )

    delta = np.linspace(-0.2, 0.2, 101)
    sigma_profile = phasesmith.profile_tch_from_gaussian_sigma(delta, sigma, lorentzian)
    fwhm_profile = phasesmith.profile_tch(
        delta, reference.GAUSSIAN_FWHM_PER_SIGMA * sigma, lorentzian
    )
    np.testing.assert_array_equal(sigma_profile.value, fwhm_profile.value)
    np.testing.assert_allclose(
        sigma_profile.d_gaussian_sigma,
        fwhm_profile.d_gaussian_fwhm * reference.GAUSSIAN_FWHM_PER_SIGMA,
        rtol=2e-15,
    )


def test_tch_profile_matches_reference_and_shape_contracts() -> None:
    gaussian = 0.071
    lorentzian = 0.023
    shape = phasesmith.tch_shape_from_fwhm(gaussian, lorentzian)
    delta = np.linspace(-0.8, 0.8, 4_001)
    actual = phasesmith.profile_tch(delta, gaussian, lorentzian)
    expected = reference.profile_tch(delta, gaussian, lorentzian)
    np.testing.assert_allclose(actual.value, expected.value, rtol=3e-15, atol=2e-15)
    np.testing.assert_allclose(actual.d_delta, expected.d_delta, rtol=5e-15, atol=2e-13)
    np.testing.assert_allclose(
        actual.d_gaussian_fwhm, expected.d_gaussian_fwhm, rtol=7e-15, atol=2e-13
    )
    np.testing.assert_allclose(
        actual.d_lorentzian_fwhm,
        expected.d_lorentzian_fwhm,
        rtol=7e-15,
        atol=2e-13,
    )

    half_maximum = phasesmith.profile_tch(
        [0.0, -shape.total_fwhm / 2.0, shape.total_fwhm / 2.0],
        gaussian,
        lorentzian,
    ).value
    np.testing.assert_allclose(half_maximum[1:], half_maximum[0] / 2.0, rtol=2e-15)
    assert actual.value[0] == actual.value[-1]
    assert np.all(actual.value > 0.0)


@pytest.mark.parametrize("parameter", ["delta", "gaussian_fwhm", "lorentzian_fwhm"])
def test_tch_profile_derivatives_match_centered_differences(parameter: str) -> None:
    delta = np.array([-0.13, -0.021, 0.0, 0.047, 0.16])
    gaussian = 0.083
    lorentzian = 0.037
    baseline = phasesmith.profile_tch(delta, gaussian, lorentzian)
    step = 1e-7
    if parameter == "delta":
        plus = phasesmith.profile_tch(delta + step, gaussian, lorentzian).value
        minus = phasesmith.profile_tch(delta - step, gaussian, lorentzian).value
        analytic = baseline.d_delta
    elif parameter == "gaussian_fwhm":
        plus = phasesmith.profile_tch(delta, gaussian + step, lorentzian).value
        minus = phasesmith.profile_tch(delta, gaussian - step, lorentzian).value
        analytic = baseline.d_gaussian_fwhm
    else:
        plus = phasesmith.profile_tch(delta, gaussian, lorentzian + step).value
        minus = phasesmith.profile_tch(delta, gaussian, lorentzian - step).value
        analytic = baseline.d_lorentzian_fwhm
    np.testing.assert_allclose(analytic, (plus - minus) / (2.0 * step), rtol=2e-8, atol=2e-8)


def test_tch_support_integral_and_centroid() -> None:
    position = 31.7
    intensity = 143.0
    gaussian = 0.062
    lorentzian = 0.028
    support = 15.0
    shape = phasesmith.tch_shape_from_fwhm(gaussian, lorentzian)
    radius = support * shape.total_fwhm
    x = np.linspace(position - radius, position + radius, 300_001)
    result = phasesmith.accumulate_tch(
        x,
        [position],
        [intensity],
        [gaussian],
        [lorentzian],
        support_fwhm=support,
    )
    gaussian_fraction = math.erf(2.0 * math.sqrt(math.log(2.0)) * support)
    lorentzian_fraction = 2.0 / math.pi * math.atan(2.0 * support)
    expected_integral = intensity * (
        shape.eta * lorentzian_fraction + (1.0 - shape.eta) * gaussian_fraction
    )
    sampled_integral = trapezoid(result.y, x)
    assert sampled_integral == pytest.approx(expected_integral, rel=2e-10)
    centroid = trapezoid(x * result.y, x) / sampled_integral
    assert centroid == pytest.approx(position, abs=5e-15)


def test_tch_accumulation_matches_independent_reference() -> None:
    rng = np.random.default_rng(73)
    x = np.cumsum(rng.uniform(0.001, 0.005, 2_001))
    positions = rng.uniform(x[0] - 0.1, x[-1] + 0.1, 29)
    intensities = rng.uniform(-20.0, 300.0, positions.size)
    gaussian = rng.uniform(0.003, 0.08, positions.size)
    lorentzian = rng.uniform(0.0, 0.06, positions.size)
    support = 5.73
    actual = phasesmith.accumulate_tch(
        x,
        positions,
        intensities,
        gaussian,
        lorentzian,
        support_fwhm=support,
    )
    expected_y, expected_jacobian = reference.accumulate_tch(
        x,
        positions,
        intensities,
        gaussian,
        lorentzian,
        support_fwhm=support,
    )
    np.testing.assert_allclose(actual.y, expected_y, rtol=5e-15, atol=2e-12)
    np.testing.assert_allclose(
        actual.derivatives.local.to_dense(x.size),
        expected_jacobian,
        rtol=5e-14,
        atol=6e-12,
    )
    assert actual.derivatives.local_parameter_names == phasesmith.TCH_PARAMETER_ORDER


@pytest.mark.parametrize("parameter", range(4), ids=phasesmith.TCH_PARAMETER_ORDER)
def test_tch_accumulation_derivatives_match_finite_differences(parameter: int) -> None:
    x = np.linspace(-0.5, 0.5, 1_001)
    parameters = np.array([[0.013, 82.0, 0.071, 0.023]], dtype=np.float64)
    support = 6.123
    baseline = phasesmith.accumulate_tch(
        x,
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
        support_fwhm=support,
        jacobian_layout="dense",
    )
    input_column = [1, 0, 2, 3][parameter]
    step = [2e-5, 2e-7, 2e-7, 2e-7][parameter]
    plus = parameters.copy()
    minus = parameters.copy()
    plus[0, input_column] += step
    minus[0, input_column] -= step

    def evaluate(values: np.ndarray) -> np.ndarray:
        return phasesmith.accumulate_tch(
            x,
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            support_fwhm=support,
        ).y

    finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
    shape = phasesmith.tch_shape_from_fwhm(parameters[0, 2], parameters[0, 3])
    active = np.abs(x - parameters[0, 0]) < support * shape.total_fwhm - 1e-10
    np.testing.assert_allclose(
        baseline.jacobian[0, parameter, active],
        finite_difference[active],
        rtol=4e-7,
        atol=3e-7,
    )


@pytest.mark.parametrize(
    ("gaussian", "lorentzian", "message"),
    [
        (0.0, 0.0, "at least one"),
        (-0.1, 0.2, "Gaussian FWHM must be non-negative"),
        (0.1, -0.2, "Lorentzian FWHM must be non-negative"),
        (float("nan"), 0.2, "Gaussian FWHM must be finite"),
    ],
)
def test_invalid_tch_widths_fail_clearly(gaussian: float, lorentzian: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        phasesmith.tch_shape_from_fwhm(gaussian, lorentzian)
    with pytest.raises(ValueError, match=message):
        phasesmith.profile_tch([0.0], gaussian, lorentzian)


@pytest.mark.parametrize(
    ("sigma", "message"),
    [(-0.1, "sigma must be non-negative"), (float("nan"), "sigma must be finite")],
)
def test_invalid_gaussian_sigma_fails_with_sigma_convention(sigma: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        phasesmith.tch_shape_from_gaussian_sigma(sigma, 0.1)
    with pytest.raises(ValueError, match=message):
        phasesmith.profile_tch_from_gaussian_sigma([0.0], sigma, 0.1)
