from __future__ import annotations

import math

import numpy as np
import pytest
import rietveld
from rietveld import reference


def test_native_profile_matches_independent_numpy_reference() -> None:
    rng = np.random.default_rng(20260804)
    delta = rng.uniform(-3.0, 3.0, 2_000)
    for fwhm, eta in zip(
        rng.uniform(0.02, 1.5, 12), rng.uniform(0.0, 1.0, 12), strict=True
    ):
        actual = rietveld.profile(delta, fwhm, eta)
        expected = reference.profile(delta, fwhm, eta)
        np.testing.assert_allclose(actual.value, expected.value, rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(actual.d_delta, expected.d_delta, rtol=4e-15, atol=2e-14)
        np.testing.assert_allclose(actual.d_fwhm, expected.d_fwhm, rtol=4e-15, atol=2e-14)
        np.testing.assert_allclose(actual.d_eta, expected.d_eta, rtol=3e-15, atol=2e-15)


def test_fwhm_and_symmetry_contract() -> None:
    fwhm = 0.37
    for eta in np.linspace(0.0, 1.0, 9):
        result = rietveld.profile([0.0, -fwhm / 2.0, fwhm / 2.0], fwhm, eta)
        np.testing.assert_allclose(result.value[1:], result.value[0] / 2.0, rtol=2e-15)
        assert result.d_delta[0] == 0.0
        assert result.value[1] == result.value[2]


def test_support_integral_matches_analytic_truncation() -> None:
    position = 0.2
    intensity = 137.0
    fwhm = 0.08
    eta = 0.42
    support = 12.0
    radius = support * fwhm
    x = np.linspace(position - radius, position + radius, 200_001)
    result = rietveld.accumulate(
        x, [position], [intensity], [fwhm], [eta], support_fwhm=support
    )

    gaussian_fraction = math.erf(2.0 * math.sqrt(math.log(2.0)) * support)
    lorentzian_fraction = 2.0 / math.pi * math.atan(2.0 * support)
    expected_integral = intensity * (
        eta * lorentzian_fraction + (1.0 - eta) * gaussian_fraction
    )
    sampled_integral = np.trapezoid(result.y, x)
    assert sampled_integral == pytest.approx(expected_integral, rel=2e-10)

    centroid = np.trapezoid(x * result.y, x) / sampled_integral
    assert centroid == pytest.approx(position, abs=2e-15)


def test_fused_accumulation_matches_reference() -> None:
    rng = np.random.default_rng(91)
    x = np.linspace(10.003, 19.997, 4_000)
    positions = np.sort(rng.uniform(10.2, 19.8, 37))
    intensities = rng.uniform(-10.0, 500.0, positions.size)
    fwhms = rng.uniform(0.015, 0.15, positions.size)
    etas = rng.uniform(0.0, 1.0, positions.size)

    actual = rietveld.accumulate(
        x, positions, intensities, fwhms, etas, support_fwhm=7.37
    )
    expected_y, expected_jacobian = reference.accumulate(
        x, positions, intensities, fwhms, etas, support_fwhm=7.37
    )
    np.testing.assert_allclose(actual.y, expected_y, rtol=3e-15, atol=2e-13)
    np.testing.assert_allclose(actual.jacobian, expected_jacobian, rtol=4e-15, atol=2e-13)
    assert rietveld.PARAMETER_ORDER == ("intensity", "position", "fwhm", "eta")


@pytest.mark.parametrize("parameter", range(4), ids=rietveld.PARAMETER_ORDER)
def test_accumulation_jacobian_matches_centered_finite_difference(parameter: int) -> None:
    x = np.linspace(-0.493, 0.507, 1_001)
    parameters = np.array([[0.013, 82.0, 0.071, 0.37]], dtype=np.float64)
    support = 5.321
    baseline = rietveld.accumulate(
        x,
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
        support_fwhm=support,
    )
    # The input table is position, intensity, FWHM, eta; the public Jacobian is
    # intentionally intensity, position, FWHM, eta.
    input_column = [1, 0, 2, 3][parameter]
    step = [2e-5, 2e-7, 2e-7, 2e-6][parameter]
    plus = parameters.copy()
    minus = parameters.copy()
    plus[0, input_column] += step
    minus[0, input_column] -= step

    def evaluate(values: np.ndarray) -> np.ndarray:
        return rietveld.accumulate(
            x,
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            support_fwhm=support,
        ).y

    finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
    active = np.abs(x - parameters[0, 0]) < support * parameters[0, 2] - 1e-10
    np.testing.assert_allclose(
        baseline.jacobian[0, parameter, active],
        finite_difference[active],
        rtol=3e-7,
        atol=2e-7,
    )


def test_support_is_inclusive_and_exact() -> None:
    result = rietveld.accumulate(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [0.0],
        [1.0],
        [1.0],
        [0.5],
        support_fwhm=1.0,
    )
    np.testing.assert_array_equal(result.y[[0, 4]], 0.0)
    assert np.all(result.y[1:4] > 0.0)
    np.testing.assert_array_equal(result.jacobian[0, :, [0, 4]], 0.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([0.0, 0.0], [0.0], [1.0], [1.0], [0.5]), "strictly increasing"),
        (([0.0], [0.0], [1.0], [0.0], [0.5]), "non-positive FWHM"),
        (([0.0], [0.0], [1.0], [1.0], [1.1]), "eta outside"),
    ],
)
def test_invalid_inputs_fail_at_python_boundary(
    arguments: tuple[list[float], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rietveld.accumulate(*arguments)
