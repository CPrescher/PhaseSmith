from __future__ import annotations

import math

import numpy as np
import phasesmith
import pytest
from phasesmith import reference


def test_package_exposes_installed_version() -> None:
    assert phasesmith.__version__ == "0.3.0"


def test_native_profile_matches_independent_numpy_reference() -> None:
    rng = np.random.default_rng(20260804)
    delta = rng.uniform(-3.0, 3.0, 2_000)
    for fwhm, eta in zip(rng.uniform(0.02, 1.5, 12), rng.uniform(0.0, 1.0, 12), strict=True):
        actual = phasesmith.profile(delta, fwhm, eta)
        expected = reference.profile(delta, fwhm, eta)
        np.testing.assert_allclose(actual.value, expected.value, rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(actual.d_delta, expected.d_delta, rtol=4e-15, atol=2e-14)
        np.testing.assert_allclose(actual.d_fwhm, expected.d_fwhm, rtol=4e-15, atol=2e-14)
        np.testing.assert_allclose(actual.d_eta, expected.d_eta, rtol=3e-15, atol=2e-15)


def test_fwhm_and_symmetry_contract() -> None:
    fwhm = 0.37
    for eta in np.linspace(0.0, 1.0, 9):
        result = phasesmith.profile([0.0, -fwhm / 2.0, fwhm / 2.0], fwhm, eta)
        np.testing.assert_allclose(result.value[1:], result.value[0] / 2.0, rtol=2e-15)
        assert result.d_delta[0] == 0.0
        assert result.value[1] == result.value[2]


def test_profile_rejects_non_finite_offsets_before_native_dispatch() -> None:
    with pytest.raises(ValueError, match="delta must contain only finite values"):
        phasesmith.profile([0.0, np.nan], 0.2, 0.5)


def test_support_integral_matches_analytic_truncation() -> None:
    position = 0.2
    intensity = 137.0
    fwhm = 0.08
    eta = 0.42
    support = 12.0
    radius = support * fwhm
    x = np.linspace(position - radius, position + radius, 200_001)
    result = phasesmith.accumulate(x, [position], [intensity], [fwhm], [eta], support_fwhm=support)

    gaussian_fraction = math.erf(2.0 * math.sqrt(math.log(2.0)) * support)
    lorentzian_fraction = 2.0 / math.pi * math.atan(2.0 * support)
    expected_integral = intensity * (eta * lorentzian_fraction + (1.0 - eta) * gaussian_fraction)
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

    actual = phasesmith.accumulate(x, positions, intensities, fwhms, etas, support_fwhm=7.37)
    expected_y, expected_jacobian = reference.accumulate(
        x, positions, intensities, fwhms, etas, support_fwhm=7.37
    )
    np.testing.assert_allclose(actual.y, expected_y, rtol=3e-15, atol=2e-13)
    assert isinstance(actual.jacobian, phasesmith.SupportJacobian)
    np.testing.assert_allclose(
        actual.jacobian.to_dense(x.size),
        expected_jacobian,
        rtol=4e-15,
        atol=2e-13,
    )
    assert phasesmith.PARAMETER_ORDER == ("intensity", "position", "fwhm", "eta")


@pytest.mark.parametrize("parameter", range(4), ids=phasesmith.PARAMETER_ORDER)
def test_accumulation_jacobian_matches_centered_finite_difference(parameter: int) -> None:
    x = np.linspace(-0.493, 0.507, 1_001)
    parameters = np.array([[0.013, 82.0, 0.071, 0.37]], dtype=np.float64)
    support = 5.321
    baseline = phasesmith.accumulate(
        x,
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
        support_fwhm=support,
        jacobian_layout="dense",
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
        return phasesmith.accumulate(
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
    result = phasesmith.accumulate(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [0.0],
        [1.0],
        [1.0],
        [0.5],
        support_fwhm=1.0,
    )
    np.testing.assert_array_equal(result.y[[0, 4]], 0.0)
    assert np.all(result.y[1:4] > 0.0)
    assert isinstance(result.jacobian, phasesmith.SupportJacobian)
    np.testing.assert_array_equal(result.jacobian.starts, [1])
    np.testing.assert_array_equal(result.jacobian.offsets, [0, 3])
    dense = result.jacobian.to_dense(result.y.size)
    np.testing.assert_array_equal(dense[0, :, [0, 4]], 0.0)


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
        phasesmith.accumulate(*arguments)


def test_dense_layout_is_explicit_compatibility_materialization() -> None:
    arguments = (
        np.linspace(-1.0, 1.0, 101),
        np.array([-0.3, 0.4]),
        np.array([2.0, 5.0]),
        np.array([0.1, 0.2]),
        np.array([0.25, 0.75]),
    )
    support = phasesmith.accumulate(*arguments, support_fwhm=3.0)
    dense = phasesmith.accumulate(*arguments, support_fwhm=3.0, jacobian_layout="dense")
    assert isinstance(support.jacobian, phasesmith.SupportJacobian)
    assert isinstance(dense.jacobian, np.ndarray)
    np.testing.assert_array_equal(support.y, dense.y)
    np.testing.assert_array_equal(support.jacobian.to_dense(arguments[0].size), dense.jacobian)
    assert dense.derivatives.local.values.size == support.jacobian.values.size


def test_sparse_memory_scales_with_active_support() -> None:
    x = np.linspace(0.0, 100.0, 10_001)
    positions = np.linspace(1.0, 99.0, 250)
    result = phasesmith.accumulate(
        x,
        positions,
        np.ones(positions.size),
        np.full(positions.size, 0.02),
        np.full(positions.size, 0.5),
        support_fwhm=2.0,
    )
    sparse = result.derivatives.local
    dense_derivative_elements = positions.size * len(phasesmith.PARAMETER_ORDER) * x.size
    assert sparse.values.size == sparse.active_sample_count * len(phasesmith.PARAMETER_ORDER)
    assert sparse.values.size < dense_derivative_elements // 500


@pytest.mark.parametrize("seed", [7, 41, 503, 8_191])
def test_randomized_support_reconstruction_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.001, 0.006, 731))
    positions = rng.uniform(x[0] - 0.2, x[-1] + 0.2, 23)
    intensities = rng.uniform(-50.0, 400.0, positions.size)
    fwhms = rng.uniform(0.004, 0.08, positions.size)
    etas = rng.uniform(0.0, 1.0, positions.size)
    support_fwhm = 3.719
    actual = phasesmith.accumulate(
        x,
        positions,
        intensities,
        fwhms,
        etas,
        support_fwhm=support_fwhm,
    )
    expected_starts = np.searchsorted(x, positions - support_fwhm * fwhms, side="left")
    expected_stops = np.searchsorted(x, positions + support_fwhm * fwhms, side="right")
    expected_offsets = np.concatenate(([0], np.cumsum(expected_stops - expected_starts)))
    expected_y, expected_jacobian = reference.accumulate(
        x,
        positions,
        intensities,
        fwhms,
        etas,
        support_fwhm=support_fwhm,
    )
    np.testing.assert_array_equal(actual.derivatives.local.starts, expected_starts)
    np.testing.assert_array_equal(actual.derivatives.local.offsets, expected_offsets)
    np.testing.assert_allclose(actual.y, expected_y, rtol=4e-15, atol=2e-12)
    np.testing.assert_allclose(
        actual.derivatives.local.to_dense(x.size),
        expected_jacobian,
        rtol=5e-15,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    ("x", "positions", "expected_starts", "expected_offsets"),
    [
        ([], [], [], [0]),
        ([0.0, 1.0], [10.0], [2], [0, 0]),
        ([0.0, 1.0, 2.0], [1.1], [1], [0, 1]),
    ],
)
def test_empty_outside_and_single_sample_supports(
    x: list[float],
    positions: list[float],
    expected_starts: list[int],
    expected_offsets: list[int],
) -> None:
    count = len(positions)
    result = phasesmith.accumulate(
        x,
        positions,
        np.ones(count),
        np.full(count, 0.1),
        np.full(count, 0.5),
        support_fwhm=1.0,
    )
    np.testing.assert_array_equal(result.derivatives.local.starts, expected_starts)
    np.testing.assert_array_equal(result.derivatives.local.offsets, expected_offsets)


def test_python_array_validation_and_layout_validation() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        phasesmith.accumulate([[0.0]], [], [], [], [])
    with pytest.raises(ValueError, match="real floating-point or integer"):
        phasesmith.accumulate(np.array([0.0 + 1.0j]), [], [], [], [])
    with pytest.raises(ValueError, match="equal length"):
        phasesmith.accumulate([0.0], [0.0], [], [1.0], [0.5])
    with pytest.raises(ValueError, match="jacobian_layout"):
        phasesmith.accumulate([0.0], [], [], [], [], jacobian_layout="csr")  # type: ignore[arg-type]


def test_support_to_dense_rejects_inconsistent_sample_count() -> None:
    result = phasesmith.accumulate([0.0, 1.0], [1.0], [1.0], [1.0], [0.5])
    with pytest.raises(ValueError, match="outside"):
        result.derivatives.local.to_dense(1)
