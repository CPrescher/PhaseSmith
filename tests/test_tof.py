from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith import reference


def instrument() -> phasesmith.TofInstrument:
    return phasesmith.TofInstrument(
        zero_us=-0.773346536757,
        difc_us_per_angstrom=5084.82763065,
        difa_us_per_angstrom2=-2.6304177486,
        difb_us_angstrom=1.25,
        alpha_coefficient=0.16,
        beta0_per_us=0.028,
        beta1_angstrom4_per_us=0.0012,
        betaq_angstrom2_per_us=0.003,
        sigma0_us2=1.5,
        sigma1_us2_per_angstrom2=15.1402867268,
        sigma2_us2_per_angstrom4=0.08,
        sigmaq_us2_per_angstrom=0.7,
        x_us_per_angstrom=0.8,
        y_us_per_angstrom2=0.15,
        z_us=1.2,
    )


def reference_parameters(d: np.ndarray) -> reference.ReferenceTofProfileParameters:
    model = instrument()
    return reference.tof_profile_parameters(
        d,
        zero_us=model.zero_us,
        difc_us_per_angstrom=model.difc_us_per_angstrom,
        difa_us_per_angstrom2=model.difa_us_per_angstrom2,
        difb_us_angstrom=model.difb_us_angstrom,
        alpha_coefficient=model.alpha_coefficient,
        beta0_per_us=model.beta0_per_us,
        beta1_angstrom4_per_us=model.beta1_angstrom4_per_us,
        betaq_angstrom2_per_us=model.betaq_angstrom2_per_us,
        sigma0_us2=model.sigma0_us2,
        sigma1_us2_per_angstrom2=model.sigma1_us2_per_angstrom2,
        sigma2_us2_per_angstrom4=model.sigma2_us2_per_angstrom4,
        sigmaq_us2_per_angstrom=model.sigmaq_us2_per_angstrom,
        x_us_per_angstrom=model.x_us_per_angstrom,
        y_us_per_angstrom2=model.y_us_per_angstrom2,
        z_us=model.z_us,
    )


def test_parameter_equations_match_independent_reference() -> None:
    d = np.array([0.45, 0.8, 1.3, 2.1, 3.6])
    actual = phasesmith.tof_profile_parameters(d, instrument())
    expected = reference_parameters(d)
    for field in expected.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=3e-15, atol=2e-13
        )


@pytest.mark.parametrize(
    ("alpha", "beta", "gaussian", "lorentzian"),
    [
        (0.08, 0.03, 22.0, 4.0),
        (0.4, 0.4, 8.0, 0.0),
        (2.5, 0.018, 35.0, 12.0),
    ],
)
def test_direct_profile_matches_high_order_independent_quadrature(
    alpha: float, beta: float, gaussian: float, lorentzian: float
) -> None:
    x = np.linspace(-250.0, 450.0, 701)
    actual = phasesmith.profile_tof(x, 17.0, alpha, beta, gaussian, lorentzian)
    expected = reference.profile_tof(
        x,
        17.0,
        alpha,
        beta,
        gaussian,
        lorentzian,
        quadrature_order=768,
    )
    for field in actual.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), rtol=2e-9, atol=5e-12
        )


@pytest.mark.parametrize(
    ("field", "step"),
    [
        ("position_us", 2e-5),
        ("alpha_per_us", 2e-7),
        ("beta_per_us", 2e-7),
        ("gaussian_fwhm_us", 2e-5),
        ("lorentzian_fwhm_us", 2e-5),
    ],
)
def test_direct_profile_derivatives_match_centered_differences(field: str, step: float) -> None:
    inputs = {
        "position_us": 5000.0,
        "alpha_per_us": 0.08,
        "beta_per_us": 0.03,
        "gaussian_fwhm_us": 22.0,
        "lorentzian_fwhm_us": 4.0,
    }
    x = np.linspace(4875.0, 5225.0, 351)
    actual = phasesmith.profile_tof(x, **inputs)
    plus = inputs | {field: inputs[field] + step}
    minus = inputs | {field: inputs[field] - step}
    finite = (
        phasesmith.profile_tof(x, **plus).value - phasesmith.profile_tof(x, **minus).value
    ) / (2.0 * step)
    derivative_name = {
        "position_us": "d_position",
        "alpha_per_us": "d_alpha",
        "beta_per_us": "d_beta",
        "gaussian_fwhm_us": "d_gaussian_fwhm",
        "lorentzian_fwhm_us": "d_lorentzian_fwhm",
    }[field]
    np.testing.assert_allclose(getattr(actual, derivative_name), finite, rtol=3e-7, atol=2e-10)


def test_normalization_centroid_and_asymmetry_moments() -> None:
    alpha = 0.08
    beta = 0.03
    tail_log = 20.0
    x = np.linspace(-300.0, 800.0, 44_001)
    y = phasesmith.profile_tof(x, 0.0, alpha, beta, 10.0, 0.0, tail_log=tail_log).value
    area = np.trapezoid(y, x)
    centroid = np.trapezoid(x * y, x) / area
    exponential_mean = (1.0 - (tail_log + 1.0) * np.exp(-tail_log)) / (1.0 - np.exp(-tail_log))
    left_fraction = beta / (alpha + beta)
    right_fraction = alpha / (alpha + beta)
    expected_centroid = exponential_mean * (-left_fraction / alpha + right_fraction / beta)
    third_moment = np.trapezoid((x - centroid) ** 3 * y, x) / area

    assert area == pytest.approx(1.0, rel=2e-5)
    assert centroid == pytest.approx(expected_centroid, abs=3e-4)
    assert third_moment > 0.0


def test_equal_rates_produce_a_symmetric_profile() -> None:
    x = np.linspace(-150.0, 150.0, 1201)
    result = phasesmith.profile_tof(x, 0.0, 0.08, 0.08, 12.0, 3.0)
    np.testing.assert_allclose(result.value, result.value[::-1], rtol=0.0, atol=2e-17)


@pytest.mark.parametrize("collapsed_side", ["left", "right"])
def test_one_sided_exponential_limits(collapsed_side: str) -> None:
    """A rate tending to infinity removes that exponential tail."""

    x = np.linspace(-250.0, 250.0, 1001)
    finite_rate = 0.04
    collapsed_rate = 1.0e8
    alpha, beta = (
        (collapsed_rate, finite_rate) if collapsed_side == "left" else (finite_rate, collapsed_rate)
    )
    actual = phasesmith.profile_tof(x, 0.0, alpha, beta, 12.0, 3.0).value

    nodes, weights = np.polynomial.legendre.leggauss(768)
    tail_log = 20.0
    t = 0.5 * tail_log * (nodes + 1.0)
    quadrature = 0.5 * tail_log * weights * np.exp(-t) / (1.0 - np.exp(-tail_log))
    shifts = t / finite_rate
    delta = x[:, None] - shifts[None, :]
    if collapsed_side == "right":
        delta = x[:, None] + shifts[None, :]
    expected = reference.profile_tch(delta, 12.0, 3.0).value @ quadrature

    np.testing.assert_allclose(actual, expected, rtol=2.0e-8, atol=2.0e-11)


def test_fused_accumulation_matches_support_limited_reference() -> None:
    d = np.array([1.58, 1.62, 1.69])
    intensities = np.array([9.0, 5.5, 13.0])
    parameters = phasesmith.tof_profile_parameters(d, instrument())
    x = np.linspace(7600.0, 9400.0, 901)
    support_fwhm = 12.0
    tail_log = 20.0
    actual = phasesmith.accumulate_tof(
        x,
        d,
        intensities,
        instrument(),
        support_fwhm=support_fwhm,
        tail_log=tail_log,
        jacobian_layout="dense",
    )
    expected = np.zeros_like(x)
    for index, intensity in enumerate(intensities):
        base_radius = support_fwhm * parameters.total_fwhm_us[index]
        left = (
            parameters.position_us[index] - base_radius - tail_log / parameters.alpha_per_us[index]
        )
        right = (
            parameters.position_us[index] + base_radius + tail_log / parameters.beta_per_us[index]
        )
        active = (x >= left) & (x <= right)
        evaluated = reference.profile_tof(
            x[active],
            parameters.position_us[index],
            parameters.alpha_per_us[index],
            parameters.beta_per_us[index],
            parameters.gaussian_fwhm_us[index],
            parameters.lorentzian_fwhm_us[index],
            tail_log=tail_log,
            quadrature_order=768,
            base_radius_us=base_radius,
        )
        expected[active] += intensity * evaluated.value

    assert actual.derivatives.local_parameter_names == phasesmith.TOF_LOCAL_PARAMETER_ORDER
    assert actual.derivatives.global_parameter_names == phasesmith.TOF_GLOBAL_PARAMETER_ORDER
    np.testing.assert_allclose(actual.y, expected, rtol=2e-9, atol=2e-12)


def test_tof_execution_policy_is_bitwise_deterministic() -> None:
    d = np.linspace(0.5, 2.8, 36)
    intensities = np.linspace(3.0, 13.0, d.size)
    x = np.linspace(1_000.0, 17_000.0, 8_001)
    serial = phasesmith.accumulate_tof(
        x,
        d,
        intensities,
        instrument(),
        execution=phasesmith.ExecutionPolicy(threads=1),
    )
    parallel = phasesmith.accumulate_tof(
        x,
        d,
        intensities,
        instrument(),
        execution=phasesmith.ExecutionPolicy(threads=2),
    )
    np.testing.assert_array_equal(parallel.y, serial.y)
    np.testing.assert_array_equal(
        parallel.derivatives.local.values,
        serial.derivatives.local.values,
    )
    np.testing.assert_array_equal(
        parallel.derivatives.global_jacobian,
        serial.derivatives.global_jacobian,
    )


GLOBAL_FIELDS = {
    "zero": "zero_us",
    "difc": "difc_us_per_angstrom",
    "difa": "difa_us_per_angstrom2",
    "difb": "difb_us_angstrom",
    "alpha": "alpha_coefficient",
    "beta0": "beta0_per_us",
    "beta1": "beta1_angstrom4_per_us",
    "betaq": "betaq_angstrom2_per_us",
    "sigma0": "sigma0_us2",
    "sigma1": "sigma1_us2_per_angstrom2",
    "sigma2": "sigma2_us2_per_angstrom4",
    "sigmaq": "sigmaq_us2_per_angstrom",
    "x": "x_us_per_angstrom",
    "y": "y_us_per_angstrom2",
    "z": "z_us",
}


@pytest.mark.parametrize("parameter", phasesmith.TOF_GLOBAL_PARAMETER_ORDER)
def test_global_derivatives_match_centered_differences(parameter: str) -> None:
    d = np.array([1.66, 1.70])
    intensities = np.array([12.0, 7.0])
    x = np.linspace(8100.0, 9000.0, 451)
    model = instrument()
    baseline = phasesmith.accumulate_tof(x, d, intensities, model, support_fwhm=100.0)
    field = GLOBAL_FIELDS[parameter]
    value = getattr(model, field)
    step = max(abs(value) * 2e-6, 2e-7)
    plus = replace(model, **{field: value + step})
    minus = replace(model, **{field: value - step})
    finite = (
        phasesmith.accumulate_tof(x, d, intensities, plus, support_fwhm=100.0).y
        - phasesmith.accumulate_tof(x, d, intensities, minus, support_fwhm=100.0).y
    ) / (2.0 * step)
    row = phasesmith.TOF_GLOBAL_PARAMETER_ORDER.index(parameter)
    scale = max(1.0, float(np.max(np.abs(finite))))
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row],
        finite,
        rtol=2e-5,
        atol=3e-8 * scale,
    )


@pytest.mark.parametrize("parameter", phasesmith.TOF_LOCAL_PARAMETER_ORDER)
def test_local_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(8100.0, 9000.0, 451)
    d = np.array([1.68])
    intensities = np.array([12.0])
    baseline = phasesmith.accumulate_tof(
        x,
        d,
        intensities,
        instrument(),
        support_fwhm=100.0,
        jacobian_layout="dense",
    )
    plus_d = d.copy()
    minus_d = d.copy()
    plus_i = intensities.copy()
    minus_i = intensities.copy()
    step = 2e-6
    if parameter == "d_spacing":
        plus_d[0] += step
        minus_d[0] -= step
    else:
        plus_i[0] += step
        minus_i[0] -= step
    plus = phasesmith.accumulate_tof(x, plus_d, plus_i, instrument(), support_fwhm=100.0).y
    minus = phasesmith.accumulate_tof(x, minus_d, minus_i, instrument(), support_fwhm=100.0).y
    finite = (plus - minus) / (2.0 * step)
    column = phasesmith.TOF_LOCAL_PARAMETER_ORDER.index(parameter)
    np.testing.assert_allclose(baseline.jacobian[0, column], finite, rtol=2e-5, atol=2e-8)


def test_support_boundaries_are_inclusive_and_coordinates_are_bin_centers() -> None:
    d = np.array([1.7])
    parameters = phasesmith.tof_profile_parameters(d, instrument())
    support_fwhm = 1.0
    tail_log = 3.0
    base_radius = support_fwhm * parameters.total_fwhm_us[0]
    left = parameters.position_us[0] - base_radius - tail_log / parameters.alpha_per_us[0]
    right = parameters.position_us[0] + base_radius + tail_log / parameters.beta_per_us[0]
    x = np.array([left - 1e-6, left, parameters.position_us[0], right, right + 1e-6])
    result = phasesmith.accumulate_tof(
        x, d, [1.0], instrument(), support_fwhm=support_fwhm, tail_log=tail_log
    )
    assert result.derivatives.local.starts.tolist() == [1]
    assert result.derivatives.local.offsets.tolist() == [0, 3]
    assert np.all(result.y[[0, 4]] == 0.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"difc_us_per_angstrom": 0.0}, "difc"),
        ({"alpha_coefficient": 0.0}, "alpha"),
        ({"beta0_per_us": -1.0}, "beta"),
        ({"sigma0_us2": -1e6}, "Gaussian variance"),
        ({"z_us": -1e6}, "Lorentzian FWHM"),
    ],
)
def test_invalid_instrument_or_derived_domains_are_clear(
    change: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        model = replace(instrument(), **change)
        phasesmith.tof_profile_parameters([1.7], model)


def test_batch_boundary_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        phasesmith.tof_profile_parameters([0.0], instrument())
    with pytest.raises(ValueError, match="equal length"):
        phasesmith.accumulate_tof([1.0, 2.0], [1.5], [], instrument())
    with pytest.raises(ValueError, match="strictly increasing"):
        phasesmith.accumulate_tof([1.0, 1.0], [1.5], [1.0], instrument())
    with pytest.raises(ValueError, match="tail_log"):
        phasesmith.profile_tof([0.0], 0.0, 1.0, 1.0, 1.0, 0.0, tail_log=0.0)
