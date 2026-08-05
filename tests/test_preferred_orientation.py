from __future__ import annotations

import numpy as np
import pytest
import rietveld
from rietveld import reference


def instrument() -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.54056,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def reflections() -> rietveld.ReflectionGeometryBatch:
    positions = np.array([25.0, 42.0, 63.0, 88.0])
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    return rietveld.ReflectionGeometryBatch(
        [[0, 0, 1], [1, 0, 0], [1, 0, 1], [-1, 0, -1]],
        d_spacing,
        positions,
        [10.0, 8.0, 6.0, 4.0],
    )


def cubic_metric() -> rietveld.ReciprocalMetric:
    return rietveld.ReciprocalMetric.orthogonal(4.0, 4.0, 4.0)


def test_reciprocal_metric_angles_match_independent_cartesian_geometry() -> None:
    metric = rietveld.ReciprocalMetric(
        [[0.08, 0.012, -0.003], [0.012, 0.06, 0.007], [-0.003, 0.007, 0.04]]
    )
    hkl = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-2.0, 1.0, 3.0]])
    axis = np.array([0.4, -0.2, 1.3])
    actual = rietveld.reciprocal_angle_geometry(hkl, axis, metric)
    reciprocal_basis = np.linalg.cholesky(metric.matrix).T
    cartesian_hkl = hkl @ reciprocal_basis.T
    cartesian_axis = reciprocal_basis @ axis
    expected = (cartesian_hkl @ cartesian_axis) ** 2 / (
        np.einsum("ij,ij->i", cartesian_hkl, cartesian_hkl) * np.dot(cartesian_axis, cartesian_axis)
    )
    np.testing.assert_allclose(actual.cosine_squared, expected, rtol=5e-15, atol=5e-16)


def test_reciprocal_axis_derivatives_match_centered_differences() -> None:
    metric = rietveld.ReciprocalMetric(
        [[0.08, 0.012, -0.003], [0.012, 0.06, 0.007], [-0.003, 0.007, 0.04]]
    )
    hkl = np.array([[1.0, 2.0, -1.0], [-2.0, 1.0, 3.0]])
    axis = np.array([0.4, -0.2, 1.3])
    actual = rietveld.reciprocal_angle_geometry(hkl, axis, metric)
    step = 1.0e-6
    for coordinate in range(3):
        plus = axis.copy()
        minus = axis.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        finite_difference = (
            rietveld.reciprocal_angle_geometry(hkl, plus, metric).cosine_squared
            - rietveld.reciprocal_angle_geometry(hkl, minus, metric).cosine_squared
        ) / (2.0 * step)
        np.testing.assert_allclose(
            actual.d_cosine_squared_d_axis_hkl[:, coordinate],
            finite_difference,
            rtol=2e-9,
            atol=2e-10,
        )


def test_march_dollase_values_and_ratio_derivative_follow_equation() -> None:
    ratio = 0.72
    provider = rietveld.MarchDollasePreferredOrientation(ratio, (0, 0, 1), cubic_metric())
    contribution = provider.evaluate(rietveld.PhysicsContext(reflections(), instrument()))
    cosine_squared = np.array([1.0, 0.0, 0.5, 0.5])
    denominator = ratio**2 * cosine_squared + (1.0 - cosine_squared) / ratio
    expected = denominator ** (-1.5)
    expected_derivative = (
        -1.5
        * denominator ** (-2.5)
        * (2.0 * ratio * cosine_squared - (1.0 - cosine_squared) / ratio**2)
    )
    np.testing.assert_allclose(contribution.intensity_multiplier, expected, rtol=3e-15)
    np.testing.assert_allclose(
        contribution.d_intensity_multiplier_d_parameters[0], expected_derivative, rtol=3e-15
    )


def test_ratio_one_recovers_unmodified_profile_exactly() -> None:
    x = np.linspace(20.0, 95.0, 7_501)
    batch = reflections()
    baseline = rietveld.calculate_cw_pattern(x, batch, instrument(), jacobian_layout="dense")
    orientation = rietveld.MarchDollasePreferredOrientation(1.0, (0, 0, 1), cubic_metric())
    actual = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=orientation, jacobian_layout="dense"
    )
    np.testing.assert_array_equal(actual.y, baseline.y)
    np.testing.assert_array_equal(actual.jacobian, baseline.jacobian)
    np.testing.assert_array_equal(
        actual.derivatives.global_jacobian[:5], baseline.derivatives.global_jacobian
    )


def test_march_ratio_profile_derivative_matches_centered_difference() -> None:
    x = np.linspace(40.0, 65.0, 5_001)
    batch = reflections()
    ratio = 0.72

    def provider(value: float) -> rietveld.MarchDollasePreferredOrientation:
        return rietveld.MarchDollasePreferredOrientation(value, (0, 0, 1), cubic_metric())

    baseline = rietveld.calculate_cw_pattern(x, batch, instrument(), physics=provider(ratio))
    step = 1.0e-6
    finite_difference = (
        rietveld.calculate_cw_pattern(x, batch, instrument(), physics=provider(ratio + step)).y
        - rietveld.calculate_cw_pattern(x, batch, instrument(), physics=provider(ratio - step)).y
    ) / (2.0 * step)
    row = baseline.derivatives.global_parameter_names.index("march_dollase.ratio")
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row], finite_difference, rtol=3e-9, atol=2e-8
    )


def test_orientation_composes_with_size_and_strain_through_reference_path() -> None:
    x = np.linspace(20.0, 95.0, 7_501)
    batch = reflections()
    provider = rietveld.CompositePhysicsProvider(
        (
            rietveld.IsotropicSizeBroadening(60.0),
            rietveld.IsotropicMicrostrainBroadening(4.0e-4),
            rietveld.MarchDollasePreferredOrientation(1.25, (0, 0, 1), cubic_metric()),
        )
    )
    contribution = provider.evaluate(rietveld.PhysicsContext(batch, instrument()))
    actual = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=provider, jacobian_layout="dense"
    )
    expected = reference.accumulate_cw_contributions(
        x,
        batch.two_theta_deg,
        batch.base_integrated_intensity,
        gaussian_variance_deg2=contribution.gaussian_variance_deg2,
        lorentzian_fwhm_deg=contribution.lorentzian_fwhm_deg,
        intensity_multiplier=contribution.intensity_multiplier,
        d_gaussian_variance_d_position=contribution.d_gaussian_variance_d_position,
        d_lorentzian_fwhm_d_position=contribution.d_lorentzian_fwhm_d_position,
        d_intensity_multiplier_d_position=contribution.d_intensity_multiplier_d_position,
        d_gaussian_variance_d_parameters=contribution.d_gaussian_variance_d_parameters,
        d_lorentzian_fwhm_d_parameters=contribution.d_lorentzian_fwhm_d_parameters,
        d_intensity_multiplier_d_parameters=contribution.d_intensity_multiplier_d_parameters,
        u_deg2=instrument().u_deg2,
        v_deg2=instrument().v_deg2,
        w_deg2=instrument().w_deg2,
        x_deg=instrument().x_deg,
        y_deg=instrument().y_deg,
    )
    np.testing.assert_allclose(actual.y, expected[0], rtol=4e-13, atol=4e-13)
    np.testing.assert_allclose(actual.jacobian, expected[1], rtol=2e-11, atol=3e-10)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian, expected[2], rtol=2e-11, atol=3e-10
    )


def test_symmetry_related_signs_receive_identical_orientation_factors() -> None:
    contribution = rietveld.MarchDollasePreferredOrientation(
        1.4, (0, 0, 1), cubic_metric()
    ).evaluate(rietveld.PhysicsContext(reflections(), instrument()))
    assert contribution.intensity_multiplier[2] == contribution.intensity_multiplier[3]


@pytest.mark.parametrize("ratio", [0.0, -1.0, np.inf, np.nan])
def test_invalid_march_ratios_are_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match="march_ratio"):
        rietveld.MarchDollasePreferredOrientation(ratio, (0, 0, 1), cubic_metric())


def test_invalid_orientation_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        rietveld.MarchDollasePreferredOrientation(1.0, (0, 0, 0), cubic_metric())
    with pytest.raises(ValueError, match="zero reflection"):
        rietveld.reciprocal_angle_geometry([[0, 0, 0]], [0, 0, 1], cubic_metric())
    with pytest.raises(ValueError, match="positive definite"):
        rietveld.ReciprocalMetric([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
