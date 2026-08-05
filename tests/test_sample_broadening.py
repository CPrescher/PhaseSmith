from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld import reference


def instrument(wavelength: float = 1.54056) -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=wavelength,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def reflections(positions: np.ndarray | None = None) -> rietveld.ReflectionGeometryBatch:
    if positions is None:
        positions = np.array([25.0, 55.0, 105.0])
    wavelength = instrument().wavelength_angstrom
    d_spacing = wavelength / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    return rietveld.ReflectionGeometryBatch(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        d_spacing,
        positions,
        [11.0, 7.0, 4.0],
    )


def composite(size: float = 48.0, strain: float = 6.0e-4) -> rietveld.CompositePhysicsProvider:
    return rietveld.CompositePhysicsProvider(
        (
            rietveld.IsotropicSizeBroadening(size),
            rietveld.IsotropicMicrostrainBroadening(strain),
        )
    )


def test_isotropic_models_follow_documented_width_equations() -> None:
    context = rietveld.PhysicsContext(reflections(), instrument())
    size = rietveld.IsotropicSizeBroadening(50.0, shape_factor=0.9).evaluate(context)
    strain = rietveld.IsotropicMicrostrainBroadening(4.0e-4).evaluate(context)
    theta = np.deg2rad(context.reflections.two_theta_deg / 2.0)
    expected_size = (
        np.rad2deg(1.0)
        * 0.9
        * context.instrument.wavelength_angstrom
        / (10.0 * 50.0 * np.cos(theta))
    )
    expected_variance = (np.rad2deg(1.0) * 2.0 * 4.0e-4 * np.tan(theta)) ** 2
    np.testing.assert_allclose(size.lorentzian_fwhm_deg, expected_size, rtol=2e-15)
    np.testing.assert_allclose(strain.gaussian_variance_deg2, expected_variance, rtol=2e-15)
    np.testing.assert_allclose(
        size.d_lorentzian_fwhm_d_parameters[0], -expected_size / 50.0, rtol=2e-15
    )
    np.testing.assert_allclose(
        strain.d_gaussian_variance_d_parameters[0], expected_variance * 2.0 / 4.0e-4
    )


def test_infinite_size_and_zero_strain_recover_instrument_values_exactly() -> None:
    x = np.linspace(20.0, 110.0, 9_001)
    batch = reflections()
    baseline = rietveld.calculate_cw_pattern(x, batch, instrument(), jacobian_layout="dense")
    disabled = rietveld.calculate_cw_pattern(
        x,
        batch,
        instrument(),
        physics=composite(size=np.inf, strain=0.0),
        jacobian_layout="dense",
    )
    np.testing.assert_array_equal(disabled.y, baseline.y)
    np.testing.assert_array_equal(disabled.jacobian, baseline.jacobian)
    np.testing.assert_array_equal(
        disabled.derivatives.global_jacobian[:5], baseline.derivatives.global_jacobian
    )
    np.testing.assert_array_equal(disabled.derivatives.global_jacobian[5:], 0.0)


def test_native_sample_accumulation_matches_independent_reference() -> None:
    x = np.linspace(23.0, 107.0, 8_401)
    batch = reflections()
    contribution = composite().evaluate(rietveld.PhysicsContext(batch, instrument()))
    actual = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=composite(), jacobian_layout="dense"
    )
    expected_y, expected_local, expected_global = reference.accumulate_cw_contributions(
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
    np.testing.assert_allclose(actual.y, expected_y, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(actual.jacobian, expected_local, rtol=8e-12, atol=2e-10)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian, expected_global, rtol=8e-12, atol=2e-10
    )


def test_randomized_native_provider_batches_match_reference_deterministically() -> None:
    rng = np.random.default_rng(20_260_805)
    positions = np.sort(rng.uniform(12.0, 138.0, 17))
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    batch = rietveld.ReflectionGeometryBatch(
        np.column_stack((np.arange(17), np.ones(17), np.zeros(17))).astype(np.int64),
        d_spacing,
        positions,
        rng.uniform(-3.0, 20.0, 17),
    )
    x = np.linspace(10.0, 140.0, 6_501)
    provider = composite(37.0, 8.0e-4)
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
    np.testing.assert_allclose(actual.y, expected[0], rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(actual.jacobian, expected[1], rtol=2e-11, atol=5e-10)
    np.testing.assert_allclose(
        actual.derivatives.global_jacobian, expected[2], rtol=2e-11, atol=5e-10
    )
    repeated = rietveld.calculate_cw_pattern(x, batch, instrument(), physics=provider)
    np.testing.assert_array_equal(repeated.y, actual.y)


def test_sample_broadening_preserves_area_and_symmetric_centroid() -> None:
    position = 55.0
    wavelength = instrument().wavelength_angstrom
    batch = rietveld.ReflectionGeometryBatch(
        [[1, 1, 0]],
        [wavelength / (2.0 * np.sin(np.deg2rad(position / 2.0)))],
        [position],
        [13.0],
    )
    x = np.linspace(30.0, 80.0, 200_001)
    support = 100.0
    result = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=composite(), support_fwhm=support
    )
    area = np.trapezoid(result.y, x)
    centroid = np.trapezoid(x * result.y, x) / area
    contribution = composite().evaluate(rietveld.PhysicsContext(batch, instrument()))
    base = rietveld.cw_profile_parameters([position], instrument())
    gaussian = reference.GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(
        base.gaussian_variance_deg2[0] + contribution.gaussian_variance_deg2[0]
    )
    lorentzian = base.lorentzian_fwhm_deg[0] + contribution.lorentzian_fwhm_deg[0]
    shape = reference.tch_shape_from_fwhm(float(gaussian), float(lorentzian))
    retained = shape.eta * 2.0 / math.pi * math.atan(2.0 * support) + (1.0 - shape.eta) * math.erf(
        2.0 * math.sqrt(math.log(2.0)) * support
    )
    assert area == pytest.approx(13.0 * retained, rel=8e-6)
    assert centroid == pytest.approx(position, abs=2e-10)


@pytest.mark.parametrize("parameter", ["size", "strain"])
def test_sample_parameter_derivatives_match_centered_differences(parameter: str) -> None:
    x = np.linspace(54.0, 56.0, 2_001)
    batch = reflections()
    size = 48.0
    strain = 6.0e-4
    baseline = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=composite(size, strain)
    )
    if parameter == "size":
        step = 2.0e-4
        plus = composite(size + step, strain)
        minus = composite(size - step, strain)
        name = "isotropic_size.crystallite_size_nm"
    else:
        step = 2.0e-8
        plus = composite(size, strain + step)
        minus = composite(size, strain - step)
        name = "isotropic_microstrain.rms"
    plus_y = rietveld.calculate_cw_pattern(x, batch, instrument(), physics=plus).y
    minus_y = rietveld.calculate_cw_pattern(x, batch, instrument(), physics=minus).y
    finite_difference = (plus_y - minus_y) / (2.0 * step)
    row = baseline.derivatives.global_parameter_names.index(name)
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row],
        finite_difference,
        rtol=7e-6,
        atol=2e-7 * max(1.0, float(np.max(np.abs(finite_difference)))),
    )


def test_position_and_intensity_derivatives_include_provider_chains() -> None:
    x = np.linspace(54.0, 56.0, 2_001)
    batch = reflections()
    baseline = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=composite(), jacobian_layout="dense"
    )
    for parameter, column, step in (("position", 1, 1.0e-5), ("intensity", 0, 1.0e-6)):
        plus_positions = batch.two_theta_deg.copy()
        minus_positions = batch.two_theta_deg.copy()
        plus_intensities = batch.base_integrated_intensity.copy()
        minus_intensities = batch.base_integrated_intensity.copy()
        if parameter == "position":
            plus_positions[1] += step
            minus_positions[1] -= step
        else:
            plus_intensities[1] += step
            minus_intensities[1] -= step
        plus_batch = rietveld.ReflectionGeometryBatch(
            batch.hkl, batch.d_spacing_angstrom, plus_positions, plus_intensities
        )
        minus_batch = rietveld.ReflectionGeometryBatch(
            batch.hkl, batch.d_spacing_angstrom, minus_positions, minus_intensities
        )
        plus = rietveld.calculate_cw_pattern(x, plus_batch, instrument(), physics=composite()).y
        minus = rietveld.calculate_cw_pattern(x, minus_batch, instrument(), physics=composite()).y
        finite_difference = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(
            baseline.jacobian[1, column], finite_difference, rtol=8e-6, atol=2e-6
        )


class ExternalQuadraticBroadening:
    """External-style provider used to prove the public protocol."""

    descriptor = rietveld.ProviderDescriptor("example.quadratic-broadening", "0.1")

    def __init__(self, amplitude: float) -> None:
        self.amplitude = amplitude
        self.call_count = 0

    def evaluate(self, context: rietveld.PhysicsContext) -> rietveld.PhysicsContribution:
        self.call_count += 1
        normalized = context.reflections.two_theta_deg / 100.0
        variance = self.amplitude * normalized**2
        count = context.reflections.reflection_count
        zeros = np.zeros(count)
        return rietveld.PhysicsContribution(
            gaussian_variance_deg2=variance,
            lorentzian_fwhm_deg=zeros,
            intensity_multiplier=np.ones(count),
            d_gaussian_variance_d_position=2.0 * self.amplitude * normalized / 100.0,
            d_lorentzian_fwhm_d_position=zeros,
            d_intensity_multiplier_d_position=zeros,
            parameter_names=("example.amplitude",),
            d_gaussian_variance_d_parameters=(normalized**2)[None, :],
            d_lorentzian_fwhm_d_parameters=zeros[None, :],
            d_intensity_multiplier_d_parameters=zeros[None, :],
        )


def test_external_provider_is_called_once_and_uses_native_derivative_path() -> None:
    x = np.linspace(54.0, 56.0, 2_001)
    batch = reflections()
    amplitude = 3.0e-4
    provider = ExternalQuadraticBroadening(amplitude)
    baseline = rietveld.calculate_cw_pattern(x, batch, instrument(), physics=provider)
    assert provider.call_count == 1
    step = 1.0e-8
    plus = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=ExternalQuadraticBroadening(amplitude + step)
    ).y
    minus = rietveld.calculate_cw_pattern(
        x, batch, instrument(), physics=ExternalQuadraticBroadening(amplitude - step)
    ).y
    finite_difference = (plus - minus) / (2.0 * step)
    row = baseline.derivatives.global_parameter_names.index("example.amplitude")
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row], finite_difference, rtol=6e-6, atol=3e-6
    )


def test_provider_boundary_rejects_version_shape_and_domain_errors() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        rietveld.ProviderDescriptor(123, "1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="incompatible"):
        bad = ExternalQuadraticBroadening(1.0e-4)
        bad.descriptor = replace(bad.descriptor, api_version=99)
        rietveld.calculate_cw_pattern([25.0], reflections(), instrument(), physics=bad)
    with pytest.raises(ValueError, match="non-negative"):
        rietveld.PhysicsContribution(
            gaussian_variance_deg2=[-1.0],
            lorentzian_fwhm_deg=[0.0],
            intensity_multiplier=[1.0],
            d_gaussian_variance_d_position=[0.0],
            d_lorentzian_fwhm_d_position=[0.0],
            d_intensity_multiplier_d_position=[0.0],
            parameter_names=(),
            d_gaussian_variance_d_parameters=np.empty((0, 1)),
            d_lorentzian_fwhm_d_parameters=np.empty((0, 1)),
            d_intensity_multiplier_d_parameters=np.empty((0, 1)),
        )
    overflow_batch = rietveld.ReflectionGeometryBatch([[1, 0, 0]], [3.0], [25.0], [1.0e308])
    neutral = rietveld.PhysicsContribution.neutral(1)
    amplified = rietveld.PhysicsContribution(
        gaussian_variance_deg2=neutral.gaussian_variance_deg2,
        lorentzian_fwhm_deg=neutral.lorentzian_fwhm_deg,
        intensity_multiplier=[2.0],
        d_gaussian_variance_d_position=neutral.d_gaussian_variance_d_position,
        d_lorentzian_fwhm_d_position=neutral.d_lorentzian_fwhm_d_position,
        d_intensity_multiplier_d_position=neutral.d_intensity_multiplier_d_position,
        parameter_names=(),
        d_gaussian_variance_d_parameters=np.empty((0, 1)),
        d_lorentzian_fwhm_d_parameters=np.empty((0, 1)),
        d_intensity_multiplier_d_parameters=np.empty((0, 1)),
    )
    with pytest.raises(ValueError, match="effective_intensity"):
        rietveld.accumulate_cw_contributions(
            [25.0],
            overflow_batch.two_theta_deg,
            overflow_batch.base_integrated_intensity,
            instrument(),
            amplified,
        )


def test_bragg_consistent_d_spacing_tracks_angular_broadening() -> None:
    context = rietveld.PhysicsContext(reflections(), instrument())
    size = rietveld.IsotropicSizeBroadening(50.0).evaluate(context)
    strain = rietveld.IsotropicMicrostrainBroadening(5.0e-4).evaluate(context)
    assert np.all(np.diff(context.reflections.d_spacing_angstrom) < 0.0)
    assert np.all(np.diff(size.lorentzian_fwhm_deg) > 0.0)
    assert np.all(np.diff(strain.gaussian_variance_deg2) > 0.0)
