from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.54056,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def reflection_batch(
    prefix: str,
    positions: np.ndarray,
    intensities: np.ndarray,
) -> phasesmith.ReflectionBatch:
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    count = positions.size
    return phasesmith.ReflectionBatch(
        [f"{prefix}-{index}" for index in range(count)],
        np.column_stack(
            (
                np.arange(1, count + 1, dtype=np.int64),
                np.ones(count, dtype=np.int64),
                np.arange(count, dtype=np.int64) % 3,
            )
        ),
        d_spacing,
        positions,
        intensities,
    )


def phases() -> tuple[phasesmith.Phase, phasesmith.Phase]:
    alpha = phasesmith.Phase(
        "alpha",
        "Alpha phase",
        reflection_batch("a", np.array([24.0, 42.0, 63.0]), np.array([10.0, 7.0, 4.0])),
        scale=1.25,
        physics=phasesmith.IsotropicSizeBroadening(70.0),
    )
    beta = phasesmith.Phase(
        "beta",
        "Beta phase",
        reflection_batch("b", np.array([33.0, 42.0, 78.0]), np.array([8.0, 3.0, 6.0])),
        scale=0.65,
        physics=phasesmith.IsotropicMicrostrainBroadening(3.0e-4),
    )
    return alpha, beta


def test_single_phase_is_exactly_equivalent_to_low_level_cw() -> None:
    x = np.linspace(20.0, 80.0, 6_001)
    reflections = reflection_batch("r", np.array([24.0, 42.0, 63.0]), np.array([10.0, 7.0, 4.0]))
    phase = phasesmith.Phase("phase-1", "Phase one", reflections)
    actual = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [phase])
    expected = phasesmith.accumulate_cw(
        x,
        reflections.geometry.two_theta_deg,
        reflections.integrated_intensity,
        instrument(),
    )
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.profile_y, expected.y)
    np.testing.assert_array_equal(
        actual.derivatives.global_jacobian[:5], expected.derivatives.global_jacobian
    )
    assert actual.derivatives.global_parameter_names[-1] == "phase[phase-1].scale"


def test_fused_multiphase_equals_separate_phase_profiles_and_adds_background_once() -> None:
    x = np.linspace(20.0, 82.0, 6_201)
    background = 0.3 + 0.002 * (x - x[0])
    pattern = phasesmith.PowderPattern(x, background=background)
    alpha, beta = phases()
    options = phasesmith.CalculationOptions(return_phase_components=True)
    fused = phasesmith.calculate_pattern(pattern, instrument(), [alpha, beta], options=options)
    alpha_only = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [alpha])
    beta_only = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [beta])
    np.testing.assert_allclose(
        fused.profile_y,
        alpha_only.profile_y + beta_only.profile_y,
        rtol=2e-16,
        atol=2e-14,
    )
    np.testing.assert_allclose(fused.y - fused.profile_y, background, rtol=0.0, atol=7e-15)
    np.testing.assert_allclose(fused.phase_y("alpha"), alpha_only.profile_y, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(fused.phase_y("beta"), beta_only.profile_y, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        fused.phase_y("alpha") + fused.phase_y("beta"),
        fused.profile_y,
        rtol=2e-16,
        atol=2e-14,
    )


def test_phase_and_reflection_labels_preserve_input_order() -> None:
    x = np.linspace(20.0, 82.0, 3_101)
    alpha, beta = phases()
    result = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [alpha, beta])
    assert result.reflection_keys == (
        ("alpha", "a-0"),
        ("alpha", "a-1"),
        ("alpha", "a-2"),
        ("beta", "b-0"),
        ("beta", "b-1"),
        ("beta", "b-2"),
    )
    np.testing.assert_array_equal(result.phase_offsets, [0, 3, 6])
    assert result.derivatives.global_parameter_names == (
        "u",
        "v",
        "w",
        "x",
        "y",
        "phase[alpha].scale",
        "phase[alpha].physics.isotropic_size.crystallite_size_nm",
        "phase[beta].scale",
        "phase[beta].physics.isotropic_microstrain.rms",
    )
    reversed_result = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x), instrument(), [beta, alpha]
    )
    assert reversed_result.reflection_keys[0] == ("beta", "b-0")
    np.testing.assert_allclose(result.y, reversed_result.y, rtol=2e-16, atol=2e-14)


@pytest.mark.parametrize("phase_index", [0, 1])
def test_phase_scale_derivatives_match_centered_differences(phase_index: int) -> None:
    x = np.linspace(20.0, 82.0, 6_201)
    original = phases()
    baseline = phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), original)
    step = 1.0e-6
    plus = list(original)
    minus = list(original)
    plus[phase_index] = replace(original[phase_index], scale=original[phase_index].scale + step)
    minus[phase_index] = replace(original[phase_index], scale=original[phase_index].scale - step)
    finite_difference = (
        phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), plus).y
        - phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), minus).y
    ) / (2.0 * step)
    name = f"phase[{original[phase_index].phase_id}].scale"
    row = baseline.derivatives.global_parameter_names.index(name)
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row], finite_difference, rtol=2e-9, atol=2e-8
    )


def test_zero_scale_phase_has_zero_profile_but_nonzero_scale_derivative() -> None:
    x = np.linspace(20.0, 70.0, 5_001)
    alpha = replace(phases()[0], scale=0.0)
    result = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x),
        instrument(),
        [alpha],
        options=phasesmith.CalculationOptions(return_phase_components=True),
    )
    np.testing.assert_array_equal(result.profile_y, 0.0)
    np.testing.assert_array_equal(result.phase_y("alpha"), 0.0)
    row = result.derivatives.global_parameter_names.index("phase[alpha].scale")
    assert np.max(np.abs(result.derivatives.global_jacobian[row])) > 0.0


def test_negative_reflection_intensities_cancel_at_duplicate_positions() -> None:
    x = np.linspace(40.0, 44.0, 4_001)
    positive = phasesmith.Phase(
        "positive",
        "Positive",
        reflection_batch("p", np.array([42.0]), np.array([5.0])),
    )
    negative = phasesmith.Phase(
        "negative",
        "Negative",
        reflection_batch("n", np.array([42.0]), np.array([-5.0])),
    )
    result = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x), instrument(), [positive, negative]
    )
    np.testing.assert_array_equal(result.profile_y, 0.0)


class CountingProvider:
    descriptor = phasesmith.ProviderDescriptor("test.counting", "1")

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: phasesmith.PhysicsContext) -> phasesmith.PhysicsContribution:
        self.calls += 1
        return phasesmith.IsotropicSizeBroadening(55.0).evaluate(context)


def test_prepared_pattern_evaluates_provider_once_and_reuses_flat_inputs() -> None:
    x = np.linspace(20.0, 70.0, 5_001)
    provider = CountingProvider()
    phase = replace(phases()[0], physics=provider)
    prepared = phasesmith.PreparedPattern(phasesmith.PowderPattern(x), instrument(), [phase])
    assert provider.calls == 1
    first = prepared.calculate()
    second = prepared.calculate()
    assert provider.calls == 1
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(
        first.derivatives.global_jacobian, second.derivatives.global_jacobian
    )


def test_stateless_multiphase_dispatches_one_native_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phasesmith.calculation as calculation

    calls = 0
    native_boundary = calculation.accumulate_cw_contributions

    def counted(*args: object, **kwargs: object) -> phasesmith.AccumulationResult:
        nonlocal calls
        calls += 1
        return native_boundary(*args, **kwargs)

    monkeypatch.setattr(calculation, "accumulate_cw_contributions", counted)
    x = np.linspace(20.0, 82.0, 3_101)
    calculation.calculate_pattern(phasesmith.PowderPattern(x), instrument(), phases())
    assert calls == 1


def test_phase_specific_provider_rows_are_prefixed_and_finite_differenced() -> None:
    x = np.linspace(20.0, 82.0, 6_201)
    alpha, beta = phases()
    size = 70.0
    baseline = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x), instrument(), [alpha, beta]
    )
    step = 2.0e-4
    plus = replace(alpha, physics=phasesmith.IsotropicSizeBroadening(size + step))
    minus = replace(alpha, physics=phasesmith.IsotropicSizeBroadening(size - step))
    finite_difference = (
        phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [plus, beta]).y
        - phasesmith.calculate_pattern(phasesmith.PowderPattern(x), instrument(), [minus, beta]).y
    ) / (2.0 * step)
    name = "phase[alpha].physics.isotropic_size.crystallite_size_nm"
    row = baseline.derivatives.global_parameter_names.index(name)
    np.testing.assert_allclose(
        baseline.derivatives.global_jacobian[row], finite_difference, rtol=7e-6, atol=2e-7
    )


def test_pattern_and_identity_validation_fail_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        phasesmith.PowderPattern([1.0, 1.0])
    with pytest.raises(ValueError, match="uncertainty"):
        phasesmith.PowderPattern([1.0, 2.0], uncertainty=[1.0, 0.0])
    with pytest.raises(ValueError, match="boolean"):
        phasesmith.PowderPattern([1.0, 2.0], mask=[0, 1])
    with pytest.raises(ValueError, match="reflection_ids"):
        phasesmith.ReflectionBatch(
            ["same", "same"], [[1, 0, 0], [0, 1, 0]], [1, 1], [20, 30], [1, 1]
        )
    alpha = phases()[0]
    with pytest.raises(ValueError, match="phase_id"):
        phasesmith.calculate_pattern(
            phasesmith.PowderPattern([20.0, 21.0]), instrument(), [alpha, alpha]
        )
    with pytest.raises(ValueError, match="phase scale"):
        replace(alpha, scale=-1.0)
    with pytest.raises(KeyError, match="no diagnostic"):
        phasesmith.calculate_pattern(
            phasesmith.PowderPattern([20.0, 21.0]), instrument(), [alpha]
        ).phase_y("alpha")
