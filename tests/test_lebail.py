from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
import pytest
from phasesmith.refinement import AffineConstraint, TerminationReason, lebail


def instrument() -> phasesmith.ConstantWavelengthInstrument:
    return phasesmith.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=2.0e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def phase(
    phase_id: str,
    positions: np.ndarray,
    intensities: np.ndarray,
) -> phasesmith.Phase:
    count = positions.size
    d_spacing = instrument().wavelength_angstrom / (2.0 * np.sin(np.deg2rad(positions / 2.0)))
    reflections = phasesmith.ReflectionBatch(
        [f"{phase_id}-{index}" for index in range(count)],
        np.column_stack(
            (
                np.arange(1, count + 1, dtype=np.int64),
                np.ones(count, dtype=np.int64),
                np.arange(count, dtype=np.int64) % 2,
            )
        ),
        d_spacing,
        positions,
        intensities,
    )
    return phasesmith.Phase(phase_id, phase_id.title(), reflections)


def observed_pattern(
    x: np.ndarray,
    truth: tuple[phasesmith.Phase, ...],
    *,
    background: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> phasesmith.PowderPattern:
    supplied_background = np.zeros_like(x) if background is None else background
    calculated = phasesmith.calculate_pattern(
        phasesmith.PowderPattern(x, background=supplied_background),
        instrument(),
        truth,
    )
    return phasesmith.PowderPattern(
        x,
        observed_y=calculated.y,
        background=supplied_background,
        uncertainty=uncertainty,
        mask=mask,
    )


def extracted(result: lebail.LeBailResult) -> np.ndarray:
    return np.asarray([item.integrated_intensity for item in result.intensities])


def test_isolated_reflections_are_recovered_from_a_short_one_call_script() -> None:
    x = np.linspace(20.0, 80.0, 6001)
    positions = np.array([30.0, 50.0, 70.0])
    truth = (phase("alpha", positions, np.array([10.0, 6.0, 3.0])),)
    starting = (phase("alpha", positions, np.ones(3)),)
    pattern = observed_pattern(x, truth, background=np.full(x.size, 0.2))
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), starting),
        lebail.LeBailOptions(max_iterations=10),
    )

    assert result.termination_reason is TerminationReason.CONVERGED
    np.testing.assert_allclose(extracted(result), [10.0, 6.0, 3.0], rtol=2e-12)
    np.testing.assert_allclose(result.calculation.y, pattern.observed_y, atol=3e-13)
    assert result.metrics.rwp < 2e-14
    assert result.calculation.reflection_keys == (
        ("alpha", "alpha-0"),
        ("alpha", "alpha-1"),
        ("alpha", "alpha-2"),
    )
    assert result.calculation.phase_components[0].phase_id == "alpha"
    np.testing.assert_allclose(
        result.calculation.phase_components[0].y,
        result.calculation.profile_y,
        atol=2e-15,
    )


def test_overlapping_reflections_converge_nonnegative_with_nonuniform_uncertainty() -> None:
    x = np.linspace(38.0, 43.0, 5001)
    positions = np.array([40.0, 40.06, 41.4])
    truth_values = np.array([9.0, 4.0, 6.0])
    truth = (phase("alpha", positions, truth_values),)
    starting = (phase("alpha", positions, np.array([2.0, 7.0, 1.0])),)
    uncertainty = 0.5 + 0.01 * (x - x[0])
    pattern = observed_pattern(x, truth, uncertainty=uncertainty)
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), starting),
        lebail.LeBailOptions(max_iterations=100, intensity_tolerance=2e-7),
    )

    assert np.all(extracted(result) >= 0.0)
    np.testing.assert_allclose(extracted(result), truth_values, rtol=3e-5, atol=2e-5)
    assert result.metrics.rwp < 2e-7
    assert len(result.history) <= 100


def test_exactly_coincident_multiplet_preserves_starting_partition_and_reports_rank() -> None:
    x = np.linspace(39.0, 41.0, 4001)
    positions = np.array([40.0, 40.0])
    truth = (phase("alpha", positions, np.array([4.0, 8.0])),)
    starting = (phase("alpha", positions, np.array([1.0, 2.0])),)
    result = lebail.refine(
        lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting),
        lebail.LeBailOptions(max_iterations=10),
    )

    np.testing.assert_allclose(extracted(result), [4.0, 8.0], rtol=2e-12)
    assert len(result.rank_deficient_groups) == 1
    group = result.rank_deficient_groups[0]
    assert group.reflection_keys == (("alpha", "alpha-0"), ("alpha", "alpha-1"))
    assert group.rank == 1


def test_shared_peak_across_phases_keeps_identity_and_reports_joint_rank() -> None:
    x = np.linspace(39.0, 41.0, 4001)
    alpha_truth = phase("alpha", np.array([40.0]), np.array([4.0]))
    beta_truth = phase("beta", np.array([40.0]), np.array([8.0]))
    alpha_start = phase("alpha", np.array([40.0]), np.array([1.0]))
    beta_start = phase("beta", np.array([40.0]), np.array([2.0]))
    pattern = observed_pattern(x, (alpha_truth, beta_truth))
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), (alpha_start, beta_start)),
        lebail.LeBailOptions(max_iterations=10),
    )

    np.testing.assert_allclose(extracted(result), [4.0, 8.0], rtol=2e-12)
    assert result.rank_deficient_groups == (
        lebail.CoincidentReflectionGroup((("alpha", "alpha-0"), ("beta", "beta-0")), rank=1),
    )


def test_zero_initialization_is_deterministic_and_absent_reflection_goes_to_zero() -> None:
    x = np.linspace(39.0, 43.0, 4001)
    truth = (phase("alpha", np.array([40.0, 42.0]), np.array([5.0, 7.0])),)
    starting = (phase("alpha", np.array([40.0, 42.0, 80.0]), np.zeros(3)),)
    input_data = lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting)
    first = lebail.initialize_intensities(input_data)
    second = lebail.initialize_intensities(input_data)
    np.testing.assert_array_equal(first, second)
    result = lebail.refine(input_data, lebail.LeBailOptions(max_iterations=20))
    np.testing.assert_allclose(extracted(result)[:2], [5.0, 7.0], rtol=3e-10)
    assert extracted(result)[2] == 0.0
    assert any("no included support" in warning for warning in result.history[-1].warnings)


def test_masked_region_does_not_contribute_to_extraction_or_metrics() -> None:
    x = np.linspace(39.0, 43.0, 4001)
    truth = (phase("alpha", np.array([40.0, 42.0]), np.array([5.0, 7.0])),)
    starting = (phase("alpha", np.array([40.0, 42.0]), np.ones(2)),)
    mask = x < 41.0
    pattern = observed_pattern(x, truth, mask=mask)
    result = lebail.refine(
        lebail.LeBailInput(pattern, instrument(), starting),
        lebail.LeBailOptions(max_iterations=20),
    )
    assert extracted(result)[0] == pytest.approx(5.0, rel=2e-10)
    assert extracted(result)[1] == 0.0
    assert np.count_nonzero(result.metrics.included) == np.count_nonzero(mask)


def test_analytical_profile_update_refines_a_reflection_position() -> None:
    x = np.linspace(39.0, 41.0, 4001)
    truth = (phase("alpha", np.array([40.0]), np.array([8.0])),)
    starting = (phase("alpha", np.array([39.985]), np.array([8.0])),)
    parameters = lebail.build_parameter_set(instrument(), starting, reflection_positions=True)
    result = lebail.refine(
        lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting, parameters),
        lebail.LeBailOptions(
            max_iterations=30,
            intensity_tolerance=1e-7,
            max_scaled_parameter_step=1.0,
        ),
    )
    refined_position = result.phases[0].reflections.two_theta_deg[0]
    assert refined_position == pytest.approx(40.0, abs=2e-7)
    assert result.metrics.rwp < 2e-6
    assert result.parameters is not None
    assert result.covariance is not None
    changes = tuple(change for item in result.history for change in item.parameter_changes)
    assert changes
    assert all(change.key == parameters.keys[0] for change in changes)
    assert changes[0].after - changes[0].before == pytest.approx(
        changes[0].scaled_change * parameters.specs[0].scale
    )


def test_analytical_profile_update_refines_an_instrument_width_parameter() -> None:
    x = np.linspace(20.0, 110.0, 9001)
    positions = np.array([30.0, 60.0, 100.0])
    truth = (phase("alpha", positions, np.array([8.0, 5.0, 3.0])),)
    starting = (phase("alpha", positions, np.array([8.0, 5.0, 3.0])),)
    starting_instrument = replace(instrument(), w_deg2=5.0e-4)
    parameters = lebail.build_parameter_set(
        starting_instrument,
        starting,
        instrument_parameters=("w_deg2",),
    )
    result = lebail.refine(
        lebail.LeBailInput(
            observed_pattern(x, truth),
            starting_instrument,
            starting,
            parameters,
        ),
        lebail.LeBailOptions(
            max_iterations=30,
            intensity_tolerance=1.0e-7,
            max_scaled_parameter_step=1.0,
        ),
    )

    assert result.instrument.w_deg2 == pytest.approx(instrument().w_deg2, abs=2.0e-8)
    assert result.metrics.rwp < 2.0e-5
    assert any(record.parameter_changes for record in result.history)


def test_parameter_constraints_are_applied_during_profile_updates() -> None:
    x = np.linspace(39.0, 43.0, 4001)
    truth = (phase("alpha", np.array([40.0, 42.0]), np.array([5.0, 7.0])),)
    starting = (phase("alpha", np.array([39.99, 41.99]), np.array([5.0, 7.0])),)
    parameters = lebail.build_parameter_set(instrument(), starting, reflection_positions=True)
    first_key, second_key = parameters.keys
    constraints = (AffineConstraint(second_key, first_key, 1.0, 2.0),)
    result = lebail.refine(
        lebail.LeBailInput(
            observed_pattern(x, truth),
            instrument(),
            starting,
            parameters,
            constraints,
        ),
        lebail.LeBailOptions(max_iterations=30, max_scaled_parameter_step=1.0),
    )
    refined = result.phases[0].reflections.two_theta_deg
    assert refined[1] - refined[0] == pytest.approx(2.0, abs=2e-13)
    np.testing.assert_allclose(refined, [40.0, 42.0], atol=3e-7)


def test_checkpoint_resume_is_identical_to_an_uninterrupted_run() -> None:
    x = np.linspace(38.0, 43.0, 5001)
    positions = np.array([40.0, 40.06, 41.4])
    truth = (phase("alpha", positions, np.array([9.0, 4.0, 6.0])),)
    starting = (phase("alpha", positions, np.array([2.0, 7.0, 1.0])),)
    input_data = lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting)
    partial = lebail.refine(
        input_data,
        lebail.LeBailOptions(max_iterations=3, min_iterations=2, intensity_tolerance=1e-12),
    )
    resumed = lebail.refine(
        input_data,
        lebail.LeBailOptions(max_iterations=30, min_iterations=2),
        checkpoint=partial.checkpoint,
    )
    uninterrupted = lebail.refine(
        input_data,
        lebail.LeBailOptions(max_iterations=30, min_iterations=2),
    )
    np.testing.assert_array_equal(extracted(resumed), extracted(uninterrupted))
    np.testing.assert_array_equal(resumed.calculation.y, uninterrupted.calculation.y)
    assert resumed.history == uninterrupted.history
    assert resumed.termination_reason is uninterrupted.termination_reason


def test_iterate_once_is_a_deterministic_lower_level_primitive() -> None:
    x = np.linspace(38.0, 43.0, 5001)
    positions = np.array([40.0, 40.06, 41.4])
    truth = (phase("alpha", positions, np.array([9.0, 4.0, 6.0])),)
    starting = (phase("alpha", positions, np.array([2.0, 7.0, 1.0])),)
    input_data = lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting)
    options = lebail.LeBailOptions(max_iterations=20, intensity_tolerance=1.0e-14)

    first = lebail.iterate_once(input_data, options)
    second = lebail.iterate_once(input_data, options, checkpoint=first.checkpoint)
    direct = lebail.refine(input_data, replace(options, min_iterations=2, max_iterations=2))

    assert len(first.history) == 1
    assert len(second.history) == 2
    np.testing.assert_array_equal(extracted(second), extracted(direct))
    np.testing.assert_array_equal(second.calculation.y, direct.calculation.y)
    assert second.history == direct.history


class RecordingOptimizer:
    def __init__(self) -> None:
        self.calls = 0

    def solve(
        self,
        residual: np.ndarray,
        jacobian: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        self.calls += 1
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        return np.clip(step, lower, upper)


def test_custom_optimizer_protocol_drives_profile_updates() -> None:
    x = np.linspace(39.0, 41.0, 4001)
    truth = (phase("alpha", np.array([40.0]), np.array([8.0])),)
    starting = (phase("alpha", np.array([39.985]), np.array([8.0])),)
    parameters = lebail.build_parameter_set(instrument(), starting, reflection_positions=True)
    optimizer = RecordingOptimizer()
    result = lebail.refine(
        lebail.LeBailInput(observed_pattern(x, truth), instrument(), starting, parameters),
        lebail.LeBailOptions(max_iterations=15, max_scaled_parameter_step=1.0),
        optimizer=optimizer,
    )

    assert optimizer.calls > 0
    assert result.phases[0].reflections.two_theta_deg[0] == pytest.approx(40.0, abs=3.0e-7)


def test_optional_scipy_adapter_imports_only_when_used() -> None:
    adapter = phasesmith.refinement.ScipyLeastSquaresAdapter()
    with pytest.raises(ImportError, match="optional dependency"):
        adapter.solve(
            np.ones(2),
            np.eye(2),
            np.full(2, -np.inf),
            np.full(2, np.inf),
        )


def test_le_bail_inputs_options_and_nonnegative_boundaries_are_validated() -> None:
    x = np.linspace(39.0, 41.0, 101)
    phase_list = [phase("alpha", np.array([40.0]), np.ones(1))]
    frozen_input = lebail.LeBailInput(
        phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
        instrument(),
        phase_list,
    )
    phase_list.clear()
    assert len(frozen_input.phases) == 1
    with pytest.raises(ValueError, match="observed_y"):
        lebail.LeBailInput(
            phasesmith.PowderPattern(x),
            instrument(),
            (phase("alpha", np.array([40.0]), np.ones(1)),),
        )
    with pytest.raises(ValueError, match="non-negative"):
        lebail.initialize_intensities(
            lebail.LeBailInput(
                phasesmith.PowderPattern(x, observed_y=np.zeros_like(x)),
                instrument(),
                (phase("alpha", np.array([40.0]), np.array([-1.0])),),
            )
        )
    with pytest.raises(ValueError, match="min_iterations"):
        lebail.LeBailOptions(max_iterations=1, min_iterations=2)
    with pytest.raises(ValueError, match="unsupported"):
        lebail.build_parameter_set(instrument(), (), instrument_parameters=("zero",))
    result = lebail.iterate_once(frozen_input)
    assert not result.checkpoint.intensities.flags.writeable


def test_phase_scale_is_available_as_a_typed_parameter_but_rank_is_reported() -> None:
    x = np.linspace(39.0, 41.0, 1001)
    starting = (phase("alpha", np.array([40.0]), np.array([8.0])),)
    parameters = lebail.build_parameter_set(instrument(), starting, phase_scales=True)
    assert parameters.keys == (lebail.phase_scale_key("alpha"),)
    changed = replace(starting[0], scale=0.8)
    values = lebail.build_parameter_set(instrument(), (changed,), phase_scales=True)
    assert values.specs[0].value == 0.8
    result = lebail.refine(
        lebail.LeBailInput(
            observed_pattern(x, starting),
            instrument(),
            (changed,),
            values,
        ),
        lebail.LeBailOptions(max_iterations=3, min_iterations=2),
    )
    assert result.covariance is None
    assert any(
        "not identifiable independently" in warning
        for record in result.history
        for warning in record.warnings
    )
