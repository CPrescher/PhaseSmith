from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rietveld
from rietveld.refinement import (
    AffineConstraint,
    Bounds,
    ConstraintTransform,
    FixedConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualOptions,
    evaluate_residuals,
    jacobian_vector_product,
    transpose_jacobian_vector_product,
)


def parameter_set() -> tuple[ParameterSet, tuple[ParameterKey, ...]]:
    keys = (
        ParameterKey("instrument", "bank-1", "u"),
        ParameterKey("instrument", "bank-1", "v"),
        ParameterKey("instrument", "bank-1", "w"),
        ParameterKey("phase", "alpha", "scale"),
    )
    parameters = ParameterSet(
        [
            ParameterSpec(keys[0], 2.0e-4, "degree^2", Bounds(0.0, 1.0), 1.0e-4),
            ParameterSpec(keys[1], -1.0e-4, "degree^2", Bounds(-1.0, 1.0), 1.0e-4),
            ParameterSpec(keys[2], 1.2e-4, "degree^2", Bounds(0.0, 1.0), 1.0e-4),
            ParameterSpec(keys[3], 1.0, "dimensionless", Bounds(0.0, 10.0), 1.0),
        ]
    )
    return parameters, keys


def test_parameter_keys_sets_and_scaled_packing_are_deterministic() -> None:
    parameters, keys = parameter_set()
    transform = ConstraintTransform(parameters)
    np.testing.assert_array_equal(transform.pack(), [2.0, -1.0, 1.2, 1.0])
    unpacked = transform.unpack([2.1, -0.8, 1.3, 0.9])
    assert tuple(unpacked) == keys
    assert unpacked[keys[0]] == pytest.approx(2.1e-4)
    assert unpacked[keys[3]] == pytest.approx(0.9)
    replaced = parameters.replace_values({keys[3]: 2.0})
    assert replaced.specs[-1].value == 2.0
    assert parameters.specs[-1].value == 1.0


def test_fixed_and_ordered_affine_constraints_expand_without_strings() -> None:
    parameters, keys = parameter_set()
    transform = ConstraintTransform(
        parameters,
        [
            FixedConstraint(keys[1], -2.0e-4),
            AffineConstraint(keys[2], keys[0], multiplier=0.5, offset=1.0e-5),
        ],
    )
    assert transform.free_keys == (keys[0], keys[3])
    values = transform.unpack([3.0, 1.5])
    assert values[keys[0]] == pytest.approx(3.0e-4)
    assert values[keys[1]] == pytest.approx(-2.0e-4)
    assert values[keys[2]] == pytest.approx(1.6e-4)
    assert values[keys[3]] == pytest.approx(1.5)
    np.testing.assert_array_equal(
        transform.derivative_matrix(),
        [
            [1.0e-4, 0.0],
            [0.0, 0.0],
            [0.5e-4, 0.0],
            [0.0, 1.0],
        ],
    )
    assert not transform.derivative_matrix().flags.writeable


def test_constraint_cycles_duplicates_bounds_and_unknowns_are_rejected() -> None:
    parameters, keys = parameter_set()
    with pytest.raises(ValueError, match="ordered without cycles"):
        ConstraintTransform(
            parameters,
            [AffineConstraint(keys[0], keys[1]), AffineConstraint(keys[1], keys[0])],
        )
    with pytest.raises(ValueError, match="only once"):
        ConstraintTransform(
            parameters,
            [FixedConstraint(keys[0], 1e-4), FixedConstraint(keys[0], 2e-4)],
        )
    with pytest.raises(ValueError, match="outside"):
        ConstraintTransform(parameters).unpack([-1.0, -1.0, 1.2, 1.0])
    clipped = ConstraintTransform(parameters).unpack([-1.0, -1.0, 1.2, 1.0], clip=True)
    assert clipped[keys[0]] == 0.0


def test_weighted_residuals_masks_and_standard_metrics() -> None:
    pattern = rietveld.PowderPattern(
        [1.0, 2.0, 3.0, 4.0],
        observed_y=[10.0, 20.0, 30.0, 40.0],
        uncertainty=[1.0, 2.0, 3.0, 4.0],
        mask=[True, False, True, True],
    )
    actual = evaluate_residuals(
        pattern,
        [11.0, 18.0, 27.0, 44.0],
        ResidualOptions(parameter_count=1),
    )
    np.testing.assert_array_equal(actual.residual, [1.0, -2.0, -3.0, 4.0])
    np.testing.assert_array_equal(actual.weighted_residual, [1.0, -1.0, -1.0, 1.0])
    assert actual.chi_square == pytest.approx(3.0)
    assert actual.reduced_chi_square == pytest.approx(1.5)
    assert actual.rp == pytest.approx(8.0 / 80.0)
    denominator = 10.0**2 + (30.0 / 3.0) ** 2 + (40.0 / 4.0) ** 2
    assert actual.rwp == pytest.approx(np.sqrt(3.0 / denominator))


def instrument() -> rietveld.ConstantWavelengthInstrument:
    return rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )


def calculation() -> rietveld.AccumulationResult:
    x = np.linspace(35.0, 45.0, 1001)
    return rietveld.accumulate_cw(
        x,
        [38.0, 41.0, 43.0],
        [12.0, 7.0, 4.0],
        instrument(),
    )


def test_hybrid_jvp_matches_dense_materialization_and_vjp_is_adjoint() -> None:
    result = calculation()
    derivatives = result.derivatives
    rng = np.random.default_rng(20260805)
    local_vector = rng.normal(size=(3, 2))
    global_vector = rng.normal(size=5)
    samples = rng.normal(size=result.y.size)
    actual = jacobian_vector_product(derivatives, local_vector, global_vector)
    dense_local = derivatives.local.to_dense(result.y.size)
    expected = np.einsum("rps,rp->s", dense_local, local_vector)
    expected += global_vector @ derivatives.global_jacobian
    np.testing.assert_allclose(actual, expected, rtol=2e-16, atol=2e-15)

    local_transpose, global_transpose = transpose_jacobian_vector_product(
        derivatives, samples
    )
    left = float(actual @ samples)
    right = float(local_vector.ravel() @ local_transpose.ravel())
    right += float(global_vector @ global_transpose)
    assert left == pytest.approx(right, rel=3e-15, abs=2e-13)


def test_selected_hybrid_jvp_matches_simultaneous_finite_difference() -> None:
    x = np.linspace(35.0, 45.0, 1001)
    positions = np.array([38.0, 41.0, 43.0])
    intensities = np.array([12.0, 7.0, 4.0])
    baseline = rietveld.accumulate_cw(x, positions, intensities, instrument())
    local_direction = np.zeros((3, 2))
    local_direction[:, 0] = [0.2, -0.1, 0.3]
    local_direction[:, 1] = [0.01, -0.02, 0.015]
    global_direction = np.array([0.02, -0.01, 0.015, -0.03, 0.01])
    analytical = jacobian_vector_product(
        baseline.derivatives, local_direction, global_direction
    )
    step = 1.0e-6
    field_names = ("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg")
    plus_values = {
        name: getattr(instrument(), name) + step * direction
        for name, direction in zip(field_names, global_direction, strict=True)
    }
    minus_values = {
        name: getattr(instrument(), name) - step * direction
        for name, direction in zip(field_names, global_direction, strict=True)
    }
    plus = rietveld.accumulate_cw(
        x,
        positions + step * local_direction[:, 1],
        intensities + step * local_direction[:, 0],
        replace(instrument(), **plus_values),
    ).y
    minus = rietveld.accumulate_cw(
        x,
        positions - step * local_direction[:, 1],
        intensities - step * local_direction[:, 0],
        replace(instrument(), **minus_values),
    ).y
    finite = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(analytical, finite, rtol=3e-6, atol=2e-7)


def test_refinement_boundary_validation_is_clear() -> None:
    with pytest.raises(ValueError, match="observed_y"):
        evaluate_residuals(rietveld.PowderPattern([1.0]), [0.0])
    with pytest.raises(ValueError, match="reserved"):
        ParameterKey("phase", "bad[id]", "scale")
    result = calculation()
    with pytest.raises(ValueError, match="local_vector"):
        jacobian_vector_product(result.derivatives, np.zeros((1, 2)), np.zeros(5))
    with pytest.raises(ValueError, match="sample_vector"):
        transpose_jacobian_vector_product(result.derivatives, np.zeros(2))
