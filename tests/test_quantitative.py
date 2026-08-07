import numpy as np
import pytest
from phasesmith.quantitative import (
    QuantitativePhase,
    quantitative_phase_analysis,
    weight_fractions_from_scale,
)


def test_equal_crystallographic_factors_reduce_to_normalized_scales() -> None:
    fractions = weight_fractions_from_scale(
        [1.0, 3.0],
        [2.0, 2.0],
        [10.0, 10.0],
        [20.0, 20.0],
    )

    np.testing.assert_allclose(fractions, [0.25, 0.75], rtol=0.0, atol=1e-15)
    assert not fractions.flags.writeable


def test_recovers_qarr_1g_weighed_composition_from_compatible_scales() -> None:
    # Representative Z, formula masses, and unit-cell volumes for the three
    # QARR phases. Scales are constructed from the independently weighed values.
    expected = np.array([0.3137, 0.3421, 0.3442])
    z = np.array([6.0, 2.0, 4.0])
    mass = np.array([101.9601, 81.379, 78.0748])
    volume = np.array([254.97, 47.62, 163.03])
    compatible_scale = expected / (z * mass * volume)

    fractions = weight_fractions_from_scale(compatible_scale, z, mass, volume)

    np.testing.assert_allclose(fractions, expected, rtol=0.0, atol=1e-15)


def test_typed_analysis_preserves_phase_order_and_normalizes() -> None:
    results = quantitative_phase_analysis(
        [
            QuantitativePhase("alpha", 2.0, 1.0, 10.0, 5.0),
            QuantitativePhase("beta", 1.0, 2.0, 10.0, 5.0),
        ]
    )

    assert [result.phase_id for result in results] == ["alpha", "beta"]
    np.testing.assert_allclose([result.weight_fraction for result in results], [0.5, 0.5])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]), "at least one"),
        (([-1.0], [1.0], [1.0], [1.0]), "nonnegative"),
        (([1.0], [0.0], [1.0], [1.0]), "formula_units_per_cell must be positive"),
        (([1.0, 2.0], [1.0], [1.0], [1.0]), "same shape"),
        (([1.0], [1.0], [np.nan], [1.0]), "finite"),
        (([1e308], [1e308], [1.0], [1.0]), "overflowed"),
    ],
)
def test_array_api_rejects_invalid_inputs(
    arguments: tuple[list[float], list[float], list[float], list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weight_fractions_from_scale(*arguments)


def test_typed_analysis_rejects_duplicate_phase_ids() -> None:
    phases = [
        QuantitativePhase("alpha", 1.0, 1.0, 1.0, 1.0),
        QuantitativePhase("alpha", 1.0, 1.0, 1.0, 1.0),
    ]
    with pytest.raises(ValueError, match="unique"):
        quantitative_phase_analysis(phases)
