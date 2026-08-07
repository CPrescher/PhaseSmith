from __future__ import annotations

import numpy as np
import pytest
from rietveld.refinement import (
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
    DifferentiableBackground,
    PointBackground,
    PolynomialBackground,
)


@pytest.mark.parametrize(
    "model",
    (
        PolynomialBackground("power", (2.0, -0.3, 0.08)),
        ChebyshevBackground("chebyshev", (2.0, -0.3, 0.08), (10.0, 80.0)),
        PointBackground("points", (10.0, 25.0, 50.0, 80.0), (1.0, 2.0, 1.5, 2.5)),
        AmorphousBackground(
            "amorphous",
            (AmorphousPeak(12.0, 38.0, 11.0), AmorphousPeak(5.0, 62.0, 8.0)),
        ),
    ),
)
def test_background_analytical_basis_matches_centered_differences(
    model: DifferentiableBackground,
) -> None:
    x = np.linspace(10.0, 80.0, 351)
    basis = model.basis(x)
    assert basis.shape == (x.size, len(model.coefficients))
    for index, value in enumerate(model.coefficients):
        step = 1.0e-6 * max(abs(value), 1.0)
        plus = np.asarray(model.coefficients)
        minus = np.asarray(model.coefficients)
        plus[index] += step
        minus[index] -= step
        finite_difference = (
            model.replace_coefficients(plus).calculate(x)
            - model.replace_coefficients(minus).calculate(x)
        ) / (2.0 * step)
        np.testing.assert_allclose(basis[:, index], finite_difference, rtol=2e-7, atol=2e-9)


def test_point_background_has_constant_ends_and_exact_knots() -> None:
    model = PointBackground("points", (20.0, 40.0, 60.0), (2.0, 5.0, 3.0))
    actual = model.calculate([10.0, 20.0, 30.0, 40.0, 60.0, 70.0])
    np.testing.assert_allclose(actual, [2.0, 2.0, 3.5, 5.0, 3.0, 3.0], rtol=0.0, atol=0.0)


def test_composite_background_preserves_component_parameter_order() -> None:
    power = PolynomialBackground("power", (1.0, 0.2))
    amorphous = AmorphousBackground("glass", (AmorphousPeak(4.0, 45.0, 10.0),))
    model = CompositeBackground("combined", (power, amorphous))
    x = np.linspace(20.0, 70.0, 101)
    np.testing.assert_allclose(model.calculate(x), power.calculate(x) + amorphous.calculate(x))
    assert model.parameter_names == (
        "power.coefficient_0",
        "power.coefficient_1",
        "glass.peak_0.area",
        "glass.peak_0.center_deg",
        "glass.peak_0.fwhm_deg",
    )
    assert isinstance(model, DifferentiableBackground)


def test_background_validation_rejects_ambiguous_domains() -> None:
    with pytest.raises(ValueError, match="domain_deg"):
        ChebyshevBackground("bad", (1.0,), (20.0, 20.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        PointBackground("bad", (20.0, 20.0), (1.0, 2.0))
    with pytest.raises(ValueError, match="FWHM"):
        AmorphousPeak(1.0, 40.0, 0.0)
