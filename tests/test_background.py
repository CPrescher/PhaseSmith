from __future__ import annotations

import sys

import numpy as np
import phasesmith
import pytest
from phasesmith import background_reference


def peak_rich_signal(samples: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(10.0, 20.0, samples)
    baseline = 3.0 + 0.15 * (x - 10.0) + 0.4 * np.cos(0.5 * x)
    peaks = (
        20.0 * np.exp(-0.5 * ((x - 12.0) / 0.035) ** 2)
        + 8.0 * np.exp(-0.5 * ((x - 15.2) / 0.08) ** 2)
        + 14.0 * np.exp(-0.5 * ((x - 18.1) / 0.05) ** 2)
    )
    return x, baseline + peaks


def test_native_smoother_matches_independent_reference_randomized() -> None:
    rng = np.random.default_rng(20260807)
    for sample_count, smooth_points, iterations in (
        (1, 0, 0),
        (8, 0, 7),
        (12, 2, 1),
        (37, 3, 9),
        (101, 12, 5),
        (11, 20, 4),
    ):
        y = rng.uniform(-2.0, 7.0, sample_count)
        actual = phasesmith.smooth_bruckner(y, smooth_points, iterations)
        expected = background_reference.smooth_bruckner(y, smooth_points, iterations)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_pinned_xypattern_cython_fixture_and_trailing_range() -> None:
    # Generated from xypattern 1.2.3 smooth_bruckner.pyx at the revision
    # exported by phasesmith.XYPATTERN_REVISION.
    y = np.array(
        [
            2.969336117675465,
            3.1174095473616203,
            2.958192636184182,
            3.1165009223502547,
            7.147173595093141,
            3.161693323546317,
            3.1752331915120955,
            3.017912282120381,
            3.15150232452551,
            3.300392341749493,
            3.384377146380255,
            3.152007862177872,
            11.123139472373492,
            3.2218442981407436,
            3.369497722325094,
            3.340209794888846,
            3.4457332514537757,
            6.359619686181661,
            3.4123141446358045,
            3.593132262352966,
            3.4092332877761398,
            3.4947639078460075,
            3.4040091970312156,
            3.32294708182363,
            3.5249457863690536,
            9.494140008111245,
            3.4954966513402197,
            3.6546202294969587,
            3.3822635231249416,
            3.7214092234334215,
            3.570209024662823,
            3.6820555995420823,
        ]
    )
    expected = np.array(
        [
            2.969336117675465,
            3.007288661484387,
            2.958192636184182,
            3.039678181060238,
            3.04184811782149,
            3.057885646771133,
            3.0933875159066457,
            3.017912282120381,
            3.1322527832561025,
            3.161847914197732,
            3.1712222363514333,
            3.152007862177872,
            3.2369523732421923,
            3.2218442981407436,
            3.2940022648751834,
            3.324834129410465,
            3.3471542395597305,
            3.36707915177254,
            3.3793292974782987,
            3.3875353422986727,
            3.3810723958411213,
            3.4015952307944257,
            3.4040091970312156,
            3.32294708182363,
            3.5249457863690536,
            9.494140008111245,
            3.4954966513402197,
            3.6546202294969587,
            3.3822635231249416,
            3.7214092234334215,
            3.570209024662823,
            3.6820555995420823,
        ]
    )
    actual = phasesmith.smooth_bruckner(y, 3, 4)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-15)
    np.testing.assert_array_equal(actual[-8:], y[-8:])
    assert len(phasesmith.XYPATTERN_REVISION) == 40


def test_raw_smoother_validates_inputs_and_returns_owned_read_only_data() -> None:
    y = np.arange(8.0)
    actual = phasesmith.smooth_bruckner(y, 1, 2)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    assert not actual.flags.writeable
    assert y.flags.writeable
    for invalid in ([], [1.0, np.nan], [1.0, np.inf]):
        with pytest.raises(ValueError):
            phasesmith.smooth_bruckner(invalid, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        phasesmith.smooth_bruckner(np.ones((2, 2)), 1)
    with pytest.raises(ValueError, match="real floating-point or integer"):
        phasesmith.smooth_bruckner(["1", "2"], 1)
    with pytest.raises(ValueError, match="non-negative"):
        phasesmith.smooth_bruckner(y, -1)
    with pytest.raises(TypeError, match="integer"):
        phasesmith.smooth_bruckner(y, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        phasesmith.smooth_bruckner(y, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        phasesmith.smooth_bruckner(y, 1, -1)


def test_physical_width_raw_model_matches_point_kernel() -> None:
    x, y = peak_rich_signal()
    model = phasesmith.SmoothBrucknerBackground(
        smooth_width=0.1,
        iterations=12,
        chebyshev_order=None,
    )
    result = model.subtract(x, y)
    assert result.smooth_points == 10
    np.testing.assert_array_equal(result.smoothed_y, phasesmith.smooth_bruckner(y, 10, 12))
    np.testing.assert_array_equal(result.background, result.smoothed_y)
    np.testing.assert_allclose(result.corrected_y, y - result.background, rtol=0.0, atol=0.0)
    assert all(
        not values.flags.writeable
        for values in (result.background, result.corrected_y, result.smoothed_y)
    )


def test_default_pipeline_matches_explicit_chebyshev_fit_and_pattern_boundary() -> None:
    x, y = peak_rich_signal()
    model = phasesmith.SmoothBrucknerBackground()
    result = model.subtract(x, y)
    normalized_x = 2.0 * (x - x[0]) / (x[-1] - x[0]) - 1.0
    expected_coefficients = np.polynomial.chebyshev.chebfit(
        normalized_x,
        result.smoothed_y,
        50,
    )
    expected = np.polynomial.chebyshev.chebval(normalized_x, expected_coefficients)
    np.testing.assert_allclose(result.background, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(model.estimate(x, y), result.background)
    pattern = phasesmith.PowderPattern(x, observed_y=y, background=result.background)
    np.testing.assert_array_equal(pattern.background, result.background)
    np.testing.assert_array_equal(pattern.observed_y - pattern.background, result.corrected_y)
    assert "xypattern" not in sys.modules


def test_physical_width_model_rejects_ambiguous_or_invalid_configuration() -> None:
    x = np.linspace(1.0, 2.0, 20)
    y = np.ones_like(x)
    with pytest.raises(ValueError, match="smooth_width"):
        phasesmith.SmoothBrucknerBackground(smooth_width=-0.1)
    with pytest.raises(ValueError, match="smooth_width"):
        phasesmith.SmoothBrucknerBackground(smooth_width=np.inf)
    with pytest.raises(TypeError, match="iterations"):
        phasesmith.SmoothBrucknerBackground(iterations=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chebyshev_order"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=-1)
    with pytest.raises(ValueError, match="at least two"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=None).estimate([1.0], [2.0])
    with pytest.raises(ValueError, match="same shape"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=None).estimate(x, y[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=None).estimate(x[::-1], y)
    nonuniform = x.copy()
    nonuniform[10:] += 0.01
    with pytest.raises(ValueError, match="uniformly spaced"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=None).estimate(nonuniform, y)
    with pytest.raises(ValueError, match="smaller than"):
        phasesmith.SmoothBrucknerBackground(chebyshev_order=x.size).estimate(x, y)


def test_result_public_constructor_copies_inputs_and_validates_shape() -> None:
    values = np.arange(4.0)
    result = phasesmith.BackgroundSubtractionResult(values, values, values, 1)
    values[0] = 99.0
    assert result.background[0] == 0.0
    assert not result.background.flags.writeable
    with pytest.raises(ValueError, match="same shape"):
        phasesmith.BackgroundSubtractionResult(np.ones(2), np.ones(3), np.ones(2), 1)
