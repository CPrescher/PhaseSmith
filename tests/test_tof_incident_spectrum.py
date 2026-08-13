"""Facility-neutral TOF incident-spectrum and normalization contracts."""

from __future__ import annotations

import numpy as np
import phasesmith
import pytest

COEFFICIENTS = np.array(
    [12.0, 40_000.0, 3.0, 2.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)


def test_numpy_equation_and_derivative_match_closed_form_differences() -> None:
    spectrum = phasesmith.TofIncidentSpectrum(500.0, 10_000.0, COEFFICIENTS)
    tof_us = np.array([1_250.0, 2_500.0, 7_500.0])
    time_ms = tof_us / 1_000.0
    x = 2.0 / time_ms - 1.0
    expected = (
        12.0
        + 40_000.0 / time_ms**5 * np.exp(-3.0 / time_ms**2)
        + 2.0 * x
        - 0.5 * (2.0 * x**2 - 1.0)
        + 0.25 * (4.0 * x**3 - 3.0 * x)
    )
    actual = spectrum.evaluate(tof_us)
    np.testing.assert_allclose(actual.values, expected, rtol=2e-15)

    step_us = 1.0e-3
    finite = (
        spectrum.evaluate(tof_us + step_us).values - spectrum.evaluate(tof_us - step_us).values
    ) / (2.0 * step_us)
    np.testing.assert_allclose(actual.d_values_d_tof_us, finite, rtol=2e-8, atol=1e-10)
    assert not actual.values.flags.writeable
    assert not actual.d_values_d_tof_us.flags.writeable


def test_pattern_normalization_scales_all_intensity_domain_arrays() -> None:
    spectrum = phasesmith.TofIncidentSpectrum(
        1_000.0,
        2_000.0,
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    pattern = phasesmith.TofPowderPattern(
        [1_000.0, 1_500.0, 2_000.0],
        observed_y=[20.0, 30.0, 40.0],
        uncertainty=[2.0, 4.0, 6.0],
        mask=[True, False, True],
        background=[4.0, 6.0, 8.0],
    )
    normalized = spectrum.normalize_pattern(pattern)
    np.testing.assert_array_equal(normalized.observed_y, [10.0, 15.0, 20.0])
    np.testing.assert_array_equal(normalized.uncertainty, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(normalized.background, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(normalized.mask, pattern.mask)
    np.testing.assert_array_equal(pattern.observed_y, [20.0, 30.0, 40.0])


def test_bounded_gsas_adapter_exposes_type_four_without_facility_assumptions() -> None:
    calibration = phasesmith.read_gsas_tof_instrument(
        "INS  2 ICONS   4368.97      0.02      2.11         0\n"
        "INS  2I ITYP    4    0.7500    8.1904     76288\n"
        "INS  2ICOFF1   0.177427E+04   0.783794E+07   0.237297E+02   0.305645E+04\n"
        "INS  2ICOFF2  -0.600307E+03  -0.146005E+03  -0.147656E+03   0.442342E+03\n"
        "INS  2ICOFF3  -0.302364E+03   0.885096E+02  -0.968997E+01   0.000000E+00\n"
        "INS  2PRCF      1   12   0.01000    0NNNNNNNNNNNNNNNNNNNN\n"
        "INS  2PRCF 1   0.000000E+00   0.142760E+00   0.557661E-01   0.344498E-02\n"
        "INS  2PRCF 2   0.000000E+00   0.977951E+02   0.000000E+00   0.000000E+00\n",
        bank=2,
    )
    spectrum = calibration.incident_spectrum
    assert spectrum is not None
    assert spectrum.min_tof_us == pytest.approx(750.0)
    assert spectrum.max_tof_us == pytest.approx(8_190.4)
    assert spectrum.coefficients[0] == pytest.approx(1_774.27)
    assert spectrum.evaluate([2_500.0]).values[0] > 0.0


def test_invalid_spectrum_shapes_ranges_domains_and_pattern_are_explicit() -> None:
    with pytest.raises(ValueError, match="12 finite"):
        phasesmith.TofIncidentSpectrum(1.0, 2.0, [1.0])
    with pytest.raises(ValueError, match="increasing"):
        phasesmith.TofIncidentSpectrum(2.0, 1.0, np.ones(12))
    spectrum = phasesmith.TofIncidentSpectrum(1_000.0, 2_000.0, np.ones(12))
    with pytest.raises(ValueError, match="calibration interval"):
        spectrum.evaluate([999.0])
    with pytest.raises(TypeError, match="observed TofPowderPattern"):
        spectrum.normalize_pattern(phasesmith.TofPowderPattern([1_000.0]))
