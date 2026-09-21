from __future__ import annotations

from dataclasses import replace

import numpy as np
import phasesmith
from phasesmith import fpa_calibration
from phasesmith._numpy_compat import trapezoid


def _options() -> phasesmith.FundamentalProfileCalibrationOptions:
    return phasesmith.FundamentalProfileCalibrationOptions(
        peak_positions_deg=(20.0, 40.0, 60.0, 80.0, 100.0, 120.0),
        window_half_width_deg=0.8,
        step_deg=0.002,
        aperture_quadrature_order=1,
        fcj_quadrature_order=48,
        support_fwhm=50.0,
        max_iterations=50,
    )


def _representable_model() -> phasesmith.BraggBrentanoFundamentalProfile:
    first_wavelength = 1.5405929
    second_wavelength = 1.5444274
    ratio = second_wavelength / first_wavelength
    return phasesmith.BraggBrentanoFundamentalProfile(
        radius_mm=217.5,
        source_width_mm=0.0,
        receiving_slit_width_mm=0.0,
        sample_half_length_mm=1.0875,
        detector_half_length_mm=1.0875,
        emission_lines=(
            phasesmith.FundamentalEmissionLine(
                first_wavelength,
                2.0,
                0.0002600,
                0.0001200,
            ),
            phasesmith.FundamentalEmissionLine(
                second_wavelength,
                1.0,
                0.0002600 * ratio,
                0.0001200 * ratio,
            ),
        ),
    )


def test_independent_fundamental_pattern_is_deterministic_and_immutable() -> None:
    model = _representable_model()
    options = _options()

    first = phasesmith.simulate_fundamental_peaks(model, options)
    second = phasesmith.simulate_fundamental_peaks(model, options)

    np.testing.assert_array_equal(first.positions_deg, options.peak_positions_deg)
    np.testing.assert_array_equal(first.grid_deg, second.grid_deg)
    np.testing.assert_array_equal(first.intensity, second.intensity)
    assert len(first.peak_slices) == len(options.peak_positions_deg)
    assert np.all(first.intensity > 0.0)
    assert not first.positions_deg.flags.writeable
    assert not first.grid_deg.flags.writeable
    assert not first.intensity.flags.writeable


def test_uniform_equatorial_apertures_add_the_first_principles_variance() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.0002, 0.0)
    narrow = phasesmith.BraggBrentanoFundamentalProfile(217.5, 0.0, 0.0, 0.0, 0.0, (line,))
    wide = replace(narrow, source_width_mm=0.1, receiving_slit_width_mm=0.2)
    options = replace(_options(), aperture_quadrature_order=12)
    narrow_pattern = phasesmith.simulate_fundamental_peaks(narrow, options)
    wide_pattern = phasesmith.simulate_fundamental_peaks(wide, options)
    active = narrow_pattern.peak_slices[0]
    grid = narrow_pattern.grid_deg[active]

    def variance(values: np.ndarray) -> float:
        weights = values / np.sum(values)
        centroid = float(weights @ grid)
        return float(weights @ (grid - centroid) ** 2)

    actual_increment = variance(wide_pattern.intensity[active]) - variance(
        narrow_pattern.intensity[active]
    )
    expected_increment = (
        np.rad2deg(0.1 / narrow.radius_mm) ** 2 + np.rad2deg(0.2 / narrow.radius_mm) ** 2
    ) / 12.0
    np.testing.assert_allclose(
        actual_increment,
        expected_increment,
        rtol=1.0e-3,
    )


def test_gaussian_spectral_passband_has_explicit_fwhm_and_immutable_values() -> None:
    passband = phasesmith.GaussianSpectralPassband(1.5406, 0.002)

    transmission = passband.transmission(np.array([1.5396, 1.5406, 1.5416]))

    np.testing.assert_allclose(transmission, [0.5, 1.0, 0.5], rtol=2.0e-13)
    assert not transmission.flags.writeable
    with np.testing.assert_raises_regex(ValueError, "positive and finite"):
        phasesmith.GaussianSpectralPassband(1.5406, 0.0)
    with np.testing.assert_raises_regex(ValueError, "positive and finite"):
        passband.transmission([1.54, -1.0])


def test_spectral_passband_narrows_a_line_and_updates_effective_components() -> None:
    lines = (
        phasesmith.FundamentalEmissionLine(1.5406, 2.0, 0.00026, 0.00012),
        phasesmith.FundamentalEmissionLine(1.5444, 1.0, 0.00026, 0.00012),
    )
    geometry = phasesmith.SollerAxialGeometry(0.0, 0.0, 0.0)
    unfiltered = phasesmith.BraggBrentanoFundamentalProfile(
        217.5, 0.0, 0.0, 0.0, 0.0, lines, geometry
    )
    filtered = replace(
        unfiltered,
        spectral_passband=phasesmith.GaussianSpectralPassband(1.5406, 0.004),
    )
    options = replace(_options(), step_deg=0.004)

    unfiltered_pattern = phasesmith.simulate_fundamental_peaks(unfiltered, options)
    filtered_pattern = phasesmith.simulate_fundamental_peaks(filtered, options)
    active = unfiltered_pattern.peak_slices[0]
    grid = unfiltered_pattern.grid_deg[active]

    def variance(values: np.ndarray) -> float:
        weights = values / np.sum(values)
        centroid = float(weights @ grid)
        return float(weights @ (grid - centroid) ** 2)

    assert variance(filtered_pattern.intensity[active]) < variance(
        unfiltered_pattern.intensity[active]
    )
    calibration = phasesmith.calibrate_fundamental_profile(filtered, options)
    assert calibration.components.normalized_intensities[1] < 0.1
    assert calibration.components.wavelengths_angstrom[1] < lines[1].wavelength_angstrom


def test_spectral_transmission_integral_matches_dense_wavelength_reference() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.0003, 0.0002)
    passband = phasesmith.GaussianSpectralPassband(1.5409, 0.0018)
    wavelength = np.linspace(1.5306, 1.5506, 200_001)
    density = fpa_calibration.reference.profile_tch(
        wavelength - line.wavelength_angstrom,
        line.gaussian_fwhm_angstrom,
        line.lorentzian_fwhm_angstrom,
    ).value
    expected = trapezoid(density * passband.transmission(wavelength), wavelength)
    expected_centroid = (
        trapezoid(wavelength * density * passband.transmission(wavelength), wavelength) / expected
    )

    actual = fpa_calibration._line_transmission(line, passband, 255)
    actual_area, actual_centroid = fpa_calibration._line_transmission_moments(line, passband, 255)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6)
    np.testing.assert_allclose(actual_area, expected, rtol=2.0e-6)
    np.testing.assert_allclose(actual_centroid, expected_centroid, rtol=2.0e-9)


def test_soller_axial_target_recovers_fcj_for_a_point_incident_beam() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.00025, 0.0001)
    fcj = phasesmith.BraggBrentanoFundamentalProfile(217.5, 0.0, 0.0, 0.0, 2.0, (line,))
    full_axial = replace(
        fcj,
        soller_axial_geometry=phasesmith.SollerAxialGeometry(
            source_full_length_mm=0.0,
            sample_full_length_mm=0.0,
            receiving_slit_full_length_mm=4.0,
        ),
    )
    options = replace(
        _options(),
        step_deg=0.004,
        aperture_quadrature_order=1,
        axial_ray_quadrature_order=48,
        fcj_quadrature_order=128,
    )

    expected = phasesmith.simulate_fundamental_peaks(fcj, options)
    actual = phasesmith.simulate_fundamental_peaks(full_axial, options)

    np.testing.assert_allclose(actual.intensity, expected.intensity, rtol=8.0e-12, atol=1.0e-12)

    symmetric = replace(fcj, detector_half_length_mm=0.0)
    symmetric_full_axial = replace(
        symmetric,
        soller_axial_geometry=phasesmith.SollerAxialGeometry(0.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(
        phasesmith.simulate_fundamental_peaks(symmetric_full_axial, options).intensity,
        phasesmith.simulate_fundamental_peaks(symmetric, options).intensity,
    )


def test_soller_filters_reduce_full_axial_broadening() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.00025, 0.0001)
    open_geometry = phasesmith.SollerAxialGeometry(12.0, 15.0, 5.0)
    filtered_geometry = replace(
        open_geometry,
        incident_soller_full_width_deg=2.5,
        diffracted_soller_full_width_deg=2.5,
    )
    open_model = phasesmith.BraggBrentanoFundamentalProfile(
        217.5,
        0.0,
        0.0,
        7.5,
        2.5,
        (line,),
        open_geometry,
    )
    filtered_model = replace(open_model, soller_axial_geometry=filtered_geometry)
    options = replace(
        _options(),
        step_deg=0.004,
        aperture_quadrature_order=1,
        axial_ray_quadrature_order=127,
    )

    open_pattern = phasesmith.simulate_fundamental_peaks(open_model, options)
    filtered_pattern = phasesmith.simulate_fundamental_peaks(filtered_model, options)
    active = open_pattern.peak_slices[0]
    grid = open_pattern.grid_deg[active]

    def moments(values: np.ndarray) -> tuple[float, float]:
        weights = values / np.sum(values)
        centroid = float(weights @ grid)
        variance = float(weights @ (grid - centroid) ** 2)
        return centroid, variance

    open_centroid, open_variance = moments(open_pattern.intensity[active])
    filtered_centroid, filtered_variance = moments(filtered_pattern.intensity[active])
    assert filtered_centroid > open_centroid
    assert filtered_variance < 0.1 * open_variance


def test_transformed_soller_quadrature_is_converged_for_nist_geometry() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.00025, 0.0001)
    geometry = phasesmith.SollerAxialGeometry(12.0, 15.0, 5.0, 6.776, 6.776)
    model = phasesmith.BraggBrentanoFundamentalProfile(217.5, 0.0, 0.0, 7.5, 2.5, (line,), geometry)
    options = replace(
        _options(),
        step_deg=0.004,
        aperture_quadrature_order=1,
        axial_ray_quadrature_order=191,
    )

    lower_order = phasesmith.simulate_fundamental_peaks(model, options)
    accepted_order = phasesmith.simulate_fundamental_peaks(
        model,
        replace(options, axial_ray_quadrature_order=255),
    )

    relative_l2 = np.linalg.norm(lower_order.intensity - accepted_order.intensity) / np.linalg.norm(
        accepted_order.intensity
    )
    assert relative_l2 < 2.0e-3


def test_representable_physical_profile_compresses_to_production_model() -> None:
    model = _representable_model()

    result = phasesmith.calibrate_fundamental_profile(model, _options())

    assert result.converged
    assert result.accepted
    assert result.relative_l2_error < 3.5e-4
    assert result.maximum_peak_relative_l2_error < 4.0e-4
    assert result.minimum_profile_correlation > 0.9999999
    assert abs(result.sh_over_l - 0.01) < 5.0e-8
    assert result.axial_geometry == phasesmith.FcjGeometry(
        result.sh_over_l / 2.0,
        result.sh_over_l / 2.0,
    )
    np.testing.assert_array_equal(
        result.components.wavelengths_angstrom,
        [1.5405929, 1.5444274],
    )
    assert result.parameter_names == (
        "u_deg2",
        "v_deg2",
        "w_deg2",
        "x_deg",
        "y_deg",
        "sh_over_l",
    )
    assert result.warnings == ()
    assert not result.target_y.flags.writeable
    assert not result.calculated_y.flags.writeable


def test_nonrepresentable_component_widths_are_rejected_with_diagnostics() -> None:
    model = phasesmith.BraggBrentanoFundamentalProfile(
        radius_mm=217.5,
        source_width_mm=0.0,
        receiving_slit_width_mm=0.0,
        sample_half_length_mm=1.0875,
        detector_half_length_mm=1.0875,
        emission_lines=(
            phasesmith.FundamentalEmissionLine(1.5405929, 2.0, 0.00026, 0.00012),
            phasesmith.FundamentalEmissionLine(1.5444274, 1.0, 0.00030, 0.00014),
        ),
    )

    result = phasesmith.calibrate_fundamental_profile(model, _options())

    assert result.converged
    assert not result.accepted
    assert result.relative_l2_error > 0.03
    assert len(result.peak_diagnostics) == 6
    assert "not represented" in result.warnings[0]


def test_unconverged_compression_cannot_be_accepted() -> None:
    result = phasesmith.calibrate_fundamental_profile(
        _representable_model(),
        replace(_options(), max_iterations=1),
    )

    assert not result.converged
    assert not result.accepted
    assert result.warnings == ("profile compression stopped before the convergence thresholds",)


def test_variable_projection_jacobian_matches_centered_finite_differences() -> None:
    model = _representable_model()
    options = replace(_options(), step_deg=0.004, support_fwhm=200.0)
    target = phasesmith.simulate_fundamental_peaks(model, options)
    parameters = np.array([8.0e-5, -1.0e-6, 2.0e-5, 1.0e-4, 0.009, 0.01])
    _calculated, jacobian, _scales = fpa_calibration._candidate(
        target.grid_deg,
        target.positions_deg,
        target.peak_slices,
        target.intensity,
        model.components,
        parameters,
        options,
    )
    steps = (1.0e-9, 1.0e-9, 1.0e-9, 1.0e-7, 1.0e-7, 1.0e-7)
    for index, step in enumerate(steps):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        plus_y, _plus_jacobian, _plus_scales = fpa_calibration._candidate(
            target.grid_deg,
            target.positions_deg,
            target.peak_slices,
            target.intensity,
            model.components,
            plus,
            options,
        )
        minus_y, _minus_jacobian, _minus_scales = fpa_calibration._candidate(
            target.grid_deg,
            target.positions_deg,
            target.peak_slices,
            target.intensity,
            model.components,
            minus,
            options,
        )
        finite_difference = (plus_y - minus_y) / (2.0 * step)
        relative_error = np.linalg.norm(jacobian[:, index] - finite_difference) / np.linalg.norm(
            finite_difference
        )
        assert relative_error < 2.0e-6


def test_fundamental_profile_inputs_reject_ambiguous_values() -> None:
    line = phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.0002, 0.0001)
    with np.testing.assert_raises(ValueError):
        phasesmith.FundamentalEmissionLine(1.5406, 1.0, 0.0, 0.0)
    with np.testing.assert_raises(ValueError):
        phasesmith.BraggBrentanoFundamentalProfile(0.0, 0.0, 0.0, 1.0, 1.0, (line,))
    with np.testing.assert_raises(TypeError):
        phasesmith.FundamentalProfileCalibrationOptions(  # type: ignore[arg-type]
            peak_positions_deg=[20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
        )
    with np.testing.assert_raises(TypeError):
        phasesmith.FundamentalProfileCalibrationOptions(aperture_quadrature_order=2.5)  # type: ignore[arg-type]
    with np.testing.assert_raises(ValueError):
        phasesmith.SollerAxialGeometry(12.0, 15.0, 5.0, 0.0, 2.5)
    with np.testing.assert_raises_regex(ValueError, "all be positive"):
        phasesmith.SollerAxialGeometry(12.0, 15.0, 0.0)
    with np.testing.assert_raises(ValueError):
        phasesmith.BraggBrentanoFundamentalProfile(
            217.5,
            0.0,
            0.0,
            1.0,
            2.5,
            (line,),
            phasesmith.SollerAxialGeometry(12.0, 15.0, 5.0),
        )
    with np.testing.assert_raises_regex(ValueError, "requires explicit full axial"):
        phasesmith.BraggBrentanoFundamentalProfile(
            217.5,
            0.0,
            0.0,
            0.0,
            0.0,
            (line,),
            spectral_passband=phasesmith.GaussianSpectralPassband(1.5406, 0.002),
        )
    with np.testing.assert_raises(ValueError):
        phasesmith.FundamentalProfileCalibrationOptions(spectral_transmission_quadrature_order=0)
