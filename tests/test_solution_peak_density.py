"""Spherical support, Fourier normalization, derivatives and assignment safety."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
from phasesmith.experimental.solution_peak_density_reference import (
    ball_reference,
    integrated_reference,
    label_mask_reference,
)
from test_solution_shared_search import synthetic

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_IntegratedPeakDensity"), reason="requires experimental-solution"
)


@pytest.mark.parametrize(
    "q,radius",
    [
        (0.0, 0.7),
        (1e-10, 0.7),
        (0.02, 0.7),
        (0.2, 0.7),
        (2.0, 0.7),
        (5.0, 1.5),
        (10.0, 2.0),
        (1.0, 1e-4),
    ],
)
def test_ball_transform_matches_independent_disk_quadrature_and_radius_derivative(q, radius):
    value, derivative = _core._solution_ball_transform(q, radius)
    ref, ref_derivative = ball_reference(q, radius)
    np.testing.assert_allclose([value, derivative], [ref, ref_derivative], rtol=3e-11, atol=2e-12)
    if radius > 1e-4 and radius < 2:
        step = 1e-6
        finite = (
            _core._solution_ball_transform(q, radius + step)[0]
            - _core._solution_ball_transform(q, radius - step)[0]
        ) / (2 * step)
        assert abs(finite - derivative) < 2e-8


def map_inputs(cell=None):
    cell = cell or UnitCell(6, 7, 8, 79, 96, 105)
    shape = np.array([16, 16, 16])
    hkl = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=np.int64)
    grid = np.stack(np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), axis=-1)
    source = np.ones(tuple(shape)) * 0.7
    for h in ([1, 0, 0], [0, 2, 1], [1, -1, 2], [2, 1, -1]):
        source += np.cos(2 * np.pi * (grid @ h) + 0.173 * sum(h))
    # Undefined phases remain absent; positive observed amplitudes are not phases.
    amplitudes = np.sqrt(np.sum(hkl * hkl, axis=1) + 2.0).astype(float)
    return cell, shape, hkl, amplitudes, source


def test_integrated_field_and_fused_derivatives_against_direct_fourier_sum():
    cell, shape, hkl, observed, source = map_inputs()
    point = np.array([0.137, 0.271, 0.413])
    radius = 0.7
    native = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), radius
    )
    expected = integrated_reference(
        cell.geometry().direct_basis, shape, hkl, observed, source, radius, point
    )
    value, gradient, dr = native.evaluate(point)
    np.testing.assert_allclose(value, expected[0], rtol=3e-11, atol=2e-13)
    np.testing.assert_allclose(gradient, expected[1], rtol=3e-11, atol=2e-13)
    np.testing.assert_allclose(dr, expected[2], rtol=3e-11, atol=2e-13)
    step = 1e-6
    for axis in range(3):
        shift = np.eye(3)[axis] * step
        finite = (native.evaluate(point + shift)[0] - native.evaluate(point - shift)[0]) / (
            2 * step
        )
        assert abs(finite - gradient[axis]) < 3e-10
    lo = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), radius - step
    )
    hi = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), radius + step
    )
    assert abs((hi.evaluate(point)[0] - lo.evaluate(point)[0]) / (2 * step) - dr) < 3e-10
    periodic = native.evaluate(point + np.array([3, -2, 1]))
    np.testing.assert_allclose([periodic[0], periodic[2]], [value, dr], rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(periodic[1], gradient, rtol=2e-12, atol=1e-13)
    scaled = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed * 3, source.ravel(), radius
    ).evaluate(point)
    np.testing.assert_allclose([scaled[0], scaled[2]], [3 * value, 3 * dr], rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(scaled[1], np.array(gradient) * 3, rtol=2e-12, atol=1e-13)


def test_constant_field_volume_normalization_and_missing_phase_zero():
    cell = UnitCell(4, 5, 6, 90, 90, 90)
    shape = [8] * 3
    hkl = np.zeros(3, dtype=np.int64)
    observed = np.array([120.0])
    radius = 0.7
    native = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), shape, hkl, observed, np.ones(512), radius
    )
    value, gradient, dr = native.evaluate([0.123, 0.234, 0.345])
    np.testing.assert_allclose(value, 4 * np.pi * radius**3 / 3, rtol=2e-14)
    np.testing.assert_allclose(dr, 4 * np.pi * radius**2, rtol=2e-14)
    np.testing.assert_allclose(gradient, 0, atol=3e-15)
    empty = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), shape, hkl, observed, np.zeros(512), radius
    )
    assert empty.phased_reflections == 0
    assert empty.evaluate([0, 0, 0]) == (0.0, [0.0, 0.0, 0.0], 0.0)


def test_equal_charge_different_widths_have_more_stable_integrals_than_heights():
    cell = UnitCell(4, 4, 4, 90, 90, 90)
    shape = np.array([32] * 3)
    grid = np.stack(np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), axis=-1)
    points = np.array([[0.2, 0.3, 0.4], [0.7, 0.3, 0.4]])
    source = np.zeros(tuple(shape))
    for point, sigma in zip(points, [0.16, 0.26], strict=True):
        delta = ((grid - point + 0.5) % 1 - 0.5) * 4
        source += (
            8
            * np.exp(-np.sum(delta * delta, axis=-1) / (2 * sigma * sigma))
            / ((2 * np.pi) ** 1.5 * sigma**3)
        )
    hkl = np.array(list(itertools.product(range(-15, 16), repeat=3)), dtype=np.int64)
    coefficients = (
        np.fft.ifftn(source) * np.prod(shape) * cell.geometry().volume_angstrom3 / np.prod(shape)
    )
    observed = np.abs(coefficients[tuple((hkl % shape).T)]).astype(float)
    native = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), 0.7
    )
    masses = np.array([native.evaluate(p)[0] for p in points])
    assert np.all(np.abs(masses - 8) < 1.0)
    assert masses.max() / masses.min() < 1.15
    heights = np.array([source[tuple(np.rint(p * shape).astype(int) % shape)] for p in points])
    assert heights.max() / heights.min() > 3


@pytest.mark.parametrize(
    "mass,scale", [(7.0, 1.0), (19.0, 1.0), (35.0, 1.0), (-1.0, 1.0), (7.0, None), (0.3, 0.1)]
)
def test_label_ambiguity_and_weak_evidence_fallback(mass, scale):
    numbers = [35, 19, 9, 7, 6]
    assert _core._solution_density_labels(mass, scale, numbers) == label_mask_reference(
        mass, scale, numbers
    )
    if mass == 7 and scale == 1:
        assert _core._solution_density_labels(mass, scale, [7, 6]) == [True, True]


@pytest.mark.parametrize("radius", [0.0, -1.0, 3.0, np.nan, np.inf])
def test_invalid_sphere_radius(radius):
    with pytest.raises(ValueError):
        _core._solution_ball_transform(1.0, radius)


def test_density_constructor_rejects_aliasing_missing_mates_and_nonfinite_centers():
    cell, shape, hkl, observed, source = map_inputs()
    with pytest.raises(ValueError):
        _core._IntegratedPeakDensity(
            list(cell.as_tuple()),
            [8] * 3,
            np.array([4, 0, 0], dtype=np.int64),
            np.ones(1),
            np.ones(512),
            0.7,
        )
    with pytest.raises(ValueError):
        _core._IntegratedPeakDensity(
            list(cell.as_tuple()), list(shape), hkl[1:].ravel(), observed[1:], source.ravel(), 0.7
        )
    native = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), 0.7
    )
    with pytest.raises(ValueError):
        native.evaluate([np.nan, 0, 0])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_density_guidance_preserves_unknown_counts_and_records_calibration(symbol):
    cell, group, data, _, _ = synthetic(symbol)
    options = SolutionOptions(
        trials=1, cycles=800, max_seconds=15, r1_target=0.005, completion=False
    )
    results = [
        solve_structure_elements(
            data,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            maximum_atoms=32,
            options=options,
            integrated_peak_density=enabled,
        )
        for enabled in (False, True)
    ]
    baseline, guided = results
    assert guided.target_met and baseline.target_met
    assert guided.composition == baseline.composition
    assert guided.best.r1 < 1e-6
    assert (
        guided.shared_search.integrated_peak_density
        and guided.shared_search.density_map_builds == 1
    )
    assert (
        guided.shared_search.density_evaluations > 0 and guided.shared_search.density_calibrations
    )
    assert (
        not baseline.shared_search.integrated_peak_density
        and baseline.shared_search.density_map_builds == 0
    )


@pytest.mark.parametrize("q", [-1.0, np.nan, np.inf])
def test_invalid_ball_frequency(q):
    with pytest.raises(ValueError):
        _core._solution_ball_transform(q, 0.7)


def test_harmonic_grid_plane_uses_positive_cell_derivative():
    cell = UnitCell(4, 5, 6, 90, 90, 90)
    shape = np.array([8] * 3)
    x = np.arange(8) / 8
    source = np.broadcast_to((1 + np.cos(2 * np.pi * x))[:, None, None], tuple(shape)).copy()
    hkl = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int64)
    amplitudes = np.array([2.0, 10.0, 2.0])
    native = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), amplitudes, source.ravel(), 0.7
    )
    value, gradient, _ = native.evaluate([0.0, 0.2, 0.3])
    step = 1e-5  # Within one linear interpolation cell; avoids subtraction cancellation.
    assert abs((native.evaluate([step, 0.2, 0.3])[0] - value) / step - gradient[0]) < 3e-10
    # The quadrature reference checks Fourier response exactly at the node.
    k0, _ = ball_reference(0.0, 0.7)
    k1, _ = ball_reference(0.25, 0.7)
    assert abs(value - (10 * k0 + 4 * k1) / 120) < 2e-13


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integrated_peak_density": 1},
        {"peak_density_radius_angstrom": 0.0},
        {"peak_density_radius_angstrom": np.nan},
        {"peak_density_radius_angstrom": True},
    ],
)
def test_public_density_controls_are_validated(kwargs):
    cell, group, data, _, _ = synthetic("P-1")
    with pytest.raises((ValueError, TypeError)):
        solve_structure_elements(data, cell=cell, space_group=group, elements=["Fe", "O"], **kwargs)


@pytest.mark.parametrize(
    "mass,scale,numbers",
    [(np.nan, 1.0, [6]), (1.0, 0.0, [6]), (1.0, 1.0, [1]), (1.0, 1.0, [200]), (1.0, 1.0, [])],
)
def test_invalid_label_evidence(mass, scale, numbers):
    with pytest.raises(ValueError):
        _core._solution_density_labels(mass, scale, numbers)


def test_unmeasured_zero_reflection_is_not_invented():
    cell, shape, hkl, observed, source = map_inputs()
    keep = np.any(hkl != 0, axis=1)
    point = np.array([0.137, 0.271, 0.413])
    radius = 0.7
    full = _core._IntegratedPeakDensity(
        list(cell.as_tuple()), list(shape), hkl.ravel(), observed, source.ravel(), radius
    ).evaluate(point)
    missing = _core._IntegratedPeakDensity(
        list(cell.as_tuple()),
        list(shape),
        hkl[keep].ravel(),
        observed[keep],
        source.ravel(),
        radius,
    ).evaluate(point)
    volume, area = ball_reference(0.0, radius)
    coefficient = observed[~keep][0] / cell.geometry().volume_angstrom3
    np.testing.assert_allclose(
        [full[0] - missing[0], full[2] - missing[2]],
        [coefficient * volume, coefficient * area],
        rtol=3e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(np.array(full[1]) - missing[1], 0, atol=2e-13)
