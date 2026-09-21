"""Accuracy policies are explicit, differentiable locally, and reproducible."""

import math
from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith import reference
from phasesmith._numpy_compat import trapezoid
from phasesmith.refinement import RefinementLimits
from phasesmith.refinement import rietveld as rv
from test_rietveld_optimization import doublet_request
from test_rietveld_refinement import request_from_cif, selection


@pytest.mark.parametrize("value", [0, -0.01, 0.11, float("nan"), float("inf"), 1e-9])
def test_invalid_tail_budgets(value):
    with pytest.raises(ValueError):
        ps.ProfileAccuracy(tail_area_tolerance=value)


def test_accuracy_types_and_defaults():
    assert rv.RietveldOptions().profile_accuracy == ps.ProfileAccuracy()
    with pytest.raises(TypeError):
        ps.ProfileAccuracy(fast_fcj=1)
    with pytest.raises(TypeError):
        ps.ProfileAccuracy(tail_area_tolerance=True)
    with pytest.raises(TypeError):
        rv.RietveldOptions(profile_accuracy="fast")


def axial_for_span(position, width, ratio, fraction):
    limit = position + np.sign(position - 90) * width * ratio
    height = np.sqrt((np.cos(np.deg2rad(limit)) / np.cos(np.deg2rad(position))) ** 2 - 1)
    return ps.FcjGeometry(float(height * fraction), float(height * (1 - fraction)))


@pytest.mark.parametrize("ratio", [0.001, 0.0199, 0.0201, 0.199, 0.201])
def test_fast_fcj_against_independent_and_high_order_integrals(ratio):
    rng = np.random.default_rng(20260917)
    accuracy = ps.ProfileAccuracy(fast_fcj=True)
    for i in range(8):
        position = float(rng.uniform(12, 75) if i % 2 else rng.uniform(105, 168))
        gaussian, lorentzian = rng.uniform(0.02, 0.08), rng.uniform(0.001, 0.03)
        width = reference.tch_shape_from_fwhm(gaussian, lorentzian).total_fwhm
        axial = axial_for_span(position, width, ratio, 0.5 if i % 3 == 0 else 0.2)
        x = position + width * np.linspace(-4, 4, 401)
        actual = ps.profile_fcj(x, position, gaussian, lorentzian, axial, profile_accuracy=accuracy)
        same_rule = reference.profile_fcj(
            x,
            position,
            gaussian,
            lorentzian,
            axial.sample_over_radius,
            axial.detector_over_radius,
            fast_fcj=True,
        )
        high_order = reference.profile_fcj(
            x,
            position,
            gaussian,
            lorentzian,
            axial.sample_over_radius,
            axial.detector_over_radius,
            quadrature_order=256,
        )
        for field in actual.__dataclass_fields__:
            actual_values = getattr(actual, field)
            ref_values = getattr(high_order, field)
            scale = max(float(np.max(np.abs(ref_values))), 1.0)
            assert np.max(np.abs(actual_values - getattr(same_rule, field))) / scale < 2e-9
            # Local scaled derivative envelope; this is not a fit-parameter error bound.
            assert np.max(np.abs(actual_values - ref_values)) / scale < 3e-7
        assert np.max(np.abs(actual.value - high_order.value)) / np.max(high_order.value) < 2e-9


@pytest.mark.parametrize("position", [0.5, 89.999, 90.0, 90.001, 179.5])
def test_fast_fcj_near_angular_limits_and_ninety_degrees(position):
    x = position + np.linspace(-0.1, 0.1, 151)
    geometry = ps.FcjGeometry(0.0001, 0.00007)
    actual = ps.profile_fcj(
        x, position, 0.04, 0.01, geometry, profile_accuracy=ps.ProfileAccuracy(fast_fcj=True)
    )
    expected = reference.profile_fcj(x, position, 0.04, 0.01, 0.0001, 0.00007, quadrature_order=256)
    for field in actual.__dataclass_fields__:
        ref_values = getattr(expected, field)
        scale = max(float(np.max(np.abs(ref_values))), 1.0)
        assert np.max(np.abs(getattr(actual, field) - ref_values)) / scale < 3e-7


def test_fast_fcj_zero_geometry_preserves_symmetric_limit():
    x = np.linspace(39.9, 40.1, 101)
    args = (x, 40.0, 0.04, 0.01, ps.FcjGeometry(0.0, 0.0))
    actual = ps.profile_fcj(*args, profile_accuracy=ps.ProfileAccuracy(fast_fcj=True))
    expected = ps.profile_fcj(*args)
    for field in actual.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_fast_fcj_all_derivatives_match_centered_differences():
    position, gaussian, lorentzian = 45.0, 0.04, 0.01
    width = reference.tch_shape_from_fwhm(gaussian, lorentzian).total_fwhm
    axial = axial_for_span(position, width, 0.008, 0.3)
    values = [position, gaussian, lorentzian, axial.sample_over_radius, axial.detector_over_radius]
    x = position + width * np.linspace(-2, 2, 151)

    def evaluate(v):
        return ps.profile_fcj(
            x, *v[:3], ps.FcjGeometry(*v[3:]), profile_accuracy=ps.ProfileAccuracy(fast_fcj=True)
        )

    actual = evaluate(values)
    for index, field in enumerate(tuple(actual.__dataclass_fields__)[1:]):
        step = 2e-6 if index == 0 else 2e-7
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        fd = (evaluate(plus).value - evaluate(minus).value) / (2 * step)
        scale = max(float(np.max(np.abs(fd))), 1)
        assert np.max(np.abs(getattr(actual, field) - fd)) / scale < 2e-6


@pytest.mark.parametrize("eta", [0.0, 0.01, 0.3, 0.9, 1.0])
def test_tail_bound_is_conservative_for_exact_continuous_area(eta):
    for tolerance in (1e-8, 0.001, 0.01, 0.02, 0.1):
        k = reference.tail_support_multiple(tolerance, eta)
        discarded = eta * 2 / math.pi * math.atan(0.5 / k) + (1 - eta) * math.erfc(
            2 * math.sqrt(math.log(2)) * k
        )
        assert discarded <= tolerance * (1 + 1e-14)


def test_fused_accuracy_policy_matches_numpy_values_and_all_derivative_rows():
    x = np.linspace(15, 140, 10001)
    positions = np.array([25.0123, 60.2345, 126.5432])
    areas = np.array([2.0, 7.0, 4.0])
    instrument = ps.ConstantWavelengthInstrument(1.54, 2e-4, -1e-4, 3e-4, 0.006, 0.002)
    accuracy = ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
    result = ps.accumulate_cw_contributions(
        x,
        positions,
        areas,
        instrument,
        ps.PhysicsContribution.neutral(3),
        geometry=ps.FcjGeometry(0.001, 0.001),
        profile_accuracy=accuracy,
        jacobian_layout="dense",
    )
    y, local, global_j = reference.accumulate_cw_fcj(
        x,
        positions,
        areas,
        u_deg2=instrument.u_deg2,
        v_deg2=instrument.v_deg2,
        w_deg2=instrument.w_deg2,
        x_deg=instrument.x_deg,
        y_deg=instrument.y_deg,
        sample_over_radius=0.001,
        detector_over_radius=0.001,
        fast_fcj=True,
        tail_area_tolerance=0.01,
    )
    np.testing.assert_allclose(result.y, y, rtol=2e-10, atol=2e-9)
    np.testing.assert_allclose(result.jacobian, local, rtol=2e-9, atol=2e-7)
    np.testing.assert_allclose(result.derivatives.global_jacobian, global_j, rtol=2e-9, atol=2e-7)
    # Discrete integration includes a small sampling error; areas are not renormalized.
    integral = trapezoid(result.y, x)
    assert 0.989 < integral / sum(areas) < 1.001
    assert trapezoid(x * result.y, x) == pytest.approx(trapezoid(x * y, x), rel=2e-10)


@pytest.mark.parametrize("callbacks", [False, True])
def test_refinement_accuracy_diagnostics_resume_and_policy_mismatch(callbacks):
    request = doublet_request()
    accuracy = ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
    options = rv.RietveldOptions(profile_accuracy=accuracy, max_scaled_parameter_step=1)
    kwargs = {"logger": lambda event: None} if callbacks else {}
    complete = rv.refine(request, options, **kwargs)
    fresh = rv.calculate(
        request.pattern,
        complete.experiment,
        complete.phases,
        background=complete.background,
        profile_accuracy=accuracy,
    )
    np.testing.assert_allclose(complete.calculation.y, fresh.y, rtol=3e-12, atol=3e-9)
    for a, b in zip(complete.calculation.phase_calculations, fresh.phase_calculations, strict=True):
        np.testing.assert_allclose(
            a.derivatives.global_jacobian, b.derivatives.global_jacobian, rtol=3e-10, atol=3e-7
        )
    partial = rv.refine(
        request, replace(options, limits=RefinementLimits(max_iterations=1)), **kwargs
    )
    resumed = rv.refine(request, options, checkpoint=partial.checkpoint, **kwargs)
    np.testing.assert_array_equal(resumed.calculation.y, complete.calculation.y)
    assert resumed.history == complete.history
    assert complete.checkpoint.profile_accuracy == accuracy
    with pytest.raises(ValueError, match="profile accuracy"):
        rv.refine(
            request,
            replace(options, profile_accuracy=ps.ProfileAccuracy()),
            checkpoint=partial.checkpoint,
            **kwargs,
        )


@pytest.mark.parametrize("doublet", [False, True])
def test_project_roundtrip_preserves_accuracy_and_calculation(tmp_path, doublet):
    request = doublet_request() if doublet else request_from_cif(selection(phase_scale=True))
    accuracy = ps.ProfileAccuracy(fast_fcj=True, tail_area_tolerance=0.01)
    project = ps.RietveldProject(
        request,
        rv.RietveldOptions(profile_accuracy=accuracy, limits=RefinementLimits(max_iterations=1)),
    )
    project.refine()
    expected = project.calculate()
    saved = project.save(tmp_path / "accuracy-project")
    restored = ps.RietveldProject.load(saved)
    assert restored.options.profile_accuracy == accuracy
    assert restored.checkpoint.profile_accuracy == accuracy
    np.testing.assert_array_equal(restored.calculate().y, expected.y)
