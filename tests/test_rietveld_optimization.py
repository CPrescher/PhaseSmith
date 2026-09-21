"""End-to-end checks for optimized native fixed-spectrum refinement."""

from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.refinement import AffineConstraint, RefinementLimits
from phasesmith.refinement import rietveld as rv
from test_rietveld_refinement import P1_CIF, component_experiment, selection


def doublet_request():
    x = np.linspace(15, 100, 2501)
    selected = replace(selection(phase_scale=True), instrument_parameters=("w_deg2",))
    experiment = replace(component_experiment(), axial_geometry=ps.FcjGeometry(0.001, 0.001))
    base = rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment,
        P1_CIF,
        phase_id="alpha",
        selection=selected,
        intensity_correction=ps.BraggBrentanoUnpolarizedLp(1.54056),
    )
    alpha = base.phases[0]
    beta = replace(alpha, phase_id="beta", name="beta", scale=0.5)
    truth = rv.calculate(base.pattern, experiment, (alpha, beta)).y
    phases = (replace(alpha, scale=0.6), replace(beta, scale=0.3))
    experiment = replace(experiment, instrument=replace(experiment.instrument, w_deg2=2.2e-4))
    params = rv.build_parameter_set(phases, (None, None), selected, experiment=experiment)
    return rv.RietveldInput(
        ps.PowderPattern(
            x, observed_y=truth, uncertainty=np.sqrt(np.maximum(truth, 1)), mask=(x < 49) | (x > 52)
        ),
        experiment,
        phases,
        (None, None),
        params,
        selection=selected,
        constraints=(
            AffineConstraint(rv.phase_scale_key("beta"), rv.phase_scale_key("alpha"), 0.5, 0),
        ),
    )


def test_native_doublet_preserves_ties_masks_and_python_reference_fit():
    request = doublet_request()
    options = rv.RietveldOptions(max_scaled_parameter_step=1)
    native = rv.refine(request, options)
    reference = rv._refine_with_executor(request, options)
    assert native.backend == "native"
    assert any(item.cg_iterations == 0 for item in native.history)
    assert native.metrics.rwp < 1e-7
    assert native.phases[0].scale == pytest.approx(1, rel=2e-7)
    assert native.phases[1].scale == pytest.approx(0.5 * native.phases[0].scale, abs=1e-15)
    np.testing.assert_array_equal(native.metrics.included, request.pattern.mask)
    np.testing.assert_allclose(native.calculation.y, reference.calculation.y, rtol=3e-7, atol=2e-7)
    assert native.covariance is not None
    assert np.isfinite(native.covariance).all()
    assert native.experiment.radiation == request.experiment.radiation


def test_native_doublet_checkpoint_continues_and_callbacks_keep_reference_path():
    request = doublet_request()
    options = rv.RietveldOptions(estimate_covariance=False, max_scaled_parameter_step=1)
    partial = rv.refine(request, replace(options, limits=RefinementLimits(max_iterations=1)))
    assert partial.backend == "native"
    assert partial.checkpoint._native is not None
    resumed = rv.refine(request, options, checkpoint=partial.checkpoint)
    complete = rv.refine(request, options)
    np.testing.assert_array_equal(resumed.calculation.y, complete.calculation.y)
    events = []
    reference = rv.refine(request, options, logger=events.append)
    assert reference.backend != "native"
    assert events
    assert reference.metrics.rwp < 1e-7


def test_doublet_custom_linearization_budget_keeps_reference_path():
    request = doublet_request()
    result = rv.refine(
        request,
        rv.RietveldOptions(
            max_linearization_elements=0,
            estimate_covariance=False,
            max_scaled_parameter_step=1,
        ),
    )
    assert result.backend != "native"
    assert result.metrics.rwp < 1e-7


def assert_complete_diagnostics(result, request):
    fresh = rv.calculate(
        request.pattern,
        result.experiment,
        result.phases,
        background=result.background,
        support_fwhm=20,
    )
    np.testing.assert_allclose(result.calculation.y, fresh.y, rtol=3e-13, atol=3e-11)
    for actual, expected in zip(
        result.calculation.phase_calculations, fresh.phase_calculations, strict=True
    ):
        np.testing.assert_allclose(actual.y, expected.y, rtol=3e-13, atol=3e-11)
        np.testing.assert_array_equal(
            actual.reflections.component_index, expected.reflections.component_index
        )
        np.testing.assert_allclose(
            actual.reflections.integrated_intensity,
            expected.reflections.integrated_intensity,
            rtol=3e-13,
            atol=3e-11,
        )
        left, right = actual.accumulation.derivatives, expected.accumulation.derivatives
        assert left.global_parameter_names == right.global_parameter_names
        np.testing.assert_allclose(
            left.global_jacobian, right.global_jacobian, rtol=3e-12, atol=3e-8
        )
        np.testing.assert_allclose(left.local.values, right.local.values, rtol=3e-12, atol=3e-8)


def test_native_diagnostics_preserve_full_axial_rows_without_python_recalculation(monkeypatch):
    request = doublet_request()

    def forbidden(*args, **kwargs):
        raise AssertionError("native result must not recalculate through Python")

    with monkeypatch.context() as scope:
        scope.setattr(rv, "calculate", forbidden)
        result = rv.refine(request, rv.RietveldOptions(max_scaled_parameter_step=1))
    assert_complete_diagnostics(result, request)


@pytest.mark.parametrize("initial_scale", [0.0, 0.6])
def test_scale_basis_preserves_constraints_covariance_and_exact_resume(initial_scale):
    request = doublet_request()
    selected = selection(phase_scale=True)
    phases = tuple(
        replace(p, scale=initial_scale * (0.5 if i else 1)) for i, p in enumerate(request.phases)
    )
    truth = rv.calculate(
        request.pattern,
        request.experiment,
        tuple(replace(p, scale=0.5 if i else 1) for i, p in enumerate(phases)),
    ).y
    request = replace(
        request,
        phases=phases,
        selection=selected,
        pattern=replace(
            request.pattern, observed_y=truth, uncertainty=np.sqrt(np.maximum(truth, 1))
        ),
        parameters=rv.build_parameter_set(
            phases, (None, None), selected, experiment=request.experiment
        ),
    )
    options = rv.RietveldOptions(max_scaled_parameter_step=0.25)
    result = rv.refine(request, options)
    reference = rv._refine_with_executor(request, options)
    assert result.metrics.rwp < 1e-7
    assert result.phases[0].scale == pytest.approx(1, rel=2e-7)
    assert result.phases[1].scale == pytest.approx(0.5, rel=2e-7)
    np.testing.assert_allclose(result.covariance, reference.covariance, rtol=2e-8, atol=1e-15)
    assert_complete_diagnostics(result, request)
    partial = rv.refine(request, replace(options, limits=RefinementLimits(max_iterations=1)))
    resumed = rv.refine(request, options, checkpoint=partial.checkpoint)
    np.testing.assert_array_equal(resumed.calculation.y, result.calculation.y)
    assert resumed.history == result.history
