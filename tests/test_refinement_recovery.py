"""Native trace isolation and damping recovery without relaxing fit tolerances."""

from dataclasses import replace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.refinement import RefinementLimits
from phasesmith.refinement import rietveld as rv
from test_rietveld_refinement import P1_CIF, experiment, selection


def occupancy_request():
    x = np.linspace(15, 100, 2501)
    selected = selection(occupancy=True)
    cif = P1_CIF.replace("O1 O 0.41 0.52 0.63 1.0 0.018", "").replace("0.9 0.012", "0.01 0.012")
    q = rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment(),
        cif,
        phase_id="alpha",
        selection=selected,
    )
    phase = q.phases[0]
    truth = replace(
        phase,
        structure=replace(
            phase.structure, sites=(replace(phase.structure.sites[0], occupancy=0.5),)
        ),
    )
    y = rv.calculate(q.pattern, q.experiment, (truth,)).y
    return replace(q, pattern=ps.PowderPattern(x, observed_y=y))


def controls():
    return rv.RietveldOptions(
        max_backtracks=0,
        max_scaled_parameter_step=100,
        objective_tolerance=1e-12,
        parameter_tolerance=1e-10,
        estimate_covariance=False,
        limits=RefinementLimits(
            max_iterations=100, max_evaluations=500, max_consecutive_rejections=100
        ),
    )


def test_native_trace_does_not_change_fit_or_checkpoint():
    q, options = occupancy_request(), controls()
    request = rv._native_request(q, options)
    plain = request.refine(None, None)
    traced = request.refine(None, None, trace=True)
    assert plain.trace_records() == []
    events = traced.trace_records()
    assert events
    times = [e[4] for e in events]
    assert times == sorted(times) and all(np.isfinite(t) and t >= 0 for t in times)
    assert traced.history() == plain.history()
    assert traced.parameter_records() == plain.parameter_records()
    assert traced.evaluations == plain.evaluations
    assert traced.termination_reason == plain.termination_reason == "converged"
    assert traced.checkpoint().parameter_records() == plain.checkpoint().parameter_records()
    np.testing.assert_array_equal(traced.calculated_y(), plain.calculated_y())


def test_python_tiny_damping_recovery_matches_native_solution_and_resumes():
    q, options = occupancy_request(), controls()
    initial = rv.calculate(q.pattern, q.experiment, q.phases)
    checkpoint = rv.RietveldCheckpoint(
        0,
        q.phases,
        q.lattice_domains,
        q.parameters,
        float(0.5 * np.sum((initial.y - q.pattern.observed_y) ** 2)),
        1e-18,
        (),
        q.experiment,
    )
    recovered = rv.refine(q, options, checkpoint=checkpoint)
    native = rv.refine(q, options)
    assert recovered.backend != "native" and native.backend == "native"
    assert recovered.termination_reason.value == native.termination_reason.value == "converged"
    assert recovered.history[0].damping >= options.initial_damping
    assert recovered.metrics.rwp < 1e-8 and native.metrics.rwp < 1e-8
    assert abs(recovered.phases[0].structure.sites[0].occupancy - 0.5) < 2e-9
    np.testing.assert_allclose(recovered.calculation.y, native.calculation.y, rtol=1e-8, atol=1e-8)
    partial = rv.refine(
        q, replace(options, limits=replace(options.limits, max_iterations=1)), checkpoint=checkpoint
    )
    assert partial.history == ()
    resumed = rv.refine(q, options, checkpoint=partial.checkpoint)
    assert resumed.history == recovered.history
    np.testing.assert_array_equal(resumed.calculation.y, recovered.calculation.y)


def test_large_damping_cannot_certify_convergence_far_from_solution():
    q = occupancy_request()
    options = replace(
        controls(), initial_damping=1e20, max_scaled_parameter_step=0.25, max_backtracks=8
    )
    native = rv.refine(q, options)
    python = rv.refine(q, options, logger=lambda event: None)
    assert native.backend == "native" and python.backend != "native"
    for result in (native, python):
        assert result.history
        assert result.termination_reason.value == "converged"
        assert result.metrics.rwp < 1e-8
        assert abs(result.phases[0].structure.sites[0].occupancy - 0.5) < 2e-9
        assert all(
            b.objective < a.objective
            for a, b in zip(result.history, result.history[1:], strict=False)
        )
    np.testing.assert_allclose(native.calculation.y, python.calculation.y, rtol=1e-8, atol=1e-8)
    for logger, complete in ((None, native), (lambda event: None, python)):
        partial = rv.refine(
            q,
            replace(options, limits=replace(options.limits, max_iterations=1)),
            logger=logger,
        )
        resumed = rv.refine(q, options, checkpoint=partial.checkpoint, logger=logger)
        assert resumed.history == complete.history
        np.testing.assert_array_equal(resumed.calculation.y, complete.calculation.y)


def test_projected_stationarity_accepts_a_physical_bound_optimum():
    q = occupancy_request()
    # Data require occupancy 2, outside its [0, 1] domain. The best feasible
    # solution is exactly 1, with a nonzero unprojected gradient.
    q = replace(q, pattern=ps.PowderPattern(q.pattern.x, observed_y=q.pattern.observed_y * 16))
    options = replace(controls(), max_backtracks=8, max_scaled_parameter_step=0.25)
    for logger in (None, lambda event: None):
        result = rv.refine(q, options, logger=logger)
        assert result.termination_reason.value == "converged"
        assert result.phases[0].structure.sites[0].occupancy == 1.0
        assert result.metrics.rwp > 0.5


def test_tiny_user_step_cap_cannot_certify_convergence():
    q = occupancy_request()
    options = replace(controls(), max_scaled_parameter_step=1e-20)
    for logger in (None, lambda event: None):
        result = rv.refine(q, options, logger=logger)
        assert result.termination_reason.value != "converged"
        assert result.metrics.rwp > 0.99


def test_small_damped_objective_change_does_not_stop_a_linear_scale_fit():
    x = np.linspace(15, 100, 2501)
    q = rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selection(phase_scale=True),
        scale=0.5,
    )
    truth = rv.calculate(q.pattern, q.experiment, (replace(q.phases[0], scale=1.0),)).y
    q = replace(q, pattern=ps.PowderPattern(x, observed_y=truth))
    # For this linear problem J = truth * initial_scale in canonical scaled
    # coordinates. Heavy damping gives a tiny accepted improvement while the
    # physical scale is still approximately 0.5 rather than its true value 1.
    curvature = float(np.sum((truth * 0.5) ** 2))
    options = replace(
        controls(),
        initial_damping=1e6 * curvature,
        objective_tolerance=1e-5,
        min_iterations=1,
        max_scaled_parameter_step=0.25,
        max_backtracks=8,
    )
    for logger in (None, lambda event: None):
        result = rv.refine(q, options, logger=logger)
        first = result.history[0]
        assert first.objective_change < options.objective_tolerance * first.objective
        assert first.scaled_step_norm > options.parameter_tolerance
        assert len(result.history) > 1
        assert result.termination_reason.value == "converged"
        assert abs(result.phases[0].scale - 1.0) < 1e-8
        assert result.metrics.rwp < 1e-8


@pytest.mark.parametrize(
    "rejection_limit, expected", [(10, "stagnated"), (3, "repeated_rejections")]
)
def test_coordinate_recovery_shares_one_trial_allowance_and_preserves_runtime_guard(
    rejection_limit, expected
):
    # Several locally promising coordinates, but the user cap prevents any
    # trial from making the required material improvement. Recovery must not
    # multiply max_backtracks by the number of free parameters.
    x = np.linspace(15, 100, 2501)
    q = rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment(),
        P1_CIF,
        phase_id="alpha",
        selection=selection(phase_scale=True, occupancy=True),
        scale=0.5,
    )
    truth = rv.calculate(q.pattern, q.experiment, (replace(q.phases[0], scale=1.0),)).y
    q = replace(q, pattern=ps.PowderPattern(x, observed_y=truth))
    options = replace(
        controls(),
        max_backtracks=8,
        max_scaled_parameter_step=1e-10,
        parameter_tolerance=1e-8,
        objective_tolerance=1e-5,
        limits=RefinementLimits(
            max_iterations=100,
            max_evaluations=500,
            max_consecutive_rejections=rejection_limit,
        ),
    )
    for logger in (None, lambda event: None):
        result = rv.refine(q, options, logger=logger)
        assert result.termination_reason.value == expected
        assert not result.history
        assert result.phases[0].scale == 0.5
        assert result.metrics.rwp > 0.49
