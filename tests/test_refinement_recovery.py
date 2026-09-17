"""Native trace isolation and damping recovery without relaxing fit tolerances."""

from dataclasses import replace

import numpy as np
import phasesmith as ps
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
