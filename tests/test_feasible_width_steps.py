"""Independent checks of optional coupled-width proposals and public controls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import phasesmith as ps
import pytest
from phasesmith.refinement import AffineConstraint, ConstraintTransform, RefinementLimits
from phasesmith.refinement import rietveld as rv
from phasesmith.refinement._feasible_step import solve_quadratic, width_inequalities
from test_rietveld_refinement import P1_CIF, component_experiment, experiment, selection


def test_quadratic_matches_exhaustive_kkt_solutions():
    a = np.array([[1, 2, 0], [0, -1, 1], [-1, 0, 0], [0, 0, -1]], dtype=float)
    b = np.array([-0.1, -0.2, -0.4, -0.3])
    for seed in range(1, 33):
        basis = np.sin(seed * 0.7 + np.arange(9).reshape(3, 3))
        h = basis.T @ basis + 0.25 * np.eye(3)
        rhs = np.cos(seed + np.arange(3) * 0.3)
        actual = solve_quadratic(h, rhs, (a, b), 1e-9)
        assert actual is not None
        objectives = []
        for mask in range(16):
            indices = [i for i in range(4) if mask & (1 << i)]
            active = a[indices]
            matrix = np.block([[h, -active.T], [active, np.zeros((len(indices), len(indices)))]])
            try:
                result = np.linalg.solve(matrix, np.r_[rhs, b[indices]])
            except np.linalg.LinAlgError:
                continue
            if np.all(a @ result[:3] >= b - 1e-8) and np.all(result[3:] >= -1e-8):
                objectives.append(0.5 * result[:3] @ h @ result[:3] - rhs @ result[:3])
        assert abs(0.5 * actual @ h @ actual - rhs @ actual - min(objectives)) < 1e-9
        assert np.all(a @ actual >= b - 1e-10)


def width_request(exp=None):
    x = np.linspace(15, 100, 2501)
    return rv.RietveldInput.from_cif(
        ps.PowderPattern(x, observed_y=np.zeros_like(x)),
        experiment() if exp is None else exp,
        P1_CIF,
        phase_id="alpha",
        selection=selection(
            phase_scale=True,
            instrument_parameters=(
                "u_deg2",
                "v_deg2",
                "w_deg2",
                "x_deg",
                "y_deg",
                "zero_shift_deg",
            ),
        ),
    )


@pytest.mark.parametrize("exp", [experiment, component_experiment])
def test_width_constraint_derivatives_include_zero_shift_and_affine_bounds(exp):
    q = width_request(exp())
    keys = {s.key.name: s.key for s in q.parameters.specs}
    transform = ConstraintTransform(
        q.parameters,
        (AffineConstraint(keys["v_deg2"], keys["u_deg2"], -0.5),),
    )
    derivative = transform.derivative_matrix()
    calculation = rv.calculate(q.pattern, q.experiment, q.phases)
    positions = np.concatenate(
        [p.reflections.two_theta_deg for p in calculation.phase_calculations]
    )
    rows, lower = width_inequalities(q.experiment, calculation, q.parameters, derivative)
    names = ("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg", "zero_shift_deg")
    indices = {s.key.name: i for i, s in enumerate(q.parameters.specs)}
    initial = np.array(
        [getattr(q.experiment.instrument, n) for n in names[:-1]] + [q.experiment.zero_shift_deg]
    )

    def widths(step):
        values = initial + np.array([derivative[indices[n]] @ step for n in names])
        u, v, w, x, y, zero = values
        theta = (positions + zero - initial[-1]) * np.pi / 360
        return np.column_stack(
            (u * np.tan(theta) ** 2 + v * np.tan(theta) + w, x / np.cos(theta) + y * np.tan(theta))
        ).ravel()

    count = len(transform.free_keys)
    np.testing.assert_allclose(
        lower[: 2 * len(positions)], -0.99 * widths(np.zeros(count)), atol=1e-18
    )
    for i in range(count):
        delta = np.eye(count)[i] * 1e-5
        finite = (widths(delta) - widths(-delta)) / 2e-5
        np.testing.assert_allclose(rows[: len(finite), i], finite, rtol=2e-7, atol=2e-12)
    # Every finite physical bound must also be represented, including v=-u/2.
    offset = 2 * len(positions)
    for i, spec in enumerate(q.parameters.specs):
        for row, bound in (
            (derivative[i], spec.bounds.lower - spec.value),
            (-derivative[i], spec.value - spec.bounds.upper),
        ):
            if np.isfinite(bound) and np.any(row):
                np.testing.assert_array_equal(rows[offset], row)
                assert lower[offset] == bound
                offset += 1
    assert offset == len(rows)


def test_ineligible_and_unverifiable_subproblems_fall_back():
    q = width_request()
    transform = ConstraintTransform(q.parameters, ())
    calculation = rv.calculate(q.pattern, q.experiment, q.phases)
    derivative = transform.derivative_matrix()
    oversized = np.zeros((len(q.parameters.specs), 65))
    assert width_inequalities(q.experiment, calculation, q.parameters, oversized) is None
    specs = list(q.parameters.specs)
    specs[0] = SimpleNamespace(key=SimpleNamespace(module="lattice", name="a_angstrom"))
    assert (
        width_inequalities(q.experiment, calculation, SimpleNamespace(specs=specs), derivative)
        is None
    )
    assert (
        solve_quadratic(
            np.array([[1.0, 2.0], [2.0, 1.0]]), np.ones(2), (np.empty((0, 2)), np.empty(0)), 1e-9
        )
        is None
    )
    assert solve_quadratic(np.eye(2), np.ones(2), (np.eye(2), np.ones(2)), 1e-9) is None


@pytest.mark.parametrize("exp", [experiment, component_experiment])
def test_feasible_native_and_python_fits_preserve_accuracy_and_resume(exp):
    q = width_request(exp())
    true_exp = replace(
        q.experiment,
        instrument=replace(
            q.experiment.instrument,
            u_deg2=3e-4,
            v_deg2=-2e-4,
            w_deg2=2.5e-4,
            x_deg=2e-3,
            y_deg=2.5e-3,
        ),
        zero_shift_deg=0.001,
    )
    truth = rv.calculate(q.pattern, true_exp, q.phases).y
    q = replace(q, pattern=ps.PowderPattern(q.pattern.x, observed_y=truth))
    options = rv.RietveldOptions(
        feasible_width_steps=True,
        objective_tolerance=1e-13,
        parameter_tolerance=1e-10,
        estimate_covariance=False,
        limits=RefinementLimits(max_iterations=100, max_evaluations=2000),
    )
    results = []
    for logger in (None, lambda event: None):
        complete = rv.refine(q, options, logger=logger)
        assert complete.termination_reason.value == "converged"
        assert complete.metrics.rwp < 1e-7
        assert all(
            b.objective < a.objective
            for a, b in zip(complete.history, complete.history[1:], strict=False)
        )
        partial = rv.refine(
            q, replace(options, limits=replace(options.limits, max_iterations=2)), logger=logger
        )
        resumed = rv.refine(q, options, checkpoint=partial.checkpoint, logger=logger)
        assert resumed.history == complete.history
        np.testing.assert_array_equal(resumed.calculation.y, complete.calculation.y)
        results.append(complete)
    assert results[0].backend == "native" and results[1].backend != "native"
    np.testing.assert_allclose(
        results[0].calculation.y, results[1].calculation.y, rtol=2e-7, atol=1e-7
    )


@pytest.mark.parametrize("exp", [experiment, component_experiment])
def test_option_roundtrips_and_defaults_off(tmp_path, exp):
    assert rv.RietveldOptions().feasible_width_steps is False
    with pytest.raises(TypeError, match="boolean"):
        rv.RietveldOptions(feasible_width_steps=1)
    q = width_request(exp())
    options = rv.RietveldOptions(feasible_width_steps=True)
    path = ps.RietveldProject(q, options).save(tmp_path / "feasible")
    restored = ps.RietveldProject.load(path)
    assert restored.options.feasible_width_steps is True


@pytest.mark.parametrize("matrix_free", [False, True])
def test_ineligible_fit_uses_identical_fallback(matrix_free):
    q = width_request()
    if not matrix_free:
        # No refined widths: even a dense solve must use the original path.
        chosen = selection(phase_scale=True)
        q = replace(
            q,
            selection=chosen,
            parameters=rv.build_parameter_set(
                q.phases, q.lattice_domains, chosen, experiment=q.experiment
            ),
        )
    truth = rv.calculate(q.pattern, q.experiment, (replace(q.phases[0], scale=1.1),)).y
    q = replace(q, pattern=ps.PowderPattern(q.pattern.x, observed_y=truth))
    options = rv.RietveldOptions(
        max_linearization_elements=1 if matrix_free else 10_000_000,
        estimate_covariance=False,
        limits=RefinementLimits(max_iterations=2),
    )
    for logger in (None, lambda event: None):
        before = rv.refine(q, options, logger=logger)
        after = rv.refine(q, replace(options, feasible_width_steps=True), logger=logger)
        assert before.history == after.history
        assert before.termination_reason == after.termination_reason
        np.testing.assert_array_equal(before.calculation.y, after.calculation.y)
