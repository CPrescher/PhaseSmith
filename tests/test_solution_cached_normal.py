"""Guarded cached normal equations against independent solves and complete fits."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_origin_reference import (
    iterative_normal_reference,
    preconditioned_normal_reference,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution"
)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("damping", [1e-8, 1e-3, 0.1, 10.0])
@pytest.mark.parametrize("kind", ["random", "rank_deficient", "tiny_columns"])
def test_cached_step_matches_independent_damped_equations(seed, damping, kind):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(64, 259))
    j *= np.geomspace(0.01, 100, 64)[:, None]
    if kind == "rank_deficient":
        j[-1] = j[0]
        j[-2] = 2 * j[1]
    elif kind == "tiny_columns":
        j[-1] = 0
        j[-2] *= 1e-12
    r = rng.normal(size=j.shape[1])
    value = _core._iterative_normal_step(j.ravel(), r, damping, cached=True)
    expected = iterative_normal_reference(j, r, damping)
    normal, rhs, scale = preconditioned_normal_reference(j, r, damping)
    x = value["step"] / scale
    # Same scaled original-system bound as the matrix-free reference tests.
    assert np.linalg.norm(normal @ x - rhs) <= 2e-8 * np.linalg.norm(rhs)
    # Near singular modes are weakly identified; compare predicted data and cost.
    np.testing.assert_allclose(j.T @ value["step"], j.T @ expected, rtol=5e-5, atol=2e-6)
    assert 0 < value["iterations"] <= 128
    if damping >= 1e-3 and kind == "random":
        assert value["used_cache"]


@pytest.mark.parametrize("n,m", [(8, 100), (40, 45)])
def test_unsuitable_shapes_use_the_unchanged_matrix_free_step(n, m):
    rng = np.random.default_rng(29)
    j, r = rng.normal(size=(n, m)), rng.normal(size=m)
    old = _core._iterative_normal_step(j.ravel(), r, 0.1)
    new = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True)
    np.testing.assert_array_equal(old["step"].view(np.uint64), new["step"].view(np.uint64))
    assert not new["used_cache"] and not new["fell_back"]


@pytest.mark.parametrize("invalid", ["nan", "length", "zero", "negative"])
def test_invalid_linear_systems_fail(invalid):
    j, r, damping = np.ones(64 * 140), np.ones(140), 0.1
    if invalid == "nan":
        j[0] = np.nan
    elif invalid == "length":
        j = j[:-1]
    elif invalid == "zero":
        damping = 0
    else:
        damping = -1
    with pytest.raises(ValueError):
        _core._iterative_normal_step(j, r, damping, cached=True)


def test_zero_rhs_and_repeated_cached_steps_are_deterministic():
    rng = np.random.default_rng(911)
    j = rng.normal(size=(40, 203))
    z = _core._iterative_normal_step(j.ravel(), np.zeros(203), 0.1, cached=True)
    assert z["iterations"] == 0 and np.all(z["step"] == 0)
    r = rng.normal(size=203)
    a = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True)
    b = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True)
    np.testing.assert_array_equal(a["step"].view(np.uint64), b["step"].view(np.uint64))
    assert a["iterations"] == b["iterations"]


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
def test_complete_anisotropic_fits_preserve_original_objective_and_derivatives(symbol):
    from test_solution_symmetry_adp import model_case

    sites = [("Si" if i < 2 else "O", p) for i, p in enumerate(
        np.random.default_rng(131).uniform(0.05, 0.95, (8, 3))
    )]
    model, _, _, _, _ = model_case(symbol, UnitCell(6, 7, 8, 90, 100, 90), sites)
    p = model.initial_parameters(0.015)
    old = model.refine(p, 8, 60.0, None, iterative=True)
    new = model.refine(p, 8, 60.0, None, iterative=True, cached_normal=True)
    assert new["cached_normal_solves"] > 0
    for field in ("objective", "r1", "residual", "jacobian", "fractional_xyz", "u_cartesian"):
        np.testing.assert_allclose(
            new["evaluation"][field], old["evaluation"][field], rtol=2e-5, atol=2e-7
        )
    # Evaluation kernel is untouched and still computes analytic values/derivatives together.
    evaluated = model.evaluate(new["parameters"])
    np.testing.assert_array_equal(evaluated["jacobian"], new["evaluation"]["jacobian"])
    again = model.refine(p, 8, 60.0, None, iterative=True, cached_normal=True)
    np.testing.assert_array_equal(
        again["parameters"].view(np.uint64), new["parameters"].view(np.uint64)
    )


def test_public_control_validates_dependency_and_records_final_fit_execution():
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic("P1")
    common = dict(cell=cell, space_group=group, elements=["Fe", "O"])
    with pytest.raises(ValueError, match="requires iterative"):
        solve_structure_elements(data, **common, cached_normal_refinement=True)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(
            data, **common, iterative_refinement=True, cached_normal_refinement=1
        )
    common.update(
        maximum_atoms=12,
        symmetry_recycling=True,
        tangent_refinement=True,
        wilson_normalization=True,
        phase_quality_selection=True,
        phase_origin_selection=True,
        chemical_assignment=True,
        iterative_refinement=True,
        score_only_evaluation=True,
        reusable_fourier_workspaces=True,
        work_sliced_phasing=True,
        isotropic_assignment_refinement=True,
        filtered_peak_search=True,
        batched_fft=True,
        options=SolutionOptions(
            phasing="dual_space", trials=4, cycles=800,
            max_seconds=60, r1_target=0.005, completion=False,
        ),
    )
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(data, **common, cached_normal_refinement=True)
    assert old.target_met and new.target_met
    assert new.composition == old.composition
    assert new.shared_search.cached_normal_refinement
    assert new.best.refinement.mode == old.best.refinement.mode == "anisotropic"
    # This exact synthetic target stops at the gradient gate before a linear solve.
    assert new.shared_search.cached_normal_solves == 0
    np.testing.assert_array_equal(new.best.fractional_xyz, old.best.fractional_xyz)


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
def test_small_final_models_keep_cholesky_and_original_fused_derivatives(symbol):
    from test_solution_symmetry_adp import model_case

    sites = [("Si", (0.137, 0.281, 0.419)), ("O", (0.621, 0.781, 0.813))]
    model, metadata, _, _, _ = model_case(symbol, UnitCell(6, 7, 8, 90, 100, 90), sites)
    assert 12 <= metadata["parameter_count"] < 32
    p = model.initial_parameters(0.015)
    old = model.refine(p, 8, 60.0, None, iterative=True)
    new = model.refine(p, 8, 60.0, None, iterative=True, cached_normal=True)
    assert new["cached_normal_solves"] > 0
    for key in ("objective", "r1", "residual", "jacobian", "fractional_xyz", "u_cartesian"):
        np.testing.assert_allclose(
            new["evaluation"][key], old["evaluation"][key], rtol=2e-8, atol=2e-10
        )
    again = model.refine(p, 8, 60.0, None, iterative=True, cached_normal=True)
    np.testing.assert_array_equal(
        new["parameters"].view(np.uint64), again["parameters"].view(np.uint64)
    )
