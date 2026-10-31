"""Blocked Gram formation checked against independent equations and complete fits."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.single_crystal.reference.solution_origin_reference import (
    iterative_normal_reference,
    preconditioned_normal_reference,
)
from phasesmith.single_crystal.reference.solution_reference import symmetric_model_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution"
)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("damping", [1e-8, 1e-3, 0.1, 10.0])
@pytest.mark.parametrize("kind", ["random", "rank_deficient", "tiny_columns"])
def test_blocked_step_matches_independent_damped_equations(seed, damping, kind):
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
    value = _core._iterative_normal_step(j.ravel(), r, damping, cached=True, blocked=True)
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
    new = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True, blocked=True)
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
        _core._iterative_normal_step(j, r, damping, cached=True, blocked=True)


def test_zero_rhs_and_repeated_cached_steps_are_deterministic():
    rng = np.random.default_rng(911)
    j = rng.normal(size=(40, 203))
    z = _core._iterative_normal_step(j.ravel(), np.zeros(203), 0.1, cached=True, blocked=True)
    assert z["iterations"] == 0 and np.all(z["step"] == 0)
    r = rng.normal(size=203)
    a = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True, blocked=True)
    b = _core._iterative_normal_step(j.ravel(), r, 0.1, cached=True, blocked=True)
    np.testing.assert_array_equal(a["step"].view(np.uint64), b["step"].view(np.uint64))
    assert a["iterations"] == b["iterations"]


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
@pytest.mark.parametrize("mode,count", [("isotropic", 10), ("anisotropic", 8)])
def test_rejected_damping_trials_reuse_but_accepted_models_invalidate(symbol, mode, count):
    from test_solution_symmetry_adp import model_case

    cell = UnitCell(6, 7, 8, 90, 100, 90)
    sites = [
        ("Si" if i < 2 else "O", point)
        for i, point in enumerate(np.random.default_rng(719).uniform(0.05, 0.95, (count, 3)))
    ]
    model, metadata, hkl, observed, factors = model_case(symbol, cell, sites, mode=mode)
    p = model.initial_parameters(0.015)
    common = dict(iterative=True, cached_normal=True)
    old = model.refine(p, 40, 60.0, None, **common)
    new = model.refine(p, 40, 60.0, None, **common, reuse_normal=True, blocked_normal=True)
    assert new["normal_matrix_builds"] > 0
    assert new["normal_matrix_reuses"] > 0
    assert new["normal_matrix_builds"] <= new["accepted_steps"] + 1
    assert new["blocked_normal_builds"] > 0
    assert new["normal_matrix_builds"] + new["normal_matrix_reuses"] >= new["cached_normal_solves"]
    assert new["normal_cache_fallbacks"] == old["normal_cache_fallbacks"] == 0
    assert new["accepted_steps"] == old["accepted_steps"]
    assert new["termination"] == old["termination"]
    assert old["normal_matrix_builds"] == old["normal_matrix_reuses"] == 0
    np.testing.assert_allclose(old["parameters"], new["parameters"], rtol=2e-5, atol=2e-7)
    for key in old["evaluation"]:
        a, b = np.asarray(old["evaluation"][key]), np.asarray(new["evaluation"][key])
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7, err_msg=key)
    ref = symmetric_model_reference(
        metadata, cell.geometry().direct_basis, hkl, observed, factors, new["parameters"]
    )
    for key in ["objective", "r1", "amplitude_scale", "residual", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(new["evaluation"][key], ref[key], rtol=5e-11, atol=2e-12)
    jac = new["evaluation"]["jacobian"].reshape(len(p), len(hkl))
    for axis in [0, len(p) - 1]:
        delta = np.zeros(len(p))
        delta[axis] = 2e-6

        def reference(q):
            return symmetric_model_reference(
                metadata, cell.geometry().direct_basis, hkl, observed, factors, q
            )["residual"]

        fd = (reference(new["parameters"] + delta) - reference(new["parameters"] - delta)) / 4e-6
        np.testing.assert_allclose(jac[axis], fd, rtol=3e-5, atol=2e-8)
    # A fresh API call must start a fresh cache, even on the same prepared model.
    again = model.refine(p, 40, 60.0, None, **common, reuse_normal=True, blocked_normal=True)
    np.testing.assert_array_equal(
        new["parameters"].view(np.uint64), again["parameters"].view(np.uint64)
    )
    assert new["normal_matrix_builds"] == again["normal_matrix_builds"]
    assert new["normal_matrix_reuses"] == again["normal_matrix_reuses"]


@pytest.mark.parametrize("iterative,cached", [(False, False), (True, False), (True, True)])
def test_small_or_disabled_cache_preserves_original_execution(iterative, cached):
    from test_solution_symmetry_adp import CASES, model_case

    symbol, cell, sites, _ = CASES[0]
    model, *_ = model_case(symbol, cell, sites, mode="isotropic")
    p = model.initial_parameters(0.015)
    a = model.refine(p, 8, 60.0, None, iterative=iterative, cached_normal=cached)
    b = model.refine(
        p,
        8,
        60.0,
        None,
        iterative=iterative,
        cached_normal=cached,
        reuse_normal=True,
        blocked_normal=True,
    )
    np.testing.assert_array_equal(a["parameters"].view(np.uint64), b["parameters"].view(np.uint64))
    assert b["normal_matrix_builds"] == b["normal_matrix_reuses"] == 0


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_models_counts_and_identical_cycle_prefixes(symbol):
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
        short_phase_starts=True,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
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
        real_fft=True,
        fused_phase_workspace=True,
        cached_normal_refinement=True,
        multiplicity_weighted_refinement=True,
        ambiguous_assignments=True,
        cached_assignment_refinement=True,
        specialized_isotropic_refinement=True,
        streamed_peak_search=True,
        reuse_normal_matrices=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    before = data.intensity.copy()
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(data, **common, blocked_normal_matrices=True)
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.refinement.mode == "anisotropic"
    np.testing.assert_allclose(old.best.fractional_xyz, new.best.fractional_xyz, rtol=0, atol=2e-7)
    assert new.shared_search.blocked_normal_matrices
    assert not old.shared_search.blocked_normal_matrices
    for phase in new.shared_search.phase_slices:
        assert phase.cycle_quota == _core._phase_cycle_quota(800, 4, phase.trial, short=True)
        assert phase.completed_cycles <= phase.cycle_quota
    for a, b in zip(old.histories, new.histories, strict=False):
        n = min(len(a), len(b))
        np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))
    np.testing.assert_array_equal(before, data.intensity)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, blocked_normal_matrices=1)
    bad = dict(common, reuse_normal_matrices=False)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **bad, blocked_normal_matrices=True)
