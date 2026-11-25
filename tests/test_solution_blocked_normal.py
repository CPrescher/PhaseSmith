"""Blocked Gram formation checked against independent equations and complete fits.

Low-level ``_iterative_normal_step(..., blocked=True)`` and small-model
invariance checks live alongside their non-blocked counterparts in
``test_solution_cached_normal.py`` and ``test_solution_normal_reuse.py``.
"""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.single_crystal.reference.solution_reference import symmetric_model_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution"
)


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
