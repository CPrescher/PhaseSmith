"""Independent stopping decisions, physical equations and full synthetic recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.single_crystal.reference.solution_reference import symmetric_model_reference
from phasesmith.single_crystal.reference.solution_schedule_reference import phase_plateau_ready

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_phase_plateau_ready"), reason="experimental-solution"
)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("omit", [False, True])
def test_phase_stop_matches_reference_for_thresholds_and_progress(seed, omit):
    rng = np.random.default_rng(seed)
    for n in [0, 32, 96, 127, 128, 129, 160, 192, 256]:
        for level in [0.79, 0.8, 0.81, 0.89]:
            a = level + rng.uniform(-0.001, 0.001, n)
            assert _core._phase_plateau_ready(a, omit) == phase_plateau_ready(
                a, omit_every_third=omit
            )
    a = np.full(128, 0.81)
    assert _core._phase_plateau_ready(a, omit)
    a[-1] = 0.79
    assert _core._phase_plateau_ready(a, omit)
    a[-1] = 0.82
    assert not _core._phase_plateau_ready(a, omit)
    a[-1] = np.nan
    assert not _core._phase_plateau_ready(a, omit)


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
def test_assignment_convergence_keeps_independent_values_and_derivatives(symbol):
    from test_solution_symmetry_adp import model_case

    cell = UnitCell(6, 7, 8, 90, 100, 90)
    sites = [
        ("Si" if i < 2 else "O", p)
        for i, p in enumerate(np.random.default_rng(719).uniform(0.05, 0.95, (10, 3)))
    ]
    model, meta, hkl, observed, factors = model_case(symbol, cell, sites, mode="isotropic")
    p = model.initial_parameters(0.015)
    common = dict(iterative=True, cached_normal=True, reuse_normal=True, blocked_normal=True)
    old = model.refine(p, 40, 60.0, None, **common)
    new = model.refine(p, 40, 60.0, None, **common, assignment_stopping=True)
    assert new["accepted_steps"] <= old["accepted_steps"]
    assert new["evaluation"]["objective"] <= old["evaluation"]["objective"] * 1.0001
    ref = symmetric_model_reference(
        meta, cell.geometry().direct_basis, hkl, observed, factors, new["parameters"]
    )
    for key in ["objective", "r1", "residual", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(new["evaluation"][key], ref[key], rtol=5e-11, atol=2e-12)
    jac = new["evaluation"]["jacobian"].reshape(len(p), len(hkl))
    for axis in [0, len(p) - 1]:
        d = np.zeros(len(p))
        d[axis] = 2e-6

        def evaluate(q):
            return symmetric_model_reference(
                meta, cell.geometry().direct_basis, hkl, observed, factors, q
            )["residual"]

        np.testing.assert_allclose(
            jac[axis],
            (evaluate(new["parameters"] + d) - evaluate(new["parameters"] - d)) / 4e-6,
            rtol=3e-5,
            atol=2e-8,
        )
    again = model.refine(p, 40, 60.0, None, **common, assignment_stopping=True)
    np.testing.assert_array_equal(
        new["parameters"].view(np.uint64), again["parameters"].view(np.uint64)
    )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_models_counts_and_identical_cycle_prefixes(symbol):
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
        short_phase_starts=True,
        blocked_normal_matrices=True,
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
    new = solve_structure_elements(
        data, **common, phase_plateau_stopping=True, assignment_convergence_stopping=True
    )
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.refinement.mode == "anisotropic"
    np.testing.assert_allclose(old.best.fractional_xyz, new.best.fractional_xyz, rtol=0, atol=2e-7)
    assert (
        new.shared_search.phase_plateau_stopping
        and new.shared_search.assignment_convergence_stopping
    )
    assert (
        not old.shared_search.phase_plateau_stopping
        and not old.shared_search.assignment_convergence_stopping
    )
    for phase in new.shared_search.phase_slices:
        assert phase.cycle_quota == _core._phase_cycle_quota(800, 4, phase.trial, short=True)
        assert phase.completed_cycles <= phase.cycle_quota
    for a, b in zip(old.histories, new.histories, strict=False):
        n = min(len(a), len(b))
        np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))
    np.testing.assert_array_equal(before, data.intensity)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, phase_plateau_stopping=1)
    bad = dict(common, work_sliced_phasing=False)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **bad, phase_plateau_stopping=True)
