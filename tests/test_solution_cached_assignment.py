"""Checked normal caches in bounded isotropic assignment fits and full searches."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_reference import symmetric_model_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="experimental-solution"
)


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
@pytest.mark.parametrize("count", [4, 10])
def test_short_isotropic_fits_preserve_equations_and_are_deterministic(symbol, count):
    from test_solution_symmetry_adp import model_case

    cell = UnitCell(6, 7, 8, 90, 100, 90)
    sites = [
        ("Si" if i < 2 else "O", p)
        for i, p in enumerate(np.random.default_rng(719).uniform(0.05, 0.95, (count, 3)))
    ]
    model, metadata, hkl, observed, factors = model_case(symbol, cell, sites, mode="isotropic")
    initial = model.initial_parameters(0.015)
    old = model.refine(initial, 8, 60.0, None, iterative=True)
    new = model.refine(initial, 8, 60.0, None, iterative=True, cached_normal=True)
    assert new["cached_normal_solves"] > 0
    assert (len(initial) >= 32) == (count == 10)
    for key in ["objective", "r1", "residual", "jacobian", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(
            new["evaluation"][key], old["evaluation"][key], rtol=2e-5, atol=2e-7
        )
    reference = symmetric_model_reference(
        metadata, cell.geometry().direct_basis, hkl, observed, factors, new["parameters"]
    )
    for key in ["objective", "r1", "residual", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(new["evaluation"][key], reference[key], rtol=5e-11, atol=2e-12)
    jac = np.asarray(new["evaluation"]["jacobian"]).reshape(len(initial), len(hkl))
    for axis in [0, len(initial) - 1]:
        delta = np.zeros(len(initial))
        delta[axis] = 2e-6
        plus = symmetric_model_reference(
            metadata,
            cell.geometry().direct_basis,
            hkl,
            observed,
            factors,
            new["parameters"] + delta,
        )
        minus = symmetric_model_reference(
            metadata,
            cell.geometry().direct_basis,
            hkl,
            observed,
            factors,
            new["parameters"] - delta,
        )
        np.testing.assert_allclose(
            jac[axis], (plus["residual"] - minus["residual"]) / 4e-6, rtol=3e-5, atol=2e-8
        )
    again = model.refine(initial, 8, 60.0, None, iterative=True, cached_normal=True)
    np.testing.assert_array_equal(again["parameters"], new["parameters"])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_complete_assignment_search_keeps_counts_and_final_mode(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _xyz, species = synthetic(symbol)
    common = dict(
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
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(data, **common, cached_assignment_refinement=True)
    expected = {e: int(np.count_nonzero(species == e)) for e in set(species)}
    assert old.target_met and new.target_met
    assert dict(old.composition) == dict(new.composition) == expected
    assert new.best.refinement.mode == "anisotropic"
    assert new.shared_search.cached_assignment_refinement
    assert old.shared_search.assignment_cached_normal_solves == 0
    assert (
        new.shared_search.assignment_cached_refinements <= new.shared_search.screening_refinements
    )
    np.testing.assert_allclose(old.best.fractional_xyz, new.best.fractional_xyz, rtol=0, atol=2e-7)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, cached_assignment_refinement=1)
    for flag in ["iterative_refinement", "isotropic_assignment_refinement"]:
        invalid = dict(common)
        invalid[flag] = False
        with pytest.raises(ValueError, match="requires"):
            solve_structure_elements(data, **invalid, cached_assignment_refinement=True)
