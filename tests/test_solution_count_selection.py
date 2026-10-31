"""Count-policy ablation, independent score, and required-type recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.reference.solution_count_reference import count_ranking_reference
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements


def test_count_score_rejects_a_small_fit_gain_from_a_large_model():
    scores, order = count_ranking_reference([0.10, 0.099, 0.02], [10, 30, 20], 200)
    assert list(order) == [2, 0, 1]
    assert scores[1] > scores[0]
    assert np.isfinite(count_ranking_reference([0], [1], 10)[0]).all()
    for j, p, n in [([-1], [1], 10), ([np.nan], [1], 10), ([1], [1.5], 10), ([1], [1], 0)]:
        with pytest.raises(ValueError):
            count_ranking_reference(j, p, n)


@pytest.mark.skipif(not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution")
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_count_policy_ablation_recovers_counts_and_uses_the_independent_score(symbol, adaptive):
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["O", "Fe"],
        space_group=group,
        maximum_atoms=32,
        adaptive_counts=adaptive,
        symmetry_recycling=True,
        tangent_refinement=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert len(result.best.species) == len(species)
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    audit = result.shared_search
    assert audit.adaptive_counts is adaptive
    assert bool(audit.bounded_map_handoffs) is adaptive
    if not adaptive:
        assert audit.count_prefix_screenings == 0
    expected, _ = count_ranking_reference(
        [result.best.objective], [result.best.refinement.parameter_count], result.family_count
    )
    assert audit.accepted_moves[-1].score == pytest.approx(expected[0], rel=3e-12, abs=3e-10)
    assert result.to_record()["shared_search"]["adaptive_counts"] is adaptive


@pytest.mark.skipif(not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution")
@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_count_policy_requires_boolean(value):
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic()
    with pytest.raises(TypeError, match="adaptive_counts"):
        solve_structure_elements(
            data, cell=cell, elements=["Fe", "O"], space_group=group, adaptive_counts=value
        )


@pytest.mark.skipif(not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution")
def test_disabling_density_guidance_also_disables_density_based_count_batches():
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic("P-1")
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["Fe", "O"],
        space_group=group,
        maximum_atoms=32,
        adaptive_counts=True,
        integrated_peak_density=False,
        symmetry_recycling=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert len(result.best.species) == len(species)
    assert result.shared_search.density_evaluations == 0
    assert result.shared_search.count_prefix_screenings == 0
