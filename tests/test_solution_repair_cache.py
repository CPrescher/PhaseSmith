"""Numerical reuse must leave categorical repair decisions unchanged."""

import numpy as np
import pytest
from phasesmith import _core

pytestmark = pytest.mark.skipif(
    not hasattr(getattr(_core, "_PeakCompletionPlan", object), "joint"),
    reason="experimental-solution",
)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_cached_beam_keeps_every_candidate_float_bit_and_logical_score(seed, symbol):
    from test_solution_ambiguity import setup

    plan, phase, factors, observed, labels, u, s2, multiplicities = setup(seed, symbol)
    plan = plan.displaced(u)
    bounds = np.column_stack([1e-5 - u, 0.2 - u])
    old = plan.joint(labels, bounds, 4)
    new = plan.joint(labels, bounds, 4, cached=True)
    from phasesmith.experimental.solution_assignment_reference import (
        joint_assignment_beam_reference,
    )

    reference = joint_assignment_beam_reference(
        phase * np.exp(-8 * np.pi**2 * s2[:, None] * u),
        factors,
        observed,
        labels,
        bounds,
        s2,
        [6, 7, 8],
        multiplicities,
        32,
        4,
    )
    assert [tuple(None if e < 0 else e for e in c["labels"]) for c in new["candidates"]] == [
        c["labels"] for c in reference["candidates"]
    ]
    assert new["evaluations"] == old["evaluations"] and new["depths"] == old["depths"]
    assert [c["labels"] for c in old["candidates"]] == [c["labels"] for c in new["candidates"]]
    for a, b in zip(old["candidates"], new["candidates"], strict=True):
        for k in ["shifts", "objective", "score"]:
            np.testing.assert_array_equal(
                np.asarray(a[k]).view(np.uint64), np.asarray(b[k]).view(np.uint64), err_msg=k
            )
    assert (
        old["profile_builds"]
        == old["profile_reuses"]
        == old["score_only_requests"]
        == old["workspace_bytes"]
        == 0
    )
    assert (
        new["profile_builds"] > 0 and new["profile_reuses"] > 0 and new["score_only_requests"] > 0
    )
    assert 0 < new["workspace_bytes"] <= 4 * 1024 * 1024 + 2048 * 104
    assert new == plan.joint(labels, bounds, 4, cached=True)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_solutions_and_validates_cache_request(symbol):
    import sys
    from pathlib import Path

    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from solution_multicore import ENABLED

    cell, group, data, _, species = synthetic(symbol)
    options = {key: True for key in ENABLED}
    options.update(
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=12,
        symmetry_recycling=True,
        tangent_refinement=True,
        joint_assignment_refinement=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    a = solve_structure_elements(data, **options)
    b = solve_structure_elements(data, **options, cached_repair_evaluation=True)
    assert a.target_met and b.target_met
    assert (
        dict(a.composition)
        == dict(b.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    np.testing.assert_array_equal(a.best.fractional_xyz, b.best.fractional_xyz)
    assert b.shared_search.cached_repair_evaluation and not a.shared_search.cached_repair_evaluation
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **options, cached_repair_evaluation=1)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(
            data, **dict(options, joint_assignment_refinement=False), cached_repair_evaluation=True
        )
