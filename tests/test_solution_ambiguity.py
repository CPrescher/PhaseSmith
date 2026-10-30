"""Coupled categorical proposals with independent displacement-aware equations."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_assignment_reference import ambiguous_assignments_reference
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(getattr(_core, "_PeakCompletionPlan", object), "ambiguous"),
    reason="experimental-solution",
)


def setup(seed=0, symbol="P1"):
    from test_solution_shared_search import unique_operations

    rng = np.random.default_rng(670 + seed)
    cell = UnitCell(6, 7, 8, 90, 100, 90)
    group = space_group_by_symbol(symbol).space_group
    points = rng.uniform(0.1, 0.9, (6, 3))
    if symbol == "Pm":
        points[0, 1] = 0
    hkl = np.array(
        [h for h in itertools.product(range(-3, 4), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    s2 = np.sum((hkl @ np.linalg.inv(cell.geometry().direct_basis)) ** 2, axis=1) / 4
    images = [unique_operations(group, p)[0] for p in points]
    phase = np.column_stack([np.exp(2j * np.pi * (hkl @ x.T)).sum(axis=1) for x in images])
    factors, _ = xray_non_resonant(["C", "N", "O"], np.sqrt(s2))
    u = rng.uniform(0.005, 0.04, len(points))
    observed = np.abs(
        np.sum(
            phase * factors[:, [0, 1, 1, 1, 0, 2]] * np.exp(-8 * np.pi**2 * s2[:, None] * u), axis=1
        )
    )
    observed *= rng.uniform(0.95, 1.05, len(observed))
    labels = [0, 0, 0, 1, 1, 2]
    plan = _core._PeakCompletionPlan(
        list(cell.as_tuple()),
        group._native,
        points.ravel(),
        ["C", "N", "O"],
        hkl.ravel(),
        observed,
        32,
    )
    return plan, phase, factors, observed, labels, u, s2, [len(x) for x in images]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_proposals_match_independent_enumeration_and_displaced_analytic_kernel(seed, symbol):
    plan, phase, factors, observed, labels, u, s2, m = setup(seed, symbol)
    expected = ambiguous_assignments_reference(
        phase, factors, observed, labels, u, s2, [6, 7, 8], m, 4
    )
    actual = plan.ambiguous(labels, u, 4)
    assert [tuple(c["labels"]) for c in actual["candidates"]] == [c[0] for c in expected]
    for c, ref in zip(actual["candidates"], expected, strict=True):
        np.testing.assert_allclose(c["site_u"], ref[1], rtol=2e-12, atol=2e-14)
        assert c["objective"] == pytest.approx(ref[2], rel=3e-12, abs=3e-15)
        value = plan.displaced(c["site_u"]).evaluate(c["labels"], 0)
        assert c["objective"] == pytest.approx(value["objective"], rel=3e-12, abs=3e-15)
        assert c["edits"] == sum(a != b for a, b in zip(labels, c["labels"], strict=True))
    assert actual["evaluations"] > 0
    assert len(actual["candidates"]) <= 4


def test_coupled_relabels_can_change_element_counts_and_preserve_original_inputs():
    plan, *_, labels, u, _s2, _m = setup()
    before = u.copy()
    result = plan.ambiguous(labels, u, 8)
    assert any(
        c["edits"] == 2 and sorted(c["labels"]) != sorted(labels) for c in result["candidates"]
    )
    np.testing.assert_array_equal(u, before)


@pytest.mark.parametrize(
    "invalid", ["short", "nan", "low", "high", "maximum", "coverage", "incomplete", "displaced"]
)
def test_invalid_or_double_applied_displacement_controls(invalid):
    plan, *_, labels, u, _s2, _m = setup()
    maximum = 4
    if invalid == "short":
        u = u[:-1]
    elif invalid == "nan":
        u[0] = np.nan
    elif invalid == "low":
        u[0] = 0
    elif invalid == "high":
        u[0] = 0.21
    elif invalid == "maximum":
        maximum = 9
    elif invalid == "coverage":
        labels = [0] * 6
    elif invalid == "incomplete":
        labels[0] = -1
    else:
        plan = plan.displaced(u)
    with pytest.raises(ValueError):
        plan.ambiguous(labels, u, maximum)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_complete_search_preserves_counts_and_final_anisotropic_mode(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _xyz, species = synthetic(symbol)
    args = dict(
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
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    result = solve_structure_elements(data, **args, ambiguous_assignments=True)
    assert result.target_met
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    assert result.best.refinement.mode == "anisotropic"
    assert result.shared_search.ambiguous_assignments
    for a in result.shared_search.phase_origins:
        assert a.ambiguity_rounds <= 2 and a.ambiguity_relaxations <= 6
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **args, ambiguous_assignments=1)
    args["isotropic_assignment_refinement"] = False
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **args, ambiguous_assignments=True)
