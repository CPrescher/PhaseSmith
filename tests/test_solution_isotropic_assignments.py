"""Ambiguous label proposals and independent-family isotropic ranking before promotion."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
from phasesmith.experimental.solution_assignment_reference import assignment_neighbors_reference
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(getattr(_core, "_PeakCompletionPlan", object), "neighbors"),
    reason="experimental-solution",
)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_neighbors_match_independent_enumeration_with_unequal_displacements(seed, symbol):
    from test_solution_shared_search import unique_operations

    rng = np.random.default_rng(134 + seed)
    cell = UnitCell(6, 7, 8, 90, 100, 90)
    group = space_group_by_symbol(symbol).space_group
    points = rng.uniform(0.1, 0.9, (5, 3))
    if symbol == "Pm":
        points[0, 1] = 0
    hkl = np.array(
        [h for h in itertools.product(range(-3, 4), repeat=3) if h != (0, 0, 0)],
        dtype=np.int64,
    )
    s2 = np.sum((hkl @ np.linalg.inv(cell.geometry().direct_basis)) ** 2, axis=1) / 4
    site_u = rng.uniform(0.005, 0.04, len(points))
    columns = np.column_stack(
        [np.exp(2j * np.pi * (hkl @ unique_operations(group, p)[0].T)).sum(axis=1) for p in points]
    ) * np.exp(-8 * np.pi**2 * s2[:, None] * site_u)
    factors, _ = xray_non_resonant(["C", "O", "Fe"], np.sqrt(s2))
    labels = [0, 1, 2, 1, 0]
    # Noisy observations give a genuine near-tie neighborhood, not only exact minima.
    observed = np.abs(np.sum(columns * factors[:, [0, 2, 1, 1, 0]], axis=1))
    observed *= rng.uniform(0.96, 1.04, len(observed))
    plan = _core._PeakCompletionPlan(
        list(cell.as_tuple()),
        group._native,
        points.ravel(),
        ["C", "O", "Fe"],
        hkl.ravel(),
        observed,
        32,
    ).displaced(site_u)
    expected = assignment_neighbors_reference(columns, factors, observed, labels, 4)
    actual = plan.neighbors(labels, 4)
    assert [tuple(c["labels"]) for c in actual["candidates"]] == [p[0] for p in expected]
    np.testing.assert_allclose(
        [c["objective"] for c in actual["candidates"]],
        [p[1] for p in expected],
        rtol=3e-12,
        atol=3e-15,
    )
    assert actual["evaluations"] > 0
    assert actual["changes"] == 0  # Proposals are not accepted changes.


def test_homometric_assignment_is_retained_as_ambiguous_not_an_improvement():
    from test_solution_peak_completion import make_plan, setup

    cell, group, points, hkl, observed, *_ = setup("P1", False)
    plan = make_plan(cell, group, points[:2], hkl, observed).displaced([0.01, 0.01])
    assert plan.reassign([1, 0])["candidates"] == []
    neighbor = plan.neighbors([1, 0])["candidates"][0]
    assert neighbor["labels"] == [0, 1]
    assert neighbor["objective"] == pytest.approx(plan.evaluate([1, 0], 0)["objective"], abs=1e-24)
    for maximum in (0, 9):
        with pytest.raises(ValueError):
            plan.neighbors([1, 0], maximum)
    with pytest.raises(ValueError):
        plan.neighbors([0, -1])
    with pytest.raises(ValueError):
        plan.neighbors([0, 0])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_counts_geometry_and_requested_final_mode(symbol):
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
    before = data.intensity.copy()
    result = solve_structure_elements(
        data,
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
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    assert result.best.refinement.mode == "anisotropic"
    assert result.shared_search.isotropic_assignment_refinement
    origins = result.shared_search.phase_origins
    assert sum(o.assignment_parent_fits for o in origins) > 0
    for origin in origins:
        assert origin.assignment_parent_fits <= 2
        assert origin.assignment_neighbor_fits <= 2
        assert origin.assignment_retained <= 4
        assert origin.assignment_scores == tuple(sorted(origin.assignment_scores))
        if origin.assignment_families:
            assert origin.assignment_families == origin.screening_families
    # Independently align the full cell in either hand and compare like elements.
    error = np.inf
    for hand in (-1, 1):
        for anchor in xyz[species == result.best.species[0]]:
            model = (
                hand * result.best.fractional_xyz + anchor - hand * result.best.fractional_xyz[0]
            ) % 1
            maximum = 0.0
            for e in set(species):
                d = (
                    xyz[species == e, None]
                    - model[None, np.asarray(result.best.species) == e]
                    + 0.5
                ) % 1 - 0.5
                distance = np.linalg.norm(d @ cell.geometry().direct_basis.T, axis=-1)
                maximum = max(maximum, distance.min(axis=0).max(), distance.min(axis=1).max())
            error = min(error, maximum)
    assert error < 1e-5
    np.testing.assert_array_equal(before, data.intensity)


def test_assignment_flag_requires_origin_and_chemical_path_and_boolean():
    from test_solution_shared_search import synthetic

    cell, group, data, *_ = synthetic("P1")
    common = dict(cell=cell, space_group=group, elements=["Fe", "O"])
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, isotropic_assignment_refinement=1)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **common, isotropic_assignment_refinement=True)
