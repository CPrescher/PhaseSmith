"""Dense byte-budget admission, input-only preflight and numerical invariance."""

import itertools
from dataclasses import asdict, replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.hkl import HklData
from phasesmith.io.space_groups import space_group_by_number
from phasesmith.single_crystal.reference.solution_reference import dense_memory_atom_cap_reference
from phasesmith.single_crystal.solution import (
    SolutionOptions,
    estimate_solution_memory,
    solve_structure,
    solve_structure_elements,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_memory_estimate"), reason="updated experimental-solution required"
)


def small_case():
    cell = UnitCell(6, 7, 8, 90, 90, 90)
    hkl = np.array([h for h in itertools.product(range(-3, 4), repeat=3) if h > (0, 0, 0)])
    xyz = np.array([[0.137, 0.241, 0.319], [0.413, 0.627, 0.821]])
    # A simple positive synthetic observation field; solver receives no coordinates.
    amplitude = np.abs(np.exp(2j * np.pi * (hkl @ xyz.T)) @ np.array([14.0, 8.0]))
    return cell, HklData(hkl, amplitude**2, np.ones(len(hkl)))


@pytest.mark.parametrize("value", [0, 7, -1, 2**30 + 1, True, 16.0, np.inf, "16000000"])
def test_invalid_memory_budget(value):
    with pytest.raises(ValueError, match="dense_memory_budget_bytes"):
        SolutionOptions(dense_memory_budget_bytes=value)
    cell, data = small_case()
    with pytest.raises(ValueError, match="dense_memory_budget_bytes"):
        estimate_solution_memory(data, cell=cell, dense_memory_budget_bytes=value)


def test_native_preflight_matches_exhaustive_reference_and_inclusive_boundary():
    cell, data = small_case()
    volume = cell.geometry().volume_angstrom3
    rows = 2 * len(data.hkl)
    edge = 8 * rows * (5 * 7 + 7)
    budgets = [8, edge - 1, edge, edge + 7, 16_000_000, 64 * 1024**2]
    budgets += np.random.default_rng(974).integers(8, 200_000, 30).tolist()
    for budget in budgets:
        report = estimate_solution_memory(data, cell=cell, dense_memory_budget_bytes=budget)
        assert report.expanded_reflection_count == rows
        assert report.maximum_atoms == dense_memory_atom_cap_reference(rows, volume, budget)
        assert report.dense_memory_budget_bytes == budget
    assert (
        estimate_solution_memory(data, cell=cell, dense_memory_budget_bytes=edge).maximum_atoms == 7
    )
    assert (
        estimate_solution_memory(data, cell=cell, dense_memory_budget_bytes=edge - 1).maximum_atoms
        == 6
    )
    with pytest.raises(ValueError, match="cannot fit one atom"):
        solve_structure_elements(
            data,
            cell=cell,
            elements=["Si", "O"],
            options=SolutionOptions(dense_memory_budget_bytes=8),
        )


def test_preflight_uses_allowed_unique_expanded_rows_and_no_reference_counts():
    group = space_group_by_number(61).space_group
    data = HklData(
        np.array([[1, 2, 3], [2, 4, 6], [1, 0, 0], [-2, -4, -6]]), [10, -2, 7, 3], [1, 1, 1, 1]
    )
    cell = UnitCell(8, 9, 10, 90, 90, 90)
    report = estimate_solution_memory(data, cell=cell, space_group=group)
    assert report.expanded_reflection_count == 16
    # Reflection order and duplicated observation errors cannot teach an atom count.
    shuffled = HklData(data.hkl[::-1], data.intensity[::-1], data.sigma[::-1])
    assert report == estimate_solution_memory(shuffled, cell=cell, space_group=group)
    assert report.maximum_atoms == 90


def test_budget_does_not_change_fixed_work_search_arithmetic_or_serialized_policy():
    cell, data = small_case()
    options = SolutionOptions(
        trials=1,
        cycles=30,
        polish_iterations=3,
        refinement="none",
        completion=False,
        max_seconds=30,
    )
    first = solve_structure(data, cell=cell, composition={"Si": 1, "O": 1}, options=options)
    second = solve_structure(
        data,
        cell=cell,
        composition={"Si": 1, "O": 1},
        options=replace(options, dense_memory_budget_bytes=64 * 1024**2),
    )
    assert first.candidates
    assert first.termination == second.termination
    assert first.target_met == second.target_met
    assert np.array_equal(first.density, second.density)
    assert len(first.histories) == len(second.histories)
    for left, right in zip(first.histories, second.histories, strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(first.candidates, second.candidates, strict=True):
        assert left.species == right.species and left.r1 == right.r1
        np.testing.assert_array_equal(left.fractional_xyz, right.fractional_xyz)
    assert first.to_record()["options"]["dense_memory_budget_bytes"] == 16_000_000
    assert second.to_record()["options"]["dense_memory_budget_bytes"] == 64 * 1024**2
    assert SolutionOptions(**asdict(second.options)) == second.options


def test_automatic_element_search_cap_uses_explicit_budget():
    cell, data = small_case()
    budget = 8 * 2 * len(data.hkl) * (5 * 4 + 7)
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["Si", "O"],
        options=SolutionOptions(
            trials=1,
            cycles=1,
            polish_iterations=0,
            symmetry_iterations=0,
            completion=False,
            dense_memory_budget_bytes=budget,
        ),
    )
    assert result.shared_search.maximum_atoms == 4


def test_explicit_over_budget_atom_request_reports_required_and_configured_bytes():
    cell, data = small_case()
    budget = 8 * 2 * len(data.hkl) * (5 * 4 + 7)
    with pytest.raises(ValueError, match=rf"requires .* bytes, configured {budget} bytes"):
        solve_structure_elements(
            data,
            cell=cell,
            elements=["Si", "O"],
            maximum_atoms=5,
            options=SolutionOptions(dense_memory_budget_bytes=budget),
        )
