"""Bounded count generation and geometric rejection for the blind experiment."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_symbol

# Benchmark scripts share helpers with their sibling modules, without installation.
BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    spec = importlib.util.spec_from_file_location(
        "count_experiment", BENCHMARKS / "solution_element_counts.py"
    )
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
finally:
    sys.path.pop(0)


def test_sampled_counts_follow_supplied_symmetry_and_bound():
    group = space_group_by_symbol("I4/mcm").space_group
    sizes, proposals = experiment.count_hypotheses(group, ["F", "Br", "K"], 52)
    assert sizes == [4, 8, 16, 32]
    assert len(proposals) == 45
    assert all(sum(p.values()) <= 52 for p in proposals)
    assert list(proposals[0]) == ["Br", "K", "F"]
    assert proposals == experiment.count_hypotheses(group, ["K", "F", "Br"], 52)[1]
    assert [sum(p.values()) for p in proposals] == sorted(sum(p.values()) for p in proposals)


@pytest.mark.parametrize("symbol,sizes", [("P1", [1]), ("P-1", [1, 2])])
def test_no_material_specific_multiplicities(symbol, sizes):
    found, _ = experiment.count_hypotheses(space_group_by_symbol(symbol).space_group, ["C"], 8)
    assert found == sizes


@pytest.mark.parametrize(
    "elements,bound",
    [([], 8), (["F", "F"], 8), (["H"], 8), (["Fe2+"], 8), (["Xx"], 8), (["C"], 0), (["C"], True)],
)
def test_count_invalid_inputs(elements, bound):
    with pytest.raises(ValueError):
        experiment.count_hypotheses(space_group_by_symbol("P1").space_group, elements, bound)


def test_periodic_overlap_and_strict_cutoff():
    cell = UnitCell(1, 1, 1, 90, 90, 90)
    assert experiment.has_overlap([[0.01, 0, 0], [0.99, 0, 0]], cell)
    assert not experiment.has_overlap([[0, 0, 0], [0.5, 0, 0]], cell)
    assert not experiment.has_overlap([[0, 0, 0]], cell)
    with pytest.raises(ValueError):
        experiment.has_overlap([[np.nan, 0, 0]], cell)


def test_skew_periodic_overlap_against_brute_images():
    cell = UnitCell(3, 4, 5, 80, 75, 25)
    rng = np.random.default_rng(132)
    xyz = rng.random((10, 3))
    delta = xyz[:, None] - xyz[None, :]
    delta = delta[np.triu_indices(len(xyz), 1)]
    import itertools

    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    minimum = np.linalg.norm(
        (delta[:, None] + shifts) @ cell.geometry().direct_basis.T, axis=-1
    ).min()
    assert experiment.has_overlap(xyz, cell, minimum + 1e-6)
    assert not experiment.has_overlap(xyz, cell, minimum - 1e-6)
