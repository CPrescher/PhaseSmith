"""Shared orbit cache equations, categorical completion, and frozen-topology derivatives."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant
from phasesmith.single_crystal.reference.solution_origin_reference import peak_completion_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PeakCompletionPlan"), reason="experimental-solution"
)


def setup(symbol="P1", perturb=True, radius=3):
    cell = UnitCell(7, 7, 9, 90, 90, 90)
    group = space_group_by_symbol(symbol).space_group
    rotations = np.asarray([op.rotation for op in group.operations])
    translations = np.asarray([[float(v) for v in op.translation] for op in group.operations])
    points = np.array([[0.137, 0.259, 0.381], [0.411, 0.217, 0.639], [0.731, 0.183, 0.552]])
    if symbol == "Pm":
        points[0, 1] = 0
    if symbol == "Fm-3m":
        cell = UnitCell(7, 7, 7, 90, 90, 90)
        points = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.0, 0.0]])
    hkl = np.array(
        [
            h
            for h in itertools.product(range(-radius, radius + 1), repeat=3)
            if any(h) and (symbol != "Fm-3m" or sum(v * v for v in h) <= radius * radius)
        ],
        dtype=np.int64,
    )
    hkl = hkl[~group.systematic_absences(hkl)]
    s_squared = np.sum((hkl @ np.linalg.inv(cell.geometry().direct_basis)) ** 2, axis=1) / 4
    factors, _ = xray_non_resonant(["Fe", "O"], np.sqrt(s_squared))
    actual = points[:2].copy()
    if perturb:
        actual += [0.003, 0.004, 0.005]
        if symbol == "Pm":
            actual[0, 1] = 0
    total = np.zeros(len(hkl), complex)
    for i, p in enumerate(actual):
        images = []
        for r, t in zip(rotations, translations, strict=True):
            q = (r @ p + t) % 1
            if not any(np.linalg.norm((q - old + 0.5) % 1 - 0.5) < 1e-8 for old in images):
                images.append(q)
        total += np.exp(2j * np.pi * (hkl @ np.asarray(images).T)).sum(axis=1) * factors[:, i]
    observed = np.abs(total) * np.exp(-8 * np.pi**2 * 0.01 * s_squared)
    return cell, group, points, hkl, observed, rotations, translations, factors, s_squared


def make_plan(cell, group, points, hkl, observed):
    return _core._PeakCompletionPlan(
        list(cell.as_tuple()), group._native, points.ravel(), ["Fe", "O"], hkl.ravel(), observed, 32
    )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm", "P4"])
def test_cached_completion_values_and_coordinate_jets_are_independent(symbol):
    cell, group, p, h, observed, r, t, f, s2 = setup(symbol)
    plan = make_plan(cell, group, p, h, observed)
    labels = [0, 1, -1]
    actual = plan.evaluate(labels, 2)
    reference = peak_completion_reference(p, labels, h, r, t, f, s2, observed, 0.01)
    assert actual["objective"] == pytest.approx(reference["objective"], rel=2e-11, abs=2e-14)
    np.testing.assert_allclose(actual["gradient"], reference["gradient"], rtol=2e-10, atol=2e-12)
    for site in range(2):
        for axis in range(3):
            if symbol == "Pm" and site == 0 and axis == 1:
                continue
            delta = np.zeros_like(p)
            delta[site, axis] = 1e-7
            left = make_plan(cell, group, p - delta, h, observed).evaluate(labels, 2)["objective"]
            right = make_plan(cell, group, p + delta, h, observed).evaluate(labels, 2)["objective"]
            assert actual["gradient"][site][axis] == pytest.approx(
                (right - left) / 2e-7, rel=2e-5, abs=3e-8
            )


@pytest.mark.parametrize("symbol,atoms", [("P1", 2), ("P-1", 4), ("Pm", 3), ("P4", 8)])
def test_shared_completion_removes_spurious_peaks_without_supplied_counts(symbol, atoms):
    cell, group, p, h, observed, _, _, _, _ = setup(symbol, False)
    result = make_plan(cell, group, p, h, observed).search(
        np.array([52.0, 16.0, 12.0]), np.array([1.7, 2.0, 2.3])
    )
    best = result["candidates"][0]
    assert best["labels"] == [0, 1, -1]
    assert best["atoms"] == atoms
    assert best["objective"] < 1e-24
    assert result["evaluations"] > 0


def test_full_pipeline_handles_default_cap_and_redundant_cubic_stabilizers():
    from phasesmith.io.hkl import HklData
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements

    cell, group, _, hkl, observed, *_ = setup("Fm-3m", False, 6)
    intensity = observed**2
    data = HklData(hkl, intensity, np.maximum(1.0, 0.02 * intensity))
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=128,
        symmetry_recycling=True,
        tangent_refinement=True,
        wilson_normalization=True,
        phase_quality_selection=True,
        phase_origin_selection=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert dict(result.composition) == {"Fe": 4, "O": 8}
    assert not any(s.diagnostic for a in result.shared_search.phase_origins for s in a.settings)


@pytest.mark.parametrize("seed", range(4))
def test_randomized_completion_equations(seed):
    cell, group, p, h, observed, r, t, f, s2 = setup()
    p += np.random.default_rng(seed).uniform(-0.02, 0.02, p.shape)
    labels = [0, 1, 0]
    value = make_plan(cell, group, p, h, observed).evaluate(labels, 3)
    reference = peak_completion_reference(p, labels, h, r, t, f, s2, observed, 0.02)
    assert value["objective"] == pytest.approx(reference["objective"], rel=2e-12, abs=2e-14)
    np.testing.assert_allclose(value["gradient"], reference["gradient"], rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("invalid", ["unsorted", "zero", "nonfinite", "elements", "cap"])
def test_completion_rejects_invalid_boundaries(invalid):
    cell, group, p, h, observed, *_ = setup()
    elements, cap = ["Fe", "O"], 32
    if invalid == "unsorted":
        h = h[::-1].copy()
    elif invalid == "zero":
        observed[:] = 0
    elif invalid == "nonfinite":
        p[0, 0] = np.nan
    elif invalid == "elements":
        elements = ["Fe", "Fe"]
    else:
        cap = 129
    with pytest.raises(ValueError):
        _core._PeakCompletionPlan(
            list(cell.as_tuple()), group._native, p.ravel(), elements, h.ravel(), observed, cap
        )
