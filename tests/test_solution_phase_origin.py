"""Phase-only origins, free gauges, enantiomorphs and independent derivatives."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.reference.solution_origin_reference import phase_origin_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PhaseOriginPlan"), reason="experimental-solution"
)


def fixture(symbol="P-1", hand=1):
    group = space_group_by_symbol(symbol).space_group
    operations = group.operations
    rotations = np.asarray([op.rotation for op in operations], dtype=np.int64)
    translations = np.asarray([[float(v) for v in op.translation] for op in operations])
    rows = set()
    for h in itertools.product(range(-3, 4), repeat=3):
        if not any(h) or sum(v * v for v in h) > 9:
            continue
        orbit = np.asarray(h) @ rotations
        rows.update(map(tuple, np.concatenate([orbit, -orbit])))
    hkl = np.asarray(sorted(rows), dtype=np.int64)
    hkl = hkl[~group.systematic_absences(hkl)]
    xyz, weights = [], []
    for p, weight in [([0.137, 0.259, 0.381], 14), ([0.411, 0.217, 0.639], 8)]:
        unique = []
        for r, t in zip(rotations, translations, strict=True):
            x = (r @ p + t) % 1
            if not any(np.linalg.norm((x - y + 0.5) % 1 - 0.5) < 1e-8 for y in unique):
                unique.append(x)
        xyz.extend(unique)
        weights.extend([weight] * len(unique))
    shift = np.array([0.173, 0.271, 0.349])
    source = hand * (np.asarray(xyz) - shift)
    radial = np.sum(hkl * hkl, axis=1)
    if symbol == "P 31 2 1":
        radial = 4 / 3 * (hkl[:, 0] ** 2 + hkl[:, 0] * hkl[:, 1] + hkl[:, 1] ** 2) + hkl[:, 2] ** 2
    factors = np.exp(-0.02 * radial)
    f = (np.exp(2j * np.pi * (hkl @ source.T)) @ weights) * factors
    amplitudes = np.abs(f)
    grid = np.zeros((32,) * 3, dtype=complex)
    grid[tuple((hkl % 32).T)] = f
    density = np.fft.fftn(grid).real / grid.size
    plan = _core._PhaseOriginPlan(group._native, hkl.ravel(), amplitudes)
    return plan, hkl, amplitudes, f, rotations, translations, density, shift


@pytest.mark.parametrize("symbol", ["P1", "P-1", "P21", "P4", "Fm-3m", "P 31 2 1"])
def test_origin_values_and_derivatives_match_independent_equations(symbol):
    plan, hkl, amp, f, r, t, density, shift = fixture(symbol)
    native = plan.map(density.ravel(), density.shape, 1)
    assert native.evaluate(shift)["alpha"] < 1e-18
    point = shift + np.array([0.003, 0.005, 0.007])
    value = native.evaluate(point)
    expected = phase_origin_reference(hkl, amp, f, r, t, point)
    for key in ["chord", "alpha", "gradient", "hessian", "alpha_gradient"]:
        np.testing.assert_allclose(value[key], expected[key], rtol=3e-10, atol=3e-11)
    for probe in np.random.default_rng(513).uniform(-0.08, 0.08, (4, 3)) + shift:
        actual = native.evaluate(probe)
        reference = phase_origin_reference(hkl, amp, f, r, t, probe)
        for key in ["chord", "alpha", "gradient", "hessian", "alpha_gradient"]:
            np.testing.assert_allclose(actual[key], reference[key], rtol=3e-10, atol=3e-11)
    step = 1e-7
    for axis in range(3):
        delta = np.eye(3)[axis] * step
        left, right = native.evaluate(point - delta), native.evaluate(point + delta)
        assert value["gradient"][axis] == pytest.approx(
            (right["chord"] - left["chord"]) / (2 * step), rel=2e-6, abs=2e-8
        )
        assert value["alpha_gradient"][axis] == pytest.approx(
            (right["alpha"] - left["alpha"]) / (2 * step), rel=2e-6, abs=2e-8
        )
        np.testing.assert_allclose(
            np.asarray(value["hessian"])[:, axis],
            (np.asarray(right["gradient"]) - left["gradient"]) / (2 * step),
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    "symbol,hand", [("P1", 1), ("P-1", 1), ("P21", 1), ("P4", 1), ("P 31 2 1", -1)]
)
def test_fft_seeded_origin_search_recovers_supplied_setting_without_atoms(symbol, hand):
    plan, _, _, _, _, _, density, _ = fixture(symbol, hand)
    result = plan.search(density.ravel(), density.shape)
    assert result["origins"][0]["alpha"] < 1e-10
    if hand == -1:
        assert result["origins"][0]["hand"] == -1
    other = plan.search((density * 713).ravel(), density.shape)
    assert other["origins"][0]["alpha"] < 1e-10


def test_invalid_origin_boundaries_fail_explicitly():
    plan, _, _, _, _, _, density, _ = fixture()
    for shape, source, hand in [
        ((7, 32, 32), density.ravel(), 1),
        (density.shape, density.ravel(), 0),
        (density.shape, np.full(density.size, np.nan), 1),
    ]:
        with pytest.raises(ValueError):
            plan.map(source, shape, hand)


def test_hand_equivalents_do_not_crowd_out_a_distinct_quarter_origin():
    plan, hkl, _, f, _, _, density, _ = fixture("Pm")
    grid = np.zeros(density.shape, dtype=complex)
    keep = hkl[:, 1] % 2 == 0
    grid[tuple((hkl[keep] % 32).T)] = f[keep]
    source = np.fft.fftn(grid).real / grid.size
    result = plan.search(source.ravel(), source.shape, maximum=2)
    assert len(result["origins"]) == 2
    a, b = result["origins"]
    assert a["hand"] == b["hand"] == 1
    assert a["alpha"] < 1e-12 and b["alpha"] < 1e-12
    assert abs((a["shift"][1] - b["shift"][1] + 0.25) % 0.5 - 0.25) == pytest.approx(0.25, abs=1e-7)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
@pytest.mark.parametrize("chemical_assignment", [False, True])
@pytest.mark.parametrize("maximum_atoms", [32, 128])
def test_phase_origin_completion_recovers_counts_and_source_frame(
    symbol, maximum_atoms, chemical_assignment
):
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=maximum_atoms,
        symmetry_recycling=True,
        tangent_refinement=True,
        wilson_normalization=True,
        phase_quality_selection=True,
        phase_origin_selection=True,
        chemical_assignment=chemical_assignment,
        iterative_refinement=chemical_assignment,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met and len(result.best.species) == len(species)
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    best = float("inf")
    for hand in [-1, 1]:
        for anchor in xyz[species == result.best.species[0]]:
            current = (
                hand * result.best.fractional_xyz + anchor - hand * result.best.fractional_xyz[0]
            ) % 1
            error = 0.0
            for element in set(species):
                d = (
                    xyz[species == element, None]
                    - current[None, np.asarray(result.best.species) == element]
                    + 0.5
                ) % 1 - 0.5
                distance = np.linalg.norm(d @ cell.geometry().direct_basis.T, axis=-1)
                error = max(error, distance.min(axis=1).max(), distance.min(axis=0).max())
            best = min(best, error)
    assert best < 1e-5
    assert result.shared_search.phase_origins
    assert result.shared_search.final_refinements <= 4 * result.shared_search.phasing_starts
    assert result.to_record()["shared_search"]["phase_origin_selection"]
