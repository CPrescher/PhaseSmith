"""Independent shared phase-jet equations and count-free orbit editing."""

import itertools
import json

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import (
    SolutionCancellation,
    SolutionOptions,
    solve_structure_elements,
    write_solution,
)
from phasesmith.experimental.solution_shared_reference import shared_orbit_reference
from phasesmith.io.hkl import HklData
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SharedOrbitPool"), reason="requires experimental-solution"
)


def unique_operations(group, point):
    images, rotations, translations = [], [], []
    for op in group.operations:
        r, t = np.array(op.rotation), np.array([float(v) for v in op.translation])
        image = (r @ point + t) % 1
        if any(np.linalg.norm((image - p + 0.5) % 1 - 0.5) < 1e-8 for p in images):
            continue
        images.append(image)
        rotations.append(r)
        translations.append(t)
    return np.array(images), np.array(rotations), np.array(translations)


@pytest.mark.parametrize(
    "symbol,point,tangents",
    [
        ("P1", [0.31, 0.42, 0.67], np.eye(3)),
        ("P-1", [0.31713, 0.42179, 0.67319], np.eye(3)),
        ("Pm", [0.31, 0, 0.67], np.array([[1, 0, 0], [0, 0, 1]])),
        ("P 31 2 1", [0.31, 0.42, 0.67], np.eye(3)),
    ],
)
def test_shared_labels_and_fused_derivatives(symbol, point, tangents):
    cell = (
        UnitCell(5, 5, 7, 90, 90, 120) if symbol == "P 31 2 1" else UnitCell(6, 7, 8, 90, 100, 90)
    )
    group = space_group_by_symbol(symbol).space_group
    hkl = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=np.int64)
    hkl = hkl[~group.systematic_absences(hkl)]
    fixed, _, _ = unique_operations(group, np.array([0.12, 0.23, 0.34]))
    s = np.linalg.norm(hkl @ np.linalg.inv(cell.geometry().direct_basis), axis=1) / 2
    labels = ["Si", "O", "Fe"]
    f, _ = xray_non_resonant([*["Si"] * len(fixed), *labels], s)
    f *= np.exp(-8 * np.pi**2 * 0.014 * s * s)[:, None]
    observed = np.random.default_rng(631).uniform(0.2, 35, len(hkl))
    pool = _core._SharedOrbitPool(
        list(cell.as_tuple()),
        group._native,
        ["Si"] * len(fixed),
        fixed.ravel(),
        labels,
        hkl.ravel(),
        observed,
        0.014,
    )
    images, rotations, translations = unique_operations(group, np.array(point))
    reference = shared_orbit_reference(
        hkl, observed, fixed, f[:, : len(fixed)], f[:, len(fixed) :], point, rotations, translations
    )
    result = pool.evaluate(point)
    assert [r["element"] for r in result] == labels
    np.testing.assert_allclose(reference["q"][np.all(hkl == 0, axis=1)], len(images), atol=1e-14)
    np.testing.assert_allclose(reference["dq"][np.all(hkl == 0, axis=1)], 0, atol=1e-14)
    for native, expected in zip(result, reference["evaluations"], strict=True):
        for key in ("objective", "r1", "amplitude_scale", "gradient", "residual"):
            np.testing.assert_allclose(native[key], expected[key], rtol=2e-11, atol=3e-13)
        np.testing.assert_allclose(
            np.array(native["jacobian"]).reshape(3, -1),
            expected["jacobian"],
            rtol=2e-11,
            atol=3e-13,
        )
    for tangent in tangents:
        step = 1e-6
        plus, minus = (
            pool.evaluate(np.array(point) + step * tangent),
            pool.evaluate(np.array(point) - step * tangent),
        )
        for index, native in enumerate(result):
            finite = (plus[index]["residual"] - minus[index]["residual"]) / (2 * step)
            analytic = tangent @ np.array(native["jacobian"]).reshape(3, -1)
            np.testing.assert_allclose(finite, analytic, rtol=3e-6, atol=3e-9)
    scaled = _core._SharedOrbitPool(
        list(cell.as_tuple()),
        group._native,
        ["Si"] * len(fixed),
        fixed.ravel(),
        labels,
        hkl.ravel(),
        observed * 3,
        0.014,
    ).evaluate(point)
    for left, right in zip(result, scaled, strict=True):
        assert abs(left["objective"] - right["objective"]) < 2e-14
        np.testing.assert_allclose(left["jacobian"], right["jacobian"], rtol=1e-11, atol=3e-13)


def synthetic(symbol="Pm"):
    cell = UnitCell(6, 7, 8, 90, 100, 90)
    group = space_group_by_symbol(symbol).space_group
    sites = [("Fe", [0.137, 0.259, 0.382]), ("O", [0.421, 0 if symbol == "Pm" else 0.148, 0.723])]
    if symbol != "P-1":
        sites.append(("O", [0.752, 0.19, 0.158]))
    xyz, species = [], []
    for e, p in sites:
        images, _, _ = unique_operations(group, np.array(p))
        xyz.extend(images)
        species.extend([e] * len(images))
    hkl = np.array(
        [h for h in itertools.product(range(-6, 7), repeat=3) if 0 < sum(i * i for i in h) <= 36],
        dtype=np.int64,
    )
    hkl = hkl[~group.systematic_absences(hkl)]
    s = np.linalg.norm(hkl @ np.linalg.inv(cell.geometry().direct_basis), axis=1) / 2
    f, _ = xray_non_resonant(species, s)
    sf = (f * np.exp(2j * np.pi * (hkl @ np.array(xyz).T))).sum(axis=1) * np.exp(
        -8 * np.pi**2 * 0.012 * s * s
    )
    return (
        cell,
        group,
        HklData(hkl, np.abs(sf) ** 2, np.maximum(1.0, 0.02 * np.abs(sf) ** 2)),
        np.array(xyz),
        np.array(species),
    )


@pytest.mark.parametrize(
    "symbol,counts",
    [("P1", {"Fe": 1, "O": 2}), ("P-1", {"Fe": 2, "O": 2}), ("Pm", {"Fe": 2, "O": 3})],
)
@pytest.mark.parametrize("recycling", [False, True])
def test_infers_counts_and_multiple_orbits_without_composition(symbol, counts, tmp_path, recycling):
    cell, group, data, xyz, species = synthetic(symbol)
    options = SolutionOptions(
        trials=1, cycles=800, max_seconds=10, r1_target=0.005, completion=False
    )
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["O", "Fe"],
        space_group=group,
        options=options,
        maximum_atoms=32,
        symmetry_recycling=recycling,
    )
    assert result.target_met
    assert result.best.trial < len(result.phasing_trials)
    assert dict(result.composition) == counts
    assert result.best.r1 < 1e-6
    assert result.shared_search.phasing_starts == 1
    assert result.shared_search.element_scores == 2 * result.shared_search.orbit_proposals
    assert result.shared_search.required_elements_present
    assert result.shared_search.symmetry_recycling == recycling
    if recycling:
        assert result.shared_search.recycling
        assert all(r.cycles == 10 for r in result.shared_search.recycling)
        assert all(r.predicted_coefficients > 0 for r in result.shared_search.recycling)
    else:
        assert not result.shared_search.recycling
    assert all(
        m.kind in {"initial", "birth", "delete", "relabel", "recycle_batch"}
        for m in result.shared_search.accepted_moves
    )
    assert result.shared_search.accepted_moves[-1].atoms == len(xyz)
    # Compare geometry up to an overall inversion/translation and element permutations.
    best = float("inf")
    for sign in (-1, 1):
        for anchor in xyz[species == result.best.species[0]]:
            shifted = (
                sign * result.best.fractional_xyz + anchor - sign * result.best.fractional_xyz[0]
            ) % 1
            errors = []
            for element in counts:
                delta = (
                    xyz[species == element, None]
                    - shifted[None, np.array(result.best.species) == element]
                    + 0.5
                ) % 1 - 0.5
                distances = np.linalg.norm(delta @ cell.geometry().direct_basis.T, axis=-1)
                errors.extend(distances.min(axis=1))
                errors.extend(distances.min(axis=0))
            best = min(best, max(errors))
    assert best < 1e-5
    record = result.to_record()
    assert record["schema"] == "phasesmith.single-crystal-solution.v1"
    assert record["composition_source"] == "inferred_shared_orbit_search"
    assert record["input_elements"] == ["Fe", "O"]
    write_solution(result, tmp_path / "output")
    assert json.loads((tmp_path / "output/result.json").read_text())["shared_search"][
        "accepted_moves"
    ]
    assert not result.best.fractional_xyz.flags.writeable
    # Same complete run with reversed user type order remains bitwise deterministic.
    again = solve_structure_elements(
        data,
        cell=cell,
        elements=["Fe", "O"],
        space_group=group,
        options=options,
        maximum_atoms=32,
        symmetry_recycling=recycling,
    )
    np.testing.assert_array_equal(result.best.fractional_xyz, again.best.fractional_xyz)
    assert result.shared_search == again.shared_search


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elements": []},
        {"elements": ["O", "O"]},
        {"elements": ["H"]},
        {"elements": ["Fe2+"]},
        {"elements": ["Xx"]},
        {"elements": "Fe"},
        {"maximum_atoms": 0},
        {"maximum_atoms": True},
        {"beam_width": 9},
        {"search_rounds": 0},
        {"symmetry_recycling": 1},
        {"recycling_cycles": 0},
        {"recycling_cycles": 51},
        {"recycling_cycles": True},
        {"options": SolutionOptions(refinement="none")},
    ],
)
def test_shared_boundary_validation(kwargs):
    cell, group, data, _, _ = synthetic()
    settings = {"cell": cell, "space_group": group, "elements": ["Fe", "O"], **kwargs}
    with pytest.raises((ValueError, TypeError)):
        solve_structure_elements(data, **settings)


def test_cancelled_and_time_limited_results_are_truthful():
    cell, group, data, _, _ = synthetic()
    cancel = SolutionCancellation()
    cancel.cancel()
    stopped = solve_structure_elements(
        data, cell=cell, space_group=group, elements=["Fe", "O"], cancellation=cancel
    )
    assert stopped.termination == "cancelled" and not stopped.target_met
    assert not stopped.candidates and stopped.shared_search.phasing_starts == 0
    limited = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        options=SolutionOptions(max_seconds=1e-9),
    )
    assert limited.termination == "max_runtime" and not limited.target_met
    assert limited.observations is data


def test_pool_rejects_duplicate_labels_and_invalid_points():
    cell, group, data, xyz, species = synthetic()
    observed = np.sqrt(data.intensity)
    args = (list(cell.as_tuple()), group._native, list(species), xyz.ravel())
    with pytest.raises(ValueError):
        _core._SharedOrbitPool(*args, ["O", "O"], data.hkl.ravel(), observed, 0.01)
    pool = _core._SharedOrbitPool(*args, ["O"], data.hkl.ravel(), observed, 0.01)
    with pytest.raises(ValueError):
        pool.evaluate([np.nan, 0, 0])


def test_exact_zero_amplitude_uses_declared_zero_derivative():
    cell = UnitCell(6, 7, 8, 90, 90, 90)
    group = space_group_by_symbol("P1").space_group
    hkl = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    pool = _core._SharedOrbitPool(
        list(cell.as_tuple()),
        group._native,
        ["Si"],
        np.zeros(3),
        ["Si"],
        hkl.ravel(),
        np.array([12.0, 3.0]),
        0.01,
    )
    result = pool.evaluate([0.5, 0, 0])[0]
    assert np.isfinite(result["objective"])
    np.testing.assert_array_equal(result["jacobian"], np.zeros(6))


def test_atom_cap_keeps_honest_partial_model_without_inventing_required_element():
    cell, group, data, _, _ = synthetic("Pm")
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=2,
        options=SolutionOptions(
            trials=1, cycles=800, max_seconds=10, r1_target=1.0, completion=False
        ),
    )
    assert result.best is not None
    assert len(result.best.species) <= 2
    assert not result.target_met
    assert not result.shared_search.required_elements_present
    assert result.termination != "r1_target"


def test_selected_map_and_origin_stay_with_the_winning_phasing_start():
    cell, group, data, _, _ = synthetic("Pm")
    options = SolutionOptions(trials=2, cycles=800, max_seconds=10, r1_target=0.0, completion=False)
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=32,
        search_rounds=2,
        options=options,
    )
    assert result.shared_search.phasing_starts == 2
    trial = result.best.trial
    assert 0 <= trial < 2
    seed = result.phasing_trials[trial].seed
    alone = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=32,
        search_rounds=2,
        options=SolutionOptions(
            seed=seed, trials=1, cycles=800, max_seconds=10, r1_target=0.0, completion=False
        ),
    )
    np.testing.assert_array_equal(result.density, alone.density)
    np.testing.assert_array_equal(result.best.fractional_xyz, alone.best.fractional_xyz)
    np.testing.assert_array_equal(
        result.best.refinement.origin_shift, alone.best.refinement.origin_shift
    )
