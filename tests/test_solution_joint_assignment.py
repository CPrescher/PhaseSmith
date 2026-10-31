"""Tensor-informed categorical repair against independent Fourier and beam equations."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant
from phasesmith.single_crystal.reference.solution_assignment_reference import (
    joint_assignment_beam_reference,
)

pytestmark = pytest.mark.skipif(
    not hasattr(getattr(_core, "_PeakCompletionPlan", object), "joint"),
    reason="experimental-solution",
)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_beam_matches_independent_equations_with_unequal_site_displacements(seed, symbol):
    from test_solution_ambiguity import setup

    plan, phase, factors, obs, labels, u, s2, m = setup(seed, symbol)
    bounds = np.column_stack([1e-5 - u, 0.2 - u])
    native = plan.displaced(u).joint(labels, bounds, 4)
    reference = joint_assignment_beam_reference(
        phase * np.exp(-8 * np.pi**2 * s2[:, None] * u),
        factors,
        obs,
        labels,
        bounds,
        s2,
        [6, 7, 8],
        m,
        32,
        4,
    )
    assert (
        native["evaluations"] == reference["evaluations"]
        and native["depths"] == reference["depths"]
    )
    assert [tuple(None if x == -1 else x for x in c["labels"]) for c in native["candidates"]] == [
        c["labels"] for c in reference["candidates"]
    ]
    for value, expected in zip(native["candidates"], reference["candidates"], strict=True):
        np.testing.assert_allclose(value["shifts"], expected["shifts"], rtol=3e-12, atol=2e-14)
        assert value["objective"] == pytest.approx(expected["objective"], rel=3e-12, abs=3e-15)
        assert value["score"] == pytest.approx(expected["score"], rel=3e-12, abs=3e-10)
    again = plan.displaced(u).joint(labels, bounds, 4)
    assert again == native


def direct_columns(meta, cell, hkl, points, tensors):

    matrix = cell.geometry().direct_basis
    reciprocal = hkl @ np.linalg.inv(matrix)
    columns = []
    for site, point, tensor in zip(meta["sites"], points, tensors, strict=True):
        value = np.zeros(len(hkl), complex)
        for image in site["images"]:
            # Metadata contains the exact fractional image operation and Cartesian rotation.
            position = np.asarray(image["rotation"]) @ point + np.asarray(image["translation"])
            rot = np.array(image["cartesian_rotation"])
            q = reciprocal @ rot
            dw = np.exp(-2 * np.pi**2 * np.einsum("hi,ij,hj->h", q, tensor, q))
            value += dw * np.exp(2j * np.pi * (hkl @ position))
        columns.append(value)
    return np.column_stack(columns)


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_tensor_seeds_cache_and_jets_match_direct_symmetry_fourier(symbol, mode):
    from test_solution_symmetry_adp import model_case

    cell = UnitCell(6, 7, 8, 90, 100, 90)
    sites = [
        ("C", (0.13, 0.24, 0.35)),
        ("N", (0.36, 0.53, 0.68)),
        ("O", (0.72, 0.18, 0.46)),
        ("C", (0.55, 0.36, 0.22)),
    ]
    model, meta, hkl, obs, _ = model_case(symbol, cell, sites, mode=mode)
    p = model.initial_parameters(0.025) + np.random.default_rng(83).normal(
        0, 0.02, meta["parameter_count"]
    )
    state = model.evaluate(p)
    tensors = np.array(state["u_cartesian"])
    points = np.array(state["fractional_xyz"])
    # A fresh model at the evaluated geometry restores tensors without resetting ellipsoids.
    import gemmi

    full = []
    species = []
    group = gemmi.find_spacegroup_by_name(symbol)
    for site, point in zip(meta["sites"], points, strict=True):
        for op in group.operations():
            full.append(op.apply_to_xyz(point))
            species.append(site["species"])
    fresh = _core._SymmetricSolutionModel(
        list(cell.as_tuple()),
        space_group_by_symbol(symbol).space_group._native,
        species,
        np.array(full).ravel(),
        hkl.ravel(),
        obs,
        mode,
        0.01,
    )
    seeded = fresh.evaluate(fresh.initial_tensors(tensors.ravel()))
    np.testing.assert_allclose(seeded["u_cartesian"], tensors, rtol=3e-12, atol=2e-14)
    elements = ["C", "N", "O"]
    labels = [elements.index(s["species"]) for s in meta["sites"]]
    plan = model.tensor_completion(p, elements, hkl.ravel(), obs, 64)
    # Use the independently described fractional operations in metadata.
    phase = direct_columns(meta, cell, hkl, points, tensors)
    s2 = np.sum((hkl @ np.linalg.inv(cell.geometry().direct_basis)) ** 2, axis=1) / 4
    factors, _ = xray_non_resonant(elements, np.sqrt(s2))

    def objective(coords):
        columns = direct_columns(meta, cell, hkl, coords, tensors)
        a = np.abs(np.sum(columns * factors[:, labels], axis=1))
        scale = (a @ obs) / (a @ a)
        return np.sum((scale * a - obs) ** 2) / np.sum(obs**2)

    actual = plan.evaluate(labels, 0)
    assert actual["objective"] == pytest.approx(objective(points), rel=4e-11, abs=3e-13)
    gradient = np.array(actual["gradient"])
    for i, axis in [(1, 0), (2, 2)]:
        delta = np.zeros_like(points)
        delta[i, axis] = 2e-6
        fd = (objective(points + delta) - objective(points - delta)) / 4e-6
        assert gradient[i, axis] == pytest.approx(fd, rel=4e-5, abs=3e-8)
    eig = np.linalg.eigvalsh(tensors)
    bounds = np.column_stack([1e-5 - eig[:, 0], 0.2 - eig[:, -1]])
    expected = joint_assignment_beam_reference(
        phase,
        factors,
        obs,
        labels,
        bounds,
        s2,
        [6, 7, 8],
        [len(s["images"]) for s in meta["sites"]],
        64,
        2,
    )
    result = plan.joint(labels, bounds, 2)
    assert [tuple(None if x == -1 else x for x in c["labels"]) for c in result["candidates"]] == [
        c["labels"] for c in expected["candidates"]
    ]


@pytest.mark.parametrize(
    "invalid", ["length", "nan", "asymmetric", "negative", "large", "symmetry"]
)
def test_invalid_tensor_seeds_are_rejected(invalid):
    from test_solution_symmetry_adp import CASES, model_case

    symbol, cell, sites, _ = CASES[2]
    model, meta, *_ = model_case(symbol, cell, sites)
    tensors = np.tile(np.eye(3) * 0.025, (len(meta["sites"]), 1, 1))
    if invalid == "length":
        tensors = tensors[:-1]
    elif invalid == "nan":
        tensors[0, 0, 0] = np.nan
    elif invalid == "negative":
        tensors[0, 0, 0] = -0.1
    elif invalid == "large":
        tensors[0, 0, 0] = 0.3
    elif invalid == "asymmetric":
        tensors[0, 0, 1] = 0.001
    else:
        tensors[0, 0, 1] = tensors[0, 1, 0] = 0.001
    with pytest.raises(ValueError):
        model.initial_tensors(tensors.ravel())


@pytest.mark.parametrize("invalid", ["bare", "coverage", "bounds", "nan", "limit", "length"])
def test_invalid_beam_controls(invalid):
    from test_solution_ambiguity import setup

    plan, _, _, _, labels, u, _, _ = setup()
    bounds = np.column_stack([1e-5 - u, 0.2 - u])
    maximum = 2
    if invalid != "bare":
        plan = plan.displaced(u)
    if invalid == "coverage":
        labels = [0] * len(labels)
    elif invalid == "bounds":
        bounds[0, 0] = 0.01
    elif invalid == "nan":
        bounds[0, 1] = np.nan
    elif invalid == "limit":
        maximum = 5
    elif invalid == "length":
        bounds = bounds[:-1]
    with pytest.raises(ValueError):
        plan.joint(labels, bounds, maximum)


@pytest.mark.parametrize("kind", ["birth", "delete", "exchange"])
def test_joint_search_recovers_whole_orbit_counts_without_count_inputs(kind):
    from test_solution_ambiguity import setup

    _, phase, factors, _, _, u, s2, _ = setup()
    cell = UnitCell(6, 7, 8, 90, 100, 90)
    group = space_group_by_symbol("P1").space_group
    points = np.random.default_rng(670).uniform(0.1, 0.9, (6, 3))
    hkl = np.array(
        [h for h in itertools.product(range(-3, 4), repeat=3) if h != (0, 0, 0)], dtype=np.int64
    )
    truth = [0, 1, 2, 0, 1, 2]
    initial = [0, 1, 2, 0, 1, -1]
    if kind == "delete":
        truth[-1] = -1
        initial[-1] = 0
    if kind == "exchange":
        truth[-1] = -1
        initial = [0, 1, 2, 0, -1, 0]
    observed = np.abs(
        sum(
            (
                phase[:, i] * factors[:, label] * np.exp(-8 * np.pi**2 * s2 * u[i])
                for i, label in enumerate(truth)
                if label >= 0
            ),
            np.zeros(len(hkl), complex),
        )
    )
    plan = _core._PeakCompletionPlan(
        list(cell.as_tuple()),
        group._native,
        points.ravel(),
        ["C", "N", "O"],
        hkl.ravel(),
        observed,
        6,
    ).displaced(u)
    result = plan.joint(initial, np.column_stack([1e-5 - u, 0.2 - u]), 4)
    assert truth in [c["labels"] for c in result["candidates"]]
    assert min(c["objective"] for c in result["candidates"]) < 1e-20
    for c in result["candidates"]:
        assert set(x for x in c["labels"] if x >= 0) == {0, 1, 2}
        assert sum(x >= 0 for x in c["labels"]) <= 6


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_preserves_clean_solution_and_rejects_invalid_feature_controls(symbol):
    import sys
    from pathlib import Path

    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from solution_multicore import ENABLED

    cell, group, data, _, species = synthetic(symbol)
    common = {k: True for k in ENABLED if k != "phase_first_selection"}
    common.update(
        symmetry_recycling=True,
        tangent_refinement=True,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=12,
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
    new = solve_structure_elements(data, **common, joint_assignment_refinement=True)
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    np.testing.assert_array_equal(old.best.fractional_xyz, new.best.fractional_xyz)
    assert (
        new.shared_search.joint_assignment_refinement
        and new.shared_search.joint_assignment_fits == 0
    )
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, joint_assignment_refinement=1)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(
            data, **dict(common, phase_origin_selection=False), joint_assignment_refinement=True
        )


@pytest.mark.parametrize("seed", range(4))
def test_diagonal_shift_derivative_matches_independent_finite_differences(seed):
    from test_solution_ambiguity import setup

    plan, phase, factors, obs, labels, u, s2, _ = setup(seed, "P-1")
    plan = plan.displaced(u)
    shifts = np.random.default_rng(seed + 719).uniform(-0.001, 0.001, len(u))

    def objective(x):
        a = np.abs(
            np.sum(
                phase * factors[:, labels] * np.exp(-8 * np.pi**2 * s2[:, None] * (u + x)), axis=1
            )
        )
        k = (a @ obs) / (a @ a)
        return np.sum((k * a - obs) ** 2) / np.sum(obs**2)

    for site in [0, 3, 5]:
        value, gradient, curvature = plan.shift_score(labels, shifts, site)
        assert value == pytest.approx(objective(shifts), rel=3e-12, abs=2e-14)
        delta = np.zeros(len(u))
        delta[site] = 2e-7
        fd = (objective(shifts + delta) - objective(shifts - delta)) / 4e-7
        assert gradient == pytest.approx(fd, rel=3e-5, abs=3e-8)
        assert np.isfinite(curvature) and curvature >= 0
    with pytest.raises(ValueError):
        plan.shift_score(labels, shifts[:-1], 0)
    shifts[0] = np.nan
    with pytest.raises(ValueError):
        plan.shift_score(labels, shifts, 0)


@pytest.mark.parametrize("seed", range(6))
def test_coherent_displacement_proposals_match_independent_sweeps(seed):
    from phasesmith.single_crystal.reference.solution_assignment_reference import (
        coherent_assignment_reference,
    )
    from test_solution_ambiguity import setup

    plan, phase, factors, obs, labels, u, s2, m = setup(seed, "Pm")
    shifts = np.random.default_rng(seed + 317).uniform(-0.004, 0.004, len(u))
    bounds = np.column_stack([1e-5 - u, 0.2 - u])
    expected = coherent_assignment_reference(
        phase * np.exp(-8 * np.pi**2 * s2[:, None] * u),
        factors,
        obs,
        labels,
        shifts,
        s2,
        [6, 7, 8],
        m,
    )
    value = plan.displaced(u).coherent(labels, shifts, bounds)
    assert value["evaluations"] == expected["evaluations"] and value["depths"] == expected["depths"]
    assert bool(value["candidates"]) == expected["changed"]
    if expected["changed"]:
        assert value["candidates"][0]["labels"] == expected["labels"]
        assert value["candidates"][0]["objective"] == pytest.approx(
            expected["objective"], rel=3e-12, abs=2e-14
        )
    with pytest.raises(ValueError):
        plan.displaced(u).coherent(labels, shifts, bounds[:-1])
