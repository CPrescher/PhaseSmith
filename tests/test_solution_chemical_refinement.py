"""Chemistry proposals, coupled moves and iterative steps against independent equations."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_origin_reference import (
    iterative_normal_reference,
    peak_completion_reference,
)
from phasesmith.io.space_groups import space_group_by_symbol
from test_solution_peak_completion import make_plan, setup

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_chemical_density_scales"), reason="experimental-solution"
)


def chemical_scales(points, masses, elements):
    return _core._chemical_density_scales(
        [8.0, 8.0, 8.0, 90.0, 90.0, 90.0],
        space_group_by_symbol("P1").space_group._native,
        np.asarray(points, dtype=float).ravel(),
        np.asarray(masses, dtype=float),
        elements,
    )


def test_oxygen_coordination_sets_a_scale_despite_an_unrepresentative_strongest_peak():
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    points = np.vstack([[0.5, 0.5, 0.5], 0.5 + directions * 1.5 / 8, [0.1, 0.1, 0.1]])
    result = chemical_scales(points, [60.0, 32.0, 32.0, 32.0, 32.0, 500.0], ["P", "O", "W"])
    assert any(
        p["kind"] == "oxygen_coordination"
        and p["support"] == 4
        and p["scale"] == pytest.approx(4.0)
        for p in result
    )
    shifted = chemical_scales(
        (points + np.array([0.46, 0.43, 0.39])) % 1,
        [60.0, 32.0, 32.0, 32.0, 32.0, 500.0],
        ["P", "O", "W"],
    )
    assert result == shifted
    assert chemical_scales(points, [60.0, 32.0, 32.0, 32.0, 32.0, 500.0], ["P", "F", "W"]) == []


def test_boron_cage_scale_requires_a_connected_shell():
    cube = np.asarray(list(itertools.product([0.0, 1.0], repeat=3))) * 1.7 / 8 + 0.3
    result = chemical_scales(cube, np.full(8, 10.0), ["B"])
    assert result[0]["kind"] == "boron_cage"
    assert result[0]["scale"] == pytest.approx(2.0)
    assert result[0]["support"] == 3
    assert chemical_scales(cube[:2], [10.0, 10.0], ["B"]) == []


def test_paired_swap_escapes_required_element_coverage_trap():
    from phasesmith.scattering_reference import xray_non_resonant

    cell, group, p, h, _, _, _, _, s2 = setup("P1", False)
    factors, _ = xray_non_resonant(["Fe", "O", "C"], np.sqrt(s2))
    coefficients = np.exp(2j * np.pi * (h @ p.T)) * factors
    obs = np.abs(coefficients.sum(axis=1)) * np.exp(-8 * np.pi**2 * 0.01 * s2)
    plan = _core._PeakCompletionPlan(
        list(cell.as_tuple()), group._native, p.ravel(), ["Fe", "O", "C"], h.ravel(), obs, 32
    ).displaced([0.01] * 3)
    initial = plan.evaluate([1, 0, 2], 0)["objective"]
    result = plan.reassign([1, 0, 2])
    assert initial > 0.01 and result["changes"] > 0
    assert result["candidates"][0]["labels"] == [0, 1, 2]
    assert result["candidates"][0]["objective"] < 1e-24
    assert result["candidates"][0]["atoms"] == 3


def test_fixed_site_displacements_preserve_values_and_fused_coordinate_jets():
    cell, group, p, h, obs, r, t, f, s2 = setup("P1", False)
    site_u = np.array([0.004, 0.04, 0.012])
    plan = make_plan(cell, group, p, h, obs).displaced(site_u)
    labels = [0, 1, 0]
    value = plan.evaluate(labels, 0)
    ref = peak_completion_reference(p, labels, h, r, t, f, s2, obs, site_u)
    assert value["objective"] == pytest.approx(ref["objective"], rel=2e-12, abs=2e-14)
    np.testing.assert_allclose(value["gradient"], ref["gradient"], rtol=2e-11, atol=2e-12)
    delta = np.zeros_like(p)
    delta[1, 0] = 1e-7
    left = (
        make_plan(cell, group, p - delta, h, obs).displaced(site_u).evaluate(labels, 0)["objective"]
    )
    right = (
        make_plan(cell, group, p + delta, h, obs).displaced(site_u).evaluate(labels, 0)["objective"]
    )
    assert value["gradient"][1][0] == pytest.approx((right - left) / 2e-7, rel=2e-5, abs=2e-8)
    with pytest.raises(ValueError):
        plan.displaced([-0.01, 0.02, 0.02])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("damping", [1e-3, 0.1, 10.0])
def test_iterative_step_matches_dense_independent_normal_equations(seed, damping):
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(64, 240))
    j[-1] = j[0] + 1e-4 * j[-1]
    j *= np.geomspace(0.01, 100.0, 64)[:, None]
    r = rng.normal(size=240)
    expected = iterative_normal_reference(j, r, damping)
    actual = _core._iterative_normal_step(j.ravel(), r, damping)
    np.testing.assert_allclose(actual["step"], expected, rtol=3e-5, atol=1e-7)
    assert 0 < actual["iterations"] <= 128
    normal = j @ j.T + np.diag(damping * np.maximum(np.sum(j * j, axis=1), 1e-8))
    residual = normal @ actual["step"] + j @ r
    # Residual is measured in the same diagonal-scaled coordinates used for CG.
    scale = np.maximum(np.sum(j * j, axis=1), 1e-8) ** -0.5
    assert np.linalg.norm(scale * residual) <= 2e-8 * np.linalg.norm(scale * (j @ r))


@pytest.mark.parametrize("damping", [0.0, -1.0, np.nan])
def test_iterative_step_rejects_invalid_damping(damping):
    with pytest.raises(ValueError):
        _core._iterative_normal_step(np.ones(12), np.ones(3), damping)


@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_iterative_refinement_preserves_dense_trajectory_on_a_many_parameter_model(mode):
    from phasesmith.crystallography import UnitCell
    from test_solution_symmetry_adp import model_case

    rng = np.random.default_rng(410)
    sites = [
        ("Fe" if i < 2 else "O", point) for i, point in enumerate(rng.uniform(0.1, 0.9, (12, 3)))
    ]
    native, metadata, *_ = model_case("P1", UnitCell(8, 9, 10, 90, 90, 90), sites, mode=mode)
    assert metadata["parameter_count"] >= 32
    parameters = native.initial_parameters(0.015)
    dense = native.refine(parameters, 5, 20, None)
    iterative = native.refine(parameters, 5, 20, None, True)
    assert iterative["accepted_steps"] == dense["accepted_steps"]
    assert iterative["evaluation"]["objective"] == pytest.approx(
        dense["evaluation"]["objective"], rel=2e-5, abs=1e-12
    )


def test_homometric_two_atom_swap_is_not_reported_as_an_improvement():
    cell, group, p, h, observed, *_ = setup("P1", False)
    plan = make_plan(cell, group, p[:2], h, observed).displaced([0.01, 0.01])
    a = plan.evaluate([0, 1], 0)["objective"]
    b = plan.evaluate([1, 0], 0)["objective"]
    assert a == pytest.approx(b, abs=1e-24)
    assert plan.reassign([1, 0])["candidates"] == []


def test_oxygen_shell_proposal_in_a_skew_cell_across_periodic_boundaries():
    from phasesmith.crystallography import UnitCell

    cell = UnitCell(8, 7, 9, 80, 100, 110)
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    ligand_fractional = (1.5 * directions) @ np.linalg.inv(cell.geometry().direct_basis).T
    points = (np.vstack([[0.0, 0.0, 0.0], ligand_fractional]) + np.array([0.98, 0.97, 0.96])) % 1
    result = _core._chemical_density_scales(
        list(cell.as_tuple()),
        space_group_by_symbol("P1").space_group._native,
        points.ravel(),
        np.array([60.0, 32.0, 32.0, 32.0, 32.0]),
        ["P", "O"],
    )
    assert any(p["support"] == 4 and p["scale"] == pytest.approx(4.0) for p in result)
