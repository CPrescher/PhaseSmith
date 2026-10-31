"""Prepared reciprocal geometry against direct sums and independent derivatives."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.single_crystal.reference.solution_reference import symmetric_model_reference
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
from test_solution_shared_search import synthetic
from test_solution_symmetry_adp import CASES, model_case

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="requires experimental-solution"
)


@pytest.mark.parametrize("symbol,cell,sites,_counts", CASES)
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_cached_values_and_fused_derivatives_against_direct_and_numpy(
    symbol, cell, sites, _counts, mode
):
    cached, metadata, hkl, observed, factors = model_case(
        symbol, cell, sites, mode=mode, shift=(0.173, 0.287, 0.391)
    )
    direct, other, *_ = model_case(
        symbol, cell, sites, mode=mode, shift=(0.173, 0.287, 0.391), cache_geometry=False
    )
    assert metadata["geometry_cache_bytes"] == len(hkl) * metadata["full_count"] * 80
    assert other["geometry_cache_bytes"] == 0
    rng = np.random.default_rng(739)
    for index in range(4):
        p = cached.initial_parameters(0.015)
        if index:
            p += rng.normal(0, 0.025, len(p))
        actual, original = cached.evaluate(p), direct.evaluate(p)
        reference = symmetric_model_reference(
            metadata, cell.geometry().direct_basis, hkl, observed, factors, p
        )
        for key in (
            "residual",
            "jacobian",
            "amplitude_scale",
            "r1",
            "objective",
            "fractional_xyz",
            "u_cartesian",
        ):
            np.testing.assert_allclose(actual[key], original[key], rtol=4e-10, atol=2e-12)
            if key in reference:
                np.testing.assert_allclose(actual[key], reference[key], rtol=4e-10, atol=2e-12)
        np.testing.assert_array_equal(actual["jacobian"], cached.evaluate(p)["jacobian"])
        jac = np.asarray(actual["jacobian"]).reshape(len(p), len(hkl))
        for j in range(len(p)):
            plus, minus = p.copy(), p.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            finite = (cached.evaluate(plus)["residual"] - cached.evaluate(minus)["residual"]) / (
                2e-6
            )
            np.testing.assert_allclose(jac[j], finite, rtol=5e-6, atol=5e-9)
    start = cached.initial_parameters(0.015)
    fit = cached.refine(start, 8, 10, None)["evaluation"]
    baseline = direct.refine(start, 8, 10, None)["evaluation"]
    assert abs(fit["r1"] - baseline["r1"]) < 2e-10
    assert abs(fit["objective"] - baseline["objective"]) < 2e-11


def test_memory_cap_falls_back_without_truncating_observations():
    group = space_group_by_symbol("Fm-3m").space_group
    points = group.expand_sites([[0.25, 0.25, 0.25]]).fractional_xyz
    assert len(points) == 8
    hkl = np.tile(np.array([[2, 0, 0]], dtype=np.int64), (60_000, 1))
    observed = np.ones(len(hkl))
    args = (
        [4.2, 4.2, 4.2, 90, 90, 90],
        group._native,
        ["Si"] * 8,
        points.ravel(),
        hkl.ravel(),
        observed,
        "isotropic",
        0.01,
    )
    cached = _core._SymmetricSolutionModel(*args, True)
    direct = _core._SymmetricSolutionModel(*args, False)
    assert cached.metadata()["geometry_cache_bytes"] == 0
    assert cached.metadata()["full_count"] == 8
    p = cached.initial_parameters(0.015)
    a, b = cached.evaluate(p), direct.evaluate(p)
    assert len(a["residual"]) == 60_000
    np.testing.assert_array_equal(a["jacobian"], b["jacobian"])
    np.testing.assert_array_equal(a["residual"], b["residual"])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_same_inferred_counts_and_recorded_cache_control(symbol):
    cell, group, data, _, _ = synthetic(symbol)
    results = [
        solve_structure_elements(
            data,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            maximum_atoms=32,
            options=SolutionOptions(
                phasing="dual_space", trials=1, cycles=800, max_seconds=15, r1_target=0.005
            ),
            cached_refinement=enabled,
        )
        for enabled in (False, True)
    ]
    baseline, cached = results
    assert baseline.target_met and cached.target_met
    assert baseline.composition == cached.composition
    assert cached.shared_search.cached_refinement
    assert not baseline.shared_search.cached_refinement
    assert abs(cached.best.r1 - baseline.best.r1) < 1e-9
    assert cached.best.r1 < 1e-6


def test_cache_control_requires_a_boolean():
    cell, group, data, _, _ = synthetic()
    with pytest.raises(TypeError, match="cached_refinement"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], cached_refinement=1
        )


def test_geometry_factorization_matches_moved_image_and_tensor():
    from phasesmith.single_crystal.reference.solution_reference import image_geometry_reference

    _, metadata, hkl, _, _ = model_case(*CASES[3][:3])
    cell = CASES[3][1]
    q = hkl @ np.linalg.inv(cell.geometry().direct_basis)
    rng = np.random.default_rng(822)
    for site in metadata["sites"]:
        tangents = np.asarray(site["coordinate_basis"]).reshape(-1, 3)
        parameters = rng.normal(0, 0.05, len(tangents))
        position = np.array(site["fractional_xyz"])
        moved = position + parameters @ tangents
        raw = rng.normal(size=(3, 3))
        u = 0.01 * np.eye(3) + raw @ raw.T * 0.002
        for image in site["images"]:
            r, t, cart = (
                np.array(image["rotation"]),
                np.array(image["translation"]),
                np.array(image["cartesian_rotation"]),
            )
            phase, coeff, weight = image_geometry_reference(hkl, q, r, t, position, tangents, cart)
            expected = 2 * np.pi * (hkl @ (r @ moved + t))
            np.testing.assert_allclose(phase + coeff @ parameters, expected, rtol=1e-13, atol=2e-13)
            direct = -2 * np.pi**2 * np.einsum("ri,ij,rj->r", q, cart @ u @ cart.T, q)
            np.testing.assert_allclose(
                np.einsum("rij,ij->r", weight, u), direct, rtol=2e-14, atol=2e-14
            )
