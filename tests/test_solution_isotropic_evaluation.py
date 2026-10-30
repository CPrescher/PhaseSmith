"""Specialized isotropic execution retains independent equations and exact reductions."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_reference import symmetric_model_reference
from phasesmith.scattering_reference import xray_non_resonant
from test_solution_refinement_families import fixture
from test_solution_symmetry_adp import CASES

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="experimental-solution"
)


@pytest.mark.parametrize("symbol,cell,sites,_", CASES)
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_exact_evaluation_and_independent_derivatives(
    symbol, cell, sites, _, mode, weighted, cache
):
    args, hkl, observed = fixture(symbol, cell, sites, mode, cache, "partial_duplicates")
    scalar = _core._SymmetricSolutionModel(*args, weight_families=weighted)
    fast = _core._SymmetricSolutionModel(
        *args, weight_families=weighted, specialized_isotropic=True
    )
    m = fast.metadata()
    assert m["specialized_isotropic_evaluation"] == (cache and mode == "isotropic")
    p = scalar.initial_parameters(0.015)
    p += np.random.default_rng(671).normal(0, 0.01, len(p))
    a, b = scalar.evaluate(p), fast.evaluate(p)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    reps = (
        np.asarray(m["reflection_representatives"], dtype=np.int64)
        if weighted
        else np.arange(len(hkl))
    )
    mult = np.asarray(m["reflection_multiplicities"]) if weighted else None
    q = hkl[reps] @ np.linalg.inv(cell.geometry().direct_basis)
    factors, _ = xray_non_resonant(
        [s["species"] for s in m["sites"]], np.linalg.norm(q, axis=1) / 2
    )

    def reference(parameters):
        return symmetric_model_reference(
            m,
            cell.geometry().direct_basis,
            hkl[reps],
            observed[reps],
            factors,
            parameters,
            multiplicities=mult if weighted else None,
        )

    ref = reference(p)
    for key in ["objective", "r1", "amplitude_scale", "residual", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(b[key], ref[key], rtol=5e-11, atol=2e-12)
    jac = b["jacobian"].reshape(len(p), len(reps))
    for axis in range(len(p)):
        delta = np.zeros(len(p))
        delta[axis] = 2e-6
        fd = (reference(p + delta)["residual"] - reference(p - delta)["residual"]) / 4e-6
        np.testing.assert_allclose(jac[axis], fd, rtol=3e-5, atol=2e-8)


@pytest.mark.parametrize("rows", [1, 3, 7, 8, 9, 17])
def test_short_rows_zero_reflection_and_u_boundaries(rows):
    symbol, cell, sites, _ = CASES[0]
    args, h, o = fixture(symbol, cell, sites, "isotropic")
    args = list(args)
    h = h[:rows].copy()
    h[0] = 0
    args[4], args[5] = h.ravel(), o[:rows]
    scalar = _core._SymmetricSolutionModel(*args)
    fast = _core._SymmetricSolutionModel(*args, specialized_isotropic=True)
    for u in [1e-5, 0.01, 0.2]:
        p = fast.initial_parameters(u)
        a, b = scalar.evaluate(p), fast.evaluate(p)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError):
        fast.evaluate(p[:-1])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_search_preserves_counts_and_exact_final_model(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
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
        ambiguous_assignments=True,
        cached_assignment_refinement=True,
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
    new = solve_structure_elements(data, **common, specialized_isotropic_refinement=True)
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.shared_search.specialized_isotropic_fits > 0
    assert old.shared_search.specialized_isotropic_fits == 0
    np.testing.assert_array_equal(old.best.fractional_xyz, new.best.fractional_xyz)
    assert old.best.r1 == new.best.r1
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, specialized_isotropic_refinement=1)
