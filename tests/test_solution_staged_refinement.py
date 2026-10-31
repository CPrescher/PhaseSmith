"""Conservative screening, exact per-site transfer and full-model equivalence."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.reference.solution_shared_reference import (
    isotropic_site_seed_reference,
    screening_shortlist_reference,
)
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
from test_solution_shared_search import synthetic
from test_solution_symmetry_adp import CASES, model_case

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_screening_shortlist"), reason="requires experimental-solution"
)


def test_selector_matches_independent_set_union_and_retains_rescues():
    rng = np.random.default_rng(81)
    for n in range(1, 35):
        scores = rng.normal(-30, 12, n).astype(np.float64)
        labels = [(None, "C", "N", "O")[i % 4] for i in range(n)]
        for limit in (1, 3, 8):
            expected = screening_shortlist_reference(scores, labels, ["O", "N"], limit)
            assert (
                _core._solution_screening_shortlist(scores, labels, ["O", "N"], limit) == expected
            )
    scores = np.array([0.0, 1.0, 3.0, np.nextafter(3.0, np.inf), 9.0])
    assert _core._solution_screening_shortlist(scores, [None, None, None, "C", "F"], ["F"], 2) == [
        0,
        1,
        2,
        4,
    ]
    assert _core._solution_screening_shortlist(np.array([1.0, 1.0, 1.0]), [None] * 3, [], 1) == [
        0,
        1,
        2,
    ]
    assert _core._solution_screening_shortlist(np.array([], dtype=float), [], [], 3) == []


def test_selector_treats_signed_zero_as_equal_in_input_order():
    assert _core._solution_screening_shortlist(np.array([0.0, -0.0, -2.0]), [None] * 3, [], 1) == [
        2,
        0,
        1,
    ]


@pytest.mark.parametrize(
    "scores,labels,limit",
    [
        (np.array([np.nan]), [None], 1),
        (np.array([np.inf]), [None], 1),
        (np.array([0.0]), [], 1),
        (np.array([0.0]), [None], 0),
        (np.array([0.0]), [None], 17),
    ],
)
def test_selector_rejects_invalid_inputs(scores, labels, limit):
    with pytest.raises(ValueError):
        _core._solution_screening_shortlist(scores, labels, [], limit)


@pytest.mark.parametrize("symbol,cell,sites,_", CASES)
def test_site_seed_is_exact_and_preserves_isotropic_structure_factors(symbol, cell, sites, _):
    anis, meta, _hkl, _observed, _ = model_case(symbol, cell, sites)
    iso, _, _, _, _ = model_case(symbol, cell, sites, mode="isotropic")
    site_u = np.linspace(0.009, 0.039, len(meta["sites"]))
    seed = anis.initial_site_parameters(site_u)
    np.testing.assert_allclose(
        seed, isotropic_site_seed_reference(meta, site_u), rtol=3e-15, atol=2e-15
    )
    a = anis.evaluate(seed)
    b = iso.evaluate(iso.initial_site_parameters(site_u))
    for key in ("objective", "r1", "amplitude_scale", "residual", "u_cartesian", "fractional_xyz"):
        np.testing.assert_allclose(a[key], b[key], rtol=3e-12, atol=2e-13)
    np.testing.assert_allclose(
        a["u_cartesian"], np.array([u * np.eye(3) for u in site_u]), rtol=4e-13, atol=1e-14
    )
    with pytest.raises(ValueError):
        anis.initial_site_parameters(np.r_[site_u, 0.01])
    bad = site_u.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        anis.initial_site_parameters(bad)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_staging_preserves_inferred_counts_and_full_refinement(symbol):
    cell, group, data, _, _ = synthetic(symbol)
    options = SolutionOptions(
        trials=1, cycles=800, max_seconds=15, r1_target=0.005, completion=False
    )
    results = [
        solve_structure_elements(
            data,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            maximum_atoms=32,
            options=options,
            staged_refinement=staged,
        )
        for staged in (False, True)
    ]
    baseline, staged = results
    assert staged.composition == baseline.composition
    assert staged.target_met and baseline.target_met
    assert staged.best.r1 < 1e-6 and baseline.best.r1 < 1e-6
    assert all(c.refinement.mode == "anisotropic" for c in staged.candidates)
    assert staged.best.objective <= staged.best.refinement.p1_objective + 1e-12
    audit = staged.shared_search
    assert audit.screening_refinements > 0 and audit.promoted_candidates > 0
    assert audit.refinements == audit.screening_refinements + audit.final_refinements
    assert audit.final_refinements == audit.framework_hypotheses + audit.promoted_candidates
    assert audit.staged_refinement
    assert not baseline.shared_search.staged_refinement
    assert baseline.shared_search.screening_refinements == 0
    assert baseline.shared_search.final_refinements == baseline.shared_search.refinements


@pytest.mark.parametrize(
    "kwargs",
    [
        {"staged_refinement": 1},
        {"screening_iterations": 0},
        {"screening_iterations": True},
        {"screening_iterations": 41},
        {"refinement_shortlist": 0},
        {"refinement_shortlist": 17},
    ],
)
def test_schedule_boundary_validation(kwargs):
    cell, group, data, _, _ = synthetic()
    with pytest.raises((ValueError, TypeError)):
        solve_structure_elements(data, cell=cell, space_group=group, elements=["Fe", "O"], **kwargs)


def test_isotropic_requested_mode_does_not_run_a_second_isotropic_stage():
    cell, group, data, _, _ = synthetic("P-1")
    options = SolutionOptions(
        trials=1, cycles=800, refinement="isotropic", r1_target=0.005, completion=False
    )
    result = solve_structure_elements(
        data, cell=cell, space_group=group, elements=["Fe", "O"], options=options, maximum_atoms=32
    )
    assert result.target_met and result.best.refinement.mode == "isotropic"
    assert not result.shared_search.staged_refinement
    assert result.shared_search.screening_refinements == 0


def test_staging_retains_a_model_with_strongly_anisotropic_displacements():
    from phasesmith.io.hkl import HklData
    from phasesmith.scattering_reference import xray_non_resonant

    cell, group, data, xyz, species = synthetic("P-1")
    q = data.hkl @ np.linalg.inv(cell.geometry().direct_basis)
    s = np.linalg.norm(q, axis=1) / 2
    f, _ = xray_non_resonant(list(species), s)
    tensors = np.array(
        [
            np.diag([0.012, 0.024, 0.045]) if e == "Fe" else np.diag([0.009, 0.018, 0.036])
            for e in species
        ]
    )
    dw = np.exp(-2 * np.pi**2 * np.einsum("hi,nij,hj->hn", q, tensors, q))
    sf = (f * dw * np.exp(2j * np.pi * (data.hkl @ xyz.T))).sum(axis=1)
    observations = HklData(data.hkl, np.abs(sf) ** 2, np.ones(len(q)))
    options = SolutionOptions(
        trials=1, cycles=800, max_seconds=15, r1_target=0.005, completion=False
    )
    results = [
        solve_structure_elements(
            observations,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            maximum_atoms=32,
            options=options,
            staged_refinement=flag,
        )
        for flag in (False, True)
    ]
    assert results[0].target_met and results[1].target_met
    assert results[0].composition == results[1].composition == (("Fe", 2), ("O", 2))
    assert results[0].best.r1 < 1e-6 and results[1].best.r1 < 1e-6
