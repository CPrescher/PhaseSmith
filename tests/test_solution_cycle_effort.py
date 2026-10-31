"""Independent origin/hand matching and longer phase plateau references."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_consensus_reference import model_agreement_distance
from phasesmith.experimental.solution_schedule_reference import late_phase_plateau_ready

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_model_agreement"), reason="experimental-solution"
)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("angles", [(90, 90, 90), (78, 96, 107), (90, 110, 90)])
def test_consensus_matches_exhaustive_reference_across_origin_hand_and_order(seed, angles):
    rng = np.random.default_rng(seed)
    cell = UnitCell(6, 7, 8, *angles)
    species = ["Si", "O", "O", "Fe", "O"]
    a = rng.uniform(0.05, 0.95, (5, 3))
    order = rng.permutation(5)
    for hand in [1, -1]:
        b = (hand * a + rng.uniform(-2, 2, 3)) % 1
        bsp = [species[i] for i in order]
        b = b[order]
        for change in [0, 0.0005, 0.08]:
            moved = b.copy()
            moved[1, 0] += change
            ref = model_agreement_distance(cell.geometry().direct_basis, species, a, bsp, moved)
            found = _core._solution_model_agreement(
                list(cell.as_tuple()), species, a.ravel(), bsp, moved.ravel()
            )
            assert (found is None) == (ref is None)
            if found is not None:
                assert 0 <= found <= 0.02


def test_consensus_rejects_wrong_labels_counts_and_malformed_arrays():
    cell = [6, 7, 8, 90, 100, 90]
    p = np.array([[0.1, 0.2, 0.3], [0.4, 0.6, 0.8]])
    assert (
        _core._solution_model_agreement(cell, ["Si", "O"], p.ravel(), ["Si", "Si"], p.ravel())
        is None
    )
    for species, xyz in [
        ([], np.array([])),
        (["Xx"], p[0]),
        (["Si"], p.ravel()),
        (["Si", "O"], np.full(6, np.nan)),
        (["O"] * 129, np.zeros(387)),
    ]:
        with pytest.raises(ValueError):
            _core._solution_model_agreement(cell, species, xyz, ["Si", "O"], p.ravel())


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("omit", [True, False])
def test_late_plateau_matches_reference_and_waits_through_genuine_progress(seed, omit):
    rng = np.random.default_rng(seed)
    for n in [0, 128, 255, 256, 257, 288, 384, 512, 800]:
        for level in [0.79, 0.8, 0.85]:
            q = level + rng.uniform(-0.001, 0.001, n)
            assert _core._phase_plateau_ready(q, omit, late=True) == late_phase_plateau_ready(
                q, omit_every_third=omit
            )
    q = np.full(256, 0.81)
    assert _core._phase_plateau_ready(q, omit, late=True)
    q[-1] = 0.83
    assert not _core._phase_plateau_ready(q, omit, late=True)
    q[-1] = np.nan
    assert not _core._phase_plateau_ready(q, omit, late=True)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_models_counts_and_identical_cycle_prefixes(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
        short_phase_starts=True,
        phase_plateau_stopping=True,
        assignment_convergence_stopping=True,
        blocked_normal_matrices=True,
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
        specialized_isotropic_refinement=True,
        streamed_peak_search=True,
        reuse_normal_matrices=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=0.005,
            completion=False,
        ),
    )
    before = data.intensity.copy()
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(
        data, **common, late_phase_plateau_stopping=True, model_consensus_stopping=True
    )
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.refinement.mode == "anisotropic"
    np.testing.assert_allclose(old.best.fractional_xyz, new.best.fractional_xyz, rtol=0, atol=2e-7)
    assert (
        new.shared_search.late_phase_plateau_stopping and new.shared_search.model_consensus_stopping
    )
    assert (
        not old.shared_search.late_phase_plateau_stopping
        and not old.shared_search.model_consensus_stopping
    )
    for phase in new.shared_search.phase_slices:
        assert phase.cycle_quota == _core._phase_cycle_quota(800, 4, phase.trial, short=True)
        assert phase.completed_cycles <= phase.cycle_quota
    for a, b in zip(old.histories, new.histories, strict=False):
        n = min(len(a), len(b))
        np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))
    np.testing.assert_array_equal(before, data.intensity)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, late_phase_plateau_stopping=1)
    bad = dict(common, work_sliced_phasing=False)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **bad, late_phase_plateau_stopping=True)


def test_noisy_independent_models_can_stop_without_claiming_requested_r1_target():
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from phasesmith.io.hkl import HklData
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic("P1")
    noise = 1 + 0.02 * np.cos(0.37 * np.sum(data.hkl * data.hkl, axis=1))
    noisy = HklData(data.hkl, data.intensity * noise, data.sigma)
    controls = dict(cell=cell, space_group=group, elements=["Fe", "O"], maximum_atoms=12)
    for name in [
        "symmetry_recycling",
        "tangent_refinement",
        "wilson_normalization",
        "phase_quality_selection",
        "phase_origin_selection",
        "chemical_assignment",
        "iterative_refinement",
        "score_only_evaluation",
        "reusable_fourier_workspaces",
        "work_sliced_phasing",
        "isotropic_assignment_refinement",
        "filtered_peak_search",
        "batched_fft",
        "real_fft",
        "fused_phase_workspace",
        "cached_normal_refinement",
        "multiplicity_weighted_refinement",
        "ambiguous_assignments",
        "cached_assignment_refinement",
        "specialized_isotropic_refinement",
        "streamed_peak_search",
        "reuse_normal_matrices",
        "short_phase_starts",
        "blocked_normal_matrices",
        "phase_plateau_stopping",
        "assignment_convergence_stopping",
        "model_consensus_stopping",
    ]:
        controls[name] = True
    result = solve_structure_elements(
        noisy,
        **controls,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=60,
            r1_target=1e-8,
            completion=False,
        ),
    )
    assert result.termination == "independent_model_consensus"
    assert not result.target_met and result.best.r1 > 1e-8
    assert result.shared_search.phasing_starts < 4
    assert result.shared_search.model_consensus_stops == 1
    i, j = result.shared_search.model_consensus_trials
    assert i != j and max(i, j) < result.shared_search.phasing_starts
    assert result.shared_search.model_consensus_distance <= 0.02
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
