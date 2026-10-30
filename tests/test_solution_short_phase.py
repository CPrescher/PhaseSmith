"""Short early starts, unchanged Fourier prefixes and full-length fallback."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_schedule_reference import phase_cycle_quotas

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_phase_cycle_quota"), reason="experimental-solution"
)


@pytest.mark.parametrize("cycles", [1, 2, 31, 32, 33, 63, 64, 127, 128, 799, 800, 20_000])
@pytest.mark.parametrize("trials", [1, 2, 3, 4, 9, 128])
def test_short_quotas_match_independent_reference_and_keep_full_fallback(cycles, trials):
    actual = [_core._phase_cycle_quota(cycles, trials, i, short=True) for i in range(trials)]
    assert actual == phase_cycle_quotas(cycles, trials, short=True)
    old = phase_cycle_quotas(cycles, trials)
    assert all(1 <= a <= b <= cycles for a, b in zip(actual, old, strict=True))
    assert actual[trials // 2 :] == [cycles] * (trials - trials // 2)
    assert actual[-1] == cycles


@pytest.mark.parametrize(
    "cycles,trials,trial", [(0, 4, 0), (20_001, 4, 0), (800, 0, 0), (800, 129, 0), (800, 4, 4)]
)
def test_invalid_native_controls_are_unchanged(cycles, trials, trial):
    with pytest.raises(ValueError):
        _core._phase_cycle_quota(cycles, trials, trial, short=True)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_keeps_models_counts_and_identical_cycle_prefixes(symbol):
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
    new = solve_structure_elements(data, **common, short_phase_starts=True)
    assert old.target_met and new.target_met
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.refinement.mode == "anisotropic"
    np.testing.assert_allclose(old.best.fractional_xyz, new.best.fractional_xyz, rtol=0, atol=2e-7)
    assert new.shared_search.short_phase_starts
    assert not old.shared_search.short_phase_starts
    for phase in new.shared_search.phase_slices:
        assert phase.cycle_quota == _core._phase_cycle_quota(800, 4, phase.trial, short=True)
        assert phase.completed_cycles <= phase.cycle_quota
    for a, b in zip(old.histories, new.histories, strict=False):
        n = min(len(a), len(b))
        np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))
    np.testing.assert_array_equal(before, data.intensity)
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, short_phase_starts=1)
    bad = dict(common, work_sliced_phasing=False)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **bad, short_phase_starts=True)
