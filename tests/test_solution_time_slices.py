"""Count-free phase quotas, fallback starts and unchanged model validation."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
from phasesmith.experimental.solution_schedule_reference import phase_cycle_quotas

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_phase_cycle_quota"), reason="experimental-solution"
)


@pytest.mark.parametrize("cycles", [1, 2, 31, 32, 33, 64, 800, 20_000])
@pytest.mark.parametrize("trials", [1, 2, 3, 4, 9, 128])
def test_cycle_quotas_match_independent_integer_reference(cycles, trials):
    actual = [_core._phase_cycle_quota(cycles, trials, i) for i in range(trials)]
    assert actual == phase_cycle_quotas(cycles, trials)
    assert actual[-1] == cycles
    assert all(1 <= q <= cycles for q in actual)


@pytest.mark.parametrize(
    "cycles,trials,trial", [(0, 4, 0), (20_001, 4, 0), (800, 0, 0), (800, 129, 0), (800, 4, 4)]
)
def test_invalid_native_work_budgets_fail(cycles, trials, trial):
    with pytest.raises(ValueError):
        _core._phase_cycle_quota(cycles, trials, trial)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_work_slices_preserve_counts_geometry_and_do_not_change_cycle_equations(symbol):
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
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
    control = solve_structure_elements(data, **common, work_sliced_phasing=False)
    work = solve_structure_elements(data, **common, work_sliced_phasing=True)
    assert control.target_met and work.target_met
    assert (
        dict(work.composition)
        == dict(control.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    for result in (control, work):
        audit = result.shared_search
        assert audit.phase_slices
        for a in audit.phase_slices:
            expected = (
                _core._phase_cycle_quota(800, 4, a.trial) if audit.work_sliced_phasing else 800
            )
            assert a.cycle_quota == expected
            assert 0 <= a.completed_cycles <= expected
            assert a.seconds >= 0
            assert a.stop_reason in {
                "phase_time_limit",
                "cycle_limit",
                "quality_target",
                "cancelled",
                "degenerate",
            }
        # Compare selected coordinates with the synthetic reference after prediction.
        error = np.inf
        for hand in (-1, 1):
            for anchor in xyz[species == result.best.species[0]]:
                model = (
                    hand * result.best.fractional_xyz
                    + anchor
                    - hand * result.best.fractional_xyz[0]
                ) % 1
                maximum = 0.0
                for e in set(species):
                    d = (
                        xyz[species == e, None]
                        - model[None, np.asarray(result.best.species) == e]
                        + 0.5
                    ) % 1 - 0.5
                    distance = np.linalg.norm(d @ cell.geometry().direct_basis.T, axis=-1)
                    maximum = max(maximum, distance.min(axis=0).max(), distance.min(axis=1).max())
                error = min(error, maximum)
        assert error < 1e-5
    # Equal seeds give exactly the same phase prefix when the wall guard is inactive.
    for a, b in zip(control.histories, work.histories, strict=False):
        n = min(len(a), len(b))
        np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))
    np.testing.assert_array_equal(data.intensity, before)
    assert work.to_record()["shared_search"]["work_sliced_phasing"]


def test_scheduler_rejects_ambiguous_placement_and_nonboolean_flags():
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic("P1")
    common = dict(cell=cell, space_group=group, elements=["Fe", "O"])
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, work_sliced_phasing=1)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **common, work_sliced_phasing=True)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(
            data,
            **common,
            phase_origin_selection=True,
            phase_quality_selection=True,
            adaptive_checkpoints=True,
            work_sliced_phasing=True,
        )
