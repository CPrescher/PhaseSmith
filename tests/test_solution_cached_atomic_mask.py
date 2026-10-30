"""Exact cached Gaussian mask geometry and complete current execution trajectories."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_dual_reference import atomic_projection_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SharedOrbitPool"), reason="experimental-solution"
)


@pytest.mark.parametrize("shape", [(8, 8, 8), (10, 14, 18), (32, 16, 32), (8, 128, 8)])
@pytest.mark.parametrize("cell", [UnitCell(6, 7, 8, 90, 90, 90), UnitCell(6, 7, 8, 78, 96, 107)])
@pytest.mark.parametrize("cycle", [0, 2, 49, 50, 53])
def test_cached_projection_is_bitwise_equal_at_periodic_edges_and_matches_numpy(shape, cell, cycle):
    density = np.random.default_rng(901).normal(0, 0.1, shape)
    density[0, 0, 0] = 6
    density[-1, 0, shape[2]//2] = 7
    for scale in [1e-200, 1.0, 1e200]:
        values = np.ascontiguousarray(density*scale).ravel()
        args = (values, shape, list(cell.as_tuple()), 3, 0.25, 0.2, 123, cycle)
        old = _core._solution_atomic_projection(*args)
        new = _core._solution_atomic_projection(*args, cached=True)
        np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64))
        np.testing.assert_array_equal(values.reshape(shape), density*scale)
        if scale == 1:
            ref = atomic_projection_reference(
                density, cell.geometry().direct_basis, 3, 0.25,
                cycle=cycle, seed=123, omit_fraction=0.2,
            )
            np.testing.assert_allclose(new.reshape(shape), ref, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("kind", ["zero", "negative", "plateau", "signed_zero"])
def test_empty_or_tied_maps_preserve_every_bit(kind):
    density = np.zeros((8, 8, 8))
    if kind == "negative":
        density.fill(-1)
    elif kind == "plateau":
        density.fill(1)
    elif kind == "signed_zero":
        density.ravel()[::2] = -0.0
    args = (density.ravel(), density.shape, [6, 7, 8, 90, 90, 90], 3, 0.25, 0.1, 55, 2)
    old = _core._solution_atomic_projection(*args)
    new = _core._solution_atomic_projection(*args, cached=True)
    np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64))


@pytest.mark.parametrize("invalid", ["nan", "length", "sigma", "support", "odd", "count"])
def test_cached_projection_rejects_original_invalid_inputs(invalid):
    density, shape, sigma, count = np.ones(512), (8, 8, 8), 0.25, 3
    if invalid == "nan":
        density[0] = np.nan
    elif invalid == "length":
        density = density[:-1]
    elif invalid == "sigma":
        sigma = 0
    elif invalid == "support":
        sigma = 5
    elif invalid == "odd":
        shape = (8, 7, 8)
    else:
        count = 0
    with pytest.raises(ValueError):
        _core._solution_atomic_projection(
            density, shape, [6, 7, 8, 90, 90, 90], count, sigma, 0.1, 55, 2, cached=True
        )


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("rank,omit", [(None, 0.0), (None, 0.1), (0, 0.1)])
def test_complete_current_trajectories_remain_bitwise_equal(rank, omit, real):
    from test_solution_dual_space import inputs

    shape, cell, h, amplitude, phases, support = inputs()
    args = (
        shape, list(cell.as_tuple()), h.ravel(), amplitude, phases, support.ravel(),
        128, 1.5, 64, 0.25, 3.0, omit, 1034, rank,
    )
    common = dict(
        reusable_workspaces=True, filtered_peak_search=True, batched_fft=True, real_fft=real
    )
    old = _core._solution_count_free_dual_space(*args, **common)
    new = _core._solution_count_free_dual_space(*args, **common, fused_phase_workspace=True)
    for key in ["density", "history"]:
        np.testing.assert_array_equal(new[key].view(np.uint64), old[key].view(np.uint64))
    assert old["adaptive_peaks"] == new["adaptive_peaks"]
    assert old["cached_mask_cycles"] == 0 and new["cached_mask_cycles"] == 128


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_complete_structure_and_counts_are_unchanged(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
        cell=cell, space_group=group, elements=["Fe", "O"], maximum_atoms=12,
        symmetry_recycling=True, tangent_refinement=True, wilson_normalization=True,
        phase_quality_selection=True, phase_origin_selection=True, chemical_assignment=True,
        iterative_refinement=True, score_only_evaluation=True, reusable_fourier_workspaces=True,
        work_sliced_phasing=True, isotropic_assignment_refinement=True, filtered_peak_search=True,
        batched_fft=True, real_fft=True, cached_normal_refinement=True,
        options=SolutionOptions(
            phasing="dual_space", trials=4, cycles=800, max_seconds=60,
            r1_target=0.005, completion=False,
        ),
    )
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(data, **common, fused_phase_workspace=True)
    assert old.target_met and new.target_met
    assert dict(old.composition) == dict(new.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    assert new.best.refinement.mode == old.best.refinement.mode == "anisotropic"
    np.testing.assert_array_equal(
        new.best.fractional_xyz.view(np.uint64), old.best.fractional_xyz.view(np.uint64)
    )
    assert new.shared_search.fused_phase_workspace
    assert new.shared_search.cached_mask_cycles == new.shared_search.fourier_workspace_cycles
    with pytest.raises(ValueError, match="requires reusable"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], fused_phase_workspace=True
        )
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, fused_phase_workspace=1)
