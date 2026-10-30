"""Exact filtered peak decisions, full phase trajectories, and structure recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_count_free_reference import adaptive_peaks_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_filtered_peaks"), reason="experimental-solution"
)


@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 24, 8), (8, 128, 8)])
@pytest.mark.parametrize("kind", ["noise", "plateau", "corners", "negative", "zero", "threshold"])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("cap", [1, 128])
def test_filtered_peaks_match_original_bits_and_independent_numpy(shape, kind, sigma, cap):
    rng = np.random.default_rng(123)
    density = rng.normal(size=shape)
    if kind == "plateau":
        density = np.round(density)
    elif kind == "corners":
        density.fill(0)
        density[0, 0, 0] = density[-1, 0, 0] = 3.0
        density[4, 4, 4] = 2.0
    elif kind == "negative":
        density = -np.abs(density)
    elif kind == "zero":
        density.fill(0)
    elif kind == "threshold":
        density.fill(1)
    for scale in [1e-250, 1.0, 1e250]:
        values = density * scale
        before = values.copy()
        old = _core._solution_filtered_peaks(values.ravel(), shape, sigma, cap, False)
        new = _core._solution_filtered_peaks(values.ravel(), shape, sigma, cap, True)
        np.testing.assert_array_equal(new["peaks"].view(np.uint64), old["peaks"].view(np.uint64))
        assert np.float64(new["effective_sigma"]).view(np.uint64) == np.float64(
            old["effective_sigma"]
        ).view(np.uint64)
        ref, threshold = adaptive_peaks_reference(values, sigma, cap)
        np.testing.assert_allclose(new["peaks"], ref, rtol=0, atol=3e-15)
        assert new["effective_sigma"] == threshold
        np.testing.assert_array_equal(before, values)


def test_threshold_equality_and_plateau_ties_are_preserved():
    rho = np.ones((8, 8, 8))
    assert len(_core._solution_filtered_peaks(rho.ravel(), rho.shape, 1.0, 128)["peaks"]) == 1
    assert (
        len(
            _core._solution_filtered_peaks(rho.ravel(), rho.shape, np.nextafter(1.0, 2.0), 128)[
                "peaks"
            ]
        )
        == 0
    )
    rho.fill(0)
    rho[::4, ::4, ::4] = 1
    result = _core._solution_filtered_peaks(rho.ravel(), rho.shape, 0.1, 1)
    assert result["effective_sigma"] > 0.1
    assert len(result["peaks"]) == 0  # Tied heights leave the selection together.


@pytest.mark.parametrize("kind", ["nan", "inf", "length", "sigma", "cap", "shape"])
def test_same_invalid_boundaries_fail(kind):
    rho = np.ones(512)
    shape, sigma, cap = (8, 8, 8), 1.5, 128
    if kind in ("nan", "inf"):
        rho[2] = np.nan if kind == "nan" else np.inf
    elif kind == "length":
        rho = rho[:-1]
    elif kind == "sigma":
        sigma = 0.0
    elif kind == "cap":
        cap = 129
    else:
        shape = (8, 7, 8)
    for filtered in (False, True):
        with pytest.raises(ValueError):
            _core._solution_filtered_peaks(rho, shape, sigma, cap, filtered)


@pytest.mark.parametrize("rank,omit", [(None, 0.0), (None, 0.1), (0, 0.1)])
def test_entire_count_free_phase_trajectory_is_bitwise_equal(rank, omit):
    from test_solution_dual_space import inputs

    shape, cell, hkl, amplitude, phases, support = inputs()
    args = (
        shape,
        list(cell.as_tuple()),
        hkl.ravel(),
        amplitude,
        phases,
        support.ravel(),
        80,
        1.5,
        64,
        0.25,
        3.0,
        omit,
        1034,
        rank,
    )
    old = _core._solution_count_free_dual_space(*args, reusable_workspaces=True)
    new = _core._solution_count_free_dual_space(
        *args, reusable_workspaces=True, filtered_peak_search=True
    )
    for key in ("density", "history"):
        np.testing.assert_array_equal(new[key].view(np.uint64), old[key].view(np.uint64))
    assert new["adaptive_peaks"] == old["adaptive_peaks"]
    assert new["best_cycle"] == old["best_cycle"]


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_shared_search_retains_models_counts_and_phase_history(symbol):
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
    new = solve_structure_elements(data, **common, filtered_peak_search=True)
    assert old.target_met and new.target_met
    assert (
        dict(new.composition)
        == dict(old.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.species == old.best.species
    np.testing.assert_array_equal(
        new.best.fractional_xyz.view(np.uint64), old.best.fractional_xyz.view(np.uint64)
    )
    for a, b in zip(new.histories, old.histories, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    assert new.shared_search.filtered_peak_search
    assert new.shared_search.filtered_peak_cycles == new.shared_search.fourier_workspace_cycles
    assert old.shared_search.filtered_peak_cycles == 0
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], filtered_peak_search=True
        )
