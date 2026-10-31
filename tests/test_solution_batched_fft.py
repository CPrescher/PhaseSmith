"""Blocked layout permutation, batch FFT conventions and complete structure recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.reference.solution_fft_reference import transform_3d_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_fft_transform"), reason="experimental-solution"
)


def transform(values, *, inverse=False, batched=False):
    values = np.asarray(values, complex)
    out = _core._solution_fft_transform(
        np.ascontiguousarray(values.real).ravel(),
        np.ascontiguousarray(values.imag).ravel(),
        values.shape,
        inverse,
        batched,
    )
    return (out["real"] + 1j * out["imaginary"]).reshape(values.shape)


@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 24, 8), (10, 14, 18), (8, 128, 8), (32, 16, 64)])
@pytest.mark.parametrize("kind", ["complex", "real", "impulse", "constant"])
@pytest.mark.parametrize("inverse", [False, True])
def test_direct_fft_is_bitwise_equal_and_matches_independent_numpy(shape, kind, inverse):
    rng = np.random.default_rng(534)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    if kind == "real":
        values.imag = 0
    elif kind == "impulse":
        values.fill(0)
        values[-1, 1, -2] = 1 + 2j
    elif kind == "constant":
        values.fill(1 - 0.3j)
    for scale in (1e-150, 1.0, 1e150):
        sample = values * scale
        before = sample.copy()
        old = transform(sample, inverse=inverse)
        new = transform(sample, inverse=inverse, batched=True)
        np.testing.assert_array_equal(new.ravel().view(np.uint64), old.ravel().view(np.uint64))
        ref = transform_3d_reference(sample, inverse=inverse)
        np.testing.assert_allclose(new, ref, rtol=2e-12, atol=2e-12 * scale)
        restored = transform(new, inverse=not inverse, batched=True) / sample.size
        np.testing.assert_allclose(restored, sample, rtol=2e-12, atol=2e-12 * scale)
        np.testing.assert_array_equal(before, sample)


@pytest.mark.parametrize("invalid", ["nan", "inf", "length", "odd", "small", "large"])
def test_fft_input_and_grid_boundaries_fail(invalid):
    real, imaginary = np.ones(512), np.zeros(512)
    shape = (8, 8, 8)
    if invalid == "nan":
        real[0] = np.nan
    elif invalid == "inf":
        imaginary[1] = np.inf
    elif invalid == "length":
        imaginary = imaginary[:-1]
    elif invalid == "odd":
        shape = (8, 7, 8)
    elif invalid == "small":
        shape = (8, 6, 8)
    else:
        shape = (8, 130, 8)
    for batched in (False, True):
        with pytest.raises(ValueError):
            _core._solution_fft_transform(real, imaginary, shape, False, batched)


@pytest.mark.parametrize("rank,omit", [(None, 0.0), (None, 0.1), (0, 0.1)])
def test_complete_dual_space_trajectory_is_bitwise_equal(rank, omit):
    from test_solution_dual_space import inputs

    shape, cell, h, amplitude, phases, support = inputs()
    args = (
        shape,
        list(cell.as_tuple()),
        h.ravel(),
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
    common = dict(reusable_workspaces=True, filtered_peak_search=True)
    old = _core._solution_count_free_dual_space(*args, **common)
    new = _core._solution_count_free_dual_space(*args, **common, batched_fft=True)
    for key in ("density", "history"):
        np.testing.assert_array_equal(new[key].view(np.uint64), old[key].view(np.uint64))
    assert new["adaptive_peaks"] == old["adaptive_peaks"]
    assert new["best_cycle"] == old["best_cycle"]


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_shared_counts_models_histories_and_final_adp_mode_are_preserved(symbol):
    from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
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
    new = solve_structure_elements(data, **common, batched_fft=True)
    assert old.target_met and new.target_met
    assert (
        dict(new.composition)
        == dict(old.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.best.species == old.best.species
    assert new.best.refinement.mode == old.best.refinement.mode == "anisotropic"
    np.testing.assert_array_equal(
        new.best.fractional_xyz.view(np.uint64), old.best.fractional_xyz.view(np.uint64)
    )
    for a, b in zip(new.histories, old.histories, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    assert old.shared_search.batched_phase_transforms == 0
    assert new.shared_search.batched_phase_transforms > 0
    assert new.shared_search.batched_fft
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], batched_fft=True
        )
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, batched_fft=1)
