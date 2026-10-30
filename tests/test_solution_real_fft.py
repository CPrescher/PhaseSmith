"""Packed real Fourier equations, general complex return maps and complete recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_fft_reference import real_density_transform_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_real_fft_transform"), reason="experimental-solution"
)


def transform(values, to_density=False):
    values = np.asarray(values, complex)
    out = _core._solution_real_fft_transform(
        np.ascontiguousarray(values.real).ravel(),
        np.ascontiguousarray(values.imag).ravel(),
        values.shape, to_density,
    )
    return (out["real"] + 1j * out["imaginary"]).reshape(values.shape)


@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 24, 8), (10, 14, 18), (8, 128, 8), (32, 16, 64)])
@pytest.mark.parametrize("kind", ["random", "impulse", "constant", "nyquist"])
@pytest.mark.parametrize("to_density", [False, True])
def test_real_operations_match_independent_numpy_at_extreme_finite_scales(shape, kind, to_density):
    rng = np.random.default_rng(312)
    values = rng.normal(size=shape).astype(complex)
    if to_density:
        values += 1j * rng.normal(size=shape)  # Deliberately non-Hermitian.
    if kind == "impulse":
        values.fill(0)
        values[-1, 1, -2] = 1 + (2j if to_density else 0)
    elif kind == "constant":
        values.fill(1 + (0.3j if to_density else 0))
    elif kind == "nyquist":
        values = (-1.0) ** np.indices(shape).sum(axis=0)
    for scale in [1e-150, 1.0, 1e150]:
        sample = values * scale
        before = sample.copy()
        got = transform(sample, to_density)
        ref = real_density_transform_reference(sample, to_density=to_density)
        np.testing.assert_allclose(got, ref, rtol=2e-12, atol=8e-12 * scale)
        np.testing.assert_array_equal(sample, before)
        again = transform(sample, to_density)
        np.testing.assert_array_equal(got.view(np.uint64), again.view(np.uint64))
        if to_density:
            assert np.all(got.imag == 0)
        else:
            back = transform(got, True) / sample.size
            np.testing.assert_allclose(back, sample, rtol=2e-12, atol=8e-12 * scale)
            assert got[0, 0, 0].real == pytest.approx(sample.real.sum(), rel=2e-12, abs=8e-12*scale)


@pytest.mark.parametrize("invalid", ["nan", "inf", "length", "odd", "small", "large", "complex"])
def test_invalid_density_or_grid_is_rejected(invalid):
    real, imaginary, shape = np.ones(512), np.zeros(512), (8, 8, 8)
    if invalid == "nan":
        real[0] = np.nan
    elif invalid == "inf":
        imaginary[0] = np.inf
    elif invalid == "length":
        imaginary = imaginary[:-1]
    elif invalid == "odd":
        shape = (8, 7, 8)
    elif invalid == "small":
        shape = (8, 6, 8)
    elif invalid == "large":
        shape = (8, 130, 8)
    else:
        imaginary[0] = 1
    with pytest.raises(ValueError):
        _core._solution_real_fft_transform(real, imaginary, shape, False)


@pytest.mark.parametrize("cycles", [1, 2, 4, 8])
@pytest.mark.parametrize("rank", [None, 0])
def test_short_phase_trajectory_agrees_and_repeats_deterministically(cycles, rank):
    from test_solution_dual_space import inputs

    shape, cell, h, amplitude, phases, support = inputs()
    args = (
        shape, list(cell.as_tuple()), h.ravel(), amplitude, phases, support.ravel(),
        cycles, 1.5, 64, 0.25, 3.0, 0.1, 1034, rank,
    )
    common = dict(reusable_workspaces=True, filtered_peak_search=True, batched_fft=True)
    old = _core._solution_count_free_dual_space(*args, **common)
    new = _core._solution_count_free_dual_space(*args, **common, real_fft=True)
    for key in ["density", "history"]:
        np.testing.assert_allclose(new[key], old[key], rtol=2e-10, atol=2e-12)
    assert new["adaptive_peaks"] == old["adaptive_peaks"]
    again = _core._solution_count_free_dual_space(*args, **common, real_fft=True)
    np.testing.assert_array_equal(new["density"].view(np.uint64), again["density"].view(np.uint64))


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_complete_element_only_search_preserves_counts_and_final_anisotropic_mode(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    common = dict(
        cell=cell, space_group=group, elements=["Fe", "O"], maximum_atoms=12,
        symmetry_recycling=True, tangent_refinement=True, wilson_normalization=True,
        phase_quality_selection=True, phase_origin_selection=True, chemical_assignment=True,
        iterative_refinement=True, score_only_evaluation=True, reusable_fourier_workspaces=True,
        work_sliced_phasing=True, isotropic_assignment_refinement=True, filtered_peak_search=True,
        batched_fft=True, cached_normal_refinement=True,
        options=SolutionOptions(
            phasing="dual_space", trials=4, cycles=800, max_seconds=60,
            r1_target=0.005, completion=False,
        ),
    )
    old = solve_structure_elements(data, **common)
    new = solve_structure_elements(data, **common, real_fft=True)
    assert old.target_met and new.target_met
    assert dict(old.composition) == dict(new.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    assert new.best.refinement.mode == old.best.refinement.mode == "anisotropic"
    assert new.shared_search.real_fft and new.shared_search.real_phase_transforms > 0
    assert old.shared_search.real_phase_transforms == 0
    with pytest.raises(ValueError, match="requires batched"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], real_fft=True
        )
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, real_fft=1)


@pytest.mark.parametrize("rank", [None, 0])
def test_long_packed_phase_runs_are_deterministic(rank):
    from test_solution_dual_space import inputs

    shape, cell, h, amplitude, phases, support = inputs()
    args = (
        shape, list(cell.as_tuple()), h.ravel(), amplitude, phases, support.ravel(),
        256, 1.5, 64, 0.25, 3.0, 0.1, 1034, rank,
    )
    common = dict(
        reusable_workspaces=True, filtered_peak_search=True, batched_fft=True, real_fft=True
    )
    a = _core._solution_count_free_dual_space(*args, **common)
    b = _core._solution_count_free_dual_space(*args, **common)
    for key in ["density", "history"]:
        np.testing.assert_array_equal(a[key].view(np.uint64), b[key].view(np.uint64))
    assert a["adaptive_peaks"] == b["adaptive_peaks"]
