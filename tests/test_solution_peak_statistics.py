"""Guarded SIMD RMS and streamed maxima retain scalar threshold decisions."""

import math

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_count_free_reference import adaptive_peaks_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_filtered_peaks"), reason="experimental-solution"
)


def select(rho, sigma, cap, streamed):
    return _core._solution_filtered_peaks(
        np.ascontiguousarray(rho).ravel(), rho.shape, sigma, cap, True, streamed=streamed
    )


@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 24, 8), (8, 128, 8)])
@pytest.mark.parametrize("kind", ["noise", "plateau", "corners", "negative", "zero", "threshold"])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("cap", [1, 128])
def test_exact_decisions_and_independent_reference(shape, kind, sigma, cap):
    rho = np.random.default_rng(123).normal(size=shape)
    if kind == "plateau":
        rho = np.round(rho)
    elif kind == "corners":
        rho.fill(0)
        rho[0, 0, 0] = rho[-1, 0, 0] = 3.0
        rho[4, 4, 4] = 2.0
    elif kind == "negative":
        rho = -np.abs(rho)
    elif kind == "zero":
        rho.fill(0)
    elif kind == "threshold":
        rho.fill(1)
    for scale in [1e-250, 1.0, 1e250]:
        values = rho * scale
        before = values.copy()
        old, new = select(values, sigma, cap, False), select(values, sigma, cap, True)
        np.testing.assert_array_equal(old["peaks"].view(np.uint64), new["peaks"].view(np.uint64))
        assert old["effective_sigma"] == new["effective_sigma"]
        ref, effective = adaptive_peaks_reference(values, sigma, cap)
        np.testing.assert_allclose(new["peaks"], ref, rtol=0, atol=3e-15)
        assert new["effective_sigma"] == effective
        np.testing.assert_array_equal(before, values)


@pytest.mark.parametrize("scale", [1e-310, 1.0, 1e300])
def test_adversarial_thresholds_use_scalar_fallback(scale):
    rho = np.random.default_rng(317).normal(size=(16, 16, 16)) * scale
    maximum = float(np.max(np.abs(rho)))
    total = 0.0
    for value in rho.ravel():
        q = float(value) / maximum
        total += q * q
    rms = maximum * math.sqrt(total / rho.size)
    sigma = float(np.max(rho)) / rms
    for cut in [np.nextafter(sigma, 0.0), sigma, np.nextafter(sigma, np.inf)]:
        old, new = select(rho, cut, 128, False), select(rho, cut, 128, True)
        assert new["peak_rms_fallbacks"] == 1
        np.testing.assert_array_equal(old["peaks"].view(np.uint64), new["peaks"].view(np.uint64))
        assert old["effective_sigma"] == new["effective_sigma"]


@pytest.mark.parametrize("invalid", ["nan", "inf", "sigma", "cap", "shape", "length"])
def test_original_invalid_boundaries(invalid):
    rho, shape, sigma, cap = np.ones(512), (8, 8, 8), 1.5, 128
    if invalid == "nan":
        rho[2] = np.nan
    elif invalid == "inf":
        rho[2] = np.inf
    elif invalid == "sigma":
        sigma = 0.0
    elif invalid == "cap":
        cap = 129
    elif invalid == "shape":
        shape = (8, 7, 8)
    else:
        rho = rho[:-1]
    for streamed in [False, True]:
        with pytest.raises(ValueError):
            _core._solution_filtered_peaks(rho, shape, sigma, cap, True, streamed=streamed)


@pytest.mark.parametrize("seed", [0, 42, 1034])
@pytest.mark.parametrize("rank", [None, 0])
@pytest.mark.parametrize("omit", [0.0, 0.1])
def test_complete_phase_history_preserves_every_float_bit(seed, rank, omit):
    from test_solution_dual_space import inputs

    shape, cell, h, amp, phases, support = inputs()
    args = (
        shape,
        list(cell.as_tuple()),
        h.ravel(),
        amp,
        phases,
        support.ravel(),
        80,
        1.5,
        64,
        0.25,
        3.0,
        omit,
        seed,
        rank,
    )
    common = dict(
        reusable_workspaces=True,
        filtered_peak_search=True,
        batched_fft=True,
        real_fft=True,
        fused_phase_workspace=True,
    )
    old = _core._solution_count_free_dual_space(*args, **common)
    new = _core._solution_count_free_dual_space(*args, **common, streamed_peak_search=True)
    for key in ["density", "history"]:
        np.testing.assert_array_equal(old[key].view(np.uint64), new[key].view(np.uint64))
    assert old["adaptive_peaks"] == new["adaptive_peaks"]
    assert old["best_cycle"] == new["best_cycle"]
    assert old["streamed_peak_cycles"] == 0
    assert new["streamed_peak_cycles"] == len(new["history"])
    assert 0 <= new["peak_rms_fallbacks"] <= new["streamed_peak_cycles"]


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
        specialized_isotropic_refinement=True,
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
    new = solve_structure_elements(data, **common, streamed_peak_search=True)
    assert old.target_met and new.target_met
    assert new.best.refinement.mode == "anisotropic"
    assert (
        dict(old.composition)
        == dict(new.composition)
        == {e: int(np.count_nonzero(species == e)) for e in set(species)}
    )
    assert new.shared_search.streamed_peak_cycles == new.shared_search.filtered_peak_cycles > 0
    assert old.shared_search.streamed_peak_cycles == 0
    np.testing.assert_array_equal(old.best.fractional_xyz, new.best.fractional_xyz)
    assert old.best.r1 == new.best.r1
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, streamed_peak_search=1)

    bad = dict(common, filtered_peak_search=False)
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **bad, streamed_peak_search=True)
