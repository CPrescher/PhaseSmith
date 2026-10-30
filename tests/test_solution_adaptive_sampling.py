"""Stable checkpoint choices, independent sampling, and full model recovery."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_checkpoint_reference import (
    checkpoint_reference,
    origin_sampling_reference,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_phase_checkpoint_select"), reason="experimental-solution"
)


@pytest.mark.parametrize("seed", [1, 9, 57])
def test_checkpoint_policy_matches_independent_reference(seed):
    rng = np.random.default_rng(seed)
    maps = [rng.random(512)]
    for i in range(1, 10):
        maps.append(maps[-1] * 1.7 + rng.random(512) * (0.002 if i % 2 else 1.0))
    cycles = [16, 64, 256, 800, 16, 64, 256, 800, 32, 64]
    quality = [0.7, 0.704, 0.71, 0.72, 0.6, 0.6, 0.8, 0.8, 0.91, 0.92]
    scores = [0.1] * 10
    ceilings = [n == 800 for n in cycles]
    native = _core._phase_checkpoint_select(
        [m.tolist() for m in maps], cycles, quality, scores, ceilings
    )
    reference = checkpoint_reference(maps, cycles, quality, scores, ceilings)
    for a, b in zip(native, reference, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if key == "map_cosine" and a[key] is not None:
                assert a[key] == pytest.approx(b[key], abs=2e-14)
            else:
                assert a[key] == b[key]
    assert native[1]["reason"] == "stable_map"
    assert native[4]["reason"] == "continue"
    assert native[-1]["reason"] == "quality_target"


@pytest.mark.parametrize(
    "maps,quality,scores",
    [
        ([[0.0] * 4], [0.7], [0.1]),
        ([[float("nan")] * 4], [0.7], [0.1]),
        ([[1.0] * 4], [float("inf")], [0.1]),
        ([[1.0] * 4], [0.7], [-0.1]),
        ([[1.0] * 4, [1.0] * 5], [0.7, 0.7], [0.1, 0.1]),
    ],
)
def test_invalid_checkpoints_fail(maps, quality, scores):
    with pytest.raises(ValueError):
        _core._phase_checkpoint_select(
            maps, [16, 64][: len(maps)], quality, scores, [False] * len(maps)
        )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "P21", "P4", "Pm", "P 31 2 1"])
def test_origin_grid_is_frequency_derived_and_padding_independent(symbol):
    from test_solution_phase_origin import fixture

    plan, h, amp, f, rotations, _, density, _ = fixture(symbol)
    expected = origin_sampling_reference(h, amp, rotations)
    results = []
    for shape in [(16, 16, 16), (32, 32, 32), (32, 64, 32)]:
        grid = np.zeros(shape, dtype=complex)
        grid[tuple((h % shape).T)] = f
        rho = np.fft.fftn(grid).real / grid.size
        actual = plan.search(rho.ravel(), shape, independent_sampling=True)
        assert tuple(actual["sampling_shape"]) == expected
        assert actual["origins"][0]["alpha"] < 1e-10
        results.append(actual["origins"][0])
    assert results[0]["hand"] == results[1]["hand"] == results[2]["hand"]
    legacy = plan.search(density.ravel(), density.shape)
    assert tuple(legacy["sampling_shape"]) == density.shape


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
@pytest.mark.parametrize("adaptive,independent", [(True, False), (False, True), (True, True)])
def test_both_controls_recover_full_models(symbol, adaptive, independent):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
    before = data.intensity.copy()
    result = solve_structure_elements(
        data,
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
        adaptive_checkpoints=adaptive,
        independent_origin_sampling=independent,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    assert len(result.best.species) == len(xyz)
    best = np.inf
    for hand in (-1, 1):
        for anchor in xyz[species == result.best.species[0]]:
            current = (
                hand * result.best.fractional_xyz + anchor - hand * result.best.fractional_xyz[0]
            ) % 1
            maximum = 0.0
            for element in set(species):
                d = (
                    xyz[species == element, None]
                    - current[None, np.asarray(result.best.species) == element]
                    + 0.5
                ) % 1 - 0.5
                distance = np.linalg.norm(d @ cell.geometry().direct_basis.T, axis=-1)
                maximum = max(maximum, distance.min(axis=1).max(), distance.min(axis=0).max())
            best = min(best, maximum)
    assert best < 1e-5
    np.testing.assert_array_equal(data.intensity, before)
    audit = result.shared_search
    assert audit.adaptive_checkpoints == adaptive
    assert audit.independent_origin_sampling == independent
    assert bool(audit.phase_checkpoints) == adaptive
    if adaptive:
        for c in audit.phase_checkpoints:
            quality = next(q for q in audit.phase_quality if q.trial == c.trial)
            assert c.cfom == quality.cfom[c.cycles - 1]
            assert c.best_cfom >= c.cfom - 1e-12 or (c.cycles - 1) % 3 == 2
        assert all(c.cycles <= 800 for c in audit.phase_checkpoints)
        assert all(
            c.reason in {"continue", "stable_map", "quality_target", "cycle_ceiling"}
            for c in audit.phase_checkpoints
        )
    assert result.to_record()["shared_search"]["adaptive_checkpoints"] == adaptive


@pytest.mark.parametrize(
    "quality,score,perturb,expected",
    [
        (0.649, 0.1, False, "continue"),
        (0.7, 0.103, False, "continue"),
        (0.72, 0.1, False, "continue"),
        (0.7, 0.1, True, "continue"),
        (0.7, 0.1, False, "stable_map"),
    ],
)
def test_plateau_requires_quality_fit_and_map_agreement(quality, score, perturb, expected):
    first = np.arange(1, 17, dtype=float)
    second = first[::-1] if perturb else first * 1e150
    records = _core._phase_checkpoint_select(
        [first.tolist(), second.tolist()], [16, 64], [0.7, quality], [0.1, score], [False, False]
    )
    assert records[-1]["reason"] == expected


@pytest.mark.parametrize("flag", ["adaptive_checkpoints", "independent_origin_sampling"])
def test_new_options_reject_invalid_dependencies_and_nonbooleans(flag):
    from phasesmith.experimental.solution import solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic("P1")
    common = dict(cell=cell, space_group=group, elements=["Fe", "O"])
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **common, **{flag: True})
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, **{flag: 1})


def test_high_harmonic_sampling_cap_is_observable_and_zero_weight_ignored():
    from phasesmith.io.space_groups import space_group_by_symbol

    group = space_group_by_symbol("P-1").space_group
    h = np.array([[-40, 0, 0], [-1, 0, 0], [1, 0, 0], [40, 0, 0]], dtype=np.int64)
    f = np.exp(2j * np.pi * h[:, 0] * 0.173)
    shape = (128, 8, 8)
    grid = np.zeros(shape, dtype=complex)
    grid[tuple((h % shape).T)] = f
    density = (np.fft.fftn(grid).real / grid.size).ravel()
    for amp, extent, sampling in [([1.0, 1.0, 1.0, 1.0], 80, 128), ([0.0, 1.0, 1.0, 0.0], 2, 16)]:
        plan = _core._PhaseOriginPlan(group._native, h.ravel(), np.array(amp))
        actual = plan.search(density, shape, independent_sampling=True)
        assert tuple(actual["harmonic_extent"]) == (extent, 0, 0)
        assert tuple(actual["sampling_shape"]) == (sampling, 16, 16)


@pytest.mark.parametrize("moving", [False, True])
def test_stored_champion_does_not_establish_live_map_convergence(moving):
    first = np.arange(1.0, 17.0)
    live = first[::-1] if moving else first
    maps = [first.tolist(), live.tolist()]
    quality, scores, best = [0.79, 0.74], [0.08, 0.09], [0.83, 0.83]
    native = _core._phase_checkpoint_select(maps, [64, 256], quality, scores, [False, False], best)
    reference = checkpoint_reference(maps, [64, 256], quality, scores, [False, False], best)
    assert native[-1]["reason"] == reference[-1]["reason"] == "continue"
    # Even equal live maps cannot pass when live quality is far below the champion.
    quality = [0.74, 0.74]
    native = _core._phase_checkpoint_select(
        maps, [64, 256], quality, [0.09, 0.09], [False, False], best
    )
    assert native[-1]["reason"] == "continue"


def test_existing_best_quality_target_remains_available_with_live_map_guard():
    records = _core._phase_checkpoint_select(
        [[1.0, 2.0], [2.0, 1.0]], [16, 32], [0.8, 0.7], [0.1, 0.2], [False, False], [0.8, 0.91]
    )
    assert records[-1]["reason"] == "quality_target"
    assert records[-1]["best_cfom"] == 0.91
    assert records[-1]["cfom"] == 0.7
