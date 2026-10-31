"""Bounded reflection execution preserves fused equations and ordered fits."""

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution_reference import symmetric_model_reference
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="experimental-solution"
)
STATS = {"requested_workers", "workers", "parallel_maps", "parallel_tasks", "max_active_workers"}


def case(symbol="P 1", mode="isotropic", cache=True, weighted=False, specialized=True):
    import gemmi

    cell = UnitCell(9, 10, 11, 90, 100, 90)
    full = []
    species = []
    for i, point in enumerate(np.random.default_rng(769).uniform(0.05, 0.95, (20, 3))):
        for op in gemmi.find_spacegroup_by_name(symbol).operations():
            full.append(op.apply_to_xyz(point))
            species.append("Si" if i < 3 else "O")
    hkl = np.array(
        [h for h in itertools.product(range(-6, 7), repeat=3) if h > tuple(-v for v in h)],
        dtype=np.int64,
    )
    obs = np.random.default_rng(713).uniform(1, 20, len(hkl))
    if weighted:
        amplitudes = {}
        for i, h in enumerate(hkl):
            family = min(
                tuple(v)
                for op in gemmi.find_spacegroup_by_name(symbol).operations()
                for v in [op.apply_to_hkl(h), [-x for x in op.apply_to_hkl(h)]]
            )
            obs[i] = amplitudes.setdefault(family, obs[i])
    model = _core._SymmetricSolutionModel(
        list(cell.as_tuple()),
        space_group_by_symbol(symbol).space_group._native,
        species,
        np.array(full).ravel(),
        hkl.ravel(),
        obs,
        mode,
        0.01,
        cache,
        weighted,
        specialized,
    )
    return model, cell, hkl, obs


def same(a, b):
    for key in set(a) - STATS:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64), err_msg=key)


@pytest.mark.parametrize("symbol", ["P 1", "P -1", "P m"])
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
@pytest.mark.parametrize(
    "cache,weighted,specialized", [(True, False, True), (False, False, False), (True, True, True)]
)
def test_worker_counts_keep_exact_values_derivatives_and_fits(
    symbol, mode, cache, weighted, specialized
):
    model, cell, hkl, obs = case(symbol, mode, cache, weighted, specialized)
    p = model.initial_parameters(0.015)
    p += np.random.default_rng(23).normal(0, 0.002, len(p))
    serial = model.evaluate(p)
    for workers in [2, 4, 8]:
        value = model.evaluate(p, workers=workers)
        same(serial, value)
        assert value["requested_workers"] == workers
        assert 1 <= value["max_active_workers"] <= value["workers"] <= workers
        if value["workers"] > 1:
            assert value["parallel_maps"] == 1 and value["parallel_tasks"] >= 2
        fit = model.refine(
            p,
            3,
            60,
            None,
            iterative=True,
            cached_normal=True,
            reuse_normal=True,
            blocked_normal=True,
            workers=workers,
        )
        original = model.refine(
            p,
            3,
            60,
            None,
            iterative=True,
            cached_normal=True,
            reuse_normal=True,
            blocked_normal=True,
        )
        np.testing.assert_array_equal(
            original["parameters"].view(np.uint64), fit["parameters"].view(np.uint64)
        )
        same(original["evaluation"], fit["evaluation"])
        assert original["termination"] == fit["termination"]
    if not weighted:
        meta = model.metadata()
        q = hkl @ np.linalg.inv(cell.geometry().direct_basis)
        factors, _ = xray_non_resonant(
            [s["species"] for s in meta["sites"]], np.linalg.norm(q, axis=1) / 2
        )

        def reference(params):
            return symmetric_model_reference(
                meta, cell.geometry().direct_basis, hkl, obs, factors, params
            )

        ref = reference(p)
        for key in [
            "residual",
            "objective",
            "r1",
            "amplitude_scale",
            "fractional_xyz",
            "u_cartesian",
        ]:
            np.testing.assert_allclose(serial[key], ref[key], rtol=5e-11, atol=2e-12)
        jac = serial["jacobian"].reshape(len(p), len(hkl))
        for axis in [0, len(p) - 1]:
            delta = np.zeros(len(p))
            delta[axis] = 2e-6
            fd = (reference(p + delta)["residual"] - reference(p - delta)["residual"]) / 4e-6
            np.testing.assert_allclose(jac[axis], fd, rtol=3e-5, atol=2e-8)


def test_independent_concurrent_calls_restore_context_and_invalid_workers():
    model, *_ = case()
    p = model.initial_parameters(0.015)
    expected = model.evaluate(p)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: model.evaluate(p, workers=2), range(6)))
    for value in values:
        same(expected, value)
    same(expected, model.evaluate(p))
    for invalid in [0, 65]:
        with pytest.raises(ValueError, match="workers"):
            model.evaluate(p, workers=invalid)


@pytest.mark.parametrize("workers", [0, 65, True, 1.5, "2"])
def test_public_worker_validation(workers):
    from phasesmith.experimental.solution import solve_structure_elements

    with pytest.raises(ValueError, match="workers"):
        solve_structure_elements(None, cell=None, elements=["Si"], workers=workers)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_full_search_retains_counts_coordinates_and_phasing_prefixes(symbol):
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
    common.update(late_phase_plateau_stopping=True, model_consensus_stopping=True)
    original = solve_structure_elements(data, **common)
    assert original.target_met
    for workers in [2, 4, 8]:
        value = solve_structure_elements(data, **common, workers=workers)
        assert value.target_met
        assert (
            dict(value.composition)
            == dict(original.composition)
            == {e: int(np.count_nonzero(species == e)) for e in set(species)}
        )
        np.testing.assert_allclose(
            original.best.fractional_xyz, value.best.fractional_xyz, rtol=0, atol=2e-7
        )
        assert 1 <= value.shared_search.max_active_workers <= value.shared_search.workers <= workers
        assert value.shared_search.requested_workers == workers
        for a, b in zip(original.histories, value.histories, strict=False):
            n = min(len(a), len(b))
            np.testing.assert_array_equal(a[:n].view(np.uint64), b[:n].view(np.uint64))


def test_small_evaluations_do_not_submit_parallel_tasks():
    from test_solution_symmetry_adp import CASES, model_case

    symbol, cell, sites, _ = CASES[0]
    model, *_ = model_case(symbol, cell, sites, mode="isotropic")
    p = model.initial_parameters(0.015)
    value = model.evaluate(p, workers=8)
    same(model.evaluate(p), value)
    assert value["parallel_maps"] == value["parallel_tasks"] == 0
    assert value["max_active_workers"] == 1
