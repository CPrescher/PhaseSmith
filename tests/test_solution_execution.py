"""Scalar merit equivalence, reused Fourier trajectories, and serial audit barriers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_quality_reference import phase_quality_reference

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PhaseQualityPlan"), reason="experimental-solution"
)


@pytest.mark.parametrize("symbol", ["P1", "P21", "Fm-3m", "P 31 2 1"])
@pytest.mark.parametrize("shells", [False, True])
def test_score_only_batch_is_bitwise_equal_to_fused_values(symbol, shells):
    from test_solution_phase_quality import case, native_plan

    cell, group, _, h, intensity, sigma = case(symbol)
    plan = native_plan(cell, group, h, intensity, sigma, calculated_shells=shells)
    m = plan.metadata()
    rng = np.random.default_rng(1034)
    batches = rng.uniform(0.05, 3, (8, len(m["target"])))
    batches[0] = 1  # Explicit undefined flat correlation convention.
    if shells and max(m["calculated_shell"]) > 0:
        batches[1, np.asarray(m["calculated_shell"]) == 0] = 0
    batches[2] *= 1e-150
    batches[3] *= 1e150
    values = plan.evaluate_values(batches.ravel(), len(batches))
    for row, value in zip(batches, values, strict=True):
        full = plan.evaluate(row)
        for key in ("correlation", "r_weak", "cfom"):
            assert np.float64(value[key]).view(np.uint64) == np.float64(full[key]).view(np.uint64)
        assert value["correlation_defined"] == full["correlation_defined"]
        ref = phase_quality_reference(
            m["target"],
            row / row.max(),
            m["e_weights"],
            m["weak"],
            m["allowed"],
            m["calculated_shell"],
        )
        for key in ("correlation", "r_weak", "cfom"):
            assert value[key] == pytest.approx(ref[key], rel=3e-12, abs=3e-12)


@pytest.mark.parametrize("kind", ["zero", "negative", "nan", "infinity", "length"])
def test_score_only_rejects_the_same_invalid_amplitudes(kind):
    from test_solution_phase_quality import case, native_plan

    cell, group, _, h, intensity, sigma = case()
    plan = native_plan(cell, group, h, intensity, sigma)
    values = np.ones(len(plan.metadata()["target"]))
    if kind == "length":
        values = values[:-1]
    elif kind == "zero":
        values[:] = 0
    else:
        values[0] = {"negative": -1, "nan": np.nan, "infinity": np.inf}[kind]
    with pytest.raises(ValueError):
        plan.evaluate_values(values)
    with pytest.raises(ValueError):
        plan.evaluate(values)


@pytest.mark.parametrize("shape", [(16, 16, 16), (16, 24, 8)])
@pytest.mark.parametrize("rank,omit", [(None, 0.0), (None, 0.1), (0, 0.1)])
def test_reused_workspaces_preserve_complete_fourier_trajectory(shape, rank, omit):
    from phasesmith.experimental.solution_dual_reference import dual_space_reference
    from test_solution_dual_space import inputs

    _, cell, h, amplitude, phases, support = inputs()
    active = np.argwhere(support == 1)
    keep = np.all(2 * np.abs(h) < shape, axis=1)
    h, amplitude, phases = h[keep], amplitude[keep], phases[keep]
    support = np.zeros(shape, dtype=np.uint8)
    support[tuple((h % shape).T)] = 2
    support[0, 0, 0] = 1
    # Preserve an actual unmeasured pair in both layouts.
    support[1, 0, 0] = support[-1, 0, 0] = 1
    assert len(active) > 0
    args = (
        shape,
        list(cell.as_tuple()),
        h.ravel(),
        amplitude,
        phases,
        support.ravel(),
        40,
        3,
        0.25,
        3.0,
        omit,
        987,
        rank,
    )
    old = _core._solution_dual_space(*args)
    new = _core._solution_dual_space(*args, reusable_workspaces=True)
    for key in ("density", "history"):
        np.testing.assert_array_equal(
            np.asarray(new[key]).view(np.uint64), np.asarray(old[key]).view(np.uint64)
        )
    assert new["best_cycle"] == old["best_cycle"]
    reference_cycles = 1 if rank is not None and omit > 0 else 40
    ref = dual_space_reference(
        shape,
        cell.geometry().direct_basis,
        h,
        amplitude,
        phases,
        support,
        cycles=reference_cycles,
        atom_count=3,
        sigma=0.25,
        omit_fraction=omit,
        seed=987,
        patterson_rank=rank,
    )
    np.testing.assert_allclose(
        new["history"][:reference_cycles], ref["history"], rtol=3e-11, atol=3e-12
    )
    if reference_cycles == 40:
        np.testing.assert_allclose(
            new["density"].reshape(shape), ref["density"], rtol=3e-10, atol=3e-12
        )


def test_interleaving_rotates_case_positions_and_saves_every_mode_before_audit():
    path = Path(__file__).parents[1] / "benchmarks/solution_execution_interleaved.py"
    spec = importlib.util.spec_from_file_location("execution_interleaved", path)
    module = importlib.util.module_from_spec(spec)
    # Corpus package may not yet have been imported by another test module.
    import sys

    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    events, audits = [], []
    modes = module.MODES

    def stream(label):
        for i in range(4):
            for method in ("phasesmith", "shelxt"):
                yield {"id": str(i), "method": method}
        yield {"reference_audit_barrier": True}
        assert sum(e.get("reference_audit_barrier", False) for _, e in events) == 4
        audits.append(label)

    streams = {mode: stream(mode) for mode in modes}
    assert module.collect_balanced(streams, lambda mode, event: events.append((mode, event))) == 8
    assert audits == list(modes)
    for i in range(4):
        order = [
            mode for mode, e in events if e.get("id") == str(i) and e.get("method") == "phasesmith"
        ]
        assert order == list(modes[i:] + modes[:i])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_all_execution_modes_preserve_models_counts_and_cycle_histories(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, xyz, species = synthetic(symbol)
    results = []
    original = (data.intensity.copy(), data.sigma.copy(), data.hkl.copy())
    for score_only, workspaces in [(False, False), (True, False), (False, True), (True, True)]:
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
            score_only_evaluation=score_only,
            reusable_fourier_workspaces=workspaces,
            options=SolutionOptions(
                phasing="dual_space",
                trials=4,
                cycles=800,
                max_seconds=60,
                r1_target=0.005,
                completion=False,
            ),
        )
        assert result.target_met
        assert dict(result.composition) == {
            e: int(np.count_nonzero(species == e)) for e in set(species)
        }
        assert len(result.best.species) == len(xyz)
        audit = result.shared_search
        cycles = sum(len(q.cfom) for q in audit.phase_quality)
        assert audit.score_only_evaluations == (cycles if score_only else 0)
        assert audit.fourier_workspace_cycles == (cycles if workspaces else 0)
        results.append(result)
    old = results[0]
    for result in results[1:]:
        assert result.best.species == old.best.species
        np.testing.assert_array_equal(
            result.best.fractional_xyz.view(np.uint64), old.best.fractional_xyz.view(np.uint64)
        )
        assert result.best.r1 == old.best.r1
        assert len(result.histories) == len(old.histories)
        for a, b in zip(result.histories, old.histories, strict=True):
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    np.testing.assert_array_equal(data.intensity, original[0])
    np.testing.assert_array_equal(data.sigma, original[1])
    np.testing.assert_array_equal(data.hkl, original[2])
    # Coordinate checks use the synthetic model only after every prediction exists.
    best = np.inf
    for hand in (-1, 1):
        for anchor in xyz[species == old.best.species[0]]:
            current = (
                hand * old.best.fractional_xyz + anchor - hand * old.best.fractional_xyz[0]
            ) % 1
            maximum = 0.0
            for element in set(species):
                delta = (
                    xyz[species == element, None]
                    - current[None, np.asarray(old.best.species) == element]
                    + 0.5
                ) % 1 - 0.5
                distance = np.linalg.norm(delta @ cell.geometry().direct_basis.T, axis=-1)
                maximum = max(maximum, distance.min(axis=0).max(), distance.min(axis=1).max())
            best = min(best, maximum)
    assert best < 1e-5


@pytest.mark.parametrize("flag", ["score_only_evaluation", "reusable_fourier_workspaces"])
def test_execution_flag_dependencies_are_explicit(flag):
    from phasesmith.experimental.solution import solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic("P1")
    kwargs = dict(cell=cell, space_group=group, elements=["Fe", "O"])
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **kwargs, **{flag: 1})
    with pytest.raises(ValueError, match="requires"):
        solve_structure_elements(data, **kwargs, **{flag: True})
