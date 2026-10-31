"""Complete symmetry-constrained tangent cycles and actual placement regressions."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.single_crystal.reference.solution_recycling_reference import recycle_reference
from phasesmith.single_crystal.solution import SolutionOptions, solve_structure_elements
from test_solution_recycling import inputs

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PreparedTangent"), reason="requires experimental-solution"
)


def coupled_inputs(symbol):
    shape, group, rho, h, _, support = inputs(symbol)
    points, weights = [], []
    for point, weight in (([0.137, 0.259, 0.381], 14), ([0.431, 0.027, 0.823], 8)):
        orbit = []
        for op in group.operations:
            image = (op.rotation @ point + np.array(op.translation, float)) % 1
            if not any(np.linalg.norm((image - other + 0.5) % 1 - 0.5) < 1e-8 for other in orbit):
                orbit.append(image)
        points.extend(orbit)
        weights.extend([weight] * len(orbit))
    amplitude = np.abs(np.exp(2j * np.pi * (h @ np.array(points).T)) @ weights)
    intensity = amplitude**2
    cell = UnitCell(8, 8, 10, 90, 90, 90)
    plan = _core._PreparedTangent(
        list(cell.as_tuple()), group._native, h.ravel(), intensity, 0.001 + 0.01 * intensity
    )
    rho = rho - rho.mean() + 0.01
    return shape, group, rho, h, amplitude, support, plan


def native(args, *, cycles=3, hand=1, shift=(0, 0, 0), seed=47, max_seconds=10.0, plan=None):
    shape, group, rho, h, a, support, prepared = args
    return _core._solution_recycle_density(
        rho.ravel(),
        shape,
        group._native,
        h.ravel(),
        a,
        support.ravel(),
        hand,
        shift,
        cycles,
        max_seconds,
        None,
        prepared if plan is None else plan,
        seed,
    )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm", "P 21 m a", "I4/mcm"])
@pytest.mark.parametrize("hand,shift", [(1, (0.0, 0.0, 0.0)), (-1, (0.137, 0.271, 0.419))])
@pytest.mark.parametrize("cycles", [0, 3])
def test_complete_coupled_cycles_against_independent_numpy(symbol, hand, shift, cycles):
    args = coupled_inputs(symbol)
    shape, group, rho, h, a, support, plan = args
    assert plan.metadata()["audit"]["triplets"] > 0
    result = native(args, cycles=cycles, hand=hand, shift=shift)
    steps = []
    expected, fractions = recycle_reference(
        rho,
        h,
        a,
        support,
        [op.rotation for op in group.operations],
        [np.array(op.translation, float) for op in group.operations],
        hand=hand,
        shift=shift,
        cycles=cycles,
        tangent=plan.metadata(),
        seed=47,
        tangent_steps=steps,
    )
    density = result["density"].reshape(shape)
    np.testing.assert_allclose(density, expected, rtol=5e-9, atol=2e-11)
    np.testing.assert_allclose(result["negative_fractions"], fractions, atol=1 / np.prod(shape))
    audit = result["tangent"]
    assert audit["steps"] == audit["annealed_steps"] == cycles
    assert audit["proposals"] == sum(s["proposals"] for s in steps)
    assert audit["changed"] == sum(s["changed"] for s in steps)
    assert audit["declined_steps"] == sum(s["blend"] == 0 for s in steps)
    assert audit["symmetry_restorations"] == sum(s["symmetry_restored"] for s in steps)
    assert audit["symmetry_declined_steps"] == sum(s["symmetry_declined"] for s in steps)
    for key in ("quartet_before", "quartet_after"):
        np.testing.assert_allclose(audit[key], [s[key] for s in steps], atol=2e-12, rtol=2e-12)
    assert np.all(np.array(audit["quartet_after"]) <= np.array(audit["quartet_before"]) + 1e-12)
    f = np.prod(shape) * np.fft.ifftn(density)
    np.testing.assert_allclose(np.abs(f[tuple((h % shape).T)]), a, atol=2e-10, rtol=1e-11)
    assert np.max(np.abs(f[support == 0])) < 2e-10
    assert result["predicted_coefficients"] == np.count_nonzero(support == 1)
    for op in group.operations:
        np.testing.assert_allclose(
            f[tuple(((h @ op.rotation) % shape).T)],
            f[tuple((h % shape).T)] * np.exp(-2j * np.pi * (h @ np.array(op.translation, float))),
            atol=2e-10,
            rtol=1e-11,
        )
    replay = native(args, cycles=cycles, hand=hand, shift=shift)
    np.testing.assert_array_equal(result["density"], replay["density"])
    assert result["tangent"] == replay["tangent"]


def test_coupled_projection_has_actual_quartets_updates_and_truthful_stop():
    args = coupled_inputs("P 21 m a")
    result = native(args, cycles=4)
    audit = result["tangent"]
    assert audit["quartets"] > 0
    assert audit["changed"] > 0 and audit["symmetry_restorations"] > 0
    stopped = native(args, max_seconds=0.0)
    assert stopped["cycles"] == stopped["tangent"]["steps"] == 0
    assert not stopped["tangent"]["quartet_after"]
    _shape, group, _, h, _, _, _ = args
    other = _core._PreparedTangent(
        [8.0, 8.0, 10.0, 90.0, 90.0, 90.0],
        group._native,
        h[2:-2].ravel(),
        np.ones(len(h) - 4),
        np.ones(len(h) - 4),
    )
    with pytest.raises(ValueError, match="plan does not match"):
        native(args, plan=other)


@pytest.mark.parametrize("placement", [False, True])
def test_public_placement_preserves_inferred_counts_and_actual_work(placement):
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic("Pm")
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["O", "Fe"],
        space_group=group,
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
        maximum_atoms=32,
        symmetry_recycling=True,
        tangent_refinement=True,
        tangent_in_recycling=placement,
    )
    assert result.target_met and result.best.r1 < 1e-6
    assert dict(result.composition) == {"Fe": 2, "O": 3}
    audit = result.shared_search
    assert audit.tangent_in_recycling is placement
    if placement:
        assert not audit.tangent_trials
        attempts = [r.tangent for r in audit.recycling if r.tangent is not None]
        assert attempts and any(t.changed > 0 for t in attempts)
    else:
        assert audit.tangent_trials and any(t.changed > 0 for t in audit.tangent_trials)
        assert all(r.tangent is None for r in audit.recycling)


def test_quartet_guard_rejects_after_symmetry_restoration():
    import itertools

    from phasesmith.io.space_groups import space_group_by_symbol

    shape = (32, 32, 32)
    group = space_group_by_symbol("P 21 m a").space_group
    h = np.array(
        [p for p in itertools.product(range(-7, 8), repeat=3) if 0 < sum(v * v for v in p) <= 49],
        dtype=np.int64,
    )
    h = h[~group.systematic_absences(h)]
    points = np.array(
        [
            (op.rotation @ p + np.array(op.translation, float)) % 1
            for p in np.random.default_rng(629).uniform(0, 1, (4, 3))
            for op in group.operations
        ]
    )
    weights = np.repeat([14.0, 8.0, 6.0, 6.0], len(group.operations))
    amplitude = np.abs(np.exp(2j * np.pi * (h @ points.T)) @ weights)
    support = np.zeros(shape, np.uint8)
    support[0, 0, 0] = 1
    support[tuple((h % shape).T)] = 2
    plan = _core._PreparedTangent(
        [8.0, 8.0, 10.0, 90.0, 90.0, 90.0],
        group._native,
        h.ravel(),
        amplitude**2,
        0.001 + 0.01 * amplitude**2,
    )
    rho = np.random.default_rng(1).normal(0, 1, shape) + 0.001
    result = native((shape, group, rho, h, amplitude, support, plan), cycles=10)
    assert result["tangent"]["quartets"] > 0
    assert result["tangent"]["symmetry_declined_steps"] > 0
    steps = []
    expected, _ = recycle_reference(
        rho,
        h,
        amplitude,
        support,
        [op.rotation for op in group.operations],
        [np.array(op.translation, float) for op in group.operations],
        cycles=10,
        tangent=plan.metadata(),
        seed=47,
        tangent_steps=steps,
    )
    np.testing.assert_allclose(result["density"].reshape(shape), expected, rtol=5e-9, atol=2e-11)
    assert (
        sum(s["symmetry_declined"] for s in steps) == result["tangent"]["symmetry_declined_steps"]
    )
    for step in steps:
        if step["symmetry_declined"]:
            assert step["changed"] == 0 and step["quartet_before"] == step["quartet_after"]


def test_zero_amplitude_signed_zeros_do_not_count_as_phase_changes():
    shape, group, rho, h, amplitude, support, _ = coupled_inputs("P-1")
    amplitude = amplitude.copy()
    amplitude[np.sum(h, axis=1) % 3 == 0] = 0.0
    plan = _core._PreparedTangent(
        [8.0, 8.0, 10.0, 90.0, 90.0, 90.0],
        group._native,
        h.ravel(),
        amplitude**2,
        0.001 + 0.01 * amplitude**2,
    )
    result = native((shape, group, rho, h, amplitude, support, plan), cycles=3)
    assert result["tangent"]["proposals"] > 0
    assert result["tangent"]["changed"] == 0
