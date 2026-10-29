"""Independent graph discovery, gradients and complete annealed reciprocal steps."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
from phasesmith.experimental.solution_tangent_reference import (
    invariant_reference,
    normalize_reference,
    small_relations_reference,
    step_reference,
)
from phasesmith.io.space_groups import space_group_by_symbol

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PreparedTangent"), reason="requires experimental-solution"
)


def observations():
    strong = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
        (2, 0, 0),
        (3, 0, 0),
        (0, 3, 0),
        (3, 3, 0),
    ]
    weak = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 2, 1), (1, 2, 2), (4, 1, 2)]
    rows = {
        tuple(sign * np.array(h)): 100.0 if h in strong else 0.0
        for h in strong + weak
        for sign in (-1, 1)
    }
    hkl = np.array(sorted(rows), dtype=np.int64)
    intensity = np.array([rows[tuple(h)] for h in hkl])
    return hkl, intensity, np.full(len(hkl), 0.01)


def plan(arrays=None, symbol="P1"):
    hkl, intensity, sigma = observations() if arrays is None else arrays
    cell = UnitCell(8, 9, 10, 90, 90, 90)
    group = space_group_by_symbol(symbol).space_group
    native = _core._PreparedTangent(
        list(cell.as_tuple()), group._native, hkl.ravel(), intensity, sigma
    )
    return native, cell, group


def quartet_codes(terms):
    codes = tuple(sorted(sign * (index + 1) for index, sign in terms))
    return min(codes, tuple(sorted(-code for code in codes)))


def test_measured_relation_discovery_against_brute_force():
    native, cell, group = plan()
    metadata = native.metadata()
    expected = normalize_reference(
        *observations(), cell.geometry().reciprocal_basis, [op.rotation for op in group.operations]
    )
    for key, array in zip(
        ("hkl", "normalized", "upper", "strong", "shell_centers", "shell_means"),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(metadata[key], array, rtol=2e-13, atol=1e-14)
    triples, quartets = small_relations_reference(*expected[:4])
    got_t = sorted(
        (target, tuple(sorted(sign * (index + 1) for index, sign in terms)), weight)
        for target, terms, weight in metadata["triplets"]
    )
    assert [(t[0], t[1]) for t in got_t] == [(t[0], t[1]) for t in sorted(triples)]
    np.testing.assert_allclose([t[2] for t in got_t], [t[2] for t in sorted(triples)], atol=1e-13)
    got_q = sorted((quartet_codes(terms), weight) for terms, weight in metadata["quartets"])
    assert [q[0] for q in got_q] == [q[0] for q in sorted(quartets)]
    np.testing.assert_allclose([q[1] for q in got_q], [q[1] for q in sorted(quartets)], atol=1e-13)
    assert metadata["triplets"] and metadata["quartets"]
    assert not metadata["audit"]["truncated"]


@pytest.mark.parametrize("seed", [0, 71, 2**64 - 1])
def test_fused_values_and_gradients(seed):
    native, _, _ = plan()
    metadata = native.metadata()
    phases = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(metadata["hkl"]))
    got, expected = native.evaluate(phases), invariant_reference(metadata, phases)
    for key in ("triplet", "negative_quartet"):
        assert abs(got[key] - expected[key]) < 3e-13
    for key, reference in (
        ("triplet_gradient", "triplet_gradient"),
        ("quartet_gradient", "negative_quartet_gradient"),
    ):
        np.testing.assert_allclose(got[key], expected[reference], rtol=3e-12, atol=3e-13)
    for index in range(len(phases)):
        plus, minus = phases.copy(), phases.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        for key, gradient in (
            ("triplet", "triplet_gradient"),
            ("negative_quartet", "quartet_gradient"),
        ):
            finite = (native.evaluate(plus)[key] - native.evaluate(minus)[key]) / 2e-6
            assert abs(finite - got[gradient][index]) < 2e-9
    shifted = phases + 2 * np.pi * (np.asarray(metadata["hkl"]) @ [0.173, 0.287, 0.391])
    for other in (shifted, -phases):
        for key in ("triplet", "negative_quartet"):
            assert abs(native.evaluate(other)[key] - got[key]) < 2e-12


@pytest.mark.parametrize("cycle", [0, 5, 11, 12, 40])
@pytest.mark.parametrize("seed", [0, 913])
def test_complete_steps_against_independent_numpy(cycle, seed):
    native, _, _ = plan()
    metadata = native.metadata()
    rng = np.random.default_rng(63)
    phases = rng.uniform(-np.pi, np.pi, len(metadata["hkl"]))
    magnitudes = rng.uniform(0.1, 2, len(phases))
    result = native.step(phases, magnitudes, cycle, seed)
    expected = step_reference(metadata, phases, magnitudes, cycle, seed)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], rtol=3e-12, atol=3e-13)
    assert result["quartet_after"] <= result["quartet_before"] + 1e-12
    np.testing.assert_array_equal(
        result["phases"], native.step(phases, magnitudes, cycle, seed)["phases"]
    )
    assert (result["temperature"] > 0) == (cycle < 12)


def test_quartets_break_trivial_phase_consistency_and_require_measured_weak_crosses():
    native, _, _ = plan()
    metadata = native.metadata()
    zero = np.zeros(len(metadata["hkl"]))
    assert native.evaluate(zero)["triplet"] == 0
    assert abs(native.evaluate(zero)["negative_quartet"] - 2) < 1e-14
    magnitude = np.ones(len(zero))
    magnitude[np.all(np.asarray(metadata["hkl"]) == [1, 0, 0], axis=1)] = 0.01
    assert native.step(zero, magnitude, 0, 471)["quartet_after"] < 2 - 1e-5
    h, intensity, sigma = observations()
    original = {quartet_codes(q[0]) for q in metadata["quartets"]}

    # Compare signed HKL quartets, since deleting a row renumbers independent variables.
    def physical(meta):
        return {
            tuple(sorted(tuple(sign * meta["hkl"][index]) for index, sign in terms))
            for terms, _ in meta["quartets"]
        }

    keep = ~np.all(np.abs(h) == [1, 1, 0], axis=1)
    missing = plan((h[keep], intensity[keep], sigma[keep]))[0].metadata()
    inflated = sigma.copy()
    inflated[~keep] = 1000
    uncertain = plan((h, intensity, inflated))[0].metadata()
    assert original
    assert len(physical(missing)) < len(physical(metadata))
    assert len(physical(uncertain)) < len(physical(metadata))


def test_empty_relation_classes_and_input_boundary():
    h = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.int64)
    native, _, _ = plan((h, np.ones(2), np.ones(2) * 0.01))
    metadata = native.metadata()
    assert not metadata["triplets"] and not metadata["quartets"]
    assert native.evaluate(np.array([0.3]))["negative_quartet"] == 0
    np.testing.assert_array_equal(native.step(np.array([0.3]), np.ones(1), 0, 0)["phases"], [0.3])
    for arrays in (
        (h[:1], np.ones(1), np.ones(1)),
        (h[::-1].copy(), np.ones(2), np.ones(2)),
        (h, np.ones(2), np.zeros(2)),
        (h, np.zeros(2), np.ones(2)),
        (h, np.array([1.0, 2.0]), np.ones(2)),
    ):
        with pytest.raises(ValueError):
            plan(arrays)
    for phases in (np.array([]), np.array([np.nan])):
        with pytest.raises(ValueError):
            native.evaluate(phases)
    for magnitude in (np.array([-1.0]), np.array([np.inf])):
        with pytest.raises(ValueError):
            native.step(np.array([0.0]), magnitude, 0, 0)
    # Metadata arrays are detached copies and cannot corrupt the prepared graph.
    metadata["hkl"][:] = 0
    assert np.any(native.metadata()["hkl"])


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_tangent_enabled_still_infers_synthetic_counts(symbol):
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic(symbol)
    options = SolutionOptions(
        phasing="dual_space",
        trials=1,
        cycles=800,
        max_seconds=20,
        r1_target=0.005,
        completion=False,
    )
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["O", "Fe"],
        space_group=group,
        options=options,
        maximum_atoms=32,
        symmetry_recycling=True,
        tangent_refinement=True,
    )
    assert result.target_met
    assert (
        dict(result.composition)
        == {"P1": {"Fe": 1, "O": 2}, "P-1": {"Fe": 2, "O": 2}, "Pm": {"Fe": 2, "O": 3}}[symbol]
    )
    assert result.best.r1 < 1e-6
    audit = result.shared_search
    assert audit.tangent_refinement and audit.tangent_in_recycling
    assert not audit.tangent_trials
    attempts = [r.tangent for r in audit.recycling if r.tangent is not None]
    assert attempts
    assert all(t.triplets > 0 and t.steps > 0 for t in attempts)
    assert all(len(t.quartet_after) == t.steps for t in attempts)


def test_public_tangent_controls_are_explicit():
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic()
    for kwargs, error in (
        ({"tangent_refinement": 1}, TypeError),
        ({"tangent_in_recycling": 1}, TypeError),
        ({"tangent_refinement": True}, ValueError),
    ):
        with pytest.raises(error):
            solve_structure_elements(
                data, cell=cell, elements=["Fe", "O"], space_group=group, **kwargs
            )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm", "P 21 m a", "I4/mcm"])
def test_resolution_shells_and_symmetry_epsilon_against_numpy(symbol):
    cell = UnitCell(8, 8, 10, 90, 90, 90)
    group = space_group_by_symbol(symbol).space_group
    h = np.array(
        [p for p in itertools.product(range(-4, 5), repeat=3) if p != (0, 0, 0)], dtype=np.int64
    )
    h = h[~group.systematic_absences(h)]
    images = []
    for point in ([0.137, 0.259, 0.381], [0.431, 0.027, 0.823]):
        orbit = []
        for op in group.operations:
            image = (op.rotation @ point + np.array(op.translation, float)) % 1
            if not any(np.linalg.norm((image - other + 0.5) % 1 - 0.5) < 1e-8 for other in orbit):
                orbit.append(image)
        images.extend(orbit)
    q = np.sum((h @ cell.geometry().reciprocal_basis.T) ** 2, axis=1)
    intensity = np.abs(np.exp(2j * np.pi * (h @ np.array(images).T)).sum(axis=1)) ** 2 * np.exp(-q)
    sigma = 0.01 + 0.01 * intensity
    native = _core._PreparedTangent(
        list(cell.as_tuple()), group._native, h.ravel(), intensity, sigma
    )
    metadata = native.metadata()
    expected = normalize_reference(
        h,
        intensity,
        sigma,
        cell.geometry().reciprocal_basis,
        [op.rotation for op in group.operations],
    )
    assert len(metadata["shell_centers"]) > 1
    for index, key in ((1, "normalized"), (2, "upper"), (4, "shell_centers"), (5, "shell_means")):
        np.testing.assert_allclose(metadata[key], expected[index], rtol=2e-12, atol=2e-13)
    for scale in (1e-12, 1e12):
        scaled = _core._PreparedTangent(
            list(cell.as_tuple()), group._native, h.ravel(), intensity * scale, sigma * scale
        ).metadata()
        for key in ("normalized", "upper"):
            np.testing.assert_allclose(scaled[key], metadata[key], rtol=2e-12, atol=2e-13)


def test_coincident_shell_centers_merge_instead_of_arbitrary_steps():
    h = np.array(
        [
            p
            for p in itertools.product(range(-18, 19), repeat=3)
            if sum(value * value for value in p) == 325
        ],
        dtype=np.int64,
    )
    canonical = {tuple(p) if tuple(p) > tuple(-p) else tuple(-p) for p in h}
    assert len(canonical) >= 64
    rng = np.random.default_rng(634)
    value = dict(zip(sorted(canonical), rng.uniform(0.5, 2.0, len(canonical)), strict=True))
    intensity = np.array([value[tuple(p) if tuple(p) > tuple(-p) else tuple(-p)] for p in h])
    sigma = np.full(len(h), 0.01)
    cell = UnitCell(8, 8, 8, 90, 90, 90)
    group = space_group_by_symbol("P1").space_group
    native = _core._PreparedTangent(
        list(cell.as_tuple()), group._native, h.ravel(), intensity, sigma
    )
    metadata = native.metadata()
    assert len(metadata["shell_centers"]) == 1
    expected = normalize_reference(
        h,
        intensity,
        sigma,
        cell.geometry().reciprocal_basis,
        [op.rotation for op in group.operations],
    )
    np.testing.assert_allclose(metadata["normalized"], expected[1], rtol=2e-12, atol=2e-13)
    assert abs(np.mean(np.asarray(metadata["normalized"]) ** 2) - 1) < 1e-12


def test_tangent_request_keeps_truthful_budget_and_cancellation_results():
    from phasesmith.experimental.solution import SolutionCancellation
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic()
    token = SolutionCancellation()
    token.cancel()
    for options, cancellation in (
        (SolutionOptions(phasing="dual_space", max_seconds=10), token),
        (SolutionOptions(phasing="dual_space", max_seconds=1e-12), None),
    ):
        result = solve_structure_elements(
            data,
            cell=cell,
            elements=["O", "Fe"],
            space_group=group,
            options=options,
            tangent_refinement=True,
            cancellation=cancellation,
        )
        assert not result.target_met
        assert result.shared_search.tangent_refinement
        assert not result.shared_search.tangent_trials
        assert result.termination == ("cancelled" if cancellation else "max_runtime")
