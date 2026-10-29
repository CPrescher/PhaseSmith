"""Independent Wilson/epsilon equations and CC/Rweak analytic derivatives."""

import itertools

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
from phasesmith.experimental.solution_quality_reference import (
    phase_quality_reference,
    reciprocal_s_squared,
    wilson_reference,
)
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_PhaseQualityPlan"), reason="requires experimental-solution"
)


def case(symbol="P1"):
    if symbol == "P 31 2 1":
        cell = UnitCell(8, 8, 10, 90, 90, 120)
    elif symbol in ("P4", "Fm-3m"):
        cell = UnitCell(8, 8, 8, 90, 90, 90)
    else:
        cell = UnitCell(7, 8, 9, 90, 103, 90)
    group = space_group_by_symbol(symbol).space_group
    rotations = np.array([op.rotation for op in group.operations], dtype=np.int64)
    h = np.array([v for v in itertools.product(range(-4, 5), repeat=3) if any(v)], dtype=np.int64)
    x = reciprocal_s_squared(cell.as_tuple(), h)
    factors, _ = xray_non_resonant(["Fe", "O"], np.sqrt(x))
    c = (factors * factors).sum(axis=1)
    eps = np.sum(np.all(np.einsum("hi,gij->hgj", h, rotations) == h[:, None], axis=-1), axis=1)
    rng = np.random.default_rng(426)
    family_values = {}
    for row in h:
        orbit = row @ rotations
        key = min(map(tuple, np.concatenate([orbit, -orbit])))
        if key not in family_values:
            family_values[key] = rng.exponential() - 0.05
    first_allowed = h[~group.systematic_absences(h)][0]
    orbit = first_allowed @ rotations
    family_values[min(map(tuple, np.concatenate([orbit, -orbit])))] = -0.05
    fluctuation = []
    for row in h:
        orbit = row @ rotations
        fluctuation.append(family_values[min(map(tuple, np.concatenate([orbit, -orbit])))])
    intensity = eps * c * np.exp(2.3 - 9 * x) * fluctuation
    intensity[group.systematic_absences(h)] = 0.0
    sigma = np.maximum(np.abs(intensity) * 0.03, 0.1)
    return cell, group, rotations, h, intensity, sigma


def native_plan(cell, group, h, intensity, sigma, *, wilson=True):
    return _core._PhaseQualityPlan(
        list(cell.as_tuple()), group._native, h.ravel(), intensity, sigma, ["Fe", "O"], wilson
    )


@pytest.mark.parametrize("symbol", ["P1", "P-1", "P21", "P4", "Fm-3m", "P 31 2 1"])
def test_normalization_matches_independent_shell_equations_and_unit_mean(symbol):
    cell, group, rotations, h, intensity, sigma = case(symbol)
    plan = native_plan(cell, group, h, intensity, sigma)
    m = plan.metadata()
    family_intensity = {}
    for row, value in zip(h, intensity, strict=True):
        orbit = row @ rotations
        key = min(map(tuple, np.concatenate([orbit, -orbit])))
        family_intensity.setdefault(key, []).append(value)
    merged = []
    for row in m["hkl"]:
        orbit = row @ rotations
        key = min(map(tuple, np.concatenate([orbit, -orbit])))
        merged.append(np.mean(family_intensity[key]))
    merged = np.array(merged)
    x = reciprocal_s_squared(cell.as_tuple(), m["hkl"])
    f, _ = xray_non_resonant(["Fe", "O"], np.sqrt(x))
    ref = wilson_reference(m["hkl"], merged, cell.as_tuple(), rotations, (f * f).sum(axis=1))
    for key in ("expected_intensity", "e_values"):
        np.testing.assert_allclose(m[key], ref[key], rtol=3e-11, atol=1e-12)
    np.testing.assert_array_equal(m["epsilon"], ref["epsilon"])
    np.testing.assert_array_equal(m["representatives"], ref["representatives"])
    assert m["normalization"]["mean_e_squared"] == pytest.approx(1, abs=2e-14)
    assert m["normalization"]["nonpositive_families"] > 0
    assert m["normalization"]["slope"] == pytest.approx(ref["slope"], rel=2e-11, abs=1e-11)
    np.testing.assert_allclose(
        [s["reduced_mean"] for s in m["normalization"]["shells"]], ref["shell_means"], rtol=2e-11
    )
    assert np.all(m["e_values"][merged < 0] == 0)
    # Metadata is detached; changing it cannot modify the prepared evaluator.
    m["e_values"][:] = 99
    assert not np.any(plan.metadata()["e_values"] == 99)


@pytest.mark.parametrize("wilson", [False, True])
def test_phase_quality_values_gradients_and_scale_invariance(wilson):
    cell, group, _, h, intensity, sigma = case("P21")
    plan = native_plan(cell, group, h, intensity, sigma, wilson=wilson)
    m = plan.metadata()
    assert len(m["quality_hkl"]) > m["allowed"]  # measured absences retained
    assert all(m["weak"][m["allowed"] :])
    rng = np.random.default_rng(428)
    c = rng.uniform(0.2, 5, len(m["quality_hkl"]))
    args = (m["target"], c, m["e_weights"], m["weak"], m["allowed"])
    ref = phase_quality_reference(*args)
    actual = plan.evaluate(c)
    for key in ("correlation", "r_weak", "cfom", "gradient"):
        np.testing.assert_allclose(actual[key], ref[key], rtol=3e-12, atol=3e-13)
    for i in rng.choice(len(c), 8, replace=False):
        step = 1e-5
        plus, minus = c.copy(), c.copy()
        plus[i] += step
        minus[i] -= step
        finite = (plan.evaluate(plus)["cfom"] - plan.evaluate(minus)["cfom"]) / (2 * step)
        assert actual["gradient"][i] == pytest.approx(finite, rel=3e-7, abs=3e-10)
    assert c @ actual["gradient"] == pytest.approx(0, abs=3e-12)
    scaled = plan.evaluate(c * 713)
    assert scaled["cfom"] == pytest.approx(actual["cfom"], abs=3e-14)
    np.testing.assert_allclose(scaled["gradient"] * 713, actual["gradient"], rtol=2e-12, atol=3e-14)
    for wrong in [np.zeros_like(c), -c, np.full_like(c, np.nan), c[:-1]]:
        with pytest.raises(ValueError):
            plan.evaluate(wrong)


@pytest.mark.parametrize("factor", [1e-40, 3.7, 1e40])
def test_intensity_unit_change_preserves_e_and_quality(factor):
    cell, group, _, h, intensity, sigma = case()
    a = native_plan(cell, group, h, intensity, sigma)
    b = native_plan(cell, group, h, intensity * factor, sigma * factor)
    ma, mb = a.metadata(), b.metadata()
    np.testing.assert_allclose(ma["e_values"], mb["e_values"], rtol=3e-12, atol=2e-14)
    np.testing.assert_allclose(
        ma["expected_intensity"] * factor, mb["expected_intensity"], rtol=3e-12
    )
    c = np.linspace(0.1, 3, len(ma["quality_hkl"]))
    assert a.evaluate(c)["cfom"] == pytest.approx(b.evaluate(c)["cfom"], abs=3e-13)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_normalized_quality_search_still_infers_synthetic_counts(symbol):
    from test_solution_shared_search import synthetic

    cell, group, data, _, species = synthetic(symbol)
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["Fe", "O"],
        space_group=group,
        maximum_atoms=32,
        wilson_normalization=True,
        phase_quality_selection=True,
        symmetry_recycling=True,
        tangent_refinement=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=800,
            max_seconds=20,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert len(result.best.species) == len(species)
    assert dict(result.composition) == {
        e: int(np.count_nonzero(species == e)) for e in set(species)
    }
    a = result.shared_search
    assert a.normalization is not None and a.normalization_diagnostic is None
    assert a.phase_quality and all(q.cfom for q in a.phase_quality)
    assert all(np.isfinite(q.cfom).all() for q in a.phase_quality)
    assert result.to_record()["shared_search"]["normalization"]["families"] == result.family_count


def test_public_quality_flags_are_boolean_and_phasing_is_explicit():
    from test_solution_shared_search import synthetic

    cell, group, data, _, _ = synthetic()
    for kwargs, error in [
        ({"wilson_normalization": 1}, TypeError),
        ({"phase_quality_selection": 1}, TypeError),
        ({"phase_quality_selection": True}, ValueError),
    ]:
        with pytest.raises(error):
            solve_structure_elements(
                data, cell=cell, elements=["Fe", "O"], space_group=group, **kwargs
            )


def test_single_resolution_shell_is_merged_without_arbitrary_steps():
    cell = UnitCell(5, 5, 5, 90, 90, 90)
    group = space_group_by_symbol("P1").space_group
    h = np.eye(3, dtype=np.int64)
    intensity = np.array([2.0, 4.0, -1.0])
    plan = native_plan(cell, group, h, intensity, np.ones(3))
    m = plan.metadata()
    assert len(m["normalization"]["shells"]) == 1
    assert m["normalization"]["slope"] == 0
    assert m["normalization"]["mean_e_squared"] == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(m["expected_intensity"], m["expected_intensity"][0], rtol=2e-14)


def test_normalization_failure_is_explicit_and_signed_sources_are_unchanged():
    from phasesmith.io.hkl import HklData

    cell, group, _, h, _, sigma = case()
    # Both radial shells have negative signed means, but a positive family remains
    # so the ordinary legacy phasing boundary itself is valid.
    intensity = np.full(len(h), -100.0)
    intensity[0] = 1.0
    intensity[-1] = 1.0
    with pytest.raises(ValueError, match=r"positive.*shell"):
        native_plan(cell, group, h, intensity, sigma)
    data = HklData(h, intensity, sigma)
    result = solve_structure_elements(
        data,
        cell=cell,
        elements=["Fe", "O"],
        space_group=group,
        maximum_atoms=8,
        wilson_normalization=True,
        phase_quality_selection=True,
        options=SolutionOptions(
            phasing="dual_space",
            trials=1,
            cycles=4,
            max_seconds=0.2,
            r1_target=0.001,
            completion=False,
        ),
    )
    assert result.shared_search.normalization is None
    assert "positive" in result.shared_search.normalization_diagnostic
    assert not result.shared_search.phase_quality
    np.testing.assert_array_equal(result.observations.intensity, intensity)


@pytest.mark.parametrize("elements", [[], ["Fe", "Fe"], ["H"], ["Fe3+"], ["Xx"]])
def test_normalization_rejects_non_type_element_lists(elements):
    cell, group, _, h, intensity, sigma = case()
    with pytest.raises(ValueError):
        _core._PhaseQualityPlan(
            list(cell.as_tuple()), group._native, h.ravel(), intensity, sigma, elements
        )
