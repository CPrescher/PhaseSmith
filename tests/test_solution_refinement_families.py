"""Multiplicity-weighted native refinement versus direct independent equations."""

import itertools

import gemmi
import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution_reference import symmetric_model_reference
from phasesmith.io.space_groups import space_group_by_symbol
from phasesmith.scattering_reference import xray_non_resonant
from test_solution_symmetry_adp import CASES

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_SymmetricSolutionModel"), reason="requires experimental-solution"
)


def fixture(symbol, cell, sites, mode, cache=True, variant="complete"):
    ops = gemmi.find_spacegroup_by_name(symbol).operations()
    families = {}
    for h in itertools.product(range(-2, 3), repeat=3):
        if h == (0, 0, 0) or ops.is_systematically_absent(h):
            continue
        orbit = sorted(
            {tuple(sign * v for v in op.apply_to_hkl(h)) for op in ops for sign in [-1, 1]}
        )
        families[orbit[0]] = orbit
    values = {h: float(i % 11 + 1) for i, orbit in enumerate(families.values()) for h in orbit}
    hkl = np.array(sorted(values), dtype=np.int64)
    observed = np.array([values[tuple(h)] for h in hkl])
    if variant == "partial_duplicates":
        indices = np.r_[np.arange(0, len(hkl), 3), [0, 0, 0, 3]]
        np.random.default_rng(63).shuffle(indices)
        hkl, observed = hkl[indices], observed[indices]
    species, points = [], []
    for element, point in sites:
        unique = []
        for op in ops:
            q = np.mod(op.apply_to_xyz(point), 1)
            if not any(np.linalg.norm((q - p + 0.5) % 1 - 0.5) < 1e-8 for p in unique):
                unique.append(q)
        species.extend([element] * len(unique))
        points.extend(unique)
    args = (
        list(cell.as_tuple()),
        space_group_by_symbol(symbol).space_group._native,
        species,
        np.ascontiguousarray(points).ravel(),
        hkl.ravel(),
        observed,
        mode,
        0.01,
        cache,
    )
    return args, hkl, observed


@pytest.mark.parametrize("symbol,cell,sites,_", CASES)
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("cache", [False, True])
def test_weighted_values_derivatives_and_normal_equations(symbol, cell, sites, _, mode, cache):
    args, hkl, observed = fixture(symbol, cell, sites, mode, cache)
    direct = _core._SymmetricSolutionModel(*args)
    weighted = _core._SymmetricSolutionModel(*args, weight_families=True)
    metadata = weighted.metadata()
    reps = np.asarray(metadata["reflection_representatives"])
    counts = np.asarray(metadata["reflection_multiplicities"])
    assert counts.sum() == len(hkl) == metadata["input_reflection_count"]
    assert len(reps) == metadata["evaluated_reflection_count"] < len(hkl)
    h = hkl[reps]
    q = h @ np.linalg.inv(cell.geometry().direct_basis)
    factors, _factor_derivative = xray_non_resonant(
        [s["species"] for s in metadata["sites"]], np.linalg.norm(q, axis=1) / 2
    )
    parameters = np.asarray(weighted.initial_parameters(0.03))
    parameters += np.random.default_rng(73).normal(0, 0.01, len(parameters))
    a, b = direct.evaluate(parameters), weighted.evaluate(parameters)
    ref = symmetric_model_reference(
        metadata,
        cell.geometry().direct_basis,
        h,
        observed[reps],
        factors,
        parameters,
        multiplicities=counts,
    )
    for key in ["objective", "amplitude_scale", "r1", "fractional_xyz", "u_cartesian"]:
        np.testing.assert_allclose(b[key], ref[key], rtol=5e-11, atol=2e-12)
        np.testing.assert_allclose(b[key], a[key], rtol=5e-11, atol=2e-12)
    np.testing.assert_allclose(b["residual"], ref["residual"], rtol=5e-11, atol=2e-12)
    ja = np.asarray(a["jacobian"]).reshape(len(parameters), len(hkl))
    jb = np.asarray(b["jacobian"]).reshape(len(parameters), len(reps))
    np.testing.assert_allclose(jb @ b["residual"], ja @ a["residual"], rtol=5e-10, atol=2e-12)
    np.testing.assert_allclose(jb @ jb.T, ja @ ja.T, rtol=5e-10, atol=2e-12)
    for axis in range(len(parameters)):
        step = np.zeros(len(parameters))
        step[axis] = 2e-6
        plus = symmetric_model_reference(
            metadata,
            cell.geometry().direct_basis,
            h,
            observed[reps],
            factors,
            parameters + step,
            multiplicities=counts,
        )
        minus = symmetric_model_reference(
            metadata,
            cell.geometry().direct_basis,
            h,
            observed[reps],
            factors,
            parameters - step,
            multiplicities=counts,
        )
        np.testing.assert_allclose(
            jb[axis], (plus["residual"] - minus["residual"]) / 4e-6, rtol=3e-5, atol=2e-8
        )
    again = weighted.evaluate(parameters)
    np.testing.assert_array_equal(b["jacobian"], again["jacobian"])


@pytest.mark.parametrize("symbol,cell,sites,_", CASES)
def test_partial_families_duplicates_and_input_order(symbol, cell, sites, _):
    args, hkl, _observed = fixture(symbol, cell, sites, "anisotropic", variant="partial_duplicates")
    a, b = (_core._SymmetricSolutionModel(*args, weight_families=v) for v in [False, True])
    pa, pb = a.initial_parameters(0.02), b.initial_parameters(0.02)
    np.testing.assert_array_equal(pa, pb)
    ma = b.metadata()
    assert sum(ma["reflection_multiplicities"]) == len(hkl)
    ea, eb = a.evaluate(pa), b.evaluate(pb)
    for key in ["r1", "objective", "amplitude_scale"]:
        np.testing.assert_allclose(ea[key], eb[key], rtol=5e-11, atol=2e-12)


def test_nonidentical_amplitudes_are_rejected_and_input_not_changed():
    symbol, cell, sites, _ = CASES[0]
    args, _h, _o = fixture(symbol, cell, sites, "anisotropic")
    args = list(args)
    args[5] = args[5].copy()
    args[5][0] += 0.1
    before = args[5].copy()
    _core._SymmetricSolutionModel(*args)  # The general expanded evaluator remains available.
    with pytest.raises(ValueError, match="identical amplitudes"):
        _core._SymmetricSolutionModel(*args, weight_families=True)
    np.testing.assert_array_equal(args[5], before)


def test_original_row_and_jacobian_limits_apply_before_reduction():
    symbol, cell, _sites, _ = CASES[0]
    sites = [("Si", [0.11 * i + 0.03, 0.17 * i + 0.04, 0.07 * i + 0.05]) for i in range(4)]
    args, _h, _o = fixture(symbol, cell, sites, "anisotropic")
    args = list(args)
    args[4] = np.tile(np.array([1, 0, 0], dtype=np.int64), 70_000)
    args[5] = np.ones(70_000)
    with pytest.raises(ValueError, match="Jacobian memory"):
        _core._SymmetricSolutionModel(*args, weight_families=True)
    args[4] = np.tile(np.array([1, 0, 0], dtype=np.int64), 100_001)
    args[5] = np.ones(100_001)
    with pytest.raises(ValueError, match="observations"):
        _core._SymmetricSolutionModel(*args, weight_families=True)


@pytest.mark.parametrize("symbol", ["P1", "P-1", "Pm"])
def test_complete_search_retains_structure_counts_and_final_mode(symbol):
    from phasesmith.experimental.solution import SolutionOptions, solve_structure_elements
    from test_solution_shared_search import synthetic

    cell, group, data, _xyz, species = synthetic(symbol)
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
    new = solve_structure_elements(data, **common, multiplicity_weighted_refinement=True)
    expected = {e: int(np.count_nonzero(species == e)) for e in set(species)}
    assert old.target_met and new.target_met
    assert dict(old.composition) == dict(new.composition) == expected
    assert new.best.refinement.mode == old.best.refinement.mode == "anisotropic"
    np.testing.assert_allclose(new.best.fractional_xyz, old.best.fractional_xyz, atol=2e-7, rtol=0)
    audit = new.shared_search
    assert audit.weighted_refinement_fits == audit.final_refinements > 0
    assert 0 < audit.weighted_refinement_family_rows < audit.weighted_refinement_input_rows
    assert old.shared_search.weighted_refinement_fits == 0
    with pytest.raises(TypeError, match="boolean"):
        solve_structure_elements(data, **common, multiplicity_weighted_refinement=1)
