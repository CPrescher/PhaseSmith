"""Count-independent atomicity equations, phasing audits and search contracts."""

import json

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.solution import (
    SolutionCancellation,
    SolutionOptions,
    solve_structure_elements,
    write_solution,
)
from phasesmith.experimental.solution_count_free_reference import (
    adaptive_projection_reference,
    count_free_dual_space_reference,
)
from test_solution_dual_space import inputs
from test_solution_shared_search import synthetic

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "_solution_count_free_dual_space"), reason="requires experimental-solution"
)


def projection(rho, sigma=2.0, cap=128, cycle=0, omit=0.1):
    _, cell, *_ = inputs()
    return _core._solution_adaptive_projection(
        rho.ravel(), rho.shape, list(cell.as_tuple()), sigma, cap, 0.25, omit, 987, cycle
    )


@pytest.mark.parametrize(
    "sigma,cap,cycle", [(2.0, 128, 0), (0.5, 5, 2), (3.0, 32, 53), (10.0, 1, 0)]
)
def test_threshold_projection_against_independent_dense_reference(sigma, cap, cycle):
    shape, cell, *_ = inputs()
    rho = np.random.default_rng(471).normal(0, 0.5, shape)
    rho[0, 0, 0], rho[8, 9, 7] = 6, 7
    result = projection(rho, sigma, cap, cycle)
    ref, peaks, threshold, _ = adaptive_projection_reference(
        rho, cell.geometry().direct_basis, sigma, cap, 0.25, cycle=cycle, seed=987
    )
    np.testing.assert_allclose(
        np.array(result["density"]).reshape(shape), ref, rtol=3e-12, atol=3e-13
    )
    np.testing.assert_allclose(result["peaks"], peaks, rtol=1e-14, atol=2e-15)
    assert result["effective_sigma"] == threshold
    assert len(peaks) <= cap
    scaled = projection(rho * 1e20, sigma, cap, cycle)
    np.testing.assert_allclose(scaled["peaks"], peaks, atol=2e-15)
    np.testing.assert_allclose(scaled["density"], result["density"] * 1e20, rtol=3e-12, atol=1e6)


def test_threshold_is_inclusive_and_never_forces_a_count():
    rho = np.ones((16, 16, 16))
    # Flat plateau has one index-tie-broken maximum, exactly 1 RMS high.
    assert len(projection(rho, 1.0)["peaks"]) == 1
    assert len(projection(rho, np.nextafter(1.0, 2.0))["peaks"]) == 0
    for values in (np.zeros_like(rho), -rho):
        result = projection(values)
        assert len(result["peaks"]) == 0
        assert not np.any(result["density"])
    rho.fill(0)
    rho[::4, ::4, ::4] = 1
    bounded = projection(rho, 0.1, 5)
    assert bounded["effective_sigma"] > 0.1
    # Tied peaks are removed together; cap does not dictate selecting five atoms.
    assert len(bounded["peaks"]) == 0


@pytest.mark.parametrize(
    "sigma,cap", [(0.0, 128), (np.nan, 128), (np.inf, 128), (10.1, 128), (2.0, 0), (2.0, 129)]
)
def test_invalid_native_thresholds(sigma, cap):
    with pytest.raises(ValueError):
        projection(np.zeros((16, 16, 16)), sigma, cap)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_maps_rejected(value):
    rho = np.zeros((16, 16, 16))
    rho.flat[1] = value
    with pytest.raises(ValueError):
        projection(rho)


@pytest.mark.parametrize(
    "m,omit,rank,cycles",
    [(1.0, 0.0, None, 6), (3.0, 0.1, None, 9), (2.5, 0.0, 0, 12), (3.0, 0.0, None, 54)],
)
def test_complete_count_free_cycles_against_numpy(m, omit, rank, cycles):
    shape, cell, hkl, amplitude, phases, support = inputs()

    def native(amp):
        return _core._solution_count_free_dual_space(
            shape,
            list(cell.as_tuple()),
            hkl.ravel(),
            amp,
            phases,
            support.ravel(),
            cycles,
            2.0,
            128,
            0.25,
            m,
            omit,
            987,
            rank,
        )

    result = native(amplitude)
    ref = count_free_dual_space_reference(
        shape,
        cell.geometry().direct_basis,
        hkl,
        amplitude,
        phases,
        support,
        cycles=cycles,
        mask_sigma=0.25,
        hybrid_m=m,
        omit_fraction=omit,
        seed=987,
        patterson_rank=rank,
    )
    np.testing.assert_allclose(
        np.array(result["density"]).reshape(shape), ref["density"], rtol=3e-9, atol=2e-11
    )
    np.testing.assert_allclose(result["history"], ref["history"], rtol=2e-10, atol=3e-12)
    assert result["best_cycle"] == ref["best_cycle"]
    assert result["adaptive_peaks"] == ref["adaptive_peaks"]
    assert len(result["history"]) == len(result["adaptive_peaks"]["selected"])
    np.testing.assert_array_equal(result["density"], native(amplitude)["density"])
    np.testing.assert_allclose(
        result["density"], native(amplitude * 1e20)["density"], rtol=3e-9, atol=2e-11
    )


@pytest.mark.parametrize(
    "symbol,counts",
    [("P1", {"Fe": 1, "O": 2}), ("P-1", {"Fe": 2, "O": 2}), ("Pm", {"Fe": 2, "O": 3})],
)
def test_count_free_search_recovers_synthetic_counts(symbol, counts, tmp_path):
    cell, group, data, _, _ = synthetic(symbol)
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=32,
        options=SolutionOptions(
            phasing="dual_space",
            trials=4,
            cycles=800,
            max_seconds=15,
            r1_target=0.005,
            completion=False,
        ),
    )
    assert result.target_met
    assert dict(result.composition) == counts
    assert result.best.r1 < 1e-6
    trial = result.phasing_trials[0]
    assert trial.method == "dual_space" and trial.atomicity_mode == "adaptive"
    assert trial.initialization == "patterson"
    audit = trial.adaptive_peaks
    assert len(audit.selected) == trial.completed_cycles <= 800
    assert trial.checkpoints[-1].accepted
    assert trial.checkpoints[-1].completed_cycles == trial.completed_cycles
    assert all(0 <= n <= audit.cap for n in audit.selected)
    out = write_solution(result, tmp_path / "result")
    record = json.loads((out / "result.json").read_text())
    assert record["phasing_trials"][0]["adaptive_peaks"]["selected"] == list(audit.selected)


def test_atom_bound_does_not_enter_phasing_and_portfolio_is_explicit():
    cell, group, data, _, _ = synthetic()
    results = [
        solve_structure_elements(
            data,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            maximum_atoms=bound,
            dual_checkpoints=False,
            search_rounds=1,
            options=SolutionOptions(
                phasing="dual_space", trials=1, cycles=12, max_seconds=15, r1_target=0
            ),
        )
        for bound in (2, 32)
    ]
    assert results[0].phasing_trials == results[1].phasing_trials
    np.testing.assert_array_equal(results[0].histories[0], results[1].histories[0])
    result = solve_structure_elements(
        data,
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        search_rounds=1,
        options=SolutionOptions(
            phasing="portfolio", trials=2, cycles=12, max_seconds=15, r1_target=0
        ),
    )
    assert [t.method for t in result.phasing_trials] == ["charge_flip", "dual_space"]
    assert result.phasing_trials[0].adaptive_peaks is None
    assert result.phasing_trials[1].adaptive_peaks is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dual_peak_sigma": 0},
        {"dual_peak_sigma": True},
        {"dual_peak_sigma": np.nan},
        {"dual_peak_cap": 0},
        {"dual_peak_cap": True},
        {"dual_peak_cap": 129},
    ],
)
def test_public_threshold_validation(kwargs):
    cell, group, data, _, _ = synthetic()
    with pytest.raises(ValueError):
        solve_structure_elements(data, cell=cell, space_group=group, elements=["Fe", "O"], **kwargs)


def test_cancelled_and_expired_count_free_search():
    cell, group, data, _, _ = synthetic()
    token = SolutionCancellation()
    token.cancel()
    for seconds, cancel, expected in [(10, token, "cancelled"), (1e-9, None, "max_runtime")]:
        result = solve_structure_elements(
            data,
            cell=cell,
            space_group=group,
            elements=["Fe", "O"],
            cancellation=cancel,
            options=SolutionOptions(phasing="dual_space", max_seconds=seconds),
        )
        assert result.termination == expected
        assert not result.target_met and not result.phasing_trials


def test_declined_checkpoints_continue_identical_fourier_trajectory():
    from phasesmith.experimental.solution_count_free_reference import checkpoint_window_reference

    cell, group, data, _, _ = synthetic("P-1")
    settings = dict(
        cell=cell,
        space_group=group,
        elements=["Fe", "O"],
        maximum_atoms=2,
        search_rounds=1,
        options=SolutionOptions(
            phasing="dual_space", trials=1, cycles=20, r1_target=0, max_seconds=15
        ),
    )
    monitored = solve_structure_elements(data, **settings)
    full = solve_structure_elements(data, **settings, dual_checkpoints=False)
    np.testing.assert_array_equal(monitored.histories[0], full.histories[0])
    assert monitored.phasing_trials[0].adaptive_peaks == full.phasing_trials[0].adaptive_peaks
    assert not monitored.target_met
    assert monitored.shared_search.dual_checkpoints
    assert not full.shared_search.dual_checkpoints
    attempts = monitored.phasing_trials[0].checkpoints
    assert [c.completed_cycles for c in attempts] == [4, 16, 20]
    start = 0
    for c in attempts:
        assert not c.accepted
        assert c.best_cycle == checkpoint_window_reference(
            monitored.histories[0], c.completed_cycles, start
        )
        assert c.score == monitored.histories[0][c.best_cycle]
        start = c.completed_cycles
    assert not full.phasing_trials[0].checkpoints


def test_checkpoint_control_requires_boolean():
    cell, group, data, _, _ = synthetic()
    with pytest.raises(TypeError, match="dual_checkpoints"):
        solve_structure_elements(
            data, cell=cell, space_group=group, elements=["Fe", "O"], dual_checkpoints=1
        )


@pytest.mark.parametrize(
    "accept,rank,cycles", [(4, None, 20), (16, None, 20), (0, None, 54), (4, 0, 20)]
)
def test_monitored_cycles_match_independent_numpy(accept, rank, cycles):
    shape, cell, hkl, amplitude, phases, support = inputs()
    native = _core._solution_count_free_dual_space(
        shape,
        list(cell.as_tuple()),
        hkl.ravel(),
        amplitude,
        phases,
        support.ravel(),
        cycles,
        2.0,
        128,
        0.25,
        3.0,
        0.0,
        987,
        rank,
        accept,
    )
    ref = count_free_dual_space_reference(
        shape,
        cell.geometry().direct_basis,
        hkl,
        amplitude,
        phases,
        support,
        cycles=cycles,
        mask_sigma=0.25,
        omit_fraction=0.0,
        seed=987,
        patterson_rank=rank,
        checkpoint_accept_cycle=accept,
    )
    np.testing.assert_allclose(
        np.array(native["density"]).reshape(shape), ref["density"], rtol=3e-9, atol=2e-11
    )
    np.testing.assert_allclose(native["history"], ref["history"], rtol=2e-10, atol=3e-12)
    assert native["adaptive_peaks"] == ref["adaptive_peaks"]
    assert native["best_cycle"] == ref["best_cycle"]
    assert len(native["checkpoints"]) == len(ref["checkpoints"])
    for a, b in zip(native["checkpoints"], ref["checkpoints"], strict=True):
        for k in ("accepted", "completed_cycles", "best_cycle"):
            assert a[k] == b[k]
        assert abs(a["score"] - b["score"]) < 3e-12
