"""Ordered spawned searches with one numerical worker and unchanged evidence."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal import grains as G
from phasesmith.single_crystal import search_execution as execution

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


def limits(monkeypatch):
    for name in execution._LIMITS:
        monkeypatch.setenv(name, "1")


def fixture():
    rng = np.random.default_rng(730284)
    a = np.diag([4.7, 6.3, 8.1]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    h = rng.integers(-4, 5, (300, 3))
    q = h @ np.linalg.inv(a).T + rng.normal(0, 0.0005, h.shape)
    return np.concatenate([q, rng.uniform(-1, 1, (151, 3))])


def compare(a, b):
    np.testing.assert_array_equal(a.labels.view(np.uint8), b.labels.view(np.uint8))
    np.testing.assert_array_equal(
        a.translation_vectors.view(np.uint8), b.translation_vectors.view(np.uint8)
    )
    np.testing.assert_array_equal(
        a.translation_scores.view(np.uint8), b.translation_scores.view(np.uint8)
    )
    assert len(a.candidates) == len(b.candidates)
    for x, y in zip(a.candidates, b.candidates, strict=True):
        for name in x.__dataclass_fields__:
            xx, yy = getattr(x, name), getattr(y, name)
            if isinstance(xx, np.ndarray):
                assert xx.tobytes(order="C") == yy.tobytes(order="C")
            else:
                assert xx == yy


def test_parallel_budget_and_environment_are_explicit_and_serial_remains_available(monkeypatch):
    for name in execution._LIMITS:
        monkeypatch.delenv(name, raising=False)
    with execution.parallel_searches(1) as session:
        assert session is None and execution.active() is None
    for count in [True, 0, -1, 1.5, 17]:
        with (
            pytest.raises(ValueError, match="ensemble workers"),
            execution.parallel_searches(count),
        ):
            pass
    with pytest.raises(ValueError, match="before starting Python"), execution.parallel_searches(2):
        pass
    assert execution.active() is None


def test_parallel_geometric_conditioned_and_stability_proposals_keep_all_bits(monkeypatch):
    limits(monkeypatch)
    q = fixture()
    options = G.GrainSearchOptions(
        grid_size=64,
        max_length=10,
        fft_peaks=48,
        basis_vectors=16,
        max_pairs=12,
        cell_refinements=8,
        max_grains=4,
        workers=2,
        native_fourier_refinement=True,
    )
    conditioning = G.GrainConditioningOptions(
        max_seeds=4, subset_search=replace(options, min_indexed=12, max_grains=1)
    )
    kwargs = dict(
        options=options,
        searches=(options, replace(options, reciprocal_tolerance=0.003)),
        conditioning=conditioning,
        stability=G.GrainStabilityOptions(resamples=1),
    )
    serial = G.find_grains_ensemble(q, **kwargs)
    parallel = G.find_grains_ensemble(q, **kwargs, ensemble_workers=2)
    compare(serial, parallel)
    assert parallel.options.workers == 1
    audit = next(d["parallel_search"] for d in parallel.diagnostics if "parallel_search" in d)
    assert audit["max_processes"] == 2 and audit["tasks"] > 2
    assert all(job["native_workers"] == 1 for job in audit["jobs"])
    assert all(set(job["numerical_environment"].values()) == {"1"} for job in audit["jobs"])
    assert execution.active() is None


def test_paired_recovery_merge_is_independent_of_process_completion_order(monkeypatch):
    from phasesmith.single_crystal.recovery import paired_proposals

    limits(monkeypatch)
    q = fixture()
    options = G.GrainSearchOptions(
        grid_size=128, max_length=15, workers=1, native_fourier_refinement=True
    )
    engine = _core.ExperimentalLatticeEngine(1)
    vectors = G._vectors(engine, q, options, diverse=True)[0]
    recovery = G.GrainRecoveryOptions(max_pairs=2)
    serial, audit = paired_proposals(engine, q, vectors, options, recovery)
    with execution.parallel_searches(2):
        parallel, other = paired_proposals(engine, q, vectors, options, recovery)
    assert audit == other
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def test_worker_exception_resets_parent_session_and_nested_budgets_cannot_expand(monkeypatch):
    limits(monkeypatch)
    with (
        pytest.raises(ValueError, match="unknown positional search task"),
        execution.parallel_searches(2) as session,
    ):
        with execution.parallel_searches(1) as same:
            assert same is session
        with (
            pytest.raises(ValueError, match="nested search sessions"),
            execution.parallel_searches(3),
        ):
            pass
        list(session.map("invalid", [()]))
    assert execution.active() is None


def test_serial_conditioning_callback_can_cancel_before_first_subset_search(monkeypatch):
    events = []

    def field(*args):
        events.append("work")
        raise AssertionError("callback should cancel before numerical field work")

    def cancel(*args):
        events.append("progress")
        raise RuntimeError("cancelled")

    monkeypatch.setattr(G, "_conditioned_seed", field)
    with pytest.raises(RuntimeError, match="cancelled"):
        G._conditioned_proposals(
            None,
            fixture(),
            G.GrainSearchOptions(max_length=10),
            G.GrainConditioningOptions(max_seeds=3),
            cancel,
            prepared=(np.eye(3), np.ones(3), {}),
        )
    assert events == ["progress"]


def assert_fits_equal(first, second):
    models, audit = first
    other, record = second
    assert audit == record
    assert len(models) == len(other)
    for a, b in zip(models, other, strict=True):
        assert a.tobytes() == b.tobytes()


def test_parallel_training_fits_trims_and_overlapping_alias_entries_are_exact(monkeypatch):
    from phasesmith.single_crystal import ownership as W
    from phasesmith.single_crystal import recovery as R
    from test_experimental_grains import positional_cloud

    limits(monkeypatch)
    q, obs, geometry, truth = positional_cloud()
    basis = np.linalg.inv(truth).T
    models = [basis.copy(), basis * 1.0003, basis * 0.9997]
    original = [a.copy() for a in models]
    engine = _core.ExperimentalLatticeEngine(1)
    options = G.GrainSearchOptions(workers=1)
    polish = G.GrainProposalRefinementOptions(trim_fractions=(0.9, 0.8, 0.7))
    own = G.GrainOwnershipOptions(alias_fit_cycles=2)
    # The second entry intentionally revisits cells updated by the first.
    entries = [
        dict(pool_start=0, cycle=0, audit=dict(proposed=3)),
        dict(pool_start=1, cycle=1, audit=dict(proposed=2)),
    ]
    serial_forward = R.fit_training_proposals(engine, q, obs, geometry, models, options)
    serial_trim = G._polish_proposals(engine, q, obs, geometry, models, options, polish, None)
    serial_alias = [a.copy() for a in models]
    alias_audit = W.polish_ownership_aliases(
        engine, q, obs, geometry, serial_alias, entries, options, own
    )
    assert serial_trim[1]["trimmed_proposed"] > 0
    with execution.parallel_searches(2) as session:
        assert_fits_equal(
            serial_forward, R.fit_training_proposals(engine, q, obs, geometry, models, options)
        )
        assert_fits_equal(
            serial_trim,
            G._polish_proposals(engine, q, obs, geometry, models, options, polish, None),
        )
        parallel_alias = [a.copy() for a in models]
        other_audit = W.polish_ownership_aliases(
            engine, q, obs, geometry, parallel_alias, entries, options, own
        )
        assert_fits_equal((serial_alias, alias_audit), (parallel_alias, other_audit))
        audit = session.statistics()
    assert audit["submitted_tasks"] == audit["tasks"]
    assert all(j["native_workers"] == 1 for j in audit["jobs"])
    assert all(set(j["numerical_environment"].values()) == {"1"} for j in audit["jobs"])
    for a, b in zip(models, original, strict=True):
        assert a.tobytes() == b.tobytes()


def test_parallel_ownership_branch_and_coherent_rng_keep_original_order(monkeypatch):
    from phasesmith.single_crystal import ownership as W
    from phasesmith.single_crystal import recovery as R
    from test_experimental_grains import positional_cloud

    limits(monkeypatch)
    q, obs, geometry, truth = positional_cloud()
    a = np.linalg.inv(truth).T
    # Duplicate and nearby proposals exercise ordered fit admission and competition.
    models = [a.copy(), a.copy(), a * 1.0003]
    engine = _core.ExperimentalLatticeEngine(1)
    options = G.GrainSearchOptions(workers=1)
    own = G.GrainOwnershipOptions(alias_rounds=0, refinement_cycles=2)
    charts, valid = W.observation_charts(obs, geometry, own)
    args = engine, q, obs, geometry, models, charts, valid, options, own, None
    serial, labels, record = W.refine_ownership_pool(*args)
    recovery = G.GrainRecoveryOptions(split_trials=32, max_split_parents=4)
    parents = [*models, a * 0.05]
    coherent = R.coherent_proposals(engine, q, obs, geometry, parents, options, recovery)
    assert any(len(r["training_rows"]) < 17 for r in coherent[1]["records"])
    assert coherent[1]["proposed"] > 0
    with execution.parallel_searches(2):
        parallel, otherlabels, audit = W.refine_ownership_pool(*args)
        assert_fits_equal((serial, record), (parallel, audit))
        assert labels.tobytes() == otherlabels.tobytes()
        assert_fits_equal(
            coherent, R.coherent_proposals(engine, q, obs, geometry, parents, options, recovery)
        )


def test_speculative_failure_is_deferred_until_original_coordinator_consumes_case():
    from phasesmith.single_crystal.fitting_tasks import _Cases

    class Ticket:
        calls = 0

        def result(self):
            self.calls += 1
            return [(4, ("accepted", None)), (7, (None, ValueError("unused fit")))]

    ticket = Ticket()
    cases = _Cases({4: ticket, 7: ticket})
    assert cases.get(4) == "accepted"
    assert cases.get(4) == "accepted"
    assert ticket.calls == 1
    with pytest.raises(ValueError, match="unused fit"):
        cases.get(7)


def test_fit_tasks_can_be_disabled_without_disabling_parallel_searches(monkeypatch):
    from phasesmith.single_crystal import fitting_tasks

    limits(monkeypatch)
    with execution.parallel_searches(2, fit_tasks=False) as session:
        assert execution.active() is session and fitting_tasks.active() is None
        with execution.parallel_searches(2):
            assert fitting_tasks.active() is None
        assert fitting_tasks.batches("forward", (), [(0, ()), (1, ())]) is None
    with pytest.raises(ValueError, match="fit_tasks"), execution.parallel_searches(2, fit_tasks=1):
        pass
