"""Training-only proposal fitting and model competition controls."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.grains import (
    GrainConditioningOptions,
    GrainOwnershipOptions,
    GrainProposalRefinementOptions,
    GrainRecoveryOptions,
    GrainSearchOptions,
    _finish,
    _polish_proposals,
    find_grains_ensemble,
)
from phasesmith.experimental.ownership import _select, observation_charts
from phasesmith.experimental.prediction import fit_crystal_proposal, refine_crystal
from phasesmith.experimental.reciprocal import reciprocal_vectors
from phasesmith.experimental.recovery import finish_training_pool
from test_experimental_grains import positional_cloud


def test_unvalidated_forward_fit_matches_validated_optimizer_coefficients():
    q, observed, geometry, truth = positional_cloud()
    training = np.arange(len(q)) % 5 != 0
    h = np.rint(q @ np.linalg.inv(truth)).astype(np.int64)
    initial = truth + np.random.default_rng(414).normal(0, 1e-4, (3, 3))
    validated = refine_crystal(observed, h, initial, geometry, training=training)
    proposal = fit_crystal_proposal(observed[training], h[training], initial, geometry)
    np.testing.assert_array_equal(proposal.reciprocal_basis, validated.reciprocal_basis)
    np.testing.assert_array_equal(proposal.history, validated.history)
    assert not hasattr(proposal, "metrics")
    assert not proposal.reciprocal_basis.flags.writeable
    with pytest.raises(ValueError, match="held-out"):
        refine_crystal(
            observed[training],
            h[training],
            initial,
            geometry,
            training=np.ones(training.sum(), bool),
        )


def test_training_pruning_resolves_identical_competitors_without_validation():
    q, observed, geometry, truth = positional_cloud()
    a = np.linalg.inv(truth).T
    o = GrainSearchOptions(workers=2)
    own = GrainOwnershipOptions()
    charts, valid = observation_charts(observed, geometry, own)
    selected, labels, audit = finish_training_pool(
        _core.ExperimentalLatticeEngine(2),
        q,
        observed,
        geometry,
        [a, a.copy()],
        charts,
        valid,
        o,
        own,
    )
    assert len(selected) == 1 and len(audit["removed"]) == 1
    assert audit["removed"][0]["owned"] == 0
    assert (labels > 0).sum() >= 20
    assert audit["forward_fits"][0]["status"] == "completed"


def test_competitive_contribution_replaces_union_veto_but_not_final_support():
    rng = np.random.default_rng(623)
    child = np.diag([4.1, 5.3, 7.7])
    parent = np.diag([2.0, 1.0, 1.0]) @ child
    hc = np.unique(rng.integers(-4, 5, (24, 3)), axis=0)
    hc = hc[np.any(hc != 0, axis=1)]
    hp = np.unique(rng.integers(-4, 5, (90, 3)), axis=0)
    hp[:, 0] = 2 * hp[:, 0] + 1
    q = np.r_[hc @ np.linalg.inv(child).T, hp @ np.linalg.inv(parent).T]
    charts = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    valid = np.ones(len(q), bool)
    engine = _core.ExperimentalLatticeEngine(2)
    o = GrainSearchOptions(max_grains=4)
    legacy, _ = _select(engine, q, [parent, child], charts, valid, o, GrainOwnershipOptions())
    competitive, _ = _select(
        engine,
        q,
        [parent, child],
        charts,
        valid,
        o,
        GrainOwnershipOptions(competitive_selection=True),
    )
    assert len(legacy) == 1 and len(competitive) == 2
    # Exact shared positions remain ambiguous: a proposal does not establish
    # independently owned training support or bypass any final gate.
    train = np.arange(len(q)) % 5 != 0
    result = _finish(
        q,
        train,
        competitive,
        np.zeros(train.sum(), np.int32),
        np.empty((0, 3)),
        np.empty(0),
        [],
        o,
        ownership=GrainOwnershipOptions(),
        charts=charts,
        chart_valid=valid,
    )
    assert not result.candidates[1].accepted


@pytest.mark.parametrize(
    "invalid", [dict(max_pairs=0), dict(split_trials=True), dict(max_split_parents=9)]
)
def test_recovery_options_validate(invalid):
    with pytest.raises(ValueError):
        GrainRecoveryOptions(**invalid)


def test_refinement_budget_reserves_distinct_candidates():
    q, observed, geometry, truth = positional_cloud()
    a = np.linalg.inv(truth).T
    models = [a, a.copy(), a.copy()]
    _, audit = _polish_proposals(
        _core.ExperimentalLatticeEngine(2),
        q,
        observed,
        geometry,
        models,
        GrainSearchOptions(),
        GrainProposalRefinementOptions(max_candidates=2, distinct_budget=True),
        None,
    )
    assert sum(r["status"] == "duplicate_fit_budget" for r in audit["candidates"]) == 2


def test_recovery_pool_reserved_independence_and_workers():
    q, observed, geometry, _ = positional_cloud()
    train = np.arange(len(q)) % 5 != 0
    o = GrainSearchOptions(grid_size=128, max_length=15, max_grains=3, workers=1)
    kwargs = dict(
        training=train,
        searches=(o,),
        geometry=geometry,
        conditioning=GrainConditioningOptions(max_seeds=4),
        ownership=GrainOwnershipOptions(max_fits=4, competitive_selection=True),
        proposal_refinement=GrainProposalRefinementOptions(max_candidates=4, distinct_budget=True),
        recovery=GrainRecoveryOptions(max_pairs=1, split_trials=2, max_split_parents=1),
    )
    first = find_grains_ensemble(q, observed=observed, options=o, **kwargs)
    observed[~train] += [41, -19, 0.15]
    q[~train] = reciprocal_vectors(observed[~train], geometry)
    second = find_grains_ensemble(q, observed=observed, options=replace(o, workers=2), **kwargs)
    np.testing.assert_array_equal(first.labels[train], second.labels[train])
    for a, b in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
    x = next(r["training_recovery"] for r in first.diagnostics if "training_recovery" in r)
    y = next(r["training_recovery"] for r in second.diagnostics if "training_recovery" in r)
    assert x == y


def test_recovery_detector_uniform_null():
    _, _, geometry, _ = positional_cloud()
    observed = np.random.default_rng(771).uniform([200, 200, -200], [1800, 1800, -30], (500, 3))
    q = reciprocal_vectors(observed, geometry)
    o = GrainSearchOptions(grid_size=128, max_length=15, max_grains=3, workers=2)
    result = find_grains_ensemble(
        q,
        observed=observed,
        geometry=geometry,
        options=o,
        searches=(o,),
        conditioning=GrainConditioningOptions(max_seeds=4),
        ownership=GrainOwnershipOptions(max_fits=4, competitive_selection=True),
        proposal_refinement=GrainProposalRefinementOptions(max_candidates=4, distinct_budget=True),
        recovery=GrainRecoveryOptions(max_pairs=2, split_trials=16, max_split_parents=1),
    )
    assert not any(c.accepted for c in result.candidates)


def test_supported_subcell_is_a_cell_alternative_not_an_extra_component():
    from phasesmith.experimental.recovery import _subcell_alternatives

    rng = np.random.default_rng(621)
    parent = np.diag([10.5, 5.3, 7.7])
    child = np.diag([2.1, 5.3, 7.7])
    h = np.unique(rng.integers(-7, 8, (900, 3)), axis=0)
    h = h[np.any(h != 0, axis=1)]
    q = h @ np.linalg.inv(parent).T + rng.normal(0, 1e-5, h.shape)
    charts = np.broadcast_to(np.eye(3) * 20, (len(q), 3, 3)).copy()
    valid = np.ones(len(q), bool)
    aliases = _subcell_alternatives(
        _core.ExperimentalLatticeEngine(2),
        q,
        [parent, child],
        charts,
        valid,
        GrainSearchOptions(),
        GrainOwnershipOptions(),
    )
    assert len(aliases) == 1
    assert aliases[0]["child"] == 1 and aliases[0]["parent"] == 0 and aliases[0]["index"] == 5
    assert aliases[0]["child_exclusive"] == 0
    # A distinct orientation does not become the same component by volume ratio.
    angle = np.radians(17)
    rotation = np.array(
        [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    other = child @ rotation
    qo = np.r_[q, h @ np.linalg.inv(other).T]
    cc = np.broadcast_to(np.eye(3) * 20, (len(qo), 3, 3)).copy()
    assert not _subcell_alternatives(
        _core.ExperimentalLatticeEngine(2),
        qo,
        [parent, other],
        cc,
        np.ones(len(qo), bool),
        GrainSearchOptions(),
        GrainOwnershipOptions(),
    )
