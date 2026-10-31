"""Independent geometry derivatives, ownership gates and frozen training scope."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.grains import (
    GrainAliasOptions,
    GrainConditioningOptions,
    GrainOwnershipOptions,
    GrainProposalRefinementOptions,
    GrainSearchOptions,
    GrainStabilityOptions,
    _rotations,
    find_grains_ensemble,
)
from phasesmith.experimental.ownership import (
    assign_observed,
    observation_charts,
    reciprocal_with_jacobian,
)
from phasesmith.experimental.reciprocal import reciprocal_vectors
from test_experimental_grains import positional_cloud

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_reciprocal_jacobian"),
    reason="new experimental-spots build required",
)


def reference_assign(q, models, charts, valid, o, options):
    # Independent exhaustive model distances, using linear solves for HKLs'
    # Cartesian locations and scalar observation coordinates.
    distance = np.full((len(q), len(models)), np.inf)
    for i, a in enumerate(models):
        h = np.rint(q @ a.T)
        residual = q - np.linalg.solve(a, h.T).T
        coordinates = np.einsum("nij,nj->ni", charts, residual)
        keep = (
            valid
            & np.any(h != 0, axis=1)
            & (np.linalg.norm(residual, axis=1) < o.reciprocal_tolerance)
        )
        keep &= (np.linalg.norm(coordinates[:, :2], axis=1) <= options.xy_gate) & (
            abs(coordinates[:, 2]) <= options.omega_gate
        )
        scaled = coordinates / np.array([options.xy_scale, options.xy_scale, options.omega_scale])
        distance[keep, i] = np.sqrt(np.sum(scaled[keep] ** 2, axis=1))
    order = np.argsort(distance, axis=1, kind="stable")
    best = distance[np.arange(len(q)), order[:, 0]]
    inside = np.isfinite(best)
    amb = np.zeros(len(q), bool)
    if len(models) > 1:
        second = distance[np.arange(len(q)), order[:, 1]]
        amb = inside & ((best > o.ambiguity_ratio * second) | ((best == 0) & (second == 0)))
    return (
        np.where(inside & ~amb, order[:, 0] + 1, 0),
        int(amb.sum()),
        int((np.isfinite(distance).sum(axis=1) > 1).sum()),
    )


def test_fused_q_jacobian_finite_differences_and_old_values():
    _, observed, geometry, _ = positional_cloud()
    observed = observed[:35]
    q, j = reciprocal_with_jacobian(observed, geometry)
    np.testing.assert_array_equal(q, reciprocal_vectors(observed, geometry))
    for column, step in enumerate([1e-3, 1e-3, 1e-4]):
        delta = np.zeros_like(observed)
        delta[:, column] = step
        finite = (
            reciprocal_vectors(observed + delta, geometry)
            - reciprocal_vectors(observed - delta, geometry)
        ) / (2 * step)
        np.testing.assert_allclose(j[:, :, column], finite, rtol=2e-6, atol=2e-11)


def test_native_reference_covariant_rotations_and_exact_workers():
    rng = np.random.default_rng(713)
    q = rng.uniform(-1, 1, (600, 3))
    models = [np.diag([4.0, 5.0, 6.0]), np.diag([4.1, 5.2, 6.1])]
    charts = np.broadcast_to(np.diag([20.0, 10.0, 3.0]), (len(q), 3, 3)).copy()
    valid = np.arange(len(q)) % 11 != 0
    o = GrainSearchOptions(reciprocal_tolerance=0.15)
    p = GrainOwnershipOptions()
    expected = reference_assign(q, models, charts, valid, o, p)
    for workers in [1, 2, 2]:
        result = assign_observed(
            _core.ExperimentalLatticeEngine(workers), q, models, charts, valid, o, p
        )
        for a, b in zip(result, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    rotation = _rotations(1, 842)[0]
    rotated = assign_observed(
        _core.ExperimentalLatticeEngine(2),
        q @ rotation,
        [a @ rotation for a in models],
        charts @ rotation,
        valid,
        o,
        p,
    )
    for a, b in zip(rotated, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_eligibility_before_ownership_boundaries_and_origin():
    eng = _core.ExperimentalLatticeEngine(2)
    p = GrainOwnershipOptions()
    o = GrainSearchOptions(reciprocal_tolerance=0.25)
    q = np.array([[1.125, 0, 0], [0, 0, 0.0]])
    charts = np.broadcast_to(np.diag([24.0, 1.0, 1.0]), (2, 3, 3)).copy()
    valid = np.ones(2, bool)
    result = assign_observed(eng, q, [np.eye(3)], charts, valid, o, p)
    np.testing.assert_array_equal(result[0], [1, 0])  # XY exactly3 is eligible.
    charts[0, 0, 0] = np.nextafter(24.0, np.inf)
    assert assign_observed(eng, q, [np.eye(3)], charts, valid, o, p)[0][0] == 0
    charts[0] = np.diag([1.0, 1.0, 1.0])
    charts[0, 2, 0] = 4.0
    assert assign_observed(eng, q, [np.eye(3)], charts, valid, o, p)[0][0] == 1  # omega exactly.5
    charts[0, 2, 0] = np.nextafter(4.0, np.inf)
    assert assign_observed(eng, q, [np.eye(3)], charts, valid, o, p)[0][0] == 0
    charts[:] = np.eye(3)
    o = replace(o, reciprocal_tolerance=0.125)
    assert assign_observed(eng, q, [np.eye(3)], charts, valid, o, p)[0][0] == 0  # q gate strict
    # An invalid observation-space candidate must not block an eligible one.
    q = np.array([[1.01, 1.1, 0.0]])
    models = [np.diag([1.0, 1 / 1.1, 1.0]), np.diag([1 / 1.01, 1.0, 1.0])]
    charts = np.array([np.diag([400.0, 1.0, 1.0])])
    result = assign_observed(
        eng, q, models, charts, np.ones(1, bool), replace(o, reciprocal_tolerance=0.15), p
    )
    assert result[0][0] == 2


@pytest.mark.parametrize(
    "kw",
    [
        dict(xy_scale=0),
        dict(omega_gate=np.nan),
        dict(rank_rtol=0.1),
        dict(max_fits=True),
        dict(alias_rounds=2),
        dict(refinement_cycles=0),
    ],
)
def test_options_validate(kw):
    with pytest.raises(ValueError):
        GrainOwnershipOptions(**kw)


def test_complete_search_reserved_independence_and_worker_repeat():
    q, observed, g, truth = positional_cloud()
    tr = np.arange(len(q)) % 5 != 0
    o = GrainSearchOptions(grid_size=128, max_length=15, max_grains=3, workers=1)
    extra = dict(
        training=tr,
        searches=(o,),
        geometry=g,
        conditioning=GrainConditioningOptions(max_seeds=4),
        stability=GrainStabilityOptions(resamples=0),
        proposal_refinement=GrainProposalRefinementOptions(max_candidates=4),
        aliases=GrainAliasOptions(max_parents=4, max_children=4),
        ownership=GrainOwnershipOptions(max_fits=4),
    )
    first = find_grains_ensemble(q, observed=observed, options=o, **extra)
    assert any(c.accepted for c in first.candidates)
    accepted = [c for c in first.candidates if c.accepted]
    assert any(
        abs(c.volume / abs(np.linalg.det(np.linalg.inv(truth).T)) - 1) < 0.01 for c in accepted
    )
    observed[~tr] += [23, -19, 0.1]
    q[~tr] = reciprocal_vectors(observed[~tr], g)
    for workers in [1, 2]:
        second = find_grains_ensemble(
            q, observed=observed, options=replace(o, workers=workers), **extra
        )
        np.testing.assert_array_equal(first.labels[tr], second.labels[tr])
        for a, b in zip(first.candidates, second.candidates, strict=True):
            np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
        audit1 = next(d["ownership"] for d in first.diagnostics if "ownership" in d)
        audit2 = next(d["ownership"] for d in second.diagnostics if "ownership" in d)
        assert audit1 == audit2


def test_observation_charts_inverse_and_invalid_shapes():
    _, observed, g, _ = positional_cloud()
    chart, valid = observation_charts(observed, g)
    _, j = reciprocal_with_jacobian(observed, g)
    assert valid.all()
    np.testing.assert_allclose(chart @ j, np.broadcast_to(np.eye(3), j.shape), atol=3e-12)
    assert not chart.flags.writeable and not valid.flags.writeable
    eng = _core.ExperimentalLatticeEngine(1)
    with pytest.raises(ValueError, match="charts"):
        eng.assign_observed(
            np.zeros((2, 3)),
            np.eye(3)[None],
            np.eye(3)[None],
            np.zeros((1, 3, 3)),
            np.ones(2, bool),
            np.array([0.015, 0.8, 2, 0.3, 3, 0.5]),
        )


@pytest.mark.parametrize("mixture", [False, True])
def test_normal_reference_and_derivatives(mixture):
    from phasesmith.experimental.grains_reference import observation_normal

    rng = np.random.default_rng(818)
    n = 81
    h = rng.integers(-8, 9, (n, 3)).astype(float)
    b = np.diag([0.17, 0.12, 0.1])
    q = h @ b + rng.normal(0, 0.02, (n, 3))
    charts = rng.normal(size=(n, 3, 3)) * 20
    scales = np.array([2.0, 2.0, 0.3])
    valid = np.arange(n) % 7 != 0
    background = rng.uniform(0.1, 2.0, n) if mixture else None
    eng = _core.ExperimentalLatticeEngine(2)

    def evaluate(basis):
        return eng.observation_normal(q, h, charts, valid, basis, scales, 2.5, background)

    result = evaluate(b)
    expected = observation_normal(q, h, charts, b, scales, valid, background=background)
    for a, v in zip(result, expected, strict=True):
        np.testing.assert_allclose(a, v, rtol=2e-12, atol=2e-10)
    for k in range(9):
        delta = np.zeros(9)
        delta[k] = 1e-7
        finite = (
            evaluate(b + delta.reshape(3, 3))[2] - evaluate(b - delta.reshape(3, 3))[2]
        ) / 2e-7
        np.testing.assert_allclose(result[1][k], finite, rtol=2e-6, atol=2e-5)
    with pytest.raises(ValueError):
        eng.observation_normal(q, h, charts, valid, b, scales, 2.5, np.zeros(n))


def test_integer_family_deferral_requires_independent_training_support():
    from phasesmith.experimental.ownership import _select

    rng = np.random.default_rng(992)
    a = np.diag([4.1, 5.3, 7.7])
    parent = np.diag([2.0, 1.0, 1.0]) @ a
    h = np.unique(rng.integers(-4, 5, (120, 3)), axis=0)
    h = h[np.any(h != 0, axis=1)]
    q = h @ np.linalg.inv(a).T + rng.normal(0, 0.0001, h.shape)
    charts = np.broadcast_to(np.eye(3) * 20, (len(q), 3, 3)).copy()
    eng = _core.ExperimentalLatticeEngine(2)
    options = GrainSearchOptions(max_grains=4)
    own = GrainOwnershipOptions()
    selected, audit = _select(eng, q, [parent, a], charts, np.ones(len(q), bool), options, own)
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0], a)
    assert audit["families"][0]["index"] == 2
    # Actual extra superlattice support must prevent automatic family deferral.
    hp = rng.integers(-4, 5, (60, 3))
    hp[:, 0] = 2 * hp[:, 0] + 1
    extra = hp @ np.linalg.inv(parent).T
    qq = np.r_[q, extra]
    cc = np.broadcast_to(np.eye(3) * 20, (len(qq), 3, 3)).copy()
    _, more = _select(eng, qq, [parent, a], cc, np.ones(len(qq), bool), options, own)
    assert not more["families"]


def test_detector_uniform_cloud_not_accepted():
    _, _, g, _ = positional_cloud()
    rng = np.random.default_rng(1127)
    observed = rng.uniform([200, 200, -200], [1800, 1800, -30], (500, 3))
    q = reciprocal_vectors(observed, g)
    o = GrainSearchOptions(grid_size=128, max_length=15, max_grains=3, workers=2)
    result = find_grains_ensemble(
        q,
        observed=observed,
        geometry=g,
        options=o,
        searches=(o,),
        conditioning=GrainConditioningOptions(max_seeds=4),
        ownership=GrainOwnershipOptions(max_fits=4),
    )
    assert not any(c.accepted for c in result.candidates)


def test_normal_empty_mask_and_overflow_boundary():
    engine = _core.ExperimentalLatticeEngine(1)
    q = np.ones((2, 3))
    charts = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    args = [q, q, charts, np.zeros(2, bool), np.eye(3), np.ones(3), 2.5]
    normal, gradient, cost, n = engine.observation_normal(*args)
    assert n == cost == 0
    np.testing.assert_array_equal(normal, np.zeros((9, 9)))
    np.testing.assert_array_equal(gradient, np.zeros(9))
    args[3][:] = True
    args[2][:] = 1e300
    args[4] *= 2
    with pytest.raises(ValueError, match="nonfinite"):
        engine.observation_normal(*args)
