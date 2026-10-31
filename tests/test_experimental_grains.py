"""Independent native checks and blind multigrain recovery, positions only."""

import json
from itertools import product

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import grains_reference as ref
from phasesmith.experimental.grains import (
    GrainAliasOptions,
    GrainConditioningOptions,
    GrainProposalRefinementOptions,
    GrainSearchOptions,
    GrainStabilityOptions,
    _consolidate,
    ensemble_search_options,
    find_grains,
    find_grains_ensemble,
    integer_lattice_basis,
    prime_congruence_bases,
)

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots feature not built"
)


def positional_cloud():
    from phasesmith.experimental.prediction import predict_spots
    from phasesmith.experimental.reciprocal import PanelGeometry, reciprocal_vectors

    rng = np.random.default_rng(643)
    geometry = PanelGeometry.from_poni(
        distance=0.2,
        poni1=0.08,
        poni2=0.08,
        pixel1=8e-5,
        pixel2=8e-5,
        rot1=0.01,
        rot2=-0.03,
        rot3=0.05,
        wavelength=0.4,
        axis=[-0.97, 0.12, 0.2],
    )
    direct = np.array([[4.7, 0, 0], [0.9, 6.3, 0], [-0.4, 0.2, 8.1]])
    direct = direct @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    b = np.linalg.inv(direct).T
    h = rng.integers(-5, 6, (500, 3))
    prediction = predict_spots(h, b, geometry, rng.uniform(-200, -30, len(h)))
    observed = prediction.values[prediction.flags == 0].copy()
    observed += rng.normal(size=observed.shape) * [0.12, 0.12, 0.025]
    return reciprocal_vectors(observed, geometry), observed, geometry, b


def test_early_polishing_rescues_below_core_cutoff_and_is_bounded():
    from phasesmith.experimental.grains import _polish_proposals, _score

    q, observed, geometry, truth = positional_cloud()
    engine = _core.ExperimentalLatticeEngine(2)
    options = GrainSearchOptions(workers=2)
    start = np.linalg.inv(truth + np.random.default_rng(987).normal(0, 0.002, (3, 3))).T
    assert _score(engine, q, start, options.reciprocal_tolerance / 4)[2].sum() < 20
    proposed, audit = _polish_proposals(
        engine,
        q,
        observed,
        geometry,
        [start, start.copy()],
        options,
        GrainProposalRefinementOptions(max_candidates=1),
        None,
    )
    assert len(proposed) == 1
    assert _score(engine, q, proposed[0], options.reciprocal_tolerance / 4)[2].mean() > 0.98
    # A small narrow-core fit should recover the lattice to sub-percent error;
    # it need not have the precision of fitting every synthetic observation.
    error = np.linalg.norm(np.linalg.inv(proposed[0]).T - truth)
    initial_error = np.linalg.norm(np.linalg.inv(start).T - truth)
    assert error < 0.1 * initial_error
    assert error / np.linalg.norm(truth) < 0.002
    assert audit["candidates"][1]["status"] == "budget_not_selected"
    record = audit["candidates"][0]
    assert len(record["training_rows"]) <= 256
    assert sum(record["internal_training"]) >= 12
    assert len(record["internal_training"]) - sum(record["internal_training"]) >= 5
    assert options.min_indexed == 20
    # Proposal-only eligibility must not bypass the final support contract.
    selected, _, _ = _consolidate(engine, q[:18], proposed, options)
    assert not selected
    _, zero_audit = _polish_proposals(
        engine,
        np.zeros((21, 3)),
        np.zeros((21, 3)),
        geometry,
        [start],
        options,
        GrainProposalRefinementOptions(),
        None,
    )
    assert zero_audit["candidates"][0]["tight_core"] == 0
    with pytest.raises(ValueError, match="8 million"):
        _polish_proposals(
            engine,
            np.zeros((100001, 3)),
            None,
            None,
            [start] * 80,
            options,
            GrainProposalRefinementOptions(),
            None,
        )


def test_exact_prime_congruences_against_independent_residue_enumeration():
    rng = np.random.default_rng(24313)
    for prime, expected in ((2, 7), (3, 13), (5, 31)):
        kernels = prime_congruence_bases((prime,))
        assert len(kernels) == expected
        distinct = set()
        for p, normal, kernel, pivot in kernels:
            assert round(np.linalg.det(kernel)) == p
            residues = ref.modular_kernel_residues(normal, p)
            assert len(residues) == p**2
            distinct.add(tuple(map(tuple, residues)))
            transformed = residues.copy()
            transformed[:, pivot] = residues @ normal // p
            np.testing.assert_array_equal(transformed @ kernel, residues)
            # Unbounded integer coefficients obey the same exact kernel;
            # this is not just a statement about small residue representatives.
            coefficients = rng.integers(-1000, 1000, (80, 3))
            indices = coefficients @ kernel
            assert np.all(indices @ normal % p == 0)
            recovered = indices.copy()
            recovered[:, pivot] = indices @ normal // p
            np.testing.assert_array_equal(recovered, coefficients)
        assert len(distinct) == expected


def test_positional_congruence_recovers_smaller_cell_despite_q_contamination():
    from phasesmith.experimental.grains import _alias_proposals, _score
    from phasesmith.experimental.prediction import predict_spots
    from phasesmith.experimental.reciprocal import reciprocal_vectors

    q, observed, geometry, b = positional_cloud()
    truth = np.linalg.inv(b).T
    parent = np.diag([2, 1, 1]) @ truth
    rng = np.random.default_rng(12413)
    h = rng.integers(-5, 6, (500, 3))
    h[:, 0] = 2 * h[:, 0] + 1
    predicted = predict_spots(h, np.linalg.inv(parent).T, geometry, rng.uniform(-200, -30, len(h)))
    nuisance = predicted.values[predicted.flags == 0][:80].copy()
    nuisance += rng.normal(size=nuisance.shape) * [0.12, 0.12, 0.025]
    points = np.r_[q, reciprocal_vectors(nuisance, geometry)]
    obs = np.r_[observed, nuisance]
    engine = _core.ExperimentalLatticeEngine(2)
    options = GrainSearchOptions(workers=2)
    # The small cell retains less than90% of the broad parent's q support.
    # These positional hypotheses cannot by themselves prove primitivity.
    assert _score(engine, points, truth, 0.015)[2].mean() < 0.9
    before = parent.copy()
    children, audit = _alias_proposals(
        engine, points, obs, geometry, [parent], options, GrainAliasOptions(), None
    )
    np.testing.assert_array_equal(parent, before)
    assert len(children) == 1
    child = children[0]
    transform = child @ np.linalg.inv(truth)
    np.testing.assert_allclose(transform, np.rint(transform), atol=0.001)
    assert abs(round(np.linalg.det(np.rint(transform)))) == 1
    assert abs(np.linalg.det(child) / np.linalg.det(parent)) == pytest.approx(0.5, rel=0.001)
    selected = _score(engine, points, child, 0.00375)[2]
    assert selected[: len(q)].mean() > 0.99
    assert selected[len(q) :].mean() < 0.01
    assert audit["children"][0]["index"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(primes=(4,)),
        dict(primes=[2, 3]),
        dict(primes=(3, 2)),
        dict(max_index=1),
        dict(max_depth=True),
        dict(min_fraction=0.5),
        dict(min_fraction=np.nan),
        dict(min_points=16),
        dict(max_children=129),
        dict(max_parents=0),
    ],
)
def test_alias_option_bounds(kwargs):
    with pytest.raises(ValueError):
        GrainAliasOptions(**kwargs)


def test_stable_polished_pool_independent_of_reserved_observations_and_workers():
    from dataclasses import replace

    q, observed, geometry, _ = positional_cloud()
    training = np.arange(len(q)) % 5 != 0
    options = GrainSearchOptions(grid_size=128, max_length=15, max_grains=2, workers=1)
    kwargs = dict(
        training=training,
        searches=(options,),
        conditioning=GrainConditioningOptions(max_seeds=4),
        stability=GrainStabilityOptions(resamples=1),
        proposal_refinement=GrainProposalRefinementOptions(max_candidates=4),
        aliases=GrainAliasOptions(max_parents=4, max_children=4),
        sparse_conditioning=GrainConditioningOptions(
            max_seeds=4,
            candidates_per_seed=3,
            subset_search=replace(
                options,
                fft_peaks=400,
                basis_vectors=80,
                max_grains=1,
                min_indexed=12,
                reciprocal_tolerance=0.003,
            ),
        ),
        geometry=geometry,
    )
    first = find_grains_ensemble(q, observed=observed, options=options, **kwargs)
    assert any(c.accepted for c in first.candidates)
    q[~training] = np.random.default_rng(963).uniform(-1, 1, ((~training).sum(), 3))
    observed[~training] += [51, -37, 2]
    for workers in (1, 2):
        second = find_grains_ensemble(
            q, observed=observed, options=replace(options, workers=workers), **kwargs
        )
        np.testing.assert_array_equal(first.labels[training], second.labels[training])
        for a, b in zip(first.candidates, second.candidates, strict=True):
            np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
        # Workers are recorded in the proposal options; the actual proposal
        # bases, selected training IDs and optimization histories must match.
        for stage in (
            "stability",
            "sparse_conditioning",
            "proposal_refinement",
            "aliases",
            "consolidation",
        ):
            a = next(d[stage] for d in first.diagnostics if stage in d)
            b = next(d[stage] for d in second.diagnostics if stage in d)
            if stage in ("stability", "sparse_conditioning"):
                for audit in (a, b):
                    for view in audit["views"]:
                        view["seed_search"].pop("workers", None)
                        view["subset_search"].pop("workers", None)
            assert a == b


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        (GrainStabilityOptions, dict(resamples=True)),
        (GrainStabilityOptions, dict(resamples=5)),
        (GrainStabilityOptions, dict(fraction=0.49)),
        (GrainStabilityOptions, dict(fraction=np.nan)),
        (GrainStabilityOptions, dict(proposal_min_indexed=5)),
        (GrainProposalRefinementOptions, dict(max_candidates=257)),
        (GrainProposalRefinementOptions, dict(min_points=16)),
        (GrainProposalRefinementOptions, dict(max_points=17, min_points=18)),
    ],
)
def test_stability_polishing_options_validate(kind, kwargs):
    with pytest.raises(ValueError):
        kind(**kwargs)


def test_stability_uniform_null_and_cancellation():
    q = np.random.default_rng(973).uniform(-1, 1, (500, 3))
    options = GrainSearchOptions(grid_size=128, max_length=15, max_grains=2, workers=2)
    extra = dict(
        searches=(options,),
        conditioning=GrainConditioningOptions(max_seeds=4),
        stability=GrainStabilityOptions(resamples=1),
    )
    result = find_grains_ensemble(q, options=options, **extra)
    assert not any(c.accepted for c in result.candidates)

    def cancel(done, total):
        if done == 2:
            raise InterruptedError("stability cancelled")

    with pytest.raises(InterruptedError, match="stability cancelled"):
        find_grains_ensemble(q, options=options, progress=cancel, **extra)
    with pytest.raises(ValueError, match="requires observed"):
        find_grains_ensemble(
            q, options=options, proposal_refinement=GrainProposalRefinementOptions()
        )


def test_conditioned_search_recovers_sparse_lattice_with_outliers():
    # A weak lattice embedded in a much larger unrelated cloud. No supplied
    # HKLs, basis or group labels enter either search.
    from dataclasses import replace

    rng = np.random.default_rng(3201)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    truth = np.diag([3.7, 5.3, 7.9]) @ rotation
    h = np.unique(rng.integers(-3, 4, (140, 3)), axis=0)
    h = h[np.any(h != 0, axis=1)]
    signal = h @ np.linalg.inv(truth).T + rng.normal(0, 0.0003, h.shape)
    q = np.r_[signal, rng.uniform(-0.9, 0.9, (2500, 3))]
    options = GrainSearchOptions(grid_size=128, max_length=20, max_grains=5, workers=2)
    result = find_grains_ensemble(
        q,
        options=options,
        searches=(replace(options, max_grains=1),),
        conditioning=GrainConditioningOptions(),
    )
    accepted = [c for c in result.candidates if c.accepted]
    assert len(accepted) == 1
    candidate = accepted[0]
    assert candidate.volume == pytest.approx(abs(np.linalg.det(truth)), rel=0.01)
    assert candidate.indexed[: len(signal)].mean() > 0.95
    assert candidate.indexed[len(signal) :].mean() < 0.01
    change = candidate.direct_basis @ np.linalg.inv(truth)
    np.testing.assert_allclose(change, np.rint(change), atol=0.025)
    audit = next(d["conditioning"] for d in result.diagnostics if "conditioning" in d)
    assert audit["proposed"] > 0
    json.dumps(result.diagnostics, allow_nan=False)


def test_conditioned_pool_is_independent_of_reserved_coordinates_and_repeats():
    q, _, _ = mixed_cloud()
    training = np.arange(len(q)) % 5 != 0
    options = GrainSearchOptions(grid_size=128, max_length=15, max_grains=3, workers=2)
    searches = (options,)
    conditioning = GrainConditioningOptions(max_seeds=4)
    first = find_grains_ensemble(
        q, training=training, options=options, searches=searches, conditioning=conditioning
    )
    q[~training] = np.random.default_rng(8839).uniform(-1, 1, ((~training).sum(), 3))
    second = find_grains_ensemble(
        q, training=training, options=options, searches=searches, conditioning=conditioning
    )
    np.testing.assert_array_equal(first.labels[training], second.labels[training])
    assert len(first.candidates) == len(second.candidates)
    for a, b in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
    # Discovery diagnostics precede the reserved-coordinate validation records.
    assert first.diagnostics[:3] == second.diagnostics[:3]


@pytest.mark.parametrize(
    "kw",
    [
        dict(max_seeds=True),
        dict(max_seeds=0),
        dict(max_seeds=81),
        dict(phase_tolerance=0),
        dict(phase_tolerance=np.nan),
        dict(seed_search={}),
        dict(subset_search=object()),
    ],
)
def test_conditioning_options_validate_boundaries(kw):
    with pytest.raises((ValueError, TypeError)):
        GrainConditioningOptions(**kw)


def test_conditioned_uniform_cloud_rejected_and_progress_can_cancel():
    q = np.random.default_rng(973).uniform(-1, 1, (500, 3))
    options = GrainSearchOptions(grid_size=128, max_length=15, max_grains=2, workers=2)
    conditioning = GrainConditioningOptions(max_seeds=8)
    result = find_grains_ensemble(
        q, options=options, searches=(options,), conditioning=conditioning
    )
    assert not any(c.accepted for c in result.candidates)
    calls = []

    def cancel(done, total):
        calls.append((done, total))
        if done == 1:
            raise InterruptedError("conditioned cancellation")

    with pytest.raises(InterruptedError, match="conditioned cancellation"):
        find_grains_ensemble(
            q, options=options, searches=(options,), conditioning=conditioning, progress=cancel
        )
    assert calls[0] == (0, 2) and calls[-1] == (1, 2)


@pytest.mark.parametrize("n", [32, 64])
def test_deposition_fft_against_numpy_and_exact_worker_repeat(n):
    q = np.random.default_rng(n).uniform(-0.65, 0.65, (301, 3))
    old = None
    for workers in (1, 4, 4):
        result = _core.ExperimentalLatticeEngine(workers).build_field(q, n, 0.05).to_numpy()
        np.testing.assert_allclose(result, ref.fourier_field(q, n, 0.05), rtol=3e-5, atol=1e-5)
        assert result[0, 0, 0] == pytest.approx(len(q), abs=3e-5)
        if old is not None:
            np.testing.assert_array_equal(result, old)
        old = result


def test_maxima_boundaries_ties_and_stable_rank():
    n = 16
    values = np.zeros((n, n, n // 2 + 1), dtype=np.float32)
    values[1, 0, 0] = values[-1, 0, 0] = 10
    values[0, 3, 2] = 8
    engine = _core.ExperimentalLatticeEngine(4)
    field = engine.field_from_array(values, 0.1)
    vectors, scores, total = field.maxima(0.5, 3.0, 40)
    freq = np.fft.fftfreq(n, 0.1)
    positive = np.fft.rfftfreq(n, 0.1)
    xyz = np.stack(np.meshgrid(freq, freq, positive, indexing="ij"), axis=-1)
    length = np.linalg.norm(xyz, axis=-1)
    masked = values.copy()
    masked[(length < 0.5) | (length > 3)] = -np.inf
    keep = np.isfinite(masked)
    for dx, dy, dz in product((-1, 0, 1), repeat=3):
        other = np.roll(masked, (dx, dy), axis=(0, 1))
        other = other[:, :, np.clip(np.arange(n // 2 + 1) + dz, 0, n // 2)]
        keep &= masked >= other
    ids = np.flatnonzero(keep)
    order = np.lexsort((ids, -masked.ravel()[ids]))[:40]
    np.testing.assert_array_equal(vectors, xyz.reshape(-1, 3)[ids[order]])
    np.testing.assert_array_equal(scores, masked.ravel()[ids[order]])
    assert total == len(ids)


def test_score_assignment_independent_and_exact_boundaries():
    rng = np.random.default_rng(823)
    models = np.array(
        [np.diag([4.0, 5.0, 6.0]), [[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 6.0]]]
    )
    q = np.r_[rng.uniform(-1, 1, (500, 3)), [[0.25, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]]]
    engine = _core.ExperimentalLatticeEngine(4)
    reciprocal = np.linalg.inv(models).transpose(0, 2, 1)
    for a, b in zip(models, reciprocal, strict=True):
        h, error = engine.score(q, a, b)
        expected = np.rint(q @ a.T)
        np.testing.assert_array_equal(h, expected)
        np.testing.assert_allclose(error, np.linalg.norm(q - expected @ b, axis=1), atol=2e-16)
    actual = engine.assign(q, models, reciprocal, 0.03, 0.8)
    expected = ref.assign(q, models, 0.03, 0.8)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    identity = np.eye(3)[None]
    labels, amb, multi = engine.assign(
        np.array([[1.0, 0, 0], [1.125, 0, 0], [0.0, 0, 0]]), identity, identity, 0.125, 0.8
    )
    np.testing.assert_array_equal(labels, [1, 0, 0])  # strict boundary; exclude 000
    doubled = np.repeat(identity, 2, axis=0)
    labels, amb, multi = engine.assign(np.array([[1.0, 0, 0]]), doubled, doubled, 0.1, 0.8)
    assert labels[0] == 0 and amb == multi == 1


def mixed_cloud(same_cell=False):
    rng = np.random.default_rng(33)
    parts, truths = [], []
    for edges in ([4.3, 5.1, 7.7], [4.3, 5.1, 7.7] if same_cell else [5.2, 6.2, 8.3]):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        a = np.diag(edges) @ rotation
        h = rng.integers(-5, 6, (500, 3))
        h = np.unique(h[np.any(h != 0, axis=1)], axis=0)
        parts.append(h @ np.linalg.inv(a).T + rng.normal(0, 0.0005, h.shape))
        truths.append(a)
    return np.concatenate([*parts, rng.uniform(-1, 1, (100, 3))]), truths, list(map(len, parts))


@pytest.mark.parametrize("same_cell", [False, True])
def test_blind_two_grains_and_same_cell_orientations(same_cell):
    q, truths, sizes = mixed_cloud(same_cell)
    options = GrainSearchOptions(max_length=15, grid_size=128, max_grains=4)
    result = find_grains(q, options=options)
    assert len(result.candidates) == 2 and all(c.accepted for c in result.candidates)
    for a, c in zip(truths, result.candidates, strict=True):
        assert c.volume == pytest.approx(abs(np.linalg.det(a)), rel=0.002)
        change = c.direct_basis @ np.linalg.inv(a)
        np.testing.assert_allclose(change, np.rint(change), atol=0.015)
    for i, block in enumerate(np.split(result.labels[: sum(sizes)], [sizes[0]]), 1):
        assert np.mean(block == i) > 0.97
    json.dumps(result.diagnostics, allow_nan=False)
    assert not result.labels.flags.writeable


def test_heldout_cannot_change_discovery_pool_or_training_ownership():
    q, _, _ = mixed_cloud()
    training = np.arange(len(q)) % 5 != 0
    options = GrainSearchOptions(max_length=15, grid_size=128, max_grains=3)
    first = find_grains(q, training=training, options=options)
    q[~training] = np.random.default_rng(920).uniform(-1, 1, ((~training).sum(), 3))
    second = find_grains(q, training=training, options=options)
    assert len(first.candidates) == len(second.candidates)
    for a, b in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
        np.testing.assert_array_equal(a.hkl[training], b.hkl[training])
        assert a.accepted and not b.accepted
    np.testing.assert_array_equal(first.labels[training], second.labels[training])
    assert first.diagnostics[:2] == second.diagnostics[:2]


def test_random_cloud_rejected_and_cancellation_propagates():
    q = np.random.default_rng(8221).uniform(-1, 1, (350, 3))
    result = find_grains(q, options=GrainSearchOptions(max_length=15, grid_size=128, max_grains=2))
    assert not any(c.accepted for c in result.candidates)

    def cancel(*args):
        raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        find_grains(q, progress=cancel)


def test_integer_generated_lattice_not_an_assumed_cell():
    h = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6], [2, 4, 6]])
    b = integer_lattice_basis(h)
    assert round(abs(np.linalg.det(b))) == 48
    np.testing.assert_allclose(h @ np.linalg.inv(b), np.rint(h @ np.linalg.inv(b)))
    assert integer_lattice_basis(h[:2]) is None
    assert round(abs(np.linalg.det(integer_lattice_basis(np.r_[h, [[1, 1, 1]]])))) == 4


@pytest.mark.parametrize(
    "kw",
    [
        {"workers": 5},
        {"grid_size": 512},
        {"max_grains": 0},
        {"null_rotations": 2},
        {"seed": -1},
        {"grid_size": True},
        {"max_length": np.nan},
        {"min_volume": -1},
        {"ambiguity_ratio": 1},
    ],
)
def test_options_reject_invalid_budgets(kw):
    with pytest.raises(ValueError):
        GrainSearchOptions(**kw)


def test_native_and_array_boundaries():
    engine = _core.ExperimentalLatticeEngine(1)
    for q, n, step in [
        (np.zeros((0, 3)), 32, 0.1),
        (np.ones((3, 3)), 17, 0.1),
        (np.ones((3, 3)), 512, 0.1),
        (np.full((3, 3), np.nan), 32, 0.1),
        (np.ones((3, 3)), 32, 0),
        (np.ones((3, 3)) * 100, 32, 0.1),
    ]:
        with pytest.raises(ValueError):
            engine.build_field(q, n, step)
    q, _, _ = mixed_cloud()
    for bad in [np.ones(len(q)), np.ones((len(q), 1), bool)]:
        with pytest.raises(ValueError, match="training"):
            find_grains(q, training=bad)
    with pytest.raises(ValueError, match="insufficient"):
        find_grains(q[:5])
    with pytest.raises(ValueError):
        integer_lattice_basis(np.ones((3, 3)))


@pytest.mark.parametrize(
    "basis",
    [
        [[4.1, 0, 0], [1.2, 5.3, 0], [-0.8, 0.5, 6.7]],
        [[5.2, 0, 0], [-2.6, 4.5033321, 0], [0, 0, 8.4]],
        [[0, 3.2, 3.2], [3.2, 0, 3.2], [3.2, 3.2, 0]],
    ],
)
def test_general_metric_without_a_crystal_system_prior(basis):
    from test_experimental_indexing import synthetic

    q, truth = synthetic(basis, outliers=100)
    result = find_grains(q, options=GrainSearchOptions(max_length=15, grid_size=256, max_grains=3))
    best = result.candidates[0]
    assert best.accepted and best.validation_fraction > 0.7
    assert best.volume == pytest.approx(abs(np.linalg.det(truth)), rel=0.01)
    change = best.direct_basis @ np.linalg.inv(truth)
    np.testing.assert_allclose(change, np.rint(change), atol=0.025)


def test_ensemble_prefers_compact_coherent_lattices_over_a_dense_superset():
    q, truths, sizes = mixed_cloud()
    training = np.arange(len(q)) % 5 != 0
    # A giant supercell can index every point of the first actual lattice, but
    # is not an additional orientation. Its extra lattice points carry a density
    # penalty, rather than erasing the compact cell through one-way overlap.
    pool = [4 * truths[0], truths[0], truths[1]]
    options = GrainSearchOptions(max_length=40, max_grains=4)
    engine = _core.ExperimentalLatticeEngine(4)
    models, labels, audit = _consolidate(engine, q[training], pool, options)
    assert len(models) == 2
    selected = [r["pool_id"] for r in audit["selected"]]
    assert set(selected) == {1, 2}
    sigma = options.reciprocal_tolerance / 4
    evidence, errors = ref.ensemble_evidence(q[training], pool, sigma)
    origin_evidence, origin_error = ref.ensemble_evidence(np.zeros((1, 3)), pool, sigma)
    np.testing.assert_array_equal(origin_evidence, 0)
    assert np.isinf(origin_error).all()
    first = audit["selected"][0]
    assert first["marginal_evidence"] == pytest.approx(evidence[first["pool_id"]].sum(), rel=2e-14)
    assert first["core_training"] == int((errors[first["pool_id"]] < sigma).sum())
    expected = np.repeat([1, 2, 0], [*sizes, 100])[training]
    mapping = {pool_id: i + 1 for i, pool_id in enumerate(selected)}
    for source, pool_id in ((1, 1), (2, 2)):
        assert np.mean(labels[expected == source] == mapping[pool_id]) > 0.97


def test_ensemble_pool_and_training_assignment_are_reserved_data_independent():
    q, _, _ = mixed_cloud()
    training = np.arange(len(q)) % 5 != 0
    o = GrainSearchOptions(max_length=15, grid_size=128, max_grains=4)
    searches = (o, GrainSearchOptions(max_length=10, grid_size=128, max_grains=4))
    first = find_grains_ensemble(q, training=training, options=o, searches=searches)
    assert sum(c.accepted for c in first.candidates) == 2
    assert first.method == "fft_grains_ensemble"
    q[~training] = np.random.default_rng(12).uniform(-1, 1, ((~training).sum(), 3))
    second = find_grains_ensemble(q, training=training, options=o, searches=searches)
    assert (
        first.diagnostics[: -len(first.candidates)] == second.diagnostics[: -len(second.candidates)]
    )
    np.testing.assert_array_equal(first.labels[training], second.labels[training])
    for a, b in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(a.direct_basis, b.direct_basis)
        np.testing.assert_array_equal(a.hkl[training], b.hkl[training])
    assert not any(c.accepted for c in second.candidates)


def test_ensemble_options_budgets_empty_pool_and_cancellation():
    o = GrainSearchOptions(max_grains=40)
    assert len(ensemble_search_options(o)) == 28
    assert sum(s.max_grains for s in ensemble_search_options(o)) <= 256
    q, _, _ = mixed_cloud()
    for searches in ((), [o] * 33, [o] * 7, [object()]):
        with pytest.raises(ValueError):
            find_grains_ensemble(q, searches=searches)
    with pytest.raises(ValueError, match="training"):
        find_grains_ensemble(q, training=np.ones(len(q)))
    with pytest.raises(ValueError, match="insufficient"):
        find_grains_ensemble(q[:5])
    with pytest.raises(TypeError):
        find_grains_ensemble(q, options=object())
    with pytest.raises(InterruptedError):
        find_grains_ensemble(
            q, searches=(o,), progress=lambda *_: (_ for _ in ()).throw(InterruptedError())
        )
    models, labels, audit = _consolidate(_core.ExperimentalLatticeEngine(1), q, [], o)
    assert models == [] and audit["selected"] == [] and not labels.any()


def test_ensemble_evidence_units_and_equal_position_density_penalty():
    q = np.array([[0.25, 0, 0], [0.125, 0, 0], [0, 0, 0]])
    models = np.array([np.eye(3) * 4, np.eye(3) * 8])
    evidence, errors = ref.ensemble_evidence(q, models, 0.01)
    # The first point is indexed exactly by both; the smaller cell has fewer
    # chance reciprocal locations and greater positional evidence there.
    assert errors[0, 0] == errors[1, 0] == 0
    assert evidence[0, 0] > evidence[1, 0]
    scaled, scaled_errors = ref.ensemble_evidence(q * 2.5, models / 2.5, 0.025)
    np.testing.assert_allclose(scaled, evidence, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(scaled_errors, errors * 2.5, rtol=2e-14, atol=1e-14)
