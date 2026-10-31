"""Qualified fitted pools merge on training evidence, never reserved outcomes."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.grains import GrainSearchOptions, find_grains_ensemble
from phasesmith.single_crystal.ownership import GrainOwnershipOptions, _physical_fields
from phasesmith.single_crystal.retention import GrainPoolMergeOptions, merge_fitted_training_pools

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


def pools():
    rng = np.random.default_rng(91381)
    bases = []
    points = []
    for lengths in ([4.0, 5.0, 6.0], [5.2, 6.2, 8.3], [4.3, 5.1, 7.7]):
        a = np.diag(lengths) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        h = np.unique(rng.integers(-3, 4, (120, 3)), axis=0)
        h = h[np.any(h != 0, axis=1)][:80]
        points.append(h @ np.linalg.inv(a).T + rng.normal(0, 0.00003, h.shape))
        bases.append(a)
    points = np.vstack([*points, rng.uniform(-1, 1, (300, 3))])
    charts = np.broadcast_to(np.eye(3) * 1024, (len(points), 3, 3)).copy()
    return points, np.array(bases), charts, np.ones(len(points), bool)


def test_merge_adds_independent_model_and_preserves_every_incumbent_and_provenance():
    q, bases, charts, valid = pools()
    candidates = np.array([bases[0] * 1.00001, bases[2], np.diag([3.1, 3.3, 3.7])])
    old = bases[:2].copy()
    result = merge_fitted_training_pools(q, old, candidates, charts=charts, chart_valid=valid)
    assert len(result.bases) == 3
    np.testing.assert_array_equal(result.bases[:2], old)
    np.testing.assert_array_equal(result.bases[2], candidates[1])
    assert result.source_kind.tolist() == [0, 0, 1]
    assert result.source_index.tolist() == [0, 1, 1]
    assert result.diagnostics["candidates"][0]["candidate"] == 1
    assert any(
        x.get("reason") == "oriented_cell_alternative" for x in result.diagnostics["candidates"]
    )
    assert np.all(np.array(result.diagnostics["incumbent_after"]) >= 20)
    assert not hasattr(result, "accepted") and not result.bases.flags.writeable
    repeat = merge_fitted_training_pools(q, old, candidates, charts=charts, chart_valid=valid)
    np.testing.assert_array_equal(result.bases, repeat.bases)
    assert result.diagnostics == repeat.diagnostics
    assert not any("selection" in key or "test" in key for key in result.diagnostics)


def test_narrow_proposal_gate_is_not_reapplied_to_post_fit_matrix():
    q, bases, charts, valid = pools()
    q = q.copy()
    charts = charts / 10
    # A geometry-consistent post-fit pool can have only 19 narrow-core points.
    # Qualification happened upstream; the final fixed physical/rank gates are
    # not the same quantity as a different pre-fit reciprocal-core gate.
    first = 160
    q[first + 19 : first + 80, 1] += 0.0045
    result = merge_fitted_training_pools(q, bases[:2], bases[2:], charts=charts, chart_valid=valid)
    e, core, physical = _physical_fields(
        _core.ExperimentalLatticeEngine(1),
        q,
        bases[2:],
        charts,
        valid,
        GrainSearchOptions(),
        GrainOwnershipOptions(native_fields=True),
    )
    assert core.sum() < 20 and physical.sum() >= 20 and e.sum() > 0
    assert len(result.bases) == 3 and result.diagnostics["upstream_proposal_gates_reused"]


def test_training_controls_reject_a_dense_uniform_noise_hypothesis():
    rng = np.random.default_rng(81)
    q = rng.uniform(-1, 1, (2000, 3))
    a = np.diag([17.0, 17.0, 17.0])[None]
    charts = np.broadcast_to(np.eye(3) * 100, (len(q), 3, 3)).copy()
    result = merge_fitted_training_pools(
        q, np.empty((0, 3, 3)), a, charts=charts, chart_valid=np.ones(len(q), bool)
    )
    assert len(result.bases) == 0
    assert result.diagnostics["candidates"][0]["reason"] == "training_control"


def test_training_pool_has_no_validation_metrics_or_reserved_coordinate_requirement(monkeypatch):
    import phasesmith.single_crystal.grains as grains

    q = np.random.default_rng(731).uniform(-1, 1, (301, 3))
    o = GrainSearchOptions(grid_size=64, max_length=8, max_grains=1, fft_peaks=40, basis_vectors=20)

    def forbidden(*a, **kw):
        raise AssertionError("validation called during proposal generation")

    monkeypatch.setattr(grains, "_finish", forbidden)
    result = find_grains_ensemble(
        q, training=np.ones(len(q), bool), options=o, searches=(o,), _training_only=True
    )
    assert result.bases.shape[1:] == (3, 3) and result.labels.shape == (len(q),)
    assert not result.bases.flags.writeable and not hasattr(result, "candidates")
    with pytest.raises(ValueError, match="insufficient"):
        find_grains_ensemble(q, training=np.ones(len(q), bool), options=o, searches=(o,))
    with pytest.raises(ValueError, match="boolean"):
        find_grains_ensemble(q, options=o, _training_only=1)


def test_fit_budget_compares_actual_rows_hkls_and_internal_training_splits():
    from phasesmith.single_crystal.grains import GrainProposalRefinementOptions, _same_fit_start

    a = np.diag([4.0, 5.0, 6.0])
    models = [a, a * 1.00001]
    error = np.full(40, 0.0001)
    hkl = np.arange(120).reshape(40, 3)
    eligible = np.ones(40, bool)
    first = (0.0, 120.0, 0, hkl, error, eligible)
    second = (0.0, 120.0, 1, hkl.copy(), error.copy(), eligible.copy())
    search = GrainSearchOptions()
    trim = GrainProposalRefinementOptions(
        distinct_budget=True,
        cohort_aware_budget=True,
        trim_fractions=(0.9,),
        retain_split_fit=False,
    )
    assert _same_fit_start(models, first, second, search, trim)
    changed_rows = list(second)
    changed_rows[5] = eligible.copy()
    changed_rows[5][0] = False
    assert not _same_fit_start(models, first, changed_rows, search, trim)
    changed_hkl = list(second)
    changed_hkl[3] = hkl.copy()
    changed_hkl[3][0, 0] += 1
    assert not _same_fit_start(models, first, changed_hkl, search, trim)
    # Legacy internal splits depend on pool index even for identical cohorts.
    assert not _same_fit_start(models, first, second, search, replace(trim, retain_split_fit=True))
    # Default matrix-only budget and an identical start remain unchanged.
    assert _same_fit_start(
        models, first, changed_rows, search, replace(trim, cohort_aware_budget=False)
    )
    assert _same_fit_start(models, first, first, search, replace(trim, retain_split_fit=True))
    with pytest.raises(ValueError, match="boolean"):
        GrainProposalRefinementOptions(cohort_aware_budget=1)


@pytest.mark.parametrize(
    "kwargs", [dict(control_rotations=1), dict(max_additions=-1), dict(max_alias_index=True)]
)
def test_merge_options_validate(kwargs):
    with pytest.raises(ValueError):
        GrainPoolMergeOptions(**kwargs)


def test_merge_budget_and_invalid_array_boundaries():
    q, bases, charts, valid = pools()
    result = merge_fitted_training_pools(
        q,
        bases[:2],
        bases[2:],
        charts=charts,
        chart_valid=valid,
        options=GrainPoolMergeOptions(max_additions=0),
    )
    np.testing.assert_array_equal(result.bases, bases[:2])
    assert result.diagnostics["candidates"][0]["reason"] == "model_budget"
    for bad in [q.astype(np.float32), q[:, :2], np.full_like(q, np.nan)]:
        with pytest.raises(ValueError):
            merge_fitted_training_pools(bad, bases[:2], bases[2:], charts=charts, chart_valid=valid)
    with pytest.raises(ValueError):
        merge_fitted_training_pools(q, bases[:2], bases[2], charts=charts, chart_valid=valid)
    with pytest.raises(ValueError):
        merge_fitted_training_pools(
            q,
            bases[:2],
            bases[2:],
            charts=charts,
            chart_valid=valid,
            search=replace(GrainSearchOptions(), max_grains=1),
        )
