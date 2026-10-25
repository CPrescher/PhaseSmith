"""Position-only lattice comparison: synthetic truths, never a sample cell prior."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental.cell_comparison import (
    CellComparisonOptions,
    compare_cells,
    reciprocal_hnfs,
)
from phasesmith.experimental.cell_metrics import (
    BRAVAIS,
    centring,
    find_bravais_settings,
    integer_det,
    metric_chain,
    metric_directions,
    metric_step,
    project_metric,
    reduce_cell,
)
from phasesmith.experimental.cell_reference import metric_basis_reference, selling_reference
from phasesmith.experimental.prediction import predict_spots
from phasesmith.experimental.reciprocal import PanelGeometry

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "experimental_reduce_lattice"), reason="experimental native build required"
)

CENTRES = {
    "P": np.eye(3, dtype=np.int64),
    "C": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=np.int64),
    "I": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int64),
    "F": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int64),
    "R": np.array([[1, -1, 0], [0, 1, -1], [1, 1, 1]], dtype=np.int64),
}


def synthetic(symbol):
    if symbol[0] == "a":
        g = np.array([[16, 3, 2], [3, 39, 4], [2, 4, 81.0]])
    elif symbol[0] == "m":
        g = np.array([[16, 0, -5], [0, 39, 0], [-5, 0, 81.0]])
    elif symbol[0] == "o":
        g = np.diag([16.0, 39.0, 81.0])
    elif symbol[0] == "t":
        g = np.diag([25.0, 25.0, 81.0])
    elif symbol[0] == "h":
        g = np.array([[25, -12.5, 0], [-12.5, 25, 0], [0, 0, 81.0]])
    else:
        g = np.eye(3) * 25
    t = CENTRES[symbol[1]]
    # Exercise an arbitrary input basis, far from its conventional setting.
    shear = np.array([[1, 7, -3], [0, 1, 4], [0, 0, 1]], dtype=np.int64)
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    return shear @ np.linalg.solve(t, np.linalg.cholesky(g)) @ rotation


@pytest.mark.parametrize("symbol", BRAVAIS)
def test_all_fourteen_sheared_rotated_lattices(symbol):
    a = synthetic(symbol)
    settings = find_bravais_settings(a, max_strain=0.001, max_angle=0.1)
    matching = [s for s in settings if s.symbol == symbol]
    assert matching, (symbol, [(s.symbol, s.cell) for s in settings])
    for s in settings:
        np.testing.assert_allclose(s.direct_basis, s.transform @ a, atol=1e-11)
        assert centring(s.transform) == s.symbol[1]
        assert integer_det(s.transform) > 0
        h = np.array([[1, 2, -3], [3, -5, 7]])
        hc = h @ s.transform.T
        np.testing.assert_allclose(
            hc @ np.linalg.inv(s.direct_basis).T, h @ np.linalg.inv(a).T, atol=2e-12
        )
        assert not s.transform.flags.writeable
    assert all(s.strain < 1e-10 for s in matching)


@pytest.mark.parametrize("symbol", BRAVAIS)
def test_metric_derivatives_and_retraction(symbol):
    a0 = synthetic(symbol)
    s = next(
        s for s in find_bravais_settings(a0, max_strain=0.001, max_angle=0.1) if s.symbol == symbol
    )
    a, _ = project_metric(s.direct_basis, symbol)
    b = np.linalg.inv(a).T
    chain = metric_chain(b, symbol)
    eps = 1e-6
    for k in range(chain.shape[1]):
        step = np.zeros(chain.shape[1])
        step[k] = eps
        fd = (metric_step(b, step, symbol) - metric_step(b, -step, symbol)) / (2 * eps)
        np.testing.assert_allclose(chain[:, k], fd.ravel(), rtol=3e-7, atol=2e-9)
    # Independent metric construction, separate from local retraction.
    directions = metric_directions(symbol)
    gram = a @ a.T
    coords = np.linalg.lstsq(directions.reshape(len(directions), 9).T, gram.ravel(), rcond=None)[0]
    q = np.linalg.cholesky(gram).T @ b
    np.testing.assert_allclose(metric_basis_reference(coords, directions, q), b, atol=1e-13)
    changed = metric_step(b, np.full(chain.shape[1], 1e-5), symbol)
    ca = np.linalg.inv(changed).T
    projected, _ = project_metric(ca, symbol)
    np.testing.assert_allclose(projected @ projected.T, ca @ ca.T, atol=1e-10)


def test_selling_independent_reference_random_and_invalid():
    rng = np.random.default_rng(741)
    for _ in range(50):
        a = rng.normal(size=(3, 3)) * 5
        if abs(np.linalg.det(a)) < 0.2:
            continue
        native = reduce_cell(a)
        reference, t, n = selling_reference(a)
        np.testing.assert_allclose(native.direct_basis, reference, atol=2e-12)
        np.testing.assert_array_equal(native.transform, t)
        assert native.iterations == n
        assert abs(integer_det(t)) == 1
        v = np.vstack((native.direct_basis, -native.direct_basis.sum(axis=0)))
        gram = v @ v.T
        assert np.max(gram - np.diag(np.diag(gram))) < 1e-9
    for a in [np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2), np.eye(3) * 1e8]:
        with pytest.raises(ValueError):
            reduce_cell(a)
    with pytest.raises(ValueError):
        reduce_cell(np.eye(3), max_iterations=0)


def test_hnf_census_and_exact_centres():
    assert [sum(round(np.linalg.det(h)) == n for h in reciprocal_hnfs(4)) for n in range(1, 5)] == [
        1,
        7,
        13,
        35,
    ]
    for c, t in CENTRES.items():
        assert centring(t) == c
    assert centring(np.diag([2, 1, 1])) is None


def test_smaller_cells_rejected_for_populated_parities_and_training_only():
    rng = np.random.default_rng(12)
    a = np.diag([4.0, 6.0, 9.0])
    h = rng.integers(-6, 7, size=(500, 3))
    h = h[np.any(h != 0, axis=1)]
    points = h @ np.linalg.inv(a).T + rng.normal(0, 1e-5, h.shape)
    train = np.arange(len(h)) % 5 != 0
    result = compare_cells(points, a, training=train)
    assert len(result.primitives) == 56
    assert result.recommended_primitive_id == 0
    assert result.recommended.setting.symbol == "oP"
    assert len([s for s in result.settings if s.setting.symbol == "mP"]) == 3
    assert all(not p.supported for p in result.primitives[1:])
    assert all(p.excluded_training > 0 for p in result.primitives[1:])
    changed = points.copy()
    changed[~train] *= 1.03
    again = compare_cells(changed, a, training=train)
    assert again.recommended_setting_index == result.recommended_setting_index
    for first, second in zip(result.settings, again.settings, strict=True):
        np.testing.assert_array_equal(first.reciprocal_basis, second.reciprocal_basis)
        assert first.training_score == second.training_score
    assert not result.parent_indexed.flags.writeable


@pytest.mark.parametrize("centre", ["C", "I", "F", "R"])
def test_find_primitive_from_centred_conventional_input(centre):
    rng = np.random.default_rng(390)
    symbol = "hR" if centre == "R" else "o" + centre
    primitive = reduce_cell(synthetic(symbol)).direct_basis
    s = next(s for s in find_bravais_settings(primitive) if s.symbol == symbol)
    conventional = s.direct_basis
    h = rng.integers(-5, 6, size=(400, 3))
    h = h[np.any(h, axis=1)]
    points = h @ np.linalg.inv(primitive).T
    train = np.arange(len(h)) % 5 != 0
    result = compare_cells(points, conventional, training=train)
    chosen = result.primitives[result.recommended_primitive_id]
    assert chosen.index == s.index
    assert chosen.excluded_training == 0
    assert chosen.volume == pytest.approx(abs(np.linalg.det(primitive)), rel=1e-10)
    assert symbol in [f.setting.symbol for f in result.settings if f.primitive_id == chosen.id]


def test_bounded_metric_candidates_not_forced_symmetry():
    a = np.array([[4.0, 0, 0], [0.7, 6.0, 0], [0.6, 0.8, 9.0]])
    assert [s.symbol for s in find_bravais_settings(a, max_angle=0.1, max_strain=0.001)] == ["aP"]
    for kwargs in [dict(coefficient_limit=0), dict(max_strain=np.nan), dict(max_angle=0)]:
        with pytest.raises(ValueError):
            find_bravais_settings(a, **kwargs)
    for kwargs in [dict(max_subcell_index=9), dict(minimum_retention=0.5), dict(q_tolerance=-1)]:
        with pytest.raises(ValueError):
            CellComparisonOptions(**kwargs)


def test_detector_comparison_native_jacobian_and_heldout_independence():
    rng = np.random.default_rng(65)
    # Sample rotation z; beam x; detector normal x, fast y, slow z.
    geo = PanelGeometry(
        np.array([[0.2, -0.08, -0.08], [0, 0.0001, 0], [0, 0, 0.0001], [1, 0, 0], [0, 0, 1.0]]), 0.5
    )
    a = np.diag([5.0, 7.0, 10.0])
    h = rng.integers(-5, 6, size=(1000, 3)).astype(np.int64)
    pred = predict_spots(h, np.linalg.inv(a).T, geo, 0.0)
    good = pred.flags == 0
    h = h[good]
    obs = pred.values[good].copy()
    obs += rng.normal(0, [0.03, 0.03, 0.002], obs.shape)
    points = h @ np.linalg.inv(a).T
    train = np.arange(len(h)) % 5 != 0
    perturb = a.copy()
    perturb[0, 0] *= 1.001
    perturb[1, 2] = 0.003
    options = CellComparisonOptions(max_subcell_index=1)
    result = compare_cells(
        points, perturb, training=train, observed=obs, geometry=geo, options=options
    )
    assert result.recommended.setting.symbol == "oP"
    assert result.recommended.training_metrics["rms_xy"] < 0.1
    changed = obs.copy()
    changed[~train, :2] += 5
    again = compare_cells(
        points, perturb, training=train, observed=changed, geometry=geo, options=options
    )
    for first, second in zip(result.settings, again.settings, strict=True):
        np.testing.assert_array_equal(first.reciprocal_basis, second.reciprocal_basis)
    assert (
        result.recommended.validation_metrics["rms_xy"]
        < again.recommended.validation_metrics["rms_xy"]
    )


@pytest.mark.parametrize("symbol", BRAVAIS)
def test_gemmi_black_box_selling_energy_and_lattice_holohedry(symbol):
    # Optional test oracle only. Documented cell APIs, never implementation code.
    gemmi = pytest.importorskip("gemmi", minversion="0.7.5")
    a = synthetic(symbol)
    from phasesmith.experimental.indexing import cell_parameters

    cell = gemmi.UnitCell(*cell_parameters(a))
    selling = gemmi.SellingVector(cell, gemmi.SpaceGroup("P 1"))
    selling.reduce()
    reduced = reduce_cell(a).direct_basis
    v = np.vstack((reduced, -reduced.sum(axis=0)))
    assert np.sum(v * v) == pytest.approx(selling.sum_b_squared(), rel=1e-10, abs=1e-9)
    setting = next(
        s for s in find_bravais_settings(a, max_strain=0.001, max_angle=0.1) if s.symbol == symbol
    )
    conventional = gemmi.UnitCell(*setting.cell)
    group = gemmi.find_lattice_symmetry(conventional, symbol[1], 0.01)
    order = {"a": 1, "m": 2, "o": 4, "t": 8, "h": 12, "c": 24}[symbol[0]]
    if symbol == "hR":
        order = 6
    # Gemmi returns proper rotations and separate centring translations.
    assert len(group.sym_ops) == order
    assert len(group.cen_ops) == setting.index


@pytest.mark.parametrize("symbol", BRAVAIS)
def test_noisy_full_comparison_recovers_synthetic_bravais(symbol):
    rng = np.random.default_rng(510)
    a = synthetic(symbol)
    reduced = reduce_cell(a).direct_basis
    h = rng.integers(-5, 6, (220, 3))
    h = h[np.any(h != 0, axis=1)]
    points = h @ np.linalg.inv(reduced).T + rng.normal(0, 2e-5, h.shape)
    distorted = a @ (np.eye(3) + rng.normal(0, 1e-4, (3, 3)))
    train = np.arange(len(points)) % 5 != 0
    result = compare_cells(
        points, distorted, training=train, options=CellComparisonOptions(max_subcell_index=1)
    )
    assert result.recommended.setting.symbol == symbol
    assert result.recommended.training_metrics["rms_q"] < 6e-5
    assert result.recommended.validation_metrics["rms_q"] < 6e-5
    assert np.all(np.diff(result.recommended.history) < 0)


def test_boundaries_snapshot_budget_and_exact_integer_determinant():
    with pytest.raises(ValueError, match="snapshot budget"):
        compare_cells(
            np.tile(np.diag([0.25, 1 / 6, 1 / 9]), (4400, 1)),
            np.diag([4.0, 6.0, 9.0]),
            training=np.ones(13200, bool),
            options=CellComparisonOptions(max_subcell_index=8),
        )
    # Large unimodular shears: intermediate products can overflow i64.
    t = np.array([[1, 10**9, 0], [0, 1, 10**9], [0, 0, 1]], dtype=np.int64)
    assert integer_det(t) == 1
    assert centring(t) == "P"
    for t in [np.eye(3), np.ones((2, 3), dtype=int)]:
        with pytest.raises(ValueError):
            centring(t)
    with pytest.raises(ValueError):
        list(reciprocal_hnfs(9))


def test_strict_parent_distance_boundary_origin_and_empty_validation():
    rng = np.random.default_rng(72)
    a = np.diag([4.0, 6.0, 9.0])
    h = rng.integers(1, 5, (80, 3))
    q = h @ np.linalg.inv(a).T
    tolerance = 1 / 16
    q[0, 0] += tolerance
    q[1, 0] = np.nextafter(q[1, 0] + tolerance, -np.inf)
    q[-1] = 0
    r = compare_cells(
        q,
        a,
        training=np.ones(len(q), bool),
        options=CellComparisonOptions(max_subcell_index=1, q_tolerance=tolerance),
    )
    assert not r.parent_indexed[0]
    assert r.parent_indexed[1]
    assert not r.parent_indexed[-1]
    assert r.recommended.validation_metrics["count"] == 0
    assert np.isnan(r.recommended.validation_metrics["rms_scaled"])


def test_explicit_experimental_feature_error(monkeypatch):
    monkeypatch.delattr(_core, "ExperimentalLatticeEngine")
    with pytest.raises(RuntimeError, match="experimental-spots"):
        compare_cells(np.ones((12, 3)), np.eye(3), training=np.ones(12, bool))


def test_extreme_input_shear_does_not_duplicate_conventional_constraints():
    from collections import Counter

    a = reduce_cell(synthetic("cF")).direct_basis
    sheared = np.array([[1, 1000, 0], [0, 1, 0], [0, 0, 1]]) @ a
    canonical = find_bravais_settings(a)
    alternative = find_bravais_settings(sheared)
    assert Counter(s.symbol for s in canonical) == Counter(s.symbol for s in alternative)
    assert sum(s.symbol == "cF" for s in alternative) == 1
