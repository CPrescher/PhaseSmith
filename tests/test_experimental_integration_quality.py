"""Independent record-level checks for advisory image quality; intensities stay exact."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from phasesmith.experimental.integration import IntegrationOptions, ReflectionTable
from phasesmith.experimental.integration_quality import (
    QualityIssue,
    QualityOptions,
    assess_integration,
)


def make_table(intensities, *, flags=None, position_offset=0.0):
    data = np.asarray(intensities, float)
    nf, n = data.shape
    frames = np.repeat(np.arange(nf), n)
    ids = np.tile(np.arange(n), nf)
    predicted = np.column_stack([np.arange(n) * 10 + 12, np.full(n, 12), np.full(n, 1.5)])
    fv = np.zeros((nf * n, 11))
    fv[:, :2] = predicted[ids, :2] + position_offset
    fv[:, 2] = data.ravel()
    fv[:, 3] = 2.0
    fv[:, 4] = 10.0
    fv[:, 9] = 9
    fv[:, 10] = 20
    boxes = np.tile([11, 11, 14, 14], (nf * n, 1))
    values = np.column_stack(
        [
            data.sum(axis=0),
            np.full(n, 2 * np.sqrt(nf)),
            np.full(n, 90 * nf),
            np.full(n, 9 * nf),
            np.full(n, nf),
            np.full(n, 20 * nf),
        ]
    )
    raw = np.zeros(nf * n, np.uint32) if flags is None else np.asarray(flags, np.uint32).ravel()
    summary = np.bitwise_or.reduce(raw.reshape(nf, n), axis=0)
    return ReflectionTable(
        np.column_stack([np.arange(n) + 1, np.zeros((n, 2), int)]),
        predicted,
        np.zeros(n, np.int64),
        values,
        summary,
        ids,
        frames,
        boxes,
        fv,
        raw,
        np.column_stack([np.arange(nf), np.arange(1, nf + 1)]),
        tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(nf)),
        IntegrationOptions(),
        "poisson_plus_read_noise_with_variance_floor",
        (1.0, 0.0, None),
    )


def test_endpoints_and_exact_no_mutation():
    t = make_table([[2, 20], [100, 100], [2, 20]])
    before = {
        name: getattr(t, name).tobytes()
        for name in ("values", "frame_values", "flags", "frame_flags")
    }
    q = assess_integration(t)
    np.testing.assert_allclose(q.reflection_values[:, 0], [2 / 104, 20 / 140], rtol=1e-15)
    assert not q.reflection_issues[0] & QualityIssue.WINDOW_END
    assert q.reflection_issues[1] & QualityIssue.WINDOW_END
    np.testing.assert_array_equal(q.column("n_window_end"), [1, 1, 1])
    assert all(getattr(t, k).tobytes() == v for k, v in before.items())
    assert not q.frame_values.flags.writeable
    assert "wider" in " ".join(q.actions(0))


def test_invalid_endpoints_not_replaced_and_short_curves_unknown():
    t = make_table([[20], [100], [20]])
    v = t.frame_values.copy()
    v[0, 2:4] = np.nan
    q = assess_integration(replace(t, frame_values=v))
    assert np.isnan(q.reflection_values[0, 0])
    assert not q.reflection_issues[0] & QualityIssue.WINDOW_END
    q = assess_integration(make_table([[100], [100]]))
    assert np.isnan(q.reflection_values[0, 0])


def test_frame_flags_pixel_fraction_geometry_and_actions():
    flags = np.zeros((3, 6), np.uint32)
    flags[0, :5] = [1, 2, 4, 8, 16]
    t = make_table(np.full((3, 6), 20.0), flags=flags, position_offset=3.0)
    v = t.frame_values.copy()
    v[1, 9] = 6
    q = assess_integration(replace(t, frame_values=v))
    np.testing.assert_allclose(q.column("masked_pixel_fraction"), [3 / 54, 0, 0], atol=1e-15)
    np.testing.assert_array_equal(q.frame_values[0, 12:], [1, 1, 1, 1, 1])
    assert not q.frame_issues[0] & QualityIssue.POSITION  # only one unflagged strong spot
    assert q.frame_issues[1] & QualityIssue.POSITION
    assert len(q.actions(0)) >= 5
    assert q.frame_issues[0] & QualityIssue.SATURATED


def test_negative_counts_are_reviewed_without_rejection():
    data = np.full((3, 12), 2.0)
    data[1, :3] = -10
    t = make_table(data)
    q = assess_integration(t)
    assert q.frame_issues[1] & QualityIssue.NEGATIVE
    assert q.column("n_negative_strong")[1] == 3
    assert t.frame_values[12, 2] == -10
    assert not q.frame_issues[0] & QualityIssue.NEGATIVE


def test_spatial_and_background_model_actions_remain_visible():
    t = make_table([[30, 30], [30, 30]], flags=[[2048, 4096], [0, 0]])
    q = assess_integration(t)
    assert q.frame_issues[0] & QualityIssue.SIGNAL_BOUNDARY
    assert q.frame_issues[0] & QualityIssue.BACKGROUND_MODEL
    assert q.reflection_issues[0] & QualityIssue.SIGNAL_BOUNDARY
    assert q.reflection_issues[1] & QualityIssue.BACKGROUND_MODEL
    assert any("spatial boundary" in a for a in q.actions(0))
    alternate = replace(t, values=t.values * [1.2, 1, 1, 1, 1, 1])
    q = assess_integration(t, comparison=alternate)
    assert q.reflection_issues[0] & QualityIssue.WINDOW_CHANGE


def test_support_assessment_retains_failed_checks_and_originals():
    from phasesmith.experimental.integration_quality import assess_support

    t = make_table([[0, 0, 0], [200, 200, 1], [0, 0, 0]])
    a = replace(t, values=t.values * [1.01, 1, 1, 1, 1, 1])
    values = t.values.copy()
    values[1, 0] *= 1.2
    b = replace(t, values=values)
    original = t.values.tobytes()
    report = assess_support(t, [a, b])
    np.testing.assert_array_equal(report.passes, [True, False, False])
    assert report.reasons[1, 1] & 32 and report.reasons[2, 0] & 2
    assert t.values.tobytes() == original
    assert not report.relative_changes.flags.writeable
    with pytest.raises(ValueError):
        assess_support(t, [a])
    with pytest.raises(ValueError):
        assess_support(t, [a, b], max_relative_change=True)
    bad = replace(a, frame_hashes=("changed",) * 3)
    with pytest.raises(ValueError):
        assess_support(t, [bad, b])


def test_empty_frames_and_angular_coverage():
    t = make_table(np.full((3, 1), 20.0))
    keep = t.frame_indices != 1
    t = replace(
        t,
        observation_ids=t.observation_ids[keep],
        frame_indices=t.frame_indices[keep],
        boxes=t.boxes[keep],
        frame_values=t.frame_values[keep],
        frame_flags=t.frame_flags[keep],
        flags=np.array([256], np.uint32),
    )
    q = assess_integration(t)
    assert q.frame_issues[1] == QualityIssue.NO_MEASUREMENTS
    assert np.isnan(q.column("median_background")[1])
    assert q.frame_issues[0] & QualityIssue.ANGULAR
    assert len(q.frame_values) == 3


def test_window_comparison_and_provenance():
    t = make_table([[2, 2], [100, 100], [2, 2]])
    v = t.values.copy()
    v[:, 0] *= [1.25, 1.05]
    other = replace(t, values=v, options=IntegrationOptions(pixel_radius=7))
    q = assess_integration(t, comparison=other)
    np.testing.assert_allclose(q.reflection_values[:, 1], [0.25, 0.05], atol=1e-15)
    assert q.reflection_issues[0] & QualityIssue.WINDOW_CHANGE
    assert not q.reflection_issues[1] & QualityIssue.WINDOW_CHANGE
    for invalid in [
        replace(other, frame_hashes=("bad",) * 3),
        replace(other, noise_parameters=(2.0, 0.0, None)),
        replace(other, predicted=other.predicted + 1),
        replace(other, noise_parameters=None),
    ]:
        with pytest.raises(ValueError):
            assess_integration(t, comparison=invalid)
    with pytest.raises(ValueError):
        assess_integration(replace(t, noise_parameters=None), comparison=other)


def test_randomized_frame_summaries_independent_loops():
    rng = np.random.default_rng(20261025)
    t = make_table(rng.normal(5, 10, (9, 17)))
    q = assess_integration(t)
    for frame in range(9):
        rows = [i for i, f in enumerate(t.frame_indices) if f == frame]
        ratios = [t.frame_values[i, 2] / t.frame_values[i, 3] for i in rows]
        strong = sum(r > 3 for r in ratios)
        negative = sum(r < -3 for r in ratios)
        np.testing.assert_allclose(
            q.frame_values[frame, [1, 2, 3, 7, 8]],
            [len(rows), len(rows), strong, np.median(ratios), negative],
            atol=1e-14,
        )
    repeated = assess_integration(t)
    assert repeated.frame_values.tobytes() == q.frame_values.tobytes()


@pytest.mark.parametrize(
    "kw",
    [
        dict(endpoint_fraction=0),
        dict(endpoint_fraction=1),
        dict(strong_sigma=np.inf),
        dict(min_position_spots=True),
        dict(min_negative_measurements=0),
        dict(position_pixels=np.nan),
    ],
)
def test_bad_options(kw):
    with pytest.raises(ValueError):
        QualityOptions(**kw)


def test_bad_shapes_indices_and_pixel_counts():
    t = make_table([[2], [100], [2]])
    for bad in [
        replace(t, frame_indices=np.array([1, 0, 2])),
        replace(t, observation_ids=np.array([9, 0, 0])),
        replace(t, boxes=np.ones((3, 4), int)),
        replace(t, frame_flags=np.zeros(3, float)),
        replace(t, frame_values=np.ones((3, 10))),
    ]:
        with pytest.raises(ValueError):
            assess_integration(bad)
    v = t.frame_values.copy()
    v[0, 9] = 10
    with pytest.raises(ValueError):
        assess_integration(replace(t, frame_values=v))
