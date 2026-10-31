"""Fused proposal evidence, independent equations and unchanged eligibility."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.experimental import grains_reference as ref
from phasesmith.experimental.grains import GrainSearchOptions
from phasesmith.experimental.ownership import GrainOwnershipOptions, _physical_fields

pytestmark = pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="experimental-spots not built"
)


def fixture():
    rng = np.random.default_rng(81231)
    bases = np.array(
        [np.diag([4.0, 5.0, 6.0]), [[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 6.0]]]
    )
    h = rng.integers(-6, 7, (1001, 3))
    q = h @ np.linalg.inv(bases[0]).T + rng.normal(0, 0.001, (len(h), 3))
    charts = np.broadcast_to(np.diag([1000.0, 1000.0, 100.0]), (len(q), 3, 3)).copy()
    enabled = np.arange(len(q)) % 7 != 0
    parameters = np.array([0.015, 2.0, 0.3, 3.0, 0.5])
    measure = (2 * np.pi) ** 1.5 * 2**2 * 0.3 / abs(np.linalg.det(charts))
    return q, bases, charts, enabled, measure, parameters


def call(engine, data):
    q, bases, charts, enabled, measure, p = data
    return engine.physical_fields(
        q,
        bases,
        np.linalg.inv(bases).transpose(0, 2, 1),
        abs(np.linalg.det(bases)),
        charts,
        enabled,
        measure,
        p,
    )


def test_fields_reference_worker_repeat_and_python_orchestration():
    data = fixture()
    q, bases, charts, enabled, measure, p = data
    expected = ref.physical_fields(q, bases, charts, enabled, measure, p)
    assert expected[2].sum() > 500
    previous = None
    for workers in (1, 4, 4):
        engine = _core.ExperimentalLatticeEngine(workers)
        actual = call(engine, data)
        np.testing.assert_allclose(actual[0], expected[0], atol=3e-12, rtol=2e-12)
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        if previous is not None:
            for x, y in zip(previous, actual, strict=True):
                np.testing.assert_array_equal(x, y)
        previous = actual
    settings = GrainOwnershipOptions()
    ordinary = _physical_fields(engine, q, bases, charts, enabled, GrainSearchOptions(), settings)
    native = _physical_fields(
        engine,
        q,
        bases,
        charts,
        enabled,
        GrainSearchOptions(),
        replace(settings, native_fields=True),
    )
    np.testing.assert_allclose(native[0], ordinary[0], atol=3e-12, rtol=2e-12)
    np.testing.assert_array_equal(native[1], ordinary[1])
    np.testing.assert_array_equal(native[2], ordinary[2])
    assert settings.native_fields is False
    with pytest.raises(ValueError):
        GrainOwnershipOptions(native_fields=1)


def test_strict_core_reciprocal_and_inclusive_physical_boundaries():
    q = np.array(
        [
            [1.0, 3 / 1024, 0.5 / 1024],
            [1.0, 3 / 1024, np.nextafter(0.5 / 1024, 1)],
            [1.0, np.nextafter(3 / 1024, 1), 0],
            [1.0, 1 / 32, 0],
            [1.125, 0, 0],
            [0.0, 0, 0],
        ]
    )
    bases = np.eye(3)[None]
    charts = np.broadcast_to(np.eye(3) * 1024, (len(q), 3, 3)).copy()
    data = (
        q,
        bases,
        charts,
        np.ones(len(q), bool),
        np.full(len(q), 1e-6),
        np.array([0.125, 2.0, 0.3, 3.0, 0.5]),
    )
    e, c, p = call(_core.ExperimentalLatticeEngine(4), data)
    assert p.tolist() == [[True, False, False, False, False, False]]
    assert c.tolist() == [[True, True, True, False, False, False]]
    assert e[0, 0] > 0 and np.count_nonzero(e) == 1
    off = list(data)
    off[3] = np.zeros(len(q), bool)
    off[4] = np.zeros(len(q))
    disabled = call(_core.ExperimentalLatticeEngine(1), off)
    np.testing.assert_array_equal(disabled[1], c)
    assert not disabled[0].any() and not disabled[2].any()


def test_rotating_models_matches_coherent_rotating_points_and_charts():
    q, bases, charts, enabled, measure, p = fixture()
    rot = np.linalg.qr(np.random.default_rng(791).normal(size=(3, 3)))[0]
    engine = _core.ExperimentalLatticeEngine(4)
    x = call(engine, (q @ rot, bases, charts @ rot, enabled, measure, p))
    y = call(engine, (q, bases @ rot.T, charts, enabled, measure, p))
    np.testing.assert_allclose(x[0], y[0], atol=3e-11, rtol=2e-11)
    np.testing.assert_array_equal(x[1], y[1])
    np.testing.assert_array_equal(x[2], y[2])


def test_empty_shapes_and_invalid_normalisations_are_bounded():
    engine = _core.ExperimentalLatticeEngine(1)
    q, bases, charts, enabled, measure, p = fixture()
    data = (q, bases[:0], charts, enabled, measure, p)
    assert all(x.shape == (0, len(q)) for x in call(engine, data))
    for index, value in [
        (2, np.zeros((len(q), 2, 2))),
        (3, enabled[:-1]),
        (4, measure[:-1]),
        (4, np.full(len(q), np.nan)),
        (4, np.zeros(len(q))),
        (5, np.array([1.0, 2.0, 0.3, 3.0, 0.5])),
    ]:
        bad = list(fixture())
        bad[index] = value
        with pytest.raises(ValueError):
            call(engine, bad)
    with pytest.raises(ValueError, match="normalisation"):
        engine.physical_fields(
            np.array([[1.0, 0.0, 0.0]]),
            np.eye(3)[None],
            np.eye(3)[None],
            np.array([1e308]),
            np.eye(3)[None],
            np.array([True]),
            np.array([1e308]),
            np.array([0.015, 2.0, 0.3, 3.0, 0.5]),
        )
    many = np.broadcast_to(np.eye(3), (8001, 3, 3)).copy()
    with pytest.raises(ValueError, match="budget"):
        engine.physical_fields(q, many, many, np.ones(len(many)), charts, enabled, measure, p)
