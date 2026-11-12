"""Exact CPU grain-search reuse: primitive-index proof and frozen fit inputs."""

from dataclasses import replace

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal import grains as G
from phasesmith.single_crystal.ownership import (
    GrainOwnershipOptions,
    _physical_fields,
    _physical_measure,
)
from phasesmith.single_crystal.reference.grains_reference import integer_lattice_index


def test_primitive_shortcut_matches_independent_exact_minor_index():
    rng = np.random.default_rng(120126)
    cases = [
        np.eye(3, dtype=np.int64),
        np.zeros((8, 3), dtype=np.int64),
        np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6], [1, 1, 1]], dtype=np.int64),
    ]
    for _ in range(180):
        h = rng.integers(-8, 9, (8, 3), dtype=np.int64)
        if rng.random() < 0.5:
            h = h @ np.diag(rng.integers(1, 5, 3))
        cases.append(h)
    for h in cases:
        full = G.integer_lattice_basis(h)
        quick = G.integer_lattice_basis(h, _stop_if_primitive=True)
        expected = integer_lattice_index(h)
        if not expected:
            assert quick is None and full is None
        elif expected == 1:
            np.testing.assert_array_equal(quick, np.eye(3, dtype=np.int64))
            assert round(abs(np.linalg.det(full))) == 1
        else:
            np.testing.assert_array_equal(quick, full)
            assert round(abs(np.linalg.det(quick))) == expected
    # Public/default representation still performs the complete row operations.
    h = np.array([[1, 37, -29], [0, 1, 41], [0, 0, 1], [5, -8, 13]], dtype=np.int64)
    full = G.integer_lattice_basis(h)
    assert full is not None and round(np.linalg.det(full)) == 1
    np.testing.assert_array_equal(G.integer_lattice_basis(h), full)
    # Validate the whole input before a primitive prefix can terminate work.
    invalid = np.vstack([np.eye(3, dtype=np.int64), [[1000001, 0, 0]]])
    for early in (False, True):
        with pytest.raises(ValueError):
            G.integer_lattice_basis(invalid, _stop_if_primitive=early)


@pytest.mark.parametrize("change", [None, "h_signed_zero", "q_signed_zero"])
def test_fit_reuse_keeps_cycles_and_distinguishes_float_bits(monkeypatch, change):
    h = np.vstack([np.eye(3), -np.eye(3), 2 * np.eye(3), -2 * np.eye(3)])
    reciprocal = np.diag([0.25, 0.2, 1 / 6])
    q = h @ reciprocal

    class Cloud:
        def __init__(self):
            self.calls = []

        def selected(self, a, b, tolerance):
            self.calls.append(tolerance)
            hh = h.copy()
            qq = q.copy()
            if len(self.calls) >= 2:
                if change == "h_signed_zero":
                    hh[0, 1] = -0.0
                if change == "q_signed_zero":
                    qq[0, 1] = -0.0
            return hh, qq, np.arange(len(h)), np.zeros(len(h))

    original = np.linalg.lstsq
    solves = []

    def solve(*args, **kwargs):
        solves.append((args[0].tobytes(), args[1].tobytes()))
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", solve)
    options = G.GrainSearchOptions()
    cloud = Cloud()
    result = G._refine(None, q, np.diag([4.0, 5.0, 6.0]), options, integer_cores=(), _cloud=cloud)
    assert result is not None
    assert cloud.calls == [
        2 * options.reciprocal_tolerance,
        2 * options.reciprocal_tolerance,
        options.reciprocal_tolerance,
        options.reciprocal_tolerance,
    ]
    assert len(solves) == (1 if change is None else 2)
    # Reuse never survives a fit call, even for identical inputs.
    before = len(solves)
    again = G._refine(None, q, np.diag([4.0, 5.0, 6.0]), options, integer_cores=(), _cloud=Cloud())
    assert len(solves) - before == (1 if change is None else 2)
    np.testing.assert_array_equal(result.view(np.uint8), again.view(np.uint8))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.skipif(
    not hasattr(_core, "ExperimentalLatticeEngine"), reason="single-crystal feature not built"
)
def test_frozen_chart_measure_reuse_preserves_all_field_bits(native):
    rng = np.random.default_rng(120127)
    q = rng.uniform(-1, 1, (301, 3))
    models = [
        np.diag([4.0, 5.0, 6.0]),
        np.array([[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 6.0]]),
    ]
    charts = rng.normal(size=(len(q), 3, 3)) * 100
    charts[0] = 0
    valid = np.arange(len(q)) % 7 != 0
    options = GrainOwnershipOptions(native_fields=native)
    engine = _core.ExperimentalLatticeEngine(2)
    search = G.GrainSearchOptions()
    measure = _physical_measure(charts, valid, options)
    expected = _physical_fields(engine, q, models, charts, valid, search, options)
    actual = _physical_fields(engine, q, models, charts, valid, search, options, _measure=measure)
    for x, y in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))
    # New charts or scales require their own calculation, with no cross-call cache.
    other = replace(options, xy_scale=3.0)
    assert not np.array_equal(measure, _physical_measure(charts, valid, other))
    assert not np.array_equal(measure, _physical_measure(charts * 2, valid, options))
