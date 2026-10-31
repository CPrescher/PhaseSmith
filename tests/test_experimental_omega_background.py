"""Differential equation, interval, noise and native-boundary contracts."""

import numpy as np
import pytest
from phasesmith import _core
from phasesmith.single_crystal.omega_background import (
    ArPLSParameters,
    arpls_masked_rows,
    arpls_rows,
    available,
)

pytestmark = pytest.mark.skipif(not available(), reason="experimental-spots build required")


@pytest.mark.parametrize("frames", [3, 5, 37, 76, 180])
@pytest.mark.parametrize("smoothness", [0.1, 300.0, 1e4, 1e6])
@pytest.mark.parametrize("solver", ["cholesky", "ldlt"])
def test_independent_reference_and_workers(frames, smoothness, solver):
    rng = np.random.default_rng(640)
    rows = rng.normal(200, 3, (8, frames))
    rows[:, frames // 2] += 100
    p = ArPLSParameters(smoothness, 1e-3, 15)
    reference = arpls_rows(rows, p, backend="scipy")
    serial = arpls_rows(rows, p, backend="native", workers=1, solver=solver)
    parallel = arpls_rows(rows, p, backend="native", workers=4, solver=solver)
    # Independent LAPACK and scalar Cholesky differ in floating-point order.
    np.testing.assert_allclose(serial[0], reference[0], rtol=2e-7, atol=2e-5)
    for a, b in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.all(serial[1] <= 15)


@pytest.mark.parametrize("solver", ["cholesky", "ldlt"])
def test_fragmented_intervals_short_runs_nonfinite_and_independent_fits(solver):
    rows = np.random.default_rng(13).normal(50, 3, (19, 76))
    mask = np.random.default_rng(14).random(rows.shape) < 0.025
    mask[0] = True
    mask[1] = False
    mask[2] = False
    mask[2, 12:25] = True
    mask[3, :] = True
    mask[3, :11] = False
    mask[4, :] = True
    mask[4, :12] = False
    rows[mask] = np.nan
    rows[3, :11] = np.inf  # Too-short samples are never fitted or validated.
    native = arpls_masked_rows(rows, mask, workers=4, solver=solver)
    reference = arpls_masked_rows(rows, mask, backend="scipy")
    np.testing.assert_allclose(native[0], reference[0], rtol=1e-9, atol=1e-6, equal_nan=True)
    for a, b in zip(native, arpls_masked_rows(rows, mask, workers=1, solver=solver), strict=True):
        np.testing.assert_array_equal(a, b)
    for i in (0, 3):
        assert np.isnan(native[0][i]).all() and native[1][i] == 0 and not native[2][i]
    assert np.isfinite(native[0][4, :12]).all()
    for start, end in [(0, 12), (25, 76)]:
        isolated = arpls_rows(rows[2:3, start:end], solver=solver)
        np.testing.assert_array_equal(native[0][2, start:end], isolated[0][0])
    assert native[1][2] == max(arpls_rows(rows[2:3, :12])[1][0], arpls_rows(rows[2:3, 25:])[1][0])
    rows[1, 10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        arpls_masked_rows(rows, mask, solver=solver)


@pytest.mark.parametrize("solver", ["cholesky", "ldlt"])
def test_signed_peaks_constant_empty_capped_and_inputs_unchanged(solver):
    x = np.arange(180)
    peaks = 1000 * np.exp(-0.5 * ((x - 80) / 1.2) ** 2)
    rows = np.vstack([peaks + 200 + 0.15 * x, np.full(180, -15.0), np.zeros(180)])
    original = rows.copy()
    baseline, _, _ = arpls_rows(rows, solver=solver)
    signal = rows[0] - baseline[0]
    window = abs(x - 80) < 6
    assert abs(signal[window].sum() / peaks[window].sum() - 1) < 0.01
    assert abs((signal[window] * x[window]).sum() / signal[window].sum() - 80) < 0.02
    assert (signal < 0).any()
    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(baseline[1:], rows[1:])
    assert arpls_rows(np.empty((0, 76)), solver=solver)[0].shape == (0, 76)
    assert arpls_masked_rows(np.empty((0, 76)), np.empty((0, 76), bool), solver=solver)[
        0
    ].shape == (0, 76)
    capped = arpls_rows(
        np.random.default_rng(1).normal(size=(8, 76)),
        ArPLSParameters(max_iterations=1),
        solver=solver,
    )
    assert np.all(capped[1] == 1) and not capped[2].all()


@pytest.mark.parametrize("workers", [0, 129, True, 1.5])
def test_invalid_workers(workers):
    with pytest.raises(ValueError):
        arpls_rows(np.ones((2, 12)), workers=workers)


@pytest.mark.parametrize(
    "rows", [np.ones(4), np.ones((2, 2)), np.ones((2, 12), complex), [[1, 2, np.inf]]]
)
def test_invalid_arrays(rows):
    with pytest.raises(ValueError):
        arpls_rows(rows)


def test_native_boundary_validates_without_python_wrapper():
    engine = _core.ExperimentalArPLSEngine(1)
    for data, mask, smoothness, minimum in [
        (np.ones((2, 2)), None, 1e4, 3),
        (np.ones((2, 12)), np.zeros((1, 12), bool), 1e4, 12),
        (np.ones((2, 12)), None, 0.0, 12),
        (np.ones((2, 12)), None, 1e4, 2),
        (np.full((2, 12), np.inf), None, 1e4, 12),
    ]:
        with pytest.raises(ValueError):
            engine.baselines(data, mask, smoothness, 0.001, 30, minimum)


@pytest.mark.parametrize("workers", [1, 10])
@pytest.mark.parametrize("solver", ["cholesky", "ldlt"])
def test_many_interval_lengths_and_repeated_parameters_remain_independent(workers, solver):
    """More workers and repeated calls must not mix interval lengths or penalties."""
    engine = _core.ExperimentalArPLSEngine(workers, solver)
    rows = np.random.default_rng(915).normal(150, 3, (49, 180))
    rows[:, 32:35] += 400
    mask = np.ones(rows.shape, bool)
    intervals = []
    for i in range(len(rows)):
        # Many different lengths plus a second disjoint interval; short leftover
        # regions remain excluded.
        runs = [(i % 5, i % 5 + 12 + i), (100, 114 + i % 29)]
        for start, end in runs:
            mask[i, start:end] = False
        intervals.append(runs)
    for smoothness in (1e-6, 1e4, 1e9, 1e4):
        p = ArPLSParameters(smoothness=smoothness)
        result = engine.baselines(rows, mask, smoothness, p.tolerance, p.max_iterations, 12)
        for i, runs in enumerate(intervals):
            counts, converged = [], []
            for start, end in runs:
                alone = arpls_rows(rows[i : i + 1, start:end], p, workers=1, solver=solver)
                np.testing.assert_array_equal(result[0][i, start:end], alone[0][0])
                counts.append(alone[1][0])
                converged.append(alone[2][0])
            assert result[1][i] == max(counts)
            assert result[2][i] == all(converged)
        assert np.isnan(result[0][mask]).all()


@pytest.mark.parametrize("frames", [3, 12, 76, 180])
@pytest.mark.parametrize("smoothness", [1e-6, 1e4, 1e9])
def test_ldlt_first_solve_satisfies_independent_dense_system(frames, smoothness):
    rows = np.random.default_rng(562).normal(200, 3, (7, frames))
    p = ArPLSParameters(smoothness=smoothness, max_iterations=1)
    baseline, counts, _ = arpls_rows(rows, p, solver="ldlt")
    differences = np.diff(np.eye(frames), n=2, axis=0)
    matrix = np.eye(frames) + smoothness * differences.T @ differences
    # Backward error remains meaningful even for the deliberately strongly
    # conditioned 1e9 penalty, without loosening any reference-fit tolerance.
    error = np.max(np.abs(baseline @ matrix.T - rows), axis=1)
    scale = np.linalg.norm(matrix, ord=np.inf) * np.max(np.abs(baseline), axis=1) + np.max(
        np.abs(rows), axis=1
    )
    assert np.all(error / scale < 5e-16)
    np.testing.assert_array_equal(counts, 1)


@pytest.mark.parametrize("solver", ["unknown", "auto", None, 1])
def test_invalid_solver_rejected(solver):
    with pytest.raises(ValueError, match="Solver"):
        arpls_rows(np.ones((2, 12)), solver=solver)
    if isinstance(solver, str):
        with pytest.raises(ValueError, match="Solver"):
            _core.ExperimentalArPLSEngine(1, solver)


@pytest.mark.parametrize("backend", ["auto", "native"])
def test_missing_native_capability_never_falls_back_implicitly(monkeypatch, backend):
    from phasesmith.single_crystal import omega_background as module

    monkeypatch.setattr(module, "available", lambda: False)
    rows = np.ones((2, 16))
    with pytest.raises(RuntimeError, match="requires the spots capability"):
        module.arpls_rows(rows, backend=backend)
    with pytest.raises(RuntimeError, match="requires the spots capability"):
        module.arpls_masked_rows(rows, np.zeros(rows.shape, dtype=bool), backend=backend)
