"""Prototype numerical checks; no frozen data or production tests are changed."""

from itertools import product

import numpy as np
import pytest
from metal_fft import MetalFFT, radial_intervals
from scipy.ndimage import maximum_filter


@pytest.mark.parametrize("n", [16, 32, 48, 64, 128, 192, 256])
def test_deposition_fft_and_peak_detector_against_independent_arrays(n):
    rng = np.random.default_rng(81231 + n)
    q = rng.uniform(-1.5, 1.5, (301, 3))
    step = 3.01 / (n - 4)
    output = MetalFFT().windowed_maxima(q, n, step, 1.5, 15, 100, debug=True)
    vectors, values, total, grid, field, axes = output
    expected = np.zeros((n, n, n), np.float32)
    position = q / step
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    for offset in product((0, 1), repeat=3):
        index = (lower + offset) % n
        weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1).astype(np.float32)
        np.add.at(expected, tuple(index.T), weight)
    # Ordered GPU deposition must reproduce every float32 voxel bit.
    np.testing.assert_array_equal(grid.view(np.uint32), expected.view(np.uint32))
    reference = np.fft.rfftn(expected).real.astype(np.float32)
    selected = reference[
        axes[:, None, None], axes[None, :, None], np.arange(field.shape[2])[None, None, :]
    ]
    assert np.max(np.abs(selected - field)) / len(q) <= 5e-7
    # Detect peaks independently on the ACTUAL GPU field, isolating FFT rounding
    # from peak topology, strict-neighbor comparison, ties and total/index order.
    scores = np.full(reference.shape, -np.inf, dtype=np.float32)
    scores[axes[:, None, None], axes[None, :, None], np.arange(field.shape[2])[None, None, :]] = (
        field
    )
    frequency = np.arange(n, dtype=float)
    frequency[n // 2 :] -= n
    frequency /= n * step
    length = (frequency[:, None, None] ** 2 + frequency[None, :, None] ** 2) + (
        np.arange(n // 2 + 1)[None, None, :] / (n * step)
    ) ** 2
    scores[(length < 1.5**2) | (length > 15**2)] = -np.inf
    local = maximum_filter(scores, size=3, mode=("wrap", "wrap", "nearest"))
    ids = np.flatnonzero(np.isfinite(scores) & (scores >= local))
    assert len(ids) == total
    ids = ids[np.lexsort((ids, -scores.reshape(-1)[ids]))[:100]]
    nz = n // 2 + 1
    expected_vectors = np.column_stack(
        (frequency[ids // (n * nz)], frequency[(ids // nz) % n], (ids % nz) / (n * step))
    )
    np.testing.assert_allclose(vectors, expected_vectors, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(values, scores.reshape(-1)[ids])


def test_radial_bounds_match_float64_inclusive_domain_exactly():
    for n in [16, 48, 192, 256]:
        step = 0.13
        ds = 1 / (n * step)
        freq = np.arange(n, dtype=float)
        freq[n // 2 :] -= n
        freq *= ds
        lengths = (freq[:, None, None] ** 2 + freq[None, :, None] ** 2) + (
            np.arange(n // 2 + 1)[None, None, :] * ds
        ) ** 2
        for lo, hi in [
            (ds, 3 * ds),
            (np.sqrt(lengths[1, 2, 3]), np.sqrt(lengths[4, 3, 2])),
            (40.0, 41.0),
            (1.0, 1.0),
        ]:
            lower, upper = radial_intervals(n, step, lo, hi)
            z = np.arange(n // 2 + 1)[None, :]
            actual = (z >= lower[:, None]) & (z < upper[:, None])
            np.testing.assert_array_equal(
                actual.reshape(lengths.shape), (lengths >= lo * lo) & (lengths <= hi * hi)
            )


def test_plateaus_duplicates_empty_annulus_and_repeats():
    from phasesmith import _core

    gpu = MetalFFT()
    cpu = _core.ExperimentalLatticeEngine(4)
    for q in [np.zeros((3, 3)), np.array([[0.0, 0.0, 0.0], [0.125, 0.0, 0.125]])]:
        for bounds in [(0.5, 4.0), (1.0, 40.0), (40.0, 41.0), (1.0, 1.0)]:
            a = gpu.windowed_maxima(q, 16, 0.25, *bounds, 97)
            b = cpu.windowed_maxima(q, 16, 0.25, *bounds, 97)
            for x, y in zip(a[:2], b[:2], strict=True):
                np.testing.assert_array_equal(x, y)
            assert a[2] == b[2]
            repeat = gpu.windowed_maxima(q, 16, 0.25, *bounds, 97)
            for x, y in zip(a, repeat, strict=True):
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "q,n,step,lo,hi,limit",
    [
        (np.empty((0, 3)), 32, 0.1, 1.0, 3.0, 80),
        (np.zeros((2, 2)), 32, 0.1, 1.0, 3.0, 80),
        (np.full((1, 3), np.nan), 32, 0.1, 1.0, 3.0, 80),
        (np.zeros((1, 3)), 15, 0.1, 1.0, 3.0, 80),
        (np.zeros((1, 3)), 258, 0.1, 1.0, 3.0, 80),
        (np.zeros((1, 3)), 32, 0.0, 1.0, 3.0, 80),
        (np.ones((1, 3)), 32, 0.001, 1.0, 3.0, 80),
        (np.zeros((1, 3)), 32, 0.1, 0.0, 3.0, 80),
        (np.zeros((1, 3)), 32, 0.1, 3.0, 1.0, 80),
        (np.zeros((1, 3)), 32, 0.1, 1.0, np.inf, 80),
        (np.zeros((1, 3)), 32, 0.1, 1.0, 3.0, 0),
        (np.zeros((1, 3)), 32, 0.1, 1.0, 3.0, 1001),
    ],
)
def test_invalid_inputs(q, n, step, lo, hi, limit):
    with pytest.raises(ValueError):
        MetalFFT().windowed_maxima(q, n, step, lo, hi, limit)
