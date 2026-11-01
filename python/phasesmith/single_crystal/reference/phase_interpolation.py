"""Independent periodic trigonometric interpolation with explicit Nyquist splitting."""

from itertools import product

import numpy as np


def interpolate_phase_density(density, target):
    """Interpolate an even real 3D array; output axes must be at least the input axes."""
    values = np.asarray(density, dtype=np.float64)
    shape = np.asarray(values.shape)
    target = np.asarray(target)
    if (
        values.ndim != 3
        or target.shape != (3,)
        or target.dtype.kind not in "iu"
        or np.any(shape < 8)
        or np.any(shape > 128)
        or np.any(shape % 2)
        or np.any(target < shape)
        or np.any(target > 128)
        or np.any(target % 2)
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 1e100)
    ):
        raise ValueError("invalid phase-density interpolation input")
    if np.array_equal(shape, target):
        return values.copy()
    spectrum = np.fft.fftn(values) / values.size
    padded = np.zeros(tuple(target), dtype=complex)
    for index in np.ndindex(values.shape):
        modes = []
        for axis, i in enumerate(index):
            frequency = i if i <= shape[axis] // 2 else i - shape[axis]
            modes.append(
                [frequency, -frequency]
                if i == shape[axis] // 2 and target[axis] > shape[axis]
                else [frequency]
            )
        combinations = list(product(*modes))
        for mode in combinations:
            padded[tuple(np.asarray(mode) % target)] += spectrum[index] / len(combinations)
    return (np.fft.ifftn(padded) * padded.size).real
