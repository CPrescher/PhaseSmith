"""Independent NumPy reference for background-estimation kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def smooth_bruckner(
    y: ArrayLike,
    smooth_points: int,
    iterations: int,
) -> NDArray[np.float64]:
    """Readable loop matching the pinned Cython scan and update ordering."""

    source = np.asarray(y, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("reference y must be a non-empty finite vector")
    if smooth_points < 0 or iterations < 0:
        raise ValueError("reference counts must be non-negative")
    point_count = int(smooth_points)
    result = np.empty(source.size + 2 * point_count, dtype=np.float64)
    result[:point_count] = source[0]
    result[point_count : point_count + source.size] = source
    result[point_count + source.size :] = source[-1]
    window_size = 2 * point_count + 1

    for _ in range(iterations):
        window_average = sum(result[:window_size]) / window_size
        for index in range(point_count, source.size - point_count - 2):
            if result[index] > window_average:
                old_value = result[index]
                result[index] = window_average
                window_average += (
                    (window_average - old_value)
                    + result[index + point_count + 1]
                    - result[index - point_count]
                ) / window_size
            else:
                window_average += (
                    result[index + point_count + 1] - result[index - point_count]
                ) / window_size
    return result[point_count : point_count + source.size]
