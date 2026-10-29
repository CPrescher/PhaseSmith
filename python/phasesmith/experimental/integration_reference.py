"""Independent pixel equations for fixed-aperture reflection measurements.

No native calls and no imports of production integration or detector equations.
Small validation inputs only: deliberately readable pixel iteration.
"""

import numpy as np


def measure_frame(
    image,
    boxes,
    *,
    mask=None,
    variance=None,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
    background_margin=4,
    min_background=12,
):
    data = np.asarray(image, float)
    excluded = np.zeros(data.shape, bool) if mask is None else np.asarray(mask)
    var = (
        np.maximum(np.maximum(data, 0) / gain + read_noise**2, 1) if variance is None else variance
    )
    coverage = np.zeros(data.shape, int)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] += 1
    result = []
    for x0, y0, x1, y1 in boxes:
        signal, background = [], []
        for y in range(max(0, y0 - background_margin), min(data.shape[0], y1 + background_margin)):
            for x in range(
                max(0, x0 - background_margin), min(data.shape[1], x1 + background_margin)
            ):
                if excluded[y, x]:
                    continue
                if x0 <= x < x1 and y0 <= y < y1:
                    signal.append((data[y, x], var[y, x]))
                elif coverage[y, x] == 0 and (saturation is None or data[y, x] < saturation):
                    background.append((data[y, x], var[y, x]))
        m, n = len(signal), len(background)
        if n < min_background or m == 0:
            result.append([np.nan, np.nan, np.nan, m, n])
        else:
            mean = sum(d for d, _ in background) / n
            intensity = sum(d for d, _ in signal) - m * mean
            error = np.sqrt(
                sum(v for _, v in signal) + (m / n) ** 2 * sum(v for _, v in background)
            )
            result.append([intensity, error, mean, m, n])
    return np.array(result).reshape(-1, 5)
