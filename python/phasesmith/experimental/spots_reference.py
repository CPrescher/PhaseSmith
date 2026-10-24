"""Independent, deliberately direct NumPy reference for detector spot equations.

No summed-area tables or native calls. Input/result contracts are shared with
the public adapter, while thresholding, connectivity and measurement arithmetic
are implemented independently here. Intended for small validation images.
"""

from __future__ import annotations

import numpy as np

from .spots import _DEFAULT_OPTIONS, SpotFlag, SpotTable, _boxes, _inputs


def _prepared(image, mask, variance, gain, read_noise, saturation):
    data, excluded, var, gain, read_noise, saturation = _inputs(
        image, mask, variance, gain, read_noise, saturation
    )
    if var is None:
        var = np.maximum(np.maximum(data, 0) / gain + read_noise**2, 1)
    saturated = np.zeros(data.shape, dtype=bool) if saturation is None else data >= saturation
    return data, excluded, var, saturated


def _expand(box, radius, shape):
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - radius),
        max(0, y0 - radius),
        min(shape[1], x1 + radius),
        min(shape[0], y1 + radius),
    )


def detect_spots(
    image,
    *,
    mask=None,
    variance=None,
    options=_DEFAULT_OPTIONS,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
):
    data, excluded, var, saturated = _prepared(image, mask, variance, gain, read_noise, saturation)
    previous = np.zeros(data.shape, dtype=bool)
    for _ in range(2):
        strong = np.zeros(data.shape, dtype=bool)
        for y, x in np.ndindex(data.shape):
            if excluded[y, x]:
                continue
            x0, y0, x1, y1 = _expand((x, y, x + 1, y + 1), options.outer_radius, data.shape)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            select = ~excluded[y0:y1, x0:x1] & ~saturated[y0:y1, x0:x1]
            select &= ~previous[y0:y1, x0:x1]
            select &= (abs(xx - x) > options.inner_radius) | (abs(yy - y) > options.inner_radius)
            count = select.sum()
            if count < options.min_background:
                continue
            bg = data[y0:y1, x0:x1][select].mean()
            bg_var = var[y0:y1, x0:x1][select].sum() / count**2
            excess = data[y, x] - bg
            strong[y, x] = (
                excess > options.threshold_absolute
                and excess > options.threshold_sigma * np.sqrt(var[y, x] + bg_var)
            )
        previous = strong.copy()
    boxes, large, rejected = [], [], 0
    for y, x in np.ndindex(data.shape):
        if not strong[y, x]:
            continue
        todo, component = [(y, x)], []
        strong[y, x] = False
        while todo:
            row, col = todo.pop()
            component.append((row, col))
            for row2 in range(max(0, row - 1), min(data.shape[0], row + 2)):
                for col2 in range(max(0, col - 1), min(data.shape[1], col + 2)):
                    if strong[row2, col2]:
                        strong[row2, col2] = False
                        todo.append((row2, col2))
        if len(component) < options.min_pixels:
            rejected += 1
            continue
        if len(boxes) >= options.max_spots:
            raise ValueError("max_spots exceeded")
        rows, cols = np.array(component).T
        box = cols.min(), rows.min(), cols.max() + 1, rows.max() + 1
        boxes.append(_expand(box, options.padding, data.shape))
        large.append(int(SpotFlag.LARGE) if len(component) > options.max_pixels else 0)
    result = extract_spots(
        image,
        np.array(boxes, dtype=np.int64).reshape(-1, 4),
        mask=mask,
        variance=variance,
        options=options,
        gain=gain,
        read_noise=read_noise,
        saturation=saturation,
    )
    return SpotTable(
        result.boxes,
        result.values,
        result.flags | np.array(large, dtype=np.uint32),
        rejected,
        result.noise_model,
    )


def extract_spots(
    image,
    boxes,
    *,
    mask=None,
    variance=None,
    options=_DEFAULT_OPTIONS,
    gain=1.0,
    read_noise=0.0,
    saturation=None,
):
    data, excluded, var, saturated = _prepared(image, mask, variance, gain, read_noise, saturation)
    boxes = _boxes(boxes)
    if len(boxes) > options.max_spots:
        raise ValueError("max_spots exceeded")
    cover = np.zeros(data.shape, dtype=np.uint32)
    for x0, y0, x1, y1 in boxes:
        if not (0 <= x0 < x1 <= data.shape[1] and 0 <= y0 < y1 <= data.shape[0]):
            raise ValueError("invalid box bounds")
        cover[y0:y1, x0:x1] += 1
    values, flags = [], []
    for box in boxes:
        x0, y0, x1, y1 = box
        flag = (
            int(SpotFlag.EDGE)
            if (x0 == 0 or y0 == 0 or x1 == data.shape[1] or y1 == data.shape[0])
            else 0
        )
        a, b, c, d = _expand(box, options.background_margin, data.shape)
        bg_select = (cover[b:d, a:c] == 0) & ~excluded[b:d, a:c] & ~saturated[b:d, a:c]
        nb = bg_select.sum()
        if nb >= options.min_background:
            bg = data[b:d, a:c][bg_select].mean()
            bg_var = var[b:d, a:c][bg_select].sum() / nb**2
        else:
            flag |= SpotFlag.BACKGROUND
            bg, bg_var = np.nan, np.nan
        sig_select = ~excluded[y0:y1, x0:x1]
        if not sig_select.all():
            flag |= SpotFlag.MASKED
        if np.any(saturated[y0:y1, x0:x1] & sig_select):
            flag |= SpotFlag.SATURATED
        if np.any(cover[y0:y1, x0:x1] > 1):
            flag |= SpotFlag.OVERLAP
        signal = data[y0:y1, x0:x1][sig_select]
        ns = len(signal)
        intensity = signal.sum() - ns * bg if ns else np.nan
        sigma = np.sqrt(var[y0:y1, x0:x1][sig_select].sum() + ns**2 * bg_var) if ns else np.nan
        yy, xx = np.mgrid[y0:y1, x0:x1]
        weights = np.where(sig_select, np.maximum(data[y0:y1, x0:x1] - bg, 0), 0)
        if np.isfinite(bg) and weights.sum() > 0:
            cx, cy = (weights * xx).sum() / weights.sum(), (weights * yy).sum() / weights.sum()
            wx = np.sqrt((weights * (xx - cx) ** 2).sum() / weights.sum())
            wy = np.sqrt((weights * (yy - cy) ** 2).sum() / weights.sum())
        else:
            flag |= SpotFlag.NO_CENTROID
            cx, cy, wx, wy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2, np.nan, np.nan
        values.append(
            [
                cx,
                cy,
                intensity,
                sigma,
                bg,
                np.sqrt(bg_var),
                signal.max() if ns else np.nan,
                wx,
                wy,
                ns,
                nb,
            ]
        )
        flags.append(int(flag))
    return SpotTable(
        boxes,
        np.array(values).reshape(-1, 11),
        np.array(flags, dtype=np.uint32),
        noise_model="supplied_independent_variance"
        if variance is not None
        else "poisson_plus_read_noise_with_variance_floor",
    )
