"""Independent NumPy least-squares reference for guarded plane extraction.

Small fixtures only. No production/native imports. Gradients are conditional on
the retained background samples and fixed supplied variances, not clipping events.
"""

import numpy as np


def measure(
    image,
    boxes,
    *,
    mask=None,
    variance=None,
    gap=3,
    margin=6,
    minimum=12,
    clip=4,
    gain=1,
    read_noise=0,
    saturation=None,
):
    data = np.asarray(image, float)
    mask = np.zeros(data.shape, bool) if mask is None else np.asarray(mask)
    var = (
        np.maximum(np.maximum(data, 0) / gain + read_noise**2, 1) if variance is None else variance
    )
    h, w = data.shape
    yy, xx = np.indices(data.shape)
    coverage = np.zeros(data.shape, int)
    guards = np.zeros(data.shape, bool)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] += 1
        guards[max(0, y0 - gap) : min(h, y1 + gap), max(0, x0 - gap) : min(w, x1 + gap)] = True
    results, diagnostics, gradients = [], [], []
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
        sx, sy = max((x1 - x0) / 2, 1), max((y1 - y0) / 2, 1)
        design = np.stack([np.ones_like(xx), (xx - cx) / sx, (yy - cy) / sy], axis=-1)
        signal = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1) & ~mask
        outer = (
            (xx >= x0 - gap - margin)
            & (xx < x1 + gap + margin)
            & (yy >= y0 - gap - margin)
            & (yy < y1 + gap + margin)
        )
        bg = outer & ~guards & ~mask
        if saturation is not None:
            bg &= data < saturation
        candidate = int(bg.sum())
        for iteration in range(9):
            x = design[bg]
            if len(x) < minimum or np.linalg.matrix_rank(x) < 3:
                raise ValueError("insufficient background reference fixture")
            coeff = np.linalg.lstsq(x, data[bg], rcond=None)[0]
            if iteration == 8:
                break
            residual = (data[bg] - x @ coeff) / np.sqrt(var[bg])
            centre = np.median(residual)
            scale = max(1, 1.4826 * np.median(abs(residual - centre)))
            keep = abs(residual - centre) <= clip * scale
            if keep.all():
                break
            indices = np.flatnonzero(bg)
            bg.flat[indices[~keep]] = False
        x = design[bg]
        inv = np.linalg.inv(x.T @ x)
        cov = inv @ ((x.T * var[bg]) @ x) @ inv
        q = design[signal].sum(axis=0)
        m = signal.sum()
        background = design @ coeff
        intensity = (data[signal] - background[signal]).sum()
        bvar = q @ cov @ q
        sigma = np.sqrt(var[signal].sum() + bvar)
        weights = np.maximum(data[signal] - background[signal], 0)
        weight = weights.sum()
        mx, my = (
            ((weights * xx[signal]).sum() / weight, (weights * yy[signal]).sum() / weight)
            if weight > 0
            else (cx, cy)
        )
        vx, vy = (
            (
                np.sqrt((weights * (xx[signal] - mx) ** 2).sum() / weight),
                np.sqrt((weights * (yy[signal] - my) ** 2).sum() / weight),
            )
            if weight > 0
            else (np.nan, np.nan)
        )
        edge = signal & ((xx == x0) | (xx == x1 - 1) | (yy == y0) | (yy == y1 - 1))
        eq = design[edge].sum(axis=0)
        ei = (data[edge] - background[edge]).sum()
        es = np.sqrt(var[edge].sum() + eq @ cov @ eq)
        rms = np.sqrt(np.sum(((data[bg] - x @ coeff) / np.sqrt(var[bg])) ** 2) / max(len(x) - 3, 1))
        results.append(
            [
                mx,
                my,
                intensity,
                sigma,
                q @ coeff / m,
                np.sqrt(bvar) / m,
                data[signal].max(),
                vx,
                vy,
                m,
                len(x),
            ]
        )
        diagnostics.append(
            [
                *coeff,
                *(-q),
                candidate,
                candidate - len(x),
                rms,
                ei,
                es,
                ei / intensity if intensity > 0 else np.nan,
            ]
        )
        gradient = np.zeros(data.shape)
        gradient[signal] = 1
        gradient[bg] = -(x @ inv @ q)
        gradients.append(gradient)
    return np.asarray(results), np.asarray(diagnostics), np.asarray(gradients)
