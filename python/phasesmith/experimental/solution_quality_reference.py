"""Independent NumPy Wilson-shell normalization and phase-map quality equations.

No native normalization, phasing, geometry or statistic evaluator is called.
Rotation matrices and element scattering factors are supplied explicitly.
"""

import numpy as np


def reciprocal_s_squared(cell, hkl):
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sg = np.sin(np.deg2rad(gamma))
    cx, cy = c * cb, c * (ca - cb * cg) / sg
    direct = np.array(
        [[a, b * cg, cx], [0, b * sg, cy], [0, 0, np.sqrt(c * c - cx * cx - cy * cy)]]
    )
    reciprocal = np.asarray(hkl) @ np.linalg.inv(direct)
    return (reciprocal * reciprocal).sum(axis=1) / 4


def wilson_reference(hkl, intensity, cell, rotations, form_squares):
    """Signed independent-family shell means, Wilson line and regularized profile."""
    h = np.asarray(hkl, dtype=np.int64)
    intensity = np.asarray(intensity, dtype=float)
    rotations = np.asarray(rotations, dtype=np.int64)
    c = np.asarray(form_squares, dtype=float)
    x = reciprocal_s_squared(cell, h)
    location = {tuple(row): i for i, row in enumerate(h)}
    remaining = set(location)
    representatives = []
    while remaining:
        key = min(remaining)
        representatives.append(location[key])
        orbit = np.asarray(key) @ rotations
        remaining.difference_update(map(tuple, np.concatenate([orbit, -orbit])))
    eps = np.sum(np.all(np.einsum("hi,gij->hgj", h, rotations) == h[:, None], axis=-1), axis=1)
    maximum = np.abs(intensity).max()
    reduced = intensity / maximum / (eps * c)
    # Stable reciprocal-shell membership across equivalent metric arithmetic.
    keys = np.floor(x / x.max() * 1e12 + 0.5)
    ordered = sorted(representatives, key=lambda i: (keys[i], *h[i]))
    bins = min(max(len(ordered) // 80, 2), 20, len(ordered))
    groups = [
        ordered[b * len(ordered) // bins : (b + 1) * len(ordered) // bins] for b in range(bins)
    ]
    sx = np.array([np.mean(x[rows]) for rows in groups])
    means = np.array([np.mean(reduced[rows]) for rows in groups])
    counts = np.array([len(rows) for rows in groups])
    combined = []
    for position, mean, count in zip(sx, means, counts, strict=True):
        if combined and abs(combined[-1][0] - position) <= x.max() * 1e-12:
            px, pm, pn = combined.pop()
            combined.append(
                (
                    (px * pn + position * count) / (pn + count),
                    (pm * pn + mean * count) / (pn + count),
                    pn + count,
                )
            )
        else:
            combined.append((position, mean, count))
    sx, means, counts = map(np.asarray, zip(*combined, strict=True))
    bins = len(sx)
    good = means > 0
    if not good.any():
        raise ValueError("no positive signed shell means")
    mx = np.average(sx[good], weights=counts[good])
    my = np.average(np.log(means[good]), weights=counts[good])
    variance = np.sum(counts[good] * (sx[good] - mx) ** 2)
    slope = (
        np.sum(counts[good] * (sx[good] - mx) * (np.log(means[good]) - my)) / variance
        if variance > 1e-30
        else 0.0
    )
    intercept = my - slope * mx
    residual = np.zeros(bins)
    residual[good] = (
        (np.log(means[good]) - intercept - slope * sx[good]) * counts[good] / (counts[good] + 80)
    )
    profile = np.interp(x, sx, residual)
    log_scale = intercept + np.log(maximum)
    expected = eps * c * np.exp(log_scale + slope * x + profile)
    mean_e = np.mean(np.maximum(intensity[representatives], 0) / expected[representatives])
    expected *= mean_e
    return {
        "expected_intensity": expected,
        "e_values": np.sqrt(np.maximum(intensity, 0) / expected),
        "epsilon": eps,
        "representatives": representatives,
        "log_scale": log_scale + np.log(mean_e),
        "slope": slope,
        "shell_x": sx,
        "shell_means": means * maximum,
        "shell_residuals": residual,
    }


def phase_quality_reference(target, calculated, e_weights, weak, allowed):
    """CC-Rweak and analytic derivatives of the complete scale-invariant statistic."""
    target, c, weights = map(np.asarray, (target, calculated, e_weights))
    weak = np.asarray(weak, dtype=bool)
    centered_t = target[:allowed] - target[:allowed].mean()
    centered_c = c[:allowed] - c[:allowed].mean()
    vt, vc = centered_t @ centered_t, centered_c @ centered_c
    defined = vt > 1e-28 and vc / np.max(c) ** 2 > 1e-28
    dcc = np.zeros_like(c)
    if defined:
        cc = centered_t @ centered_c / np.sqrt(vt * vc)
        dcc[:allowed] = centered_t / np.sqrt(vt * vc) - cc * centered_c / vc
    else:
        cc = 0.0
    total = np.mean(c * c * weights)
    rw = np.mean((c * c * weights)[weak]) / total
    dr = 2 * c * weights * (weak.astype(float) / weak.sum() - rw / len(c)) / total
    return {"correlation": cc, "r_weak": rw, "cfom": cc - rw, "gradient": dcc - dr}
