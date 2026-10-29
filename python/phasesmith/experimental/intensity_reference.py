"""Independent NumPy equations for slab and empirical intensity kernels.

No Rust calls, image readers, datasets or structure factors enter this module.
"""

from itertools import pairwise

import numpy as np


def slab_response(cosine, thickness, mu):
    c = np.asarray(cosine, float)
    tau = mu * thickness / c
    if mu * thickness == 0:
        return np.column_stack(
            [
                np.zeros_like(c),
                c,
                np.full_like(c, thickness / 2),
                np.zeros_like(c),
                np.ones_like(c),
                np.zeros_like(c),
            ]
        )
    # Series is evaluated only where needed to avoid subtraction near zero.
    eta = -np.expm1(-tau)
    relative = -np.expm1(-mu * thickness) / eta
    depth = np.empty_like(c)
    dh = np.empty_like(c)
    small = tau < 1e-3
    t = tau[small]
    depth[small] = thickness * (0.5 - t / 12 + t**3 / 720 - t**5 / 30240)
    dh[small] = thickness / c[small] * (t / 12 - t**3 / 240 + t**5 / 6048)
    t = tau[~small]
    e = np.exp(-t)
    depth[~small] = thickness * (1 / t - e / eta[~small])
    dh[~small] = thickness / c[~small] * (1 / t - t * e / eta[~small] ** 2)
    de = -tau * np.exp(-tau) / c
    return np.column_stack([eta, relative, depth, de, -relative * de / eta, dh])


def empirical_model(frames, detector, harmonics, coefficients, floor=1e-8):
    nf, nd = frames.shape[1], detector.shape[1]
    d = 1 + detector @ coefficients[nf : nf + nd]
    a = 1 + harmonics @ coefficients[nf + nd :]
    value = frames @ coefficients[:nf] + np.log(np.maximum(d, floor)) + np.log(np.maximum(a, floor))
    jacobian = np.column_stack(
        [
            frames,
            detector * ((d > floor) / np.maximum(d, floor))[:, None],
            harmonics * ((a > floor) / np.maximum(a, floor))[:, None],
        ]
    )
    return value, np.column_stack([d, a]), jacobian


def family_center(values, groups, weights):
    x = np.asarray(values, float)
    sums = np.zeros((int(np.max(groups, initial=-1)) + 1, x.shape[1]))
    totals = np.zeros(len(sums))
    np.add.at(sums, groups, weights[:, None] * x)
    np.add.at(totals, groups, weights)
    return x - sums[groups] / totals[groups, None]


def profile_precision(weights, basis, offsets, covariance):
    """Small independent dense reference; production never forms this matrix."""
    w = np.diag(weights)
    for g, (lo, hi) in enumerate(pairwise(offsets)):
        x = basis[lo:hi]
        diagonal = np.diag(weights[lo:hi])
        c = covariance[g].reshape(3, 3)
        update = np.linalg.solve(np.eye(3) + c @ (x.T @ diagonal @ x), c)
        w[lo:hi, lo:hi] -= diagonal @ x @ update @ x.T @ diagonal
    return w
