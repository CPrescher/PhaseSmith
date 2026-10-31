"""Independent readable NumPy equations for the experimental scan-scale model."""

import numpy as np


def fit(
    intensity,
    sigma,
    x,
    groups,
    training,
    *,
    minimum_snr=10.0,
    relative_error_floor=0.02,
    huber=2.5,
    max_iterations=30,
    tolerance=1e-10,
):
    intensity, sigma, x = map(lambda a: np.asarray(a, float), (intensity, sigma, x))
    groups = np.asarray(groups, int)
    training = np.asarray(training, bool)
    keep = training & (intensity / sigma > minimum_snr)
    for group in np.unique(groups):
        rows = (groups == group) & keep
        if rows.sum() < 2 or np.ptp(x[rows]) <= 1e-10:
            keep[groups == group] = False
    ids = np.flatnonzero(keep)
    xx = x[ids]
    yy = np.log(intensity[ids])
    gg = groups[ids]
    if len(ids) < 3:
        raise ValueError("insufficient references")
    var = (sigma[ids] / intensity[ids]) ** 2 + relative_error_floor**2
    base = 1 / var
    weights = base.copy()
    beta = 0.0
    converged = False
    dx = np.zeros(len(ids))
    dy = dx.copy()
    for iteration in range(max_iterations):
        for group in np.unique(gg):
            rows = gg == group
            w = weights[rows]
            dx[rows] = xx[rows] - np.average(xx[rows], weights=w)
            dy[rows] = yy[rows] - np.average(yy[rows], weights=w)
        den = np.sum(weights * dx**2)
        if den <= 1e-14 * weights.sum():
            raise ValueError("unidentifiable scan slope")
        new = np.sum(weights * dx * dy) / den
        residual = dy - new * dx
        if abs(new - beta) < tolerance:
            beta = new
            converged = True
            break
        beta = new
        if iteration + 1 < max_iterations:
            weights = base * np.minimum(1, huber / np.maximum(abs(residual) / np.sqrt(var), 1e-30))
    dof = max(1, len(ids) - len(np.unique(gg)) - 1)
    variance = max(1, np.sum(weights * residual**2) / dof) / den
    return beta, variance, ids, weights, iteration + 1, converged


def apply(intensity, sigma, omega, omega_range, coefficient, variance):
    half = (omega_range[1] - omega_range[0]) / 2
    x = (np.asarray(omega) - (omega_range[0] + half)) / half
    factor = np.exp(-coefficient * x)
    value = np.asarray(intensity) * factor
    d_beta = -x * value
    return np.column_stack(
        (
            factor,
            value,
            np.asarray(sigma) * factor,
            d_beta,
            -coefficient * value / half,
            abs(d_beta) * np.sqrt(variance),
        )
    )
