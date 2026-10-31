"""Optional relative error floor calibrated from independent equivalent pairs.

This is an empirical observation-error model, not a correction to intensity or
an uncertainty model for shared calibration/scale coefficients. No atoms enter.
"""

from dataclasses import dataclass

import numpy as np

from .empirical_scaling import _real, paired_mask

__all__ = ["EquivalentError", "estimate_equivalent_error"]


@dataclass(frozen=True)
class EquivalentError:
    relative: float
    sigma: np.ndarray
    systematic_sigma: np.ndarray
    d_sigma_d_relative: np.ndarray
    family_intensity: np.ndarray
    training_pairs: int
    training_families: int
    training_robust_width: float


def estimate_equivalent_error(
    intensity, sigma, groups, reference_mask, training_mask, *, maximum=0.5, max_pairs=1_000_000
):
    """Solve family-balanced robust pair-width=1 for a nonnegative error floor.

    var(I_i) = sigma_i^2 + (beta*mu_family)^2. ``mu`` is the nonnegative
    within-family median of measured intensities. Only declared references
    in complete training families determine beta. Unknown/missing rows should
    be excluded before this optional error-model fit; signed finite rows remain.
    """
    i = _real(intensity, "intensity", 1)
    s = _real(sigma, "sigma", 1)
    g = np.asarray(groups)
    references = paired_mask(g, reference_mask)
    train = np.asarray(training_mask)
    if (
        s.shape != i.shape
        or g.shape != i.shape
        or np.any(s <= 0)
        or train.shape != i.shape
        or train.dtype != bool
        or not np.isscalar(maximum)
        or not np.isfinite(maximum)
        or maximum <= 0
        or type(max_pairs) is not int
        or not 1 <= max_pairs <= 1_000_000
    ):
        raise ValueError("invalid equivalent-error inputs")
    families, labels = np.unique(g, return_inverse=True)
    means = np.empty(len(families))
    pairs, weights = [], []
    count = 0
    for f in range(len(families)):
        members = labels == f
        if train[members].any() and not train[members].all():
            raise ValueError("error-model training must withhold complete families")
        means[f] = max(0.0, float(np.median(i[members])))
        ids = np.flatnonzero(members & references & train)
        size = len(ids) * (len(ids) - 1) // 2
        count += size
        if count > max_pairs:
            raise ValueError("equivalent pair budget exceeded")
        if size:
            a, b = np.triu_indices(len(ids), 1)
            pairs.append(np.c_[ids[a], ids[b]])
            weights.append(np.full(size, 1 / size))
    if not pairs:
        raise ValueError("no paired equivalent training references")
    pairs, weight = np.concatenate(pairs), np.concatenate(weights)
    a, b = pairs.T
    delta, variance = i[a] - i[b], s[a] ** 2 + s[b] ** 2
    mu = means[labels[a]]

    def width(beta):
        residual = abs(delta) / np.sqrt(variance + 2 * (beta * mu) ** 2)
        order = np.argsort(residual, kind="stable")
        k = np.searchsorted(np.cumsum(weight[order]), 0.5 * weight.sum())
        return float(residual[order[min(k, len(order) - 1)]] / 0.6744897501960817)

    if width(maximum) > 1:
        raise ValueError("equivalent error exceeds the declared relative bound")
    lo, hi = 0.0, float(maximum)
    if width(0.0) <= 1:
        hi = 0.0
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            if width(mid) > 1:
                lo = mid
            else:
                hi = mid
    beta = (lo + hi) / 2
    systematic = beta * means[labels]
    total = np.hypot(s, systematic)
    derivative = beta * means[labels] ** 2 / total
    return EquivalentError(
        beta,
        total,
        systematic,
        derivative,
        means[labels],
        len(pairs),
        len(np.unique(labels[a])),
        width(beta),
    )
