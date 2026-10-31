"""Independent NumPy count-selection scores; no native search calls.

A descriptive Gaussian-residual/BIC ranking with unknown common residual scale.
Symmetry-constrained parameter counts are supplied; one scale is added explicitly.
This is not a calibrated probability for correlated crystallographic residuals.
"""

import numpy as np


def count_ranking_reference(objectives, parameter_counts, families):
    """Return N log(J) + (p+1) log(N), and a stable ascending ranking."""
    j = np.asarray(objectives, dtype=np.float64)
    p = np.asarray(parameter_counts, dtype=np.float64)
    if (
        j.ndim != 1
        or p.shape != j.shape
        or not np.isfinite(j).all()
        or not np.isfinite(p).all()
        or np.any(j < 0)
        or np.any(p < 0)
        or np.any(p != np.floor(p))
        or isinstance(families, bool)
        or not isinstance(families, int)
        or families < 2
    ):
        raise ValueError("finite nonnegative objectives/counts and at least two families required")
    scores = families * np.log(np.maximum(j, 1e-300)) + (p + 1) * np.log(families)
    return scores, np.argsort(scores, kind="stable")
