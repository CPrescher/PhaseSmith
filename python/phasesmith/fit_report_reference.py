"""Independent NumPy residual diagnostic reference (never calls Rust)."""

from __future__ import annotations

import numpy as np


def residual_diagnostics_reference(x, residual, weighted, included, region_count):
    """Return mathematical evidence for validated, finite input arrays."""
    edges = np.linspace(x[0], x[-1], region_count + 1)
    selected = x[included]
    squares = weighted[included] ** 2
    counts = np.histogram(selected, bins=edges)[0]
    sums = np.histogram(selected, bins=edges, weights=squares)[0]
    pairs = included[1:] & included[:-1]
    total = np.dot(weighted[included], weighted[included])
    difference = np.diff(weighted)[pairs]
    return {
        "count": int(np.sum(included)),
        "pairs": int(np.sum(pairs)),
        "chi_square": float(total),
        "mean": float(np.mean(residual[included])),
        "weighted_rms": float(np.sqrt(np.mean(squares))),
        "durbin_watson": None
        if total == 0 or not np.any(pairs)
        else float(difference @ difference / total),
        "region_counts": counts,
        "region_chi_square": sums,
    }
