"""Independent small-array reference: all-pairs graph search and direct NumPy sums."""

from __future__ import annotations

import numpy as np


def link_spots(frames, xy, segments, radius):
    """Simple explicit graph traversal; no native/sweep/union-find helpers."""
    neighbors = [[] for _ in frames]
    for j in range(len(frames)):
        for i in range(j):
            if (
                frames[j] == frames[i] + 1
                and segments[frames[j]] == segments[frames[i]]
                and np.sum((xy[j] - xy[i]) ** 2) <= radius**2
            ):
                neighbors[i].append(j)
                neighbors[j].append(i)
    labels = np.full(len(frames), -1, dtype=np.int64)
    track = 0
    for seed in range(len(frames)):
        if labels[seed] >= 0:
            continue
        queue = [seed]
        labels[seed] = track
        while queue:
            for other in neighbors[queue.pop()]:
                if labels[other] < 0:
                    labels[other] = track
                    queue.append(other)
        track += 1
    return labels


def summarize(samples):
    """Direct positive-weight position and midpoint moments."""
    good = np.isfinite(samples[:, :4]).all(axis=1)
    r = samples[good]
    w = np.maximum(r[:, 3], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xyz = (r[:, :3] * w[:, None]).sum(axis=0) / w.sum()
        rms = np.sqrt(np.sum(w * (r[:, 2] - xyz[2]) ** 2) / w.sum())
    return np.r_[xyz, rms, len(samples)]
