"""Independent NumPy reference for experimental FFT deposition and ownership.

Readable equations, no native calls. Float32 deposition order is observable.
"""

from itertools import product

import numpy as np


def fourier_field(points, n, step):
    grid = np.zeros((n, n, n), dtype=np.float32)
    position = np.asarray(points) / step
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    for offset in product((0, 1), repeat=3):
        index = (lower + offset) % n
        weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1).astype(np.float32)
        np.add.at(grid, tuple(index.T), weight)
    return np.fft.rfftn(grid).real.astype(np.float32)


def assign(points, models, tolerance, ratio):
    if not len(models):
        return np.zeros(len(points), dtype=np.int32), 0, 0
    distances = []
    for rows in models:
        h = np.rint(points @ rows.T)
        error = np.linalg.norm(points - h @ np.linalg.inv(rows).T, axis=1)
        error[~np.any(h != 0, axis=1)] = np.inf
        distances.append(error)
    distances = np.column_stack(distances)
    order = np.argsort(distances, axis=1, kind="stable")
    best = distances[np.arange(len(points)), order[:, 0]]
    inside = best < tolerance
    ambiguous = np.zeros(len(points), dtype=bool)
    if len(models) > 1:
        second = distances[np.arange(len(points)), order[:, 1]]
        ambiguous = inside & ((best > ratio * second) | ((best == 0) & (second == 0)))
    labels = np.where(inside & ~ambiguous, order[:, 0] + 1, 0).astype(np.int32)
    return labels, int(ambiguous.sum()), int(((distances < tolerance).sum(axis=1) > 1).sum())
