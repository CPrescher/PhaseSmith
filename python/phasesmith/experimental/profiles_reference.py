"""Independent dense NumPy profile equations for small validation fixtures only."""

import numpy as np


def evaluate(profile, bounds, step=(1.0, 1.0, 1.0), centre=(0.0, 0.0, 0.0), scales=(1.0, 1.0, 1.0)):
    scale = np.asarray(scales)
    endpoints = (np.asarray(bounds).reshape(-1, 3, 2) - np.asarray(centre)[None, :, None]) * scale[
        None, :, None
    ]
    bounds = np.sort(endpoints, axis=2).reshape(-1, 6)
    centre = np.zeros(3)
    z, y, x = np.indices(profile.shape)
    index = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    step = np.asarray(step)
    dimensions = np.array(profile.shape[::-1])
    centre = np.asarray(centre)
    lower = (index - dimensions / 2) * step + centre
    upper = lower + step
    out = []
    for b in np.asarray(bounds):
        overlap = np.maximum(np.minimum(upper, b[1::2]) - np.maximum(lower, b[::2]), 0)
        fraction = overlap / step
        derivative = ((upper < b[1::2]).astype(float) - (lower > b[::2]).astype(float)) / step
        derivative[overlap <= 0] = 0
        mass = np.sum(profile.ravel() * np.prod(fraction, axis=1))
        d = [
            np.sum(
                profile.ravel()
                * derivative[:, j]
                * np.prod(fraction[:, [k for k in range(3) if k != j]], axis=1)
            )
            for j in range(3)
        ]
        out.append([mass, *d])
    out = np.asarray(out)
    out[:, 1:] *= scale
    return out


def sample_bounds(offsets, boxes, angles):
    rows = []
    design = []
    for g, (x0, y0, x1, y1) in enumerate(boxes):
        for y in range(y0, y1):
            for x in range(x0, x1):
                rows.append([x - 0.5, x + 0.5, y - 0.5, y + 0.5, *angles[g]])
                design.append(
                    [
                        1,
                        (x - (x0 + x1 - 1) / 2) / max((x1 - x0) / 2, 1),
                        (y - (y0 + y1 - 1) / 2) / max((y1 - y0) / 2, 1),
                    ]
                )
        assert len(rows) == offsets[g + 1]
    return np.array(rows), np.array(design)


def fit(
    samples,
    valid,
    offsets,
    boxes,
    ids,
    angles,
    covariance,
    centres,
    profiles,
    assignments,
    shifts,
    step=(1.0, 1.0, 1.0),
    scales=None,
):
    scales = np.ones_like(centres) if scales is None else scales
    bounds, design = sample_bounds(offsets, boxes, angles)
    out = []
    for reflection, centre in enumerate(centres):
        model = assignments[reflection]
        centre = centre + shifts[model] / scales[reflection]
        selected = np.zeros(len(samples), bool)
        c = np.zeros((len(samples), len(samples)))
        for g in np.flatnonzero(ids == reflection):
            indices = np.arange(offsets[g], offsets[g + 1])
            selected[indices] = valid[indices]
            x = design[indices]
            c[np.ix_(indices, indices)] = (
                np.diag(samples[indices, 1]) + x @ covariance[g].reshape(3, 3) @ x.T
            )
        selected = np.flatnonzero(selected)
        v = evaluate(profiles[model], bounds[selected], step, centre, scales[reflection])
        foreground = v[:, 0] > 0
        selected = selected[foreground]
        v = v[foreground]
        d = samples[selected, 0]
        p = v[:, 0]
        c = c[np.ix_(selected, selected)]
        wp = np.linalg.solve(c, p)
        wd = np.linalg.solve(c, d)
        amp = p @ wd / (p @ wp)
        sigma = 1 / np.sqrt(p @ wp)
        r = d - amp * p
        wr = np.linalg.solve(c, r)
        chi = r @ wr / max(len(r) - 1, 1)
        gradient = np.r_[-2 * p @ wr, -2 * amp * (v[:, 1:].T @ wr)]
        out.append([amp, sigma, p.sum(), chi, len(r), np.corrcoef(p, d)[0, 1], *gradient])
    return np.asarray(out)


def learn_one(
    samples,
    valid,
    offsets,
    boxes,
    ids,
    angles,
    centres,
    references,
    shape,
    step,
    iterations,
    scales=None,
    reference_weights=None,
):
    """Dense projected least squares for one bank, without native training gates/cutoff."""
    scales = np.ones_like(centres) if scales is None else scales
    bounds, _ = sample_bounds(offsets, boxes, angles)
    voxels = np.prod(shape)
    operator = []
    observed = []
    shifts = []
    rowweights = []
    reference_weights = np.ones(len(centres)) if reference_weights is None else reference_weights
    for r in np.flatnonzero(references):
        groups = np.flatnonzero(ids == r)
        indices = np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in groups])
        assert valid[indices].all()
        total = samples[indices, 0].sum()
        xyz = (bounds[indices, ::2] + bounds[indices, 1::2]) / 2
        centroid = np.sum(samples[indices, 0, None] * xyz, axis=0) / total
        centroid[2] = centres[r, 2]
        shifts.append((centroid - centres[r]) * scales[r])
        z, y, x = np.indices(shape[::-1])
        index = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        lo = (index - np.array(shape) / 2) * step
        hi = lo + step
        for i in indices:
            ends = (bounds[i].reshape(3, 2) - centroid[:, None]) * scales[r, :, None]
            transformed = np.sort(ends, axis=1).ravel()
            fraction = (
                np.maximum(np.minimum(hi, transformed[1::2]) - np.maximum(lo, transformed[::2]), 0)
                / step
            )
            operator.append(np.prod(fraction, axis=1))
            observed.append(samples[i, 0] / total)
            rowweights.append(reference_weights[r])
    a = np.asarray(operator).reshape(-1, voxels)
    d = np.asarray(observed)
    root = np.sqrt(rowweights)
    a = a * root[:, None]
    d = d * root
    denominator = a.T @ a.sum(axis=1)
    p = np.divide(a.T @ d, denominator, out=np.zeros(voxels), where=denominator > 0)
    p = np.maximum(p, 0)
    for _ in range(iterations):
        gradient = a.T @ (d - a @ p)
        p = np.maximum(
            p + np.divide(0.9 * gradient, denominator, out=np.zeros(voxels), where=denominator > 0),
            0,
        )
    # Independently identify the face-connected positive component of the peak.
    volume = p.reshape(shape[::-1])
    seed = np.unravel_index(np.argmax(volume), volume.shape)
    keep = np.zeros(volume.shape, bool)
    pending = [seed]
    while pending:
        coord = pending.pop()
        if keep[coord] or volume[coord] <= 0:
            continue
        keep[coord] = True
        for axis in range(3):
            for direction in (-1, 1):
                neighbour = list(coord)
                neighbour[axis] += direction
                if 0 <= neighbour[axis] < volume.shape[axis]:
                    pending.append(tuple(neighbour))
    p[~keep.ravel()] = 0
    return p / p.sum(), np.median(shifts, axis=0)
