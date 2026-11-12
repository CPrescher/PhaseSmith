"""Independent NumPy reference for experimental FFT deposition and ownership.

Readable equations, no native calls. Float32 deposition order is observable.
"""

from itertools import product

import numpy as np


def integer_lattice_index(indices):
    """Independent index: gcd of all exact 3x3 minors; zero means rank <3.

    This small-array reference uses Python integers and the determinant formula,
    not Euclidean row reduction or native code. A full-rank row lattice is Z^3
    exactly when that gcd is one.
    """
    from itertools import combinations
    from math import gcd

    index = 0
    for rows in combinations(indices, 3):
        a, b, c = ([int(x) for x in row] for row in rows)
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        index = gcd(index, abs(determinant))
        if index == 1:
            break
    return index


def fourier_newton(points, initial, radius):
    """Independent NumPy implementation of the bounded 20-pass ascent.

    Uses the readable array Fourier equations, not a native evaluator or the
    production small-matrix solver. No early convergence tolerance is applied.
    """
    from .reciprocal_reference import lattice_fourier

    vectors = np.array(initial, dtype=float, copy=True)
    moments = lattice_fourier(points, vectors)
    for _ in range(20):
        h = -moments[:, 4:].reshape(-1, 3, 3)
        h += np.maximum(0.0, 0.01 - np.linalg.eigvalsh(h)[:, 0])[:, None, None] * np.eye(3)
        delta = np.linalg.solve(h, moments[:, 1:4, None])[:, :, 0]
        delta *= np.minimum(1, 0.3 / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        improved = np.zeros(len(vectors), bool)
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(vectors + scale * delta, initial - radius, initial + radius)
            candidate = lattice_fourier(points, trial)
            better = (candidate[:, 0] > moments[:, 0]) & ~improved
            vectors[better] = trial[better]
            moments[better] = candidate[better]
            improved |= better
        if not improved.any():
            break
    return vectors, moments


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


def ensemble_evidence(points, models, sigma):
    """Independent positional/counting evidence relative to a uniform cloud.

    Nearest rounded lattice location, excluding 000. The Gaussian factor is
    divided by reciprocal-lattice point density (direct cell volume). This is
    a proposal-ranking heuristic; it does not fit or integrate intensities.
    """
    evidence = []
    errors = []
    for a in models:
        h = np.rint(np.asarray(points) @ np.asarray(a).T)
        predicted = np.linalg.solve(np.asarray(a), h.T).T
        d2 = np.sum((points - predicted) ** 2, axis=1)
        d2[~np.any(h != 0, axis=1)] = np.inf
        normalisation = abs(np.linalg.det(a)) * (2 * np.pi * sigma**2) ** 1.5
        evidence.append(np.log1p(np.exp(-d2 / (2 * sigma**2)) / normalisation))
        errors.append(np.sqrt(d2))
    return np.asarray(evidence), np.asarray(errors)


def modular_kernel_residues(normal, prime):
    """Independent exhaustive residue definition of a homogeneous congruence.

    No matrix construction, normal canonicalization or native code is shared
    with the production proposal enumeration. Intended for small-prime tests.
    """
    return np.asarray(
        [
            r
            for r in product(range(prime), repeat=3)
            if sum(int(a) * int(b) for a, b in zip(r, normal, strict=True)) % prime == 0
        ],
        dtype=np.int64,
    )


def observation_normal(
    points, hkl, charts, reciprocal, scales, enabled, huber=2.5, background=None
):
    """Independent dense reference for fixed-HKL observation-space fitting.

    Huber norm loss, or negative Gaussian-plus-frozen-background log likelihood.
    The returned normal is a positive Gauss-Newton approximation, not the Hessian.
    """
    residual = np.einsum("nij,nj->ni", charts, points - hkl @ reciprocal) / scales
    design = (
        -np.einsum("na,nub->nuab", hkl, charts).reshape(-1, 3, 9)
        / np.asarray(scales)[None, :, None]
    )
    r, d = residual[enabled], design[enabled]
    norm = np.linalg.norm(r, axis=1)
    if background is None:
        weight = np.minimum(1, huber / np.maximum(norm, 1e-300))
        cost = np.where(norm <= huber, norm**2 / 2, huber * (norm - huber / 2)).sum()
    else:
        likelihood = np.exp(-(norm**2) / 2)
        b = np.asarray(background)[enabled]
        weight = likelihood / (b + likelihood)
        cost = np.log1p((1 - likelihood) / (b + likelihood)).sum()
    return (
        np.einsum("n,nua,nub->ab", weight, d, d),
        np.einsum("n,nua,nu->a", weight, d, r),
        float(cost),
        int(np.sum(enabled)),
    )


def candidate_counts(points, bases, charts, enabled, tolerances, xy_gate, omega_gate):
    """Independent array equations for three shared proposal score cutoffs.

    Rounding is ties-to-even, 000 is excluded, q radii are strict, physical
    gates are inclusive. Counts and error sums precede any model selection.
    """
    out = []
    for basis in bases:
        h = np.rint(points @ basis.T)
        delta = points - np.linalg.solve(basis, h.T).T
        error = np.linalg.norm(delta, axis=1)
        residual = np.einsum("nij,nj->ni", charts, delta)
        physical = (
            enabled
            & (np.linalg.norm(residual[:, :2], axis=1) <= xy_gate)
            & (abs(residual[:, 2]) <= omega_gate)
        )
        row = []
        for tolerance in tolerances:
            keep = (error < tolerance) & np.any(h != 0, axis=1)
            row.extend(
                (int(keep.sum()), float(np.sum(error[keep] ** 2)), int((keep & physical).sum()))
            )
        out.append(row)
    return np.asarray(out).reshape(len(bases), 9)


def phase_triplets(points, vectors, options):
    """Readable unbatched phase-intersection ranking, independent of Rust bits."""
    from itertools import combinations

    phase = points @ vectors.T
    masks = abs(phase - np.rint(phase)) < options.fractional_tolerance
    pairs = sorted(
        combinations(range(len(vectors)), 2),
        key=lambda ids: (-int(np.count_nonzero(masks[:, ids[0]] & masks[:, ids[1]])), ids),
    )[: options.max_pairs]
    seen, records = set(), []
    for i, j in pairs:
        for k in range(len(vectors)):
            ids = tuple(sorted((i, j, k)))
            if len(set(ids)) != 3 or ids in seen:
                continue
            seen.add(ids)
            a = vectors[list(ids)]
            length = np.linalg.norm(a, axis=1)
            volume = abs(float(np.linalg.det(a)))
            if not (
                np.isfinite(a).all()
                and length.min() >= options.min_length
                and length.max() <= options.max_length
                and options.min_volume <= volume <= options.max_volume
                and np.linalg.cond(a) <= options.max_condition
            ):
                continue
            records.append((int(np.all(masks[:, list(ids)], axis=1).sum()), volume, ids))
    return sorted(records, key=lambda row: (-row[0], row[1], row[2]))


def physical_fields(points, bases, charts, enabled, measure, parameters):
    """Independent array equations for bounded positional hypothesis evidence.

    Each nonzero nearest-HKL residual has a strict reciprocal cutoff, an
    inclusive physical cylinder, and a Gaussian/uniform density ratio. The
    narrow core remains defined even at a chart-disabled observation.
    """
    tolerance, xy_scale, omega_scale, xy_gate, omega_gate = parameters
    evidence, cores, physical = [], [], []
    for basis in bases:
        h = np.rint(points @ basis.T)
        delta = points - np.linalg.solve(basis, h.T).T
        error = np.linalg.norm(delta, axis=1)
        nonzero = np.any(h != 0, axis=1)
        r = np.einsum("nij,nj->ni", charts, delta)
        good = (
            enabled
            & nonzero
            & (error < tolerance)
            & (np.linalg.norm(r[:, :2], axis=1) <= xy_gate)
            & (abs(r[:, 2]) <= omega_gate)
        )
        d2 = np.sum((r[:, :2] / xy_scale) ** 2, axis=1) + (r[:, 2] / omega_scale) ** 2
        row = np.zeros(len(points))
        normalisation = abs(np.linalg.det(basis)) * measure[good]
        row[good] = np.logaddexp(0, -0.5 * d2[good] - np.log(normalisation))
        evidence.append(row)
        cores.append(nonzero & (error < tolerance / 4))
        physical.append(good)
    shape = (len(bases), len(points))
    return (
        np.asarray(evidence).reshape(shape),
        np.asarray(cores, bool).reshape(shape),
        np.asarray(physical, bool).reshape(shape),
    )


def fitting_selection(points, rows, reciprocal, tolerance):
    """Independent ordered nearest-HKL equations and strict selected rows.

    Explicit component arithmetic matches the positional convention without
    invoking a native scorer, BLAS dot, or any production selection helper.
    """
    q = np.asarray(points)
    h = np.column_stack(
        [np.rint(q[:, 0] * row[0] + q[:, 1] * row[1] + q[:, 2] * row[2]) for row in rows]
    )
    predicted = np.column_stack(
        [
            h[:, 0] * reciprocal[0, j] + h[:, 1] * reciprocal[1, j] + h[:, 2] * reciprocal[2, j]
            for j in range(3)
        ]
    )
    delta = q - predicted
    error = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    ids = np.flatnonzero((error < tolerance) & np.any(h != 0, axis=1))
    return h[ids], q[ids], ids, error[ids]
