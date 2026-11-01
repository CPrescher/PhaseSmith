"""Independent measured-only intensity/absence assessment equations.

Validation reference only: it neither imports native assessment nor drives search.
Exact operations are provided as plain records by a database fixture/adapter.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _laue(operations):
    rotations = set()
    for op in operations:
        r = tuple(tuple(row) for row in op["rotation"])
        rotations.add(r)
        rotations.add(tuple(tuple(-x for x in row) for row in r))
    return tuple(sorted(rotations))


def _mean(rows):
    rows = np.asarray(rows, dtype=float)
    minimum = rows[:, 1].min()
    weights = (minimum / rows[:, 1]) ** 2
    return float(np.dot(rows[:, 0], weights / weights.sum())), float(
        minimum / np.sqrt(weights.sum())
    )


def assess(observations, cell, settings):
    """Return independent Laue and shell/absence evidence for supplied measured rows."""
    source = np.asarray(observations, dtype=float)
    if source.ndim != 2 or source.shape[1] != 5 or not len(source):
        raise ValueError("observations must have shape (n,5)")
    if not np.isfinite(source).all() or (source[:, 4] <= 0).any():
        raise ValueError("finite measured rows and positive sigma required")
    hkl = source[:, :3].astype(np.int64)
    if not np.array_equal(hkl, source[:, :3]) or (np.abs(hkl) > 1024).any():
        raise ValueError("integer Miller indices required")
    if (hkl == 0).all(axis=1).any():
        raise ValueError("000 is not a diffraction observation")
    families = defaultdict(list)
    for h, row in zip(hkl, source, strict=True):
        key = min(tuple(h), tuple(-h))
        families[key].append(row[3:])
    keys = sorted(families)
    rows = np.asarray([_mean(families[h]) for h in keys])
    hkl = np.asarray(keys)
    identity = np.eye(3, dtype=int)
    friedel = tuple(sorted((tuple(map(tuple, identity)), tuple(map(tuple, -identity)))))
    sets = sorted({friedel, *(_laue(s["operations"]) for s in settings)})
    laue = []
    for rotations in sets:
        equivalent = defaultdict(list)
        for i, h in enumerate(hkl):
            equivalent[min(tuple(np.asarray(r).T @ h) for r in rotations)].append(i)
        chi, residual, signal, constraints = 0.0, 0.0, 0.0, 0
        for members in equivalent.values():
            if len(members) < 2:
                continue
            y, sigma = rows[members].T
            variance = sigma**2 + (0.25 * y) ** 2
            weights = 1 / variance
            mean = np.dot(y, weights) / weights.sum()
            chi += float(np.sum((y - mean) ** 2 / variance))
            residual += float(np.abs(y - mean).sum())
            signal += float(np.abs(y).sum())
            constraints += len(members) - 1
        reduced = chi / constraints if constraints else None
        status = (
            "friedel_only"
            if len(rotations) == 2
            else "insufficient_data"
            if constraints < 20
            else "contradicted"
            if reduced > 4
            else "supported"
        )
        laue.append(
            dict(
                rotations=rotations,
                families=len(equivalent),
                constraints=constraints,
                reduced_chi_square=reduced,
                r_int=residual / signal if signal else None,
                status=status,
            )
        )
    a, b, c, alpha, beta, gamma = np.asarray(cell, float)
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    metric = np.array(
        [
            [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
        ]
    )
    q2 = np.einsum("ni,ij,nj->n", hkl, np.linalg.inv(metric), hkl)
    order = sorted(range(len(rows)), key=lambda i: (np.rint(q2[i] / q2.max() * 1e12), i))
    shells = min(8, len(rows))
    bins = np.empty(len(rows), int)
    for rank, i in enumerate(order):
        bins[i] = rank * shells // len(rows)
    mu = np.asarray(
        [
            max(rows[bins == j, 0].mean(), rows[bins == j, 1].mean(), np.finfo(float).tiny)
            for j in range(shells)
        ]
    )
    evidence = []
    for s in settings:
        absent = np.zeros(len(rows), dtype=bool)
        phases = np.zeros(len(rows), dtype=complex)
        stabilizers = np.zeros(len(rows), dtype=int)
        for op in s["operations"]:
            r = np.asarray(op["rotation"])
            translation = np.array([n / d for n, d in op["translation"]])
            stabilizer = (hkl @ r == hkl).all(axis=1)
            phases[stabilizer] += np.exp(2j * np.pi * (hkl[stabilizer] @ translation))
            stabilizers += stabilizer
        absent = np.abs(phases) < 1e-8 * stabilizers
        y, sigma = rows.T
        ratio = 1 + (mu[bins] / sigma) ** 2
        log_bf = 0.5 * (np.log(ratio) + ((y - mu[bins]) / sigma) ** 2 / ratio - (y / sigma) ** 2)
        evidence.append(
            dict(
                laue=sets.index(_laue(s["operations"])),
                absent_families=int(absent.sum()),
                strong_violations=int((absent & (y > 5 * sigma)).sum()),
                absence_log_evidence=float(np.clip(log_bf[absent], -25, 5).sum()),
            )
        )
    return dict(measured_rows=len(source), friedel_families=len(rows), laue=laue, settings=evidence)
