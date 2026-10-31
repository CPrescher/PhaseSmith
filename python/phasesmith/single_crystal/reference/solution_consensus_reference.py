"""Independent exhaustive small-model agreement; no native calls."""

from collections import Counter
from itertools import permutations, product

import numpy as np


def model_agreement_distance(cell_basis, a_species, a_xyz, b_species, b_xyz):
    """Search same-element bijections and an eigenvalue-bounded image lattice.

    This reference is limited to seven atoms because permutation enumeration is
    factorial. The Rust production matcher has its separate 128-atom cap.
    """
    a = np.asarray(a_xyz, dtype=float)
    b = np.asarray(b_xyz, dtype=float)
    basis = np.asarray(cell_basis, dtype=float)
    if (
        a.shape != (len(a_species), 3)
        or b.shape != (len(b_species), 3)
        or not 1 <= len(a) <= 7
        or not 1 <= len(b) <= 7
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or basis.shape != (3, 3)
        or not np.isfinite(basis).all()
    ):
        raise ValueError("invalid small reference model")
    if Counter(a_species) != Counter(b_species):
        return None
    metric = basis.T @ basis
    eigen_min = np.linalg.eigvalsh(metric)[0]
    if eigen_min <= 0:
        raise ValueError("cell metric must be positive definite")
    counts = Counter(a_species)
    anchor_species = min(counts, key=lambda s: (counts[s], s))
    anchor = a_species.index(anchor_species)
    bijections = [
        p
        for p in permutations(range(len(b)))
        if all(a_species[i] == b_species[p[i]] for i in range(len(a)))
    ]
    for hand in (1.0, -1.0):
        for j, species in enumerate(b_species):
            if species != anchor_species:
                continue
            shift = b[j] - hand * a[anchor]
            delta = (hand * a + shift)[:, None, :] - b[None, :, :]
            delta -= np.rint(delta)
            upper_squared = np.einsum("...i,ij,...j->...", delta, metric, delta)
            # lambda_min*||delta+image||^2 bounds the metric norm below.
            # An improving image must lie inside this finite integer cube.
            radius = int(np.ceil(np.sqrt(upper_squared.max() / eigen_min) + 0.5))
            images = np.array(list(product(range(-radius, radius + 1), repeat=3)))
            vectors = delta[:, :, None, :] + images
            squared = np.einsum("...i,ij,...j->...", vectors, metric, vectors)
            distances = np.sqrt(np.maximum(0, squared.min(axis=2)))
            for permutation in bijections:
                if permutation[anchor] != j:
                    continue
                worst = max(distances[i, permutation[i]] for i in range(len(a)))
                if worst <= 0.02:
                    return float(worst)
    return None
