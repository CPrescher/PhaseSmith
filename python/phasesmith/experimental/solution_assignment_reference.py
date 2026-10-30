"""Independent NumPy enumeration of bounded fixed-geometry label alternatives."""

import itertools

import numpy as np


def assignment_neighbors_reference(columns, factors, observed, labels, maximum=4):
    """Columns include each orbit's phases and fixed site displacement attenuation.

    Return ordered (labels, objective) alternatives inside the explicit proposal
    band. This band guides relaxation; it is not a statistical confidence region.
    """
    columns = np.asarray(columns, complex)
    factors = np.asarray(factors, float)
    observed = np.asarray(observed, float)
    labels = tuple(labels)
    elements = factors.shape[1]
    if (
        columns.shape != (len(observed), len(labels))
        or factors.shape[0] != len(observed)
        or not 1 <= maximum <= 8
        or any(label is None or not 0 <= label < elements for label in labels)
        or set(labels) != set(range(elements))
    ):
        raise ValueError("complete labels and bounded maximum required")

    def objective(assignment):
        amplitude = np.abs(np.sum(columns * factors[:, assignment], axis=1))
        scale = amplitude @ observed / (amplitude @ amplitude)
        return float(np.sum((scale * amplitude - observed) ** 2) / np.sum(observed**2))

    ceiling = 1.25 * objective(labels) + 1e-4
    candidates = {}
    priorities = []
    for site in range(len(labels)):
        best = np.inf
        for element in range(elements):
            if element == labels[site]:
                continue
            proposed = list(labels)
            proposed[site] = element
            score = objective(proposed)
            best = min(best, score)
            if set(proposed) == set(labels) and score <= ceiling:
                candidates[tuple(proposed)] = score
        priorities.append((best, site))
    selected = [site for _, site in sorted(priorities)[:24]]
    for i, j in itertools.combinations(selected, 2):
        if labels[i] == labels[j]:
            continue
        proposed = list(labels)
        proposed[i], proposed[j] = proposed[j], proposed[i]
        score = objective(proposed)
        if score <= ceiling:
            candidates[tuple(proposed)] = score
    return sorted(candidates.items(), key=lambda p: (p[1], p[0]))[:maximum]
