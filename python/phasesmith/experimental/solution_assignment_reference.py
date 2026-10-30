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


def ambiguous_assignments_reference(
    phases, factors, observed, labels, site_u, s_squared, atomic_numbers, multiplicities, maximum=4
):
    """Independent exhaustive bounded one/two-edit proposals with explicit attenuation.

    No native evaluator or derivatives are used. Proposal bands and count-vector
    diversity are deterministic heuristics, not model probabilities.
    """
    phases, factors = np.asarray(phases), np.asarray(factors)
    observed = np.asarray(observed)
    site_u, s_squared = np.asarray(site_u), np.asarray(s_squared)
    labels = tuple(labels)
    atomic_numbers = np.asarray(atomic_numbers)
    multiplicities = np.asarray(multiplicities)
    width = factors.shape[1]
    if (
        phases.shape != (len(observed), len(labels))
        or factors.shape[0] != len(observed)
        or site_u.shape != (len(labels),)
        or s_squared.shape != observed.shape
        or not np.all(np.isfinite(site_u))
        or np.any(site_u < 1e-5)
        or np.any(site_u > 0.2)
        or not 1 <= maximum <= 8
        or set(labels) != set(range(width))
    ):
        raise ValueError("complete bounded assignments required")
    beta = 8 * np.pi**2

    def objective(assignment, displacement):
        columns = (
            phases * factors[:, assignment] * np.exp(-beta * s_squared[:, None] * displacement)
        )
        amplitude = np.abs(columns.sum(axis=1))
        denominator = amplitude @ amplitude
        if denominator <= 1e-250:
            return np.inf
        scale = amplitude @ observed / denominator
        return float(np.sum((scale * amplitude - observed) ** 2) / np.sum(observed**2))

    ceiling = 1.25 * objective(labels, site_u) + 1e-4
    s2 = np.sum(s_squared * observed**2) / np.sum(observed**2)
    anchor = int(np.argmin(np.abs(s_squared - s2)))
    singles = []
    candidates = []

    def retain(next_labels, displacement, value, edits):
        if np.isfinite(value) and value <= ceiling and set(next_labels) == set(labels):
            previous = next((p for p in candidates if p[0] == tuple(next_labels)), None)
            if previous is not None:
                if previous[2] <= value:
                    return
                candidates.remove(previous)
            candidates.append((tuple(next_labels), tuple(displacement), value, edits))
            candidates.sort(key=lambda p: (p[2], p[0]))
            del candidates[32:]

    for site, old in enumerate(labels):
        for element in range(width):
            if element == old or abs(int(atomic_numbers[element]) - int(atomic_numbers[old])) > 2:
                continue
            ratio = factors[anchor, element] / factors[anchor, old]
            shifted = (
                float(
                    np.clip(
                        site_u[site] + np.clip(np.log(ratio) / (beta * s2), -0.02, 0.02), 1e-5, 0.2
                    )
                )
                if ratio > 0 and np.isfinite(ratio) and s2 > 0
                else site_u[site]
            )
            new_labels = list(labels)
            new_labels[site] = element
            for u in dict.fromkeys((site_u[site], shifted)):
                displacement = site_u.copy()
                displacement[site] = u
                value = objective(new_labels, displacement)
                retain(new_labels, displacement, value, 1)
                singles.append((value, site, element, u))
                singles.sort()
                del singles[12:]
    for a, b in itertools.combinations(singles, 2):
        if a[1] == b[1]:
            continue
        new_labels = list(labels)
        new_labels[a[1]] = a[2]
        new_labels[b[1]] = b[2]
        displacement = site_u.copy()
        displacement[a[1]] = a[3]
        displacement[b[1]] = b[3]
        retain(new_labels, displacement, objective(new_labels, displacement), 2)
    selected, signatures = [], set()
    for p in candidates:
        signature = tuple(
            sum(m for e, m in zip(p[0], multiplicities, strict=True) if e == i)
            for i in range(width)
        )
        if signature not in signatures and len(selected) < maximum:
            signatures.add(signature)
            selected.append(p)
    for p in candidates:
        if len(selected) == maximum:
            break
        if not any(q[0] == p[0] for q in selected):
            selected.append(p)
    return sorted(selected, key=lambda p: (p[2], p[0]))
