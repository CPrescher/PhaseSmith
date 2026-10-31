"""Independent dense NumPy equations for measured-only phase invariants.

This reference does not import the native phase planner or execute a solver.
"""

import itertools

import numpy as np


def normalize_reference(hkl, intensity, sigma, reciprocal_basis, rotations):
    hkl = np.asarray(hkl)
    rows = np.array([i for i, h in enumerate(hkl) if tuple(h) > tuple(-h)])
    canonical = hkl[rows]
    values, uncertainty = np.asarray(intensity)[rows], np.asarray(sigma)[rows]
    epsilon = np.array([sum(np.array_equal(h @ r, h) for r in rotations) for h in canonical])
    q = np.sum((canonical @ np.asarray(reciprocal_basis).T) ** 2, axis=1)
    scaled = np.maximum(values, 0) / np.max(values) / epsilon
    bins = max(1, min(8, len(rows) // 32))
    order = sorted(range(len(rows)), key=lambda i: (q[i], i))
    centers, means = [], []
    for index in range(bins):
        subset = order[index * len(rows) // bins : (index + 1) * len(rows) // bins]
        centers.append(float(np.mean(q[subset])))
        mean = float(np.mean(scaled[subset]))
        means.append(mean if mean > np.mean(scaled) * 1e-12 else float(np.mean(scaled)))
    populations = [(b + 1) * len(rows) // bins - b * len(rows) // bins for b in range(bins)]
    merged = []
    for center, mean, population in zip(centers, means, populations, strict=True):
        if merged and abs(center - merged[-1][0]) <= 1e-12 * (1 + abs(merged[-1][0])):
            previous = merged.pop()
            total = previous[2] + population
            merged.append(
                (
                    (previous[0] * previous[2] + center * population) / total,
                    (previous[1] * previous[2] + mean * population) / total,
                    total,
                )
            )
        else:
            merged.append((center, mean, population))
    centers, means = [row[0] for row in merged], [row[1] for row in merged]
    mean = np.interp(q, centers, means)
    e = np.sqrt(scaled / mean)
    upper = np.sqrt(np.maximum(values + 2 * uncertainty, 0) / np.max(values) / epsilon / mean)
    strong = sorted(
        np.flatnonzero((e >= 1.2) & (values >= 2 * uncertainty)), key=lambda i: (-e[i], i)
    )[:256]
    return canonical, e, upper, strong, np.array(centers), np.array(means)


def small_relations_reference(hkl, e, upper, strong):
    """Brute-force small relations, independent of native pair-group discovery."""
    if len(strong) > 12:
        raise ValueError("brute-force reference limited to 12 selected variables")
    lookup = {}
    for index, h in enumerate(hkl):
        lookup[tuple(h)] = index + 1
        lookup[tuple(-h)] = -(index + 1)
    signed = sorted((h, code) for h, code in lookup.items() if abs(code) - 1 in strong)
    triplets = []
    for target in strong:
        found = []
        for (h, first), (k, second) in itertools.combinations(signed, 2):
            if len({target, abs(first) - 1, abs(second) - 1}) != 3:
                continue
            if np.array_equal(np.array(h) + k, hkl[target]):
                key = tuple(sorted((first, second)))
                found.append((e[abs(first) - 1] * e[abs(second) - 1], key))
        for weight, key in sorted(found, key=lambda item: (-item[0], item[1]))[:64]:
            triplets.append((target, key, weight))
    quartets = {}
    for indices in itertools.combinations(sorted(strong), 4):
        for signs in itertools.product((-1, 1), repeat=4):
            signed_hkl = np.array(signs)[:, None] * np.asarray(hkl)[list(indices)]
            if np.any(np.sum(signed_hkl, axis=0)):
                continue
            cross = [signed_hkl[0] + signed_hkl[j] for j in (1, 2, 3)]
            if any(tuple(h) not in lookup for h in cross):
                continue
            bounds = [upper[abs(lookup[tuple(h)]) - 1] for h in cross]
            if any(value >= 0.9 for value in bounds):
                continue
            codes = tuple(sorted(s * (i + 1) for i, s in zip(indices, signs, strict=True)))
            key = min(codes, tuple(sorted(-c for c in codes)))
            quartets[key] = np.prod(e[list(indices)]) * np.prod(1 - np.array(bounds) / 0.9)
    selected, counts = [], np.zeros(len(hkl), int)
    for codes, weight in sorted(quartets.items(), key=lambda item: (-item[1], item[0])):
        indices = np.abs(codes) - 1
        if np.any(counts[indices] >= 16):
            continue
        counts[indices] += 1
        selected.append((codes, weight))
    return triplets, selected[:4096]


def incidence_reference(metadata):
    n = len(metadata["hkl"])
    triple, quartet = [], []
    for target, partners, _ in metadata["triplets"]:
        row = np.zeros(n)
        row[target] = 1
        for index, sign in partners:
            row[index] -= sign
        triple.append(row)
    for terms, _ in metadata["quartets"]:
        row = np.zeros(n)
        for index, sign in terms:
            row[index] += sign
        quartet.append(row)
    return np.array(triple).reshape(-1, n), np.array(quartet).reshape(-1, n)


def invariant_reference(metadata, phases):
    triple, quartet = incidence_reference(metadata)
    result = {}
    for name, matrix, weights, direction in (
        ("triplet", triple, [r[2] for r in metadata["triplets"]], -1),
        ("negative_quartet", quartet, [r[1] for r in metadata["quartets"]], 1),
    ):
        angle = matrix @ phases
        weights = np.array(weights)
        if len(weights):
            weights /= np.sum(weights)
        result[name] = float(weights @ (1 + direction * np.cos(angle)))
        result[name + "_gradient"] = -direction * matrix.T @ (weights * np.sin(angle))
    return result


def uniform_reference(seed, cycle, index, lane):
    mask = 2**64 - 1
    value = seed ^ ((cycle * 0x9E3779B97F4A7C15) & mask)
    value ^= (index * 0xBF58476D1CE4E5B9) & mask
    value ^= (lane * 0x94D049BB133111EB) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return ((value >> 12) + 0.5) / 2**52


def step_reference(metadata, phases, magnitudes, cycle, seed, *, relative_tie_tolerance=0.0):
    old = np.array(phases)
    proposed = old.copy()
    strong = metadata["strong"]
    frozen = set(sorted(strong, key=lambda i: (-magnitudes[i], i))[: 2 * len(strong) // 5])
    temperature = 0.35 * 0.85**cycle if cycle < 12 else 0.0
    _, q = incidence_reference(metadata)
    weights = np.array([item[1] for item in metadata["quartets"]])
    proposals = 0
    for index in strong:
        relations = [t for t in metadata["triplets"] if t[0] == index]
        if index in frozen or not relations:
            continue
        phasor = sum(
            weight * np.exp(1j * sum(sign * old[j] for j, sign in terms))
            for _, terms, weight in relations
        )
        if abs(phasor) <= 1e-12:
            continue
        delta = min(np.pi / 3, temperature * -np.log(uniform_reference(seed, cycle, index, 0)))
        plus, minus = np.angle(phasor) + delta, np.angle(phasor) - delta
        incident = np.flatnonzero(q[:, index])
        if len(incident):

            def score(value, index=index, incident=incident):
                trial = old.copy()
                trial[index] = value
                return weights[incident] @ (1 + np.cos(q[incident] @ trial))

            plus_cost, minus_cost = score(plus), score(minus)
            tolerance = relative_tie_tolerance * (1 + abs(plus_cost) + abs(minus_cost))
            proposed[index] = plus if plus_cost <= minus_cost + tolerance else minus
        else:
            proposed[index] = minus if uniform_reference(seed, cycle, index, 1) < 0.5 else plus
        proposals += 1
    before = invariant_reference(metadata, old)["negative_quartet"]
    blend = 0.25
    for _ in range(7):
        values = old + blend * ((proposed - old + np.pi) % (2 * np.pi) - np.pi)
        after = invariant_reference(metadata, values)["negative_quartet"]
        if after <= before + 1e-12:
            return dict(
                phases=values,
                temperature=temperature,
                proposals=proposals,
                changed=int(np.count_nonzero(np.abs(values - old) > 1e-14)),
                blend=blend,
                quartet_before=before,
                quartet_after=after,
            )
        blend /= 2
    return dict(
        phases=old,
        temperature=temperature,
        proposals=proposals,
        changed=0,
        blend=0.0,
        quartet_before=before,
        quartet_after=before,
    )
