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


def joint_assignment_beam_reference(
    phases,
    factors,
    observed,
    labels,
    bounds,
    s_squared,
    atomic_numbers,
    multiplicities,
    maximum_atoms=128,
    maximum=2,
):
    phases, factors, observed = np.asarray(phases), np.asarray(factors), np.asarray(observed)
    bounds, s_squared = np.asarray(bounds), np.asarray(s_squared)
    numbers, multiplicities = np.asarray(atomic_numbers), np.asarray(multiplicities)
    labels = tuple(None if x == -1 else int(x) for x in labels)
    n, width = len(observed), factors.shape[1]
    if (
        phases.shape != (n, len(labels))
        or bounds.shape != (len(labels), 2)
        or s_squared.shape != observed.shape
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > 0)
        or np.any(bounds[:, 1] < 0)
        or np.any(bounds[:, 0] < -0.2)
        or np.any(bounds[:, 1] > 0.2)
        or not 1 <= maximum <= 4
        or set(x for x in labels if x is not None) != set(range(width))
    ):
        raise ValueError("invalid categorical repair inputs")
    beta = 8 * np.pi**2
    anchor_s2 = np.sum(s_squared * observed**2) / np.sum(observed**2)
    anchor = np.argmin(np.abs(s_squared - anchor_s2))

    def column(i, label, shift):
        return (
            np.zeros(n, complex)
            if label is None
            else phases[:, i] * factors[:, label] * np.exp(-beta * shift * s_squared)
        )

    def total(state):
        return sum(
            (column(i, label, state["shifts"][i]) for i, label in enumerate(state["labels"])),
            np.zeros(n, complex),
        )

    def objective(value):
        a = np.abs(value)
        if a @ a <= 1e-250:
            return np.inf
        scale = (a @ observed) / (a @ a)
        return float(np.sum((scale * a - observed) ** 2) / np.sum(observed**2))

    def shift_profile(rest, i, label, shift):
        part = column(i, label, shift)
        f = rest + part
        a = np.abs(f)
        da = np.divide(
            -beta * s_squared * np.real(np.conj(f) * part),
            a,
            out=np.zeros_like(a),
            where=a > 1e-100,
        )
        square = a @ a
        if square <= 1e-250:
            return np.inf, 0.0, 0.0
        scale = (a @ observed) / square
        ds = (da @ observed - 2 * scale * (a @ da)) / square
        r = scale * a - observed
        dr = scale * da + ds * a
        norm = observed @ observed
        return float((r @ r) / norm), float(2 * (r @ dr) / norm), float(2 * (dr @ dr) / norm)

    def rank(value, assignment):
        return n * np.log(max(value, 1e-300)) + (
            4 * sum(x is not None for x in assignment) + 2
        ) * np.log(n)

    def order(state):
        return state["score"], tuple(-1 if e is None else e for e in state["labels"])

    parent = dict(labels=labels, shifts=np.zeros(len(labels)))
    parent["objective"] = objective(total(parent))
    parent["score"] = rank(parent["objective"], labels)
    beam = [parent]
    visited = {labels}
    retained = []
    evaluations = depths = 0
    for _ in range(min(8, max(len(labels), 1))):
        proposed = {}
        for state in beam:
            current = total(state)
            for i, old in enumerate(state["labels"]):
                original = column(i, old, state["shifts"][i])
                for new in [None, *range(width)]:
                    if new == old or (
                        old is not None and new is not None and abs(numbers[old] - numbers[new]) > 2
                    ):
                        continue
                    assignment = list(state["labels"])
                    assignment[i] = new
                    assignment = tuple(assignment)
                    if (
                        set(e for e in assignment if e is not None) != set(range(width))
                        or sum(
                            m
                            for m, e in zip(multiplicities, assignment, strict=False)
                            if e is not None
                        )
                        > maximum_atoms
                        or assignment in visited
                    ):
                        continue
                    delta = 0.0
                    if old is not None and new is not None and anchor_s2 > 0:
                        ratio = factors[anchor, new] / factors[anchor, old]
                        if ratio > 0:
                            delta = np.clip(np.log(ratio) / (beta * anchor_s2), -0.02, 0.02)
                    shifted = np.clip(state["shifts"][i] + delta, *bounds[i])
                    rest = current - original
                    polished = (
                        shifted
                        if not np.any(np.abs(beta * shifted * s_squared) > 600)
                        else state["shifts"][i]
                    )
                    for _ in range(3):
                        value, gradient, curvature = shift_profile(rest, i, new, polished)
                        evaluations += 1
                        if curvature <= 1e-20:
                            break
                        step = np.clip(-gradient / curvature, -0.01, 0.01)
                        if abs(step) < 1e-8:
                            break
                        accepted = False
                        for factor in [1.0, 0.5, 0.25]:
                            trial = np.clip(polished + factor * step, *bounds[i])
                            if np.any(np.abs(beta * trial * s_squared) > 600):
                                continue
                            cost, _, _ = shift_profile(rest, i, new, trial)
                            evaluations += 1
                            if cost < value - 1e-14:
                                polished = trial
                                accepted = True
                                break
                        if not accepted:
                            break
                    for variant, shift in enumerate([state["shifts"][i], shifted, polished]):
                        if (variant and shift == state["shifts"][i]) or (
                            variant == 2 and shift == shifted
                        ):
                            continue
                        if np.any(np.abs(beta * shift * s_squared) > 600):
                            continue
                        value = objective(current - original + column(i, new, shift))
                        evaluations += 1
                        if not np.isfinite(value) or value > 1.25 * parent["objective"] + 1e-4:
                            continue
                        candidate = dict(
                            labels=assignment,
                            shifts=state["shifts"].copy(),
                            objective=value,
                            score=rank(value, assignment),
                        )
                        candidate["shifts"][i] = shift
                        if (
                            assignment not in proposed
                            or candidate["score"] < proposed[assignment]["score"]
                        ):
                            proposed[assignment] = candidate
        if not proposed:
            break
        beam = sorted(proposed.values(), key=order)[:4]
        depths += 1
        for state in beam:
            visited.add(state["labels"])
            if state["score"] <= parent["score"] + 6.0:
                retained.append(state)
        retained = sorted(retained, key=order)[:32]
    selected = []
    signatures = set()
    for state in retained:
        signature = tuple(
            sum(m for m, label in zip(multiplicities, state["labels"], strict=False) if label == e)
            for e in range(width)
        )
        if signature not in signatures:
            selected.append(state)
            signatures.add(signature)
        if len(selected) == maximum:
            break
    return dict(candidates=selected, evaluations=evaluations, depths=depths)


def coherent_assignment_reference(
    phases, factors, observed, labels, shifts, s_squared, atomic_numbers, multiplicities
):
    """Independent coordinate sweeps with a fixed externally proposed tensor shift."""
    labels = list(labels)
    original = labels.copy()
    n = len(observed)
    width = factors.shape[1]
    columns = phases * np.exp(-8 * np.pi**2 * np.asarray(s_squared)[:, None] * np.asarray(shifts))

    def objective(assignment):
        total = sum(
            (columns[:, i] * factors[:, e] for i, e in enumerate(assignment) if e >= 0),
            np.zeros(n, complex),
        )
        amplitude = np.abs(total)
        square = amplitude @ amplitude
        if square <= 1e-250:
            return np.inf
        scale = (amplitude @ observed) / square
        return float(np.sum((scale * amplitude - observed) ** 2) / np.sum(observed**2))

    value = objective(labels)
    evaluations = depths = 0
    for _ in range(4):
        changed = False
        for i, old in enumerate(labels):
            if old < 0:
                continue
            best = None
            for new in range(width):
                if new == old or abs(atomic_numbers[old] - atomic_numbers[new]) > 2:
                    continue
                proposed = labels.copy()
                proposed[i] = new
                if set(e for e in proposed if e >= 0) != set(range(width)):
                    continue
                cost = objective(proposed)
                evaluations += 1
                if cost < value - 1e-12 and (best is None or cost < best[0]):
                    best = (cost, proposed)
            if best is not None:
                value, labels = best
                changed = True
        depths += 1
        if not changed:
            break
    return dict(
        labels=labels,
        objective=value,
        evaluations=evaluations,
        depths=depths,
        changed=labels != original,
    )
