"""Independent measured-phase origin equations, with no native evaluator calls."""

import numpy as np


def peak_completion_reference(
    points, labels, hkl, rotations, translations, factors, s_squared, observed, u_iso
):
    """Orbit phase values/jets and profiled amplitude objective, independently in NumPy."""
    hkl, factors = np.asarray(hkl), np.asarray(factors)
    total = np.zeros(len(hkl), dtype=complex)
    derivatives = np.zeros((len(points), 3, len(hkl)), dtype=complex)
    for i, (point, label) in enumerate(zip(points, labels, strict=True)):
        if label < 0:
            continue
        images, jacobians = [], []
        for rotation, translation in zip(rotations, translations, strict=True):
            image = (rotation @ point + translation) % 1
            if not any(np.linalg.norm((image - p + 0.5) % 1 - 0.5) < 1e-8 for p in images):
                images.append(image)
                jacobians.append(rotation)
        phases = np.exp(2j * np.pi * (hkl @ np.asarray(images).T))
        total += phases.sum(axis=1) * factors[:, label]
        for a in range(3):
            wave = np.asarray([hkl @ r[:, a] for r in jacobians]).T
            derivatives[i, a] = (2j * np.pi * wave * phases).sum(axis=1) * factors[:, label]
    attenuation = np.exp(-8 * np.pi**2 * u_iso * np.asarray(s_squared))
    magnitude = np.abs(total)
    amplitude = magnitude * attenuation
    k = amplitude @ observed / (amplitude @ amplitude)
    residual = k * amplitude - observed
    norm = np.asarray(observed) @ observed
    direction = np.divide(
        total.conj(), magnitude, out=np.zeros_like(total), where=magnitude > 1e-100
    )
    gradient = (
        2
        * k
        / norm
        * np.einsum(
            "n,ian->ia", residual, (direction[None, None, :] * derivatives).real * attenuation
        )
    )
    return {"objective": residual @ residual / norm, "gradient": gradient}


def phase_origin_reference(hkl, amplitudes, coefficients, rotations, translations, shift, hand=1):
    hkl = np.asarray(hkl, dtype=int)
    rotations, translations = np.asarray(rotations), np.asarray(translations)
    locations = {tuple(h): i for i, h in enumerate(hkl)}
    remaining = set(locations)
    reps = []
    while remaining:
        h = min(remaining)
        reps.append(locations[h])
        orbit = np.asarray(h) @ rotations
        remaining.difference_update(map(tuple, np.concatenate([orbit, -orbit])))
    reps = sorted(reps, key=lambda i: (-amplitudes[i], *hkl[i]))[:2048]
    chosen_ops = {}
    for rotation, translation in zip(rotations, translations, strict=True):
        chosen_ops.setdefault(tuple(rotation.ravel()), (rotation, translation))
    qs, angles, weights = [], [], []
    threshold = np.max(np.abs(coefficients)) * 1e-12
    for i in reps:
        rows = [(hkl[i] @ r, t) for r, t in chosen_ops.values() if np.any(hkl[i] @ r != hkl[i])]
        for mapped, translation in rows:
            a = coefficients[locations[tuple(hand * hkl[i])]]
            b = coefficients[locations[tuple(hand * mapped)]]
            if abs(a) <= threshold or abs(b) <= threshold or amplitudes[i] == 0:
                continue
            q = mapped - hkl[i]
            qs.append(q)
            angles.append(
                np.angle(b) - np.angle(a) + 2 * np.pi * (hkl[i] @ translation + q @ shift)
            )
            weights.append((amplitudes[i] / np.max(amplitudes)) ** 2 / len(rows))
    if not weights:
        return dict(
            chord=0.0,
            alpha=0.0,
            gradient=np.zeros(3),
            hessian=np.zeros((3, 3)),
            alpha_gradient=np.zeros(3),
        )
    q, weights = np.asarray(qs), np.asarray(weights)
    eta = np.angle(np.exp(1j * np.asarray(angles)))
    w = weights / weights.sum()
    return {
        "chord": w @ (1 - np.cos(eta)),
        "alpha": 3 / np.pi**2 * (w @ eta**2),
        "gradient": 2 * np.pi * ((w * np.sin(eta)) @ q),
        "hessian": (2 * np.pi) ** 2 * np.einsum("n,na,nb->ab", w * np.cos(eta), q, q),
        "alpha_gradient": 12 / np.pi * ((w * eta) @ q),
    }
