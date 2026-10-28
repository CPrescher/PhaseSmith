"""Independent readable NumPy implementation of the published dual-space equations.

Dense whole-grid Gaussian evaluation intentionally differs from the bounded native
neighbourhood loops. This module never invokes the Rust phasing or peak routines.
"""

from __future__ import annotations

import itertools

import numpy as np


def peaks_reference(density, count):
    rho = np.asarray(density)
    identifiers = np.arange(rho.size).reshape(rho.shape)
    selected = rho > 0
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        other = np.roll(rho, tuple(-v for v in offset), axis=(0, 1, 2))
        other_ids = np.roll(identifiers, tuple(-v for v in offset), axis=(0, 1, 2))
        selected &= (rho > other) | ((rho == other) & (identifiers < other_ids))
    indices = np.flatnonzero(selected)
    order = np.lexsort((indices, -rho.ravel()[indices]))[:count]
    result = []
    for index in indices[order]:
        position = np.array(np.unravel_index(index, rho.shape))
        refined = []
        for axis in range(3):
            left, right = position.copy(), position.copy()
            left[axis] -= 1
            right[axis] += 1
            a = rho[tuple(left % rho.shape)]
            b = rho[tuple(position)]
            c = rho[tuple(right % rho.shape)]
            denominator = a - 2 * b + c
            offset = np.clip(0.5 * (a - c) / denominator, -0.5, 0.5) if denominator < 0 else 0
            refined.append(((position[axis] + offset) / rho.shape[axis]) % 1)
        result.append(refined)
    return np.array(result).reshape(-1, 3)


def hash64(value):
    mask = 2**64 - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def atomic_projection_reference(
    density, basis, atom_count, sigma, *, cycle=0, seed=0, omit_fraction=0.1
):
    rho = np.asarray(density)
    count = min(2 * atom_count, 128) if cycle < 50 else atom_count
    peaks = peaks_reference(rho, count)
    excluded = set()
    if cycle % 3 == 2 and len(peaks) >= 2 and omit_fraction > 0:
        number = min(len(peaks) - 1, max(1, int(np.floor(omit_fraction * len(peaks)))))
        ranking = sorted(
            range(len(peaks)), key=lambda i: (hash64(seed ^ hash64(cycle) ^ hash64(i)), i)
        )
        excluded.update(ranking[:number])
    coordinates = np.moveaxis(np.indices(rho.shape), 0, -1) / rho.shape
    gaussian_mask = np.zeros(rho.shape)
    for index, peak in enumerate(peaks):
        if index in excluded:
            continue
        delta = (coordinates - peak + 0.5) % 1 - 0.5
        cartesian = delta @ np.asarray(basis).T
        squared = np.einsum("...i,...i->...", cartesian, cartesian)
        values = np.exp(-0.5 * squared / sigma**2) / ((2 * np.pi) ** 1.5 * sigma**3)
        gaussian_mask += np.where(squared <= (3 * sigma) ** 2 * (1 + 1e-12), values, 0)
    return np.maximum(rho, 0) * gaussian_mask


def patterson_reference(shape, hkl, amplitude, rank=0):
    coefficients = np.zeros(shape, dtype=complex)
    coefficients[tuple((hkl % shape).T)] = amplitude**2
    patterson = np.fft.fftn(coefficients).real / np.prod(shape)
    candidates = []
    for peak in peaks_reference(patterson, 128):
        vector = np.floor(peak * shape + 0.5).astype(int)
        index = np.ravel_multi_index(vector % shape, shape)
        negative = np.ravel_multi_index(-vector % shape, shape)
        if index == 0 or index > negative:
            continue
        phase = (hkl * (vector / shape)).sum(axis=1)
        if np.all(np.abs(phase - np.rint(phase)) < 1e-9):
            continue
        candidates.append(vector)
    if not candidates:
        return None, None
    vector = candidates[rank % len(candidates)]
    shifted = np.roll(patterson, tuple(vector), axis=(0, 1, 2))
    return np.maximum(np.minimum(patterson, shifted), 0), vector


def dual_space_reference(
    shape,
    basis,
    hkl,
    amplitude,
    phases,
    support,
    *,
    cycles,
    atom_count,
    sigma,
    hybrid_m=3.0,
    omit_fraction=0.1,
    seed=0,
    patterson_rank=None,
):
    hkl, amplitude = np.asarray(hkl), np.asarray(amplitude)
    amplitude = amplitude / amplitude.max()
    target = amplitude / np.sqrt(np.mean(amplitude**2))
    initial, vector = (
        patterson_reference(shape, hkl, target, patterson_rank)
        if patterson_rank is not None
        else (None, None)
    )
    coefficients = np.zeros(shape, dtype=complex)
    indices = tuple((hkl % shape).T)
    coefficients[indices] = target * np.exp(1j * phases)
    rho = np.fft.fftn(coefficients).real / np.prod(shape) if initial is None else initial
    support = np.asarray(support).reshape(shape)
    history, best, best_score, best_cycle = [], rho.copy(), np.inf, 0
    for cycle in range(cycles):
        projected = atomic_projection_reference(
            rho, basis, atom_count, sigma, cycle=cycle, seed=seed, omit_fraction=omit_fraction
        )
        calculated = np.prod(shape) * np.fft.ifftn(projected)
        magnitude = np.abs(calculated[indices])
        norm = magnitude @ magnitude
        if not np.isfinite(norm) or norm <= 1e-250:
            history.append(1.0)
            break
        scale = (target @ magnitude) / norm
        score = np.sum((scale * magnitude - target) ** 2) / (target @ target)
        history.append(score)
        if (
            (cycle % 3 != 2 or omit_fraction == 0)
            and (cycle >= 50 or cycles <= 50)
            and score < best_score
        ):
            best, best_score, best_cycle = projected.copy(), score, cycle
        calculated *= scale
        calculated[support == 0] = 0
        values = calculated[indices]
        norms = np.abs(values)
        phase = np.divide(values, norms, out=np.ones_like(values), where=norms > 0)
        calculated[indices] = phase * (hybrid_m * target - (hybrid_m - 1) * norms)
        rho = np.fft.fftn(calculated).real / np.prod(shape)
    if not np.isfinite(best_score):
        best = rho
    return {
        "density": best,
        "history": np.array(history),
        "best_cycle": best_cycle,
        "patterson_vector_grid": vector,
    }
