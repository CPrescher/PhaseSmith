"""Independent NumPy threshold-selected atomicity and complete Fourier cycles.

No atom count enters this calculation. Dense Gaussian grids deliberately differ
from the production bounded-neighbourhood implementation.
"""

from __future__ import annotations

import itertools

import numpy as np

from .solution_dual_reference import hash64, patterson_reference, peaks_reference


def adaptive_peaks_reference(density, sigma=2.0, cap=128):
    rho = np.asarray(density)
    ids = np.arange(rho.size).reshape(rho.shape)
    selected = rho > 0
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        other = np.roll(rho, tuple(-v for v in offset), axis=(0, 1, 2))
        other_ids = np.roll(ids, tuple(-v for v in offset), axis=(0, 1, 2))
        selected &= (rho > other) | ((rho == other) & (ids < other_ids))
    heights = rho[selected]
    scale = np.max(np.abs(rho))
    rms = scale * np.sqrt(np.mean((rho / scale) ** 2)) if scale else 0
    effective = sigma
    count = np.count_nonzero(heights >= effective * rms)
    while count > cap:
        effective *= 1.5
        count = np.count_nonzero(heights >= effective * rms)
    return peaks_reference(rho, count), effective


def adaptive_projection_reference(
    density, basis, sigma, cap, mask_sigma, *, cycle=0, seed=0, omit_fraction=0.1
):
    rho = np.asarray(density)
    peaks, effective = adaptive_peaks_reference(rho, sigma, cap)
    excluded = set()
    if cycle % 3 == 2 and len(peaks) >= 2 and omit_fraction > 0:
        number = min(len(peaks) - 1, max(1, int(np.floor(omit_fraction * len(peaks)))))
        ranks = sorted(
            range(len(peaks)), key=lambda i: (hash64(seed ^ hash64(cycle) ^ hash64(i)), i)
        )
        excluded.update(ranks[:number])
    coordinates = np.moveaxis(np.indices(rho.shape), 0, -1) / rho.shape
    mask = np.zeros(rho.shape)
    for index, peak in enumerate(peaks):
        if index in excluded:
            continue
        delta = (coordinates - peak + 0.5) % 1 - 0.5
        cartesian = delta @ np.asarray(basis).T
        squared = np.einsum("...i,...i->...", cartesian, cartesian)
        gaussian = np.exp(-0.5 * squared / mask_sigma**2) / ((2 * np.pi) ** 1.5 * mask_sigma**3)
        mask += np.where(squared <= (3 * mask_sigma) ** 2 * (1 + 1e-12), gaussian, 0)
    return np.maximum(rho, 0) * mask, peaks, effective, len(excluded)


def count_free_dual_space_reference(
    shape,
    basis,
    hkl,
    amplitude,
    phases,
    support,
    *,
    cycles,
    peak_sigma=2.0,
    peak_cap=128,
    mask_sigma=0.25,
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
    selected, thresholds, omitted = [], [], []
    for cycle in range(cycles):
        projected, peaks, threshold, skipped = adaptive_projection_reference(
            rho,
            basis,
            peak_sigma,
            peak_cap,
            mask_sigma,
            cycle=cycle,
            seed=seed,
            omit_fraction=omit_fraction,
        )
        selected.append(len(peaks))
        thresholds.append(threshold)
        omitted.append(skipped)
        calculated = np.prod(shape) * np.fft.ifftn(projected)
        magnitude = np.abs(calculated[indices])
        norm = magnitude @ magnitude
        if not np.isfinite(norm) or norm <= 1e-250:
            history.append(1.0)
            rho = projected
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
        "adaptive_peaks": {
            "sigma": peak_sigma,
            "cap": peak_cap,
            "selected": selected,
            "thresholds": thresholds,
            "omitted": omitted,
        },
    }
