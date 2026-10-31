"""Independent NumPy checkpoint policy and frequency-sampling conventions."""

import numpy as np


def checkpoint_reference(maps, cycles, quality, scores, ceilings, best_quality=None):
    best_quality = quality if best_quality is None else best_quality
    result = []
    for i, (a, n, q, score, ceiling) in enumerate(
        zip(maps, cycles, quality, scores, ceilings, strict=True)
    ):
        a = np.asarray(a, dtype=float)
        a = a / np.max(np.abs(a))
        cosine = None
        stable = False
        if i:
            b = np.asarray(maps[i - 1], dtype=float)
            b = b / np.max(np.abs(b))
            cosine = float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
            stable = (
                n >= 64
                and q >= 0.65
                and abs(q - quality[i - 1]) <= 0.015
                and best_quality[i] - q <= 0.015
                and abs(score - scores[i - 1]) <= 0.02 * max(scores[i - 1], 1e-12)
                and cosine >= 0.995
            )
        reason = (
            "quality_target"
            if n >= 32 and best_quality[i] >= 0.9
            else "stable_map"
            if stable
            else "cycle_ceiling"
            if ceiling
            else "continue"
        )
        result.append(
            dict(
                cycles=n,
                cfom=q,
                best_cfom=best_quality[i],
                score=score,
                map_cosine=cosine,
                selected=reason != "continue",
                reason=reason,
            )
        )
    return result


def origin_sampling_reference(hkl, amplitudes, rotations):
    """Independent family selection and nonzero-weight harmonic extents."""
    hkl, rotations, amplitudes = map(np.asarray, (hkl, rotations, amplitudes))
    indices = {tuple(row): i for i, row in enumerate(hkl)}
    remaining = set(indices)
    representatives = []
    while remaining:
        row = min(remaining)
        representatives.append(indices[row])
        orbit = np.asarray(row) @ rotations
        remaining.difference_update(map(tuple, np.concatenate((orbit, -orbit))))
    chosen = sorted(representatives, key=lambda i: (-amplitudes[i], *hkl[i]))[:2048]
    waves = [
        row @ rotation - row
        for i in chosen
        if amplitudes[i] > 0
        for row in [hkl[i]]
        for rotation in rotations
    ]
    extent = np.max(np.abs(waves), axis=0) if waves else np.zeros(3, dtype=int)
    return tuple(min(128, max(16, 2 ** int(int(2 * q).bit_length()))) for q in extent)
