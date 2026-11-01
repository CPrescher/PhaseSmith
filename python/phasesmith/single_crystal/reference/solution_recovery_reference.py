"""Independent recovery-policy algebra; no production/native calls."""

from fractions import Fraction

import numpy as np


def pullback_tensors(full, rotations, source_indices):
    """Mean Q.T U_image Q in one Cartesian site basis, U in angstrom squared."""
    tensors = np.asarray(full, dtype=float)
    q = np.asarray(rotations, dtype=float)
    indices = np.asarray(source_indices)
    if (
        tensors.ndim != 3
        or tensors.shape[1:] != (3, 3)
        or q.shape != (len(indices), 3, 3)
        or len(indices) == 0
        or indices.dtype.kind not in "iu"
        or (indices < 0).any()
        or (indices >= len(tensors)).any()
        or not np.isfinite(tensors).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("invalid full tensor/rotation/source topology")
    return np.mean([r.T @ tensors[i] @ r for r, i in zip(q, indices, strict=True)], axis=0)


def model_score(objective, independent_families, parameters):
    """Existing normalized-amplitude BIC-like score with actual tensor parameters."""
    if (
        not np.isfinite(objective)
        or objective < 0
        or independent_families <= parameters
        or parameters < 1
    ):
        raise ValueError("invalid/underdetermined candidate score")
    return independent_families * np.log(max(objective, 1e-300)) + parameters * np.log(
        independent_families
    )


def symmorphic_intensity_group(operations):
    """Exact lattice-coset test for an already validated closed operation set."""
    ops = [
        (np.asarray(op["rotation"]), tuple(Fraction(n, d) % 1 for n, d in op["translation"]))
        for op in operations
    ]
    translations = {t for r, t in ops if np.array_equal(r, np.eye(3, dtype=int))}
    return bool(ops) and all(t in translations for _, t in ops)


def align_stored_frame(points, hand, shift):
    """Coordinate convention only; group bijection must still be checked separately."""
    xyz = np.asarray(points, dtype=float)
    origin = np.asarray(shift, dtype=float)
    if (
        hand not in (-1, 1)
        or xyz.ndim != 2
        or xyz.shape[1:] != (3,)
        or not 1 <= len(xyz) <= 128
        or origin.shape != (3,)
        or not np.isfinite(xyz).all()
        or not np.isfinite(origin).all()
    ):
        raise ValueError("invalid source frame")
    return np.mod(hand * xyz + np.mod(origin, 1), 1)


def consensus_limit(target, native_policy=False):
    """Requested target can tighten the existing 0.1 stopping ceiling."""
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("invalid requested target")
    return min(0.1, target) if native_policy else 0.1
