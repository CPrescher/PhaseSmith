"""Bounded metric setting search, without composition or reference-cell inputs."""

from dataclasses import dataclass
from itertools import product
from math import gcd

import numpy as np

from .indexing import cell_parameters
from .prediction import _basis, _readonly


@dataclass(frozen=True)
class OrthorhombicSetting:
    transform: np.ndarray  # A_setting = T @ A_input; h_setting = h_input @ T.T
    direct_basis: np.ndarray
    centring: str
    index: int
    max_angle_deviation: float

    def __post_init__(self):
        object.__setattr__(self, "transform", _readonly(self.transform, np.int64))
        object.__setattr__(self, "direct_basis", _readonly(self.direct_basis))

    @property
    def cell(self):
        return cell_parameters(self.direct_basis)


def _centring(t, index):
    # Direct-lattice translations expressed in the candidate conventional cell.
    # An index <=4 integer sublattice has all coset orders dividing index.
    inverse = np.linalg.inv(t)
    residues = {
        tuple(np.rint(np.mod(np.asarray(v) @ inverse, 1) * index).astype(int) % index)
        for v in product(range(index), repeat=3)
    }
    expected = {
        (1, "P"): {(0, 0, 0)},
        (2, "A"): {(0, 0, 0), (0, 1, 1)},
        (2, "B"): {(0, 0, 0), (1, 0, 1)},
        (2, "C"): {(0, 0, 0), (1, 1, 0)},
        (2, "I"): {(0, 0, 0), (1, 1, 1)},
        (4, "F"): {(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)},
    }
    return next((name for (n, name), r in expected.items() if n == index and residues == r), None)


def find_orthorhombic_settings(direct_basis, *, max_angle=0.3, coefficient_limit=3):
    """Search short integer axes for P/A/B/C/I/F orthorhombic metric candidates.

    This bounded search is not a proof of Bravais symmetry or primitivity.
    Index 1, 2 and 4 conventional cells preserve the supplied lattice exactly;
    no inferred fractional subcell, missing-reflection rule or material prior.
    Axes are length ordered, right handed. Results prefer the smallest index.
    """
    a = _basis(direct_basis).copy()
    if np.linalg.det(a) <= 0:
        raise ValueError("direct basis must be right handed")
    if not np.isfinite(max_angle) or not 0 < max_angle <= 2:
        raise ValueError("max_angle must be in (0,2] degrees")
    if type(coefficient_limit) is not int or not 1 <= coefficient_limit <= 4:
        raise ValueError("coefficient_limit must be an integer in [1,4]")
    # Integer size reduction with an exact record of each row operation.
    reduction = np.eye(3, dtype=np.int64)
    for _ in range(32):
        order = np.argsort(np.sum(a * a, axis=1), kind="stable")
        a, reduction = a[order], reduction[order]
        changed = False
        for i in range(1, 3):
            for j in range(i - 1, -1, -1):
                k = int(np.rint(a[i] @ a[j] / (a[j] @ a[j])))
                if k:
                    if abs(k) > 10000 or np.max(abs(reduction)) > 100000:
                        raise ValueError("setting reduction coefficient budget exceeded")
                    a[i] -= k * a[j]
                    reduction[i] -= k * reduction[j]
                    changed = True
        if not changed:
            break
    n = coefficient_limit
    vectors = np.array(
        [
            v
            for v in product(range(-n, n + 1), repeat=3)
            if any(v) and next(x for x in v if x) > 0 and gcd(gcd(*v[:2]), v[2]) == 1
        ],
        dtype=np.int64,
    )
    axes = vectors @ a
    lengths = np.linalg.norm(axes, axis=1)
    order = np.argsort(lengths, kind="stable")
    axes, vectors, lengths = axes[order], vectors[order], lengths[order]
    cosine = axes @ axes.T / lengths[:, None] / lengths[None, :]
    adjacent = abs(cosine) <= np.sin(np.deg2rad(max_angle))
    result = []
    for i in range(len(axes)):
        for j in np.flatnonzero(adjacent[i] & (np.arange(len(axes)) > i)):
            for k in np.flatnonzero(adjacent[i] & adjacent[j] & (np.arange(len(axes)) > j)):
                t = vectors[[i, j, k]] @ reduction
                determinant = round(np.linalg.det(t))
                index = abs(determinant)
                if index not in (1, 2, 4):
                    continue
                if determinant < 0:
                    t[2] *= -1
                centring = _centring(t, index)
                if centring is None:
                    continue
                basis = t @ np.asarray(direct_basis)
                deviation = float(np.max(abs(cell_parameters(basis)[3:] - 90)))
                result.append(OrthorhombicSetting(t, basis, centring, index, deviation))
    result.sort(key=lambda s: (s.index, s.max_angle_deviation, tuple(s.transform.ravel())))
    return tuple(result)


def project_orthorhombic(basis):
    """Polar projection of normalized reciprocal rows, retaining their norms."""
    b = _basis(basis)
    lengths = np.linalg.norm(b, axis=1)
    u, _, vt = np.linalg.svd(b / lengths[:, None])
    return lengths[:, None] * (u @ vt)


def orthorhombic_chain(basis):
    """dB/d(log reciprocal lengths, local active sample rotation), shape (9,6)."""
    chain = np.zeros((3, 3, 6))
    for i in range(3):
        chain[i, :, i] = basis[i]
    for k in range(3):
        chain[:, :, 3 + k] = np.cross(np.eye(3)[k], basis)
    return chain.reshape(9, 6)


def orthorhombic_step(basis, step):
    """Positive row scaling and an exact Rodrigues rotation; keeps 90° angles."""
    vector = step[3:]
    theta = np.linalg.norm(vector)
    cross = np.array(
        [[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]]
    )
    rotation = (
        np.eye(3)
        + np.sinc(theta / np.pi) * cross
        + 0.5 * np.sinc(theta / (2 * np.pi)) ** 2 * (cross @ cross)
    )
    return np.exp(step[:3, None]) * basis @ rotation.T
