"""Explicit detector/sample geometry, without material or unit-cell priors."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from phasesmith import _core


def triples(values, name):
    a = np.asarray(values)
    if a.dtype.kind not in "fiu" or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be real arrays of shape (n,3)")
    a = np.array(a, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


class PixelCoordinateMap(Protocol):
    """Application-owned coordinates around the native affine plane.

    Batch methods consume/return finite float (n,2) arrays. inverse_jacobian
    returns d(image xy)/d(plane xy), shape (n,2,2), evaluated at plane xy.
    Boundary and nonunique-inverse conventions belong to the map's provenance.
    """

    def to_plane(self, xy: np.ndarray) -> np.ndarray: ...
    def from_plane(self, xy: np.ndarray) -> np.ndarray: ...
    def inverse_jacobian(self, xy: np.ndarray) -> np.ndarray: ...


def mapped_xy(pixel_map, xy, *, inverse=False):
    result = np.asarray((pixel_map.from_plane if inverse else pixel_map.to_plane)(xy))
    if result.dtype.kind not in "fiu" or result.shape != xy.shape or not np.isfinite(result).all():
        raise ValueError("pixel coordinate map must return finite real (n,2) coordinates")
    return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True)
class PanelGeometry:
    """Five lab vectors: pixel(0,0) centre, fast/slow pixel steps (m), beam, axis.

    All use one right-handed basis. Omega is a right-handed rotation about
    axis, in degrees. Wavelength is in angstroms; output q excludes 2*pi.
    """

    vectors: np.ndarray
    wavelength: float
    pixel_map: PixelCoordinateMap | None = None

    def __post_init__(self):
        a = triples(self.vectors, "geometry vectors")
        if a.shape != (5, 3):
            raise ValueError("geometry vectors must have shape (5,3)")
        norms = np.linalg.norm(a[1:], axis=1)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("geometry directions must have finite nonzero norms")
        if abs(np.dot(a[1] / norms[0], a[2] / norms[1])) > 1 - 1e-10:
            raise ValueError("pixel directions must be independent")
        if not math.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        a.flags.writeable = False
        object.__setattr__(self, "vectors", a)
        if self.pixel_map is not None and not all(
            callable(getattr(self.pixel_map, name, None))
            for name in ("to_plane", "from_plane", "inverse_jacobian")
        ):
            raise TypeError("pixel_map must implement to_plane, from_plane and inverse_jacobian")

    @classmethod
    def from_poni(
        cls,
        *,
        distance,
        poni1,
        poni2,
        pixel1,
        pixel2,
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength,
        axis,
    ):
        """Undistorted pyFAI plane in its right-handed (1,2,3) basis.

        Parameters in metres/radians, wavelength in angstroms. x/y refer to
        the calibration's own array orientation. No implicit image flip.
        """
        p = np.array([distance, poni1, poni2, pixel1, pixel2, rot1, rot2, rot3])
        if not np.isfinite(p).all() or min(distance, pixel1, pixel2) <= 0:
            raise ValueError("invalid planar PONI parameters")
        c1, c2, c3 = map(math.cos, [rot1, rot2, rot3])
        s1, s2, s3 = map(math.sin, [rot1, rot2, rot3])
        r1 = np.array([[1, 0, 0], [0, c1, s1], [0, -s1, c1]])
        r2 = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
        r3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
        rotation = r3 @ r2 @ r1
        vectors = np.array(
            [
                rotation @ [pixel1 / 2 - poni1, pixel2 / 2 - poni2, distance],
                rotation @ [0, pixel2, 0],
                rotation @ [pixel1, 0, 0],
                [0, 0, 1],
                axis,
            ]
        )
        return cls(vectors, wavelength)


def reciprocal_vectors(xyomega, geometry):
    """Correct each observation before combining tracks; O(N) time/storage."""
    a = triples(xyomega, "xyomega")
    if not isinstance(geometry, PanelGeometry):
        raise TypeError("geometry must be PanelGeometry")
    if not hasattr(_core, "experimental_reciprocal"):
        raise RuntimeError("build PhaseSmith with --features experimental-spots")
    if geometry.pixel_map is not None:
        a[:, :2] = mapped_xy(geometry.pixel_map, a[:, :2])
    return _core.experimental_reciprocal(a, geometry.vectors, geometry.wavelength)


@dataclass(frozen=True)
class ReciprocalPositions:
    """One equal-observation mean q per linked track, and its 3D RMS spread."""

    points: np.ndarray
    rms: np.ndarray
    observation_vectors: np.ndarray


def correct_positions(positions, geometry):
    """Combine only existing links, never collapse distinct reciprocal neighbours.

    Equal weights avoid inventing unavailable per-observation intensity weights.
    RMS includes finite exposure width and is not a statistical standard error.
    """
    obs = np.asarray(positions.observations)
    if obs.ndim != 2 or obs.shape[1] != 5 or not np.isfinite(obs).all():
        raise ValueError("expected finite (track,frame,x,y,omega) observations")
    ids = obs[:, 0].astype(np.int64)
    if (ids != obs[:, 0]).any() or (ids < 0).any() or (ids >= len(positions)).any():
        raise ValueError("invalid track IDs")
    q = reciprocal_vectors(obs[:, 2:5], geometry)
    counts = np.bincount(ids, minlength=len(positions))
    if (counts == 0).any():
        raise ValueError("each track needs at least one observation")
    means = (
        np.column_stack([np.bincount(ids, weights=v, minlength=len(counts)) for v in q.T])
        / counts[:, None]
    )
    rms = np.sqrt(
        np.bincount(ids, weights=np.sum((q - means[ids]) ** 2, axis=1), minlength=len(counts))
        / counts
    )
    for a in [means, rms, q]:
        a.flags.writeable = False
    return ReciprocalPositions(means, rms, q)


def lattice_fourier(points, vectors):
    """Fused mean-cosine value, gradient and Hessian; rows contain 1+3+9 values."""
    q, v = triples(points, "points"), triples(vectors, "vectors")
    return _core.experimental_lattice_fourier(q, v)
