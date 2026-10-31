"""Explicit detector-section centroid maps learned from strong positional residuals.

These are empirical pixel-coordinate corrections, not independent sensor
metrology. Section bounds/centres must be supplied; camera names are not used.
"""

from dataclasses import dataclass

import numpy as np

from .empirical_scaling import _real

__all__ = ["SensorLayout", "SensorOffsets", "fit_sensor_offsets"]


@dataclass(frozen=True)
class SensorLayout:
    bounds: np.ndarray  # half-open x0,y0,x1,y1 pixel-coordinate sections
    centres: np.ndarray
    coordinate_scale: float = 256.0
    rotations: bool = True
    shared_symmetric: bool = True

    def __post_init__(self):
        b = _real(self.bounds, "sensor bounds", 2).copy()
        c = _real(self.centres, "sensor centres", 2).copy()
        if (
            not 1 <= len(b) <= 64
            or b.shape != (len(b), 4)
            or c.shape != (len(b), 2)
            or np.any(b[:, :2] >= b[:, 2:])
            or np.any(c < b[:, :2])
            or np.any(c >= b[:, 2:])
            or not np.isscalar(self.coordinate_scale)
            or not np.isfinite(self.coordinate_scale)
            or self.coordinate_scale <= 0
            or type(self.rotations) is not bool
            or type(self.shared_symmetric) is not bool
        ):
            raise ValueError("invalid sensor layout")
        for i in range(len(b)):
            overlap = np.all(
                np.minimum(b[i, 2:], b[i + 1 :, 2:]) > np.maximum(b[i, :2], b[i + 1 :, :2]), axis=1
            )
            if overlap.any():
                raise ValueError("sensor section bounds overlap")
        b.flags.writeable = c.flags.writeable = False
        object.__setattr__(self, "bounds", b)
        object.__setattr__(self, "centres", c)

    def to_dict(self):
        """JSON-safe explicit detector-section recipe."""
        return dict(
            bounds=self.bounds.tolist(),
            centres=self.centres.tolist(),
            coordinate_scale=self.coordinate_scale,
            rotations=self.rotations,
            shared_symmetric=self.shared_symmetric,
        )

    @property
    def parameters(self):
        return len(self.bounds) * (3 if self.rotations else 2) + (3 if self.shared_symmetric else 0)

    def sections(self, xy):
        xy = _real(xy, "sensor coordinates", 2)
        if xy.shape[1] != 2:
            raise ValueError("sensor coordinates must have two columns")
        inside = np.all(
            (xy[:, None] >= self.bounds[None, :, :2]) & (xy[:, None] < self.bounds[None, :, 2:]),
            axis=2,
        )
        if np.any(inside.sum(axis=1) != 1):
            raise ValueError("coordinates outside declared sensor sections")
        return inside.argmax(axis=1)

    def design(self, xy, sections=None):
        xy = _real(xy, "sensor coordinates", 2)
        sid = self.sections(xy) if sections is None else np.asarray(sections)
        if (
            xy.shape[1] != 2
            or sid.shape != (len(xy),)
            or sid.dtype.kind not in "iu"
            or np.any(sid < 0)
            or np.any(sid >= len(self.bounds))
        ):
            raise ValueError("invalid explicit sensor assignment")
        n, ns = len(xy), len(self.bounds)
        d = (xy - self.centres[sid]) / self.coordinate_scale
        out = np.zeros((n, 2, self.parameters))
        out[np.arange(n), 0, 2 * sid] = 1
        out[np.arange(n), 1, 2 * sid + 1] = 1
        k = 2 * ns
        if self.rotations:
            out[np.arange(n), 0, k + sid] = -d[:, 1]
            out[np.arange(n), 1, k + sid] = d[:, 0]
            k += ns
        if self.shared_symmetric:
            out[:, 0, k] = d[:, 0]
            out[:, 1, k + 1] = d[:, 1]
            out[:, 0, k + 2] = d[:, 1]
            out[:, 1, k + 2] = d[:, 0]
        return out


@dataclass(frozen=True)
class SensorOffsets:
    layout: SensorLayout
    coefficients: np.ndarray
    provenance: str

    def __post_init__(self):
        b = _real(self.coefficients, "sensor coefficients", 1).copy()
        if (
            b.shape != (self.layout.parameters,)
            or not isinstance(self.provenance, str)
            or not self.provenance.strip()
        ):
            raise ValueError("sensor coefficients/provenance required")
        b.flags.writeable = False
        object.__setattr__(self, "coefficients", b)

    def to_dict(self):
        """A portable frozen centroid map for ``ReductionConfig.sensor``."""
        return dict(
            layout=self.layout.to_dict(),
            coefficients=self.coefficients.tolist(),
            provenance=self.provenance,
        )

    def evaluate(self, xy):
        return np.einsum("nip,p->ni", self.layout.design(xy), self.coefficients)


def fit_sensor_offsets(
    layout,
    xy,
    shifts,
    training,
    *,
    provenance,
    huber_pixels=0.3,
    regularization=2.0,
    iterations=8,
    penalty_offsets=None,
):
    """Robust fit to declared training centroids, preserving held-out rows.

    Regularization penalizes differential displacement at four section-edge
    probes, leaving translations unpenalized. ``penalty_offsets`` optionally
    declares four x/y probe offsets per section, in pixels, independently of
    assignment bounds that may include gaps. Default probes use section half
    extents. It does not target R1 or U values.
    """
    x = _real(xy, "sensor coordinates", 2)
    y = _real(shifts, "centroid shifts", 2)
    m = np.asarray(training)
    if (
        y.shape != x.shape
        or m.shape != (len(x),)
        or m.dtype != bool
        or not np.isfinite([huber_pixels, regularization]).all()
        or huber_pixels <= 0
        or regularization < 0
        or type(iterations) is not int
        or not 1 <= iterations <= 100
    ):
        raise ValueError("invalid sensor fitting inputs")
    design = layout.design(x)
    if np.any(np.bincount(layout.sections(x[m]), minlength=len(layout.bounds)) < 2):
        raise ValueError("each declared section needs at least two training centroids")
    c = np.repeat(layout.centres, 4, axis=0)
    sid = np.repeat(np.arange(len(layout.bounds)), 4)
    half = (layout.bounds[:, 2:] - layout.bounds[:, :2]) / 2
    default_offsets = np.stack(
        [
            np.c_[half[:, 0], half[:, 0] * 0],
            np.c_[-half[:, 0], half[:, 0] * 0],
            np.c_[half[:, 1] * 0, half[:, 1]],
            np.c_[half[:, 1] * 0, -half[:, 1]],
        ],
        axis=1,
    )
    offsets = (
        default_offsets
        if penalty_offsets is None
        else _real(penalty_offsets, "sensor penalty offsets", 3)
    )
    if offsets.shape != (len(layout.bounds), 4, 2):
        raise ValueError("sensor penalty offsets must be (sections,4,2)")
    if np.any(np.linalg.matrix_rank(offsets) < 2):
        raise ValueError("sensor penalty offsets must span both spatial directions")
    probes = c + offsets.reshape(-1, 2)
    q = (layout.design(probes, sid) - layout.design(c, sid)).reshape(-1, layout.parameters)
    reg = regularization * q.T @ q
    a = design[m].reshape(-1, layout.parameters)
    w = np.ones(m.sum())
    for _ in range(iterations):
        ww = np.repeat(w, 2)
        normal = a.T @ (ww[:, None] * a) + reg
        if np.linalg.matrix_rank(normal) < layout.parameters:
            raise ValueError("sensor correction is not identifiable on training data")
        b = np.linalg.solve(normal, a.T @ (ww * y[m].ravel()))
        error = (a @ b).reshape(-1, 2) - y[m]
        w = np.minimum(1.0, huber_pixels / np.maximum(np.linalg.norm(error, axis=1), 1e-30))
    return SensorOffsets(layout, b, provenance)
