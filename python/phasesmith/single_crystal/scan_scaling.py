"""Opt-in relative scan scaling; input intensities must already include LP.

Confirmed equivalent-family labels are explicit input. No atom coordinates,
structure factors, CrysAlis values or inferred crystal symmetry are accepted.
See docs/single-crystal-scan-scaling.md for equations and limitations.
"""

from dataclasses import dataclass

import numpy as np

from phasesmith import _core

__all__ = ["ScanScaleModel", "ScanScaleOptions", "fit_scan_scale"]


def _vector(v, name, n=None):
    a = np.asarray(v)
    if a.ndim != 1 or a.dtype.kind not in "fiu" or (n is not None and len(a) != n):
        raise ValueError(f"{name} must be a real vector of matching length")
    return np.array(a, dtype=float, order="C", copy=True)


@dataclass(frozen=True)
class ScanScaleOptions:
    minimum_snr: float = 10.0
    relative_error_floor: float = 0.02
    huber: float = 2.5
    max_iterations: int = 30
    tolerance: float = 1e-10

    def __post_init__(self):
        for name in ("minimum_snr", "relative_error_floor", "huber", "tolerance"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not np.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if (
            self.minimum_snr < 3
            or not 0 <= self.relative_error_floor <= 1
            or self.huber <= 0
            or self.tolerance <= 0
        ):
            raise ValueError("invalid reference or robust-fit controls")
        if (
            isinstance(self.max_iterations, (bool, np.bool_))
            or not isinstance(self.max_iterations, (int, np.integer))
            or not 1 <= self.max_iterations <= 1000
        ):
            raise ValueError("max_iterations must be an integer in 1..1000")


@dataclass(frozen=True)
class ScanScaleModel:
    coefficient: float
    variance: float
    omega_range: tuple
    reference_ids: np.ndarray
    reference_weights: np.ndarray
    iterations: int
    converged: bool
    options: ScanScaleOptions

    def __post_init__(self):
        if not isinstance(self.options, ScanScaleOptions):
            raise TypeError("options must be ScanScaleOptions")
        if (
            isinstance(self.iterations, (bool, np.bool_))
            or not isinstance(self.iterations, (int, np.integer))
            or not 1 <= self.iterations <= self.options.max_iterations
            or not isinstance(self.converged, (bool, np.bool_))
        ):
            raise ValueError("invalid iteration count or convergence flag")
        if not np.isfinite(self.coefficient) or not np.isfinite(self.variance) or self.variance < 0:
            raise ValueError("coefficient and nonnegative variance must be finite")
        r = np.asarray(self.omega_range, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all() or not r[0] < r[1]:
            raise ValueError("omega_range must be increasing finite endpoints")
        object.__setattr__(self, "omega_range", tuple(map(float, r)))
        ids = np.asarray(self.reference_ids)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or len(ids) < 3
            or (ids < 0).any()
            or (ids >= 1_000_000).any()
            or (np.diff(ids.astype(np.int64)) <= 0).any()
        ):
            raise ValueError("reference_ids must be nonnegative integers")
        weights = _vector(self.reference_weights, "reference_weights", len(ids))
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("reference weights must be positive and finite")
        for name, value in (
            ("reference_ids", np.array(ids, dtype=np.int64)),
            ("reference_weights", weights),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def apply(self, intensity, sigma, omega):
        """Return factor, signed I, counting sigma, dI/dbeta, dI/domega, model sigma.

        Model sigma is separate because its errors correlate reflections.
        No angular extrapolation; both declared endpoints are included.
        Missing/flagged data must be selected out explicitly by the caller.
        """
        i = _vector(intensity, "intensity")
        s = _vector(sigma, "sigma", len(i))
        om = _vector(omega, "omega", len(i))
        if len(i) > 1_000_000:
            raise ValueError("scan-scaling application exceeds one-million observations")
        result = _core.experimental_scan_scale_apply(
            i, s, om, self.omega_range, self.coefficient, self.variance
        )
        result.flags.writeable = False
        return result


def fit_scan_scale(
    intensity, sigma, omega, family_ids, *, omega_range, training=None, eligible=None, options=None
):
    """Fit a strong-reference relative log slope with unknown family intensities.

    Supply labels only for confirmed symmetry equivalents of the same crystal.
    Training and eligibility are independent masks; withheld families must have
    training=False for every member. Signed/weak/negative rows are retained in
    the input but cannot train a logarithmic scale model. Their later correction
    remains linear and signed. At least two strong observations per family and
    nonzero within-family angular spread are required.
    """
    if not hasattr(_core, "experimental_scan_scale_fit"):
        raise RuntimeError("rebuild PhaseSmith with single-crystal scan scaling")
    options = options or ScanScaleOptions()
    if not isinstance(options, ScanScaleOptions):
        raise TypeError("options must be ScanScaleOptions")
    i = _vector(intensity, "intensity")
    n = len(i)
    s = _vector(sigma, "sigma", n)
    om = _vector(omega, "omega", n)
    if not 1 <= n <= 1_000_000:
        raise ValueError("expected 1..1000000 observations")
    labels = np.asarray(family_ids)
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or (labels < 0).any():
        raise ValueError("family_ids must be nonnegative integers with observation length")
    _, groups = np.unique(labels, return_inverse=True)
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    mask = np.ones(n, bool)
    for values, name in ((training, "training"), (eligible, "eligible")):
        if values is not None:
            v = np.asarray(values)
            if v.shape != (n,) or v.dtype != bool:
                raise ValueError(f"{name} must be a boolean observation vector")
            mask &= v
    bounds = np.asarray(omega_range, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or not bounds[0] < bounds[1]
        or not np.isfinite(om).all()
        or (om < bounds[0]).any()
        or (om > bounds[1]).any()
    ):
        raise ValueError("finite omegas must lie in the declared increasing angular range")
    half = (bounds[1] - bounds[0]) / 2
    if not np.isfinite(half):
        raise ValueError("angular normalization overflows")
    x = np.ascontiguousarray((om - (bounds[0] + half)) / half)
    beta, variance, ids, weights, iterations, converged = _core.experimental_scan_scale_fit(
        i,
        s,
        x,
        groups,
        mask,
        options.minimum_snr,
        options.relative_error_floor,
        options.huber,
        options.max_iterations,
        options.tolerance,
    )
    return ScanScaleModel(
        beta, variance, tuple(bounds), ids, weights, iterations, converged, options
    )
