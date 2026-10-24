"""Fixed-instrument rotation prediction and positional crystal refinement."""

import math
from dataclasses import dataclass

import numpy as np

from phasesmith import _core

from .indexing import cell_parameters
from .reciprocal import PanelGeometry, triples


def _readonly(a, dtype=None):
    a = np.array(a, dtype=dtype, copy=True, order="C")
    a.flags.writeable = False
    return a


def _hkl(hkl):
    a = np.asarray(hkl)
    if a.ndim != 2 or a.shape[1] != 3 or a.dtype.kind not in "iu":
        raise ValueError("hkl must be integer shape (n,3)")
    if len(a) > 1_000_000 or (a > 1_000_000).any() or (a < -1_000_000).any():
        raise ValueError("prediction HKL/work budget exceeded")
    return np.array(a, dtype=np.int64, order="C", copy=True)


def _basis(basis):
    b = triples(basis, "reciprocal basis")
    if b.shape != (3, 3) or not np.isfinite(np.linalg.det(b)) or abs(np.linalg.det(b)) < 1e-15:
        raise ValueError("reciprocal basis must be nonsingular (3,3)")
    return b


@dataclass(frozen=True)
class Predictions:
    values: np.ndarray  # x/y/omega, pixels/degrees
    jacobian: np.ndarray  # (n,3,9), row-major reciprocal basis coefficients
    flags: np.ndarray  # 1 no crossing, 2 tangent, 4 no panel; zero valid

    def __post_init__(self):
        for name in ("values", "jacobian", "flags"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def predict_spots(hkl, reciprocal_basis, geometry, omega_hint, *, other=False):
    """Solve the Ewald equation and project onto the calibrated detector plane.

    Hints select/unroll the nearer of two crossings; other=True selects the
    other root. Invalid crossings return flags and NaNs, never fabricated spots.
    """
    h, b = _hkl(hkl), _basis(reciprocal_basis)
    if not isinstance(geometry, PanelGeometry):
        raise TypeError("geometry must be PanelGeometry")
    hints = np.asarray(omega_hint)
    if hints.dtype.kind not in "fiu":
        raise ValueError("omega hints must be real")
    hints = np.broadcast_to(hints, (len(h),)).astype(np.float64).copy()
    if not np.isfinite(hints).all() or (abs(hints) > 1e8).any():
        raise ValueError("omega hints must be finite within +/-1e8 degrees")
    if not hasattr(_core, "experimental_predict_spots"):
        raise RuntimeError("build PhaseSmith with --features experimental-spots for prediction")
    values, jacobian, flags = _core.experimental_predict_spots(
        h, b, hints, geometry.vectors, geometry.wavelength, bool(other)
    )
    return Predictions(values, jacobian.reshape(-1, 3, 9), flags)


@dataclass(frozen=True)
class RefinementOptions:
    xy_scale: float = 2.0
    omega_scale: float = 0.3
    huber: float = 2.5
    max_iterations: int = 40
    initial_xy_gate: float = 25.0
    initial_omega_gate: float = 2.0

    def __post_init__(self):
        for name in ("xy_scale", "omega_scale", "huber", "initial_xy_gate", "initial_omega_gate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("refinement scales/gates must be finite and positive")
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 200:
            raise ValueError("max_iterations must be an integer in [1,200]")


@dataclass(frozen=True)
class CrystalRefinement:
    initial_basis: np.ndarray
    reciprocal_basis: np.ndarray
    initial_predictions: Predictions
    predictions: Predictions
    observed: np.ndarray
    hkl: np.ndarray
    training: np.ndarray
    eligible: np.ndarray
    fit_mask: np.ndarray
    history: np.ndarray  # accepted training robust costs
    metrics: dict
    options: RefinementOptions
    converged: bool

    def __post_init__(self):
        for name in (
            "initial_basis",
            "reciprocal_basis",
            "observed",
            "hkl",
            "training",
            "eligible",
            "fit_mask",
            "history",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def direct_basis(self):
        return _readonly(np.linalg.inv(self.reciprocal_basis).T)

    @property
    def cell(self):
        return cell_parameters(self.direct_basis)


def orientation_matrix(reciprocal_basis):
    """Columns map canonical orthonormal crystal coordinates to sample frame.

    The canonical direct cell uses lower-triangular Cholesky rows, so its
    first axis follows a. No material or conventional-cell setting is supplied.
    """
    a = np.linalg.inv(_basis(reciprocal_basis)).T
    if np.linalg.det(a) <= 0:
        raise ValueError("orientation requires a right-handed basis")
    canonical = np.linalg.cholesky(a @ a.T)
    return np.linalg.solve(canonical, a).T


def _statistics(observed, initial, final, training, eligible, scale):
    result = {}
    for label, part in (("training", training), ("validation", ~training)):
        mask = part & eligible
        entry = {"count": int(mask.sum())}
        for name, predicted in (("before", initial), ("after", final)):
            residual = predicted[mask] - observed[mask]
            entry[name] = dict(
                rms_xy=float(np.sqrt(np.mean(np.sum(residual[:, :2] ** 2, axis=1)))),
                rms_omega=float(np.sqrt(np.mean(residual[:, 2] ** 2))),
                rms_scaled=float(np.sqrt(np.mean(np.sum((residual / scale) ** 2, axis=1)))),
            )
        result[label] = entry
    return result


def refine_crystal(
    observed,
    hkl,
    reciprocal_basis,
    geometry,
    *,
    training,
    eligible=None,
    options=None,
    progress=None,
):
    """Robust, training-only refinement of nine reciprocal-basis coefficients.

    Fixed assignments, fixed instrument, no symmetry constraint. Measurements
    are one original weighted x/y/omega centroid per linked track, not every
    frame observation. Scales are declared loss weights, not calibrated errors.
    """
    observed, h, b = triples(observed, "observed"), _hkl(hkl), _basis(reciprocal_basis)
    if len(observed) != len(h):
        raise ValueError("observed and hkl sizes differ")
    options = RefinementOptions() if options is None else options
    if not isinstance(options, RefinementOptions):
        raise TypeError("options must be RefinementOptions")
    training = np.asarray(training)
    eligible = np.ones(len(h), bool) if eligible is None else np.asarray(eligible)
    if (
        training.dtype != bool
        or eligible.dtype != bool
        or training.shape != (len(h),)
        or eligible.shape != (len(h),)
    ):
        raise ValueError("training/eligible must be boolean shape (n,)")
    initial = predict_spots(h, b, geometry, observed[:, 2])
    differences = initial.values - observed
    eligible = (
        eligible
        & (initial.flags == 0)
        & (np.linalg.norm(differences[:, :2], axis=1) < options.initial_xy_gate)
        & (abs(differences[:, 2]) < options.initial_omega_gate)
    )
    fit = eligible & training
    if fit.sum() < 12 or (eligible & ~training).sum() < 5:
        raise ValueError("need >=12 training and >=5 held-out eligible centroids")
    if np.linalg.matrix_rank(h[fit]) < 3:
        raise ValueError("training HKLs do not span three dimensions")
    b0 = b.copy()
    scale = np.array([options.xy_scale, options.xy_scale, options.omega_scale])

    def evaluate(basis):
        prediction = predict_spots(h[fit], basis, geometry, observed[fit, 2])
        if prediction.flags.any():
            return None
        residual = (prediction.values - observed[fit]) / scale
        lengths = np.linalg.norm(residual, axis=1)
        weights = np.minimum(1.0, options.huber / np.maximum(lengths, 1e-15))
        cost = np.where(
            lengths <= options.huber,
            0.5 * lengths**2,
            options.huber * (lengths - 0.5 * options.huber),
        ).sum()
        return float(cost), residual, prediction.jacobian / scale[None, :, None], weights

    current = evaluate(b)
    history = [current[0]]
    damping, converged = 1e-3, False
    for iteration in range(options.max_iterations):
        if progress is not None:
            progress(iteration, options.max_iterations)
        cost, residual, jac, weights = current
        j = (jac * np.sqrt(weights)[:, None, None]).reshape(-1, 9)
        r = (residual * np.sqrt(weights)[:, None]).ravel()
        if np.linalg.matrix_rank(j) < 9:
            raise ValueError("centroid refinement is rank deficient")
        normal, gradient = j.T @ j, j.T @ r
        diagonal = np.maximum(np.diag(normal), 1e-12)
        step = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient).reshape(3, 3)
        trial = b + step
        if np.linalg.det(trial) * np.linalg.det(b0) <= 0 or np.linalg.norm(
            trial - b0
        ) > 0.2 * np.linalg.norm(b0):
            damping *= 10
            continue
        candidate = evaluate(trial)
        if candidate is not None and candidate[0] < cost:
            b, current = trial, candidate
            history.append(current[0])
            damping = max(damping / 3, 1e-12)
            if np.linalg.norm(step) < 1e-10 or cost - current[0] < 1e-10 * max(cost, 1):
                converged = True
                break
        else:
            if np.linalg.norm(step) < 1e-10:
                converged = True
                break
            damping *= 10
        if damping > 1e14:
            break
    final = predict_spots(h, b, geometry, observed[:, 2])
    # No invalid validation predictions are dropped after fitting: failure is
    # visible as infinite validation metrics, rather than improved selection.
    metrics = _statistics(observed, initial.values, final.values, training, eligible, scale)
    for label, mask in (("training", fit), ("validation", eligible & ~training)):
        if final.flags[mask].any():
            metrics[label]["after"] = dict(
                rms_xy=float("inf"), rms_omega=float("inf"), rms_scaled=float("inf")
            )
    metrics["validation_improved"] = bool(
        metrics["validation"]["after"]["rms_scaled"] < metrics["validation"]["before"]["rms_scaled"]
    )
    return CrystalRefinement(
        b0,
        b,
        initial,
        final,
        observed,
        h,
        training,
        eligible,
        fit,
        np.asarray(history),
        metrics,
        options,
        converged,
    )


@dataclass(frozen=True)
class ScanPredictions:
    hkl: np.ndarray
    values: np.ndarray
    frame_indices: np.ndarray
    qmax: float

    def __post_init__(self):
        for name in ("hkl", "values", "frame_indices"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def predict_scan(
    reciprocal_basis,
    geometry,
    omega_bounds,
    image_shape,
    *,
    qmax,
    max_hkl=250_000,
    max_predictions=1_000_000,
):
    """Enumerate possible integer-HKL crossings; no extinctions/intensities.

    Detector footprint and each exposure use half-open boundaries. Both roots
    are enumerated. Wide (>180 degree) scan envelopes are rejected explicitly.
    """
    from .rotation import angle_segments

    bounds, _ = angle_segments(omega_bounds)
    if len(bounds) > 10_000:
        raise ValueError("scan prediction supports at most 10000 exposures")
    b = _basis(reciprocal_basis)
    if not math.isfinite(qmax) or qmax <= 0:
        raise ValueError("qmax must be finite and positive")
    if len(image_shape) != 2 or any(type(v) is not int or v < 1 for v in image_shape):
        raise ValueError("image_shape must be two positive integer dimensions")
    if (
        type(max_hkl) is not int
        or not 1 <= max_hkl <= 1_000_000
        or type(max_predictions) is not int
        or not 1 <= max_predictions <= 1_000_000
    ):
        raise ValueError("prediction budgets must lie in [1,1000000]")
    lo, hi = bounds.min(), bounds.max()
    if hi - lo > 180:
        raise ValueError("this experimental predictor supports scan envelopes <=180 degrees")
    a = np.linalg.inv(b).T
    extent = np.ceil(qmax * np.linalg.norm(a, axis=1))
    if not np.isfinite(extent).all() or (extent > max_hkl).any():
        raise ValueError("HKL enumeration budget exceeded")
    limits = extent.astype(np.int64)
    sizes = 2 * limits + 1
    if np.prod(sizes.astype(object)) > max_hkl:
        raise ValueError("HKL enumeration budget exceeded")
    h = np.indices(tuple(sizes), dtype=np.int64).reshape(3, -1).T - limits
    norm = np.linalg.norm(h @ b, axis=1)
    h = h[(norm <= qmax) & (norm > 0)]
    height, width = image_shape
    parts = []
    count = 0
    for other in (False, True):
        p = predict_spots(h, b, geometry, (lo + hi) / 2, other=other)
        valid = (
            (p.flags == 0)
            & (p.values[:, 2] >= lo)
            & (p.values[:, 2] < hi)
            & (p.values[:, 0] >= -0.5)
            & (p.values[:, 0] < width - 0.5)
            & (p.values[:, 1] >= -0.5)
            & (p.values[:, 1] < height - 0.5)
        )
        indices = np.flatnonzero(valid)
        for frame, (start, end) in enumerate(bounds):
            selected = indices[
                (p.values[indices, 2] >= min(start, end)) & (p.values[indices, 2] < max(start, end))
            ]
            count += len(selected)
            if count > max_predictions:
                raise ValueError("prediction count budget exceeded")
            parts.append((h[selected], p.values[selected], np.full(len(selected), frame, np.int64)))
    hkls, values, frames = [np.concatenate([p[i] for p in parts], axis=0) for i in range(3)]
    order = np.lexsort((hkls[:, 2], hkls[:, 1], hkls[:, 0], values[:, 2], frames))
    return ScanPredictions(hkls[order], values[order], frames[order], qmax)
