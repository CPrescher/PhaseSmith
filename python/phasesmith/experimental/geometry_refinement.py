"""Opt-in indexed-position detector calibration; no intensity/structure model.

Waterman et al. (2016), doi:10.1107/S2059798316002187. Rust evaluates
Ewald intersections and both analytical Jacobians in a single reflection pass;
Python supplies parameter chains, robust optimization, partitions and reporting.
"""

from dataclasses import dataclass

import numpy as np

from phasesmith import _core

from .prediction import (
    _basis,
    _hkl,
    _readonly,
    _statistics,
    map_predictions,
    orientation_matrix,
    predict_spots,
)
from .reciprocal import PanelGeometry, triples


def rotation(angles):
    """Rz @ Ry @ Rx and analytical partials in radians."""
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
    dx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]], float)
    dy = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]], float)
    dz = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]], float)
    return rz @ ry @ rx, np.array([rz @ ry @ dx, rz @ dy @ rx, dz @ ry @ rx])


def reference_basis(basis, cell):
    """Replace the metric, preserving initial canonical crystal orientation."""
    p = np.asarray(cell)
    if p.shape != (6,) or p.dtype.kind not in "fiu" or not np.isfinite(p).all():
        raise ValueError("reference_cell must contain six finite cell parameters")
    if (p[:3] <= 0).any() or (p[3:] <= 0).any() or (p[3:] >= 180).any():
        raise ValueError("invalid reference cell lengths/angles")
    alpha, beta, gamma = np.deg2rad(p[3:])
    gram = np.diag(p[:3] ** 2).astype(float)
    for i, j, c in [(0, 1, np.cos(gamma)), (0, 2, np.cos(beta)), (1, 2, np.cos(alpha))]:
        gram[i, j] = gram[j, i] = p[i] * p[j] * c
    try:
        direct = np.linalg.cholesky(gram) @ orientation_matrix(basis).T
    except np.linalg.LinAlgError as error:
        raise ValueError("reference cell has nonpositive volume") from error
    return np.linalg.inv(direct).T


def predict_with_detector(hkl, basis, geometry, hints):
    """Values, B partials, detector-vector partials and flags (owned arrays)."""
    if not hasattr(_core, "experimental_predict_spots_with_detector"):
        raise RuntimeError("rebuild PhaseSmith with experimental-spots geometry derivatives")
    h, b = _hkl(hkl), _basis(basis)
    if not isinstance(geometry, PanelGeometry):
        raise TypeError("geometry must be PanelGeometry")
    hint = np.asarray(hints)
    if hint.shape != (len(h),) or hint.dtype.kind not in "fiu" or not np.isfinite(hint).all():
        raise ValueError("hints must be a finite vector matching hkl")
    v, j, d, f = _core.experimental_predict_spots_with_detector(
        h, b, np.ascontiguousarray(hint, float), geometry.vectors, geometry.wavelength, False
    )
    j, d = j.reshape(-1, 3, 9), d.reshape(-1, 3, 9)
    v, (j, d) = map_predictions(v, (j, d), f, geometry)
    return v, j, d, f


@dataclass(frozen=True)
class GeometryOptions:
    xy_scale: float = 0.4
    omega_scale: float = 0.2
    huber: float = 2.5
    max_evaluations: int = 150
    translation_limit_mm: float = 2.0
    rotation_limit_deg: float = 2.0
    log_length_limit: float = 0.02
    initial_xy_gate: float = 5.0
    initial_omega_gate: float = 1.0

    def __post_init__(self):
        for name in (
            "xy_scale",
            "omega_scale",
            "huber",
            "translation_limit_mm",
            "rotation_limit_deg",
            "log_length_limit",
            "initial_xy_gate",
            "initial_omega_gate",
        ):
            v = getattr(self, name)
            if (
                isinstance(v, (bool, np.bool_))
                or not np.isscalar(v)
                or not np.isfinite(v)
                or v <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_evaluations) is not int or not 1 <= self.max_evaluations <= 1000:
            raise ValueError("max_evaluations must be an integer in 1..1000")
        if (
            self.rotation_limit_deg > 10
            or self.log_length_limit > 0.2
            or self.translation_limit_mm > 100
        ):
            raise ValueError("geometry refinement exceeds local parameter bounds")


@dataclass(frozen=True)
class GeometryCalibration:
    initial_geometry: PanelGeometry
    geometry: PanelGeometry
    initial_basis: np.ndarray
    reciprocal_basis: np.ndarray
    observed: np.ndarray
    hkl: np.ndarray
    training: np.ndarray
    eligible: np.ndarray
    parameters: np.ndarray
    initial_predictions: object
    predictions: object
    metrics: dict
    options: GeometryOptions
    converged: bool
    at_bounds: bool
    scaled_condition: float
    maximum_parameter_correlation: float
    reference_cell: np.ndarray | None
    refine_lengths: bool
    evaluations: int

    def __post_init__(self):
        for name in (
            "initial_basis",
            "reciprocal_basis",
            "observed",
            "hkl",
            "training",
            "eligible",
            "parameters",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.reference_cell is not None:
            object.__setattr__(self, "reference_cell", _readonly(self.reference_cell))

    @property
    def acceptable(self):
        """Advisory acceptance: finite identified fit and withheld improvement."""
        before = self.metrics["validation"]["before"]["rms_scaled"]
        after = self.metrics["validation"]["after"]["rms_scaled"]
        return bool(
            self.converged and not self.at_bounds and self.scaled_condition < 1e6 and after < before
        )


def parameter_state(parameters, basis, geometry, pivot, refine_lengths):
    """Physical state and exact chains wrt Euler angles, mm shifts and log lengths."""
    p = np.asarray(parameters, float)
    n = 12 if refine_lengths else 9
    if p.shape != (n,) or not np.isfinite(p).all():
        raise ValueError("invalid geometry parameter vector")
    rc, dc = rotation(p[:3])
    rd, dd = rotation(p[6:9])
    b0 = basis.copy()
    if refine_lengths:
        b0 *= np.exp(p[9:12, None])
    b = b0 @ rc.T
    v = geometry.vectors.copy()
    v[0] = pivot + 0.001 * p[3:6] + rd @ (geometry.vectors[0] - pivot)
    v[1:3] = geometry.vectors[1:3] @ rd.T
    jb = np.zeros((9, n))
    jg = np.zeros((9, n))
    for k in range(3):
        jb[:, k] = (b0 @ dc[k].T).ravel()
        jg[:3, 3 + k] = 0.001 * np.eye(3)[k]
        jg[:, 6 + k] = np.concatenate(
            [
                dd[k] @ (geometry.vectors[0] - pivot),
                dd[k] @ geometry.vectors[1],
                dd[k] @ geometry.vectors[2],
            ]
        )
        if refine_lengths:
            jb[3 * k : 3 * k + 3, 9 + k] = b[k]
    return b, PanelGeometry(v, geometry.wavelength, geometry.pixel_map), jb, jg


def refine_geometry(
    observed,
    hkl,
    reciprocal_basis,
    geometry,
    image_shape,
    *,
    training,
    eligible=None,
    reference_cell=None,
    refine_lengths=False,
    options=None,
    progress=None,
):
    """Fit rigid detector pose and crystal orientation to confirmed indexed spots.

    Cell can be fixed to a declared reference or have its three lengths refined
    with initial angles fixed. Beam, wavelength, pixel metric, axis and omega-zero
    gauge remain fixed. No HKL reassignment, structure factors or intensity loss.
    Masks and held-out families belong to the caller; all inputs are snapshotted.
    """
    from scipy.optimize import least_squares

    obs = triples(observed, "observed")
    h = _hkl(hkl)
    b = _basis(reciprocal_basis)
    if len(obs) != len(h) or len(h) > 1_000_000:
        raise ValueError("observed and hkl sizes differ or exceed budget")
    if not isinstance(geometry, PanelGeometry):
        raise TypeError("geometry must be PanelGeometry")
    if type(refine_lengths) is not bool:
        raise TypeError("refine_lengths must be boolean")
    if len(image_shape) != 2 or any(
        not isinstance(v, (int, np.integer)) or v < 1 for v in image_shape
    ):
        raise ValueError("image_shape must contain two positive integer dimensions")
    opt = GeometryOptions() if options is None else options
    if not isinstance(opt, GeometryOptions):
        raise TypeError("options must be GeometryOptions")
    train = np.asarray(training)
    use = np.ones(len(h), bool) if eligible is None else np.asarray(eligible)
    if (
        train.shape != (len(h),)
        or use.shape != train.shape
        or train.dtype != bool
        or use.dtype != bool
    ):
        raise ValueError("training and eligible must be boolean vectors")
    train = train.copy()
    use = use.copy()
    initial = predict_spots(h, b, geometry, obs[:, 2])
    e = initial.values - obs
    use &= (
        (initial.flags == 0)
        & (np.linalg.norm(e[:, :2], axis=1) < opt.initial_xy_gate)
        & (abs(e[:, 2]) < opt.initial_omega_gate)
    )
    if (
        (train & use).sum() < 30
        or (~train & use).sum() < 10
        or np.linalg.matrix_rank(h[train & use]) < 3
    ):
        raise ValueError("need >=30 training and >=10 withheld indexed spots spanning 3D")
    base = b if reference_cell is None else reference_basis(b, reference_cell)
    pivot = (
        geometry.vectors[0]
        + (image_shape[1] - 1) / 2 * geometry.vectors[1]
        + (image_shape[0] - 1) / 2 * geometry.vectors[2]
    )
    select = train & use
    scale = np.array([opt.xy_scale, opt.xy_scale, opt.omega_scale])
    n = 12 if refine_lengths else 9
    units = np.array([0.001] * 3 + [0.1] * 3 + [0.001] * 6)[:n]
    limits = np.array(
        [np.deg2rad(opt.rotation_limit_deg)] * 3
        + [opt.translation_limit_mm] * 3
        + [np.deg2rad(opt.rotation_limit_deg)] * 3
        + [opt.log_length_limit] * 3
    )[:n]
    cache = {}
    evaluations = 0

    def evaluate(p):
        nonlocal evaluations
        if "p" not in cache or not np.array_equal(cache["p"], p):
            if progress:
                progress(evaluations, opt.max_evaluations)
            bb, gg, jb, jg = parameter_state(p, base, geometry, pivot, refine_lengths)
            v, j, d, f = predict_with_detector(h[select], bb, gg, obs[select, 2])
            if f.any():
                raise ValueError("trial geometry loses an indexed crossing; narrow the input gate")
            cache.update(
                p=p.copy(),
                r=((v - obs[select]) / scale).ravel(),
                j=((j @ jb + d @ jg) / scale[None, :, None]).reshape(-1, n),
            )
            evaluations += 1
        return cache

    fit = least_squares(
        lambda p: evaluate(p)["r"],
        np.zeros(n),
        jac=lambda p: evaluate(p)["j"],
        bounds=(-limits, limits),
        x_scale=units,
        loss="soft_l1",
        f_scale=opt.huber,
        max_nfev=opt.max_evaluations,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-9,
    )
    bb, gg, _, _ = parameter_state(fit.x, base, geometry, pivot, refine_lengths)
    final = predict_spots(h, bb, gg, obs[:, 2])
    if np.any(use & (final.flags != 0)):
        raise ValueError("refined geometry loses withheld indexed crossings")
    singular = np.linalg.svd(fit.jac * units, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    cov = np.linalg.pinv(fit.jac.T @ fit.jac)
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    correlation = np.divide(
        cov, np.outer(std, std), out=np.zeros_like(cov), where=np.outer(std, std) > 0
    )
    np.fill_diagonal(correlation, 0)
    metrics = _statistics(obs, initial.values, final.values, train, use, scale)
    return GeometryCalibration(
        geometry,
        gg,
        b,
        bb,
        obs,
        h,
        train,
        use,
        fit.x,
        initial,
        final,
        metrics,
        opt,
        bool(fit.success),
        bool(fit.active_mask.any()),
        condition,
        float(abs(correlation).max()),
        reference_cell,
        refine_lengths,
        evaluations,
    )
