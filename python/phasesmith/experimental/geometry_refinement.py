"""Opt-in indexed-position detector calibration; no intensity/structure model.

Waterman et al. (2016), doi:10.1107/S2059798316002187. Rust evaluates
Ewald intersections and both analytical Jacobians in a single reflection pass;
Rust also owns parameter chains, robust optimization and diagnostics. Python
validates/copies arrays and presents the result. No SciPy runtime dependency.
"""

from dataclasses import dataclass

import numpy as np

from phasesmith import _core

from .prediction import (
    Predictions,
    _basis,
    _hkl,
    _readonly,
    map_predictions,
)
from .reciprocal import PanelGeometry, triples


def _native():
    if not hasattr(_core, "experimental_geometry_fit"):
        raise RuntimeError("rebuild PhaseSmith with experimental-spots native geometry fitting")
    return _core


def rotation(angles):
    """Native Rz @ Ry @ Rx and analytical partials in radians."""
    a = np.asarray(angles)
    if a.shape != (3,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("angles must contain three finite real values")
    result = _native().experimental_geometry_rotation(tuple(a))
    return result["rotation"], result["partials"].reshape(3, 3, 3)


def reference_basis(basis, cell):
    """Replace the metric in Rust, preserving canonical crystal orientation."""
    p = np.asarray(cell)
    if p.shape != (6,) or p.dtype.kind not in "fiu" or not np.isfinite(p).all():
        raise ValueError("reference_cell must contain six finite cell parameters")
    return _native().experimental_geometry_reference_basis(_basis(basis), tuple(p))


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
    covariance: np.ndarray | None = None
    objective_history: tuple = ()
    backend: str = "phasesmith-spots Rust"

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
        if self.covariance is not None:
            object.__setattr__(self, "covariance", _readonly(self.covariance))
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
    """Native physical state and exact angle/mm/log-length parameter chains."""
    p = np.asarray(parameters)
    n = 12 if refine_lengths else 9
    v = np.asarray(pivot)
    if p.shape != (n,) or p.dtype.kind not in "fiu" or not np.isfinite(p).all():
        raise ValueError("invalid geometry parameter vector")
    if v.shape != (3,) or v.dtype.kind not in "fiu" or not np.isfinite(v).all():
        raise ValueError("pivot must contain three finite values")
    if not isinstance(geometry, PanelGeometry) or type(refine_lengths) is not bool:
        raise TypeError("geometry must be PanelGeometry and refine_lengths must be bool")
    result = _native().experimental_geometry_state(
        np.ascontiguousarray(p, float),
        _basis(basis),
        geometry.vectors,
        geometry.wavelength,
        tuple(v),
        refine_lengths,
    )
    return (
        result["basis"],
        PanelGeometry(result["vectors"], geometry.wavelength, geometry.pixel_map),
        result["basis_jacobian"],
        result["detector_jacobian"],
    )


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
    cell = None if reference_cell is None else np.asarray(reference_cell)
    if cell is not None and (
        cell.shape != (6,) or cell.dtype.kind not in "fiu" or not np.isfinite(cell).all()
    ):
        raise ValueError("reference_cell must contain six finite cell parameters")
    result = _native().experimental_geometry_fit(
        obs,
        h,
        b,
        geometry.vectors,
        geometry.wavelength,
        tuple(image_shape),
        np.ascontiguousarray(train),
        np.ascontiguousarray(use),
        None if cell is None else tuple(cell),
        refine_lengths,
        (
            opt.xy_scale,
            opt.omega_scale,
            opt.huber,
            opt.max_evaluations,
            opt.translation_limit_mm,
            opt.rotation_limit_deg,
            opt.log_length_limit,
            opt.initial_xy_gate,
            opt.initial_omega_gate,
        ),
        progress,
        geometry.pixel_map,
    )

    def predictions(record):
        return Predictions(record["values"], record["jacobian"].reshape(-1, 3, 9), record["flags"])

    metrics = {}
    for i, name in enumerate(("training", "validation")):
        part = dict(count=result["counts"][i])
        for j, when in enumerate(("before", "after")):
            part[when] = dict(
                zip(("rms_xy", "rms_omega", "rms_scaled"), result["rms"][i][j], strict=True)
            )
        metrics[name] = part
    return GeometryCalibration(
        geometry,
        PanelGeometry(result["vectors"], geometry.wavelength, geometry.pixel_map),
        b,
        result["basis"],
        obs,
        h,
        train,
        result["eligible"],
        result["parameters"],
        predictions(result["initial"]),
        predictions(result["predictions"]),
        metrics,
        opt,
        result["converged"],
        result["at_bounds"],
        result["condition"],
        result["correlation"],
        cell,
        refine_lengths,
        result["evaluations"],
        covariance=result["covariance"],
        objective_history=tuple(result["history"]),
    )
