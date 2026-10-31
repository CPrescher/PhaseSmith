"""Structure-independent F(frame)*D(panel)*A(ray) scaling from explicit families.

No atomic model, external integration or displacement parameter enters the fit.
Designs and family partitions are caller supplied and retained in the result.
Rust owns the model/partials, family centering, robust weights and bounded optimization.
"""

from dataclasses import dataclass

import numpy as np

from phasesmith import _core


def _real(value, name, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite real {ndim}D array")
    return np.ascontiguousarray(a, float)


@dataclass(frozen=True)
class ScalingDesign:
    frames: np.ndarray
    detector: np.ndarray
    harmonics: np.ndarray
    penalty: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for name in ("frames", "detector", "harmonics", "penalty", "lower", "upper"):
            a = _real(getattr(self, name), name, 1 if name in ("lower", "upper") else 2).copy()
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        n = len(self.frames)
        p = sum(a.shape[1] for a in (self.frames, self.detector, self.harmonics))
        if (
            not 1 <= p <= 256
            or n * p > 20_000_000
            or len(self.detector) != n
            or len(self.harmonics) != n
            or self.penalty.shape[1] != p
            or self.lower.shape != (p,)
            or self.upper.shape != (p,)
            or np.any(self.lower >= self.upper)
            or np.any(self.lower >= 0)
            or np.any(self.upper <= 0)
        ):
            raise ValueError("inconsistent scaling design, zero gauge or parameter budget")

    @property
    def parameters(self):
        return len(self.lower)

    def evaluate(self, coefficients, ids=None):
        b = _real(coefficients, "coefficients", 1)
        if b.shape != self.lower.shape:
            raise ValueError("coefficient shape differs from design")
        if ids is None:
            ids = slice(None)
        return _core.experimental_empirical_model(
            *(np.ascontiguousarray(a[ids]) for a in (self.frames, self.detector, self.harmonics)),
            b,
            1e-8,
        )


def family_center(values, groups, weights):
    a = _real(values, "family values", 2)
    w = _real(weights, "weights", 1)
    g = np.asarray(groups)
    if (
        g.shape != (len(a),)
        or g.dtype.kind not in "iu"
        or np.any(g < 0)
        or np.any(g >= len(a))
        or w.shape != (len(a),)
        or np.any(w <= 0)
    ):
        raise ValueError("invalid family labels or positive weights")
    return _core.experimental_family_center(a, np.ascontiguousarray(g, np.int64), w)


@dataclass(frozen=True)
class ScalingOptions:
    strength: float = 1000.0
    huber: float = 2.5
    iterations: int = 100
    minimum_component: float = 0.2

    def __post_init__(self):
        if (
            not np.isfinite([self.strength, self.huber, self.minimum_component]).all()
            or self.strength < 0
            or self.huber <= 0
            or not 0 < self.minimum_component < 1
            or type(self.iterations) is not int
            or not 1 <= self.iterations <= 200
        ):
            raise ValueError("invalid empirical scaling options")


@dataclass(frozen=True)
class ScalingResult:
    coefficients: np.ndarray
    correction: np.ndarray
    components: np.ndarray
    reference_mask: np.ndarray
    converged: bool
    history: tuple

    def apply(self, intensity, sigma):
        i = _real(intensity, "intensity", 1)
        s = _real(sigma, "sigma", 1)
        if i.shape != self.correction.shape or s.shape != i.shape or np.any(s < 0):
            raise ValueError("scaling input dimensions or uncertainty invalid")
        # Conditional counting uncertainty only; fitted scale errors correlate
        # reflections and are not represented by an invented diagonal floor.
        return np.column_stack([i * self.correction, s * self.correction])


def paired_mask(groups, mask):
    """Retain entire caller-selected families having at least two references."""
    g = np.asarray(groups)
    m = np.asarray(mask)
    if (
        g.ndim != 1
        or g.dtype.kind not in "iu"
        or np.any(g < 0)
        or np.any(g >= len(g))
        or m.shape != g.shape
        or m.dtype != bool
    ):
        raise ValueError("invalid family/reference arrays")
    count = np.bincount(g[m].astype(np.int64), minlength=len(g))
    return m & (count[g] >= 2)


def fit_empirical_scale(
    design, intensity, log_variance, groups, reference_mask, *, options=None, start=None
):
    """Native robust within-family log fit, preserving every signed output row."""
    if not isinstance(design, ScalingDesign):
        raise TypeError("design must be ScalingDesign")
    o = ScalingOptions() if options is None else options
    if not isinstance(o, ScalingOptions):
        raise TypeError("options must be ScalingOptions")
    y = _real(intensity, "intensity", 1)
    variance = _real(log_variance, "log variance", 1)
    mask = paired_mask(groups, reference_mask)
    b = np.zeros(design.parameters) if start is None else _real(start, "start", 1)
    if not hasattr(_core, "experimental_empirical_fit"):
        raise RuntimeError("rebuild PhaseSmith with experimental-spots native scaling optimization")
    result = _core.experimental_empirical_fit(
        design.frames,
        design.detector,
        design.harmonics,
        design.penalty,
        y,
        variance,
        np.ascontiguousarray(groups, np.int64),
        np.ascontiguousarray(mask),
        design.lower,
        design.upper,
        np.ascontiguousarray(b),
        o.strength,
        o.huber,
        o.iterations,
        o.minimum_component,
    )
    history = tuple(
        dict(
            parameter_change=float(row[0]), weight_change=float(row[1]), inner_success=bool(row[2])
        )
        for row in result["history"]
    )
    return ScalingResult(
        result["coefficients"],
        result["correction"],
        result["components"],
        result["references"],
        result["converged"],
        history,
    )


def panel_mesh(xy, detector_shape, grid=(5, 5)):
    """Bilinear node weights on measured pixel-centre coordinates (y,x shape)."""
    a = _real(xy, "xy", 2)
    if (
        a.shape[1] != 2
        or len(detector_shape) != 2
        or any(type(v) is not int or v <= 1 for v in detector_shape)
        or len(grid) != 2
        or any(type(v) is not int or not 2 <= v <= 8 for v in grid)
    ):
        raise ValueError("invalid panel/grid dimensions")
    h, w = detector_shape
    if np.any(a < 0) or np.any(a[:, 0] > w - 1) or np.any(a[:, 1] > h - 1):
        raise ValueError("mesh coordinates outside detector")
    nx, ny = grid
    u = a / np.array([w - 1, h - 1]) * [nx - 1, ny - 1]
    low = np.minimum(np.floor(u).astype(int), [nx - 2, ny - 2])
    frac = u - low
    out = np.zeros((len(a), nx * ny))
    for dy in (0, 1):
        for dx in (0, 1):
            out[np.arange(len(a)), (low[:, 1] + dy) * nx + low[:, 0] + dx] = (
                frac[:, 0] if dx else 1 - frac[:, 0]
            ) * (frac[:, 1] if dy else 1 - frac[:, 1])
    return out


def ray_harmonics(vectors):
    """Complete real degree-1 and degree-2 sectors, no radial/B-factor term."""
    v = _real(vectors, "rays", 2)
    if v.shape[1] != 3 or not np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("rays must be unit vectors")
    x, y, z = v.T
    return np.c_[x, y, z, x * x - y * y, 2 * z * z - x * x - y * y, 2 * x * y, 2 * x * z, 2 * y * z]


def score_equivalents(design, intensity, log_variance, groups, mask, coefficients):
    """Family-balanced held-out log residuals; no atom or reference-I targets."""
    y = _real(intensity, "intensity", 1)
    v = _real(log_variance, "log variance", 1)
    if y.shape != (len(design.frames),) or v.shape != y.shape or np.any(v <= 0):
        raise ValueError("invalid score inputs")
    use = paired_mask(groups, mask)
    if not use.any() or np.any(y[use] <= 0):
        raise ValueError("score needs positive paired observations")
    ids = np.flatnonzero(use)
    families, g = np.unique(np.asarray(groups)[ids], return_inverse=True)
    residual = family_center(
        (np.log(y[ids]) - design.evaluate(coefficients, ids)[0])[:, None], g, 1 / v[ids]
    )[:, 0]
    count = np.bincount(g)
    loss = np.bincount(g, weights=residual**2) / count
    return dict(
        families=families,
        family_loss=loss,
        rms_log=float(np.sqrt(loss.mean())),
        observations=len(ids),
    )


def select_common_strength(
    design, intensities, log_variance, groups, training, selection, strengths, *, options=None
):
    """Select one neutral-prior strength by pooled complete-family validation.

    Every integration has its own coefficients and nuisance family means, but
    shares the model, masks, variance, priors and selected strength. The most
    regularized candidate within one paired-family standard error of the best
    mean squared log residual is selected. Test families are never evaluated.
    """
    from dataclasses import replace

    ys = _real(intensities, "integrations", 2)
    candidates = _real(strengths, "strengths", 1)
    if (
        ys.shape[1] != len(design.frames)
        or not len(ys)
        or not len(candidates)
        or np.any(candidates < 0)
        or np.any(np.diff(candidates) <= 0)
    ):
        raise ValueError("strengths must be sorted distinct nonnegative values")
    train = paired_mask(groups, training)
    valid = paired_mask(groups, selection)
    g = np.asarray(groups)
    if np.intersect1d(g[train], g[valid]).size:
        raise ValueError("training and selection must withhold complete families")
    o = ScalingOptions() if options is None else options
    records = []
    for strength in candidates:
        fits, losses = [], []
        for intensity in ys:
            fit = fit_empirical_scale(
                design,
                intensity,
                log_variance,
                groups,
                train,
                options=replace(o, strength=float(strength)),
            )
            if not fit.converged:
                raise ValueError("candidate scaling fit failed to converge")
            score = score_equivalents(
                design, intensity, log_variance, groups, valid, fit.coefficients
            )
            fits.append(fit)
            losses.append(score["family_loss"])
        paired = np.mean(losses, axis=0)
        if len(paired) < 2:
            raise ValueError("at least two validation families required")
        records.append(
            dict(
                strength=float(strength),
                mean_loss=float(paired.mean()),
                family_se=float(paired.std(ddof=1) / np.sqrt(len(paired))),
                fits=tuple(fits),
            )
        )
    best = min(records, key=lambda r: r["mean_loss"])
    threshold = best["mean_loss"] + best["family_se"]
    chosen = max(r["strength"] for r in records if r["mean_loss"] <= threshold)
    return chosen, tuple(records)
