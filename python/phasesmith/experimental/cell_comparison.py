"""Position-only primitive hypotheses and training-only Bravais comparison.

Experimental orchestration over the native lattice assignment, Selling reduction
and fused spot prediction kernels. Fixed HKLs and fixed eligible populations make
nested metric fits comparable. No reference cell, material or intensity input.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from phasesmith import _core

from .cell_metrics import (
    BRAVAIS,
    BravaisSetting,
    find_bravais_settings,
    metric_chain,
    metric_directions,
    metric_step,
    reduce_cell,
)
from .indexing import cell_parameters
from .prediction import _basis, _readonly, predict_spots
from .reciprocal import PanelGeometry, triples


@dataclass(frozen=True)
class CellComparisonOptions:
    max_subcell_index: int = 4
    minimum_retention: float = 0.9
    q_tolerance: float = 0.015  # strict <, inverse Angstrom, no 2pi
    q_scale: float = 0.005
    max_strain: float = 0.01
    max_angle: float = 1.0
    coefficient_limit: int = 2
    xy_scale: float = 2.0
    omega_scale: float = 0.3
    initial_xy_gate: float = 25.0
    initial_omega_gate: float = 2.0
    huber: float = 2.5
    max_iterations: int = 60

    def __post_init__(self):
        for name, lo, hi in [
            ("max_subcell_index", 1, 8),
            ("coefficient_limit", 1, 3),
            ("max_iterations", 1, 200),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be integer in [{lo},{hi}]")
        for name in (
            "q_tolerance",
            "q_scale",
            "xy_scale",
            "omega_scale",
            "initial_xy_gate",
            "initial_omega_gate",
            "huber",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.minimum_retention) or not 0.5 < self.minimum_retention <= 1:
            raise ValueError("minimum_retention must lie in (.5,1]")
        if not np.isfinite(self.max_strain) or not 0 < self.max_strain <= 0.1:
            raise ValueError("max_strain must lie in (0,.1]")
        if not np.isfinite(self.max_angle) or not 0 < self.max_angle <= 5:
            raise ValueError("max_angle must lie in (0,5] degrees")


@dataclass(frozen=True)
class PrimitiveHypothesis:
    id: int
    reciprocal_hnf: np.ndarray  # exact: A_hypothesis_initial=H^-T A_input
    direct_basis: np.ndarray  # training-only unconstrained least squares on fixed HKLs
    hkl: np.ndarray
    retained: np.ndarray  # membership within fixed parent population
    training_count: int
    validation_count: int
    excluded_training: int
    excluded_validation: int
    supported: bool
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "supported", bool(self.supported))
        for name in ("reciprocal_hnf", "direct_basis", "hkl", "retained"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def index(self):
        return round(np.linalg.det(self.reciprocal_hnf))

    @property
    def cell(self):
        return cell_parameters(self.direct_basis)

    @property
    def volume(self):
        return float(abs(np.linalg.det(self.direct_basis)))


@dataclass(frozen=True)
class SettingFit:
    primitive_id: int
    setting: BravaisSetting
    reciprocal_basis: np.ndarray  # refined conventional reciprocal rows
    hkl: np.ndarray
    eligible: np.ndarray  # same population for every setting of this hypothesis
    history: np.ndarray
    training_metrics: dict
    validation_metrics: dict
    parameter_count: int
    training_score: float  # 2*Huber cost + p*log(3*N_train), fixed scales
    converged: bool
    status: str

    def __post_init__(self):
        for name in ("reciprocal_basis", "hkl", "eligible", "history"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def cell(self):
        return cell_parameters(np.linalg.inv(self.reciprocal_basis).T)


@dataclass(frozen=True)
class CellComparison:
    primitives: tuple[PrimitiveHypothesis, ...]
    settings: tuple[SettingFit, ...]
    parent_hkl: np.ndarray
    parent_indexed: np.ndarray
    training: np.ndarray
    recommended_primitive_id: int
    recommended_setting_index: int
    options: CellComparisonOptions
    mode: str
    searched_types: tuple[str, ...] = BRAVAIS

    def __post_init__(self):
        for name in ("parent_hkl", "parent_indexed", "training"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def recommended(self):
        return self.settings[self.recommended_setting_index]


def reciprocal_hnfs(max_index):
    """All upper row-Hermite reciprocal sublattices up to the declared index."""
    if type(max_index) is not int or not 1 <= max_index <= 8:
        raise ValueError("max_index must be integer in [1,8]")
    for index in range(1, max_index + 1):
        for a in range(1, index + 1):
            for d in range(1, index + 1):
                if index % (a * d):
                    continue
                f = index // (a * d)
                for b, c, e in product(range(d), range(f), range(f)):
                    yield np.array([[a, b, c], [0, d, e], [0, 0, f]], dtype=np.int64)


def _statistics(residual, mask, scale, mode):
    r = residual[mask]
    if not len(r):
        return dict(count=0, rms_scaled=float("nan"))
    scaled = r / scale
    stats = dict(
        count=int(mask.sum()), rms_scaled=float(np.sqrt(np.mean(np.sum(scaled * scaled, axis=1))))
    )
    if mode == "reciprocal":
        stats["rms_q"] = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
    else:
        stats["rms_xy"] = float(np.sqrt(np.mean(np.sum(r[:, :2] ** 2, axis=1))))
        stats["rms_omega"] = float(np.sqrt(np.mean(r[:, 2] ** 2)))
    return stats


def _fit_setting(points, observed, geometry, training, primitive, setting, eligible, options):
    h = primitive.hkl @ setting.transform.T
    fit = eligible & training
    mode = "reciprocal" if observed is None else "detector"
    scale = (
        np.full(3, options.q_scale)
        if observed is None
        else np.array([options.xy_scale, options.xy_scale, options.omega_scale])
    )
    target = points if observed is None else observed
    p = len(metric_directions(setting.symbol)) + 3
    b = np.linalg.inv(setting.projected_basis).T
    b0 = b.copy()

    def evaluate(basis, mask, derivatives=False):
        if observed is None:
            residual = h[mask] @ basis - target[mask]
            if derivatives:
                jac = np.einsum("ni,jk->njik", h[mask], np.eye(3)).reshape(-1, 3, 9)
        else:
            prediction = predict_spots(h[mask], basis, geometry, observed[mask, 2])
            residual = prediction.values - target[mask]
            if prediction.flags.any():
                residual = residual.copy()
                residual[prediction.flags != 0] = np.inf
            if derivatives:
                jac = prediction.jacobian
        if not derivatives:
            return residual
        residual = residual / scale
        length = np.linalg.norm(residual, axis=1)
        weights = np.minimum(1.0, options.huber / np.maximum(length, 1e-15))
        cost = np.where(
            length <= options.huber, 0.5 * length**2, options.huber * (length - 0.5 * options.huber)
        ).sum()
        return float(cost), residual, jac / scale[None, :, None], weights

    current = evaluate(b, fit, True)
    history = [current[0]]
    status = "iteration limit"
    converged = False
    damping = 1e-3
    for _ in range(options.max_iterations):
        cost, r, jac, weights = current
        if not np.isfinite(cost):
            status = "invalid training prediction"
            break
        chain = metric_chain(b, setting.symbol)
        j = ((jac @ chain) * np.sqrt(weights)[:, None, None]).reshape(-1, p)
        rr = (r * np.sqrt(weights)[:, None]).ravel()
        if np.linalg.matrix_rank(j) < p:
            status = "rank deficient training fit"
            break
        normal = j.T @ j
        step = np.linalg.solve(
            normal + damping * np.diag(np.maximum(np.diag(normal), 1e-12)), -j.T @ rr
        )
        try:
            trial = metric_step(b, step, setting.symbol)
            if np.linalg.norm(trial - b0) > 0.2 * np.linalg.norm(b0):
                raise ValueError("step budget")
            candidate = evaluate(trial, fit, True)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            candidate = None
        if candidate is not None and candidate[0] < cost:
            b, current = trial, candidate
            history.append(current[0])
            damping = max(damping / 3, 1e-12)
            if np.linalg.norm(step) < 1e-10 or cost - current[0] < 1e-10 * max(1.0, cost):
                converged = True
                status = "converged"
                break
        else:
            if np.linalg.norm(step) < 1e-10:
                converged = True
                status = "converged"
                break
            damping *= 10
        if damping > 1e14:
            status = "damping limit"
            break
    # Validation is evaluated after fitting; never influences acceptance or ranking.
    residual = evaluate(b, np.ones(len(points), bool))
    score = 2 * history[-1] + p * np.log(3 * int(fit.sum()))
    if status in ("rank deficient training fit", "invalid training prediction"):
        score = float("inf")
    return SettingFit(
        primitive.id,
        setting,
        b,
        h,
        eligible,
        np.array(history),
        _statistics(residual, fit, scale, mode),
        _statistics(residual, eligible & ~training, scale, mode),
        p,
        float(score),
        converged,
        status,
    )


def compare_cells(
    points, direct_basis, *, training, observed=None, geometry=None, options=None, progress=None
):
    """Compare primitive hypotheses and Bravais settings of an inferred lattice.

    ``points``: unindexed Cartesian reciprocal positions, no 2pi, inverse Å.
    ``direct_basis``: a training-derived inferred basis (rows, Å), not a prior.
    ``training``: explicit boolean split. Held-out points never enter a fit,
    support decision or recommendation. Optional observed (x,y,omega) uses the
    native fused Ewald prediction/Jacobian and a fixed PanelGeometry.

    Smaller cells must retain >=minimum_retention of fixed parent training
    assignments. Rejected hypotheses and excluded-reflection counts are retained.
    A recommendation is bounded evidence, not proof of primitivity or symmetry.
    Ranking uses a declared robust parameter penalty, not a calibrated BIC or a
    space-group probability, and is only compared within one primitive population.
    """
    points = triples(points, "points")
    a = _basis(direct_basis)
    if np.linalg.det(a) <= 0 or np.linalg.cond(a) > 1e8 or np.max(abs(a)) > 1e5:
        raise ValueError("direct basis must be right handed within conditioning/magnitude budgets")
    if not 12 <= len(points) <= 1_000_000 or np.max(abs(points)) > 1e3:
        raise ValueError("position count/magnitude budget exceeded")
    training = np.asarray(training)
    if training.dtype != bool or training.shape != (len(points),) or training.sum() < 12:
        raise ValueError("training must be boolean shape (n,) with >=12 training positions")
    options = CellComparisonOptions() if options is None else options
    if not isinstance(options, CellComparisonOptions):
        raise TypeError("options must be CellComparisonOptions")
    if (observed is None) != (geometry is None):
        raise ValueError("observed and geometry must be provided together")
    if observed is not None:
        observed = triples(observed, "observed")
        if observed.shape != points.shape or not isinstance(geometry, PanelGeometry):
            raise ValueError("observed must match points and geometry must be PanelGeometry")
    if not hasattr(_core, "ExperimentalLatticeEngine"):
        raise RuntimeError(
            "build PhaseSmith with --features experimental-spots for cell comparison"
        )
    # Assign in a reduced basis, then convert indices back exactly. Direct
    # rounding in a heavily sheared input basis is unstable even at small noise.
    reduced = reduce_cell(a)
    u = reduced.transform
    inverse = np.rint(np.linalg.inv(u)).astype(np.int64)
    if (
        np.max(abs(u)) > 1_000_000
        or np.max(abs(inverse)) > 1_000_000
        or not np.array_equal(u @ inverse, np.eye(3, dtype=np.int64))
    ):
        raise ValueError("assignment reduction transform budget exceeded")
    engine = _core.ExperimentalLatticeEngine(1)
    hr, error = engine.score(points, reduced.direct_basis, np.linalg.inv(reduced.direct_basis).T)
    if np.max(abs(hr)) > 1_000_000:
        raise ValueError("assignment HKL budget exceeded")
    h = hr.astype(np.int64) @ inverse.T
    if np.max(abs(h)) > 1_000_000:
        raise ValueError("input-basis HKL budget exceeded")
    parent = (error < options.q_tolerance) & np.any(h != 0, axis=1)
    if (parent & training).sum() < 12 or np.linalg.matrix_rank(h[parent & training]) < 3:
        raise ValueError("need >=12 rank-three parent training assignments")
    ntrain = int((parent & training).sum())
    nvalid = int((parent & ~training).sum())
    hnfs = tuple(reciprocal_hnfs(options.max_subcell_index))
    if len(hnfs) * len(points) > 5_000_000:
        raise ValueError(
            "primitive comparison snapshot budget exceeded (5 million position/hypothesis pairs)"
        )
    primitives = []
    for pid, hnf in enumerate(hnfs):
        # HNF entries/index <=8: adjugate and products with bounded HKLs
        # fit comfortably in i64. Membership is exact integer divisibility.
        denominator = int(hnf[0, 0] * hnf[1, 1] * hnf[2, 2])
        adjugate = np.column_stack(
            (np.cross(hnf[1], hnf[2]), np.cross(hnf[2], hnf[0]), np.cross(hnf[0], hnf[1]))
        )
        numerator = h @ adjugate
        hp = np.rint(numerator / denominator).astype(np.int64)
        retained = parent & np.all(numerator % denominator == 0, axis=1)
        fit = retained & training
        nt, nv = int(fit.sum()), int((retained & ~training).sum())
        supported = (
            nt >= 12
            and np.linalg.matrix_rank(hp[fit]) == 3
            and nt >= options.minimum_retention * ntrain
        )
        ap = np.linalg.inv(hnf).T @ a
        if nt >= 12 and np.linalg.matrix_rank(hp[fit]) == 3:
            bp = np.linalg.lstsq(hp[fit], points[fit], rcond=None)[0]
            if np.linalg.det(bp) > 1e-15:
                ap = np.linalg.inv(bp).T
            else:
                supported = False
        reason = (
            "supported within index bound"
            if supported
            else "insufficient retained rank-three training support"
        )
        primitives.append(
            PrimitiveHypothesis(
                pid, hnf, ap, hp, retained, nt, nv, ntrain - nt, nvalid - nv, supported, reason
            )
        )
    settings = []
    for primitive in primitives:
        if not primitive.supported:
            continue
        eligible = primitive.retained.copy()
        if observed is not None:
            initial = predict_spots(
                primitive.hkl, np.linalg.inv(primitive.direct_basis).T, geometry, observed[:, 2]
            )
            difference = initial.values - observed
            eligible &= (
                (initial.flags == 0)
                & (np.linalg.norm(difference[:, :2], axis=1) < options.initial_xy_gate)
                & (abs(difference[:, 2]) < options.initial_omega_gate)
            )
        if (eligible & training).sum() < 12:
            continue
        candidates = find_bravais_settings(
            primitive.direct_basis,
            max_strain=options.max_strain,
            max_angle=options.max_angle,
            coefficient_limit=options.coefficient_limit,
        )
        for setting in candidates:
            if (len(settings) + 1) * len(points) > 5_000_000:
                raise ValueError(
                    "setting comparison snapshot budget exceeded (5 million position/setting pairs)"
                )
            if progress is not None:
                progress(primitive.id, setting.symbol)
            settings.append(
                _fit_setting(
                    points, observed, geometry, training, primitive, setting, eligible, options
                )
            )
    finite = [i for i, s in enumerate(settings) if np.isfinite(s.training_score)]
    if not finite:
        raise ValueError("no finite, rank-complete metric fits in the bounded search")
    # Only training support and exact index decide the preferred primitive;
    # nested metric scores are compared on its one fixed population.
    available = {settings[i].primitive_id for i in finite}
    chosen = min(
        (p for p in primitives if p.id in available),
        key=lambda p: (-p.index, -p.training_count, p.id),
    )
    recommended = min(
        (i for i in finite if settings[i].primitive_id == chosen.id),
        key=lambda i: (settings[i].training_score, i),
    )
    return CellComparison(
        tuple(primitives),
        tuple(settings),
        h,
        parent,
        training,
        chosen.id,
        recommended,
        options,
        "reciprocal" if observed is None else "detector",
    )
