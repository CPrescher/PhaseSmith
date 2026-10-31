"""Training-only combination of already fitted, qualified hypothesis pools.

No acceptance labels, reserved observations, reference cells or chemistry are
inputs. Final validation remains the caller's separate unchanged stage.
"""

from dataclasses import dataclass

import numpy as np

from phasesmith import _core

from .grains import GrainSearchOptions, _rotations, _valid_basis
from .ownership import GrainOwnershipOptions, _physical_fields, assign_observed

__all__ = ["GrainPoolMergeOptions", "TrainingPoolMerge", "merge_fitted_training_pools"]


@dataclass(frozen=True)
class GrainPoolMergeOptions:
    control_rotations: int = 8
    max_additions: int = 40
    max_alias_index: int = 12

    def __post_init__(self):
        for name, low, high in [
            ("control_rotations", 8, 64),
            ("max_additions", 0, 40),
            ("max_alias_index", 1, 64),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")


@dataclass(frozen=True)
class TrainingPoolMerge:
    """Unvalidated fitted bases and origin indices; deliberately no accepted flag."""

    bases: np.ndarray
    source_kind: np.ndarray  # zero incumbent, one candidate
    source_index: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        for name in ("bases", "source_kind", "source_index"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def _equivalent(a, b, max_index):
    for x, y in ((a, b), (b, a)):
        transform = x @ np.linalg.inv(y)
        integer = np.rint(transform)
        if not 1 <= abs(round(np.linalg.det(integer))) <= max_index:
            continue
        predicted = integer @ y
        if np.max(abs(transform - integer)) >= 0.025 or np.linalg.norm(
            x - predicted
        ) >= 0.02 * np.linalg.norm(x):
            continue
        u, singular, vt = np.linalg.svd(np.linalg.solve(predicted, x))
        rotation = u @ vt
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        if np.linalg.det(rotation) > 0 and angle < 2 and np.max(abs(singular - 1)) < 0.02:
            return True
    return False


def merge_fitted_training_pools(
    points,
    incumbents,
    candidates,
    *,
    charts,
    chart_valid,
    search=None,
    ownership=None,
    options=None,
):
    """Freeze a larger fitted pool using only global-training observations.

    Both inputs must already have passed the upstream proposal qualification
    and training fit stages. In particular, the narrow-core proposal gate is
    not reapplied to a different *post-fit* basis. Physical support, distinct
    HKL/rank, training null and protected-incumbent ownership gates determine
    additions. Every incumbent, including one that may later fail validation,
    remains present and unchanged. Reuse each model's completed final fit;
    fitting or pruning on reserved acceptance outcomes is not part of this API.
    """
    search = GrainSearchOptions() if search is None else search
    ownership = (
        GrainOwnershipOptions(native_fields=True, competitive_selection=True)
        if ownership is None
        else ownership
    )
    options = GrainPoolMergeOptions() if options is None else options
    if not isinstance(search, GrainSearchOptions) or not isinstance(
        ownership, GrainOwnershipOptions
    ):
        raise TypeError("requires GrainSearchOptions and GrainOwnershipOptions")
    if not isinstance(options, GrainPoolMergeOptions):
        raise TypeError("options must be GrainPoolMergeOptions")
    q = np.asarray(points)
    if (
        q.dtype != np.float64
        or q.ndim != 2
        or q.shape[1:] != (3,)
        or not 1 <= len(q) <= 100000
        or not np.isfinite(q).all()
    ):
        raise ValueError("points must be finite float64 shape(N,3), 1<=N<=100000")
    arrays = []
    for data in (incumbents, candidates):
        a = np.asarray(data)
        if (
            a.dtype != np.float64
            or a.ndim != 3
            or a.shape[1:] != (3, 3)
            or len(a) > 10000
            or len(a) * len(q) > 8_000_000
            or not np.isfinite(a).all()
            or any(not _valid_basis(b, search) for b in a)
        ):
            raise ValueError("fitted pools must be finite valid float64 (M,3,3), <=8million pairs")
        arrays.append(a.copy())
    incumbents, candidates = arrays
    if len(incumbents) > search.max_grains:
        raise ValueError("incumbents exceed max_grains")
    charts, valid = np.asarray(charts), np.asarray(chart_valid)
    if (
        charts.dtype != np.float64
        or charts.shape != (len(q), 3, 3)
        or not np.isfinite(charts).all()
        or valid.dtype != bool
        or valid.shape != (len(q),)
    ):
        raise ValueError("requires finite float64 charts[N,3,3] and boolean chart_valid[N]")
    engine = _core.ExperimentalLatticeEngine(search.workers)
    models = list(incumbents)
    kinds, origins = [0] * len(models), list(range(len(models)))
    evidence, _, _ = _physical_fields(engine, q, models, charts, valid, search, ownership)
    ce, cc, cp = _physical_fields(engine, q, candidates, charts, valid, search, ownership)
    controls = []
    for rotation in _rotations(options.control_rotations, search.seed + 881):
        _, _, physical = _physical_fields(
            engine, q, candidates @ rotation.T, charts, valid, search, ownership
        )
        controls.append(physical.sum(axis=1))
    controls = np.asarray(controls).reshape(options.control_rotations, len(candidates))
    labels, _, _ = assign_observed(engine, q, models, charts, valid, search, ownership)
    initial = np.bincount(labels, minlength=len(models) + 1)[1:]
    lower = np.minimum(initial, search.min_indexed)
    best = np.max(evidence, axis=0) if len(models) else np.zeros(len(q))
    audit, added = [], []
    pending = list(range(len(candidates)))
    while pending:
        i = min(
            pending,
            key=lambda j: (
                -float(np.maximum(ce[j] - best, 0).sum()),
                abs(np.linalg.det(candidates[j])),
                j,
            ),
        )
        pending.remove(i)
        a = candidates[i]
        gain = float(np.maximum(ce[i] - best, 0).sum())
        record = dict(
            candidate=i,
            tight=int(cc[i].sum()),
            physical=int(cp[i].sum()),
            null_max=int(controls[:, i].max()),
            gain=gain,
            status="rejected",
        )
        reason = None
        if len(models) >= search.max_grains or len(added) >= options.max_additions:
            reason = "model_budget"
        elif cp[i].sum() < search.min_indexed:
            reason = "fitted_training_support"
        elif cp[i].sum() <= controls[:, i].max():
            reason = "training_control"
        elif gain <= 0:
            reason = "no_training_evidence_gain"
        elif any(_equivalent(a, b, options.max_alias_index) for b in models):
            reason = "oriented_cell_alternative"
        if reason is None:
            labels, _, _ = assign_observed(
                engine, q, [*models, a], charts, valid, search, ownership
            )
            counts = np.bincount(labels, minlength=len(models) + 2)[1:]
            if counts[-1] < search.min_indexed or np.any(counts[:-1] < lower):
                reason = "owned_training_guard"
            else:
                h = np.rint(q[labels == len(models) + 1] @ a.T)
                if len(np.unique(h, axis=0)) < search.min_indexed or np.linalg.matrix_rank(h) < 3:
                    reason = "training_rank"
        if reason is not None:
            record["reason"] = reason
        else:
            models.append(a)
            kinds.append(1)
            origins.append(i)
            added.append(i)
            best = np.maximum(best, ce[i])
            lower = np.r_[lower, search.min_indexed]
            record.update(status="added", owned=int(counts[-1]))
        audit.append(record)
    labels, _, _ = assign_observed(engine, q, models, charts, valid, search, ownership)
    final = np.bincount(labels, minlength=len(models) + 1)[1:]
    return TrainingPoolMerge(
        np.asarray(models, dtype=float).reshape(-1, 3, 3),
        np.asarray(kinds, np.int8),
        np.asarray(origins, np.int64),
        dict(
            candidates=audit,
            added=added,
            incumbent_before=initial.tolist(),
            incumbent_after=final[: len(incumbents)].tolist(),
            controls=controls.tolist(),
            upstream_proposal_gates_reused=True,
            scope="Global training only; no accepted flags or reserved data",
        ),
    )
