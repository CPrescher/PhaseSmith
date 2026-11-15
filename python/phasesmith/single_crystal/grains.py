"""Position-only FFT grain search with a frozen training/validation split.

No input HKLs, cell, chemistry, Bravais class, or space group. Native FFT and
ownership kernels; readable array orchestration; no Qt or external checkout.
"""

import math
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import combinations, product

import numpy as np

from phasesmith import _core

from . import discovery_cache, search_execution
from .indexing import IndexingResult, LatticeCandidate, cell_parameters, reduce_basis
from .ownership import GrainOwnershipOptions
from .reciprocal import lattice_fourier, triples
from .recovery import GrainRecoveryOptions

__all__ = [
    "GrainAliasOptions",
    "GrainConditioningOptions",
    "GrainProposalRefinementOptions",
    "GrainSearchOptions",
    "GrainStabilityOptions",
    "find_grains",
    "find_grains_ensemble",
]


@dataclass(frozen=True)
class _GrainTrainingPool:
    """Internal unvalidated output. It deliberately has no acceptance fields."""

    bases: np.ndarray
    labels: np.ndarray
    translation_vectors: np.ndarray
    translation_scores: np.ndarray
    diagnostics: tuple
    options: object

    def __post_init__(self):
        for name in ("bases", "labels", "translation_vectors", "translation_scores"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GrainSearchOptions:
    min_length: float = 1.5
    max_length: float = 30.0
    grid_size: int = 192
    fft_peaks: int = 192
    basis_vectors: int = 64
    max_pairs: int = 96
    cell_refinements: int = 32
    max_grains: int = 8
    workers: int = 4
    fractional_tolerance: float = 0.15  # translation proposal screening only
    reciprocal_tolerance: float = 0.015
    min_indexed: int = 20
    min_volume: float = 15.0
    max_volume: float = 20000.0
    max_condition: float = 20.0
    ambiguity_ratio: float = 0.8
    null_rotations: int = 64
    seed: int = 842024
    native_fourier_refinement: bool = False

    def __post_init__(self):
        if type(self.native_fourier_refinement) is not bool:
            raise ValueError("native_fourier_refinement must be boolean")
        integers = dict(
            grid_size=(16, 256),
            fft_peaks=(3, 400),
            basis_vectors=(3, 80),
            max_pairs=(1, 256),
            cell_refinements=(1, 80),
            max_grains=(1, 40),
            workers=(1, 4),
            min_indexed=(6, 100000),
            null_rotations=(8, 128),
            seed=(0, 2**32 - 1),
        )
        for name, (lo, hi) in integers.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
        if self.grid_size % 2 or self.basis_vectors > self.fft_peaks:
            raise ValueError("grid must be even and basis_vectors <= fft_peaks")
        for name in (
            "min_length",
            "max_length",
            "fractional_tolerance",
            "reciprocal_tolerance",
            "min_volume",
            "max_volume",
            "max_condition",
            "ambiguity_ratio",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_length < self.max_length <= 100:
            raise ValueError("require 0 < min_length < max_length <= 100 angstrom")
        if not 0 < self.fractional_tolerance < 0.5 or not 0 < self.reciprocal_tolerance < 1:
            raise ValueError("invalid positional tolerances")
        if not 0 < self.min_volume < self.max_volume <= 1e6 or not 1 < self.max_condition <= 100:
            raise ValueError("invalid volume or conditioning bounds")
        if not 0 < self.ambiguity_ratio < 1:
            raise ValueError("ambiguity_ratio must lie in (0,1)")


def _valid_basis(a, o):
    lengths = np.linalg.norm(a, axis=1)
    return (
        np.all(np.isfinite(a))
        and lengths.min() >= o.min_length
        and lengths.max() <= o.max_length
        and o.min_volume <= abs(np.linalg.det(a)) <= o.max_volume
        and np.linalg.cond(a) <= o.max_condition
    )


def _score(engine, q, a, tolerance):
    h, error = engine.score(q, a, np.linalg.inv(a).T)
    return h, error, (error < tolerance) & np.any(h != 0, axis=1)


def _labels(engine, q, models, o):
    if not models:
        return np.zeros(len(q), dtype=np.int32), 0, 0
    rows = np.asarray(models)
    return engine.assign(
        q, rows, np.linalg.inv(rows).transpose(0, 2, 1), o.reciprocal_tolerance, o.ambiguity_ratio
    )


def _vector_order(vectors, scores, o, diverse):
    order = np.argsort(-scores, kind="stable")
    if not diverse:
        return order
    # Keep half of the amplitude shortlist, then visit angular/radial strata.
    # A direction and its negative denote the same integer-phase constraint.
    length = np.linalg.norm(vectors, axis=1)
    axis = np.argmax(abs(vectors), axis=1)
    canonical = vectors * np.where(vectors[np.arange(len(vectors)), axis] < 0, -1, 1)[:, None]
    buckets = {}
    for i in order:
        other = [j for j in range(3) if j != axis[i]]
        angular = (
            int(axis[i]) * 4 + int(canonical[i, other[0]] < 0) * 2 + int(canonical[i, other[1]] < 0)
        )
        radial = min(2, max(0, int(3 * (length[i] - o.min_length) / (o.max_length - o.min_length))))
        buckets.setdefault((radial, angular), []).append(int(i))
    prefix = list(map(int, order[: o.basis_vectors // 2]))
    seen = set(prefix)
    cursors = dict.fromkeys(buckets, 0)
    while True:
        added = False
        for key in sorted(buckets):
            bucket = buckets[key]
            while cursors[key] < len(bucket) and bucket[cursors[key]] in seen:
                cursors[key] += 1
            if cursors[key] < len(bucket):
                i = bucket[cursors[key]]
                prefix.append(i)
                seen.add(i)
                cursors[key] += 1
                added = True
        if not added:
            return np.asarray(prefix, dtype=np.int64)


def _fourier_evaluator(engine, q):
    callback = engine.fourier if hasattr(engine, "fourier") else lattice_fourier
    ordinary = partial(callback, q)
    if not hasattr(engine, "prepare_fourier"):
        return ordinary
    cloud = None

    def evaluate(vectors):
        nonlocal cloud
        # Prepared coefficients help small late-Newton batches. Keep the
        # compact point-only path for large batches where their memory traffic
        # has not shown a reliable benefit. Both evaluate identical equations.
        if len(vectors) > 32:
            return ordinary(vectors)
        if cloud is None:
            cloud = engine.prepare_fourier(q)
        return cloud.evaluate(vectors)

    return evaluate


@discovery_cache.fourier
def _fourier_peaks(engine, q, o):
    n = o.grid_size
    step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * o.max_length))
    if step <= 0 or np.max(abs(q)) / step >= n / 2 - 1:
        raise ValueError(
            "FFT grid too small for reciprocal extent and maximum edge; "
            "increase grid or lower edge bound"
        )
    if hasattr(engine, "windowed_maxima"):
        initial, _, total = engine.windowed_maxima(
            q, n, step, o.min_length, o.max_length, o.fft_peaks
        )
    else:
        field = engine.build_field(q, n, step)
        initial, _, total = field.maxima(o.min_length, o.max_length, o.fft_peaks)
        del field
    if not len(initial):
        return (
            initial,
            np.empty(0),
            dict(
                reciprocal_grid_step=float(step),
                direct_grid_step=float(1 / (n * step)),
                fft_local_maxima=int(total),
            ),
        )
    vectors = initial.copy()
    # Native fused mean cosine, gradient and Hessian. Bounded Newton ascent
    # replaces the original study's L-BFGS-B without adding a SciPy dependency.
    radius = 1.5 / (n * step)
    if o.native_fourier_refinement:
        vectors, moments, _ = engine.refine_fourier(q, initial, radius)
        return (
            vectors,
            moments[:, 0],
            dict(
                reciprocal_grid_step=float(step),
                direct_grid_step=float(1 / (n * step)),
                fft_local_maxima=int(total),
            ),
        )
    fourier = _fourier_evaluator(engine, q)
    moments = fourier(vectors)
    alive = np.ones(len(vectors), dtype=bool)
    for _ in range(20):
        current = moments[alive]
        h = -current[:, 4:].reshape(-1, 3, 3)
        h += np.maximum(0.0, 0.01 - np.linalg.eigvalsh(h)[:, 0])[:, None, None] * np.eye(3)
        step_delta = np.linalg.solve(h, current[:, 1:4, None])[:, :, 0]
        step_delta *= np.minimum(1.0, 0.3 / np.maximum(np.linalg.norm(step_delta, axis=1), 1e-12))[
            :, None
        ]
        delta = np.zeros_like(vectors)
        delta[alive] = step_delta
        improved = np.zeros(len(vectors), dtype=bool)
        for scale in (1.0, 0.5, 0.25, 0.125):
            active = np.flatnonzero(alive & ~improved)
            if not len(active):
                break
            trial = np.clip(
                vectors[active] + scale * delta[active],
                initial[active] - radius,
                initial[active] + radius,
            )
            evaluated = fourier(trial)
            better = evaluated[:, 0] > moments[active, 0]
            accepted = active[better]
            vectors[accepted] = trial[better]
            moments[accepted] = evaluated[better]
            improved[accepted] = True
        if not np.any(improved):
            break
        # An unchanged vector has the same next step and all the same rejected
        # line trials forever. Other vectors cannot change its local objective.
        alive &= improved
    scores = moments[:, 0]
    return (
        vectors,
        scores,
        dict(
            reciprocal_grid_step=float(step),
            direct_grid_step=float(1 / (n * step)),
            fft_local_maxima=int(total),
        ),
    )


def _vectors(engine, q, o, *, diverse=False):
    vectors, scores, layout = _fourier_peaks(engine, q, o)
    chosen = []
    lengths = np.linalg.norm(vectors, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = abs(vectors @ vectors.T) / (lengths[:, None] * lengths[None, :])
    for i in _vector_order(vectors, scores, o, diverse):
        length = lengths[i]
        if not o.min_length <= length <= o.max_length:
            continue
        comparisons = cosine[i, chosen]
        # A three-term dot/norm quotient has a small roundoff envelope. Preserve
        # the original scalar arithmetic whenever a comparison could cross its
        # strict boundary; away from it the batched decision is identical.
        near = abs(comparisons - 0.9995) <= 32 * np.finfo(float).eps
        if np.any(near):
            comparisons = comparisons.copy()
            for k in np.flatnonzero(near):
                j = chosen[k]
                comparisons[k] = abs(np.dot(vectors[i], vectors[j])) / (
                    length * np.linalg.norm(vectors[j])
                )
        if np.any(comparisons > 0.9995):
            continue
        chosen.append(int(i))
        if len(chosen) >= o.basis_vectors:
            break
    return (
        vectors[chosen],
        scores[chosen],
        layout,
    )


def _triplets(q, vectors, o, engine=None, *, limit=None):
    phase = q @ vectors.T
    support = (abs(phase - np.rint(phase)) < o.fractional_tolerance).T
    engine = _core.ExperimentalLatticeEngine(o.workers) if engine is None else engine
    bits = (
        engine.pack_support(support)
        if hasattr(engine, "pack_support") and len(vectors) <= 400
        else None
    )
    native_ranking = bits is not None and hasattr(bits, "candidate_keys")
    if native_ranking:
        keys = bits.candidate_keys(o.max_pairs)
    else:
        counts = (
            bits.pair_counts().astype(np.int64)
            if bits is not None
            else support.astype(float) @ support.T.astype(float)
        )
        keys = _triplet_keys(counts, len(vectors), o.max_pairs)
    if not len(keys):
        return []
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("triplet limit must be a positive integer or None")
    rows = vectors[keys]
    lengths = np.linalg.norm(rows, axis=2)
    good = (
        np.isfinite(rows).all(axis=(1, 2))
        & (lengths.min(axis=1) >= o.min_length)
        & (lengths.max(axis=1) <= o.max_length)
    )
    keys, rows = keys[good], rows[good]
    volume = np.abs(np.linalg.det(rows))
    good = (volume >= o.min_volume) & (volume <= o.max_volume)
    keys, rows, volume = keys[good], rows[good], volume[good]
    if native_ranking:
        order, totals = bits.rank_triplets(np.ascontiguousarray(keys), volume)
        if limit is None:
            order = order[np.linalg.cond(rows[order]) <= o.max_condition] if len(order) else order
        selected = []
        for i in order:
            if limit is not None and np.linalg.cond(rows[i]) > o.max_condition:
                continue
            selected.append((int(totals[i]), float(volume[i]), tuple(map(int, keys[i]))))
            if limit is not None and len(selected) >= limit:
                break
        return selected
    if bits is not None:
        totals = bits.triplet_counts(np.ascontiguousarray(keys, dtype=np.int64)).tolist()
    else:
        totals = []
        for start in range(0, len(keys), 128):
            batch = keys[start : start + 128]
            totals.extend(
                np.count_nonzero(
                    support[batch[:, 0]] & support[batch[:, 1]] & support[batch[:, 2]], axis=1
                ).tolist()
            )
    records = [
        (count, float(vol), tuple(map(int, key)))
        for count, vol, key in zip(totals, volume, keys, strict=True)
    ]
    records.sort(key=lambda r: (-r[0], r[1], r[2]))
    if limit is None:
        if not records:
            return []
        rows = vectors[np.asarray([record[2] for record in records])]
        good = np.linalg.cond(rows) <= o.max_condition
        return [record for record, keep in zip(records, good, strict=True) if keep]
    selected = []
    for record in records:
        if np.linalg.cond(vectors[list(record[2])]) <= o.max_condition:
            selected.append(record)
            if len(selected) >= limit:
                break
    return selected


def _triplet_keys(counts, count, max_pairs):
    """Readable fallback for the exact strongest-pair candidate set."""
    pairs = sorted(combinations(range(count), 2), key=lambda p: (-counts[p], p))[:max_pairs]
    if not pairs:
        return []
    # Batched three-by-three geometry and bounded support chunks retain the
    # original candidate set, NumPy SVD/determinant conventions and final order.
    keys = np.concatenate(
        [
            np.column_stack((np.full(count, i), np.full(count, j), np.arange(count)))
            for i, j in pairs
        ]
    )
    keys.sort(axis=1)
    keys = np.unique(keys, axis=0)
    keys = keys[(keys[:, 0] != keys[:, 1]) & (keys[:, 1] != keys[:, 2])]
    return keys


def integer_lattice_basis(indices, *, _stop_if_primitive=False):
    """Exact Euclidean row operations on inferred HKLs; no supplied indexing.

    Returns the integer lattice generated by these rows, or None for rank <3.
    Used to propose smaller direct cells, never as proof of primitivity.
    The private index-only path may return as soon as the rows generate Z^3;
    later integer rows cannot enlarge that lattice. The default retains the
    original complete row operations and returned basis representation.
    """
    h = np.asarray(indices)
    if (
        h.ndim != 2
        or h.shape[1] != 3
        or h.dtype.kind not in "iu"
        or np.any(h > 1000000)
        or (h.dtype.kind == "i" and np.any(h < -1000000))
    ):
        raise ValueError("indices must be integer (n,3) with absolute values <=1000000")
    basis = [None] * 3
    for item in h:
        row = [int(v) for v in item]
        for column in range(3):
            while row[column]:
                if basis[column] is None:
                    basis[column] = row if row[column] > 0 else [-v for v in row]
                    row = [0, 0, 0]
                    break
                pivot = basis[column]
                quotient = row[column] // pivot[column]
                row = [a - quotient * b for a, b in zip(row, pivot, strict=True)]
                if row[column]:
                    if row[column] < 0:
                        row = [-v for v in row]
                    basis[column], row = row, pivot
        if _stop_if_primitive and all(
            pivot is not None and pivot[i] == 1 for i, pivot in enumerate(basis)
        ):
            return np.eye(3, dtype=np.int64)
    if any(row is None for row in basis) or any(abs(v) > 2**53 for row in basis for v in row):
        return None
    return np.asarray(basis, dtype=np.int64)


def _fitting_selection(engine, q, a, tolerance, cloud=None):
    if cloud is not None:
        return cloud.selected(a, np.linalg.inv(a).T, tolerance)
    h, error, selected = _score(engine, q, a, tolerance)
    ids = np.flatnonzero(selected)
    return h[selected], q[selected], ids, error[selected]


def _refine(engine, q, a, o, *, integer_cores=(24, 48, 96), _cloud=None):
    a = a.copy()
    previous_inputs = previous_update = None
    for cycle in range(8):
        h, selected_q, ids, _ = _fitting_selection(
            engine, q, a, o.reciprocal_tolerance * (2 if cycle < 2 else 1), _cloud
        )
        if len(ids) < 6:
            return None
        # Keep every selection/cycle and convergence comparison. Only repeat
        # identical ordered linear algebra once; byte keys retain signed zeros.
        inputs = (
            h.shape,
            h.dtype.str,
            h.tobytes(),
            selected_q.shape,
            selected_q.dtype.str,
            selected_q.tobytes(),
        )
        if inputs == previous_inputs:
            updated = previous_update
        else:
            if np.linalg.matrix_rank(h) < 3:
                return None
            reciprocal = np.linalg.lstsq(h, selected_q, rcond=None)[0]
            if abs(np.linalg.det(reciprocal)) < 1e-12:
                return None
            updated = np.linalg.inv(reciprocal).T
            previous_inputs, previous_update = inputs, updated
        delta = np.max(abs(a - updated))
        a = updated
        if cycle >= 2 and delta < 1e-8:
            break
    a = reduce_basis(a)
    if not _valid_basis(a, o):
        return None
    h, _, ids, error = _fitting_selection(engine, q, a, o.reciprocal_tolerance, _cloud)
    selected = np.zeros(len(q), dtype=bool)
    selected[ids] = True
    order = np.argsort(error, kind="stable")
    alternatives = [a]
    for size in integer_cores:
        if len(order) < size:
            continue
        integer = integer_lattice_basis(h[order[:size]].astype(np.int64), _stop_if_primitive=True)
        if integer is None or abs(round(np.linalg.det(integer))) <= 1:
            continue
        candidate = reduce_basis(np.linalg.inv(integer @ np.linalg.inv(a).T).T)
        if not _valid_basis(candidate, o):
            continue
        _, _, mask = _score(engine, q, candidate, o.reciprocal_tolerance)
        if np.count_nonzero(mask & selected) >= 0.9 * selected.sum():
            alternatives.append(candidate)
    return min(alternatives, key=lambda b: abs(np.linalg.det(b)))


def _refine_many(engine, q, bases, options, *, integer_cores=(24, 48, 96)):
    """Batch only the unchanged first-cycle rejection, keeping every survivor."""
    if not len(bases):
        return []
    if not len(q):
        return [None] * len(bases)
    if hasattr(engine, "initial_support"):
        # Keep the scalar inverse convention of the original first _score.
        reciprocal = np.array([np.linalg.inv(a).T for a in bases])
        counts = engine.initial_support(
            q, np.asarray(bases), reciprocal, 2 * options.reciprocal_tolerance
        )
    else:
        counts = np.full(len(bases), 6)
    cloud = engine.prepare_fitting(q) if hasattr(engine, "prepare_fitting") else None
    return [
        None
        if count < 6
        else _refine(engine, q, a, options, integer_cores=integer_cores, _cloud=cloud)
        for a, count in zip(bases, counts, strict=True)
    ]


def _joint(engine, q, models, o):
    labels, _, _ = _labels(engine, q, models, o)
    history = []
    for cycle in range(8):
        updated = []
        stable_h = True
        for i, a in enumerate(models, 1):
            block = q[labels == i]
            h, _, _ = _score(engine, block, a, o.reciprocal_tolerance)
            b = a
            if len(block) >= 6 and np.linalg.matrix_rank(h) == 3:
                reciprocal = np.linalg.lstsq(h, block, rcond=None)[0]
                if abs(np.linalg.det(reciprocal)) > 1e-12:
                    trial = np.linalg.inv(reciprocal).T
                    if _valid_basis(trial, o):
                        b = trial
            stable_h &= np.array_equal(h, np.rint(block @ b.T))
            updated.append(b)
        assigned, ambiguous, multiple = _labels(engine, q, updated, o)
        changed = int(np.count_nonzero(labels != assigned))
        history.append(dict(cycle=cycle, changed=changed, ambiguous=ambiguous, multiple=multiple))
        models, labels = updated, assigned
        if not changed and stable_h:
            break
    return models, labels, history


def _rotations(count, seed):
    quat = np.random.default_rng(seed).normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    x, y, z, w = quat.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _discover(fit, o, progress=None):
    """Training-only proposals and reassignment; no reserved coordinates."""
    engine = _core.ExperimentalLatticeEngine(o.workers)
    models, diagnostics = [], []
    labels = np.zeros(len(fit), dtype=np.int32)
    vectors, scores = np.empty((0, 3)), np.empty(0)
    for iteration in range(o.max_grains):
        if progress:
            progress(iteration, o.max_grains)
        explained = np.zeros(len(fit), dtype=bool)
        for a in models:
            explained |= _score(engine, fit, a, o.reciprocal_tolerance)[2]
        block = fit[(labels == 0) & ~explained]
        if len(block) < o.min_indexed:
            break
        v, s, layout = _vectors(engine, block, o)
        if iteration == 0:
            vectors, scores = v, s
        records = _triplets(block, v, o, engine, limit=o.cell_refinements)
        choices, audit = [], []
        starts = [v[list(ids)] for _, _, ids in records[: o.cell_refinements]]
        refined = _refine_many(engine, block, starts, o)
        for (_, _, ids), a in zip(records[: o.cell_refinements], refined, strict=True):
            if a is None:
                audit.append(dict(vectors=list(ids), valid=False))
                continue
            _, error, mask = _score(engine, block, a, o.reciprocal_tolerance)
            duplicate = (
                any(
                    np.mean(_score(engine, block[mask], old, o.reciprocal_tolerance)[2]) >= 0.8
                    for old in models
                )
                if np.any(mask)
                else True
            )
            count = int(mask.sum())
            volume = abs(float(np.linalg.det(a)))
            excess = count - len(block) * volume * 4 * np.pi / 3 * o.reciprocal_tolerance**3
            rms = float(np.sqrt(np.mean(error[mask] ** 2))) if count else None
            valid = count >= o.min_indexed and not duplicate
            audit.append(
                dict(
                    vectors=list(ids),
                    valid=valid,
                    duplicate=bool(duplicate),
                    basis=a.tolist(),
                    count=count,
                    volume=volume,
                    rms=rms,
                    excess=float(excess),
                )
            )
            if valid:
                choices.append((-excess, rms, volume, len(audit) - 1, a))
        diagnostic = dict(
            iteration=iteration + 1,
            residual_training=len(block),
            layout=layout,
            returned_triplets=len(records),
            triplet_limit=o.cell_refinements,
            candidates=audit,
        )
        diagnostics.append(diagnostic)
        if not choices:
            break
        best = min(choices, key=lambda r: r[:4])
        diagnostic["selected"] = best[3]
        models.append(best[-1])
        models, labels, joint = _joint(engine, fit, models, o)
        diagnostic["reassignment"] = joint
    return models, labels, vectors, scores, diagnostics


def _finish(
    q,
    train,
    models,
    labels,
    vectors,
    scores,
    diagnostics,
    o,
    progress=None,
    *,
    method="fft_grains",
    ownership=None,
    charts=None,
    chart_valid=None,
):
    engine = _core.ExperimentalLatticeEngine(o.workers)
    fit = q[train]
    # First use of held-out coordinates. Pool/order stays fixed, even on failure.
    if ownership is None:
        held_labels, ambiguous, multiple = _labels(engine, q[~train], models, o)
    else:
        from .ownership import assign_observed

        labels, _, _ = assign_observed(
            engine, q[train], models, charts[train], chart_valid[train], o, ownership
        )
        held_labels, ambiguous, multiple = assign_observed(
            engine, q[~train], models, charts[~train], chart_valid[~train], o, ownership
        )
    null = []
    for rotation in _rotations(o.null_rotations, o.seed + 1):
        if ownership is None:
            control, _, _ = _labels(engine, q[~train] @ rotation, models, o)
        else:
            control, _, _ = assign_observed(
                engine,
                q[~train] @ rotation,
                models,
                charts[~train] @ rotation,
                chart_valid[~train],
                o,
                ownership,
            )
        null.append(np.bincount(control, minlength=len(models) + 1)[1:])
    null = np.asarray(null).reshape(o.null_rotations, len(models))
    all_labels = np.zeros(len(q), dtype=np.int32)
    all_labels[train], all_labels[~train] = labels, held_labels
    candidates = []
    for i, a in enumerate(models, 1):
        h, error, _ = _score(engine, q, a, o.reciprocal_tolerance)
        indexed = all_labels == i
        ntrain, ntest = (
            int(np.count_nonzero(indexed & train)),
            int(np.count_nonzero(indexed & ~train)),
        )
        controls = null[:, i - 1]
        mean = float(controls.mean())
        sd = math.sqrt(max(float(controls.var(ddof=1)), mean, 1.0))
        checks = dict(
            training=ntrain >= o.min_indexed,
            heldout=ntest >= 6,
            rotated_max=ntest > int(controls.max()),
            fold=ntest >= 2 * max(mean, 1.0),
            null_sd=ntest >= mean + 4 * sd,
            retention=ntest >= 0.5 * ntrain * np.count_nonzero(~train) / len(fit),
            rank=len(np.unique(h[indexed & train], axis=0)) >= o.min_indexed
            and np.linalg.matrix_rank(h[indexed & train]) == 3,
        )
        checks = {name: bool(value) for name, value in checks.items()}
        cell = cell_parameters(a)
        h = h.astype(np.int64)
        for arr in (a, cell, h, indexed, error):
            arr.flags.writeable = False
        candidates.append(
            LatticeCandidate(
                a,
                cell,
                abs(float(np.linalg.det(a))),
                h,
                indexed,
                error,
                ntrain / len(fit),
                ntest / np.count_nonzero(~train),
                int(np.count_nonzero(~train)),
                float(controls.max() / np.count_nonzero(~train)),
                bool(all(checks.values())),
            )
        )
        diagnostics.append(
            dict(
                model=i,
                acceptance=checks,
                null_counts=controls.tolist(),
                heldout_ambiguous=ambiguous,
                heldout_multiple=multiple,
            )
        )
    for arr in (train, vectors, scores, all_labels):
        arr.flags.writeable = False
    if progress:
        progress(o.max_grains, o.max_grains)
    passed = sum(c.accepted for c in candidates)
    message = (
        f"{len(candidates)} grain candidates; {passed} pass held-out checks. "
        "Same-cell orientations may represent different grains; "
        "primitivity and symmetry remain unproven."
    )
    return IndexingResult(
        tuple(candidates),
        train,
        vectors,
        scores,
        message,
        o,
        method=method,
        labels=all_labels,
        diagnostics=tuple(diagnostics),
    )


def find_grains(
    points, *, training=None, options=None, progress=None, method="fft", row_options=None
):
    """Discover up to max_grains competing lattices; validate a frozen pool.

    Labels are one-based candidate IDs; zero means residual or ambiguous.
    Rejected models remain in the pool/export to preserve validation competition.
    All discovery, HKL inference, refinement and selection use training only.
    `progress(done, total)` is a synchronous observer between discovery rounds;
    exceptions propagate, allowing cancellation without GUI dependencies.
    method="rows" selects Row search (DAFi-based) grouping before per-group
    lattice inference and the existing frozen-pool validation. Its exact raw
    grouping is retained in result.row_grouping; final lattice ownership is
    separate. The default FFT path is unchanged.
    """
    if method not in ("fft", "rows"):
        raise ValueError("method must be 'fft' or 'rows'")
    if method == "fft" and row_options is not None:
        raise ValueError("row_options requires method='rows'")
    o = GrainSearchOptions() if options is None else options
    if not isinstance(o, GrainSearchOptions):
        raise TypeError("options must be GrainSearchOptions")
    q = triples(points, "points")
    if len(q) > 100000 or np.max(abs(q), initial=0) > 1000:
        raise ValueError(
            "bounded search requires <=100000 positions and |q|<=1000 inverse angstrom"
        )
    train = np.arange(len(q)) % 5 != 0 if training is None else np.asarray(training)
    if train.dtype != bool or train.shape != (len(q),):
        raise ValueError("training must be boolean shape (n,)")
    train = train.copy()
    fit = q[train]
    if len(fit) < o.min_indexed or np.count_nonzero(~train) < 6:
        raise ValueError("insufficient training or held-out positions")
    if not hasattr(_core, "ExperimentalLatticeEngine"):
        raise RuntimeError("FFT grain search requires a PhaseSmith spots build")
    if method == "rows":
        from .row_grains import discover_rows

        models, labels, vectors, scores, diagnostics, grouping = discover_rows(
            fit, o, row_options, progress
        )
        result = _finish(
            q,
            train,
            models,
            labels,
            vectors,
            scores,
            diagnostics,
            o,
            progress,
            method="row_grains",
        )
        return replace(result, row_grouping=grouping)
    models, labels, vectors, scores, diagnostics = _discover(fit, o, progress)
    return _finish(q, train, models, labels, vectors, scores, diagnostics, o, progress)


def ensemble_search_options(options):
    """A fixed 28-search geometric grid; no supplied material or learned test cell."""
    if not isinstance(options, GrainSearchOptions):
        raise TypeError("options must be GrainSearchOptions")
    options = replace(options, max_grains=min(options.max_grains, 8))
    searches = []
    for length, tolerance, phase in product(
        (1 / 3, 0.5, 2 / 3, 1.0), (0.2, 8 / 15, 1.0), (0.06, 0.15)
    ):
        maximum = options.max_length * length
        if maximum <= options.min_length:
            continue
        searches.append(
            replace(
                options,
                max_length=maximum,
                reciprocal_tolerance=options.reciprocal_tolerance * tolerance,
                fractional_tolerance=phase,
            )
        )
    for tolerance, volume in product((0.2, 1.0), (0.05, 0.15)):
        maximum = options.max_volume * volume
        if maximum > options.min_volume:
            searches.append(
                replace(
                    options,
                    max_volume=maximum,
                    reciprocal_tolerance=options.reciprocal_tolerance * tolerance,
                )
            )
    return tuple(searches)


def _consolidate(engine, fit, models, options):
    """Training-only marginal positional evidence, then existing joint ownership.

    A large direct cell has a denser reciprocal lattice. Dividing the local
    Gaussian by its volume prevents equally close supercells receiving the same
    proposal evidence. This is a bounded ranking heuristic, not a calibrated
    probability or an integrated-reflection-intensity model.
    """
    if not models:
        return [], np.zeros(len(fit), np.int32), dict(selected=[], candidates=[])
    if len(fit) * len(models) > 8_000_000:
        raise ValueError("ensemble consolidation exceeds 8 million position/model pairs")
    sigma = options.reciprocal_tolerance / 4
    errors = []
    for a in models:
        h, error = engine.score(fit, a, np.linalg.inv(a).T)
        error[~np.any(h != 0, axis=1)] = np.inf
        errors.append(error)
    errors = np.asarray(errors)
    volumes = np.abs(np.linalg.det(models))
    masks = errors < sigma
    counts = masks.sum(axis=1)
    evidence = np.logaddexp(
        0.0, -0.5 * (errors / sigma) ** 2 - np.log(volumes[:, None] * (2 * np.pi) ** 1.5 * sigma**3)
    )
    previous = np.zeros(len(fit))
    covered = np.zeros(len(fit), bool)
    selected, audit = [], []
    for _ in range(options.max_grains):
        choices = []
        for i in range(len(models)):
            if i in selected or counts[i] < options.min_indexed:
                continue
            # Symmetric overlap: a broad superset must not erase a smaller,
            # coherent candidate solely because it contains its positions.
            if any(
                np.count_nonzero(masks[i] & masks[j]) >= 0.8 * max(counts[i], counts[j])
                for j in selected
            ):
                continue
            novel = int(np.count_nonzero(masks[i] & ~covered))
            if novel < options.min_indexed:
                continue
            gain = float(np.maximum(evidence[i] - previous, 0.0).sum())
            if gain > 0:
                choices.append((-gain, volumes[i], i, novel))
        if not choices:
            break
        neg_gain, _, i, novel = min(choices)
        selected.append(i)
        previous = np.maximum(previous, evidence[i])
        covered |= masks[i]
        audit.append(
            dict(
                pool_id=i,
                core_training=int(counts[i]),
                novel_training=novel,
                marginal_evidence=-neg_gain,
            )
        )
    chosen, labels, history = _joint(engine, fit, [models[i] for i in selected], options)
    return (
        chosen,
        labels,
        dict(
            selected=audit, joint_reassignment=history, core_tolerance=sigma, pool_size=len(models)
        ),
    )


@dataclass(frozen=True)
class GrainConditioningOptions:
    """Optional training-only Fourier-plane proposals for sparse lattices.

    A selector retains |q.r-rint(q.r)| < phase_tolerance. It is not an
    assignment or an acceptance test. The final ensemble ownership, refinement
    and validation still use its original GrainSearchOptions.
    """

    max_seeds: int = 80
    phase_tolerance: float = 0.08
    seed_search: GrainSearchOptions | None = None
    subset_search: GrainSearchOptions | None = None
    candidates_per_seed: int = 1
    rank_by_population: bool = False

    def __post_init__(self):
        if type(self.rank_by_population) is not bool:
            raise ValueError("rank_by_population must be boolean")
        if isinstance(self.max_seeds, bool) or not isinstance(self.max_seeds, int):
            raise ValueError("max_seeds must be an integer in [1,80]")
        if not 1 <= self.max_seeds <= 80:
            raise ValueError("max_seeds must be an integer in [1,80]")
        if type(self.candidates_per_seed) is not int or not 1 <= self.candidates_per_seed <= 3:
            raise ValueError("candidates_per_seed must be an integer in [1,3]")
        if not math.isfinite(self.phase_tolerance) or not 0 < self.phase_tolerance < 0.5:
            raise ValueError("conditioning phase_tolerance must lie in (0,0.5)")
        for search in (self.seed_search, self.subset_search):
            if search is not None and not isinstance(search, GrainSearchOptions):
                raise TypeError("conditioning searches must be GrainSearchOptions or None")


def _conditioned_seed(engine, fit, block, vector, seed_id, options, subset, diverse):
    record = dict(seed=seed_id, vector=vector.tolist(), training_subset=len(block))
    if len(block) < subset.min_indexed:
        record["status"] = "insufficient_subset"
        return record, []
    # Grid expansion only addresses reciprocal extent; scientific edge and
    # phase bounds stay fixed. Never crop q to make a grid fit.
    actual = subset
    step = min(2 * np.max(abs(block)) / (actual.grid_size - 4), 1 / (2.1 * actual.max_length))
    if np.max(abs(block)) / step >= actual.grid_size / 2 - 1:
        actual = replace(actual, grid_size=256)
    v, _, _ = _vectors(engine, block, actual, diverse=diverse)
    triplets = _triplets(block, v, actual, engine, limit=actual.cell_refinements)
    choices, candidates = [], []
    starts = [v[list(ids)] for _, _, ids in triplets[: actual.cell_refinements]]
    refined = _refine_many(engine, block, starts, actual, integer_cores=(8, 12, 16, 20, 24, 48, 96))
    for (_, _, ids), a in zip(triplets[: actual.cell_refinements], refined, strict=True):
        if a is None or not _valid_basis(a, options):
            continue
        _, error, core = _score(engine, fit, a, actual.reciprocal_tolerance)
        count = int(core.sum())
        if count < options.min_indexed:
            continue
        volume = abs(float(np.linalg.det(a)))
        excess = count - len(fit) * volume * 4 * np.pi / 3 * actual.reciprocal_tolerance**3
        rms = float(np.sqrt(np.mean(error[core] ** 2)))
        candidates.append(
            dict(
                vectors=list(ids),
                training_core=count,
                volume=volume,
                excess=float(excess),
                rms=rms,
                basis=a.tolist(),
            )
        )
        choices.append((-excess, rms, volume, len(candidates) - 1, a))
    record.update(
        status="completed",
        grid_size=actual.grid_size,
        returned_triplets=len(triplets),
        triplet_limit=actual.cell_refinements,
        candidates=candidates,
    )
    return record, choices


def _conditioning_options(options, conditioning):
    # Half the caller's edge bound is a geometric search window, not a cell
    # supplied by the comparator. All defaults and overrides are exported.
    seed = conditioning.seed_search
    if seed is None:
        maximum = options.max_length / 2
        if maximum <= options.min_length:
            raise ValueError("conditioning half-edge window is empty; supply seed_search")
        seed = replace(
            options,
            max_length=maximum,
            fft_peaks=400,
            basis_vectors=80,
            reciprocal_tolerance=options.reciprocal_tolerance / 5,
            max_grains=1,
        )
    seed = replace(seed, workers=options.workers)
    subset = conditioning.subset_search
    if subset is None:
        subset = replace(
            seed,
            grid_size=min(seed.grid_size, 128),
            fft_peaks=96,
            basis_vectors=40,
            max_pairs=64,
            cell_refinements=16,
        )
    subset = replace(subset, workers=options.workers)
    return seed, subset


def _conditioned_proposals(
    engine, fit, options, conditioning, progress=None, *, diverse=False, prepared=None, _defer=False
):
    seed, subset = _conditioning_options(options, conditioning)
    if prepared is None:
        vectors, scores, layout = _vectors(engine, fit, seed, diverse=diverse)
    else:
        # Private within-call reuse: caller owns the exact same q/seed settings.
        vectors, scores, layout = prepared
    phase = (
        np.column_stack([fit @ vector for vector in vectors])
        if len(vectors)
        else np.empty((len(fit), 0))
    )
    phase_counts = np.count_nonzero(
        abs(phase - np.rint(phase)) < conditioning.phase_tolerance, axis=0
    )
    seed_order = (
        np.argsort(-phase_counts, kind="stable")
        if conditioning.rank_by_population
        else np.arange(len(vectors))
    )
    jobs = []
    for i in seed_order[: conditioning.max_seeds]:
        i = int(i)
        selected = abs(phase[:, i] - np.rint(phase[:, i])) < conditioning.phase_tolerance
        jobs.append((fit, fit[selected], vectors[i], i, options, subset, diverse))
    session = search_execution.active()

    def serial_results():
        for visit, job in enumerate(jobs):
            if progress:
                progress(visit, min(len(vectors), conditioning.max_seeds))
            yield _conditioned_seed(engine, *job)

    results = session.map("conditioned_seed", jobs) if session is not None else serial_results()

    def merge_results():
        models, audit = [], []
        for visit, (record, choices) in enumerate(results):
            if progress and session is not None:
                progress(visit, min(len(vectors), conditioning.max_seeds))
            if record["status"] == "insufficient_subset":
                audit.append(record)
                continue
            if choices:
                choices.sort(key=lambda r: r[:4])
                best = choices[0]
                record["selected"] = best[3]
                selected = []
                for choice in choices:
                    if len(selected) >= conditioning.candidates_per_seed or len(models) >= 80:
                        break
                    if any(_same_oriented_basis(choice[-1], old) for old in selected):
                        continue
                    selected.append(choice[-1])
                    models.append(choice[-1])
                record["selected_all"] = [
                    c[3] for c in choices if any(c[-1] is a for a in selected)
                ]
            audit.append(record)
        return models, dict(
            seed_search=asdict(seed),
            subset_search=asdict(subset),
            phase_tolerance=conditioning.phase_tolerance,
            max_seeds=conditioning.max_seeds,
            candidates_per_seed=conditioning.candidates_per_seed,
            rank_by_population=conditioning.rank_by_population,
            phase_counts=phase_counts.tolist(),
            seed_order=seed_order.tolist(),
            integer_cores=[8, 12, 16, 20, 24, 48, 96],
            layout=layout,
            seed_vectors=vectors.tolist(),
            seed_scores=scores.tolist(),
            seeds=audit,
            proposed=len(models),
        )

    return merge_results if _defer else merge_results()


def _same_oriented_basis(a, b):
    """Strict training-proposal deduplication, including unimodular row changes."""
    transform = a @ np.linalg.inv(b)
    integer = np.rint(transform)
    return (
        abs(round(np.linalg.det(integer))) == 1
        and np.max(abs(transform - integer)) < 0.005
        and np.linalg.norm(a - integer @ b) < 0.002 * np.linalg.norm(a)
    )


@dataclass(frozen=True)
class GrainAliasOptions:
    """Training-only homogeneous HKL congruences propose smaller direct cells."""

    primes: tuple[int, ...] = (2, 3)
    max_index: int = 12
    max_depth: int = 3
    max_parents: int = 64
    max_children: int = 96
    min_points: int = 18
    min_fraction: float = 0.8

    def __post_init__(self):
        if (
            not isinstance(self.primes, tuple)
            or not self.primes
            or any(type(p) is not int or p not in (2, 3, 5, 7) for p in self.primes)
            or tuple(sorted(set(self.primes))) != self.primes
        ):
            raise ValueError("alias primes must be a sorted unique tuple drawn from 2,3,5,7")
        for name, lo, hi in (
            ("max_index", 2, 24),
            ("max_depth", 1, 4),
            ("max_parents", 1, 128),
            ("max_children", 1, 128),
            ("min_points", 17, 100000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
        if not math.isfinite(self.min_fraction) or not 0.5 < self.min_fraction <= 1:
            raise ValueError("alias min_fraction must lie in (0.5,1]")


def prime_congruence_bases(primes=(2, 3)):
    """Canonical index-p kernels {h in Z³: h.n=0 mod p}, by first principles.

    Normal vectors are projective: their first nonzero coefficient is one.
    H has rows e_i-n_i e_j (i!=j), p e_j. Thus det(H)=p and h=kH
    exactly when h.n is divisible by p. This is a proposal enumeration only.
    """
    if (
        not isinstance(primes, tuple)
        or not primes
        or any(type(p) is not int or p not in (2, 3, 5, 7) for p in primes)
        or tuple(sorted(set(primes))) != primes
    ):
        raise ValueError("primes must be a sorted unique tuple drawn from 2,3,5,7")
    result = []
    for p in primes:
        for values in product(range(p), repeat=3):
            normal = np.asarray(values, dtype=np.int64)
            nonzero = np.flatnonzero(normal)
            if not len(nonzero) or normal[nonzero[0]] != 1:
                continue
            pivot = int(nonzero[0])
            h = np.eye(3, dtype=np.int64)
            h[:, pivot] = -normal
            h[pivot, pivot] = p
            result.append((p, normal, h, pivot))
    return tuple(result)


def _alias_proposals(engine, fit, observed, geometry, models, options, aliases, progress):
    from .prediction import RefinementOptions, predict_spots, refine_crystal

    if len(fit) * (len(models) + aliases.max_children) > 8_000_000:
        raise ValueError("ensemble alias proposals exceed 8 million position/model pairs")
    kernels = prime_congruence_bases(aliases.primes)
    native = RefinementOptions()

    def plans(a, index):
        h, _, near = _score(engine, fit, a, options.reciprocal_tolerance)
        predicted = predict_spots(h.astype(np.int64), np.linalg.inv(a).T, geometry, observed[:, 2])
        delta = predicted.values - observed
        strict = near & (predicted.flags == 0) & np.isfinite(delta).all(axis=1)
        strict &= (np.linalg.norm(delta[:, :2], axis=1) <= 3) & (abs(delta[:, 2]) <= 0.5)
        ids = np.flatnonzero(strict)
        if len(ids) < aliases.min_points or np.linalg.matrix_rank(h[ids]) < 3:
            return []
        choices = []
        for ki, (p, normal, kernel, pivot) in enumerate(kernels):
            if index * p > aliases.max_index:
                continue
            keep = h[ids] @ normal % p == 0
            count = int(keep.sum())
            if count < aliases.min_points or count < aliases.min_fraction * len(ids):
                continue
            selected = ids[keep]
            hk = h[selected].astype(np.int64)
            hk[:, pivot] = (h[selected] @ normal).astype(np.int64) // p
            if np.linalg.matrix_rank(hk) < 3:
                continue
            child = np.linalg.solve(kernel.T, a)
            if _valid_basis(reduce_basis(child), options):
                choices.append((ki, selected, hk, child, len(ids)))
        return choices

    roots = []
    for i, a in enumerate(models):
        if progress:
            progress()
        choices = plans(a, 1)
        if choices:
            roots.append((-max(len(c[1]) for c in choices), abs(np.linalg.det(a)), i, choices))
    queue = [(i, 0, 1, c) for _, _, i, c in sorted(roots)[: aliases.max_parents]]
    additions, audit = [], []
    cursor = 0
    while cursor < len(queue) and len(additions) < aliases.max_children:
        root, depth, index, choices = queue[cursor]
        cursor += 1
        for ki, ids, hk, child, total in choices:
            if len(additions) >= aliases.max_children:
                break
            if progress:
                progress()
            p, normal, kernel, _ = kernels[ki]
            order = np.random.default_rng(
                options.seed + 3 + root * 997 + depth * 131 + ki
            ).permutation(len(ids))
            nv = max(5, min(len(ids) - 12, math.ceil(0.2 * len(ids))))
            train = np.ones(len(ids), bool)
            train[order[:nv]] = False
            record = dict(
                root_pool_id=root,
                depth=depth + 1,
                index=index * p,
                prime=p,
                normal=normal.tolist(),
                kernel=kernel.tolist(),
                training_rows=ids.tolist(),
                internal_training=train.tolist(),
                retained=len(ids),
                parent_strict=total,
            )
            try:
                result = refine_crystal(
                    observed[ids],
                    hk,
                    np.linalg.inv(child).T,
                    geometry,
                    training=train,
                    options=native,
                )
                a = reduce_basis(np.linalg.inv(result.reciprocal_basis).T)
                if not _valid_basis(a, options):
                    record["status"] = "invalid_refined_basis"
                elif any(_same_oriented_basis(a, old) for old in additions):
                    record["status"] = "duplicate_child"
                else:
                    additions.append(a)
                    _, _, mask = _score(engine, fit, a, options.reciprocal_tolerance / 4)
                    record.update(
                        status="added",
                        added_id=len(additions) - 1,
                        basis=a.tolist(),
                        volume=abs(float(np.linalg.det(a))),
                        tight_core=int(mask.sum()),
                        converged=result.converged,
                        history=result.history.tolist(),
                    )
                    if depth + 1 < aliases.max_depth:
                        more = plans(a, index * p)
                        if more:
                            queue.append((root, depth + 1, index * p, more))
            except (ValueError, np.linalg.LinAlgError) as ex:
                record.update(status="fit_failure", reason=str(ex))
            audit.append(record)
    return additions, dict(
        options=asdict(aliases),
        root_parents=len(roots),
        selected_roots=[i for _, _, i, _ in sorted(roots)[: aliases.max_parents]],
        proposed=len(additions),
        children=audit,
        scope="Only global training rows; parent proposals retained",
    )


@dataclass(frozen=True)
class GrainStabilityOptions:
    """Additional diverse plane proposals on fixed training-track subsets."""

    resamples: int = 2
    fraction: float = 0.8
    proposal_min_indexed: int = 12

    def __post_init__(self):
        for name, lo, hi in (("resamples", 0, 4), ("proposal_min_indexed", 6, 100000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
        if not math.isfinite(self.fraction) or not 0.5 <= self.fraction <= 1:
            raise ValueError("stability fraction must lie in [0.5,1]")


@dataclass(frozen=True)
class GrainProposalRefinementOptions:
    """Bounded training-only positional polishing before tight-core selection."""

    max_candidates: int = 64
    min_points: int = 18
    max_points: int = 256
    distinct_budget: bool = False
    trim_fractions: tuple[float, ...] = ()
    retain_split_fit: bool = True
    cohort_aware_budget: bool = False

    def __post_init__(self):
        if type(self.cohort_aware_budget) is not bool:
            raise ValueError("cohort_aware_budget must be boolean")
        if type(self.retain_split_fit) is not bool:
            raise ValueError("retain_split_fit must be boolean")
        if not self.retain_split_fit and not self.trim_fractions:
            raise ValueError("discarding the split fit requires explicit trim_fractions")
        if (
            not isinstance(self.trim_fractions, tuple)
            or len(self.trim_fractions) > 3
            or any(
                not isinstance(x, (int, float, np.floating))
                or isinstance(x, bool)
                or not math.isfinite(x)
                or not 0.5 <= x < 1
                for x in self.trim_fractions
            )
            or len(set(self.trim_fractions)) != len(self.trim_fractions)
        ):
            raise ValueError(
                "trim_fractions must contain up to three distinct fractions in [0.5,1)"
            )
        if type(self.distinct_budget) is not bool:
            raise ValueError("distinct_budget must be boolean")
        for name, lo, hi in (
            ("max_candidates", 1, 256),
            ("min_points", 17, 100000),
            ("max_points", 17, 100000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
        if self.max_points < self.min_points:
            raise ValueError("proposal max_points must be >= min_points")


def _stability_proposals(engine, fit, options, conditioning, stability, progress):
    proposal = replace(options, min_indexed=stability.proposal_min_indexed)
    rng = np.random.default_rng(options.seed + 2)
    subsets = [np.arange(len(fit))]
    size = max(proposal.min_indexed, math.ceil(stability.fraction * len(fit)))
    if size > len(fit):
        raise ValueError("insufficient training points for stability proposal minimum")
    subsets.extend(
        np.sort(rng.choice(len(fit), size, replace=False)) for _ in range(stability.resamples)
    )
    models, views = [], []
    for i, ids in enumerate(subsets):
        found, audit = _conditioned_proposals(
            engine, fit[ids], proposal, conditioning, progress, diverse=True
        )
        views.append(dict(view=i, training_rows=ids.tolist(), pool_start=len(models), **audit))
        models.extend(found)
    return models, dict(
        options=asdict(stability),
        seed=options.seed + 2,
        diverse=True,
        views=views,
        proposed=len(models),
    )


def _trimmed_proposal_fits(observed, hkl, basis, geometry, fractions):
    """Unvalidated deterministic proposals; every input row is global training.

    The all-row robust fit orders squared physical residuals. Each fixed retained
    fraction is fit independently from that same anchor, never from another
    fraction's result. Native fused positional derivatives remain unchanged.
    """
    from .prediction import RefinementOptions, fit_crystal_proposal, predict_spots

    native = RefinementOptions()
    full = fit_crystal_proposal(observed, hkl, np.linalg.inv(basis).T, geometry, options=native)
    predicted = predict_spots(hkl, full.reciprocal_basis, geometry, observed[:, 2])
    residual = (predicted.values - observed) / [
        native.xy_scale,
        native.xy_scale,
        native.omega_scale,
    ]
    distance = np.sum(residual**2, axis=1)
    distance[(predicted.flags != 0) | ~np.isfinite(distance)] = np.inf
    order = np.argsort(distance, kind="stable")
    fits = [(1.0, np.arange(len(observed)), full)]
    failures = []
    for fraction in fractions:
        count = max(12, math.ceil(fraction * len(observed)))
        ids = order[:count]
        try:
            fit = fit_crystal_proposal(
                observed[ids], hkl[ids], full.reciprocal_basis, geometry, options=native
            )
            fits.append((fraction, ids, fit))
        except (ValueError, np.linalg.LinAlgError) as ex:
            failures.append(dict(retained_fraction=fraction, reason=str(ex)))
    return fits, failures


def _proposal_fit_rows(error, eligible, options, polishing):
    """The exact input cohort of an early positional fit, before any split."""
    for factor in (0.25, 0.5, 1.0):
        ids = np.flatnonzero(eligible & (error < options.reciprocal_tolerance * factor))
        if len(ids) >= polishing.min_points:
            break
    ids = ids[np.argsort(error[ids], kind="stable")][: polishing.max_points]
    return ids, factor


def _same_fit_start(models, candidate, previous, options, polishing):
    i, j = candidate[2], previous[2]
    if not _same_oriented_basis(models[i], models[j]):
        return False
    if not polishing.cohort_aware_budget:
        return True
    rows_i, _ = _proposal_fit_rows(candidate[4], candidate[5], options, polishing)
    rows_j, _ = _proposal_fit_rows(previous[4], previous[5], options, polishing)
    # Sorted sets determine the all-row anchor objective; a changed membership
    # is a different fit even when its starting matrices are very close.
    rows_i, rows_j = np.sort(rows_i), np.sort(rows_j)
    if not np.array_equal(rows_i, rows_j):
        return False
    transform = np.rint(models[i] @ np.linalg.inv(models[j])).astype(np.int64)
    if not np.array_equal(candidate[3][rows_i], previous[3][rows_i] @ transform.T):
        return False
    if polishing.retain_split_fit:
        selected = []
        for choice, index in ((candidate, i), (previous, j)):
            ids, _ = _proposal_fit_rows(choice[4], choice[5], options, polishing)
            permutation = np.random.default_rng(options.seed + 2 + index).permutation(len(ids))
            count = max(5, min(len(ids) - 12, math.ceil(0.2 * len(ids))))
            keep = np.ones(len(ids), dtype=bool)
            keep[permutation[:count]] = False
            selected.append(np.sort(ids[keep]))
        if not np.array_equal(selected[0], selected[1]):
            return False
    return True


def _polish_one(
    engine, fit, observed, geometry, options, polishing, basis, i, h, error, eligible, record
):
    from .prediction import RefinementOptions, refine_crystal

    native = RefinementOptions()
    sigma = options.reciprocal_tolerance / 4
    added, trimmed_added = [], []
    record = record.copy()
    # Use the narrowest training q core with enough valid starting predictions.
    # The internal withheld rows are ALSO global training rows. No global
    # selection/test observation reaches this fitting API.
    ids, factor = _proposal_fit_rows(error, eligible, options, polishing)
    if polishing.trim_fractions:
        record.update(
            training_rows=ids.tolist(),
            internal_training=[True] * len(ids),
            proposal_q_tolerance=options.reciprocal_tolerance * factor,
            variants=[],
            trimmed_status="proposals",
        )
        try:
            variants, failures = _trimmed_proposal_fits(
                observed[ids],
                h[ids].astype(np.int64),
                basis,
                geometry,
                polishing.trim_fractions,
            )
            record["failed_variants"] = failures
            for fraction, local_rows, fitted in variants:
                a = np.linalg.inv(fitted.reciprocal_basis).T
                if not _valid_basis(a, options):
                    continue
                _, err, mask = _score(engine, fit, a, options.reciprocal_tolerance)
                variant = dict(
                    retained_fraction=fraction,
                    training_rows=ids[local_rows].tolist(),
                    basis=a.tolist(),
                    history=fitted.history.tolist(),
                    converged=fitted.converged,
                    refined_tight_core=int(((err < sigma) & mask).sum()),
                )
                record["variants"].append(variant)
                trimmed_added.append(a)
        except (ValueError, np.linalg.LinAlgError) as ex:
            record.update(trimmed_status="fit_failure", trimmed_reason=str(ex))
        if not polishing.retain_split_fit:
            return record, added, trimmed_added
        # The conservative default preserves the original split-fit output
        # prefix. Explicit trim-only experiments keep its alternative fit
        # policy separate, rather than selecting it on validation outcomes.
    order = np.random.default_rng(options.seed + 2 + i).permutation(len(ids))
    nv = max(5, min(len(ids) - 12, math.ceil(0.2 * len(ids))))
    training = np.ones(len(ids), bool)
    training[order[:nv]] = False
    record.update(
        training_rows=ids.tolist(),
        internal_training=training.tolist(),
        proposal_q_tolerance=options.reciprocal_tolerance * factor,
    )
    try:
        fit_result = refine_crystal(
            observed[ids],
            h[ids].astype(np.int64),
            np.linalg.inv(basis).T,
            geometry,
            training=training,
            options=native,
        )
        a = np.linalg.inv(fit_result.reciprocal_basis).T
        record.update(converged=fit_result.converged, history=fit_result.history.tolist())
        if not _valid_basis(a, options):
            record["status"] = "invalid_refined_basis"
        elif np.max(abs(a - basis)) < 1e-10:
            record["status"] = "stationary"
        else:
            _, err, mask = _score(engine, fit, a, options.reciprocal_tolerance)
            record.update(
                status="added",
                basis=a.tolist(),
                refined_tight_core=int(((err < sigma) & mask).sum()),
            )
            added.append(a)
    except (ValueError, np.linalg.LinAlgError) as ex:
        record.update(status="fit_failure", reason=str(ex))
    return record, added, trimmed_added


def _polish_proposals(engine, fit, observed, geometry, models, options, polishing, progress):
    from .prediction import RefinementOptions, predict_spots

    native = RefinementOptions()
    if len(fit) * len(models) > 8_000_000:
        raise ValueError("ensemble proposal refinement exceeds 8 million position/model pairs")
    sigma = options.reciprocal_tolerance / 4
    choices, records = [], []
    budget_evidence = []
    for i, a in enumerate(models):
        h, error, near = _score(engine, fit, a, options.reciprocal_tolerance)
        predicted = predict_spots(h.astype(np.int64), np.linalg.inv(a).T, geometry, observed[:, 2])
        delta = predicted.values - observed
        eligible = near & (predicted.flags == 0) & np.isfinite(delta).all(axis=1)
        eligible &= np.linalg.norm(delta[:, :2], axis=1) < native.initial_xy_gate
        eligible &= abs(delta[:, 2]) < native.initial_omega_gate
        record = dict(
            pool_id=i,
            initial_eligible=int(eligible.sum()),
            tight_core=int(((error < sigma) & near).sum()),
            status="insufficient_proposal_support",
        )
        point_evidence = np.zeros(len(fit))
        if eligible.sum() >= polishing.min_points and np.linalg.matrix_rank(h[eligible]) == 3:
            density = abs(np.linalg.det(a)) * (2 * np.pi) ** 1.5 * sigma**3
            evidence = float(
                np.logaddexp(0, -0.5 * (error[eligible] / sigma) ** 2 - np.log(density)).sum()
            )
            choices.append((-evidence, abs(np.linalg.det(a)), i, h, error, eligible))
            record.update(status="budget_not_selected", evidence=evidence)
            physical = (
                eligible & (np.linalg.norm(delta[:, :2], axis=1) <= 3) & (abs(delta[:, 2]) <= 0.5)
            )
            point_evidence[physical] = np.logaddexp(
                0, -0.5 * (error[physical] / sigma) ** 2 - np.log(density)
            )
        budget_evidence.append(point_evidence)
        records.append(record)
    added = []
    trimmed_added = []
    budgeted = []
    pending = sorted(choices, key=lambda r: r[:3])
    previous = np.zeros(len(fit))
    while pending:
        if polishing.distinct_budget and len(budgeted) < polishing.max_candidates:
            gains = [float(np.maximum(budget_evidence[c[2]] - previous, 0).sum()) for c in pending]
            winner = min(range(len(pending)), key=lambda j: (-gains[j], *pending[j][:3]))
        else:
            winner = 0
        choice = pending.pop(winner)
        i = choice[2]
        if polishing.distinct_budget and any(
            _same_fit_start(models, choice, old, options, polishing) for old in budgeted
        ):
            records[i]["status"] = "duplicate_fit_budget"
            continue
        if len(budgeted) < polishing.max_candidates:
            budgeted.append(choice)
            previous = np.maximum(previous, budget_evidence[i])
    from . import fitting_tasks

    common = fit, observed, geometry, options, polishing
    cases = [
        (i, (models[i].copy(order="K"), i, h, error, eligible, records[i]))
        for _, _, i, h, error, eligible in budgeted
    ]
    pending_fits = fitting_tasks.batches("polish", common, cases)
    for i, case in cases:
        if progress:
            progress()
        record, legacy, trimmed = (
            _polish_one(engine, *common, *case) if pending_fits is None else pending_fits.get(i)
        )
        records[i] = record
        added.extend(legacy)
        for a in trimmed:
            if not any(np.max(abs(a - old)) < 1e-8 for old in trimmed_added):
                trimmed_added.append(a)
    return added + trimmed_added, dict(
        options=asdict(polishing),
        refinement_options=asdict(native),
        proposed=len(added) + len(trimmed_added),
        legacy_proposed=len(added),
        trimmed_proposed=len(trimmed_added),
        candidates=records,
        scope="Only global training rows; original proposals retained",
    )


@discovery_cache.scope
@search_execution.ensemble
def find_grains_ensemble(
    points,
    *,
    training=None,
    options=None,
    searches=None,
    progress=None,
    conditioning=None,
    stability=None,
    proposal_refinement=None,
    observed=None,
    geometry=None,
    sparse_conditioning=None,
    aliases=None,
    ownership=None,
    recovery=None,
    ensemble_workers=1,
    _training_only=False,
):
    """Pool bounded FFT searches, consolidate on training, validate a frozen pool.

    Default: 28 geometric search configurations. Explicit ``searches`` accepts
    1-32 GrainSearchOptions, with at most 256 total model proposals. Search
    acceptance flags never select the pool: reserved coordinates are first used
    after training-only consolidation and joint reassignment. Final acceptance,
    nearest-lattice ownership, ambiguity cuts and rotated controls match the
    single-search API. Labels remain one-based, with zero residual/ambiguous.
    Optional conditioning and stability add training-only Fourier-plane subset
    searches. Stability uses diverse direction shortlists and fixed resamples.
    Optional proposal_refinement additionally needs detector ``observed`` rows
    and a PanelGeometry; it polishes only training observations before the same
    consolidation. No comparator groups, cells or labels enter these stages.
    sparse_conditioning adds further fixed training views with independently
    specified conditional budgets. aliases additionally proposes exact HKL
    congruence reductions using training detector observations. All proposal
    options default to None, preserving the original path exactly.
    ``ensemble_workers>1`` enables ordered spawned-process scheduling, forcing
    one native worker per search. Set the three documented numerical thread
    environment limits to 1 before Python startup and use a main guard.
    Pool startup/shutdown are included in a complete call. The default is serial.
    """
    if type(_training_only) is not bool:
        raise ValueError("_training_only must be boolean")
    o = GrainSearchOptions() if options is None else options
    if not isinstance(o, GrainSearchOptions):
        raise TypeError("options must be GrainSearchOptions")
    session = search_execution.active()
    if session is not None:
        o = replace(o, workers=1)
    if conditioning is not None and not isinstance(conditioning, GrainConditioningOptions):
        raise TypeError("conditioning must be GrainConditioningOptions or None")
    if stability is not None and not isinstance(stability, GrainStabilityOptions):
        raise TypeError("stability must be GrainStabilityOptions or None")
    if sparse_conditioning is not None and not isinstance(
        sparse_conditioning, GrainConditioningOptions
    ):
        raise TypeError("sparse_conditioning must be GrainConditioningOptions or None")
    if aliases is not None and not isinstance(aliases, GrainAliasOptions):
        raise TypeError("aliases must be GrainAliasOptions or None")
    if ownership is not None and not isinstance(ownership, GrainOwnershipOptions):
        raise TypeError("ownership must be GrainOwnershipOptions or None")
    if recovery is not None and not isinstance(recovery, GrainRecoveryOptions):
        raise TypeError("recovery must be GrainRecoveryOptions or None")
    if recovery is not None and ownership is None:
        raise ValueError("recovery requires observation-aware ownership")
    if proposal_refinement is not None and not isinstance(
        proposal_refinement, GrainProposalRefinementOptions
    ):
        raise TypeError("proposal_refinement must be GrainProposalRefinementOptions or None")
    q = triples(points, "points")
    if len(q) > 100000 or np.max(abs(q), initial=0) > 1000:
        raise ValueError("bounded ensemble requires <=100000 positions and |q|<=1000")
    train = np.arange(len(q)) % 5 != 0 if training is None else np.asarray(training)
    if train.dtype != bool or train.shape != (len(q),):
        raise ValueError("training must be boolean shape (n,)")
    train = train.copy()
    fit = q[train]
    fit_observed = None
    if proposal_refinement is not None or aliases is not None or ownership is not None:
        from .reciprocal import PanelGeometry

        if observed is None:
            raise ValueError("proposal refinement requires observed detector positions")
        observation = triples(observed, "observed")
        if observation.shape != q.shape or not isinstance(geometry, PanelGeometry):
            raise ValueError("proposal refinement needs observed shape (n,3) and PanelGeometry")
        fit_observed = observation[train]
    charts, chart_valid = None, None
    if ownership is not None:
        from .ownership import observation_charts

        charts, chart_valid = observation_charts(observation, geometry, ownership)
    if len(fit) < o.min_indexed or (not _training_only and np.count_nonzero(~train) < 6):
        raise ValueError("insufficient training or held-out positions")
    searches = ensemble_search_options(o) if searches is None else tuple(searches)
    if not 1 <= len(searches) <= 32 or any(not isinstance(s, GrainSearchOptions) for s in searches):
        raise ValueError("searches must contain 1-32 GrainSearchOptions")
    if sum(s.max_grains for s in searches) > 256:
        raise ValueError("ensemble proposal budget exceeds 256 models")
    if not hasattr(_core, "ExperimentalLatticeEngine"):
        raise RuntimeError("FFT grain ensemble requires an spots build")
    if session is not None:
        searches = tuple(replace(search, workers=1) for search in searches)
    engine = _core.ExperimentalLatticeEngine(o.workers)
    search_results = (
        session.map("discover", [(fit, search) for search in searches]) if session else None
    )
    parallel_views = search_execution.conditioned_views(
        engine, fit, o, conditioning, stability, sparse_conditioning
    )
    models, diagnostics = [], []
    vectors, scores = np.empty((0, 3)), np.empty(0)
    total = (
        len(searches)
        + int(conditioning is not None)
        + int(stability is not None)
        + int(proposal_refinement is not None)
        + int(sparse_conditioning is not None)
        + int(aliases is not None)
        + int(ownership is not None)
        + 2 * int(recovery is not None)
    )
    for i, search in enumerate(searches):
        if progress:
            progress(i, total)
        found, _, v, s, audit = (
            _discover(fit, search) if search_results is None else next(search_results)
        )
        if i == 0:
            vectors, scores = v, s
        diagnostics.append(
            dict(
                search=i,
                options=asdict(search),
                pool_start=len(models),
                proposed=len(found),
                discovery=audit,
            )
        )
        models.extend(found)
    if conditioning is not None:
        observer = (lambda *_: progress(len(searches), total)) if progress else None
        found, audit = (
            _conditioned_proposals(engine, fit, o, conditioning, observer)
            if parallel_views is None
            else parallel_views.take("conditioning")
        )
        diagnostics.append(dict(conditioning=audit, pool_start=len(models)))
        models.extend(found)
    stage = len(searches) + int(conditioning is not None)
    if stability is not None:
        observer = (lambda *_: progress(stage, total)) if progress else None
        found, audit = (
            _stability_proposals(
                engine, fit, o, conditioning or GrainConditioningOptions(), stability, observer
            )
            if parallel_views is None
            else parallel_views.take("stability")
        )
        diagnostics.append(dict(stability=audit, pool_start=len(models)))
        models.extend(found)
        stage += 1
    if sparse_conditioning is not None:
        observer = (lambda *_: progress(stage, total)) if progress else None
        found, audit = (
            _stability_proposals(
                engine, fit, o, sparse_conditioning, stability or GrainStabilityOptions(), observer
            )
            if parallel_views is None
            else parallel_views.take("sparse")
        )
        diagnostics.append(dict(sparse_conditioning=audit, pool_start=len(models)))
        models.extend(found)
        stage += 1
    if recovery is not None:
        from .recovery import coherent_proposals, paired_proposals

        seed_vectors = None
        for diagnostic in diagnostics:
            if "stability" in diagnostic:
                seed_vectors = np.asarray(diagnostic["stability"]["views"][0]["seed_vectors"])
                break
        if seed_vectors is None:
            seed = replace(
                o,
                max_length=o.max_length / 2,
                fft_peaks=400,
                basis_vectors=80,
                max_grains=1,
                reciprocal_tolerance=o.reciprocal_tolerance / 5,
            )
            seed_vectors, _, _ = _vectors(engine, fit, seed, diverse=True)
        observer = (lambda: progress(stage, total)) if progress else None
        if recovery.paired_search:
            paired, paired_audit = paired_proposals(
                engine, fit, seed_vectors, o, recovery, observer
            )
            coherent, coherent_audit = coherent_proposals(
                engine, fit, fit_observed, geometry, paired, o, recovery, observer
            )
        else:
            paired, coherent = [], []
            paired_audit, coherent_audit = dict(proposed=0), dict(proposed=0)
        diagnostics.append(
            dict(
                recovery_proposals=dict(paired=paired_audit, coherent=coherent_audit),
                pool_start=len(models),
                proposed=len(paired) + len(coherent),
            )
        )
        models.extend(paired)
        models.extend(coherent)
        stage += 1
    if proposal_refinement is not None:
        observer = (lambda: progress(stage, total)) if progress else None
        found, audit = _polish_proposals(
            engine, fit, fit_observed, geometry, models, o, proposal_refinement, observer
        )
        diagnostics.append(dict(proposal_refinement=audit, pool_start=len(models)))
        models.extend(found)
        stage += 1
    if aliases is not None:
        observer = (lambda: progress(stage, total)) if progress else None
        found, audit = _alias_proposals(
            engine, fit, fit_observed, geometry, models, o, aliases, observer
        )
        diagnostics.append(dict(aliases=audit, pool_start=len(models)))
        models.extend(found)
        stage += 1
    if ownership is None:
        models, labels, audit = _consolidate(engine, fit, models, o)
        diagnostics.append(dict(consolidation=audit))
    else:
        from .ownership import refine_ownership_pool

        observer = (lambda: progress(stage, total)) if progress else None
        models, labels, audit = refine_ownership_pool(
            engine,
            fit,
            fit_observed,
            geometry,
            models,
            charts[train],
            chart_valid[train],
            o,
            ownership,
            aliases,
            observer,
            input_aliases=[
                dict(cycle=-1, pool_start=d["pool_start"], audit=d["aliases"])
                for d in diagnostics
                if "aliases" in d
            ],
        )
        diagnostics.append(dict(ownership=audit))
        stage += 1
    if recovery is not None:
        from .recovery import finish_training_pool

        observer = (lambda: progress(stage, total)) if progress else None
        models, labels, audit = finish_training_pool(
            engine,
            fit,
            fit_observed,
            geometry,
            models,
            charts[train],
            chart_valid[train],
            o,
            ownership,
            observer,
        )
        diagnostics.append(dict(training_recovery=audit))
    if progress:
        progress(total, total)
    diagnostics.append(dict(discovery_reuse=discovery_cache.statistics()))
    if session is not None:
        diagnostics.append(dict(parallel_search=session.statistics()))
    if _training_only:
        return _GrainTrainingPool(
            np.asarray(models).reshape(-1, 3, 3), labels, vectors, scores, tuple(diagnostics), o
        )
    return _finish(
        q,
        train,
        models,
        labels,
        vectors,
        scores,
        diagnostics,
        o,
        method="fft_grains_ensemble",
        ownership=ownership,
        charts=charts,
        chart_valid=chart_valid,
    )
