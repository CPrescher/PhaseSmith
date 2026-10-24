"""Position-only FFT grain search with a frozen training/validation split.

No input HKLs, cell, chemistry, Bravais class, or space group. Native FFT and
ownership kernels; readable array orchestration; no Qt or external checkout.
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from phasesmith import _core

from .indexing import IndexingResult, LatticeCandidate, cell_parameters, reduce_basis
from .reciprocal import lattice_fourier, triples


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

    def __post_init__(self):
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


def _vectors(engine, q, o):
    n = o.grid_size
    step = min(2 * np.max(abs(q)) / (n - 4), 1 / (2.1 * o.max_length))
    if step <= 0 or np.max(abs(q)) / step >= n / 2 - 1:
        raise ValueError(
            "FFT grid too small for reciprocal extent and maximum edge; "
            "increase grid or lower edge bound"
        )
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
    for _ in range(20):
        moments = lattice_fourier(q, vectors)
        h = -moments[:, 4:].reshape(-1, 3, 3)
        h += np.maximum(0.0, 0.01 - np.linalg.eigvalsh(h)[:, 0])[:, None, None] * np.eye(3)
        delta = np.linalg.solve(h, moments[:, 1:4, None])[:, :, 0]
        delta *= np.minimum(1.0, 0.3 / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
        improved = np.zeros(len(vectors), dtype=bool)
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(vectors + scale * delta, initial - radius, initial + radius)
            good = (~improved) & (lattice_fourier(q, trial)[:, 0] > moments[:, 0])
            vectors[good] = trial[good]
            improved |= good
        if not np.any(improved):
            break
    scores = lattice_fourier(q, vectors)[:, 0]
    chosen = []
    for i in np.argsort(-scores, kind="stable"):
        r = vectors[i]
        length = np.linalg.norm(r)
        if not o.min_length <= length <= o.max_length:
            continue
        if any(
            abs(np.dot(r, vectors[j])) / (length * np.linalg.norm(vectors[j])) > 0.9995
            for j in chosen
        ):
            continue
        chosen.append(i)
        if len(chosen) >= o.basis_vectors:
            break
    return (
        vectors[chosen],
        scores[chosen],
        dict(
            reciprocal_grid_step=float(step),
            direct_grid_step=float(1 / (n * step)),
            fft_local_maxima=int(total),
        ),
    )


def _triplets(q, vectors, o):
    phase = q @ vectors.T
    support = (abs(phase - np.rint(phase)) < o.fractional_tolerance).T
    counts = support.astype(float) @ support.T.astype(float)
    pairs = sorted(combinations(range(len(vectors)), 2), key=lambda p: (-counts[p], p))[
        : o.max_pairs
    ]
    records, seen = [], set()
    for i, j in pairs:
        for k in range(len(vectors)):
            key = tuple(sorted((i, j, k)))
            if len(set(key)) < 3 or key in seen:
                continue
            seen.add(key)
            rows = vectors[list(key)]
            if not _valid_basis(rows, o):
                continue
            count = int(np.count_nonzero(np.all(support[list(key)], axis=0)))
            records.append((count, abs(float(np.linalg.det(rows))), key))
    records.sort(key=lambda r: (-r[0], r[1], r[2]))
    return records


def integer_lattice_basis(indices):
    """Exact Euclidean row operations on inferred HKLs; no supplied indexing.

    Returns the integer lattice generated by these rows, or None for rank <3.
    Used to propose smaller direct cells, never as proof of primitivity.
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
    if any(row is None for row in basis) or any(abs(v) > 2**53 for row in basis for v in row):
        return None
    return np.asarray(basis, dtype=np.int64)


def _refine(engine, q, a, o):
    a = a.copy()
    for cycle in range(8):
        h, _, selected = _score(engine, q, a, o.reciprocal_tolerance * (2 if cycle < 2 else 1))
        if selected.sum() < 6 or np.linalg.matrix_rank(h[selected]) < 3:
            return None
        reciprocal = np.linalg.lstsq(h[selected], q[selected], rcond=None)[0]
        if abs(np.linalg.det(reciprocal)) < 1e-12:
            return None
        updated = np.linalg.inv(reciprocal).T
        delta = np.max(abs(a - updated))
        a = updated
        if cycle >= 2 and delta < 1e-8:
            break
    a = reduce_basis(a)
    if not _valid_basis(a, o):
        return None
    h, error, selected = _score(engine, q, a, o.reciprocal_tolerance)
    order = np.flatnonzero(selected)
    order = order[np.argsort(error[order], kind="stable")]
    alternatives = [a]
    for size in (24, 48, 96):
        if len(order) < size:
            continue
        integer = integer_lattice_basis(h[order[:size]].astype(np.int64))
        if integer is None or abs(round(np.linalg.det(integer))) <= 1:
            continue
        candidate = reduce_basis(np.linalg.inv(integer @ np.linalg.inv(a).T).T)
        if not _valid_basis(candidate, o):
            continue
        _, _, mask = _score(engine, q, candidate, o.reciprocal_tolerance)
        if np.count_nonzero(mask & selected) >= 0.9 * selected.sum():
            alternatives.append(candidate)
    return min(alternatives, key=lambda b: abs(np.linalg.det(b)))


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


def find_grains(points, *, training=None, options=None, progress=None):
    """Discover up to max_grains competing lattices; validate a frozen pool.

    Labels are one-based candidate IDs; zero means residual or ambiguous.
    Rejected models remain in the pool/export to preserve validation competition.
    All discovery, HKL inference, refinement and selection use training only.
    `progress(done, total)` is a synchronous observer between discovery rounds;
    exceptions propagate, allowing cancellation without GUI dependencies.
    """
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
        raise RuntimeError("FFT grain search requires a PhaseSmith experimental-spots build")
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
        records = _triplets(block, v, o)
        choices, audit = [], []
        for _, _, ids in records[: o.cell_refinements]:
            a = _refine(engine, block, v[list(ids)], o)
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
            screened_triplets=len(records),
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
    # First use of held-out coordinates. Pool/order stays fixed, even on failure.
    held_labels, ambiguous, multiple = _labels(engine, q[~train], models, o)
    null = []
    for rotation in _rotations(o.null_rotations, o.seed + 1):
        control, _, _ = _labels(engine, q[~train] @ rotation, models, o)
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
        method="fft_grains",
        labels=all_labels,
        diagnostics=tuple(diagnostics),
    )
