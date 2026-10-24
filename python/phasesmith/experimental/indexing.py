"""Bounded, material-independent lattice proposals from measured q.

FFT proposes direct-space translations. Actual unbinned observations refine
them. Training-only triplet selection is followed by held-out validation.
No database, chemistry, Bravais class, supplied cell, or known UB is accepted.
"""

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .reciprocal import lattice_fourier, triples


@dataclass(frozen=True)
class IndexingOptions:
    min_length: float = 1.5
    max_length: float = 30.0
    grid_size: int = 256
    fft_peaks: int = 160
    basis_vectors: int = 28
    fractional_tolerance: float = 0.15
    reciprocal_tolerance: float = 0.015
    min_indexed: int = 20
    min_fraction: float = 0.2
    consensus_slack: float = 0.02
    max_candidates: int = 40

    def __post_init__(self):
        for name in ("grid_size", "fft_peaks", "basis_vectors", "min_indexed", "max_candidates"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        for name in (
            "min_length",
            "max_length",
            "fractional_tolerance",
            "reciprocal_tolerance",
            "min_fraction",
            "consensus_slack",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("indexing limits must be finite")
        if not 0 < self.min_length < self.max_length <= 100:
            raise ValueError("require 0 < min_length < max_length <= 100 angstroms")
        if self.grid_size not in (64, 128, 256):
            raise ValueError("grid_size must be 64, 128 or 256")
        if not 3 <= self.basis_vectors <= 40 or not self.basis_vectors <= self.fft_peaks <= 400:
            raise ValueError("invalid candidate budgets")
        if not 0 < self.fractional_tolerance < 0.5 or not 0 < self.reciprocal_tolerance < 1:
            raise ValueError("invalid residual tolerances")
        if not 6 <= self.min_indexed <= 1_000_000 or not 0 < self.min_fraction <= 1:
            raise ValueError("invalid consensus requirements")
        if not 0 <= self.consensus_slack <= 0.05:
            raise ValueError("consensus_slack must lie in [0,.05]")
        if not 1 <= self.max_candidates <= 80:
            raise ValueError("max_candidates must lie in [1,80]")


@dataclass(frozen=True)
class LatticeCandidate:
    direct_basis: np.ndarray  # rows, angstroms; primitivity unproven
    cell: np.ndarray  # a,b,c,alpha,beta,gamma
    volume: float
    hkl: np.ndarray
    indexed: np.ndarray
    residual: np.ndarray  # reciprocal Euclidean distance, inverse angstroms
    training_fraction: float
    validation_fraction: float
    validation_count: int
    null_fraction: float
    accepted: bool


@dataclass(frozen=True)
class IndexingResult:
    candidates: tuple[LatticeCandidate, ...]
    training: np.ndarray
    translation_vectors: np.ndarray
    translation_scores: np.ndarray
    message: str
    options: object  # IndexingOptions or GrainSearchOptions
    method: str = "compact_fft"
    labels: np.ndarray | None = None  # one-based model ID; zero residual/ambiguous
    diagnostics: tuple = ()


def cell_parameters(basis):
    """General triclinic metric of a row-vector real-space basis."""
    a = triples(basis, "basis")
    if a.shape != (3, 3) or abs(np.linalg.det(a)) < 1e-10:
        raise ValueError("basis must be nonsingular (3,3)")
    lengths = np.linalg.norm(a, axis=1)
    angles = [
        np.rad2deg(np.arccos(np.clip(np.dot(a[i], a[j]) / (lengths[i] * lengths[j]), -1, 1)))
        for i, j in [(1, 2), (0, 2), (0, 1)]
    ]
    return np.r_[lengths, angles]


def reduce_basis(basis):
    """Deterministic integer size reduction; not a Niggli/conventional cell."""
    a = triples(basis, "basis")
    if a.shape != (3, 3) or not np.isfinite(np.linalg.det(a)) or abs(np.linalg.det(a)) < 1e-10:
        raise ValueError("basis must be finite nonsingular (3,3)")
    for _ in range(100):
        a = a[np.argsort(np.sum(a * a, axis=1), kind="stable")]
        changed = False
        for j in (1, 2):
            for i in range(j - 1, -1, -1):
                multiple = np.rint(np.dot(a[i], a[j]) / np.dot(a[i], a[i]))
                trial = a[j] - multiple * a[i]
                if np.dot(trial, trial) < np.dot(a[j], a[j]) - 1e-10:
                    a[j] = trial
                    changed = True
        if not changed:
            break
    if np.linalg.det(a) < 0:
        a[-1] *= -1
    return a


def _translations(q, options):
    n = options.grid_size
    spacing = 1 / (2 * options.max_length)
    # Fixed grid/memory bound. High-q observations outside the grid remain in
    # continuous refinement and final validation, never silently discarded there.
    inside = np.max(abs(q), axis=1) < (n / 2 - 2) * spacing
    if np.count_nonzero(inside) < options.min_indexed:
        return np.empty((0, 3)), np.empty(0)
    xyz = q[inside] / spacing
    lower = np.floor(xyz).astype(np.int64)
    fraction = xyz - lower
    grid = np.zeros((n, n, n), dtype=float)
    for offset in product((0, 1), repeat=3):
        offset = np.array(offset)
        weights = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
        index = (lower + offset) % n
        np.add.at(grid, tuple(index.T), weights)
    # Real part is mean cosine; Friedel ±q would give the same real transform.
    spectrum = np.fft.fftn(grid).real
    del grid
    frequency = np.fft.fftfreq(n, d=spacing)
    r2 = (
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    maxima = (r2 >= options.min_length**2) & (r2 <= options.max_length**2)
    del r2
    # Retain only one of each ± pair; all 26 neighbours are checked.
    maxima &= (
        (frequency[:, None, None] > 0)
        | ((frequency[:, None, None] == 0) & (frequency[None, :, None] > 0))
        | (
            (frequency[:, None, None] == 0)
            & (frequency[None, :, None] == 0)
            & (frequency[None, None, :] > 0)
        )
    )
    for delta in product((-1, 0, 1), repeat=3):
        if delta != (0, 0, 0):
            maxima &= spectrum > np.roll(spectrum, delta, axis=(0, 1, 2))
    indices = np.flatnonzero(maxima)
    ranked = indices[np.argsort(-spectrum.ravel()[indices], kind="stable")[: options.fft_peaks]]
    vectors = frequency[np.array(np.unravel_index(ranked, spectrum.shape)).T]
    del spectrum, maxima
    if not len(vectors):
        return vectors, np.empty(0)
    # Damped Newton ascent on the continuous mean cosine, with fused native
    # gradient/Hessian; bounded steps and monotonic acceptance.
    for _ in range(12):
        moments = lattice_fourier(q, vectors)
        h = -moments[:, 4:].reshape(-1, 3, 3)
        eig = np.linalg.eigvalsh(h)
        h += np.maximum(0.0, 0.01 - eig[:, 0])[:, None, None] * np.eye(3)
        step = np.linalg.solve(h, moments[:, 1:4, None])[:, :, 0]
        step *= np.minimum(1.0, 0.3 / np.maximum(np.linalg.norm(step, axis=1), 1e-12))[:, None]
        trial = vectors + step
        good = lattice_fourier(q, trial)[:, 0] >= moments[:, 0]
        vectors[good] = trial[good]
        if np.max(np.linalg.norm(step[good], axis=1), initial=0.0) < 1e-6:
            break
    scores = lattice_fourier(q, vectors)[:, 0]
    keep = []
    for i in np.argsort(-scores, kind="stable"):
        length = np.linalg.norm(vectors[i])
        if not options.min_length <= length <= options.max_length:
            continue
        if any(
            min(np.linalg.norm(vectors[i] - vectors[j]), np.linalg.norm(vectors[i] + vectors[j]))
            < 0.15
            for j in keep
        ):
            continue
        keep.append(i)
        if len(keep) >= options.basis_vectors:
            break
    return vectors[keep], scores[keep]


def _assign(q, basis, options):
    fractional = q @ basis.T
    hkl = np.rint(fractional)
    reciprocal = np.linalg.inv(basis).T
    residual = np.linalg.norm(q - hkl @ reciprocal, axis=1)
    indexed = (
        (np.max(abs(fractional - hkl), axis=1) < options.fractional_tolerance)
        & (residual < options.reciprocal_tolerance)
        & np.any(hkl != 0, axis=1)
    )
    return hkl.astype(np.int64), indexed, residual


def _refine(q, basis, options):
    for _ in range(8):
        hkl, selected, _ = _assign(q, basis, options)
        if (
            np.count_nonzero(selected) < options.min_indexed
            or np.linalg.matrix_rank(hkl[selected]) < 3
        ):
            return None
        reciprocal = np.linalg.lstsq(hkl[selected], q[selected], rcond=None)[0]
        if abs(np.linalg.det(reciprocal)) < 1e-10:
            return None
        new = np.linalg.inv(reciprocal).T
        if np.max(abs(new - basis)) < 1e-7:
            basis = new
            break
        basis = new
    basis = reduce_basis(basis)
    lengths = np.linalg.norm(basis, axis=1)
    if lengths.min() < options.min_length or lengths.max() > options.max_length:
        return None
    return basis


def _subdivide(q, basis, options):
    """Test small-denominator subcells against observations, with no metric prior.

    This is a bounded supercell check, not a proof of a primitive lattice.
    Missing reflections or multiple lattices can leave the cell ambiguous.
    """
    required = np.count_nonzero(_assign(q, basis, options)[1]) - math.ceil(
        options.consensus_slack * len(q)
    )
    for _ in range(4):
        trials = []
        for row in range(3):
            other = [i for i in range(3) if i != row]
            for denominator in (2, 3):
                for j, k in product(range(denominator), repeat=2):
                    trial = basis.copy()
                    trial[row] = (
                        basis[row] + j * basis[other[0]] + k * basis[other[1]]
                    ) / denominator
                    if np.linalg.norm(reduce_basis(trial), axis=1).min() < options.min_length:
                        continue
                    _, selected, _ = _assign(q, trial, options)
                    if np.count_nonzero(selected) >= max(required, options.min_indexed):
                        refined = _refine(q, trial, options)
                        if refined is not None and np.count_nonzero(
                            _assign(q, refined, options)[1]
                        ) >= max(required, options.min_indexed):
                            trials.append(refined)
        if not trials:
            break
        basis = min(
            trials,
            key=lambda a: (abs(np.linalg.det(a)), -np.count_nonzero(_assign(q, a, options)[1])),
        )
    return basis


def find_lattices(points, *, training=None, options=None):
    """Propose bounded size-reduced cells with a training/validation separation.

    Supply a boolean training mask partitioned by frame/track for real scans.
    Default holds out every fifth input point. No validation points contribute
    to FFT, vector refinement, basis choice or least-squares fitting.
    """
    q = triples(points, "points")
    options = IndexingOptions() if options is None else options
    if not isinstance(options, IndexingOptions):
        raise TypeError("options must be IndexingOptions")
    if len(q) > 100_000:
        raise ValueError("at most 100000 reciprocal positions per bounded search")
    if training is None:
        training = np.arange(len(q)) % 5 != 0
    training = np.asarray(training)
    if training.dtype != bool or training.shape != (len(q),):
        raise ValueError("training must be boolean shape (n,)")
    training = training.copy()
    fit = q[training]
    if len(fit) < options.min_indexed or np.count_nonzero(~training) < 5:
        raise ValueError("insufficient training or held-out positions")
    vectors, scores = _translations(fit, options)
    preliminary = []
    # Upper bound by per-translation integer proximity, before matrix inversion.
    close = abs(fit @ vectors.T - np.rint(fit @ vectors.T)) < options.fractional_tolerance
    for ids in combinations(range(len(vectors)), 3):
        support = np.all(close[:, ids], axis=1)
        count = np.count_nonzero(support)
        if count < options.min_indexed:
            continue
        basis = vectors[list(ids)]
        volume = abs(np.linalg.det(basis))
        if volume < options.min_length**3 / 4:
            continue
        if volume / np.prod(np.linalg.norm(basis, axis=1)) < 0.08:
            continue
        _, selected, residual = _assign(fit, basis, options)
        count = np.count_nonzero(selected)
        if count >= options.min_indexed:
            preliminary.append((count, float(np.mean(residual[selected] ** 2)), volume, ids))
    best_support = max((row[0] for row in preliminary), default=0)
    cutoff = best_support - math.ceil(options.consensus_slack * len(fit))
    # Additional real-space volume can fit occasional outliers by inventing
    # supercells. Prefer the smallest cell within a declared consensus slack.
    preliminary.sort(
        key=lambda row: (row[0] < cutoff, row[2] if row[0] >= cutoff else -row[0], row[1], row[3])
    )
    proposals = []
    for _, _, _, ids in preliminary[:80]:
        basis = _refine(fit, vectors[list(ids)], options)
        if basis is None:
            continue
        basis = _subdivide(fit, basis, options)
        # Equivalent integer changes of basis describe one lattice.
        equivalent = False
        for old in proposals:
            transform = basis @ np.linalg.inv(old)
            if (
                np.max(abs(transform - np.rint(transform))) < 0.05
                and abs(round(np.linalg.det(transform))) == 1
            ):
                equivalent = True
                break
        if not equivalent:
            proposals.append(basis)
    best_support = max(
        (np.count_nonzero(_assign(fit, a, options)[1]) for a in proposals), default=0
    )
    cutoff = best_support - math.ceil(options.consensus_slack * len(fit))

    def rank(basis):
        _, selected, residual = _assign(fit, basis, options)
        count = np.count_nonzero(selected)
        return (
            count < cutoff,
            abs(np.linalg.det(basis)) if count >= cutoff else -count,
            float(np.mean(residual[selected] ** 2)),
        )

    proposals.sort(key=rank)
    candidates = []
    for basis in proposals[: options.max_candidates]:
        hkl, indexed, residual = _assign(q, basis, options)
        train_fraction = float(np.mean(indexed[training]))
        validation_fraction = float(np.mean(indexed[~training]))
        # Descriptive off-origin controls, not calibrated p-values. Preserves
        # observed reciprocal-space shape but shifts away from lattice origin.
        inverse = np.linalg.inv(basis).T
        null = [
            np.mean(_assign(q[~training] + np.array(shift) @ inverse, basis, options)[1])
            for shift in product((0.27, 0.53), repeat=3)
        ]
        null_fraction = float(max(null))
        accepted = (
            train_fraction >= options.min_fraction
            and np.count_nonzero(indexed[~training]) >= 5
            and validation_fraction >= options.min_fraction
            and validation_fraction >= 3 * null_fraction + 0.05
            and len(np.unique(hkl[indexed & training], axis=0)) >= options.min_indexed
            and np.linalg.matrix_rank(hkl[indexed & training]) == 3
        )
        cell = cell_parameters(basis)
        for array in [basis, cell, hkl, indexed, residual]:
            array.flags.writeable = False
        candidates.append(
            LatticeCandidate(
                basis,
                cell,
                abs(float(np.linalg.det(basis))),
                hkl,
                indexed,
                residual,
                train_fraction,
                validation_fraction,
                int(np.count_nonzero(~training)),
                null_fraction,
                bool(accepted),
            )
        )
    for array in [training, vectors, scores]:
        array.flags.writeable = False
    message = (
        "Lattice proposal passes the held-out checks; inspect geometry and competing cells."
        if candidates and candidates[0].accepted
        else "No reliable lattice found within the declared geometry and search bounds."
    )
    return IndexingResult(tuple(candidates), training, vectors, scores, message, options)
