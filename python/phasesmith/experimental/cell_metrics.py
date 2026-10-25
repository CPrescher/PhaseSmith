"""Lattice metric spaces, exact centring cosets and bounded Bravais settings.

Row convention: A_conventional=T A_primitive, h_conventional=h_primitive T.T.
Selling reduction is lattice preserving; conventional centring is not extinction
inference. No intensity, composition or reference-cell model is used here.
"""

from dataclasses import dataclass
from itertools import product
from math import gcd

import numpy as np

from phasesmith import _core

from .indexing import cell_parameters
from .prediction import _basis, _readonly

BRAVAIS = ("aP", "mP", "mC", "oP", "oC", "oI", "oF", "tP", "tI", "hP", "hR", "cP", "cI", "cF")


@dataclass(frozen=True)
class ReducedCell:
    direct_basis: np.ndarray
    transform: np.ndarray
    iterations: int

    def __post_init__(self):
        object.__setattr__(self, "direct_basis", _readonly(self.direct_basis))
        object.__setattr__(self, "transform", _readonly(self.transform, np.int64))


def reduce_cell(direct_basis, *, tolerance=1e-12, max_iterations=10000):
    """Native Selling-Delaunay reduction, not a unique Niggli representative.

    Select the shortest three members of an obtuse superbase, right handed.
    ``transform`` is unimodular and exact; A_reduced=transform @ A_input.
    """
    a = _basis(direct_basis)
    if np.max(abs(a)) > 1e6 or np.linalg.cond(a) > 1e10:
        raise ValueError("lattice magnitude/conditioning budget exceeded")
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
        raise ValueError("tolerance must lie in [0,1e-6]")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 10000:
        raise ValueError("max_iterations must be integer in [1,10000]")
    if not hasattr(_core, "experimental_reduce_lattice"):
        raise RuntimeError("build PhaseSmith with --features experimental-spots for cell reduction")
    a, t, n = _core.experimental_reduce_lattice(a, float(tolerance), max_iterations)
    return ReducedCell(a, t, n)


def metric_directions(symbol):
    """Linear direct Gram constraints in conventional axes (monoclinic unique b)."""
    if symbol not in BRAVAIS:
        raise ValueError("unknown Bravais symbol")
    family = symbol[0]
    if family == "c":
        return np.eye(3)[None]
    if family in "th":
        ab = np.diag([1.0, 1.0, 0.0])
        if family == "h":
            ab[0, 1] = ab[1, 0] = -0.5  # gamma=120 degrees, including hR hexagonal axes
        return np.array([ab, np.diag([0.0, 0.0, 1.0])])
    d = list(np.eye(3)[i, :, None] * np.eye(3)[i, None, :] for i in range(3))
    for i, j in [(0, 2)] if family == "m" else [(0, 1), (0, 2), (1, 2)] if family == "a" else []:
        e = np.zeros((3, 3))
        e[i, j] = e[j, i] = 1
        d.append(e)
    return np.array(d)


def project_metric(a, symbol):
    """Frobenius Gram projection, then the closest Cartesian orientation."""
    g = a @ a.T
    d = metric_directions(symbol)
    coordinates = np.linalg.lstsq(d.reshape(len(d), 9).T, g.ravel(), rcond=None)[0]
    projected = np.einsum("k,kij->ij", coordinates, d)
    c = np.linalg.cholesky(projected)
    u, _, vt = np.linalg.svd(c.T @ a)
    q = u @ vt
    if np.linalg.det(q) <= 0:
        raise ValueError("right-handed direct basis required")
    # Principal stretches of the projected metric relative to the supplied metric.
    old = np.linalg.cholesky(g)
    relative = np.linalg.solve(old, projected) @ np.linalg.inv(old.T)
    strain = float(np.max(abs(np.sqrt(np.linalg.eigvalsh(relative)) - 1)))
    return c @ q, strain


def metric_chain(b, symbol):
    """dB/d(dimensionless local Gram coordinates, Cartesian rotation), (9,p).

    dC=C Phi(C^-1 dG C^-T), Phi lower triangular with half diagonal.
    Metric direction scales are local, frozen for this derivative/retraction.
    """
    a = np.linalg.inv(b).T
    g = a @ a.T
    c = np.linalg.cholesky(g)
    ci = np.linalg.inv(c)
    d = metric_directions(symbol)
    scale = np.linalg.norm(g) / np.linalg.norm(d, axis=(1, 2))
    chain = []
    for direction, s in zip(d, scale, strict=True):
        m = ci @ (s * direction) @ ci.T
        phi = np.tril(m, -1) + 0.5 * np.diag(np.diag(m))
        dc = c @ phi
        chain.append((-ci.T @ dc.T @ b).ravel())
    chain += [np.cross(np.eye(3)[k], b).ravel() for k in range(3)]
    return np.array(chain).T


def metric_step(b, step, symbol):
    a = np.linalg.inv(b).T
    g = a @ a.T
    d = metric_directions(symbol)
    scale = np.linalg.norm(g) / np.linalg.norm(d, axis=(1, 2))
    newg = g + np.einsum("k,kij->ij", step[:-3] * scale, d)
    c = np.linalg.cholesky(g)
    q = c.T @ b
    w = step[-3:]
    theta = np.linalg.norm(w)
    cross = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    rotation = (
        np.eye(3)
        + np.sinc(theta / np.pi) * cross
        + 0.5 * np.sinc(theta / (2 * np.pi)) ** 2 * (cross @ cross)
    )
    return np.linalg.solve(np.linalg.cholesky(newg).T, q @ rotation.T)


def integer_det(t):
    a, b, c = [[int(x) for x in row] for row in t]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def centring(t):
    """Exact integer adjugate residues; None for nonstandard conventional cells."""
    t = np.asarray(t)
    if t.shape != (3, 3) or t.dtype.kind not in "iu" or (abs(t.astype(float)) > 1e12).any():
        raise ValueError("centring transform must be bounded integer (3,3)")
    det = integer_det(t)
    n = abs(det)
    if n not in (1, 2, 3, 4):
        return None
    if n == 1:
        return "P"
    # Reduce before products: the adjugate residues are unchanged, avoiding overflow.
    t = (t % n).astype(np.int64)
    adj = np.column_stack((np.cross(t[1], t[2]), np.cross(t[2], t[0]), np.cross(t[0], t[1])))
    residues = {tuple((np.asarray(v) @ adj) % n) for v in product(range(n), repeat=3)}
    expected = {
        "A": {(0, 0, 0), (0, 1, 1)},
        "B": {(0, 0, 0), (1, 0, 1)},
        "C": {(0, 0, 0), (1, 1, 0)},
        "I": {(0, 0, 0), (1, 1, 1)},
        "F": {(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)},
        "R": {(0, 0, 0), (2, 1, 1), (1, 2, 2)},
    }
    return next((key for key, value in expected.items() if residues == value), None)


@dataclass(frozen=True)
class BravaisSetting:
    symbol: str
    transform: np.ndarray  # integer from supplied primitive hypothesis to conventional
    direct_basis: np.ndarray  # unprojected exact basis
    projected_basis: np.ndarray
    strain: float

    def __post_init__(self):
        for name in ("transform", "direct_basis", "projected_basis"):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), np.int64 if name == "transform" else None),
            )

    @property
    def cell(self):
        return cell_parameters(self.direct_basis)

    @property
    def index(self):
        return abs(integer_det(self.transform))


def _constraint_key(t, symbol):
    inverse = np.linalg.inv(t)
    d = np.array([inverse @ e @ inverse.T for e in metric_directions(symbol)])
    u, _, _ = np.linalg.svd(d.reshape(len(d), 9).T, full_matrices=False)
    return u @ u.T


def find_bravais_settings(direct_basis, *, max_strain=0.01, max_angle=1.0, coefficient_limit=2):
    """Compare all 14 Bravais types, in short axes of a Selling-reduced lattice.

    Bounds are explicit: primitive coefficient vectors in [-limit,limit]^3,
    conventional indices 1/2/3/4, strain <=max_strain, constrained angles within
    max_angle degrees. Equivalent constraint subspaces are deduplicated; distinct
    monoclinic axes are retained. Missing types are not a proof of impossibility.
    """
    if not np.isfinite(max_strain) or not 0 < max_strain <= 0.1:
        raise ValueError("max_strain must lie in (0,.1]")
    if not np.isfinite(max_angle) or not 0 < max_angle <= 5:
        raise ValueError("max_angle must lie in (0,5] degrees")
    if type(coefficient_limit) is not int or not 1 <= coefficient_limit <= 3:
        raise ValueError("coefficient_limit must be integer in [1,3]")
    reduced = reduce_cell(direct_basis)
    a, reduction = reduced.direct_basis, reduced.transform
    v = np.array(
        [
            r
            for r in product(range(-coefficient_limit, coefficient_limit + 1), repeat=3)
            if any(r) and next(x for x in r if x) > 0 and gcd(gcd(r[0], r[1]), r[2]) == 1
        ],
        dtype=np.int64,
    )
    axes = v @ a
    length = np.linalg.norm(axes, axis=1)
    cosine = axes @ axes.T / length[:, None] / length[None, :]
    perpendicular = abs(cosine) <= np.sin(np.deg2rad(max_angle))
    hexagonal = (abs(cosine) >= np.cos(np.deg2rad(60 + max_angle))) & (
        abs(cosine) <= np.cos(np.deg2rad(60 - max_angle))
    )
    equal = abs(length[:, None] / length[None, :] - 1) <= 4 * max_strain
    base, strain = project_metric(a, "aP")
    result = [BravaisSetting("aP", reduction, a, base, strain)]
    keys = [("aP", _constraint_key(np.eye(3), "aP"))]
    scores = [(float(np.sum(a * a)), tuple(reduction.ravel()))]
    # b perpendicular a and c covers monoclinic and all orthogonal settings;
    # the second pass covers hexagonal a/b, c perpendicular to both.
    triples = set()
    for j in range(len(v)):
        neighbors = np.flatnonzero(perpendicular[j])
        triples.update((int(i), j, int(k)) for i in neighbors for k in neighbors if i != k)
    for k in range(len(v)):
        for i in np.flatnonzero(perpendicular[k]):
            triples.update(
                (int(i), int(j), k)
                for j in np.flatnonzero(perpendicular[k] & hexagonal[i] & equal[i])
            )
    for i, j, k in sorted(triples):
        t0 = v[[i, j, k]].copy()
        determinant = integer_det(t0)
        if abs(determinant) not in (1, 2, 3, 4):
            continue
        orth = perpendicular[i, j] and perpendicular[i, k] and perpendicular[j, k]
        hexa = hexagonal[i, j] and equal[i, j] and perpendicular[i, k] and perpendicular[j, k]
        # Signs permit obverse R centring and the conventional 120-degree gamma.
        for sign in (
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ):
            if determinant * np.prod(sign) < 0:
                continue
            t = np.array(sign)[:, None] * t0
            centre = centring(t)
            symbols = []
            if perpendicular[i, j] and perpendicular[j, k] and centre in ("P", "C"):
                symbols.append("m" + centre)
            if orth:
                if centre in ("P", "C", "I", "F"):
                    symbols.append("o" + centre)
                if equal[i, j] and centre in ("P", "I"):
                    symbols.append("t" + centre)
                if equal[i, j] and equal[i, k] and centre in ("P", "I", "F"):
                    symbols.append("c" + centre)
            if hexa and cosine[i, j] * sign[0] * sign[1] < 0 and centre in ("P", "R"):
                symbols.append("h" + centre)
            if not symbols:
                continue
            reduced_t = t
            t = reduced_t @ reduction
            exact = t @ np.asarray(direct_basis)
            for symbol in symbols:
                try:
                    projected, strain = project_metric(exact, symbol)
                except np.linalg.LinAlgError:
                    continue
                if strain > max_strain:
                    continue
                key = _constraint_key(reduced_t, symbol)
                duplicate = next(
                    (
                        r
                        for r, (name, p) in enumerate(keys)
                        if name == symbol and np.max(abs(p - key)) < 1e-8
                    ),
                    None,
                )
                # Prefer short conventional axes, beta>=90, a<=b for orthogonal equal roles.
                cell = cell_parameters(exact)
                penalty = int(symbol[0] == "m" and cell[4] < 90 - 1e-8) + int(
                    symbol[0] in "oc" and cell[0] > cell[1] + 1e-8
                )
                score = (float(np.sum(exact * exact)) * (1 + penalty), tuple(t.ravel()))
                setting = BravaisSetting(symbol, t, exact, projected, strain)
                if duplicate is None:
                    keys.append((symbol, key))
                    scores.append(score)
                    result.append(setting)
                elif score < scores[duplicate]:
                    scores[duplicate] = score
                    result[duplicate] = setting
    return tuple(
        sorted(
            result, key=lambda s: (BRAVAIS.index(s.symbol), s.strain, tuple(s.transform.ravel()))
        )
    )
