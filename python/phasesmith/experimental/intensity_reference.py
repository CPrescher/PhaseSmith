"""Independent NumPy equations for slab and empirical intensity kernels.

No Rust calls, image readers, datasets or structure factors enter this module.
"""

from itertools import pairwise

import numpy as np


def slab_response(cosine, thickness, mu):
    c = np.asarray(cosine, float)
    tau = mu * thickness / c
    if mu * thickness == 0:
        return np.column_stack(
            [
                np.zeros_like(c),
                c,
                np.full_like(c, thickness / 2),
                np.zeros_like(c),
                np.ones_like(c),
                np.zeros_like(c),
            ]
        )
    # Series is evaluated only where needed to avoid subtraction near zero.
    eta = -np.expm1(-tau)
    relative = -np.expm1(-mu * thickness) / eta
    depth = np.empty_like(c)
    dh = np.empty_like(c)
    small = tau < 1e-3
    t = tau[small]
    depth[small] = thickness * (0.5 - t / 12 + t**3 / 720 - t**5 / 30240)
    dh[small] = thickness / c[small] * (t / 12 - t**3 / 240 + t**5 / 6048)
    t = tau[~small]
    e = np.exp(-t)
    depth[~small] = thickness * (1 / t - e / eta[~small])
    dh[~small] = thickness / c[~small] * (1 / t - t * e / eta[~small] ** 2)
    de = -tau * np.exp(-tau) / c
    return np.column_stack([eta, relative, depth, de, -relative * de / eta, dh])


def empirical_model(frames, detector, harmonics, coefficients, floor=1e-8):
    nf, nd = frames.shape[1], detector.shape[1]
    d = 1 + detector @ coefficients[nf : nf + nd]
    a = 1 + harmonics @ coefficients[nf + nd :]
    value = frames @ coefficients[:nf] + np.log(np.maximum(d, floor)) + np.log(np.maximum(a, floor))
    jacobian = np.column_stack(
        [
            frames,
            detector * ((d > floor) / np.maximum(d, floor))[:, None],
            harmonics * ((a > floor) / np.maximum(a, floor))[:, None],
        ]
    )
    return value, np.column_stack([d, a]), jacobian


def family_center(values, groups, weights):
    x = np.asarray(values, float)
    sums = np.zeros((int(np.max(groups, initial=-1)) + 1, x.shape[1]))
    totals = np.zeros(len(sums))
    np.add.at(sums, groups, weights[:, None] * x)
    np.add.at(totals, groups, weights)
    return x - sums[groups] / totals[groups, None]


def profile_precision(weights, basis, offsets, covariance):
    """Small independent dense reference; production never forms this matrix."""
    w = np.diag(weights)
    for g, (lo, hi) in enumerate(pairwise(offsets)):
        x = basis[lo:hi]
        diagonal = np.diag(weights[lo:hi])
        c = covariance[g].reshape(3, 3)
        update = np.linalg.solve(np.eye(3) + c @ (x.T @ diagonal @ x), c)
        w[lo:hi, lo:hi] -= diagonal @ x @ update @ x.T @ diagonal
    return w


def _real(value, name, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite real")
    return np.ascontiguousarray(a, float)


def reference_paired_mask(groups, mask):
    groups, mask = np.asarray(groups), np.asarray(mask)
    count = np.bincount(groups[mask], minlength=len(groups))
    return mask & (count[groups] >= 2)


def fit_empirical_scale_reference(
    design, intensity, log_variance, groups, reference_mask, *, options=None, start=None
):
    """Robust within-family log fit, preserving every signed output observation.

    References must be positive with positive log variance. Weak/negative rows
    remain in output, but do not teach their own multiplicative correction.
    ``groups`` must represent independently declared intensity equivalence;
    lattice metric alone does not establish that equivalence.
    """
    from types import SimpleNamespace

    from scipy.optimize import least_squares

    o = (
        SimpleNamespace(strength=1000.0, huber=2.5, iterations=100, minimum_component=0.2)
        if options is None
        else options
    )
    y = _real(intensity, "intensity", 1)
    var = _real(log_variance, "log variance", 1)
    if y.shape != (len(design.frames),) or var.shape != y.shape or np.any(var <= 0):
        raise ValueError("invalid intensity/variance dimensions")
    mask = reference_paired_mask(groups, reference_mask)
    if np.count_nonzero(mask) < 2 or np.any(y[mask] <= 0):
        raise ValueError("scaling requires positive paired references")
    # Compress labels after subsetting: Rust budgets groups by reference count.
    ids = np.flatnonzero(mask)
    _, g = np.unique(np.asarray(groups)[ids], return_inverse=True)
    target = np.log(y[ids])
    basew = 1 / var[ids]
    w = basew.copy()
    b = np.zeros(design.parameters) if start is None else _real(start, "start", 1).copy()
    if b.shape != design.lower.shape or np.any(b < design.lower) or np.any(b > design.upper):
        raise ValueError("starting coefficients outside design bounds")

    def evaluate(coefficients, rows=slice(None)):
        return empirical_model(
            design.frames[rows], design.detector[rows], design.harmonics[rows], coefficients
        )

    history = []
    rootreg = np.sqrt(o.strength)
    for _ in range(o.iterations):
        root = np.sqrt(w)
        centered_target = family_center(target[:, None], g, w)[:, 0]

        def residual(p, root=root, w=w, centered_target=centered_target):
            v, _, _ = evaluate(p, ids)
            _, all_components, _ = evaluate(p)
            # Same explicit positive-component barrier for every integration.
            barrier = 1000 * min(float(all_components.min()) - o.minimum_component, 0.0)
            return np.r_[
                root * (family_center(v[:, None], g, w)[:, 0] - centered_target),
                rootreg * (design.penalty @ p),
                barrier,
            ]

        def jacobian(p, root=root, w=w):
            _, _, j = evaluate(p, ids)
            _, comp, _ = evaluate(p)
            k, sector = np.unravel_index(np.argmin(comp), comp.shape)
            barrier = np.zeros(design.parameters)
            if comp[k, sector] < o.minimum_component:
                nf = design.frames.shape[1]
                nd = design.detector.shape[1]
                sl = slice(nf, nf + nd) if sector == 0 else slice(nf + nd, None)
                raw = design.detector if sector == 0 else design.harmonics
                barrier[sl] = 1000 * raw[k]
            return np.r_[
                root[:, None] * family_center(j, g, w), rootreg * design.penalty, barrier[None]
            ]

        fit = least_squares(
            residual,
            b,
            jac=jacobian,
            bounds=(design.lower, design.upper),
            x_scale="jac",
            max_nfev=80,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-7,
        )
        v = evaluate(fit.x, ids)[0]
        r = family_center((target - v)[:, None], g, w)[:, 0]
        neww = basew * np.minimum(1.0, o.huber / np.maximum(abs(r) / np.sqrt(var[ids]), 1e-30))
        db = float(np.linalg.norm(fit.x - b))
        dw = float(np.max(abs(neww - w) / basew))
        b, w = fit.x, neww
        history.append(dict(parameter_change=db, weight_change=dw, inner_success=bool(fit.success)))
        if db < 1e-6 and dw < 1e-5:
            break
    v, components, _ = evaluate(b)
    if np.any(components < o.minimum_component - 1e-6) or not np.isfinite(np.exp(-v)).all():
        raise ValueError("empirical correction leaves its declared physical domain")
    return SimpleNamespace(
        coefficients=b,
        correction=np.exp(-v),
        components=components,
        reference_mask=mask,
        converged=bool(db < 1e-6 and dw < 1e-5),
        history=tuple(history),
    )
