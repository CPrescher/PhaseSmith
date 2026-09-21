"""Independent small active-set subproblems for coupled CW instrument widths.

These are proposal constraints, not changes to the physical profile domain.
All accepted trials still require the ordinary complete model evaluation.
"""

from __future__ import annotations

import numpy as np


def width_inequalities(experiment, calculation, parameters, derivative):
    count = derivative.shape[1]
    positions = np.concatenate(
        [p.reflections.two_theta_deg for p in calculation.phase_calculations]
    )
    if count == 0 or count > 64 or len(positions) + len(parameters.specs) > 2048:
        return None
    indices = [None] * 6
    names = ("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg", "zero_shift_deg")
    for i, spec in enumerate(parameters.specs):
        if not np.any(derivative[i]):
            continue
        if spec.key.module == "lattice":
            return None
        if spec.key.module == "instrument":
            if spec.key.name not in names:
                return None
            indices[names.index(spec.key.name)] = i
    if all(index is None for index in indices[:5]):
        return None
    instrument = experiment.instrument
    rows, lower = [], []

    def append(row, bound):
        if np.isfinite(bound) and np.any(row):
            rows.append(row)
            lower.append(bound)

    for position in positions:
        theta = position * np.pi / 360
        tangent, secant = np.tan(theta), 1 / np.cos(theta)
        gaussian = instrument.u_deg2 * tangent**2 + instrument.v_deg2 * tangent + instrument.w_deg2
        lorentzian = instrument.x_deg * secant + instrument.y_deg * tangent
        gaussian_row = (
            tangent**2,
            tangent,
            1.0,
            0.0,
            0.0,
            (2 * instrument.u_deg2 * tangent + instrument.v_deg2) * secant**2 * np.pi / 360,
        )
        lorentzian_row = (
            0.0,
            0.0,
            0.0,
            secant,
            tangent,
            (instrument.x_deg * secant * tangent + instrument.y_deg * secant**2) * np.pi / 360,
        )
        for coefficients, value in ((gaussian_row, gaussian), (lorentzian_row, lorentzian)):
            row = np.zeros(count)
            for index, coefficient in zip(indices, coefficients, strict=True):
                if index is not None:
                    row += coefficient * derivative[index]
            append(row, -0.99 * value)
    for i, spec in enumerate(parameters.specs):
        append(derivative[i], spec.bounds.lower - spec.value)
        append(-derivative[i], spec.value - spec.bounds.upper)
    return np.asarray(rows).reshape((-1, count)), np.asarray(lower)


def solve_quadratic(normal, rhs, constraints, tolerance):
    """Minimize 1/2 d'Hd - rhs'd with A d >= b from the feasible zero step.

    Use diagonal variable scaling and a QR null space of the working set.
    Return None if a finite, primal/dual/stationarity-checked answer is unavailable.
    """
    count = len(rhs)
    if count == 0 or count > 64:
        return None
    scale = np.sqrt(np.diag(normal))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        return None
    h = normal / scale[:, None] / scale[None, :]
    g = -rhs / scale
    a, b = constraints
    a = a / scale[None, :]
    norms = np.linalg.norm(a, axis=1)
    valid = (norms > 0) & np.isfinite(norms)
    a, b = a[valid] / norms[valid, None], b[valid] / norms[valid]
    if np.any(~np.isfinite(b)) or np.any(b > 1e-12):
        return None
    step = np.zeros(count)
    active = []
    try:
        for _ in range(20 * (count + 1)):
            gradient = h @ step + g
            if active:
                rank = len(active)
                q, r = np.linalg.qr(a[active].T, mode="complete")
                if np.any(np.abs(np.diag(r[:rank, :rank])) < 1e-10):
                    return None
                z = q[:, rank:]
                if rank == count:
                    direction = np.zeros(count)
                else:
                    factor = np.linalg.cholesky(z.T @ h @ z)
                    direction = -z @ np.linalg.solve(
                        factor.T, np.linalg.solve(factor, z.T @ gradient)
                    )
                multipliers = np.linalg.solve(r[:rank, :rank], q[:, :rank].T @ gradient)
            else:
                factor = np.linalg.cholesky(h)
                direction = -np.linalg.solve(factor.T, np.linalg.solve(factor, gradient))
                multipliers = np.empty(0)
            if np.any(~np.isfinite(direction)):
                return None
            if np.linalg.norm(direction) <= 1e-9 * (1 + np.linalg.norm(step)):
                if len(multipliers) and np.min(multipliers) < -1e-8:
                    active.pop(int(np.argmin(multipliers)))
                    continue
                residual = gradient - a[active].T @ multipliers
                if np.linalg.norm(residual) > tolerance * max(np.linalg.norm(g), 1.0) or np.any(
                    a @ step - b < -1e-10 * (1 + np.abs(b))
                ):
                    return None
                return step / scale
            alpha, blocker = 1.0, None
            change, slack = a @ direction, a @ step - b
            for i in range(len(b)):
                if i not in active and change[i] < -1e-12:
                    candidate = max(slack[i], 0.0) / -change[i]
                    if candidate < alpha:
                        alpha, blocker = candidate, i
            step += alpha * direction
            if blocker is not None:
                if len(active) == count:
                    return None
                active.append(blocker)
    except np.linalg.LinAlgError:
        return None
    return None
