"""Independent dense covariance equations for signed empirical shape variations."""

import numpy as np

from .profiles_reference import evaluate, sample_bounds


def fit(observations, model, factors, assignments, fixed_fractions=None):
    o = observations
    t = o.summation
    k = len(factors)
    angles = np.sort(t.omega_bounds[t.frame_indices], axis=1)
    bounds, design = sample_bounds(o.offsets, t.boxes, angles)
    values = []
    coefficients = []
    covariances = []
    for reflection, centre in enumerate(o.centres):
        bank = assignments[reflection]
        centre = centre.copy()
        centre[:2] += np.linalg.solve(
            o.planar_transforms[reflection], model.shifts[bank, :2] / o.scales[reflection, :2]
        )
        centre[2] += model.shifts[bank, 2] / o.scales[reflection, 2]
        ids = np.flatnonzero(np.repeat(t.observation_ids == reflection, np.diff(o.offsets)))
        c = np.zeros((len(ids), len(ids)))
        position = {int(v): i for i, v in enumerate(ids)}
        for g in np.flatnonzero(t.observation_ids == reflection):
            local = np.arange(o.offsets[g], o.offsets[g + 1])
            where = np.array([position[int(v)] for v in local])
            x = design[local]
            c[np.ix_(where, where)] = (
                np.diag(o.samples[local, 1]) + x @ o.covariance[g].reshape(3, 3) @ x.T
            )
        ps = []
        partials = []
        for factor in factors:
            step = np.array(model.options.step) * factor
            if np.array_equal(o.planar_transforms[reflection], np.eye(2)):
                v = evaluate(model.volumes[bank], bounds[ids], step, centre, o.scales[reflection])
            else:
                from .profiles_affine_reference import evaluate as affine

                v = affine(
                    model.volumes[bank],
                    bounds[ids],
                    step,
                    centre,
                    o.planar_transforms[reflection],
                    o.scales[reflection],
                )
            ps.append(v[:, 0])
            partials.append(v[:, 1:])
        p = np.column_stack(ps)
        dpartial = np.stack(partials, axis=1)
        selected = o.valid[ids] & (p.sum(axis=1) > 0)
        coverage = p[o.valid[ids]].sum(axis=0)
        p = p[selected]
        dpartial = dpartial[selected]
        d = o.samples[ids[selected], 0]
        c = c[np.ix_(selected, selected)]
        w = np.linalg.solve(c, np.column_stack([p, d]))
        wp = w[:, :k]
        wd = w[:, -1]
        covariance = np.linalg.inv(p.T @ wp)
        normal = p.T @ wp
        rhs = p.T @ wd
        beta = np.zeros(k)
        best = 0.0
        for mask in range(1, 2**k):
            active = np.array([j for j in range(k) if mask & (1 << j)])
            candidate = np.zeros(k)
            candidate[active] = np.linalg.solve(normal[np.ix_(active, active)], rhs[active])
            if not (np.all(candidate >= 0) or np.all(candidate <= 0)):
                continue
            cost = candidate @ normal @ candidate - 2 * candidate @ rhs
            if cost < best:
                beta = candidate
                best = cost
        if fixed_fractions is not None:
            f = fixed_fractions[reflection]
            den = f @ normal @ f
            amplitude = f @ rhs / den
            beta = amplitude * f
            covariance = np.outer(f, f) / den
        intensity = beta.sum()
        coverage = beta @ coverage / intensity if intensity != 0 else coverage.min()
        signal = p @ beta
        residual = d - signal
        wr = np.linalg.solve(c, residual)
        chi = residual @ wr / max(len(d) - (1 if fixed_fractions is not None else k), 1)
        gradient = -2 * np.einsum("j,ijk,i->k", beta, dpartial, wr)
        correlation = np.corrcoef(signal / intensity, d)[0, 1]
        values.append(
            [
                intensity,
                np.sqrt(covariance.sum()),
                coverage,
                chi,
                len(d),
                correlation,
                -2 * (signal / intensity) @ wr,
                *gradient,
            ]
        )
        coefficients.append(beta)
        covariances.append(covariance)
    return np.array(values), np.array(coefficients), np.array(covariances)
