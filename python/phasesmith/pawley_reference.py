"""Independent NumPy Pawley objective and small exhaustive bounded LS oracle.

This module never calls the native profile/objective/solver. Large production
fits belong to phasesmith.refinement.pawley, not this deliberately dense reference.
"""

from itertools import product

import numpy as np

from . import reference
from .refinement.core import ConstraintTransform
from .refinement.pawley import PawleyOptions, parameter_key


def _positions_and_chain(phase, values, wavelength):
    if phase.lattice is None:
        return phase.two_theta_deg, None
    par = phase.lattice.parameterization
    independent = np.array(
        [values[parameter_key("lattice", phase.phase_id, name)] for name in par.parameter_names]
    )
    cell = par.to_cell(independent)
    lengths = np.array(cell.as_tuple()[:3])
    angles = np.radians(cell.as_tuple()[3:])
    a, b, c = lengths
    ca, cb, cg = np.cos(angles)
    metric = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ]
    )
    dm = np.zeros((6, 3, 3))
    for j in range(3):
        for r in range(3):
            for s in range(3):
                dm[j, r, s] = metric[r, s] * ((r == j) + (s == j)) / lengths[j]
    for j, (r, s) in enumerate([(1, 2), (0, 2), (0, 1)]):
        dm[j + 3, r, s] = dm[j + 3, s, r] = (
            -lengths[r] * lengths[s] * np.sin(angles[j]) * np.pi / 180
        )
    reciprocal_h = np.linalg.solve(metric, phase.hkl.T).T
    q2 = np.einsum("ij,ij->i", phase.hkl, reciprocal_h)
    dq2 = -np.einsum("ri,jik,rk->rj", reciprocal_h, dm, reciprocal_h)
    spacing = 1 / np.sqrt(q2)
    ds = -0.5 * q2[:, None] ** (-1.5) * dq2 @ par.cell_parameter_jacobian(independent)
    argument = wavelength / (2 * spacing)
    positions = 2 * np.degrees(np.arcsin(argument))
    dp = -180 / np.pi * wavelength / (spacing**2 * np.sqrt(1 - argument**2))
    return positions, dp[:, None] * ds


_DEFAULT_OPTIONS = PawleyOptions()


def evaluate(request, free=None, options=_DEFAULT_OPTIONS):
    """Return full-grid values, scaled-free analytical Jacobian and weighted residual."""
    transform = ConstraintTransform(request.parameters, request.constraints)
    values = transform.unpack(transform.pack() if free is None else free)
    keys = request.parameters.keys
    index = {k: j for j, k in enumerate(keys)}
    x = request.pattern.x
    physical = np.zeros((len(x), len(keys)))
    profile = np.zeros(len(x))
    names = ("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg")
    instrument = {name: values[parameter_key("profile", "instrument", name)] for name in names}
    for phase in request.phases:
        positions, chain = _positions_and_chain(
            phase, values, request.instrument.wavelength_angstrom
        )
        areas = np.array(
            [
                values[parameter_key("intensity", phase.phase_id, name)]
                for name in phase.reflection_ids
            ]
        )
        kwargs = dict(instrument, support_fwhm=options.support_fwhm)
        if request.axial_geometry is None:
            y, local, global_j = reference.accumulate_cw(x, positions, areas, **kwargs)
        else:
            g = request.axial_geometry
            y, local, global_j = reference.accumulate_cw_fcj(
                x,
                positions,
                areas,
                sample_over_radius=g.sample_over_radius,
                detector_over_radius=g.detector_over_radius,
                **kwargs,
            )
        profile += y
        for r, name in enumerate(phase.reflection_ids):
            physical[:, index[parameter_key("intensity", phase.phase_id, name)]] = local[r, 0]
        for j, name in enumerate(names):
            physical[:, index[parameter_key("profile", "instrument", name)]] += global_j[j]
        if chain is not None:
            for j, name in enumerate(phase.lattice.parameterization.parameter_names):
                physical[:, index[parameter_key("lattice", phase.phase_id, name)]] = (
                    local[:, 1].T @ chain[:, j]
                )
    background = request.pattern.background.copy()
    if request.background is not None:
        bg = request.background
        coefficients = tuple(
            values[parameter_key("background", bg.background_id, name)]
            for name in bg.parameter_names
        )
        bg = bg.replace_coefficients(coefficients)
        background += bg.calculate(x)
        basis = bg.basis(x)
        for j, name in enumerate(bg.parameter_names):
            physical[:, index[parameter_key("background", bg.background_id, name)]] = basis[:, j]
    y = profile + background
    residual = y - request.pattern.observed_y
    weights = np.ones_like(x)
    if options.use_uncertainty and request.pattern.uncertainty is not None:
        weights /= request.pattern.uncertainty
    if request.pattern.mask is not None:
        weights *= request.pattern.mask
    return y, physical @ transform.derivative_matrix(), residual * weights


def exhaustive_linear_fit(design, target, nonnegative):
    """Small independent active-face enumeration using NumPy rank-revealing lstsq.

    nonnegative identifies bounded columns; other columns are unconstrained.
    This test oracle is intentionally limited to at most 12 bounded columns.
    """
    design, target = np.asarray(design), np.asarray(target)
    bounded = tuple(nonnegative)
    if len(bounded) > 12:
        raise ValueError("exhaustive reference supports at most 12 bounded columns")
    best = None
    for active in product((False, True), repeat=len(bounded)):
        fixed = {j for j, at_bound in zip(bounded, active, strict=True) if at_bound}
        free = [j for j in range(design.shape[1]) if j not in fixed]
        areas = np.zeros(design.shape[1])
        if free:
            areas[free] = np.linalg.lstsq(design[:, free], target, rcond=1e-12)[0]
        if any(areas[j] < -1e-10 for j in bounded):
            continue
        residual = design @ areas - target
        gradient = design.T @ residual
        if any(gradient[j] < -1e-8 for j in fixed):
            continue
        objective = residual @ residual
        if best is None or objective < best[0]:
            best = objective, areas
    if best is None:
        raise ValueError("reference found no feasible least-squares face")
    return best[1]
