"""Independent NumPy TOF Pawley values and analytical parameter chains.

No native profile or optimizer calls are made. The reference is deliberately
small and dense; production evaluation and iteration are owned by Rust.
"""

from dataclasses import asdict

import numpy as np

from . import reference
from .instrument import TofInstrument
from .refinement.core import ConstraintTransform
from .refinement.pawley import PawleyOptions, parameter_key

_NAMES = (
    "zero",
    "difc",
    "difa",
    "difb",
    "alpha",
    "beta0",
    "beta1",
    "betaq",
    "sigma0",
    "sigma1",
    "sigma2",
    "sigmaq",
    "x",
    "y",
    "z",
)
_DEFAULT_OPTIONS = PawleyOptions()


def _geometry(shared, phase, values):
    par = shared.parameterization
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
    rh = np.linalg.solve(metric, phase.hkl.T).T
    q2 = np.einsum("ij,ij->i", phase.hkl, rh)
    dq2 = -np.einsum("ri,jik,rk->rj", rh, dm, rh)
    return 1 / np.sqrt(q2), -0.5 * q2[:, None] ** (-1.5) * dq2 @ par.cell_parameter_jacobian(
        independent
    )


def evaluate(request, free=None, options=_DEFAULT_OPTIONS):
    """Return concatenated values, scaled-free Jacobian and weighted residual."""
    t = ConstraintTransform(request.parameters, request.constraints)
    values = t.unpack(t.pack() if free is None else free)
    index = {k: j for j, k in enumerate(request.parameters.keys)}
    n = request.sample_offsets[-1]
    physical = np.zeros((n, len(index)))
    ys = []
    residuals = []
    cells = {c.phase_id: c for c in request.shared_lattice}
    for bi, bank in enumerate(request.banks):
        x = bank.pattern.tof_us
        offset = request.sample_offsets[bi]
        y = bank.pattern.background.copy()
        coefficients = [values[parameter_key("profile", bank.bank_id, name)] for name in _NAMES]
        instrument = TofInstrument(*coefficients)
        for phase in bank.phases:
            shared = cells.get(phase.phase_id)
            d, cell_chain = (
                (phase.d_spacing_angstrom, None)
                if shared is None
                else _geometry(shared, phase, values)
            )
            derived = reference.tof_profile_parameters(d, **asdict(instrument))
            owner = f"{bank.bank_id}:{phase.phase_id}"
            for r, name in enumerate(phase.reflection_ids):
                area = values[parameter_key("intensity", owner, name)]
                dr = d[r]
                radius = (
                    options.support_fwhm
                    * reference.tch_shape_from_fwhm(
                        derived.gaussian_fwhm_us[r], derived.lorentzian_fwhm_us[r]
                    ).total_fwhm
                )
                active = (
                    x
                    >= derived.position_us[r] - radius - request.tail_log / derived.alpha_per_us[r]
                ) & (
                    x <= derived.position_us[r] + radius + request.tail_log / derived.beta_per_us[r]
                )
                point = reference.profile_tof(
                    x[active],
                    derived.position_us[r],
                    derived.alpha_per_us[r],
                    derived.beta_per_us[r],
                    derived.gaussian_fwhm_us[r],
                    derived.lorentzian_fwhm_us[r],
                    tail_log=request.tail_log,
                    base_radius_us=radius,
                    quadrature_order=768,
                )
                samples = offset + np.flatnonzero(active)
                y[active] += area * point.value
                physical[samples, index[parameter_key("intensity", owner, name)]] = point.value
                direct = np.array(
                    [
                        point.d_position,
                        point.d_alpha,
                        point.d_beta,
                        point.d_gaussian_fwhm,
                        point.d_lorentzian_fwhm,
                    ]
                ).T
                chain = np.zeros((5, 15))
                chain[0, :4] = [1, dr, dr**2, 1 / dr]
                chain[1, 4] = 1 / dr
                chain[2, 5:8] = [1, dr**-4, dr**-2]
                g = 2.3548200450309493 / (2 * np.sqrt(derived.gaussian_variance_us2[r]))
                chain[3, 8:12] = g * np.array([1, dr**2, dr**4, dr])
                chain[4, 12:15] = [dr, dr**2, 1]
                global_j = area * direct @ chain
                for j, pname in enumerate(_NAMES):
                    physical[samples, index[parameter_key("profile", bank.bank_id, pname)]] += (
                        global_j[:, j]
                    )
                if cell_chain is not None:
                    c = coefficients
                    dchain = np.array(
                        [
                            c[1] + 2 * c[2] * dr - c[3] / dr**2,
                            -c[4] / dr**2,
                            -4 * c[6] / dr**5 - 2 * c[7] / dr**3,
                            g * (2 * c[9] * dr + 4 * c[10] * dr**3 + c[11]),
                            c[12] + 2 * c[13] * dr,
                        ]
                    )
                    local = area * direct @ dchain
                    for j, pname in enumerate(shared.parameterization.parameter_names):
                        physical[
                            samples, index[parameter_key("lattice", phase.phase_id, pname)]
                        ] += local * cell_chain[r, j]
        if bank.background is not None:
            bg = bank.background
            lo, hi = bg.domain_us
            basis = np.polynomial.chebyshev.chebvander(
                2 * (x - lo) / (hi - lo) - 1, len(bg.coefficients) - 1
            )
            for j in range(len(bg.coefficients)):
                key = parameter_key("background", bank.bank_id, f"c{j}")
                y += values[key] * basis[:, j]
                physical[offset : offset + len(x), index[key]] = basis[:, j]
        residual = y - bank.pattern.observed_y
        if options.use_uncertainty and bank.pattern.uncertainty is not None:
            residual = residual / bank.pattern.uncertainty
        if bank.pattern.mask is not None:
            residual = residual * bank.pattern.mask
        ys.append(y)
        residuals.append(residual)
    return np.concatenate(ys), physical @ t.derivative_matrix(), np.concatenate(residuals)
