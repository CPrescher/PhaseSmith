"""Independent dense NumPy equations for shared orbit/element proposals.

Scattering columns and distinct physical operations are supplied explicitly.
No native shared-search or orbit evaluator is called here.
"""

import numpy as np


def shared_orbit_reference(
    hkl, observed, fixed_xyz, fixed_factors, added_factors, point, rotations, translations
):
    """Return phase jet and profiled fit/Jacobians for each candidate species."""
    h = np.asarray(hkl, dtype=float)
    observed = np.asarray(observed)
    images = (np.einsum("gij,j->gi", rotations, point) + translations) % 1
    phase = np.exp(2j * np.pi * (h @ images.T))
    q = phase.sum(axis=1)
    dq = 2j * np.pi * np.einsum("hg,hgj->hj", phase, np.einsum("hi,gij->hgj", h, rotations))
    base = (np.exp(2j * np.pi * (h @ np.asarray(fixed_xyz).T)) * fixed_factors).sum(axis=1)
    results = []
    for factor in np.asarray(added_factors).T:
        total = base + factor * q
        amplitude = np.abs(total)
        da = np.divide(
            (total.conj()[:, None] * factor[:, None] * dq).real,
            amplitude[:, None],
            out=np.zeros_like(dq.real),
            where=amplitude[:, None] > 1e-14,
        )
        denominator = amplitude @ amplitude
        scale = (amplitude @ observed) / denominator
        dk = da.T @ (observed - 2 * scale * amplitude) / denominator
        residual = (scale * amplitude - observed) / np.linalg.norm(observed)
        jacobian = (scale * da.T + dk[:, None] * amplitude) / np.linalg.norm(observed)
        results.append(
            {
                "objective": residual @ residual,
                "r1": np.abs(scale * amplitude - observed).sum() / observed.sum(),
                "amplitude_scale": scale,
                "residual": residual,
                "jacobian": jacobian,
                "gradient": 2 * jacobian @ residual,
            }
        )
    return {"q": q, "dq": dq, "images": images, "evaluations": results}
