"""Independent NumPy equations for partial-framework completion.

All scattering values are explicit inputs; no production SF or native projection
routine is invoked. These dense equations prioritize clarity over performance.
"""

from __future__ import annotations

import itertools

import numpy as np


def orbit_reference(
    hkl,
    observed,
    fixed_xyz,
    fixed_scattering,
    added_scattering,
    s,
    u_iso,
    point,
    rotations,
    translations,
):
    hkl, observed = np.asarray(hkl), np.asarray(observed)
    dw = np.exp(-8 * np.pi**2 * u_iso * np.asarray(s) ** 2)
    base = (
        np.asarray(fixed_scattering)
        * dw[:, None]
        * np.exp(2j * np.pi * (hkl @ np.asarray(fixed_xyz).T))
    ).sum(axis=1)
    images = np.asarray(rotations) @ np.asarray(point) + np.asarray(translations)
    contribution = (
        np.asarray(added_scattering)[:, None] * dw[:, None] * np.exp(2j * np.pi * (hkl @ images.T))
    )
    calculated = base + contribution.sum(axis=1)
    amplitude = np.abs(calculated)
    scale = (amplitude @ observed) / (amplitude @ amplitude)
    residual = (scale * amplitude - observed) / np.linalg.norm(observed)
    return {
        "base": base,
        "fractional_xyz": images % 1,
        "objective": residual @ residual,
        "residual": residual,
        "amplitude_scale": scale,
        "r1": np.abs(scale * amplitude - observed).sum() / observed.sum(),
    }


def difference_reference(shape, hkl, observed, base):
    hkl, observed, base = np.asarray(hkl), np.asarray(observed), np.asarray(base)
    amplitude = np.abs(base)
    scale = (observed @ amplitude) / (amplitude @ amplitude)
    phase = np.divide(
        base, amplitude, out=np.zeros_like(base), where=amplitude > amplitude.max() * 1e-10
    )
    values = (observed / scale - amplitude) * phase
    values[np.all(hkl == 0, axis=1)] = 0
    coefficients = np.zeros(shape, dtype=complex)
    coefficients[tuple((hkl % shape).T)] = values
    return np.fft.fftn(coefficients).real / np.prod(shape), scale


def project_reference(basis, seed, rotations, translations, tolerance):
    basis, seed = np.asarray(basis), np.asarray(seed)
    jacobian, residual = [], []
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    for rotation, translation in zip(rotations, translations, strict=True):
        raw = np.asarray(rotation) @ seed + translation - seed
        images = raw - np.rint(raw) + shifts
        norms = np.linalg.norm(images @ basis.T, axis=1)
        difference = images[np.argmin(norms)]
        jacobian.append(basis @ (np.asarray(rotation) - np.eye(3)) @ np.linalg.inv(basis))
        residual.append(basis @ difference)
    if not jacobian:
        return seed % 1
    matrix = np.vstack(jacobian)
    rhs = np.concatenate(residual)
    delta = np.linalg.lstsq(matrix, -rhs, rcond=1e-10)[0]
    if (
        np.linalg.norm(delta) > tolerance * (1 + 1e-12)
        or np.max(np.linalg.norm((matrix @ delta + rhs).reshape(-1, 3), axis=1)) > 1e-8
    ):
        return None
    return (seed + np.linalg.solve(basis, delta)) % 1
