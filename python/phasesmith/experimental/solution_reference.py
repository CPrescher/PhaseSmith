"""Independent NumPy equations for experimental phasing and amplitude polishing.

This verification code never calls the Rust solver or its FFT implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def charge_flip_reference(
    shape: tuple[int, int, int],
    hkl: ArrayLike,
    amplitudes: ArrayLike,
    phases: ArrayLike,
    support: ArrayLike,
    *,
    cycles: int,
    delta_sigma: float,
    weak_fraction: float = 0.2,
    cleanup_cycles: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Readable complete cycles; callers supply already-validated Hermitian inputs."""
    hkl, amplitudes, phases = np.asarray(hkl), np.asarray(amplitudes), np.asarray(phases)
    support = np.asarray(support).reshape(shape)
    indices = tuple((hkl % shape).T)
    n = int(np.prod(shape))
    f = np.zeros(shape, dtype=complex)
    f[indices] = amplitudes * np.exp(1j * phases)
    rho = np.fft.fftn(f).real / n
    delta = delta_sigma * np.sqrt(np.mean(rho**2))
    weak_count = int(np.floor(weak_fraction * len(hkl)))
    weak = (
        amplitudes <= np.sort(amplitudes)[weak_count - 1]
        if weak_count
        else np.zeros(len(hkl), bool)
    )
    signs = np.array([1.0 if tuple(h) > tuple(-h) else -1.0 for h in hkl])
    best, best_score, best_cycle = rho.copy(), np.inf, 0
    history = []

    def constrain(g: np.ndarray, perturb: bool) -> np.ndarray:
        g[support == 0] = 0
        values = g[indices]
        norms = np.abs(values)
        phase = np.divide(values, norms, out=np.ones_like(values), where=norms > 0)
        g[indices] = (
            np.where(weak, values * 1j * signs, amplitudes * phase)
            if perturb
            else amplitudes * phase
        )
        return g

    for cycle in range(cycles):
        g = n * np.fft.ifftn(np.where(rho < delta, -rho, rho))
        norms = np.abs(g[indices])
        score = (
            float(np.sum(np.abs(amplitudes / amplitudes.sum() - norms / norms.sum())))
            if norms.sum()
            else 2.0
        )
        history.append(score)
        if (cycle >= 20 or cycles <= 20) and score < best_score:
            best, best_score, best_cycle = rho.copy(), score, cycle
        rho = np.fft.fftn(constrain(g, True)).real / n
    if not np.isfinite(best_score):
        best = rho
    for _ in range(cleanup_cycles):
        g = n * np.fft.ifftn(np.where(best < delta, 0.0, best))
        best = np.fft.fftn(constrain(g, False)).real / n
    return best, np.array(history), best_cycle


def amplitude_model_reference(
    hkl: ArrayLike,
    observed: ArrayLike,
    xyz: ArrayLike,
    scattering: ArrayLike,
    s: ArrayLike,
    u: float,
) -> dict[str, object]:
    """Direct complex sums and analytic derivatives, independently of native SF."""
    hkl, observed, xyz = np.asarray(hkl), np.asarray(observed), np.asarray(xyz)
    terms = np.asarray(scattering) * np.exp(-8 * np.pi**2 * u * np.asarray(s)[:, None] ** 2)
    terms = terms * np.exp(2j * np.pi * (hkl @ xyz.T))
    f = terms.sum(axis=1)
    amplitude = np.abs(f)
    columns = []
    for site in range(1, len(xyz)):
        for axis in range(3):
            df = terms[:, site] * (2j * np.pi * hkl[:, axis])
            columns.append(
                np.divide(
                    (f.conj() * df).real,
                    amplitude,
                    out=np.zeros_like(amplitude),
                    where=amplitude > 1e-14,
                )
            )
    columns.append(-8 * np.pi**2 * np.asarray(s) ** 2 * amplitude)
    derivative = np.asarray(columns)
    denominator = amplitude @ amplitude
    scale = (amplitude @ observed) / denominator
    dk = (derivative @ observed - 2 * scale * (derivative @ amplitude)) / denominator
    norm = np.linalg.norm(observed)
    residual = (scale * amplitude - observed) / norm
    return {
        "residual": residual,
        "jacobian": (scale * derivative + dk[:, None] * amplitude) / norm,
        "objective": float(residual @ residual),
        "amplitude_scale": float(scale),
        "r1": float(np.abs(scale * amplitude - observed).sum() / observed.sum()),
    }


def symmetric_model_reference(metadata, cell_basis, hkl, observed, scattering, parameters):
    """Independent direct NumPy sums for a fixed symmetry topology.

    The topology supplies exact operations and tangent bases; no native values,
    eigensolver, structure-factor evaluator or analytical derivatives are used.
    Returns physical tensors and normalized residuals. Derivatives can be
    checked by centered differences of this function.
    """
    parameters = np.asarray(parameters)
    hkl, observed = np.asarray(hkl), np.asarray(observed)
    reciprocal = hkl @ np.linalg.inv(np.asarray(cell_basis))
    calculated = np.zeros(len(hkl), dtype=complex)
    positions, tensors = [], []
    for i, site in enumerate(metadata["sites"]):
        coordinate_offset, tensor_offset = metadata["offsets"][i]
        directions = np.asarray(site["coordinate_basis"]).reshape(-1, 3)
        position = (
            np.asarray(site["fractional_xyz"])
            + parameters[coordinate_offset : coordinate_offset + len(directions)] @ directions
        )
        basis = np.asarray(metadata["tensor_bases"][i])
        logarithm = np.einsum(
            "i,ijk->jk", parameters[tensor_offset : tensor_offset + len(basis)], basis
        )
        eigenvalues, eigenvectors = np.linalg.eigh(logarithm)
        tensor = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
        positions.append(position)
        tensors.append(tensor)
        for image in site["images"]:
            xyz = np.asarray(image["rotation"]) @ position + image["translation"]
            rotation = np.asarray(image["cartesian_rotation"])
            rotated = rotation @ tensor @ rotation.T
            exponent = -2 * np.pi**2 * np.einsum("ri,ij,rj->r", reciprocal, rotated, reciprocal)
            calculated += scattering[:, i] * np.exp(exponent + 2j * np.pi * (hkl @ xyz))
    amplitudes = np.abs(calculated)
    scale = (amplitudes @ observed) / (amplitudes @ amplitudes)
    residual = (scale * amplitudes - observed) / np.linalg.norm(observed)
    return {
        "amplitudes": amplitudes,
        "amplitude_scale": scale,
        "residual": residual,
        "objective": residual @ residual,
        "r1": np.abs(scale * amplitudes - observed).sum() / observed.sum(),
        "fractional_xyz": np.asarray(positions),
        "u_cartesian": np.asarray(tensors),
    }
