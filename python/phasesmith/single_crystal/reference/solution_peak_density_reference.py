"""Independent quadrature and direct-sum reference for spherical peak masses."""

import itertools

import numpy as np


def ball_reference(q, radius):
    """Integrate Cartesian cross-section disks with Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    cosine = np.cos(2 * np.pi * np.asarray(q)[..., None] * radius * nodes)
    value = np.pi * radius**3 * np.sum(cosine * (1 - nodes**2) * weights, axis=-1)
    radial = 2 * np.pi * radius**2 * np.sum(cosine * weights, axis=-1)
    return value, radial


def integrated_reference(basis, shape, hkl, amplitudes, source, radius, point):
    """Direct Fourier sums at eight interpolation nodes, without a native kernel."""
    hkl = np.asarray(hkl)
    source_f = np.fft.ifftn(source) * np.prod(shape)
    phases = source_f[tuple((hkl % shape).T)]
    valid = np.abs(phases) > np.abs(source_f).max() * 1e-12
    phase = np.divide(phases, np.abs(phases), out=np.zeros_like(phases), where=valid)
    q = np.linalg.norm(hkl @ np.linalg.inv(basis), axis=1)
    kernel, radial = ball_reference(q, radius)
    coefficients = np.asarray(amplitudes) * phase / abs(np.linalg.det(basis))
    grid = np.mod(point, 1) * shape
    base = np.floor(grid).astype(int)
    delta = grid - base
    value = 0.0
    gradient = np.zeros(3)
    dr = 0.0
    for bits in itertools.product((0, 1), repeat=3):
        center = (base + bits) % shape / shape
        waves = np.exp(-2j * np.pi * (hkl @ center))
        local = np.real(np.sum(coefficients * kernel * waves))
        local_dr = np.real(np.sum(coefficients * radial * waves))
        factors = np.where(bits, delta, 1 - delta)
        weight = np.prod(factors)
        value += weight * local
        dr += weight * local_dr
        for axis in range(3):
            remaining = np.delete(factors, axis)
            gradient[axis] += (1 if bits[axis] else -1) * shape[axis] * np.prod(remaining) * local
    return value, gradient, dr


def label_mask_reference(mass, scale, numbers):
    if scale is None or mass <= 0:
        return [True] * len(numbers)
    error = np.abs(np.log((mass / scale) / np.asarray(numbers)))
    return list((error <= error.min() + np.log(1.25)) | (error <= np.log(1.5)))
