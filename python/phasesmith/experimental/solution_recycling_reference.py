"""Independent dense NumPy phase averaging and positive-density cycles."""

import numpy as np


def recycle_reference(
    source, hkl, amplitude, support, rotations, translations, *, hand=1, shift=(0, 0, 0), cycles=10
):
    rho = np.asarray(source)
    shape = np.array(rho.shape)
    hkl = np.asarray(hkl)
    f = np.prod(shape) * np.fft.ifftn(rho)
    frequencies = np.moveaxis(np.indices(shape), 0, -1)
    signed = np.where(frequencies < shape // 2, frequencies, frequencies - shape)
    aligned = np.zeros(rho.shape, dtype=complex)
    active = np.argwhere(np.asarray(support) > 0)
    for index in active:
        h = signed[tuple(index)]
        aligned[tuple(index)] = f[tuple(hand * h % shape)] * np.exp(2j * np.pi * (h @ shift))
    target = np.full(rho.shape, np.nan)
    target[tuple((hkl % shape).T)] = amplitude

    def project(values):
        output = np.zeros_like(values)
        for index in active:
            h = signed[tuple(index)]
            mapped = np.einsum("i,gij->gj", h, rotations)
            positive = values[tuple((mapped % shape).T)]
            negative = values[tuple((-mapped % shape).T)].conj()
            norms = np.abs(positive)
            other = np.abs(negative)
            units = np.divide(positive, norms, out=np.zeros_like(positive), where=norms > 1e-100)
            units += np.divide(negative, other, out=np.zeros_like(negative), where=other > 1e-100)
            mean = np.mean(units * np.exp(2j * np.pi * (np.asarray(translations) @ h))) / 2
            if abs(mean) <= 1e-12:
                raise ValueError("reference test crosses a degenerate phasor average")
            phase = mean / abs(mean)
            scale = target[tuple(index)]
            if np.isnan(scale):
                scale = np.mean(np.r_[norms, other])
            output[tuple(index)] = phase * scale
        return output

    fractions = []
    for _ in range(cycles):
        rho = np.fft.fftn(project(aligned)).real / np.prod(shape)
        fractions.append(float(np.mean(rho < 0)))
        aligned = np.prod(shape) * np.fft.ifftn(np.maximum(rho, 0))
    return np.fft.fftn(project(aligned)).real / np.prod(shape), fractions
