"""Independent NumPy prediction equations; never calls native evaluators."""

import numpy as np


def coefficient_reference(value, expected, moment, correlation, gain):
    z = np.asarray(value, dtype=float)
    radius = np.linalg.norm(z)
    r_squared = min((radius / expected) ** 2 / moment, 16.0)
    amplitude = gain * correlation * expected * np.sqrt(1.0 - correlation + correlation * r_squared)
    return amplitude * z / np.sqrt(radius * radius + 0.01 * expected * expected * moment)


def statistics_reference(coefficients, measured, unit_scale=1.0):
    m = np.asarray(measured)
    calculated = (
        np.abs(np.asarray(coefficients).ravel()[m[:, 0].astype(int)]) * unit_scale / m[:, 2]
    ) ** 2
    observed = (m[:, 3] / m[:, 2]) ** 2
    result = []
    for i in range(8):
        selected = m[:, 1] == i
        if np.count_nonzero(selected) < 12:
            selected = np.ones(len(m), bool)
        a, b = observed[selected], calculated[selected]
        moment = float(np.clip(np.mean(b), 1e-12, 1e12))
        centered_a, centered_b = a - a.mean(), b - b.mean()
        denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
        c = (
            float(np.clip(centered_a @ centered_b / denominator, 0.0, 1.0))
            if denominator > 1e-100
            else 0.0
        )
        result.append((moment, c))
    return np.asarray(result)


def prediction_update_reference(coefficients, predicted, measured, cycle, unit_scale=1.0):
    out = np.asarray(coefficients).copy()
    flat = out.ravel()
    stats = statistics_reference(out, measured, unit_scale)
    ramp = float(np.clip((cycle - 7) / 16, 0.0, 1.0))
    for index, mate, bin_index, expected, taper, extended in predicted:
        index, mate, bin_index = int(index), int(mate), int(bin_index)
        z = flat[index] * unit_scale
        target = coefficient_reference([z.real, z.imag], expected, *stats[bin_index], ramp * taper)
        value = complex(*target)
        if not extended:
            value += (1 - ramp) * z
        flat[index] = value / unit_scale
        flat[mate] = flat[index].conjugate()
    return out
