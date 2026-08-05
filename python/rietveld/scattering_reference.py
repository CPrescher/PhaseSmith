"""Small independent NumPy reference for reviewed scattering equations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Authored transcription of selected rows from the pinned source. Production
# code does not import this compact verification table.
XRAY_COEFFICIENTS = {
    "H": (
        (0.413048, 0.294953, 0.187491, 0.080701, 0.023736),
        (15.569946, 32.398468, 5.711404, 61.889874, 1.334118),
        0.000049,
    ),
    "C": (
        (2.657506, 1.078079, 1.490909, -4.241070, 0.713791),
        (14.780758, 0.776775, 42.086842, -0.000294, 0.239535),
        4.297983,
    ),
    "O": (
        (2.960427, 2.508818, 0.637853, 0.722838, 1.142756),
        (14.182259, 5.936858, 0.112726, 34.958481, 0.390240),
        0.027014,
    ),
    "Si": (
        (5.275329, 3.191038, 1.511514, 1.356849, 2.519114),
        (2.631338, 33.730728, 0.081119, 86.288643, 1.170087),
        0.145073,
    ),
    "Fe": (
        (12.311098, 1.876623, 3.066177, 2.070451, 6.975185),
        (5.009415, 0.014461, 18.743040, 82.767876, 0.346506),
        -0.304931,
    ),
    "Fe3+": (
        (9.721638, 63.403847, 2.141347, 2.629274, 7.033846),
        (4.869297, 0.000293, 4.867602, 13.539076, 0.338520),
        -61.930725,
    ),
}

NEUTRON_B_C_FM = {
    "H": -3.7409,
    "H-1": -3.7395,
    "H-2": 6.6681,
    "C": 6.6472,
    "O": 5.8037,
    "Si": 4.15071,
    "Fe": 9.45,
}


def xray_non_resonant(
    keys: tuple[str, ...] | list[str], s_inverse_angstrom: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the five-Gaussian equation and analytical `df0/ds`."""

    s = np.asarray(s_inverse_angstrom, dtype=np.float64)
    values = np.empty((s.size, len(keys)), dtype=np.float64)
    derivatives = np.empty_like(values)
    for site, key in enumerate(keys):
        a, b, c = XRAY_COEFFICIENTS[key]
        a_array = np.asarray(a)
        b_array = np.asarray(b)
        gaussian = a_array[None, :] * np.exp(-(s[:, None] ** 2) * b_array[None, :])
        values[:, site] = c + np.sum(gaussian, axis=1)
        derivatives[:, site] = -2.0 * s * np.sum(b_array[None, :] * gaussian, axis=1)
    return values, derivatives


def neutron_nuclear(
    keys: tuple[str, ...] | list[str], reflection_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast constant real bound coherent lengths and zero derivatives."""

    row = np.asarray([NEUTRON_B_C_FM[key] for key in keys], dtype=np.float64)
    return np.broadcast_to(row, (reflection_count, row.size)).copy(), np.zeros(
        (reflection_count, row.size)
    )
