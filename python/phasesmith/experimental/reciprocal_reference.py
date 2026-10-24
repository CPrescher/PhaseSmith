"""Independent NumPy equations; does not import or invoke the Rust extension."""

import numpy as np


def reciprocal_vectors(xyomega, geometry):
    v = geometry.vectors
    points = v[0] + xyomega[:, :1] * v[1] + xyomega[:, 1:2] * v[2]
    rays = points / np.linalg.norm(points, axis=1)[:, None]
    q = (rays - v[3] / np.linalg.norm(v[3])) / geometry.wavelength
    axis = v[4] / np.linalg.norm(v[4])
    # Quaternion-derived rotation matrix, independent of native Rodrigues pass.
    result = []
    for vector, omega in zip(q, xyomega[:, 2], strict=True):
        a = np.cos(np.deg2rad(omega) / 2)
        b, c, d = -axis * np.sin(np.deg2rad(omega) / 2)
        rotation = np.array(
            [
                [1 - 2 * (c * c + d * d), 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), 1 - 2 * (b * b + d * d), 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b * b + c * c)],
            ]
        )
        result.append(rotation @ vector)
    return np.asarray(result).reshape(-1, 3)


def lattice_fourier(points, vectors):
    phase = 2 * np.pi * (vectors @ points.T)
    score = np.cos(phase).mean(axis=1)
    grad = -2 * np.pi * np.sin(phase) @ points / len(points)
    hessian = (
        -((2 * np.pi) ** 2)
        * np.einsum("ki,ia,ib->kab", np.cos(phase), points, points)
        / len(points)
    )
    return np.column_stack([score, grad, hessian.reshape(-1, 9)])
