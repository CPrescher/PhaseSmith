"""Independent NumPy Ewald roots and ray/plane projection; no Rust calls."""

import numpy as np


def predict(hkl, basis, geometry, hints, *, other=False):
    beam = geometry.vectors[3] / np.linalg.norm(geometry.vectors[3])
    axis = geometry.vectors[4] / np.linalg.norm(geometry.vectors[4])
    origin, fast, slow = geometry.vectors[:3]
    normal = np.cross(fast, slow)
    result = np.full((len(hkl), 3), np.nan)
    for i, (h, hint) in enumerate(zip(hkl, hints, strict=True)):
        g = h @ basis
        parallel = axis * np.dot(axis, g)
        # Independent sine-phase formulation of the two Ewald solutions.
        cosine = np.dot(beam, g - parallel)
        sine = np.dot(beam, np.cross(axis, g))
        radius = np.hypot(cosine, sine)
        target = -geometry.wavelength * np.dot(g, g) / 2 - np.dot(beam, parallel)
        if radius <= 1e-14 or abs(target) >= radius:
            continue
        phase = np.arctan2(cosine, sine)
        alpha = np.arcsin(target / radius)
        roots = np.rad2deg(np.array([np.pi - alpha - phase, alpha - phase]))
        roots += 360 * np.floor((hint - roots) / 360 + 0.5)
        order = np.argsort(abs(roots - hint), kind="stable")
        theta = np.deg2rad(roots[order[int(other)]])
        # Matrix exponential's closed form for the skew-axis matrix.
        x, y, z = axis
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)
        ray = beam + geometry.wavelength * (rotation @ g)
        denominator = np.dot(normal, ray)
        if abs(denominator) / np.linalg.norm(normal) < 1e-12:
            continue
        distance = np.dot(normal, origin) / denominator
        if distance <= 0:
            continue
        xy = np.linalg.lstsq(np.column_stack([fast, slow]), distance * ray - origin, rcond=None)[0]
        result[i] = [*xy, np.rad2deg(theta)]
    return result
