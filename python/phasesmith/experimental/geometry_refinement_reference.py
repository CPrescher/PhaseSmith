"""Independent NumPy detector-plane partials from a 3x3 implicit solve."""

from types import SimpleNamespace

import numpy as np

from .prediction_reference import predict


def detector_partials(hkl, basis, geometry, hints):
    plane = SimpleNamespace(
        vectors=geometry.vectors, wavelength=geometry.wavelength, pixel_map=None
    )
    values = predict(hkl, basis, plane, hints)
    jac = np.full((len(hkl), 3, 9), np.nan)
    origin, fast, slow = geometry.vectors[:3]
    for i, (x, y, omega) in enumerate(values):
        if not np.isfinite(omega):
            continue
        ray = origin + x * fast + y * slow
        inverse = np.linalg.inv(np.column_stack([fast, slow, -ray]))
        jac[i] = 0
        jac[i, :2, :3] = -inverse[:2]
        jac[i, :2, 3:6] = -x * inverse[:2]
        jac[i, :2, 6:9] = -y * inverse[:2]
    if getattr(geometry, "pixel_map", None) is not None:
        good = np.isfinite(values).all(axis=1)
        xy = values[good, :2].copy()
        chain = geometry.pixel_map.inverse_jacobian(xy)
        jac[good, :2] = np.einsum("nij,njk->nik", chain, jac[good, :2])
        values[good, :2] = geometry.pixel_map.from_plane(xy)
    return values, jac
