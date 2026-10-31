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


def orientation_reference(basis):
    """Independent canonical-cell orientation from direct-cell Cholesky rows."""
    direct = np.linalg.inv(basis).T
    if np.linalg.det(direct) <= 0:
        raise ValueError("reference orientation requires a right-handed basis")
    return np.linalg.solve(np.linalg.cholesky(direct @ direct.T), direct).T


def rotation_reference(angles):
    """Rz @ Ry @ Rx and analytical partials in radians."""
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
    dx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]], float)
    dy = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]], float)
    dz = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]], float)
    return rz @ ry @ rx, np.array([rz @ ry @ dx, rz @ dy @ rx, dz @ ry @ rx])


def reference_basis_reference(basis, cell):
    """Replace the metric, preserving initial canonical crystal orientation."""
    p = np.asarray(cell)
    if p.shape != (6,) or p.dtype.kind not in "fiu" or not np.isfinite(p).all():
        raise ValueError("reference_cell must contain six finite cell parameters")
    if (p[:3] <= 0).any() or (p[3:] <= 0).any() or (p[3:] >= 180).any():
        raise ValueError("invalid reference cell lengths/angles")
    alpha, beta, gamma = np.deg2rad(p[3:])
    gram = np.diag(p[:3] ** 2).astype(float)
    for i, j, c in [(0, 1, np.cos(gamma)), (0, 2, np.cos(beta)), (1, 2, np.cos(alpha))]:
        gram[i, j] = gram[j, i] = p[i] * p[j] * c
    try:
        direct = np.linalg.cholesky(gram) @ orientation_reference(basis).T
    except np.linalg.LinAlgError as error:
        raise ValueError("reference cell has nonpositive volume") from error
    return np.linalg.inv(direct).T


def parameter_state_reference(parameters, basis, geometry, pivot, refine_lengths):
    """Physical state and exact chains wrt Euler angles, mm shifts and log lengths."""
    p = np.asarray(parameters, float)
    n = 12 if refine_lengths else 9
    if p.shape != (n,) or not np.isfinite(p).all():
        raise ValueError("invalid geometry parameter vector")
    rc, dc = rotation_reference(p[:3])
    rd, dd = rotation_reference(p[6:9])
    b0 = basis.copy()
    if refine_lengths:
        b0 *= np.exp(p[9:12, None])
    b = b0 @ rc.T
    v = geometry.vectors.copy()
    v[0] = pivot + 0.001 * p[3:6] + rd @ (geometry.vectors[0] - pivot)
    v[1:3] = geometry.vectors[1:3] @ rd.T
    jb = np.zeros((9, n))
    jg = np.zeros((9, n))
    for k in range(3):
        jb[:, k] = (b0 @ dc[k].T).ravel()
        jg[:3, 3 + k] = 0.001 * np.eye(3)[k]
        jg[:, 6 + k] = np.concatenate(
            [
                dd[k] @ (geometry.vectors[0] - pivot),
                dd[k] @ geometry.vectors[1],
                dd[k] @ geometry.vectors[2],
            ]
        )
        if refine_lengths:
            jb[3 * k : 3 * k + 3, 9 + k] = b[k]
    return (
        b,
        SimpleNamespace(vectors=v, wavelength=geometry.wavelength, pixel_map=geometry.pixel_map),
        jb,
        jg,
    )


def prediction_partials_reference(hkl, basis, geometry, hints):
    """Independent implicit 3x3 solves for both reciprocal and detector partials."""
    plane = SimpleNamespace(
        vectors=geometry.vectors, wavelength=geometry.wavelength, pixel_map=None
    )
    values, detector = detector_partials(hkl, basis, plane, hints)
    jacobian = np.full((len(hkl), 3, 9), np.nan)
    axis = geometry.vectors[4] / np.linalg.norm(geometry.vectors[4])
    beam = geometry.vectors[3] / np.linalg.norm(geometry.vectors[3])
    origin, fast, slow = geometry.vectors[:3]
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    for i, (h, value) in enumerate(zip(hkl, values, strict=True)):
        if not np.isfinite(value).all():
            continue
        theta = np.deg2rad(value[2])
        rotate = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
        reciprocal = h @ basis
        lab = rotate @ reciprocal
        angular = np.cross(axis, lab)
        slope = beam @ angular
        dtheta = -(rotate.T @ beam + geometry.wavelength * reciprocal) / slope
        ray = beam + geometry.wavelength * lab
        inverse = np.linalg.inv(np.column_stack([fast, slow, -ray]))
        distance = np.linalg.norm(origin + value[0] * fast + value[1] * slow) / np.linalg.norm(ray)
        dray = geometry.wavelength * (rotate + np.outer(angular, dtheta))
        dxydg = inverse[:2] @ (distance * dray)
        jacobian[i, :2] = np.einsum("aj,k->akj", dxydg, h).reshape(2, 9)
        jacobian[i, 2] = np.outer(h, np.rad2deg(dtheta)).ravel()
    if getattr(geometry, "pixel_map", None) is not None:
        good = np.isfinite(values).all(axis=1)
        xy = values[good, :2].copy()
        chain = geometry.pixel_map.inverse_jacobian(xy)
        jacobian[good, :2] = np.einsum("nij,njk->nik", chain, jacobian[good, :2])
        detector[good, :2] = np.einsum("nij,njk->nik", chain, detector[good, :2])
        values[good, :2] = geometry.pixel_map.from_plane(xy)
    return values, jacobian, detector


def refine_reference(
    observed,
    hkl,
    basis,
    geometry,
    image_shape,
    *,
    training,
    eligible=None,
    reference_cell=None,
    refine_lengths=False,
    options=None,
):
    """Independent NumPy forward model/partials with a SciPy control optimizer.

    No production geometry helpers or native predictors are called. Inputs are
    expected to have passed the public array/partition boundary validation.
    """
    from scipy.optimize import least_squares

    if options is None:
        options = SimpleNamespace(
            xy_scale=0.4,
            omega_scale=0.2,
            huber=2.5,
            max_evaluations=150,
            rotation_limit_deg=2.0,
            translation_limit_mm=2.0,
            log_length_limit=0.02,
            initial_xy_gate=5.0,
            initial_omega_gate=1.0,
        )
    initial = predict(hkl, basis, geometry, observed[:, 2])
    use = np.ones(len(hkl), bool) if eligible is None else np.array(eligible, copy=True)
    residual = initial - observed
    use &= (
        np.isfinite(initial).all(axis=1)
        & (np.linalg.norm(residual[:, :2], axis=1) < options.initial_xy_gate)
        & (abs(residual[:, 2]) < options.initial_omega_gate)
    )
    selected = use & training
    base = basis if reference_cell is None else reference_basis_reference(basis, reference_cell)
    pivot = (
        geometry.vectors[0]
        + (image_shape[1] - 1) / 2 * geometry.vectors[1]
        + (image_shape[0] - 1) / 2 * geometry.vectors[2]
    )
    n = 12 if refine_lengths else 9
    limits = np.array(
        [np.deg2rad(options.rotation_limit_deg)] * 3
        + [options.translation_limit_mm] * 3
        + [np.deg2rad(options.rotation_limit_deg)] * 3
        + [options.log_length_limit] * 3
    )[:n]
    units = np.array([0.001] * 3 + [0.1] * 3 + [0.001] * 6)[:n]
    scale = np.array([options.xy_scale, options.xy_scale, options.omega_scale])
    cache = {}

    def evaluate(parameters):
        if "p" not in cache or not np.array_equal(parameters, cache["p"]):
            b, g, jb, jg = parameter_state_reference(
                parameters, base, geometry, pivot, refine_lengths
            )
            v, j, d = prediction_partials_reference(hkl[selected], b, g, observed[selected, 2])
            cache.update(
                p=parameters.copy(),
                r=((v - observed[selected]) / scale).ravel(),
                j=((j @ jb + d @ jg) / scale[None, :, None]).reshape(-1, n),
            )
        return cache

    result = least_squares(
        lambda p: evaluate(p)["r"],
        np.zeros(n),
        jac=lambda p: evaluate(p)["j"],
        bounds=(-limits, limits),
        x_scale=units,
        loss="soft_l1",
        f_scale=options.huber,
        max_nfev=options.max_evaluations,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-9,
    )
    b, g, _, _ = parameter_state_reference(result.x, base, geometry, pivot, refine_lengths)
    return SimpleNamespace(
        parameters=result.x,
        basis=b,
        geometry=g,
        fit=result,
        eligible=use,
        predictions=predict(hkl, b, g, observed[:, 2]),
    )
