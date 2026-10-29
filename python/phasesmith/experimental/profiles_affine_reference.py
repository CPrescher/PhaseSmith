"""Independent affine mass equations using candidate vertices, not polygon clipping."""

import numpy as np


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersection_area(polygon, lower, upper):
    """Intersect convex quadrilateral and rectangle by enumerating all vertices."""
    p = np.asarray(polygon)
    if cross(p[1] - p[0], p[2] - p[1]) < 0:
        p = p[::-1]
    rectangle = np.array([[lower[0], lower[1]], [upper[0], lower[1]], upper, [lower[0], upper[1]]])

    def inside(v):
        return np.all(cross(np.roll(p, -1, axis=0) - p, v - p) >= -1e-12)

    points = [v for v in p if np.all(v >= lower - 1e-12) and np.all(v <= upper + 1e-12)]
    points.extend(v for v in rectangle if inside(v))
    for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
        for c, d in zip(rectangle, np.roll(rectangle, -1, axis=0), strict=True):
            matrix = np.column_stack([b - a, c - d])
            det = np.linalg.det(matrix)
            if abs(det) < 1e-14:
                continue
            t, u = np.linalg.solve(matrix, c - a)
            if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
                points.append(a + t * (b - a))
    if len(points) < 3:
        return 0.0
    vertices = np.unique(np.round(points, 13), axis=0)
    mid = vertices.mean(axis=0)
    angle = np.arctan2(vertices[:, 1] - mid[1], vertices[:, 0] - mid[0])
    vertices = vertices[np.argsort(angle)]
    return abs(np.sum(cross(vertices, np.roll(vertices, -1, axis=0)))) / 2


def evaluate(profile, bounds, step, centre, planar, scales=(1.0, 1.0, 1.0)):
    """Exact affine masses; numerical reference partials avoid native flux equations."""
    step = np.asarray(step)
    centre = np.asarray(centre)
    scales = np.asarray(scales)
    matrix = np.asarray(planar) * scales[:2, None]
    z, y, x = np.indices(profile.shape)
    indices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    active = profile.ravel() > 0
    indices = indices[active]
    values = profile.ravel()[active]
    lo = (indices - np.array(profile.shape[::-1]) / 2) * step
    hi = lo + step

    def mass(b, c):
        corners = np.array([[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]]])
        polygon = (corners - c[:2]) @ matrix.T
        omega = np.sort((b[4:] - c[2]) * scales[2])
        result = 0.0
        for lower, upper, value in zip(lo, hi, values, strict=True):
            angular = max(min(upper[2], omega[1]) - max(lower[2], omega[0]), 0) / step[2]
            if angular > 0:
                area = intersection_area(polygon, lower[:2], upper[:2])
                result += value * area / (step[0] * step[1]) * angular
        return result

    out = []
    for b in bounds:
        derivative = []
        for axis in range(3):
            delta = np.eye(3)[axis] * 1e-5
            derivative.append((mass(b, centre + delta) - mass(b, centre - delta)) / 2e-5)
        out.append([mass(b, centre), *derivative])
    return np.array(out)
