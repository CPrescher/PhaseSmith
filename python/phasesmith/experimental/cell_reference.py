"""Independent, small NumPy reference for Selling moves and metric derivatives.

No extension imports. Used for numerical validation, not by the production API.
"""

import numpy as np


def selling_reference(basis, tolerance=1e-12, max_iterations=10000):
    a = np.asarray(basis, dtype=float)
    c = np.vstack((np.eye(3, dtype=np.int64), -np.ones(3, dtype=np.int64)))
    for iteration in range(max_iterations + 1):
        v = c @ a
        gram = v @ v.T
        candidates = [(gram[i, j], i, j) for i in range(4) for j in range(i + 1, 4)]
        largest = max(x[0] for x in candidates)
        if largest <= tolerance * np.diag(gram).max():
            order = sorted(range(4), key=lambda i: (gram[i, i], tuple(c[i])))[:3]
            t = c[order].copy()
            if np.linalg.det(t @ a) < 0:
                t[[0, 1]] = t[[1, 0]]
            return t @ a, t, iteration
        if iteration == max_iterations:
            raise ValueError("Selling iteration budget exceeded")
        _, i, j = next(x for x in candidates if x[0] == largest)
        old = c[i].copy()
        for k in set(range(4)) - {i, j}:
            c[k] += old
        c[i] = -old
    raise AssertionError("unreachable")


def metric_basis_reference(parameters, directions, rotation):
    """Construct B from linear direct Gram coordinates and fixed Cartesian Q."""
    g = np.einsum("k,kij->ij", parameters, directions)
    return np.linalg.solve(np.linalg.cholesky(g).T, rotation)
