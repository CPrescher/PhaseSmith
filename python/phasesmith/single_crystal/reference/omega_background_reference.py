# SPDX-License-Identifier: MIT
"""Independent SciPy equation reference for experimental omega arPLS.

SciPy is required only for this reference, never for the Rust engine.
The same equation reference validates both native Cholesky and LDLT solvers;
it constructs the penalty independently and shares no factorization code.
"""

import numpy as np


def scipy_baselines(data, p):
    from scipy.linalg import solveh_banded
    from scipy.special import expit

    # Construct D.T @ D by accumulating each [1, -2, 1] row's outer
    # product directly into LAPACK's lower-band representation.
    n = data.shape[1]
    penalty = np.zeros((3, n))
    for j in range(n - 2):
        penalty[0, j : j + 3] += p.smoothness * np.array([1.0, 4.0, 1.0])
        penalty[1, j : j + 2] -= 2 * p.smoothness
        penalty[2, j] += p.smoothness
    result = np.empty_like(data)
    iterations = np.zeros(len(data), dtype=np.int32)
    converged = np.zeros(len(data), dtype=bool)
    for row_index, original in enumerate(data):
        offset = original[0]
        scale = np.max(np.abs(original - offset))
        if scale == 0:
            result[row_index] = original
            converged[row_index] = True
            continue
        y = (original - offset) / scale
        weights = np.ones(n)
        for iteration in range(p.max_iterations):
            band = penalty.copy()
            band[0] += weights
            z = solveh_banded(band, weights * y, lower=True, check_finite=False)
            iterations[row_index] = iteration + 1
            residual = y - z
            negative = residual[residual < 0]
            if negative.size < 2 or np.max(np.abs(residual)) < 1e-10:
                converged[row_index] = True
                break
            # Zero negative-residual variance is common for noiseless sparse
            # peaks. Use the limiting sharp transition, rather than returning
            # the initial symmetric smoother (which would absorb the peak).
            sigma = max(float(np.std(negative)), 1e-8)
            argument = 2 * (residual - (2 * sigma - negative.mean())) / sigma
            new_weights = np.maximum(expit(-np.clip(argument, -40, 40)), 1e-8)
            if np.linalg.norm(new_weights - weights) / np.linalg.norm(weights) < p.tolerance:
                converged[row_index] = True
                break
            weights = new_weights
        result[row_index] = z * scale + offset
    return result, iterations, converged
