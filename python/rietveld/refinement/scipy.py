"""Optional adapter to SciPy's maintained bounded linear least-squares solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ScipyLeastSquaresAdapter:
    """Use :func:`scipy.optimize.lsq_linear` for one linearized step.

    SciPy is imported only when :meth:`solve` is called. Normal installation,
    calculation, and NumPy-only Le Bail refinement do not require it.
    """

    tolerance: float = 1.0e-12
    max_iterations: int = 200

    def __post_init__(self) -> None:
        """Validate solver controls without importing SciPy."""

        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def solve(
        self,
        residual: NDArray[np.float64],
        jacobian: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Minimize ``||residual + jacobian @ step||`` within step bounds."""

        try:
            from scipy.optimize import lsq_linear
        except ImportError as error:
            raise ImportError(
                "ScipyLeastSquaresAdapter requires the 'refinement' optional dependency"
            ) from error
        result = lsq_linear(
            jacobian,
            -residual,
            bounds=(lower, upper),
            tol=self.tolerance,
            max_iter=self.max_iterations,
        )
        if not result.success:
            raise RuntimeError(f"SciPy bounded least squares failed: {result.message}")
        return np.asarray(result.x, dtype=np.float64)
