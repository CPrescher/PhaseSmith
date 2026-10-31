# SPDX-License-Identifier: MIT
"""Single-crystal CPU arPLS for pixel-major uniformly spaced omega sequences.

Baek et al., Analyst 140 (2015), 250-257, doi:10.1039/C4AN01061B.
Application inspired by Konstantin Glazyrin's background-sc; no upstream code
reused. No files, angular interpretation, seat-mask estimation or GUI state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .. import _core

__all__ = [
    "ArPLSParameters",
    "arpls_masked_rows",
    "arpls_rows",
    "available",
    "native_solver_available",
]

SOLVERS = ("cholesky", "ldlt")


@dataclass(frozen=True)
class ArPLSParameters:
    smoothness: float = 1e4
    tolerance: float = 1e-3
    max_iterations: int = 30

    def __post_init__(self):
        if not np.isfinite(self.smoothness) or not 1e-6 <= self.smoothness <= 1e9:
            raise ValueError("Smoothness must be between 1e-6 and 1e9.")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("Tolerance must be between 0 and 1.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or not 1 <= self.max_iterations <= 1000
        ):
            raise ValueError("Maximum iterations must be an integer from 1 to 1000.")


def available():
    """Whether the installed opt-in native build includes omega fitting."""
    return hasattr(_core, "ExperimentalArPLSEngine")


def native_solver_available(solver):
    """Whether this installed extension supports a named native factorization."""
    return available() and (
        solver == "cholesky" or solver in getattr(_core, "OMEGA_BACKGROUND_SOLVERS", ())
    )


@lru_cache(maxsize=4)
def _engine(workers, solver="cholesky"):
    # Reuse private pools across bounded tiles; idle pools do not consume CPU.
    if solver == "cholesky":
        return _core.ExperimentalArPLSEngine(workers)
    if solver not in getattr(_core, "OMEGA_BACKGROUND_SOLVERS", ()):
        raise RuntimeError("Build the matching experimental PhaseSmith engine for LDLT support")
    return _core.ExperimentalArPLSEngine(workers, solver)


def _inputs(rows, parameters, workers, backend, solver):
    p = parameters or ArPLSParameters()
    if not isinstance(p, ArPLSParameters):
        raise TypeError("parameters must be ArPLSParameters")
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] < 3 or rows.dtype.kind not in "fiu":
        raise ValueError("Expected real (pixels, frames >= 3) rows")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 128:
        raise ValueError("CPU workers must be an integer from 1 to 128")
    if backend not in ("auto", "native", "scipy"):
        raise ValueError("Backend must be auto, native or scipy")
    if not isinstance(solver, str) or solver not in SOLVERS:
        raise ValueError("Solver must be cholesky or ldlt")
    if backend == "native" and not available():
        raise RuntimeError(
            "Build PhaseSmith with --features spots (legacy experimental-spots) for omega arPLS"
        )
    return np.ascontiguousarray(rows, dtype=np.float64), p


def arpls_rows(rows, parameters=None, *, workers=1, backend="auto", solver="cholesky"):
    """Return float64 baseline, int32 solve counts and convergence flags.

    Natural unpadded second differences; float64 scalar reductions per pixel.
    Signed residuals are obtained by subtracting the baseline, without clipping.
    Inputs are not modified. Worker count is explicit and defaults to serial.
    Native solver defaults to original Cholesky arithmetic; ldlt solves the same
    equations with fewer divisions and different rounding. SciPy is independent.
    """
    data, p = _inputs(rows, parameters, workers, backend, solver)
    if not np.isfinite(data).all():
        raise ValueError("Pixel sequences must contain only finite values")
    if backend != "scipy" and available():
        return _engine(workers, solver).baselines(
            data, None, p.smoothness, p.tolerance, p.max_iterations, 3
        )
    from ..experimental.omega_background_reference import scipy_baselines

    return scipy_baselines(data, p)


def arpls_masked_rows(
    rows,
    excluded,
    parameters=None,
    *,
    workers=1,
    backend="auto",
    minimum_frames=12,
    solver="cholesky",
):
    """Fit contiguous unmasked runs of at least minimum_frames; rest is NaN.

    No interpolation or time compression. Counts are the maximum over fitted
    intervals; convergence requires at least one fitted interval and every run
    to converge. Nonfinite values are allowed only outside fitted intervals.
    """
    data, p = _inputs(rows, parameters, workers, backend, solver)
    mask = np.asarray(excluded)
    if mask.shape != data.shape or mask.dtype != np.bool_:
        raise ValueError("Expected equally shaped rows and boolean exclusions")
    if (
        isinstance(minimum_frames, bool)
        or not isinstance(minimum_frames, int)
        or minimum_frames < 3
    ):
        raise ValueError("An illuminated fit interval needs at least three frames")
    if backend != "scipy" and available():
        return _engine(workers, solver).baselines(
            data,
            np.ascontiguousarray(mask),
            p.smoothness,
            p.tolerance,
            p.max_iterations,
            minimum_frames,
        )
    from ..experimental.omega_background_reference import scipy_baselines

    result = np.full(data.shape, np.nan)
    counts = np.zeros(len(data), np.int32)
    converged = np.zeros(len(data), bool)
    for i, (row, excluded_row) in enumerate(zip(data, mask, strict=True)):
        edges = np.diff(np.pad((~excluded_row).astype(np.int8), (1, 1)))
        fitted, all_ok = False, True
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            if end - start < minimum_frames:
                continue
            part = row[None, start:end]
            if not np.isfinite(part).all():
                raise ValueError("Fitted arPLS samples must be finite")
            b, it, ok = scipy_baselines(part, p)
            result[i, start:end] = b[0]
            counts[i] = max(counts[i], it[0])
            all_ok &= bool(ok[0])
            fitted = True
        converged[i] = fitted and all_ok
    return result, counts, converged
