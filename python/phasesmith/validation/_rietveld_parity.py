"""Shared exact-linear/staged-nonlinear helpers for matched oracle workflows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..execution import ExecutionPolicy
from ..pattern import PowderPattern
from ..radiation import ConstantWavelengthExperiment
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..refinement.runtime import RefinementLimits


def solve_linear_profile_block(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    background: ChebyshevBackground,
    execution: ExecutionPolicy,
    *,
    support_fwhm: float = 30.0,
) -> tuple[tuple[rietveld.RietveldPhase, ...], ChebyshevBackground]:
    """Exactly solve non-negative phase scales and an unconstrained background block."""

    if pattern.observed_y is None or pattern.uncertainty is None:
        raise ValueError("linear profile block requires observed values and uncertainties")
    if not phases:
        raise ValueError("linear profile block requires at least one phase")
    unit_profiles = np.column_stack(
        [
            rietveld.calculate(
                pattern,
                experiment,
                (replace(phase, scale=1.0),),
                support_fwhm=support_fwhm,
                execution=execution,
            ).profile_y
            for phase in phases
        ]
    )
    background_basis = background.basis(pattern.x)
    target = pattern.observed_y - pattern.background
    inverse_uncertainty = 1.0 / pattern.uncertainty
    best_objective = np.inf
    best_solution: NDArray[np.float64] | None = None
    phase_count = len(phases)
    # These validation cases contain three phases. Enumerating active scale
    # sets gives the exact small NNLS result without adding a SciPy dependency.
    for mask in range(1 << phase_count):
        active = [index for index in range(phase_count) if mask & (1 << index)]
        design = np.column_stack((unit_profiles[:, active], background_basis))
        weighted_design = design * inverse_uncertainty[:, None]
        solution = np.linalg.lstsq(
            weighted_design,
            target * inverse_uncertainty,
            rcond=None,
        )[0]
        if np.any(solution[: len(active)] < 0.0):
            continue
        residual = weighted_design @ solution - target * inverse_uncertainty
        objective = float(residual @ residual)
        if objective < best_objective:
            complete = np.zeros(phase_count + background_basis.shape[1], dtype=np.float64)
            complete[active] = solution[: len(active)]
            complete[phase_count:] = solution[len(active) :]
            best_solution = complete
            best_objective = objective
    if best_solution is None:
        raise ValueError("linear scale/background block has no feasible solution")
    updated_phases = tuple(
        replace(phase, scale=float(scale))
        for phase, scale in zip(phases, best_solution[:phase_count], strict=True)
    )
    return updated_phases, background.replace_coefficients(best_solution[phase_count:])


def refine_nonlinear_block(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    background: ChebyshevBackground,
    selection: rietveld.RietveldParameterSelection,
    execution: ExecutionPolicy,
    *,
    iterations: int,
    step: float,
    support_fwhm: float = 30.0,
) -> rietveld.RietveldResult:
    """Refine one nonlinear block while holding the exact linear block fixed."""

    domains = (None,) * len(phases)
    input_data = rietveld.RietveldInput(
        pattern,
        experiment,
        phases,
        domains,
        rietveld.build_parameter_set(
            phases,
            domains,
            selection,
            experiment=experiment,
            background=background,
        ),
        selection=selection,
        background=background,
    )
    return rietveld.refine(
        input_data,
        rietveld.RietveldOptions(
            limits=RefinementLimits(max_iterations=iterations, max_evaluations=2_000),
            min_iterations=2,
            max_scaled_parameter_step=step,
            support_fwhm=support_fwhm,
            estimate_covariance=False,
            execution=execution,
        ),
    )
