"""Controlled orthorhombic-Stephens ablations for the citrate holdouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..execution import ExecutionPolicy
from ..extensions import CompositePhysicsProvider
from ..oracle import orthorhombic_stephens_from_gsasii
from ..refinement import AffineConstraint, ConstraintTransform, FixedConstraint, rietveld
from ..refinement.runtime import RefinementLimits
from ..sample import IsotropicSizeBroadening, StephensOrthorhombicBroadening
from ._rietveld_parity import solve_linear_profile_block
from .citrate_broadening_ablation import (
    CitrateCase,
    _prepare,
    _quantitative,
    _sample_linear_correlation,
)

_SOURCE_PARAMETERS = {
    "tripotassium": {
        "coefficients": (0.46, 0.026, 0.049, 0.057, 0.012, 0.026),
        "lorentzian_fraction": 0.9000,
        "dominant_phase": "tripotassium_citrate",
    },
    "trirubidium": {
        "coefficients": (0.11, 0.012, 0.0084, 0.017, 0.020, 0.012),
        "lorentzian_fraction": 0.3652,
        "dominant_phase": "trirubidium_citrate",
    },
}


@dataclass(frozen=True, slots=True)
class CitrateStephensModelResult:
    """One fixed or refined source-shape Stephens ablation result."""

    model: str
    poisson_rwp: float
    delta_poisson_rwp: float
    profile_correlation: float
    weight_fractions: dict[str, float]
    crystallite_size_nm: float | None
    source_shape_multiplier: float
    repeat_rwp_spread: float
    free_parameter_count: int
    jacobian_rank: int | None
    maximum_sample_linear_correlation: float | None
    identifiable: bool
    qualifications: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitrateStephensAblationResult:
    """Source-shape Stephens sufficiency and identifiability evidence."""

    case: CitrateCase
    scope: str
    dominant_phase: str
    sample_count: int
    source_coefficients_gsasii: tuple[float, ...]
    source_lorentzian_fraction: float
    model_results: tuple[CitrateStephensModelResult, ...]
    conclusion: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _model(
    source: StephensOrthorhombicBroadening,
    multiplier: float,
    size_nm: float | None,
) -> object:
    stephens = replace(
        source,
        coefficients_angstrom_minus4=tuple(
            multiplier * value for value in source.coefficients_angstrom_minus4
        ),
    )
    if size_nm is None:
        return stephens
    return CompositePhysicsProvider((IsotropicSizeBroadening(size_nm), stephens))


def _stephens(provider: object) -> StephensOrthorhombicBroadening:
    if type(provider) is StephensOrthorhombicBroadening:
        return provider
    if type(provider) is CompositePhysicsProvider:
        return next(
            child for child in provider.providers if type(child) is StephensOrthorhombicBroadening
        )
    raise TypeError("expected a Stephens citrate model")


def _size(provider: object) -> float | None:
    if type(provider) is CompositePhysicsProvider:
        return next(
            child.crystallite_size_nm
            for child in provider.providers
            if type(child) is IsotropicSizeBroadening
        )
    return None


def _shape_constraints(
    phase_id: str,
    source: StephensOrthorhombicBroadening,
) -> tuple[FixedConstraint | AffineConstraint, ...]:
    def key(name: str) -> rietveld.ParameterKey:
        return rietveld.sample_parameter_key(phase_id, name)

    source_key = key("stephens.S400")
    constraints: list[FixedConstraint | AffineConstraint] = [
        FixedConstraint(key("stephens.lorentzian_fraction"), source.lorentzian_fraction)
    ]
    for name, value in zip(
        source.coefficient_names[1:], source.coefficients_angstrom_minus4[1:], strict=True
    ):
        constraints.append(
            AffineConstraint(
                key(f"stephens.{name}"),
                source_key,
                value / source.coefficients_angstrom_minus4[0],
            )
        )
    return tuple(constraints)


def run_citrate_stephens_ablation(
    bundle_directory: str | Path,
    case: CitrateCase,
    *,
    execution: ExecutionPolicy | None = None,
) -> CitrateStephensAblationResult:
    """Test deposited Stephens shape, a common scale, and size composition.

    All instrument, geometry, structure, and silicon-width terms remain fixed.
    The deposited six-coefficient ratios and mixing fraction are never changed.
    Refined variants move one common non-negative basin scale (represented by
    ``S400``), both phase scales, one residual-background constant, and,
    where stated, coherent-domain size. Three deterministic starts test basin
    repeatability.
    """

    if case not in _SOURCE_PARAMETERS:
        raise ValueError("case must be 'tripotassium' or 'trirubidium'")
    configuration = _SOURCE_PARAMETERS[case]
    manifest, pattern, experiment, base_phases, base_background = _prepare(
        Path(bundle_directory), case
    )
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    dominant_phase = str(configuration["dominant_phase"])
    source = orthorhombic_stephens_from_gsasii(
        configuration["coefficients"], configuration["lorentzian_fraction"]
    )
    selection = rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=True,
    )
    specifications = {
        "source_fixed": ((1.0, None),),
        "source_shape_scale": ((0.25, None), (1.0, None), (4.0, None)),
        "size_source_fixed": ((1.0, 20.0), (1.0, 100.0), (1.0, 500.0)),
        "size_source_shape_scale": ((0.25, 20.0), (1.0, 100.0), (4.0, 500.0)),
    }
    raw: dict[str, list[tuple[float, Any, Any, int, int | None, float | None]]] = {}
    for model_name, starts in specifications.items():
        repeats = []
        for multiplier, size_nm in starts:
            phases = tuple(
                replace(phase, physics=_model(source, multiplier, size_nm))
                if phase.phase_id == dominant_phase
                else phase
                for phase in base_phases
            )
            phases, background = solve_linear_profile_block(
                pattern, experiment, phases, base_background, selected_execution
            )
            if model_name == "source_fixed":
                calculation = rietveld.calculate(
                    pattern,
                    experiment,
                    phases,
                    background=background,
                    support_fwhm=30.0,
                    execution=selected_execution,
                )
                weight = 1.0 / np.maximum(pattern.observed_y, 1.0)
                residual = calculation.y - pattern.observed_y
                rwp = float(
                    np.sqrt(np.sum(weight * residual**2) / np.sum(weight * pattern.observed_y**2))
                )
                repeats.append((rwp, calculation, phases, 3, None, None))
                continue
            parameters = rietveld.build_parameter_set(
                phases,
                (None,) * len(phases),
                selection,
                experiment=experiment,
                background=background,
            )
            constraints = _shape_constraints(dominant_phase, source)
            if model_name == "size_source_fixed":
                coefficient_constraints = tuple(
                    FixedConstraint(
                        rietveld.sample_parameter_key(dominant_phase, f"stephens.{name}"),
                        value,
                    )
                    for name, value in zip(
                        source.coefficient_names,
                        source.coefficients_angstrom_minus4,
                        strict=True,
                    )
                )
                constraints = (
                    *coefficient_constraints,
                    FixedConstraint(
                        rietveld.sample_parameter_key(
                            dominant_phase, "stephens.lorentzian_fraction"
                        ),
                        source.lorentzian_fraction,
                    ),
                )
            free_count = len(ConstraintTransform(parameters, constraints).free_keys)
            result = rietveld.refine(
                rietveld.RietveldInput(
                    pattern,
                    experiment,
                    phases,
                    (None,) * len(phases),
                    parameters,
                    constraints,
                    selection,
                    background,
                ),
                rietveld.RietveldOptions(
                    limits=RefinementLimits(max_iterations=25, max_evaluations=2_000),
                    min_iterations=2,
                    max_scaled_parameter_step=0.30,
                    support_fwhm=30.0,
                    estimate_covariance=True,
                    unresolved_correlation=0.0,
                    execution=selected_execution,
                ),
            )
            repeats.append(
                (
                    float(result.metrics.rwp),
                    result.calculation,
                    result.phases,
                    free_count,
                    result.jacobian_rank,
                    _sample_linear_correlation(result),
                )
            )
        raw[model_name] = repeats

    base_model = next(item for item in raw["source_fixed"])
    base_rwp = base_model[0]
    model_results = []
    for model_name, repeats in raw.items():
        rwps = [item[0] for item in repeats]
        best = repeats[int(np.argmin(rwps))]
        rwp, calculation, phases, free_count, rank, correlation = best
        provider = next(phase.physics for phase in phases if phase.phase_id == dominant_phase)
        fitted = _stephens(provider)
        multiplier = fitted.coefficients_angstrom_minus4[0] / source.coefficients_angstrom_minus4[0]
        size_nm = _size(provider)
        spread = max(rwps) - min(rwps)
        boundary = multiplier <= 0.0 or (size_nm is not None and size_nm >= 1.0e5)
        identifiable = model_name == "source_fixed" or (
            rank == free_count
            and spread <= 5.0e-5
            and not boundary
            and correlation is not None
            and correlation < 0.98
        )
        qualifications = []
        if rank is not None and rank != free_count:
            qualifications.append("weighted Jacobian is rank deficient")
        if spread > 5.0e-5:
            qualifications.append("three deterministic starts do not reach the same Rwp basin")
        if boundary:
            qualifications.append("a refined width term is inactive at its boundary")
        if correlation is not None and correlation >= 0.98:
            qualifications.append(
                "a sample-width column is at least 0.98 correlated with scale/background"
            )
        if not qualifications:
            qualifications.append("passes rank, repeatability, boundary, and correlation gates")
        model_results.append(
            CitrateStephensModelResult(
                model_name,
                rwp,
                rwp - base_rwp,
                float(
                    np.corrcoef(pattern.observed_y - calculation.background, calculation.profile_y)[
                        0, 1
                    ]
                ),
                _quantitative(phases),
                None if size_nm is None else float(size_nm),
                float(multiplier),
                float(spread),
                free_count,
                rank,
                correlation,
                identifiable,
                tuple(qualifications),
            )
        )
    accepted = [item for item in model_results if item.identifiable]
    best = min(accepted, key=lambda item: item.poisson_rwp)
    target = float(manifest["legacy_gsas_reference"]["rwp"])
    conclusion = (
        f"{best.model} is the best identifiable deposited-shape model at Poisson Rwp "
        f"{best.poisson_rwp:.6f}; the remaining {best.poisson_rwp - target:.6f} gap to "
        "the deposited curve shows whether Stephens is sufficient under fixed nuisance terms."
    )
    return CitrateStephensAblationResult(
        case,
        str(manifest["scope"]),
        dominant_phase,
        int(pattern.x.size),
        tuple(configuration["coefficients"]),
        float(configuration["lorentzian_fraction"]),
        tuple(model_results),
        conclusion,
    )
