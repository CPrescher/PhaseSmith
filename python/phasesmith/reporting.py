"""Plain JSON and CSV reports for structural Rietveld results."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .pattern import PowderPattern
from .refinement.rietveld import RietveldResult


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _package_version() -> str:
    try:
        return version("phasesmith")
    except PackageNotFoundError:  # pragma: no cover - source-tree fallback
        return "0+local"


def rietveld_result_record(result: RietveldResult) -> dict[str, Any]:
    """Build a finite JSON-compatible summary without embedding sample arrays."""

    if not isinstance(result, RietveldResult):
        raise TypeError("result must be RietveldResult")
    metrics = result.metrics
    return {
        "schema": "phasesmith.result-report.v1",
        "provenance": {
            "package": "phasesmith",
            "version": _package_version(),
            "numerical_core": "rust",
        },
        "termination": {
            "reason": result.termination_reason.value,
            "message": result.termination_message,
            "evaluations": result.evaluations,
            "accepted_iterations": len(result.history),
        },
        "metrics": {
            "rp": _finite(metrics.rp),
            "rwp": _finite(metrics.rwp),
            "chi_square": _finite(metrics.chi_square),
            "reduced_chi_square": _finite(metrics.reduced_chi_square),
        },
        "parameters": [
            {
                "label": spec.key.label,
                "module": spec.key.module,
                "owner_id": spec.key.owner_id,
                "name": spec.key.name,
                "value": spec.value,
                "unit": spec.unit,
                "lower": _finite(spec.bounds.lower),
                "upper": _finite(spec.bounds.upper),
                "scale": spec.scale,
            }
            for spec in result.parameters.specs
        ],
        "diagnostics": {
            "jacobian_rank": result.jacobian_rank,
            "logger_error": None if result.logger_error is None else str(result.logger_error),
            "unresolved_correlations": [
                {
                    "left": item.left.label,
                    "right": item.right.label,
                    "correlation": item.correlation,
                }
                for item in result.unresolved_correlations
            ],
        },
        "phases": [
            {
                "phase_id": phase.phase_id,
                "name": phase.name,
                "scale": phase.scale,
                "reflection_count": phase.reflections.reflection_count,
                "component_reflection_count": len(calculation.reflections.reflection_ids),
                "reflections": [
                    {
                        "reflection_id": reflection_id,
                        "hkl": list(map(int, hkl)),
                        "component_index": int(component_index),
                        "d_spacing_angstrom": float(d_spacing),
                        "two_theta_deg": float(position),
                        "integrated_intensity": float(intensity),
                    }
                    for reflection_id, hkl, component_index, d_spacing, position, intensity in zip(
                        calculation.reflections.reflection_ids,
                        phase.reflections.hkl[calculation.reflections.base_reflection_index],
                        calculation.reflections.component_index,
                        calculation.reflections.d_spacing_angstrom,
                        calculation.reflections.two_theta_deg,
                        calculation.reflections.integrated_intensity,
                        strict=True,
                    )
                ],
            }
            for phase, calculation in zip(
                result.phases, result.calculation.phase_calculations, strict=True
            )
        ],
        "history": [
            {
                "iteration": item.iteration,
                "rp": _finite(item.rp),
                "rwp": _finite(item.rwp),
                "chi_square": _finite(item.chi_square),
                "reduced_chi_square": _finite(item.reduced_chi_square),
                "objective": item.objective,
                "objective_change": item.objective_change,
                "scaled_step_norm": item.scaled_step_norm,
                "damping": item.damping,
                "cg_iterations": item.cg_iterations,
                "backtracks": item.backtracks,
                "topology_changes": list(item.topology_changes),
            }
            for item in result.history
        ],
    }


def write_rietveld_json(result: RietveldResult, path: str | Path) -> Path:
    """Write an auditable finite JSON result report."""

    destination = Path(path)
    destination.write_text(
        json.dumps(
            rietveld_result_record(result), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_rietveld_csv(
    result: RietveldResult,
    pattern: PowderPattern,
    path: str | Path,
) -> Path:
    """Write one plain sample row per measured coordinate."""

    if not isinstance(result, RietveldResult):
        raise TypeError("result must be RietveldResult")
    if not isinstance(pattern, PowderPattern) or pattern.observed_y is None:
        raise ValueError("pattern must be an observed PowderPattern")
    if pattern.x.shape != result.calculation.y.shape:
        raise ValueError("pattern and result sample counts must match")
    included = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    uncertainty = (
        np.full(pattern.x.size, np.nan)
        if pattern.uncertainty is None
        else pattern.uncertainty
    )
    weight = included.astype(np.float64)
    if pattern.uncertainty is not None:
        weight /= np.square(pattern.uncertainty)
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "x_deg",
                "observed_y",
                "calculated_y",
                "profile_y",
                "background_y",
                "residual_calculated_minus_observed",
                "uncertainty",
                "weight",
                "included",
            )
        )
        writer.writerows(
            zip(
                pattern.x,
                pattern.observed_y,
                result.calculation.y,
                result.calculation.profile_y,
                result.calculation.background,
                result.metrics.residual,
                uncertainty,
                weight,
                included.astype(np.int8),
                strict=True,
            )
        )
    return destination
