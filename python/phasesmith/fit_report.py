"""Read-only fit evidence and conservative next steps, without model attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from . import _core
from .pattern import PowderPattern
from .refinement.rietveld import RietveldResult


@dataclass(frozen=True, slots=True)
class ResidualRegion:
    """An equal-coordinate interval; the last interval includes its upper edge."""

    lower: float
    upper: float
    count: int
    chi_square: float
    chi_square_fraction: float
    maximum_absolute_weighted_residual: float


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Native mask-aware evidence; weighted residuals need not be sigma units."""

    included_count: int
    adjacent_pair_count: int
    chi_square: float
    mean: float
    weighted_rms: float
    durbin_watson: float | None
    regions: tuple[ResidualRegion, ...]

    def worst_regions(self, count: int = 3) -> tuple[ResidualRegion, ...]:
        """Rank nonempty intervals by chi-square, preserving coordinate order for ties."""
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return tuple(
            sorted((r for r in self.regions if r.count), key=lambda r: -r.chi_square)[:count]
        )


def diagnose_residuals(
    x: ArrayLike,
    residual: ArrayLike,
    *,
    weighted_residual: ArrayLike | None = None,
    included: ArrayLike | None = None,
    region_count: int = 20,
) -> ResidualDiagnostics:
    """Localize calculated-minus-observed residuals on a CW, TOF, or other grid.

    Pass the fit's actual weighted residuals to preserve its weighting policy.
    If omitted, weights are one. Excluded samples never form artificial
    adjacent pairs. No threshold here declares a physical cause or good fit.
    """
    if type(region_count) is not int or not 1 <= region_count <= 4096:
        raise ValueError("region_count must be an integer between 1 and 4096")

    def vector(value: ArrayLike, name: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a real numeric vector")
        return np.ascontiguousarray(raw, dtype=np.float64)

    coordinates = vector(x, "x")
    residuals = vector(residual, "residual")
    weighted = (
        residuals if weighted_residual is None else vector(weighted_residual, "weighted_residual")
    )
    mask = np.ones(coordinates.size, dtype=np.bool_) if included is None else np.asarray(included)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("included must be a boolean vector")
    values = _core.residual_diagnostics(
        coordinates, residuals, weighted, np.ascontiguousarray(mask), region_count
    )
    return ResidualDiagnostics(*values[:-1], tuple(ResidualRegion(*r) for r in values[-1]))


@dataclass(frozen=True, slots=True)
class FitAdvice:
    """A review action, never an automatically authorized parameter change."""

    code: str
    message: str
    parameter_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FitReport:
    """Versioned evidence and advice for one accepted CW Rietveld result."""

    residuals: ResidualDiagnostics
    termination_reason: str
    jacobian_rank: int | None
    covariance_available: bool
    advice: tuple[FitAdvice, ...]
    attribution_available: bool = False
    attribution_unavailable_reason: str = (
        "Residual location alone does not distinguish background, profile, structure, "
        "missing phases, or experimental errors. No cause or parameter gain is inferred."
    )

    def to_record(self) -> dict[str, object]:
        """Return finite JSON-compatible evidence with a separately versioned schema."""
        return {"schema": "phasesmith.fit-report.v1", **asdict(self)}


def build_fit_report(
    result: RietveldResult, pattern: PowderPattern, *, region_count: int = 20
) -> FitReport:
    """Review a result using its original observed pattern and actual fit weights.

    The function performs no calculation, refinement, or state mutation. The
    caller must supply the original coordinate grid; residuals and masks are
    checked against the result to reject mismatched observations.
    """
    if not isinstance(result, RietveldResult) or not isinstance(pattern, PowderPattern):
        raise TypeError("expected RietveldResult and its observed PowderPattern")
    if pattern.observed_y is None or pattern.x.shape != result.calculation.y.shape:
        raise ValueError("the observed pattern must match the result sample count")
    mask = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    if not np.array_equal(result.metrics.residual, result.calculation.y - pattern.observed_y):
        raise ValueError("pattern observations do not match result residuals")
    if not np.array_equal(mask, result.metrics.included):
        raise ValueError("pattern mask does not match result residuals")
    residuals = diagnose_residuals(
        pattern.x,
        result.metrics.residual,
        weighted_residual=result.metrics.weighted_residual,
        included=mask,
        region_count=region_count,
    )
    reason = result.termination_reason.value
    advice = []
    if result.checkpoint.empirical_gaussian is not None:
        convention = result.checkpoint.empirical_gaussian
        advice.append(
            FitAdvice(
                "empirical_gaussian_convention",
                f"RMS strain of {convention.reference_phase_id} is fixed by convention to "
                f"{convention.reference_rms_microstrain:g}. Instrument Gaussian widths, sample "
                "strains and their conditional uncertainties depend on this assumption; "
                "they are not an independent instrument/sample measurement.",
            )
        )
    if reason != "converged":
        advice.append(
            FitAdvice(
                "review_termination",
                f"Refinement stopped with {reason}: {result.termination_message}",
            )
        )
    if result.jacobian_rank is None:
        advice.append(
            FitAdvice(
                "identifiability_unassessed",
                "Jacobian rank was not assessed; it is not known to be full.",
            )
        )
    if result.covariance is None:
        advice.append(
            FitAdvice(
                "uncertainty_unavailable",
                "No covariance is available; do not infer parameter precision.",
            )
        )
    for correlation in result.unresolved_correlations:
        advice.append(
            FitAdvice(
                "review_correlated_parameters",
                "These parameters are unresolved under the fit's correlation threshold. "
                "Review the physical model and constraints before freeing more parameters.",
                (correlation.left.label, correlation.right.label),
            )
        )
    bounds = tuple(
        spec.key.label
        for spec in result.parameters.specs
        if spec.value == spec.bounds.lower or spec.value == spec.bounds.upper
    )
    if bounds:
        advice.append(
            FitAdvice(
                "review_active_bounds",
                "Parameters are exactly on a bound; inspect their physical meaning.",
                bounds,
            )
        )
    if residuals.chi_square > 0:
        advice.append(
            FitAdvice(
                "inspect_residual_regions",
                "Inspect the intervals with the largest chi-square contributions alongside "
                "observed/calculated curves. Error concentration does not establish its cause.",
            )
        )
    return FitReport(
        residuals, reason, result.jacobian_rank, result.covariance is not None, tuple(advice)
    )
