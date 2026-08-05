"""Dioptas-oriented NumPy boundary with no Dioptas runtime dependency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..calculation import CalculationOptions, calculate_pattern
from ..control import CancellationCallback, ProgressCallback
from ..instrument import ConstantWavelengthInstrument
from ..pattern import PatternCalculationResult, PowderPattern
from ..phase import Phase
from ..refinement.lebail import LeBailInput, LeBailOptions, LeBailResult, refine


@dataclass(frozen=True, slots=True, init=False)
class DioptasPatternData:
    """Validated display/input arrays and plain metadata at the GUI boundary."""

    x_unit: ClassVar[str] = "degree_2theta"
    intensity_unit: ClassVar[str] = "arbitrary_intensity"
    pattern: PowderPattern
    metadata: Mapping[str, Any]

    def __init__(
        self,
        x_deg: ArrayLike,
        observed_y: ArrayLike,
        *,
        background_y: ArrayLike | None = None,
        uncertainty_y: ArrayLike | None = None,
        included_mask: ArrayLike | None = None,
        excluded_mask: ArrayLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Copy Dioptas-style arrays and normalize mask semantics.

        Exactly one of ``included_mask`` or ``excluded_mask`` may be supplied.
        The core convention is always ``True = included``.
        """

        if included_mask is not None and excluded_mask is not None:
            raise ValueError("supply included_mask or excluded_mask, not both")
        if excluded_mask is not None:
            raw = np.asarray(excluded_mask)
            if raw.dtype != np.bool_:
                raise ValueError("excluded_mask must contain boolean values")
            included_mask = np.logical_not(raw)
        selected_metadata = {} if metadata is None else dict(metadata)
        if any(not isinstance(key, str) for key in selected_metadata):
            raise ValueError("metadata keys must be strings")
        pattern = PowderPattern(
            x_deg,
            observed_y=observed_y,
            background=background_y,
            uncertainty=uncertainty_y,
            mask=included_mask,
        )
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "metadata", MappingProxyType(selected_metadata))

    @property
    def x_deg(self) -> NDArray[np.float64]:
        return self.pattern.x

    @property
    def observed_y(self) -> NDArray[np.float64]:
        if self.pattern.observed_y is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("observed intensity is unavailable")
        return self.pattern.observed_y

    @property
    def background_y(self) -> NDArray[np.float64]:
        return self.pattern.background

    @property
    def uncertainty_y(self) -> NDArray[np.float64] | None:
        return self.pattern.uncertainty

    @property
    def included_mask(self) -> NDArray[np.bool_] | None:
        return self.pattern.mask


@dataclass(frozen=True, slots=True)
class DisplayCurve:
    """One labeled, display-ready curve in input sample order."""

    label: str
    y: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("curve label must be a non-empty string")
        if self.y.dtype != np.float64 or self.y.ndim != 1 or not np.isfinite(self.y).all():
            raise ValueError("curve y must be a finite float64 vector")


@dataclass(frozen=True, slots=True)
class DioptasDisplayResult:
    """Calculated/display arrays with background kept explicitly separate."""

    x_unit: ClassVar[str] = "degree_2theta"
    intensity_unit: ClassVar[str] = "arbitrary_intensity"
    x_deg: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    calculated_y: NDArray[np.float64]
    background_y: NDArray[np.float64]
    difference_y: NDArray[np.float64]
    included_mask: NDArray[np.bool_]
    phase_curves: tuple[DisplayCurve, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate all display arrays against one shared sample count."""

        count = self.x_deg.size
        for name, value, dtype in (
            ("x_deg", self.x_deg, np.float64),
            ("observed_y", self.observed_y, np.float64),
            ("calculated_y", self.calculated_y, np.float64),
            ("background_y", self.background_y, np.float64),
            ("difference_y", self.difference_y, np.float64),
            ("included_mask", self.included_mask, np.bool_),
        ):
            if value.dtype != dtype or value.shape != (count,):
                raise ValueError(f"{name} must match x_deg shape and documented dtype")
            if dtype == np.float64 and not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if count > 1 and np.any(np.diff(self.x_deg) <= 0.0):
            raise ValueError("x_deg must be strictly increasing")
        if any(curve.y.shape != (count,) for curve in self.phase_curves):
            raise ValueError("phase curves must match the display sample count")


@runtime_checkable
class DioptasConsumer(Protocol):
    """Minimal glue implemented inside Dioptas or another GUI application."""

    def read_pattern(self) -> DioptasPatternData:
        """Return current arrays without exposing GUI-owned objects."""

    def publish_result(self, result: DioptasDisplayResult) -> None:
        """Receive display-ready copies on the GUI-owned thread."""


def _display_result(
    source: DioptasPatternData,
    calculation: PatternCalculationResult,
    diagnostics: Mapping[str, Any],
) -> DioptasDisplayResult:
    observed = source.observed_y
    included = (
        np.ones(observed.size, dtype=np.bool_)
        if source.included_mask is None
        else np.array(source.included_mask, copy=True)
    )
    difference = observed - calculation.y
    difference.flags.writeable = False
    included.flags.writeable = False
    return DioptasDisplayResult(
        source.x_deg,
        observed,
        calculation.y,
        calculation.background,
        difference,
        included,
        tuple(
            DisplayCurve(component.phase_id, component.y)
            for component in calculation.phase_components
        ),
        MappingProxyType(dict(diagnostics)),
    )


def calculate(
    source: DioptasPatternData,
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: CalculationOptions | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> DioptasDisplayResult:
    """Calculate one GUI-facing pattern through the ordinary public API."""

    if not isinstance(source, DioptasPatternData):
        raise TypeError("source must be DioptasPatternData")
    selected = CalculationOptions(return_phase_components=True) if options is None else options
    if not selected.return_phase_components:
        selected = CalculationOptions(
            selected.support_fwhm,
            selected.jacobian_layout,
            return_phase_components=True,
        )
    calculation = calculate_pattern(
        source.pattern,
        instrument,
        phases,
        options=selected,
        progress=progress,
        cancellation=cancellation,
    )
    return _display_result(source, calculation, {"method": "calculation"})


def refine_lebail(
    source: DioptasPatternData,
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: LeBailOptions | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> tuple[DioptasDisplayResult, LeBailResult]:
    """Run Le Bail and return both display arrays and the full typed result."""

    if not isinstance(source, DioptasPatternData):
        raise TypeError("source must be DioptasPatternData")
    result = refine(
        LeBailInput(source.pattern, instrument, tuple(phases)),
        options,
        progress=progress,
        cancellation=cancellation,
    )
    display = _display_result(
        source,
        result.calculation,
        {
            "method": "lebail",
            "termination_reason": result.termination_reason.value,
            "rp": result.metrics.rp,
            "rwp": result.metrics.rwp,
            "chi_square": result.metrics.chi_square,
            "iteration_count": len(result.history),
            "rank_deficient_group_count": len(result.rank_deficient_groups),
        },
    )
    return display, result


def run_calculation(
    consumer: DioptasConsumer,
    instrument: ConstantWavelengthInstrument,
    phases: tuple[Phase, ...] | list[Phase],
    *,
    options: CalculationOptions | None = None,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCallback | None = None,
) -> DioptasDisplayResult:
    """Read, calculate, publish, and return through the minimal consumer protocol."""

    if not isinstance(consumer, DioptasConsumer):
        raise TypeError("consumer must implement DioptasConsumer")
    result = calculate(
        consumer.read_pattern(),
        instrument,
        phases,
        options=options,
        progress=progress,
        cancellation=cancellation,
    )
    consumer.publish_result(result)
    return result
