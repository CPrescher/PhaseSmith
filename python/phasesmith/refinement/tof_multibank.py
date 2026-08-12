"""Typed Python facade for joint multi-bank TOF geometry refinement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..crystallography import UnitCell
from ..instrument import TofInstrument
from .core import ResidualEvaluation
from .lattice import LatticeParameterBounds, LatticeParameterization
from .tof_lebail import (
    TofLeBailCancellation,
    TofLeBailInput,
    TofLeBailOptions,
    _metrics,
)

TofInstrumentParameter = Literal[
    "zero",
    "difc",
    "difa",
    "difb",
    "alpha",
    "beta0",
    "beta1",
    "betaq",
    "sigma0",
    "sigma1",
    "sigma2",
    "sigmaq",
    "x",
    "y",
    "z",
]

TOF_INSTRUMENT_PARAMETERS: tuple[str, ...] = (
    "zero",
    "difc",
    "difa",
    "difb",
    "alpha",
    "beta0",
    "beta1",
    "betaq",
    "sigma0",
    "sigma1",
    "sigma2",
    "sigmaq",
    "x",
    "y",
    "z",
)


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _native_bank(bank_id: str, input_: TofLeBailInput) -> object:
    offsets = np.zeros(len(input_.phases) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(phase.reflection_ids) for phase in input_.phases])
    return _core._TofLeBailBank(
        bank_id,
        input_.pattern.tof_us,
        input_.pattern.observed_y,
        input_.pattern.uncertainty,
        input_.pattern.mask,
        input_.pattern.background,
        input_.instrument.as_tuple(),
        [phase.phase_id for phase in input_.phases],
        [phase.name for phase in input_.phases],
        offsets,
        [item for phase in input_.phases for item in phase.reflection_ids],
        np.vstack([phase.hkl for phase in input_.phases]),
        np.concatenate([phase.d_spacing_angstrom for phase in input_.phases]),
        np.concatenate([phase.integrated_intensity for phase in input_.phases]),
        np.asarray([phase.scale for phase in input_.phases]),
        None if input_.background is None else input_.background.coefficients,
        "tof-background" if input_.background is None else input_.background.background_id,
    )


@dataclass(frozen=True, slots=True)
class TofLeBailBank:
    """One bank-local observation/model block in a joint TOF request."""

    bank_id: str
    input: TofLeBailInput
    _native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bank_id or self.bank_id != self.bank_id.strip():
            raise ValueError("bank_id must be a non-empty trimmed string")
        if not isinstance(self.input, TofLeBailInput):
            raise TypeError("input must be a TofLeBailInput")
        object.__setattr__(self, "_native", _native_bank(self.bank_id, self.input))


@dataclass(frozen=True, slots=True)
class TofSharedLatticePhase:
    """One bounded setting-aware cell shared by every detector bank."""

    phase_id: str
    parameterization: LatticeParameterization
    bounds: LatticeParameterBounds
    initial_cell: UnitCell
    _native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.phase_id or self.phase_id != self.phase_id.strip():
            raise ValueError("phase_id must be a non-empty trimmed string")
        if not isinstance(self.parameterization, LatticeParameterization):
            raise TypeError("parameterization must be a LatticeParameterization")
        if not isinstance(self.bounds, LatticeParameterBounds):
            raise TypeError("bounds must be LatticeParameterBounds")
        if not isinstance(self.initial_cell, UnitCell):
            raise TypeError("initial_cell must be a UnitCell")
        values = self.parameterization.values_from_cell(self.initial_cell)
        if self.bounds.parameter_names != self.parameterization.parameter_names:
            raise ValueError("bounds use a different lattice parameter order")
        if np.any(values < self.bounds.lower) or np.any(values > self.bounds.upper):
            raise ValueError("initial_cell lies outside its declared bounds")
        native = _core._TofSharedLatticePhase(
            self.parameterization.space_group._native,
            self.phase_id,
            self.initial_cell.as_tuple(),
            self.bounds.lower.tolist(),
            self.bounds.upper.tolist(),
        )
        object.__setattr__(self, "_native", native)


@dataclass(frozen=True, slots=True)
class TofInstrumentParameterBound:
    """Closed finite physical interval for one bank-local coefficient."""

    parameter: TofInstrumentParameter
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.parameter not in TOF_INSTRUMENT_PARAMETERS:
            raise ValueError(f"unknown TOF instrument parameter {self.parameter!r}")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError("TOF instrument bounds must be finite and increasing")


@dataclass(frozen=True, slots=True)
class TofBankInstrumentModel:
    """Selected coefficients for one bank in deterministic solver order."""

    bank_id: str
    bounds: tuple[TofInstrumentParameterBound, ...]
    _native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bank_id or self.bank_id != self.bank_id.strip():
            raise ValueError("bank_id must be a non-empty trimmed string")
        bounds = tuple(self.bounds)
        if not bounds or any(not isinstance(item, TofInstrumentParameterBound) for item in bounds):
            raise ValueError("bounds must contain at least one TofInstrumentParameterBound")
        if len({item.parameter for item in bounds}) != len(bounds):
            raise ValueError("selected instrument parameters must be unique within a bank")
        native = _core._TofBankInstrumentModel(
            self.bank_id,
            [(item.parameter, item.lower, item.upper) for item in bounds],
        )
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "_native", native)


@dataclass(frozen=True, slots=True)
class TofMultiBankGeometryInput:
    """Two or more TOF banks plus shared cells and local instrument selections."""

    banks: tuple[TofLeBailBank, ...]
    lattice_phases: tuple[TofSharedLatticePhase, ...]
    instrument_models: tuple[TofBankInstrumentModel, ...]

    def __post_init__(self) -> None:
        banks = tuple(self.banks)
        lattice = tuple(self.lattice_phases)
        instruments = tuple(self.instrument_models)
        if len(banks) < 2 or any(not isinstance(bank, TofLeBailBank) for bank in banks):
            raise ValueError("banks must contain at least two TofLeBailBank values")
        if len({bank.bank_id for bank in banks}) != len(banks):
            raise ValueError("bank IDs must be unique")
        if not lattice or any(not isinstance(item, TofSharedLatticePhase) for item in lattice):
            raise ValueError("lattice_phases must contain at least one shared phase")
        if len({item.phase_id for item in lattice}) != len(lattice):
            raise ValueError("shared lattice phase IDs must be unique")
        if not instruments or any(
            not isinstance(item, TofBankInstrumentModel) for item in instruments
        ):
            raise ValueError("instrument_models must contain at least one selected bank")
        if len({item.bank_id for item in instruments}) != len(instruments):
            raise ValueError("instrument model bank IDs must be unique")
        bank_ids = {bank.bank_id for bank in banks}
        if any(item.bank_id not in bank_ids for item in instruments):
            raise ValueError("instrument model references a bank absent from the request")
        object.__setattr__(self, "banks", banks)
        object.__setattr__(self, "lattice_phases", lattice)
        object.__setattr__(self, "instrument_models", instruments)


@dataclass(frozen=True, slots=True)
class TofMultiBankGeometryOptions:
    """Extraction and bounded joint-geometry controls."""

    lebail: TofLeBailOptions = field(default_factory=TofLeBailOptions)
    geometry_damping: float = 1.0e-10
    max_scaled_geometry_step: float = 0.08
    max_geometry_backtracks: int = 10
    unresolved_correlation: float = 0.98

    def __post_init__(self) -> None:
        if not isinstance(self.lebail, TofLeBailOptions):
            raise TypeError("lebail must be TofLeBailOptions")
        if not np.isfinite(self.geometry_damping) or self.geometry_damping < 0.0:
            raise ValueError("geometry_damping must be finite and nonnegative")
        if not np.isfinite(self.max_scaled_geometry_step) or self.max_scaled_geometry_step <= 0.0:
            raise ValueError("max_scaled_geometry_step must be positive and finite")
        if (
            isinstance(self.max_geometry_backtracks, bool)
            or not isinstance(self.max_geometry_backtracks, int)
            or self.max_geometry_backtracks < 0
        ):
            raise ValueError("max_geometry_backtracks must be a nonnegative integer")
        if not np.isfinite(self.unresolved_correlation) or not (
            0.0 <= self.unresolved_correlation <= 1.0
        ):
            raise ValueError("unresolved_correlation must lie in [0, 1]")


class TofMultiBankGeometryCheckpoint:
    """Opaque complete accepted state for exact joint continuation."""

    __slots__ = ("_native",)

    def __init__(self, native: object) -> None:
        if not isinstance(native, _core._TofMultiBankGeometryCheckpoint):
            raise TypeError("native must be a PhaseSmith multi-bank TOF geometry checkpoint")
        self._native = native

    @property
    def completed_iterations(self) -> int:
        """Return the number of accepted joint cycles."""

        return int(self._native.completed_iterations)


@dataclass(frozen=True, slots=True)
class TofMultiBankMetrics:
    """Residual totals over the concatenated included bank observations."""

    included_samples: int
    rp: float
    rwp: float
    chi_square: float
    reduced_chi_square: float


@dataclass(frozen=True, slots=True)
class TofGeometryParameterKey:
    """Stable identity of one shared-cell or bank-local instrument column."""

    family: Literal["lattice", "instrument"]
    owner_id: str
    parameter_name: str


@dataclass(frozen=True, slots=True)
class TofGeometryCorrelation:
    """One unresolved pair of normalized weighted Jacobian columns."""

    left: TofGeometryParameterKey
    right: TofGeometryParameterKey
    correlation: float


@dataclass(frozen=True, slots=True)
class TofGeometryDiagnostics:
    """Numerical rank and named unresolved geometry correlations."""

    parameter_count: int
    jacobian_rank: int
    maximum_absolute_correlation: float | None
    unresolved_correlations: tuple[TofGeometryCorrelation, ...]


@dataclass(frozen=True, slots=True)
class TofLatticeParameterChange:
    """One accepted physical shared-cell change."""

    phase_id: str
    parameter_name: str
    before: float
    after: float
    scaled_change: float


@dataclass(frozen=True, slots=True)
class TofInstrumentParameterChange:
    """One accepted physical bank-local instrument change."""

    bank_id: str
    parameter: str
    before: float
    after: float
    scaled_change: float


@dataclass(frozen=True, slots=True)
class TofMultiBankGeometryIteration:
    """One atomically accepted extraction and joint-geometry cycle."""

    iteration: int
    bank_metrics: tuple[ResidualEvaluation, ...]
    metrics: TofMultiBankMetrics
    maximum_relative_intensity_change: float
    maximum_absolute_background_change: float
    scaled_geometry_step_norm: float
    lattice_parameter_changes: tuple[TofLatticeParameterChange, ...]
    instrument_parameter_changes: tuple[TofInstrumentParameterChange, ...]


@dataclass(frozen=True, slots=True)
class TofMultiBankGeometryBankResult:
    """One bank's final immutable calculation and extracted state."""

    bank_id: str
    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background_y: NDArray[np.float64]
    reflection_keys: tuple[tuple[str, str], ...]
    phase_offsets: NDArray[np.int64]
    integrated_intensity: NDArray[np.float64]
    background_coefficients: NDArray[np.float64] | None
    metrics: ResidualEvaluation


@dataclass(frozen=True, slots=True)
class TofSharedLatticeState:
    """One final shared cell."""

    phase_id: str
    cell: UnitCell


@dataclass(frozen=True, slots=True)
class TofBankInstrumentState:
    """One final bank-local instrument."""

    bank_id: str
    instrument: TofInstrument


@dataclass(frozen=True, slots=True)
class TofMultiBankGeometryResult:
    """Complete final joint multi-bank TOF geometry result."""

    banks: tuple[TofMultiBankGeometryBankResult, ...]
    lattice_phases: tuple[TofSharedLatticeState, ...]
    instruments: tuple[TofBankInstrumentState, ...]
    metrics: TofMultiBankMetrics
    diagnostics: TofGeometryDiagnostics
    history: tuple[TofMultiBankGeometryIteration, ...]
    termination_reason: str
    checkpoint: TofMultiBankGeometryCheckpoint


def _aggregate_metrics(record: dict[str, object]) -> TofMultiBankMetrics:
    return TofMultiBankMetrics(
        int(record["included_samples"]),
        float(record["rp"]),
        float(record["rwp"]),
        float(record["chi_square"]),
        float(record["reduced_chi_square"]),
    )


def _geometry_key(record: tuple[str, str, str]) -> TofGeometryParameterKey:
    return TofGeometryParameterKey(record[0], record[1], record[2])  # type: ignore[arg-type]


def _bank_result(record: dict[str, object]) -> TofMultiBankGeometryBankResult:
    arrays = {
        name: np.asarray(record[name], dtype=np.float64)
        for name in ("y", "profile_y", "background_y", "integrated_intensity")
    }
    offsets = np.asarray(record["phase_offsets"], dtype=np.int64)
    coefficients = record["background_coefficients"]
    background = None if coefficients is None else np.asarray(coefficients, dtype=np.float64)
    for array in (*arrays.values(), offsets, background):
        if array is not None:
            _freeze(array)
    return TofMultiBankGeometryBankResult(
        str(record["bank_id"]),
        arrays["y"],
        arrays["profile_y"],
        arrays["background_y"],
        tuple(tuple(item) for item in record["reflection_keys"]),  # type: ignore[arg-type]
        offsets,
        arrays["integrated_intensity"],
        background,
        _metrics(record["metrics"]),  # type: ignore[arg-type]
    )


def refine_tof_multibank_geometry(
    input_: TofMultiBankGeometryInput,
    options: TofMultiBankGeometryOptions | None = None,
    *,
    cancellation: TofLeBailCancellation | None = None,
    checkpoint: TofMultiBankGeometryCheckpoint | None = None,
    progress: Callable[[dict[str, object]], object] | None = None,
) -> TofMultiBankGeometryResult:
    """Run one native joint shared-cell/bank-instrument TOF refinement."""

    if not isinstance(input_, TofMultiBankGeometryInput):
        raise TypeError("input_ must be TofMultiBankGeometryInput")
    selected = TofMultiBankGeometryOptions() if options is None else options
    if not isinstance(selected, TofMultiBankGeometryOptions):
        raise TypeError("options must be TofMultiBankGeometryOptions")
    if cancellation is not None and not isinstance(cancellation, TofLeBailCancellation):
        raise TypeError("cancellation must be TofLeBailCancellation or None")
    if checkpoint is not None and not isinstance(checkpoint, TofMultiBankGeometryCheckpoint):
        raise TypeError("checkpoint must be TofMultiBankGeometryCheckpoint or None")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    lebail = selected.lebail
    record = _core._refine_tof_multibank_geometry(
        [bank._native for bank in input_.banks],
        [phase._native for phase in input_.lattice_phases],
        [model._native for model in input_.instrument_models],
        lebail.cycles,
        lebail.redistribution_damping,
        lebail.initial_intensity_floor,
        lebail.minimum_calculated,
        lebail.support_fwhm,
        lebail.tail_log,
        lebail.use_uncertainty,
        lebail.redistribution_use_uncertainty,
        selected.geometry_damping,
        selected.max_scaled_geometry_step,
        selected.max_geometry_backtracks,
        selected.unresolved_correlation,
        lebail.execution._native,
        None if cancellation is None else cancellation._native,
        None if checkpoint is None else checkpoint._native,
        progress,
    )
    diagnostics_record = record["diagnostics"]
    diagnostics = TofGeometryDiagnostics(
        int(diagnostics_record["parameter_count"]),
        int(diagnostics_record["jacobian_rank"]),
        None
        if diagnostics_record["maximum_absolute_correlation"] is None
        else float(diagnostics_record["maximum_absolute_correlation"]),
        tuple(
            TofGeometryCorrelation(_geometry_key(item[0]), _geometry_key(item[1]), float(item[2]))
            for item in diagnostics_record["unresolved_correlations"]
        ),
    )
    history = tuple(
        TofMultiBankGeometryIteration(
            int(item["iteration"]),
            tuple(_metrics(value) for value in item["bank_metrics"]),
            _aggregate_metrics(item["metrics"]),
            float(item["maximum_relative_intensity_change"]),
            float(item["maximum_absolute_background_change"]),
            float(item["scaled_geometry_step_norm"]),
            tuple(TofLatticeParameterChange(*value) for value in item["lattice_parameter_changes"]),
            tuple(
                TofInstrumentParameterChange(*value)
                for value in item["instrument_parameter_changes"]
            ),
        )
        for item in record["history"]
    )
    return TofMultiBankGeometryResult(
        tuple(_bank_result(item) for item in record["banks"]),
        tuple(
            TofSharedLatticeState(str(phase_id), UnitCell(*cell_values))
            for phase_id, cell_values in record["lattice_phases"]
        ),
        tuple(
            TofBankInstrumentState(str(bank_id), TofInstrument(*values))
            for bank_id, values in record["instruments"]
        ),
        _aggregate_metrics(record["metrics"]),
        diagnostics,
        history,
        str(record["termination_reason"]),
        TofMultiBankGeometryCheckpoint(record["checkpoint"]),
    )
