"""Public guarded structural multi-bank neutron TOF refinement."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _core
from ..crystallography import UnitCell
from ..execution import ExecutionPolicy
from ..instrument import TofBankGeometry, TofInstrument
from ..intensity_corrections import (
    NeutralIntegratedIntensityCorrection,
    TimeOfFlightNeutronLorentz,
)
from ..io.cif import CifReadLimits, read_cif
from ..io.powder import PowderReadLimits, TofPowderFormat, read_tof_powder_data
from ..io.tof_instrument import GsasTofInstrumentReadLimits, read_gsas_tof_instrument
from ..pattern import TofPowderPattern
from ..phase import RietveldPhase, StructuralReflectionBatch
from ..scattering import NeutronNuclear
from ..structural_calculation import _native_phase
from ..structure import CrystalStructure
from ..symmetry import PreparedReflectionGenerator, TofRange
from .core import (
    Bounds,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    TerminationReason,
)
from .lattice import LatticeParameterBounds, LatticeParameterization
from .runtime import RefinementLimits
from .tof_lebail import TofChebyshevBackground, TofLeBailCancellation, _metrics
from .tof_multibank import TofInstrumentParameterBound

StructuralTofCancellation = TofLeBailCancellation


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


@dataclass(frozen=True, slots=True)
class StructuralTofSourceDigest:
    """Immutable identity for one file or inline-text request source."""

    source_name: str | None
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.source_name is not None and (
            not isinstance(self.source_name, str) or not self.source_name
        ):
            raise ValueError("source_name must be a non-empty string or None")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class StructuralTofRequestProvenance:
    """Source hashes and explicit reduction/physics choices for a file request."""

    pattern: StructuralTofSourceDigest
    instrument: StructuralTofSourceDigest
    structure: StructuralTofSourceDigest
    reduction: StructuralTofSourceDigest
    reduction_embedded_in_pattern: bool
    bank: int
    incident_normalization: Literal["already_normalized", "calibration_type4"]
    correction: Literal["neutral", "already_applied", "tof_lorentz"]
    sample_corrections: Literal["none", "already_applied"]
    fixed_background_supplied: bool
    fixed_background_sha256: str

    def __post_init__(self) -> None:
        for name in ("pattern", "instrument", "structure", "reduction"):
            if not isinstance(getattr(self, name), StructuralTofSourceDigest):
                raise TypeError(f"{name} must be a StructuralTofSourceDigest")
        if not isinstance(self.reduction_embedded_in_pattern, bool):
            raise TypeError("reduction_embedded_in_pattern must be boolean")
        if isinstance(self.bank, bool) or not isinstance(self.bank, int) or self.bank <= 0:
            raise ValueError("bank must be a positive integer")
        if self.incident_normalization not in {"already_normalized", "calibration_type4"}:
            raise ValueError("unsupported incident_normalization provenance")
        if self.correction not in {"neutral", "already_applied", "tof_lorentz"}:
            raise ValueError("unsupported correction provenance")
        if self.sample_corrections not in {"none", "already_applied"}:
            raise ValueError("unsupported sample_corrections provenance")
        if not isinstance(self.fixed_background_supplied, bool):
            raise TypeError("fixed_background_supplied must be boolean")
        if len(self.fixed_background_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.fixed_background_sha256
        ):
            raise ValueError("fixed_background_sha256 must be a lowercase SHA-256 digest")


def _source_digest(source: str | Path, max_bytes: int) -> StructuralTofSourceDigest:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("source digest max_bytes must be a positive integer")
    if isinstance(source, Path):
        size = source.stat().st_size
        if size > max_bytes:
            raise ValueError(f"request source exceeds max_bytes: {size} > {max_bytes}")
        payload = source.read_bytes()
        source_name = str(source)
    elif not isinstance(source, str):
        raise TypeError("request sources must be strings or pathlib.Path values")
    elif "\n" in source or "\r" in source or source.lstrip().lower().startswith("data_"):
        payload = source.encode("utf-8")
        source_name = None
    else:
        candidate = Path(source)
        try:
            is_path = candidate.exists()
        except OSError:
            is_path = False
        if is_path:
            size = candidate.stat().st_size
            if size > max_bytes:
                raise ValueError(f"request source exceeds max_bytes: {size} > {max_bytes}")
            payload = candidate.read_bytes()
            source_name = str(candidate)
        else:
            payload = source.encode("utf-8")
            source_name = None
    if len(payload) > max_bytes:
        raise ValueError(f"request source exceeds max_bytes: {len(payload)} > {max_bytes}")
    return StructuralTofSourceDigest(
        source_name,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


@dataclass(frozen=True, slots=True)
class StructuralTofSelection:
    """Shared structural parameter families selected for refinement."""

    lattice: bool = False
    coordinates: bool = False
    occupancy: bool = False
    u_iso: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.lattice, self.coordinates, self.occupancy, self.u_iso)
        ):
            raise TypeError("structural TOF parameter selections must be boolean")


@dataclass(frozen=True, slots=True)
class StructuralTofBank:
    """One observed bank and its explicit local structural-TOF contract."""

    bank_id: str
    pattern: TofPowderPattern
    instrument: TofInstrument
    geometry: TofBankGeometry
    correction: NeutralIntegratedIntensityCorrection | TimeOfFlightNeutronLorentz
    scale: float = 1.0
    scale_bounds: Bounds = field(default_factory=lambda: Bounds(0.0, np.inf))
    refine_scale: bool = True
    background: TofChebyshevBackground | None = None
    refine_background: bool = False
    instrument_bounds: tuple[TofInstrumentParameterBound, ...] = ()
    _native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bank_id, str)
            or not self.bank_id
            or self.bank_id != self.bank_id.strip()
        ):
            raise ValueError("bank_id must be a non-empty trimmed string")
        if not isinstance(self.pattern, TofPowderPattern) or self.pattern.observed_y is None:
            raise ValueError("structural TOF requires an observed TofPowderPattern")
        if not isinstance(self.instrument, TofInstrument):
            raise TypeError("instrument must be a TofInstrument")
        if not isinstance(self.geometry, TofBankGeometry):
            raise TypeError("geometry must be a TofBankGeometry")
        if type(self.correction) is NeutralIntegratedIntensityCorrection:
            correction = "neutral"
        elif type(self.correction) is TimeOfFlightNeutronLorentz:
            if self.correction.two_theta_deg != self.geometry.two_theta_deg:
                raise ValueError("TOF Lorentz correction angle must match bank geometry exactly")
            correction = "time_of_flight_neutron_lorentz"
        else:
            raise TypeError("correction must be neutral or TimeOfFlightNeutronLorentz")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("scale must be non-negative and finite")
        if not isinstance(self.scale_bounds, Bounds) or not self.scale_bounds.contains(self.scale):
            raise ValueError("scale must lie inside scale_bounds")
        if not isinstance(self.refine_scale, bool) or not isinstance(self.refine_background, bool):
            raise TypeError("bank refinement selections must be boolean")
        if self.background is not None and not isinstance(self.background, TofChebyshevBackground):
            raise TypeError("background must be a TofChebyshevBackground or None")
        if self.refine_background and self.background is None:
            raise ValueError("refine_background requires a background model")
        instrument_bounds = tuple(self.instrument_bounds)
        if any(not isinstance(value, TofInstrumentParameterBound) for value in instrument_bounds):
            raise TypeError("instrument_bounds must contain TofInstrumentParameterBound values")
        if len({value.parameter for value in instrument_bounds}) != len(instrument_bounds):
            raise ValueError("instrument bounds must select unique parameters")
        object.__setattr__(self, "instrument_bounds", instrument_bounds)
        native = _core._StructuralTofBank(
            self.bank_id,
            self.pattern.tof_us,
            self.pattern.observed_y,
            self.pattern.uncertainty,
            self.pattern.mask,
            self.pattern.background,
            self.instrument.as_tuple(),
            self.geometry.two_theta_deg,
            correction,
            self.scale,
            self.scale_bounds.lower,
            self.scale_bounds.upper,
            self.refine_scale,
            None if self.background is None else self.background.coefficients,
            "tof-background" if self.background is None else self.background.background_id,
            self.refine_background,
            [(value.parameter, value.lower, value.upper) for value in instrument_bounds],
        )
        object.__setattr__(self, "_native", native)


@dataclass(frozen=True, slots=True)
class StructuralTofMultiBankInput:
    """One shared neutron structure and one or more explicit detector banks."""

    phase: RietveldPhase
    banks: tuple[StructuralTofBank, ...]
    selection: StructuralTofSelection = StructuralTofSelection()
    lattice_bounds: LatticeParameterBounds | None = None
    support_fwhm: float = 20.0
    tail_log: float = 20.0
    use_uncertainty: bool = True
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    provenance: StructuralTofRequestProvenance | None = None
    _native_phase: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, RietveldPhase):
            raise TypeError("phase must be a RietveldPhase")
        if type(self.phase.scattering) is not NeutronNuclear:
            raise TypeError("structural TOF requires built-in neutron nuclear scattering")
        if type(self.phase.intensity_correction) is not NeutralIntegratedIntensityCorrection:
            raise TypeError("the shared phase must use neutral placeholder correction")
        if self.phase.scale != 1.0:
            raise ValueError("the shared phase must use unit placeholder scale")
        if self.phase.physics is not None:
            raise ValueError("CW sample physics is not part of the structural TOF contract")
        banks = tuple(self.banks)
        if not banks or any(not isinstance(bank, StructuralTofBank) for bank in banks):
            raise ValueError("banks must contain at least one StructuralTofBank value")
        if len({bank.bank_id for bank in banks}) != len(banks):
            raise ValueError("structural TOF bank IDs must be unique")
        if not isinstance(self.selection, StructuralTofSelection):
            raise TypeError("selection must be a StructuralTofSelection")
        parameterization = LatticeParameterization(
            self.phase.structure.space_group, self.phase.structure.cell
        )
        if self.selection.lattice:
            if not isinstance(self.lattice_bounds, LatticeParameterBounds):
                raise ValueError("lattice refinement requires LatticeParameterBounds")
            if self.lattice_bounds.parameter_names != parameterization.parameter_names:
                raise ValueError("lattice bounds use a different parameterization")
        elif self.lattice_bounds is not None:
            raise ValueError("lattice_bounds must be None when lattice refinement is disabled")
        if not np.isfinite(self.support_fwhm) or self.support_fwhm <= 0.0:
            raise ValueError("support_fwhm must be positive and finite")
        if not np.isfinite(self.tail_log) or self.tail_log <= 0.0:
            raise ValueError("tail_log must be positive and finite")
        if not isinstance(self.use_uncertainty, bool):
            raise TypeError("use_uncertainty must be boolean")
        if not isinstance(self.execution, ExecutionPolicy):
            raise TypeError("execution must be an ExecutionPolicy")
        if self.provenance is not None and not isinstance(
            self.provenance, StructuralTofRequestProvenance
        ):
            raise TypeError("provenance must be a StructuralTofRequestProvenance or None")
        native = _native_phase(self.phase, self.execution)
        if native is None:
            raise TypeError("phase cannot be represented by the native structural TOF engine")
        object.__setattr__(self, "banks", banks)
        object.__setattr__(
            self,
            "_native_phase",
            _core._StructuralTofPhase(
                native,
                self.phase.phase_id,
                self.phase.name,
                [site.site_id for site in self.phase.structure.sites],
            ),
        )

    @classmethod
    def from_files(
        cls,
        pattern_path: str | Path,
        instrument_path: str | Path,
        cif_path: str | Path,
        *,
        bank: int,
        incident_normalization: Literal["already_normalized", "calibration_type4"],
        correction: Literal["neutral", "already_applied", "tof_lorentz"],
        sample_corrections: Literal["none", "already_applied"],
        bank_id: str | None = None,
        pattern_format: TofPowderFormat = "auto",
        search_min_d_angstrom: float = 0.25,
        search_max_d_angstrom: float = 5.0,
        fixed_background: ArrayLike | None = None,
        reduction_path: str | Path | None = None,
        powder_limits: PowderReadLimits | None = None,
        instrument_limits: GsasTofInstrumentReadLimits | None = None,
        cif_limits: CifReadLimits | None = None,
    ) -> StructuralTofMultiBankInput:
        """Build one explicit reduced-data/calibration/CIF structural request.

        Normalization, intensity correction, and sample-correction handling are
        mandatory declarations, so file or facility names never select physics
        implicitly. ``already_applied`` records upstream correction without
        applying it again. ``reduction_path`` can identify a separate reduction
        record; otherwise the pattern bytes are the checksum-pinned reduction
        record as well.
        """

        if sample_corrections not in {"none", "already_applied"}:
            raise ValueError(
                "sample_corrections must be 'none' or 'already_applied'; "
                "named correction models are not implemented"
            )

        selected_powder_limits = powder_limits or PowderReadLimits()
        if not isinstance(selected_powder_limits, PowderReadLimits):
            raise TypeError("powder_limits must be a PowderReadLimits or None")
        selected_instrument_limits = instrument_limits or GsasTofInstrumentReadLimits()
        if not isinstance(selected_instrument_limits, GsasTofInstrumentReadLimits):
            raise TypeError("instrument_limits must be a GsasTofInstrumentReadLimits or None")
        selected_cif_limits = cif_limits or CifReadLimits()
        if not isinstance(selected_cif_limits, CifReadLimits):
            raise TypeError("cif_limits must be a CifReadLimits or None")
        pattern_digest = _source_digest(pattern_path, selected_powder_limits.max_bytes)
        instrument_digest = _source_digest(instrument_path, selected_instrument_limits.max_bytes)
        structure_digest = _source_digest(cif_path, selected_cif_limits.max_bytes)
        reduction_digest = (
            pattern_digest
            if reduction_path is None
            else _source_digest(reduction_path, selected_powder_limits.max_bytes)
        )

        powder = read_tof_powder_data(
            pattern_path,
            format=pattern_format,
            bank=bank,
            limits=selected_powder_limits,
        )
        calibration = read_gsas_tof_instrument(
            instrument_path,
            bank=bank,
            limits=selected_instrument_limits,
        )
        geometry = calibration.bank_geometry
        if geometry is None:
            raise ValueError("structural TOF calibration must supply explicit bank geometry")
        pattern = powder.to_pattern(background=fixed_background)
        if incident_normalization == "calibration_type4":
            if calibration.incident_spectrum is None:
                raise ValueError("calibration_type4 requires an incident spectrum in calibration")
            pattern = calibration.incident_spectrum.normalize_pattern(pattern)
        elif incident_normalization != "already_normalized":
            raise ValueError(
                "incident_normalization must be 'already_normalized' or 'calibration_type4'"
            )
        if correction in {"neutral", "already_applied"}:
            correction_model = NeutralIntegratedIntensityCorrection()
        elif correction == "tof_lorentz":
            correction_model = TimeOfFlightNeutronLorentz(geometry.two_theta_deg)
        else:
            raise ValueError("correction must be 'neutral', 'already_applied', or 'tof_lorentz'")
        structure = read_cif(cif_path, limits=selected_cif_limits).structure
        generated = PreparedReflectionGenerator(structure.space_group).generate(
            structure.cell,
            TofRange(
                float(pattern.tof_us[0]),
                float(pattern.tof_us[-1]),
                search_min_d_angstrom,
                search_max_d_angstrom,
                calibration.instrument.zero_us,
                calibration.instrument.difc_us_per_angstrom,
                calibration.instrument.difa_us_per_angstrom2,
                calibration.instrument.difb_us_angstrom,
            ),
        )
        phase = RietveldPhase(
            structure.structure_id,
            structure.name,
            structure,
            StructuralReflectionBatch.from_generated(generated),
            NeutronNuclear(),
            NeutralIntegratedIntensityCorrection(),
        )
        structural_bank = StructuralTofBank(
            bank_id or f"bank-{bank}",
            pattern,
            calibration.instrument,
            geometry,
            correction_model,
        )
        background_digest = hashlib.sha256(
            np.ascontiguousarray(pattern.background, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        provenance = StructuralTofRequestProvenance(
            pattern_digest,
            instrument_digest,
            structure_digest,
            reduction_digest,
            reduction_path is None,
            bank,
            incident_normalization,
            correction,
            sample_corrections,
            fixed_background is not None,
            background_digest,
        )
        return cls(phase, (structural_bank,), provenance=provenance)


@dataclass(frozen=True, slots=True)
class StructuralTofRefinementOptions:
    """Validated controls for bounded dense structural TOF refinement."""

    limits: RefinementLimits = field(
        default_factory=lambda: RefinementLimits(max_iterations=50, max_evaluations=5_000)
    )
    min_iterations: int = 1
    objective_tolerance: float = 1.0e-10
    parameter_tolerance: float = 1.0e-7
    initial_damping: float = 1.0e-6
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    max_scaled_parameter_step: float = 0.25
    max_backtracks: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.limits, RefinementLimits):
            raise TypeError("limits must be RefinementLimits")
        if not isinstance(self.min_iterations, int) or isinstance(self.min_iterations, bool):
            raise ValueError("min_iterations must be a positive integer")
        if self.min_iterations <= 0 or self.min_iterations > self.limits.max_iterations:
            raise ValueError("min_iterations must lie inside the iteration limit")
        positive = (
            self.objective_tolerance,
            self.parameter_tolerance,
            self.initial_damping,
            self.damping_increase,
            self.damping_decrease,
            self.max_scaled_parameter_step,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("solver tolerances, damping, and step cap must be positive")
        if self.damping_increase <= 1.0 or self.damping_decrease >= 1.0:
            raise ValueError("damping must increase above one and decrease below one")
        if not isinstance(self.max_backtracks, int) or isinstance(self.max_backtracks, bool):
            raise ValueError("max_backtracks must be a non-negative integer")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a non-negative integer")


class StructuralTofMultiBankCheckpoint:
    """Opaque accepted state resumed only with the exact original input."""

    __slots__ = ("_native", "_provenance")

    def __init__(
        self,
        native: object,
        provenance: StructuralTofRequestProvenance | None = None,
    ) -> None:
        if not isinstance(native, _core._StructuralTofMultiBankCheckpoint):
            raise TypeError("native must be a PhaseSmith structural TOF checkpoint")
        if provenance is not None and not isinstance(provenance, StructuralTofRequestProvenance):
            raise TypeError("provenance must be a StructuralTofRequestProvenance or None")
        self._native = native
        self._provenance = provenance

    @property
    def completed_iterations(self) -> int:
        """Return the number of accepted structural iterations."""

        return int(self._native.completed_iterations)


@dataclass(frozen=True, slots=True)
class StructuralTofParameterChange:
    """One accepted physical shared or bank-local parameter update."""

    key: ParameterKey
    before: float
    after: float
    scaled_change: float


@dataclass(frozen=True, slots=True)
class StructuralTofIteration:
    """One atomically accepted all-bank structural TOF step."""

    iteration: int
    objective: float
    objective_change: float
    scaled_step_norm: float
    damping: float
    backtracks: int
    parameter_changes: tuple[StructuralTofParameterChange, ...]


@dataclass(frozen=True, slots=True)
class StructuralTofBankResult:
    """One bank's final immutable calculation and residual state."""

    bank_id: str
    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background_y: NDArray[np.float64]
    d_spacing_angstrom: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]
    metrics: ResidualEvaluation


@dataclass(frozen=True, slots=True)
class StructuralTofMultiBankResult:
    """Last accepted structural/bank state and auditable termination data."""

    input: StructuralTofMultiBankInput
    banks: tuple[StructuralTofBankResult, ...]
    parameters: ParameterSet
    objective: float
    history: tuple[StructuralTofIteration, ...]
    termination_reason: TerminationReason
    checkpoint: StructuralTofMultiBankCheckpoint
    evaluations: int


def _updated_input(
    original: StructuralTofMultiBankInput, record: dict[str, object]
) -> StructuralTofMultiBankInput:
    phase_record = record["phase"]
    cell = UnitCell(*phase_record["cell"])
    xyz = phase_record["fractional_xyz"]
    occupancy = np.asarray(phase_record["occupancy"], dtype=np.float64)
    u_iso = np.asarray(phase_record["u_iso_angstrom2"], dtype=np.float64)
    sites = tuple(
        replace(
            site,
            fractional_xyz=tuple(map(float, coordinate)),
            occupancy=float(occ),
            u_iso_angstrom2=(
                float(displacement)
                if original.selection.u_iso or site.u_iso_angstrom2 is not None
                else None
            ),
        )
        for site, coordinate, occ, displacement in zip(
            original.phase.structure.sites, xyz, occupancy, u_iso, strict=True
        )
    )
    structure: CrystalStructure = replace(original.phase.structure, cell=cell, sites=sites)
    phase = replace(original.phase, structure=structure)
    updated_banks = []
    for bank, bank_record in zip(original.banks, record["banks"], strict=True):
        coefficients = bank_record["background_coefficients"]
        background = (
            None
            if bank.background is None
            else replace(
                bank.background,
                coefficients=tuple(float(value) for value in coefficients),
            )
        )
        updated_banks.append(
            replace(
                bank,
                scale=float(bank_record["scale"]),
                instrument=TofInstrument(*bank_record["instrument"]),
                background=background,
            )
        )
    return replace(original, phase=phase, banks=tuple(updated_banks))


def _parameters(records: object) -> ParameterSet:
    return ParameterSet(
        tuple(
            ParameterSpec(
                ParameterKey(*key),
                float(value),
                str(unit),
                Bounds(float(lower), float(upper)),
                float(scale),
            )
            for key, value, unit, lower, upper, scale in records
        )
    )


def _bank_result(record: dict[str, object]) -> StructuralTofBankResult:
    arrays = tuple(
        np.asarray(record[name], dtype=np.float64)
        for name in (
            "y",
            "profile_y",
            "background_y",
            "d_spacing_angstrom",
            "integrated_intensity",
        )
    )
    for array in arrays:
        _freeze(array)
    return StructuralTofBankResult(str(record["bank_id"]), *arrays, _metrics(record["metrics"]))


def refine_structural_tof_multibank(
    input_: StructuralTofMultiBankInput,
    options: StructuralTofRefinementOptions | None = None,
    *,
    cancellation: StructuralTofCancellation | None = None,
    checkpoint: StructuralTofMultiBankCheckpoint | None = None,
    progress: Callable[[dict[str, object]], object] | None = None,
) -> StructuralTofMultiBankResult:
    """Refine one shared neutron structure against an atomic sum of TOF banks."""

    if not isinstance(input_, StructuralTofMultiBankInput):
        raise TypeError("input_ must be a StructuralTofMultiBankInput")
    selected = StructuralTofRefinementOptions() if options is None else options
    if not isinstance(selected, StructuralTofRefinementOptions):
        raise TypeError("options must be StructuralTofRefinementOptions")
    if cancellation is not None and not isinstance(cancellation, TofLeBailCancellation):
        raise TypeError("cancellation must be TofLeBailCancellation or None")
    if checkpoint is not None and not isinstance(checkpoint, StructuralTofMultiBankCheckpoint):
        raise TypeError("checkpoint must be StructuralTofMultiBankCheckpoint or None")
    if checkpoint is not None and checkpoint._provenance != input_.provenance:
        raise ValueError("checkpoint provenance does not match the structural TOF input")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    lattice_bounds = input_.lattice_bounds
    limits = selected.limits
    record = _core._refine_structural_tof_multibank(
        input_._native_phase,
        input_.selection.lattice,
        input_.selection.coordinates,
        input_.selection.occupancy,
        input_.selection.u_iso,
        None if lattice_bounds is None else lattice_bounds.lower.tolist(),
        None if lattice_bounds is None else lattice_bounds.upper.tolist(),
        [bank._native for bank in input_.banks],
        input_.support_fwhm,
        input_.tail_log,
        input_.use_uncertainty,
        input_.execution._native,
        limits.max_iterations,
        limits.max_evaluations,
        limits.max_runtime_seconds,
        limits.max_consecutive_rejections,
        selected.min_iterations,
        selected.objective_tolerance,
        selected.parameter_tolerance,
        selected.initial_damping,
        selected.damping_increase,
        selected.damping_decrease,
        selected.max_scaled_parameter_step,
        selected.max_backtracks,
        None if cancellation is None else cancellation._native,
        None if checkpoint is None else checkpoint._native,
        progress,
    )
    history = tuple(
        StructuralTofIteration(
            int(row["iteration"]),
            float(row["objective"]),
            float(row["objective_change"]),
            float(row["scaled_step_norm"]),
            float(row["damping"]),
            int(row["backtracks"]),
            tuple(
                StructuralTofParameterChange(
                    ParameterKey(*key), float(before), float(after), float(scaled)
                )
                for key, before, after, scaled in row["parameter_changes"]
            ),
        )
        for row in record["history"]
    )
    return StructuralTofMultiBankResult(
        _updated_input(input_, record),
        tuple(_bank_result(bank) for bank in record["banks"]),
        _parameters(record["parameters"]),
        float(record["objective"]),
        history,
        TerminationReason(record["termination_reason"]),
        StructuralTofMultiBankCheckpoint(record["checkpoint"], input_.provenance),
        int(record["evaluations"]),
    )
