"""Public fixed-instrument TOF Le Bail workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import _core
from ..execution import ExecutionPolicy
from ..instrument import TofInstrument
from ..io.cif import read_cif
from ..io.powder import PowderReadLimits, TofPowderFormat, read_tof_powder_data
from ..io.tof_instrument import (
    GsasTofInstrumentReadLimits,
    read_gsas_tof_instrument,
)
from ..pattern import TofPowderPattern
from ..structure import CrystalStructure
from ..symmetry import PreparedReflectionGenerator, TofRange
from .core import ResidualEvaluation


def _float_vector(
    values: ArrayLike, name: str, *, nonnegative: bool = False
) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, init=False)
class TofLeBailPhase:
    """One fixed-cell phase with stable TOF reflection families."""

    phase_id: str
    name: str
    reflection_ids: tuple[str, ...]
    hkl: NDArray[np.int64]
    d_spacing_angstrom: NDArray[np.float64]
    integrated_intensity: NDArray[np.float64]
    scale: float

    def __init__(
        self,
        phase_id: str,
        name: str,
        reflection_ids: tuple[str, ...] | list[str],
        hkl: ArrayLike,
        d_spacing_angstrom: ArrayLike,
        integrated_intensity: ArrayLike,
        *,
        scale: float = 1.0,
    ) -> None:
        """Copy and validate one phase in stable reflection order."""

        if not phase_id or phase_id != phase_id.strip():
            raise ValueError("phase_id must be a non-empty trimmed string")
        if not name or not name.strip():
            raise ValueError("name must be non-empty")
        ids = tuple(reflection_ids)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("reflection_ids must be non-empty and unique")
        indices = np.array(hkl, dtype=np.int64, copy=True, order="C")
        if indices.shape != (len(ids), 3):
            raise ValueError("hkl must have shape (reflection_count, 3)")
        spacing = _float_vector(d_spacing_angstrom, "d_spacing_angstrom")
        intensity = _float_vector(
            integrated_intensity, "integrated_intensity", nonnegative=True
        )
        if spacing.shape != (len(ids),) or np.any(spacing <= 0.0):
            raise ValueError("d_spacing_angstrom must be positive and match reflection_ids")
        if intensity.shape != (len(ids),):
            raise ValueError("integrated_intensity must match reflection_ids")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        indices.flags.writeable = False
        object.__setattr__(self, "phase_id", phase_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "reflection_ids", ids)
        object.__setattr__(self, "hkl", indices)
        object.__setattr__(self, "d_spacing_angstrom", spacing)
        object.__setattr__(self, "integrated_intensity", intensity)
        object.__setattr__(self, "scale", float(scale))

    @classmethod
    def from_structure(
        cls,
        structure: CrystalStructure,
        pattern: TofPowderPattern,
        instrument: TofInstrument,
        *,
        phase_id: str | None = None,
        name: str | None = None,
        search_min_d_angstrom: float = 0.25,
        search_max_d_angstrom: float = 5.0,
        initial_intensity: float = 0.0,
        scale: float = 1.0,
    ) -> TofLeBailPhase:
        """Generate allowed reflection families for a structure and TOF window."""

        if not isinstance(structure, CrystalStructure):
            raise TypeError("structure must be a CrystalStructure")
        if not isinstance(pattern, TofPowderPattern):
            raise TypeError("pattern must be a TofPowderPattern")
        if not isinstance(instrument, TofInstrument):
            raise TypeError("instrument must be a TofInstrument")
        if not np.isfinite(initial_intensity) or initial_intensity < 0.0:
            raise ValueError("initial_intensity must be nonnegative and finite")
        generated = PreparedReflectionGenerator(structure.space_group).generate(
            structure.cell,
            TofRange(
                float(pattern.tof_us[0]),
                float(pattern.tof_us[-1]),
                search_min_d_angstrom,
                search_max_d_angstrom,
                instrument.zero_us,
                instrument.difc_us_per_angstrom,
                instrument.difa_us_per_angstrom2,
                instrument.difb_us_angstrom,
            ),
        )
        return cls(
            phase_id or structure.structure_id,
            name or structure.name,
            generated.reflection_ids,
            generated.hkl,
            generated.d_spacing_angstrom,
            np.full(len(generated.reflection_ids), initial_intensity),
            scale=scale,
        )


@dataclass(frozen=True, slots=True)
class TofChebyshevBackground:
    """Refinable Chebyshev residual background on the complete TOF grid."""

    coefficients: tuple[float, ...]
    background_id: str = "tof-background"

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.coefficients)
        if not coefficients or not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be a non-empty finite sequence")
        if not self.background_id or self.background_id != self.background_id.strip():
            raise ValueError("background_id must be a non-empty trimmed string")
        object.__setattr__(self, "coefficients", coefficients)


@dataclass(frozen=True, slots=True)
class TofLeBailInput:
    """Typed fixed-instrument TOF extraction request."""

    pattern: TofPowderPattern
    instrument: TofInstrument
    phases: tuple[TofLeBailPhase, ...]
    background: TofChebyshevBackground | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, TofPowderPattern):
            raise TypeError("pattern must be a TofPowderPattern")
        if self.pattern.observed_y is None:
            raise ValueError("TOF Le Bail extraction requires observed_y")
        if not isinstance(self.instrument, TofInstrument):
            raise TypeError("instrument must be a TofInstrument")
        phases = tuple(self.phases)
        if not phases or any(not isinstance(phase, TofLeBailPhase) for phase in phases):
            raise ValueError("phases must contain at least one TofLeBailPhase")
        if len({phase.phase_id for phase in phases}) != len(phases):
            raise ValueError("phase IDs must be unique")
        if self.background is not None and not isinstance(
            self.background, TofChebyshevBackground
        ):
            raise TypeError("background must be a TofChebyshevBackground or None")
        object.__setattr__(self, "phases", phases)

    @classmethod
    def from_files(
        cls,
        pattern_path: str | Path,
        instrument_path: str | Path,
        cif_path: str | Path,
        *,
        bank: int,
        pattern_format: TofPowderFormat = "auto",
        search_min_d_angstrom: float = 0.25,
        search_max_d_angstrom: float = 5.0,
        background_terms: int = 16,
        fixed_background: ArrayLike | None = None,
        powder_limits: PowderReadLimits | None = None,
        instrument_limits: GsasTofInstrumentReadLimits | None = None,
    ) -> TofLeBailInput:
        """Build one supported reduced-TOF + legacy GSAS calibration + CIF request."""

        powder = read_tof_powder_data(
            pattern_path,
            format=pattern_format,
            bank=bank,
            limits=powder_limits,
        )
        calibration = read_gsas_tof_instrument(
            instrument_path, bank=bank, limits=instrument_limits
        )
        pattern = powder.to_pattern(background=fixed_background)
        structure = read_cif(cif_path).structure
        phase = TofLeBailPhase.from_structure(
            structure,
            pattern,
            calibration.instrument,
            search_min_d_angstrom=search_min_d_angstrom,
            search_max_d_angstrom=search_max_d_angstrom,
        )
        if isinstance(background_terms, bool) or not isinstance(background_terms, int):
            raise TypeError("background_terms must be an integer")
        if background_terms < 0:
            raise ValueError("background_terms must be nonnegative")
        background = (
            None
            if background_terms == 0
            else TofChebyshevBackground(tuple(0.0 for _ in range(background_terms)))
        )
        return cls(pattern, calibration.instrument, (phase,), background)


@dataclass(frozen=True, slots=True)
class TofLeBailOptions:
    """Deterministic native TOF intensity-extraction controls."""

    cycles: int = 50
    redistribution_damping: float = 1.0
    initial_intensity_floor: float = 1.0e-12
    minimum_calculated: float = 1.0e-15
    support_fwhm: float = 20.0
    tail_log: float = 20.0
    use_uncertainty: bool = True
    redistribution_use_uncertainty: bool = True
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def __post_init__(self) -> None:
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        if not 0.0 < self.redistribution_damping <= 1.0:
            raise ValueError("redistribution_damping must lie in (0, 1]")
        for name in (
            "initial_intensity_floor",
            "minimum_calculated",
            "support_fwhm",
            "tail_log",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.execution, ExecutionPolicy):
            raise TypeError("execution must be an ExecutionPolicy")


@dataclass(frozen=True, slots=True)
class TofLeBailIteration:
    """One accepted native extraction cycle."""

    iteration: int
    metrics: ResidualEvaluation
    maximum_relative_intensity_change: float
    maximum_absolute_background_change: float


@dataclass(frozen=True, slots=True)
class TofLeBailResult:
    """Final pattern, stable intensities, metrics, and complete cycle history."""

    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background_y: NDArray[np.float64]
    reflection_keys: tuple[tuple[str, str], ...]
    phase_offsets: NDArray[np.int64]
    integrated_intensity: NDArray[np.float64]
    background_coefficients: NDArray[np.float64] | None
    metrics: ResidualEvaluation
    history: tuple[TofLeBailIteration, ...]


def _metrics(record: dict[str, object]) -> ResidualEvaluation:
    arrays = [
        np.asarray(record["included"], dtype=np.bool_),
        np.asarray(record["residual"], dtype=np.float64),
        np.asarray(record["weighted_residual"], dtype=np.float64),
    ]
    for array in arrays:
        array.flags.writeable = False
    return ResidualEvaluation(
        arrays[0],
        arrays[1],
        arrays[2],
        float(record["rp"]),
        float(record["rwp"]),
        float(record["chi_square"]),
        float(record["reduced_chi_square"]),
    )


def refine_tof_lebail(
    input_: TofLeBailInput,
    options: TofLeBailOptions | None = None,
) -> TofLeBailResult:
    """Run native nonnegative fixed-instrument TOF Le Bail extraction."""

    if not isinstance(input_, TofLeBailInput):
        raise TypeError("input_ must be a TofLeBailInput")
    selected = options or TofLeBailOptions()
    if not isinstance(selected, TofLeBailOptions):
        raise TypeError("options must be a TofLeBailOptions")
    offsets = np.zeros(len(input_.phases) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(phase.reflection_ids) for phase in input_.phases])
    record = _core._refine_tof_lebail(
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
        selected.cycles,
        selected.redistribution_damping,
        selected.initial_intensity_floor,
        selected.minimum_calculated,
        selected.support_fwhm,
        selected.tail_log,
        selected.use_uncertainty,
        selected.redistribution_use_uncertainty,
        selected.execution._native,
    )
    history = tuple(
        TofLeBailIteration(
            int(item["iteration"]),
            _metrics(item["metrics"]),
            float(item["maximum_relative_intensity_change"]),
            float(item["maximum_absolute_background_change"]),
        )
        for item in record["history"]
    )
    arrays = {
        name: np.asarray(record[name], dtype=np.float64)
        for name in ("y", "profile_y", "background_y", "integrated_intensity")
    }
    phase_offsets = np.asarray(record["phase_offsets"], dtype=np.int64)
    coefficients = record["background_coefficients"]
    background_coefficients = (
        None if coefficients is None else np.asarray(coefficients, dtype=np.float64)
    )
    for array in (*arrays.values(), phase_offsets, background_coefficients):
        if array is not None:
            array.flags.writeable = False
    return TofLeBailResult(
        arrays["y"],
        arrays["profile_y"],
        arrays["background_y"],
        tuple(tuple(value) for value in record["reflection_keys"]),
        phase_offsets,
        arrays["integrated_intensity"],
        background_coefficients,
        _metrics(record["metrics"]),
        history,
    )
