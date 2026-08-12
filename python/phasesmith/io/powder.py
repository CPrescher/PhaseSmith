"""Small, deterministic readers for common text powder-pattern formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..pattern import PowderPattern, TofPowderPattern

PowderFormat = Literal["auto", "columns", "gsas_fxye", "gsas_std"]
TofPowderFormat = Literal["auto", "columns", "gsas_slog_fxye", "gsas_const_std"]


@dataclass(frozen=True, slots=True)
class PowderReadLimits:
    """Resource limits applied before allocating pattern arrays."""

    max_bytes: int = 64 * 1024 * 1024
    max_rows: int = 5_000_000

    def __post_init__(self) -> None:
        if self.max_bytes <= 0 or self.max_rows <= 0:
            raise ValueError("powder read limits must be positive")


@dataclass(frozen=True, slots=True)
class PowderData:
    """One observed powder pattern with immutable arrays and source metadata."""

    x: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    uncertainty: NDArray[np.float64] | None
    format: Literal["columns", "gsas_fxye", "gsas_std"]
    source_name: str | None = None
    bank: int | None = None
    mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        count = self.x.size
        if self.x.dtype != np.float64 or self.x.shape != (count,):
            raise ValueError("x must be a one-dimensional float64 array")
        if self.observed_y.dtype != np.float64 or self.observed_y.shape != (count,):
            raise ValueError("observed_y must match x")
        if count == 0:
            raise ValueError("powder data must contain at least one row")
        if not np.isfinite(self.x).all() or not np.isfinite(self.observed_y).all():
            raise ValueError("powder data must contain only finite values")
        if count > 1 and np.any(np.diff(self.x) <= 0.0):
            raise ValueError("powder coordinates must be strictly increasing")
        if self.uncertainty is not None:
            if self.uncertainty.dtype != np.float64 or self.uncertainty.shape != (count,):
                raise ValueError("uncertainty must match x")
            if not np.isfinite(self.uncertainty).all() or np.any(self.uncertainty <= 0.0):
                raise ValueError("uncertainty must contain only finite positive values")
        if self.mask is not None and (self.mask.dtype != np.bool_ or self.mask.shape != (count,)):
            raise ValueError("mask must be a one-dimensional boolean array matching x")
        if self.format not in {"columns", "gsas_fxye", "gsas_std"}:
            raise ValueError("unsupported powder-data format")
        if self.format in {"gsas_fxye", "gsas_std"} and self.bank is None:
            raise ValueError("GSAS powder data must identify its bank")
        for array in (self.x, self.observed_y, self.uncertainty, self.mask):
            if array is not None:
                array.flags.writeable = False

    def to_pattern(self, *, background: object = None, mask: object = None) -> PowderPattern:
        """Construct the normal script-facing pattern model."""

        return PowderPattern(
            self.x,
            observed_y=self.observed_y,
            uncertainty=self.uncertainty,
            background=background,
            mask=self.mask if mask is None else mask,
        )


@dataclass(frozen=True, slots=True)
class TofPowderData:
    """One reduced TOF pattern with an immutable microsecond axis."""

    tof_us: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    uncertainty: NDArray[np.float64] | None
    format: Literal["columns", "gsas_slog_fxye", "gsas_const_std"]
    source_name: str | None
    bank: int | None
    logarithmic_grid: bool
    mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        count = self.tof_us.size
        if self.tof_us.dtype != np.float64 or self.tof_us.shape != (count,):
            raise ValueError("tof_us must be a one-dimensional float64 array")
        if self.observed_y.dtype != np.float64 or self.observed_y.shape != (count,):
            raise ValueError("observed_y must match tof_us")
        if count == 0:
            raise ValueError("TOF powder data must contain at least one row")
        if not np.isfinite(self.tof_us).all() or not np.isfinite(self.observed_y).all():
            raise ValueError("TOF powder data must contain only finite values")
        if count > 1 and np.any(np.diff(self.tof_us) <= 0.0):
            raise ValueError("TOF coordinates must be strictly increasing")
        if self.format not in {"columns", "gsas_slog_fxye", "gsas_const_std"}:
            raise ValueError("unsupported TOF powder-data format")
        if self.format == "columns" and self.bank is not None:
            raise ValueError("plain TOF columns must not identify a GSAS bank")
        if self.format != "columns" and (
            isinstance(self.bank, bool) or not isinstance(self.bank, int) or self.bank <= 0
        ):
            raise ValueError("GSAS TOF data must identify a positive bank")
        if not isinstance(self.logarithmic_grid, bool):
            raise TypeError("logarithmic_grid must be a bool")
        if self.uncertainty is not None:
            if self.uncertainty.dtype != np.float64 or self.uncertainty.shape != (count,):
                raise ValueError("uncertainty must match tof_us")
            if not np.isfinite(self.uncertainty).all() or np.any(self.uncertainty <= 0.0):
                raise ValueError("uncertainty must contain only finite positive values")
        if self.mask is not None and (
            self.mask.dtype != np.bool_ or self.mask.shape != (count,)
        ):
            raise ValueError("mask must be a one-dimensional boolean array matching tof_us")
        for array in (self.tof_us, self.observed_y, self.uncertainty, self.mask):
            if array is not None:
                array.flags.writeable = False

    def to_pattern(self, *, background: object = None, mask: object = None) -> TofPowderPattern:
        """Construct the script-facing microsecond-domain pattern model."""

        return TofPowderPattern(
            self.tof_us,
            observed_y=self.observed_y,
            uncertainty=self.uncertainty,
            background=background,
            mask=self.mask if mask is None else mask,
        )


def read_powder_data(
    path_or_text: str | Path,
    *,
    format: PowderFormat = "auto",
    bank: int = 1,
    limits: PowderReadLimits | None = None,
) -> PowderData:
    """Read columns or an unpacked/packed constant-wavelength GSAS bank.

    Plain columns are ``x, observed_y[, uncertainty]`` in caller-selected
    coordinate units. GSAS FXYE stores constant-wavelength coordinates in
    centidegrees; this adapter exposes degrees. The packed ``CONST ... STD``
    form stores ten fixed-width normalization/intensity records per line.
    Other packed GSAS encodings are rejected instead of being guessed.
    """

    selected_limits = limits or PowderReadLimits()
    if not isinstance(selected_limits, PowderReadLimits):
        raise TypeError("limits must be PowderReadLimits")
    if format not in {"auto", "columns", "gsas_fxye", "gsas_std"}:
        raise ValueError("format must be 'auto', 'columns', 'gsas_fxye', or 'gsas_std'")
    if isinstance(bank, bool) or not isinstance(bank, int) or bank <= 0:
        raise ValueError("bank must be a positive integer")
    native_arguments = (
        format,
        bank,
        selected_limits.max_bytes,
        selected_limits.max_rows,
    )
    if isinstance(path_or_text, Path):
        arrays = _core._read_powder_file(str(path_or_text), *native_arguments)
        return PowderData(*arrays)
    if not isinstance(path_or_text, str):
        raise TypeError("path_or_text must be a string or pathlib.Path")
    if "\n" in path_or_text or "\r" in path_or_text:
        arrays = _core._parse_powder_text(path_or_text, *native_arguments)
        return PowderData(*arrays)
    candidate = Path(path_or_text)
    try:
        if candidate.exists():
            arrays = _core._read_powder_file(str(candidate), *native_arguments)
            return PowderData(*arrays)
    except OSError:
        pass
    arrays = _core._parse_powder_text(path_or_text, *native_arguments)
    return PowderData(*arrays)


def read_tof_powder_data(
    path_or_text: str | Path,
    *,
    format: TofPowderFormat = "auto",
    bank: int = 1,
    limits: PowderReadLimits | None = None,
) -> TofPowderData:
    """Read reduced TOF data without angle-domain conversion.

    Plain columns are ``bin_center_us, intensity_density[, uncertainty]``.
    GSAS SLOG/FXYE and CONST/STD inputs are converted from bin boundaries and
    width-integrated values into that same center/density convention.
    """

    selected_limits = limits or PowderReadLimits()
    if not isinstance(selected_limits, PowderReadLimits):
        raise TypeError("limits must be PowderReadLimits")
    if format not in {"auto", "columns", "gsas_slog_fxye", "gsas_const_std"}:
        raise ValueError(
            "format must be 'auto', 'columns', 'gsas_slog_fxye', or 'gsas_const_std'"
        )
    if isinstance(bank, bool) or not isinstance(bank, int) or bank <= 0:
        raise ValueError("bank must be a positive integer")
    native_arguments = (format, bank, selected_limits.max_bytes, selected_limits.max_rows)
    if isinstance(path_or_text, Path):
        return TofPowderData(*_core._read_tof_powder_file(str(path_or_text), *native_arguments))
    if not isinstance(path_or_text, str):
        raise TypeError("path_or_text must be a string or pathlib.Path")
    if "\n" in path_or_text or "\r" in path_or_text:
        return TofPowderData(*_core._parse_tof_powder_text(path_or_text, *native_arguments))
    candidate = Path(path_or_text)
    try:
        if candidate.exists():
            return TofPowderData(*_core._read_tof_powder_file(str(candidate), *native_arguments))
    except OSError:
        pass
    return TofPowderData(*_core._parse_tof_powder_text(path_or_text, *native_arguments))
