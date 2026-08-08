"""Small, deterministic readers for common text powder-pattern formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..pattern import PowderPattern

PowderFormat = Literal["auto", "columns", "gsas_fxye", "gsas_std"]


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
        if self.format not in {"columns", "gsas_fxye", "gsas_std"}:
            raise ValueError("unsupported powder-data format")
        if self.format in {"gsas_fxye", "gsas_std"} and self.bank is None:
            raise ValueError("GSAS powder data must identify its bank")
        for array in (self.x, self.observed_y, self.uncertainty):
            if array is not None:
                array.flags.writeable = False

    def to_pattern(self, *, background: object = None, mask: object = None) -> PowderPattern:
        """Construct the normal script-facing pattern model."""

        return PowderPattern(
            self.x,
            observed_y=self.observed_y,
            uncertainty=self.uncertainty,
            background=background,
            mask=mask,
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
