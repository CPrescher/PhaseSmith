"""Small, deterministic readers for common text powder-pattern formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..pattern import PowderPattern

PowderFormat = Literal["auto", "columns", "gsas_fxye"]


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
    format: Literal["columns", "gsas_fxye"]
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
        if self.format not in {"columns", "gsas_fxye"}:
            raise ValueError("unsupported powder-data format")
        if self.format == "gsas_fxye" and self.bank is None:
            raise ValueError("GSAS FXYE data must identify its bank")
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
    """Read two/three-column text or a constant-wavelength GSAS FXYE bank.

    Plain columns are ``x, observed_y[, uncertainty]`` in caller-selected
    coordinate units. GSAS FXYE stores constant-wavelength coordinates in
    centidegrees; this adapter exposes degrees. Packed GSAS formats are
    intentionally rejected instead of being guessed.
    """

    selected_limits = limits or PowderReadLimits()
    if not isinstance(selected_limits, PowderReadLimits):
        raise TypeError("limits must be PowderReadLimits")
    if format not in {"auto", "columns", "gsas_fxye"}:
        raise ValueError("format must be 'auto', 'columns', or 'gsas_fxye'")
    if isinstance(bank, bool) or not isinstance(bank, int) or bank <= 0:
        raise ValueError("bank must be a positive integer")
    text, source_name = _read_source(path_or_text, selected_limits.max_bytes)
    selected_format = _detect_format(text, source_name) if format == "auto" else format
    if selected_format == "columns":
        return _read_columns(text, source_name, selected_limits.max_rows)
    return _read_gsas_fxye(text, source_name, bank, selected_limits.max_rows)


def _read_source(path_or_text: str | Path, max_bytes: int) -> tuple[str, str | None]:
    if isinstance(path_or_text, Path):
        return _read_path(path_or_text, max_bytes)
    if not isinstance(path_or_text, str):
        raise TypeError("path_or_text must be a string or pathlib.Path")
    if "\n" in path_or_text or "\r" in path_or_text:
        return _validate_text_size(path_or_text, max_bytes), None
    candidate = Path(path_or_text)
    try:
        if candidate.exists():
            return _read_path(candidate, max_bytes)
    except OSError:
        pass
    return _validate_text_size(path_or_text, max_bytes), None


def _read_path(path: Path, max_bytes: int) -> tuple[str, str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"powder file exceeds max_bytes: {size} > {max_bytes}")
    return path.read_text(encoding="utf-8"), str(path)


def _validate_text_size(text: str, max_bytes: int) -> str:
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("powder text exceeds max_bytes")
    return text


def _detect_format(text: str, source_name: str | None) -> Literal["columns", "gsas_fxye"]:
    if any(line.lstrip().upper().startswith("BANK ") for line in text.splitlines()):
        return "gsas_fxye"
    if source_name is not None and Path(source_name).suffix.lower() == ".fxye":
        return "gsas_fxye"
    return "columns"


def _numeric_rows(text: str, *, expected_columns: int | None, max_rows: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "!", ";")):
            continue
        for marker in ("#", "!"):
            line = line.split(marker, maxsplit=1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if expected_columns is None:
            expected_columns = len(fields)
            if expected_columns not in {2, 3}:
                raise ValueError(f"line {line_number}: expected two or three columns")
        if len(fields) != expected_columns:
            raise ValueError(
                f"line {line_number}: expected {expected_columns} columns, got {len(fields)}"
            )
        try:
            rows.append([float(field) for field in fields])
        except ValueError as error:
            raise ValueError(f"line {line_number}: non-numeric powder value") from error
        if len(rows) > max_rows:
            raise ValueError(f"powder data exceeds max_rows: {max_rows}")
    if not rows:
        raise ValueError("powder data contains no numeric rows")
    return rows


def _read_columns(text: str, source_name: str | None, max_rows: int) -> PowderData:
    rows = _numeric_rows(text, expected_columns=None, max_rows=max_rows)
    data = np.asarray(rows, dtype=np.float64)
    uncertainty = None if data.shape[1] == 2 else np.array(data[:, 2], copy=True)
    return PowderData(
        x=np.array(data[:, 0], copy=True),
        observed_y=np.array(data[:, 1], copy=True),
        uncertainty=uncertainty,
        format="columns",
        source_name=source_name,
    )


def _read_gsas_fxye(
    text: str, source_name: str | None, selected_bank: int, max_rows: int
) -> PowderData:
    bank_lines: dict[int, list[str]] = {}
    active_bank: int | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line.upper().startswith("BANK "):
            fields = line.split()
            try:
                active_bank = int(fields[1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"line {line_number}: invalid GSAS bank header") from error
            if fields[-1].upper() != "FXYE":
                raise ValueError(f"line {line_number}: only unpacked GSAS FXYE banks are supported")
            if active_bank in bank_lines:
                raise ValueError(f"line {line_number}: duplicate GSAS bank {active_bank}")
            bank_lines[active_bank] = []
        elif active_bank is not None:
            bank_lines[active_bank].append(raw_line)
    if selected_bank not in bank_lines:
        available = ", ".join(str(value) for value in sorted(bank_lines)) or "none"
        raise ValueError(f"GSAS bank {selected_bank} not found; available banks: {available}")
    rows = _numeric_rows(
        "\n".join(bank_lines[selected_bank]), expected_columns=3, max_rows=max_rows
    )
    data = np.asarray(rows, dtype=np.float64)
    return PowderData(
        x=np.array(data[:, 0] / 100.0, copy=True),
        observed_y=np.array(data[:, 1], copy=True),
        uncertainty=np.array(data[:, 2], copy=True),
        format="gsas_fxye",
        source_name=source_name,
        bank=selected_bank,
    )
