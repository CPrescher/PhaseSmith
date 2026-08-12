"""Bounded legacy GSAS TOF calibration import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import _core
from ..instrument import TofBankGeometry, TofIncidentSpectrum, TofInstrument


@dataclass(frozen=True, slots=True)
class GsasTofInstrumentReadLimits:
    """Resource limit applied before decoding a legacy calibration file."""

    max_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")


@dataclass(frozen=True, slots=True)
class GsasTofInstrumentData:
    """One translated profile-function-1 or -3 bank plus source metadata."""

    instrument: TofInstrument
    bank: int
    profile_function: int
    source_name: str | None = None
    bank_geometry: TofBankGeometry | None = None
    incident_spectrum: TofIncidentSpectrum | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, TofInstrument):
            raise TypeError("instrument must be a TofInstrument")
        if self.bank <= 0:
            raise ValueError("bank must be positive")
        if self.profile_function not in {1, 3}:
            raise ValueError("only GSAS TOF profile functions 1 and 3 are supported")
        if self.bank_geometry is not None and not isinstance(self.bank_geometry, TofBankGeometry):
            raise TypeError("bank_geometry must be a TofBankGeometry or None")
        if self.incident_spectrum is not None and not isinstance(
            self.incident_spectrum, TofIncidentSpectrum
        ):
            raise TypeError("incident_spectrum must be a TofIncidentSpectrum or None")


def read_gsas_tof_instrument(
    path_or_text: str | Path,
    *,
    bank: int = 1,
    limits: GsasTofInstrumentReadLimits | None = None,
) -> GsasTofInstrumentData:
    """Read one legacy GSAS profile-function-1 or -3 bank into the typed model."""

    selected_limits = limits or GsasTofInstrumentReadLimits()
    if not isinstance(selected_limits, GsasTofInstrumentReadLimits):
        raise TypeError("limits must be GsasTofInstrumentReadLimits")
    if isinstance(bank, bool) or not isinstance(bank, int) or not 1 <= bank <= 99:
        raise ValueError("bank must be an integer in 1..=99")
    native_arguments = (bank, selected_limits.max_bytes)
    if isinstance(path_or_text, Path):
        record = _core._read_gsas_tof_instrument_file(str(path_or_text), *native_arguments)
    elif not isinstance(path_or_text, str):
        raise TypeError("path_or_text must be a string or pathlib.Path")
    elif "\n" in path_or_text or "\r" in path_or_text:
        record = _core._parse_gsas_tof_instrument_text(path_or_text, *native_arguments)
    else:
        candidate = Path(path_or_text)
        try:
            is_path = candidate.exists()
        except OSError:
            is_path = False
        if is_path:
            record = _core._read_gsas_tof_instrument_file(str(candidate), *native_arguments)
        else:
            record = _core._parse_gsas_tof_instrument_text(path_or_text, *native_arguments)
    (
        coefficients,
        selected_bank,
        profile_function,
        source_name,
        two_theta_deg,
        spectrum_record,
    ) = record
    return GsasTofInstrumentData(
        TofInstrument(*coefficients),
        int(selected_bank),
        int(profile_function),
        source_name,
        None if two_theta_deg is None else TofBankGeometry(float(two_theta_deg)),
        None
        if spectrum_record is None
        else TofIncidentSpectrum(
            float(spectrum_record[0]),
            float(spectrum_record[1]),
            spectrum_record[2],
        ),
    )
