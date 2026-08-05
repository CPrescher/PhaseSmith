"""Pinned-oracle unit translations into public physical models."""

from __future__ import annotations

from ..instrument import ConstantWavelengthInstrument


def cw_instrument_from_gsas_centidegrees(
    *,
    wavelength_angstrom: float,
    u: float,
    v: float,
    w: float,
    x: float,
    y: float,
) -> ConstantWavelengthInstrument:
    """Convert GSAS-style U/V/W/X/Y coefficients to public degree units."""

    return ConstantWavelengthInstrument(
        wavelength_angstrom=float(wavelength_angstrom),
        u_deg2=float(u) / 10_000.0,
        v_deg2=float(v) / 10_000.0,
        w_deg2=float(w) / 10_000.0,
        x_deg=float(x) / 100.0,
        y_deg=float(y) / 100.0,
    )
