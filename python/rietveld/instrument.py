"""Typed instrument models for scripted calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ConstantWavelengthInstrument:
    """Constant-wavelength U/V/W/X/Y profile parameters in physical units.

    ``u_deg2``, ``v_deg2``, and ``w_deg2`` produce Gaussian variance in
    degrees squared. ``x_deg`` and ``y_deg`` produce Lorentzian FWHM in
    degrees. GSAS-II stored-unit conversion belongs in the oracle adapter.
    """

    wavelength_angstrom: float
    u_deg2: float
    v_deg2: float
    w_deg2: float
    x_deg: float
    y_deg: float

    def __post_init__(self) -> None:
        """Reject invalid scalar metadata before entering a calculation."""

        values = (
            self.wavelength_angstrom,
            self.u_deg2,
            self.v_deg2,
            self.w_deg2,
            self.x_deg,
            self.y_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("constant-wavelength instrument parameters must be finite")
        if self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive")
