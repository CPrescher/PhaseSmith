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


@dataclass(frozen=True, slots=True)
class FcjGeometry:
    """Finger-Cox-Jephcoat axial geometry as dimensionless half-heights."""

    sample_over_radius: float
    detector_over_radius: float

    def __post_init__(self) -> None:
        """Reject nonphysical geometry before entering a calculation."""

        values = (self.sample_over_radius, self.detector_over_radius)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("FCJ axial ratios must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("FCJ axial ratios must be non-negative")


@dataclass(frozen=True, slots=True)
class TofBankGeometry:
    """Facility-neutral fixed geometry for one focused TOF detector bank."""

    two_theta_deg: float

    def __post_init__(self) -> None:
        """Require a finite scattering angle strictly inside the physical domain."""

        if not np.isfinite(self.two_theta_deg) or not 0.0 < self.two_theta_deg < 180.0:
            raise ValueError("two_theta_deg must be finite and strictly within (0, 180)")

    @property
    def theta_radians(self) -> float:
        """Return the half scattering angle in radians."""

        return float(np.deg2rad(0.5 * self.two_theta_deg))


@dataclass(frozen=True, slots=True)
class TofInstrument:
    """TOF calibration and d-dependent profile coefficients in public units."""

    zero_us: float
    difc_us_per_angstrom: float
    difa_us_per_angstrom2: float
    difb_us_angstrom: float
    alpha_coefficient: float
    beta0_per_us: float
    beta1_angstrom4_per_us: float
    betaq_angstrom2_per_us: float
    sigma0_us2: float
    sigma1_us2_per_angstrom2: float
    sigma2_us2_per_angstrom4: float
    sigmaq_us2_per_angstrom: float
    x_us_per_angstrom: float
    y_us_per_angstrom2: float
    z_us: float

    def __post_init__(self) -> None:
        """Reject non-finite coefficients and non-positive linear calibration."""

        if not all(np.isfinite(value) for value in self.as_tuple()):
            raise ValueError("TOF instrument coefficients must be finite")
        if self.difc_us_per_angstrom <= 0.0:
            raise ValueError("difc_us_per_angstrom must be positive")

    def as_tuple(self) -> tuple[float, ...]:
        """Return coefficients in the stable native/global-derivative order."""

        return (
            self.zero_us,
            self.difc_us_per_angstrom,
            self.difa_us_per_angstrom2,
            self.difb_us_angstrom,
            self.alpha_coefficient,
            self.beta0_per_us,
            self.beta1_angstrom4_per_us,
            self.betaq_angstrom2_per_us,
            self.sigma0_us2,
            self.sigma1_us2_per_angstrom2,
            self.sigma2_us2_per_angstrom4,
            self.sigmaq_us2_per_angstrom,
            self.x_us_per_angstrom,
            self.y_us_per_angstrom2,
            self.z_us,
        )
