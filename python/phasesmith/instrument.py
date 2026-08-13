"""Typed instrument models for scripted calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .pattern import TofPowderPattern


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


@dataclass(frozen=True, slots=True, init=False)
class TofIncidentSpectrumEvaluation:
    """Immutable incident intensities and analytical TOF derivatives."""

    values: NDArray[np.float64]
    d_values_d_tof_us: NDArray[np.float64]

    def __init__(self, values: ArrayLike, d_values_d_tof_us: ArrayLike) -> None:
        intensity = np.array(values, dtype=np.float64, copy=True, order="C")
        derivative = np.array(d_values_d_tof_us, dtype=np.float64, copy=True, order="C")
        if (
            intensity.ndim != 1
            or derivative.shape != intensity.shape
            or not np.isfinite(intensity).all()
            or not np.isfinite(derivative).all()
            or np.any(intensity <= 0.0)
        ):
            raise ValueError("incident values must be positive finite matching vectors")
        intensity.flags.writeable = False
        derivative.flags.writeable = False
        object.__setattr__(self, "values", intensity)
        object.__setattr__(self, "d_values_d_tof_us", derivative)


@dataclass(frozen=True, slots=True, init=False)
class TofIncidentSpectrum:
    """Facility-neutral Maxwellian-plus-Chebyshev TOF spectrum.

    With ``t`` in milliseconds and ``x = 2/t - 1``, the twelve coefficients
    ``P1..P12`` define ``P1 + P2*t^-5*exp(-P3/t^2)`` plus
    ``sum(Pj*T_(j-3)(x), j=4..12)``. Public TOF values and the inclusive
    calibration interval are in microseconds.
    """

    min_tof_us: float
    max_tof_us: float
    coefficients: tuple[float, ...]

    def __init__(self, min_tof_us: float, max_tof_us: float, coefficients: ArrayLike) -> None:
        fitted = np.array(coefficients, dtype=np.float64, copy=True, order="C")
        if (
            not np.isfinite(min_tof_us)
            or not np.isfinite(max_tof_us)
            or min_tof_us <= 0.0
            or max_tof_us <= min_tof_us
        ):
            raise ValueError("incident-spectrum TOF range must be finite, positive, and increasing")
        if fitted.shape != (12,) or not np.isfinite(fitted).all():
            raise ValueError("incident-spectrum coefficients must contain 12 finite values")
        object.__setattr__(self, "min_tof_us", float(min_tof_us))
        object.__setattr__(self, "max_tof_us", float(max_tof_us))
        object.__setattr__(self, "coefficients", tuple(float(value) for value in fitted))

    def evaluate(self, tof_us: ArrayLike) -> TofIncidentSpectrumEvaluation:
        """Evaluate positive intensities and ``dI/d(tof_us)`` together."""

        time_us = np.array(tof_us, dtype=np.float64, copy=True, order="C")
        if (
            time_us.ndim != 1
            or not np.isfinite(time_us).all()
            or np.any(time_us < self.min_tof_us)
            or np.any(time_us > self.max_tof_us)
        ):
            raise ValueError("tof_us must be a finite vector within the calibration interval")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        time_ms = time_us / 1_000.0
        inverse_time = 1.0 / time_ms
        inverse_time2 = inverse_time * inverse_time
        x = 2.0 * inverse_time - 1.0
        d_x_d_time_ms = -2.0 * inverse_time2
        maxwell = coefficients[1] * inverse_time**5 * np.exp(-coefficients[2] * inverse_time2)
        values = coefficients[0] + maxwell
        derivatives_ms = maxwell * (-5.0 * inverse_time + 2.0 * coefficients[2] * inverse_time**3)
        previous = np.ones_like(x)
        d_previous = np.zeros_like(x)
        current = x.copy()
        d_current = d_x_d_time_ms.copy()
        for index, coefficient in enumerate(coefficients[3:]):
            if index > 0:
                next_polynomial = 2.0 * x * current - previous
                next_derivative = 2.0 * (d_x_d_time_ms * current + x * d_current) - d_previous
                previous, current = current, next_polynomial
                d_previous, d_current = d_current, next_derivative
            values = values + coefficient * current
            derivatives_ms = derivatives_ms + coefficient * d_current
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("incident-spectrum intensity must be positive and finite")
        return TofIncidentSpectrumEvaluation(values, derivatives_ms / 1_000.0)

    def normalize_pattern(self, pattern: TofPowderPattern) -> TofPowderPattern:
        """Divide observed values, uncertainties, and background by the spectrum."""

        from .pattern import TofPowderPattern

        if not isinstance(pattern, TofPowderPattern) or pattern.observed_y is None:
            raise TypeError("pattern must be an observed TofPowderPattern")
        intensity = self.evaluate(pattern.tof_us).values
        return TofPowderPattern(
            pattern.tof_us,
            observed_y=pattern.observed_y / intensity,
            uncertainty=None if pattern.uncertainty is None else pattern.uncertainty / intensity,
            mask=pattern.mask,
            background=pattern.background / intensity,
        )


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
