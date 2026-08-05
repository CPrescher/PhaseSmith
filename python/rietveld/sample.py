"""Built-in vectorized sample-broadening providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .extensions import PhysicsContext, PhysicsContribution, ProviderDescriptor

_DEG_PER_RAD = 180.0 / np.pi
_HALF_ANGLE_RAD_PER_DEG = np.pi / 360.0


def _width_only_contribution(
    gaussian_variance_deg2: np.ndarray,
    lorentzian_fwhm_deg: np.ndarray,
    d_gaussian_d_position: np.ndarray,
    d_lorentzian_d_position: np.ndarray,
    parameter_name: str,
    d_gaussian_d_parameter: np.ndarray,
    d_lorentzian_d_parameter: np.ndarray,
) -> PhysicsContribution:
    count = gaussian_variance_deg2.size
    zeros = np.zeros(count, dtype=np.float64)
    return PhysicsContribution(
        gaussian_variance_deg2=gaussian_variance_deg2,
        lorentzian_fwhm_deg=lorentzian_fwhm_deg,
        intensity_multiplier=np.ones(count, dtype=np.float64),
        d_gaussian_variance_d_position=d_gaussian_d_position,
        d_lorentzian_fwhm_d_position=d_lorentzian_d_position,
        d_intensity_multiplier_d_position=zeros,
        parameter_names=(parameter_name,),
        d_gaussian_variance_d_parameters=d_gaussian_d_parameter[None, :],
        d_lorentzian_fwhm_d_parameters=d_lorentzian_d_parameter[None, :],
        d_intensity_multiplier_d_parameters=zeros[None, :],
    )


@dataclass(frozen=True, slots=True)
class IsotropicSizeBroadening:
    """Lorentzian Scherrer broadening for one coherent-domain size."""

    crystallite_size_nm: float
    shape_factor: float = 0.9
    descriptor: ClassVar[ProviderDescriptor] = ProviderDescriptor("rietveld.isotropic-size", "1")

    def __post_init__(self) -> None:
        """Validate the coherent-domain and fixed shape-factor convention."""

        if np.isnan(self.crystallite_size_nm) or self.crystallite_size_nm <= 0.0:
            raise ValueError("crystallite_size_nm must be positive or positive infinity")
        if not np.isfinite(self.shape_factor) or self.shape_factor <= 0.0:
            raise ValueError("shape_factor must be positive and finite")

    def evaluate(self, context: PhysicsContext) -> PhysicsContribution:
        """Evaluate Scherrer FWHM and chains for all reflection positions."""

        theta = context.reflections.two_theta_deg * _HALF_ANGLE_RAD_PER_DEG
        count = context.reflections.reflection_count
        zeros = np.zeros(count, dtype=np.float64)
        if np.isinf(self.crystallite_size_nm):
            lorentzian = zeros
            d_size = zeros
            d_position = zeros
        else:
            scale = (
                _DEG_PER_RAD
                * self.shape_factor
                * context.instrument.wavelength_angstrom
                / (10.0 * self.crystallite_size_nm)
            )
            lorentzian = scale / np.cos(theta)
            d_size = -lorentzian / self.crystallite_size_nm
            d_position = lorentzian * _HALF_ANGLE_RAD_PER_DEG * np.tan(theta)
        return _width_only_contribution(
            zeros,
            lorentzian,
            zeros,
            d_position,
            "isotropic_size.crystallite_size_nm",
            zeros,
            d_size,
        )


@dataclass(frozen=True, slots=True)
class IsotropicMicrostrainBroadening:
    """Gaussian broadening from dimensionless RMS ``delta d / d``."""

    rms_microstrain: float
    descriptor: ClassVar[ProviderDescriptor] = ProviderDescriptor(
        "rietveld.isotropic-microstrain", "1"
    )

    def __post_init__(self) -> None:
        """Reject negative or non-finite RMS strain."""

        if not np.isfinite(self.rms_microstrain) or self.rms_microstrain < 0.0:
            raise ValueError("rms_microstrain must be non-negative and finite")

    def evaluate(self, context: PhysicsContext) -> PhysicsContribution:
        """Evaluate Gaussian variance and analytical chains for every reflection."""

        theta = context.reflections.two_theta_deg * _HALF_ANGLE_RAD_PER_DEG
        tangent = np.tan(theta)
        secant_squared = 1.0 / np.cos(theta) ** 2
        coefficient = (2.0 * _DEG_PER_RAD) ** 2
        strain = self.rms_microstrain
        variance = coefficient * strain**2 * tangent**2
        d_strain = 2.0 * coefficient * strain * tangent**2
        d_position = (
            2.0 * coefficient * strain**2 * tangent * secant_squared * _HALF_ANGLE_RAD_PER_DEG
        )
        zeros = np.zeros(context.reflections.reflection_count, dtype=np.float64)
        return _width_only_contribution(
            variance,
            zeros,
            d_position,
            zeros,
            "isotropic_microstrain.rms",
            d_strain,
            zeros,
        )
