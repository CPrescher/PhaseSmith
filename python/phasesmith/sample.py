"""Built-in vectorized sample-broadening providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .extensions import PhysicsContext, PhysicsContribution, ProviderDescriptor
from .phase import ReciprocalMetric

_DEG_PER_RAD = 180.0 / np.pi
_HALF_ANGLE_RAD_PER_DEG = np.pi / 360.0
_CELL_PARAMETER_NAMES = (
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
)


@dataclass(frozen=True, slots=True)
class ReciprocalAngleGeometry:
    """Squared reciprocal-space angle cosine and preferred-axis chains."""

    cosine_squared: NDArray[np.float64]
    d_cosine_squared_d_axis_hkl: NDArray[np.float64]


def reciprocal_angle_geometry(
    hkl: ArrayLike,
    preferred_axis_hkl: ArrayLike,
    reciprocal_metric: ReciprocalMetric,
) -> ReciprocalAngleGeometry:
    """Evaluate reciprocal-metric angles and axis-coordinate derivatives."""

    reflections = np.asarray(hkl, dtype=np.float64)
    axis = np.asarray(preferred_axis_hkl, dtype=np.float64)
    if reflections.ndim != 2 or reflections.shape[1] != 3 or not np.isfinite(reflections).all():
        raise ValueError("hkl must be a finite array with shape (reflection_count, 3)")
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("preferred_axis_hkl must contain three finite coordinates")
    metric = reciprocal_metric.matrix
    reflection_metric = reflections @ metric
    axis_metric = metric @ axis
    reflection_norm_squared = np.einsum("ij,ij->i", reflection_metric, reflections)
    axis_norm_squared = float(axis @ axis_metric)
    if axis_norm_squared <= 0.0:
        raise ValueError("preferred_axis_hkl must be non-zero")
    if np.any(reflection_norm_squared <= 0.0):
        raise ValueError("hkl must not contain the zero reflection")
    projection = reflection_metric @ axis
    denominator = reflection_norm_squared * axis_norm_squared
    cosine_squared = projection**2 / denominator
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any((cosine_squared < -tolerance) | (cosine_squared > 1.0 + tolerance)):
        raise ValueError("reciprocal metric produced an invalid reflection angle")
    cosine_squared = np.clip(cosine_squared, 0.0, 1.0)
    derivative = (
        2.0 * projection[:, None] * reflection_metric / denominator[:, None]
        - 2.0 * cosine_squared[:, None] * axis_metric[None, :] / axis_norm_squared
    )
    cosine_squared = np.ascontiguousarray(cosine_squared)
    derivative = np.ascontiguousarray(derivative)
    cosine_squared.flags.writeable = False
    derivative.flags.writeable = False
    return ReciprocalAngleGeometry(cosine_squared, derivative)


def _reciprocal_metric_cell_derivatives(context: PhysicsContext) -> NDArray[np.float64]:
    cell = context.unit_cell
    if cell is None:
        return np.empty((0, 3, 3), dtype=np.float64)
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell.as_tuple()
    alpha, beta, gamma = np.radians((alpha_deg, beta_deg, gamma_deg))
    direct_derivatives = np.zeros((6, 3, 3), dtype=np.float64)
    direct_derivatives[0] = (
        (2.0 * a, b * np.cos(gamma), c * np.cos(beta)),
        (b * np.cos(gamma), 0.0, 0.0),
        (c * np.cos(beta), 0.0, 0.0),
    )
    direct_derivatives[1] = (
        (0.0, a * np.cos(gamma), 0.0),
        (a * np.cos(gamma), 2.0 * b, c * np.cos(alpha)),
        (0.0, c * np.cos(alpha), 0.0),
    )
    direct_derivatives[2] = (
        (0.0, 0.0, a * np.cos(beta)),
        (0.0, 0.0, b * np.cos(alpha)),
        (a * np.cos(beta), b * np.cos(alpha), 2.0 * c),
    )
    per_degree = np.pi / 180.0
    direct_derivatives[3, 1, 2] = direct_derivatives[3, 2, 1] = (
        -b * c * np.sin(alpha) * per_degree
    )
    direct_derivatives[4, 0, 2] = direct_derivatives[4, 2, 0] = (
        -a * c * np.sin(beta) * per_degree
    )
    direct_derivatives[5, 0, 1] = direct_derivatives[5, 1, 0] = (
        -a * b * np.sin(gamma) * per_degree
    )
    reciprocal = cell.geometry().reciprocal_metric
    return np.ascontiguousarray(
        np.asarray([-reciprocal @ derivative @ reciprocal for derivative in direct_derivatives])
    )


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
    descriptor: ClassVar[ProviderDescriptor] = ProviderDescriptor("phasesmith.isotropic-size", "1")

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
        "phasesmith.isotropic-microstrain", "1"
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


@dataclass(frozen=True, slots=True)
class MarchDollasePreferredOrientation:
    """March--Dollase integrated-intensity correction for one fixed axis."""

    march_ratio: float
    preferred_axis_hkl: tuple[float, float, float]
    reciprocal_metric: ReciprocalMetric
    descriptor: ClassVar[ProviderDescriptor] = ProviderDescriptor("phasesmith.march-dollase", "1")

    def __post_init__(self) -> None:
        """Validate the positive ratio and persistence-safe axis tuple."""

        if not np.isfinite(self.march_ratio) or self.march_ratio <= 0.0:
            raise ValueError("march_ratio must be positive and finite")
        axis = np.asarray(self.preferred_axis_hkl, dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.all(axis == 0.0):
            raise ValueError("preferred_axis_hkl must contain a non-zero finite triplet")
        if not isinstance(self.reciprocal_metric, ReciprocalMetric):
            raise TypeError("reciprocal_metric must be a ReciprocalMetric")
        object.__setattr__(self, "preferred_axis_hkl", tuple(float(value) for value in axis))

    def evaluate(self, context: PhysicsContext) -> PhysicsContribution:
        """Evaluate reflection multipliers and the March-ratio chain."""

        reciprocal_metric = (
            self.reciprocal_metric
            if context.unit_cell is None
            else ReciprocalMetric(context.unit_cell.geometry().reciprocal_metric)
        )
        geometry = reciprocal_angle_geometry(
            context.reflections.hkl,
            self.preferred_axis_hkl,
            reciprocal_metric,
        )
        cosine_squared = geometry.cosine_squared
        sine_squared = 1.0 - cosine_squared
        ratio = self.march_ratio
        denominator = ratio**2 * cosine_squared + sine_squared / ratio
        multiplier = denominator ** (-1.5)
        d_denominator = 2.0 * ratio * cosine_squared - sine_squared / ratio**2
        d_ratio = -1.5 * denominator ** (-2.5) * d_denominator
        metric_derivatives = _reciprocal_metric_cell_derivatives(context)
        cell_multiplier_derivatives = []
        if metric_derivatives.size:
            reflections = np.asarray(context.reflections.hkl, dtype=np.float64)
            axis = np.asarray(self.preferred_axis_hkl, dtype=np.float64)
            metric = reciprocal_metric.matrix
            reflection_norm = np.einsum("ri,ij,rj->r", reflections, metric, reflections)
            axis_norm = float(axis @ metric @ axis)
            projection = np.einsum("ri,ij,j->r", reflections, metric, axis)
            d_multiplier_d_cosine = (
                -1.5 * denominator ** (-2.5) * (ratio**2 - 1.0 / ratio)
            )
            for derivative in metric_derivatives:
                d_reflection_norm = np.einsum(
                    "ri,ij,rj->r", reflections, derivative, reflections
                )
                d_axis_norm = float(axis @ derivative @ axis)
                d_projection = np.einsum("ri,ij,j->r", reflections, derivative, axis)
                d_cosine = (
                    2.0 * projection * d_projection / (reflection_norm * axis_norm)
                    - cosine_squared
                    * (d_reflection_norm / reflection_norm + d_axis_norm / axis_norm)
                )
                cell_multiplier_derivatives.append(d_multiplier_d_cosine * d_cosine)
        count = context.reflections.reflection_count
        zeros = np.zeros(count, dtype=np.float64)
        parameter_names = (
            "march_dollase.ratio",
            *tuple(f"march_dollase.cell.{name}" for name in _CELL_PARAMETER_NAMES)[
                : len(cell_multiplier_derivatives)
            ],
        )
        intensity_derivatives = np.vstack((d_ratio, *cell_multiplier_derivatives))
        parameter_count = len(parameter_names)
        return PhysicsContribution(
            gaussian_variance_deg2=zeros,
            lorentzian_fwhm_deg=zeros,
            intensity_multiplier=multiplier,
            d_gaussian_variance_d_position=zeros,
            d_lorentzian_fwhm_d_position=zeros,
            d_intensity_multiplier_d_position=zeros,
            parameter_names=parameter_names,
            d_gaussian_variance_d_parameters=np.zeros((parameter_count, count)),
            d_lorentzian_fwhm_d_parameters=np.zeros((parameter_count, count)),
            d_intensity_multiplier_d_parameters=intensity_derivatives,
        )
