"""Independent NumPy reference for the first profile vertical slice.

This implementation favors transparent equations over speed and must not call
the Rust extension. It is the first differential-testing layer for native code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FOUR_LN_2 = 4.0 * np.log(2.0)
GAUSSIAN_NORMALIZATION = np.sqrt(FOUR_LN_2 / np.pi)
PARAMETER_COUNT = 4
GAUSSIAN_FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Reference profile values and first derivatives."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_fwhm: NDArray[np.float64]
    d_eta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceTchShape:
    """Reference TCH width transform and component-width derivatives."""

    total_fwhm: float
    eta: float
    d_total_fwhm_d_gaussian_fwhm: float
    d_total_fwhm_d_lorentzian_fwhm: float
    d_eta_d_gaussian_fwhm: float
    d_eta_d_lorentzian_fwhm: float


@dataclass(frozen=True, slots=True)
class ReferenceTchProfile:
    """Reference TCH profile values and direct-input derivatives."""

    value: NDArray[np.float64]
    d_delta: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceCwProfileParameters:
    """Independent CW widths and derivatives for a reflection array."""

    gaussian_variance_deg2: NDArray[np.float64]
    gaussian_fwhm_deg: NDArray[np.float64]
    lorentzian_fwhm_deg: NDArray[np.float64]
    total_fwhm_deg: NDArray[np.float64]
    eta: NDArray[np.float64]
    d_gaussian_fwhm_d_instrument: NDArray[np.float64]
    d_lorentzian_fwhm_d_instrument: NDArray[np.float64]
    d_component_fwhm_d_two_theta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceFcjProfile:
    """FCJ-convolved TCH profile and direct-input derivatives."""

    value: NDArray[np.float64]
    d_position: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]
    d_sample_over_radius: NDArray[np.float64]
    d_detector_over_radius: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceTofProfile:
    """Independent truncated double-exponential TCH convolution."""

    value: NDArray[np.float64]
    d_position: NDArray[np.float64]
    d_alpha: NDArray[np.float64]
    d_beta: NDArray[np.float64]
    d_gaussian_fwhm: NDArray[np.float64]
    d_lorentzian_fwhm: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceTofProfileParameters:
    """Independent TOF calibration, rates, and component-width equations."""

    position_us: NDArray[np.float64]
    alpha_per_us: NDArray[np.float64]
    beta_per_us: NDArray[np.float64]
    gaussian_variance_us2: NDArray[np.float64]
    gaussian_fwhm_us: NDArray[np.float64]
    lorentzian_fwhm_us: NDArray[np.float64]


def profile(delta: ArrayLike, fwhm: float, eta: float) -> ReferenceProfile:
    """Evaluate the symmetric pseudo-Voigt directly from its component equations."""

    delta = np.asarray(delta, dtype=np.float64)
    inverse_fwhm = 1.0 / fwhm
    z_squared = np.square(delta * inverse_fwhm)

    gaussian = GAUSSIAN_NORMALIZATION * inverse_fwhm * np.exp(-FOUR_LN_2 * z_squared)
    denominator = 1.0 + 4.0 * z_squared
    lorentzian = 2.0 * inverse_fwhm / (np.pi * denominator)

    d_gaussian_delta = gaussian * (-2.0 * FOUR_LN_2 * delta * inverse_fwhm**2)
    d_lorentzian_delta = lorentzian * (-8.0 * delta * inverse_fwhm**2 / denominator)
    d_gaussian_fwhm = gaussian * inverse_fwhm * (-1.0 + 2.0 * FOUR_LN_2 * z_squared)
    d_lorentzian_fwhm = lorentzian * inverse_fwhm * (-1.0 + 8.0 * z_squared / denominator)
    return ReferenceProfile(
        value=eta * lorentzian + (1.0 - eta) * gaussian,
        d_delta=eta * d_lorentzian_delta + (1.0 - eta) * d_gaussian_delta,
        d_fwhm=eta * d_lorentzian_fwhm + (1.0 - eta) * d_gaussian_fwhm,
        d_eta=lorentzian - gaussian,
    )


def tch_shape_from_fwhm(gaussian_fwhm: float, lorentzian_fwhm: float) -> ReferenceTchShape:
    """Evaluate the published TCH transform directly from component FWHMs."""

    width_scale = max(float(gaussian_fwhm), float(lorentzian_fwhm))
    gaussian = float(gaussian_fwhm) / width_scale
    lorentzian = float(lorentzian_fwhm) / width_scale
    polynomial = (
        gaussian**5
        + 2.69269 * gaussian**4 * lorentzian
        + 2.42843 * gaussian**3 * lorentzian**2
        + 4.47163 * gaussian**2 * lorentzian**3
        + 0.07842 * gaussian * lorentzian**4
        + lorentzian**5
    )
    normalized_total_fwhm = polynomial**0.2
    total_fwhm = width_scale * normalized_total_fwhm
    d_polynomial_d_gaussian = (
        5.0 * gaussian**4
        + 4.0 * 2.69269 * gaussian**3 * lorentzian
        + 3.0 * 2.42843 * gaussian**2 * lorentzian**2
        + 2.0 * 4.47163 * gaussian * lorentzian**3
        + 0.07842 * lorentzian**4
    )
    d_polynomial_d_lorentzian = (
        2.69269 * gaussian**4
        + 2.0 * 2.42843 * gaussian**3 * lorentzian
        + 3.0 * 4.47163 * gaussian**2 * lorentzian**2
        + 4.0 * 0.07842 * gaussian * lorentzian**3
        + 5.0 * lorentzian**4
    )
    derivative_scale = 1.0 / (5.0 * normalized_total_fwhm**4)
    d_total_d_gaussian = d_polynomial_d_gaussian * derivative_scale
    d_total_d_lorentzian = d_polynomial_d_lorentzian * derivative_scale

    ratio = lorentzian / normalized_total_fwhm
    eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    d_eta_d_ratio = 1.36603 - 2.0 * 0.47719 * ratio + 3.0 * 0.11116 * ratio**2
    d_ratio_d_gaussian = -ratio * d_total_d_gaussian / total_fwhm
    d_ratio_d_lorentzian = (1.0 - ratio * d_total_d_lorentzian) / total_fwhm
    return ReferenceTchShape(
        total_fwhm=total_fwhm,
        eta=eta,
        d_total_fwhm_d_gaussian_fwhm=d_total_d_gaussian,
        d_total_fwhm_d_lorentzian_fwhm=d_total_d_lorentzian,
        d_eta_d_gaussian_fwhm=d_eta_d_ratio * d_ratio_d_gaussian,
        d_eta_d_lorentzian_fwhm=d_eta_d_ratio * d_ratio_d_lorentzian,
    )


def tch_shape_from_gaussian_sigma(
    gaussian_sigma: float, lorentzian_fwhm: float
) -> ReferenceTchShape:
    """Evaluate TCH after an explicit Gaussian sigma-to-FWHM conversion."""

    return tch_shape_from_fwhm(GAUSSIAN_FWHM_PER_SIGMA * gaussian_sigma, lorentzian_fwhm)


def profile_tch(
    delta: ArrayLike, gaussian_fwhm: float, lorentzian_fwhm: float
) -> ReferenceTchProfile:
    """Compose the independent TCH transform with the primitive profile."""

    shape = tch_shape_from_fwhm(gaussian_fwhm, lorentzian_fwhm)
    primitive = profile(delta, shape.total_fwhm, shape.eta)
    return ReferenceTchProfile(
        value=primitive.value,
        d_delta=primitive.d_delta,
        d_gaussian_fwhm=(
            primitive.d_fwhm * shape.d_total_fwhm_d_gaussian_fwhm
            + primitive.d_eta * shape.d_eta_d_gaussian_fwhm
        ),
        d_lorentzian_fwhm=(
            primitive.d_fwhm * shape.d_total_fwhm_d_lorentzian_fwhm
            + primitive.d_eta * shape.d_eta_d_lorentzian_fwhm
        ),
    )


def cw_profile_parameters(
    two_theta_deg: ArrayLike,
    *,
    u_deg2: float,
    v_deg2: float,
    w_deg2: float,
    x_deg: float,
    y_deg: float,
) -> ReferenceCwProfileParameters:
    """Evaluate U/V/W/X/Y broadening directly from the documented equations."""

    two_theta = np.asarray(two_theta_deg, dtype=np.float64)
    theta = np.deg2rad(two_theta / 2.0)
    tangent = np.tan(theta)
    secant = 1.0 / np.cos(theta)
    variance = u_deg2 * tangent**2 + v_deg2 * tangent + w_deg2
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(variance)
    lorentzian = x_deg * secant + y_deg * tangent
    total = np.empty_like(two_theta)
    eta = np.empty_like(two_theta)
    for index, (gaussian_value, lorentzian_value) in enumerate(
        zip(gaussian, lorentzian, strict=True)
    ):
        shape = tch_shape_from_fwhm(float(gaussian_value), float(lorentzian_value))
        total[index] = shape.total_fwhm
        eta[index] = shape.eta

    d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * np.sqrt(variance))
    d_gaussian = np.zeros((two_theta.size, 5), dtype=np.float64)
    d_gaussian[:, 0] = d_gaussian_d_variance * tangent**2
    d_gaussian[:, 1] = d_gaussian_d_variance * tangent
    d_gaussian[:, 2] = d_gaussian_d_variance
    d_lorentzian = np.zeros((two_theta.size, 5), dtype=np.float64)
    d_lorentzian[:, 3] = secant
    d_lorentzian[:, 4] = tangent

    radians_per_two_theta_degree = np.pi / 360.0
    d_tangent_d_position = radians_per_two_theta_degree * secant**2
    d_secant_d_position = radians_per_two_theta_degree * secant * tangent
    d_variance_d_position = (2.0 * u_deg2 * tangent + v_deg2) * d_tangent_d_position
    d_position = np.column_stack(
        (
            d_gaussian_d_variance * d_variance_d_position,
            x_deg * d_secant_d_position + y_deg * d_tangent_d_position,
        )
    )
    return ReferenceCwProfileParameters(
        gaussian_variance_deg2=variance,
        gaussian_fwhm_deg=gaussian,
        lorentzian_fwhm_deg=lorentzian,
        total_fwhm_deg=total,
        eta=eta,
        d_gaussian_fwhm_d_instrument=d_gaussian,
        d_lorentzian_fwhm_d_instrument=d_lorentzian,
        d_component_fwhm_d_two_theta=d_position,
    )


def profile_fcj(
    x_deg: ArrayLike,
    position_deg: float,
    gaussian_fwhm_deg: float,
    lorentzian_fwhm_deg: float,
    sample_over_radius: float,
    detector_over_radius: float,
    *,
    quadrature_order: int = 32,
    support_radius_deg: float | None = None,
) -> ReferenceFcjProfile:
    """Convolve TCH with FCJ axial divergence using a regular height integral.

    The published angular aberration has an integrable singularity at the
    Bragg angle.  This independent reference changes variables to normalized
    axial height, where the sample/detector overlap density is trapezoidal and
    the integrands are regular.
    """

    x_values = np.asarray(x_deg, dtype=np.float64)
    position = float(position_deg)
    sample = float(sample_over_radius)
    detector = float(detector_over_radius)
    if not 0.0 < position < 180.0:
        raise ValueError("position_deg must lie within (0, 180)")
    if sample < 0.0 or detector < 0.0 or not np.isfinite((sample, detector)).all():
        raise ValueError("FCJ axial ratios must be non-negative and finite")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    if support_radius_deg is not None and (
        not np.isfinite(support_radius_deg) or support_radius_deg <= 0.0
    ):
        raise ValueError("support_radius_deg must be positive and finite")

    position_rad = np.deg2rad(position)
    maximum_height = sample + detector
    limit_argument = np.cos(position_rad) * np.sqrt(1.0 + maximum_height**2)
    if abs(limit_argument) > 1.0:
        raise ValueError("FCJ axial ratios extend beyond the angular domain")
    if maximum_height == 0.0:
        symmetric = profile_tch(x_values - position, gaussian_fwhm_deg, lorentzian_fwhm_deg)
        if support_radius_deg is not None:
            active = np.abs(x_values - position) <= support_radius_deg
            symmetric = ReferenceTchProfile(
                value=np.where(active, symmetric.value, 0.0),
                d_delta=np.where(active, symmetric.d_delta, 0.0),
                d_gaussian_fwhm=np.where(active, symmetric.d_gaussian_fwhm, 0.0),
                d_lorentzian_fwhm=np.where(active, symmetric.d_lorentzian_fwhm, 0.0),
            )
        zeros = np.zeros_like(x_values)
        return ReferenceFcjProfile(
            value=symmetric.value,
            d_position=-symmetric.d_delta,
            d_gaussian_fwhm=symmetric.d_gaussian_fwhm,
            d_lorentzian_fwhm=symmetric.d_lorentzian_fwhm,
            d_sample_over_radius=zeros,
            d_detector_over_radius=zeros.copy(),
        )

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    t = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    major = max(sample, detector)
    minor = min(sample, detector)
    difference = major - minor

    def evaluate_height(
        height: NDArray[np.float64],
    ) -> tuple[
        ReferenceTchProfile,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        square_root = np.sqrt(1.0 + height**2)
        apparent_rad = np.arccos(np.cos(position_rad) * square_root)
        apparent_deg = np.rad2deg(apparent_rad)
        evaluated = profile_tch(
            x_values[None, :] - apparent_deg[:, None],
            gaussian_fwhm_deg,
            lorentzian_fwhm_deg,
        )
        if support_radius_deg is not None:
            active = np.abs(x_values[None, :] - apparent_deg[:, None]) <= support_radius_deg
            evaluated = ReferenceTchProfile(
                value=np.where(active, evaluated.value, 0.0),
                d_delta=np.where(active, evaluated.d_delta, 0.0),
                d_gaussian_fwhm=np.where(active, evaluated.d_gaussian_fwhm, 0.0),
                d_lorentzian_fwhm=np.where(active, evaluated.d_lorentzian_fwhm, 0.0),
            )
        sine_apparent = np.sin(apparent_rad)
        d_apparent_d_height_rad = -np.cos(position_rad) * height / (square_root * sine_apparent)
        d_apparent_d_height_deg = np.rad2deg(d_apparent_d_height_rad)
        d_value_d_height = -evaluated.d_delta * d_apparent_d_height_deg[:, None]
        d_apparent_d_position = np.sin(position_rad) * square_root / sine_apparent
        d_value_d_position = -evaluated.d_delta * d_apparent_d_position[:, None]
        geometry = 1.0 / (square_root * sine_apparent)
        cotangent_apparent = np.cos(apparent_rad) / sine_apparent
        d_geometry_d_height = geometry * (
            -height / (1.0 + height**2) - cotangent_apparent * d_apparent_d_height_rad
        )
        d_geometry_d_position = (
            geometry * -cotangent_apparent * d_apparent_d_position * np.pi / 180.0
        )
        return (
            evaluated,
            d_value_d_height,
            d_value_d_position,
            geometry,
            d_geometry_d_height,
            d_geometry_d_position,
        )

    height_flat = difference * t
    (
        flat,
        flat_d_height,
        flat_d_position,
        flat_geometry,
        flat_geometry_d_height,
        flat_geometry_d_position,
    ) = evaluate_height(height_flat)
    height_slope = difference + 2.0 * minor * t
    (
        slope,
        slope_d_height,
        slope_d_position,
        slope_geometry,
        slope_geometry_d_height,
        slope_geometry_d_position,
    ) = evaluate_height(height_slope)
    weighted_flat = weights[:, None]
    weighted_slope = (weights * (1.0 - t))[:, None]

    def flat_integral(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(weighted_flat * values, axis=0)

    def slope_integral(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(weighted_slope * values, axis=0)

    flat_weighted_value = flat_geometry[:, None] * flat.value
    slope_weighted_value = slope_geometry[:, None] * slope.value
    flat_numerator = flat_integral(flat_weighted_value)
    slope_numerator = slope_integral(slope_weighted_value)
    flat_denominator = float(np.sum(weights * flat_geometry))
    slope_denominator = float(np.sum(weights * (1.0 - t) * slope_geometry))
    numerator = difference * flat_numerator + 2.0 * minor * slope_numerator
    denominator = difference * flat_denominator + 2.0 * minor * slope_denominator
    value = numerator / denominator

    flat_position_integrand = (
        flat_geometry_d_position[:, None] * flat.value + flat_geometry[:, None] * flat_d_position
    )
    slope_position_integrand = (
        slope_geometry_d_position[:, None] * slope.value
        + slope_geometry[:, None] * slope_d_position
    )
    numerator_position = difference * flat_integral(
        flat_position_integrand
    ) + 2.0 * minor * slope_integral(slope_position_integrand)
    denominator_position = difference * float(
        np.sum(weights * flat_geometry_d_position)
    ) + 2.0 * minor * float(np.sum(weights * (1.0 - t) * slope_geometry_d_position))
    d_position = (numerator_position - value * denominator_position) / denominator
    d_gaussian = (
        difference * flat_integral(flat_geometry[:, None] * flat.d_gaussian_fwhm)
        + 2.0 * minor * slope_integral(slope_geometry[:, None] * slope.d_gaussian_fwhm)
    ) / denominator
    d_lorentzian = (
        difference * flat_integral(flat_geometry[:, None] * flat.d_lorentzian_fwhm)
        + 2.0 * minor * slope_integral(slope_geometry[:, None] * slope.d_lorentzian_fwhm)
    ) / denominator

    flat_numerator_d_height = (
        flat_geometry_d_height[:, None] * flat.value + flat_geometry[:, None] * flat_d_height
    )
    slope_numerator_d_height = (
        slope_geometry_d_height[:, None] * slope.value + slope_geometry[:, None] * slope_d_height
    )
    numerator_major = (
        flat_numerator
        + difference * flat_integral(flat_numerator_d_height * t[:, None])
        + 2.0 * minor * slope_integral(slope_numerator_d_height)
    )
    denominator_major = (
        flat_denominator
        + difference * float(np.sum(weights * flat_geometry_d_height * t))
        + 2.0 * minor * float(np.sum(weights * (1.0 - t) * slope_geometry_d_height))
    )
    d_value_d_major = (numerator_major - value * denominator_major) / denominator

    slope_minor_coordinate = -1.0 + 2.0 * t
    numerator_minor = (
        -flat_numerator
        - difference * flat_integral(flat_numerator_d_height * t[:, None])
        + 2.0 * slope_numerator
        + 2.0 * minor * slope_integral(slope_numerator_d_height * slope_minor_coordinate[:, None])
    )
    denominator_minor = (
        -flat_denominator
        - difference * float(np.sum(weights * flat_geometry_d_height * t))
        + 2.0 * slope_denominator
        + 2.0
        * minor
        * float(np.sum(weights * (1.0 - t) * slope_geometry_d_height * slope_minor_coordinate))
    )
    d_value_d_minor = (numerator_minor - value * denominator_minor) / denominator

    if detector > sample:
        d_detector, d_sample = d_value_d_major, d_value_d_minor
    elif sample > detector:
        d_sample, d_detector = d_value_d_major, d_value_d_minor
    else:
        equal_derivative = 0.5 * (d_value_d_major + d_value_d_minor)
        d_sample = equal_derivative
        d_detector = equal_derivative.copy()
    return ReferenceFcjProfile(
        value=value,
        d_position=d_position,
        d_gaussian_fwhm=d_gaussian,
        d_lorentzian_fwhm=d_lorentzian,
        d_sample_over_radius=d_sample,
        d_detector_over_radius=d_detector,
    )


def profile_tof(
    x_us: ArrayLike,
    position_us: float,
    alpha_per_us: float,
    beta_per_us: float,
    gaussian_fwhm_us: float,
    lorentzian_fwhm_us: float,
    *,
    tail_log: float = 20.0,
    quadrature_order: int = 96,
    base_radius_us: float | None = None,
) -> ReferenceTofProfile:
    """Evaluate the independent truncated two-sided exponential convolution."""

    x = np.asarray(x_us, dtype=np.float64)
    nodes, legendre_weights = np.polynomial.legendre.leggauss(quadrature_order)
    alpha = float(alpha_per_us)
    beta = float(beta_per_us)
    denominator = alpha + beta
    left_fraction = beta / denominator
    right_fraction = alpha / denominator
    d_left_fraction_d_alpha = -beta / denominator**2
    d_left_fraction_d_beta = alpha / denominator**2
    if base_radius_us is not None:
        output = [np.zeros_like(x) for _ in range(6)]
        normalization = 1.0 - np.exp(-tail_log)

        def integrate_interval(
            delta: float, low: float, high: float, rate: float, direction: float
        ) -> tuple[float, float, float, float, float]:
            if low >= high:
                return (0.0, 0.0, 0.0, 0.0, 0.0)
            t_local = low + (high - low) * (nodes + 1.0) / 2.0
            local_weights = (high - low) * legendre_weights / 2.0 * np.exp(-t_local) / normalization
            evaluated = profile_tch(
                delta + direction * t_local / rate,
                gaussian_fwhm_us,
                lorentzian_fwhm_us,
            )
            return (
                float(evaluated.value @ local_weights),
                float(evaluated.d_delta @ local_weights),
                float(evaluated.d_gaussian_fwhm @ local_weights),
                float(evaluated.d_lorentzian_fwhm @ local_weights),
                float((evaluated.d_delta * (-direction * t_local / rate**2)) @ local_weights),
            )

        for index, coordinate in enumerate(x.flat):
            delta = float(coordinate - position_us)
            left_low = float(np.clip(alpha * (-base_radius_us - delta), 0.0, tail_log))
            left_high = float(np.clip(alpha * (base_radius_us - delta), 0.0, tail_log))
            right_low = float(np.clip(beta * (delta - base_radius_us), 0.0, tail_log))
            right_high = float(np.clip(beta * (delta + base_radius_us), 0.0, tail_log))
            left = integrate_interval(delta, left_low, left_high, alpha, 1.0)
            right = integrate_interval(delta, right_low, right_high, beta, -1.0)
            output[0].flat[index] = left_fraction * left[0] + right_fraction * right[0]
            output[1].flat[index] = -(left_fraction * left[1] + right_fraction * right[1])
            output[2].flat[index] = (
                d_left_fraction_d_alpha * left[0]
                + left_fraction * left[4]
                - d_left_fraction_d_alpha * right[0]
            )
            output[3].flat[index] = (
                d_left_fraction_d_beta * left[0]
                + right_fraction * right[4]
                - d_left_fraction_d_beta * right[0]
            )
            output[4].flat[index] = left_fraction * left[2] + right_fraction * right[2]
            output[5].flat[index] = left_fraction * left[3] + right_fraction * right[3]
        return ReferenceTofProfile(*output)

    t = tail_log * (nodes + 1.0) / 2.0
    weights = tail_log * legendre_weights / 2.0 * np.exp(-t)
    weights /= np.sum(weights)
    left_delta = x[:, None] - position_us + t[None, :] / alpha
    right_delta = x[:, None] - position_us - t[None, :] / beta
    left = profile_tch(left_delta, gaussian_fwhm_us, lorentzian_fwhm_us)
    right = profile_tch(right_delta, gaussian_fwhm_us, lorentzian_fwhm_us)

    def integrate(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values @ weights

    left_value = integrate(left.value)
    right_value = integrate(right.value)
    left_d_delta = left.d_delta
    right_d_delta = right.d_delta
    d_delta = left_fraction * integrate(left_d_delta) + right_fraction * integrate(right_d_delta)
    d_alpha_shift = integrate(left_d_delta * (-t[None, :] / alpha**2))
    d_beta_shift = integrate(right_d_delta * (t[None, :] / beta**2))
    return ReferenceTofProfile(
        value=left_fraction * left_value + right_fraction * right_value,
        d_position=-d_delta,
        d_alpha=(
            d_left_fraction_d_alpha * left_value
            + left_fraction * d_alpha_shift
            - d_left_fraction_d_alpha * right_value
        ),
        d_beta=(
            d_left_fraction_d_beta * left_value
            + right_fraction * d_beta_shift
            - d_left_fraction_d_beta * right_value
        ),
        d_gaussian_fwhm=(
            left_fraction * integrate(left.d_gaussian_fwhm)
            + right_fraction * integrate(right.d_gaussian_fwhm)
        ),
        d_lorentzian_fwhm=(
            left_fraction * integrate(left.d_lorentzian_fwhm)
            + right_fraction * integrate(right.d_lorentzian_fwhm)
        ),
    )


def tof_profile_parameters(
    d_spacing_angstrom: ArrayLike,
    *,
    zero_us: float,
    difc_us_per_angstrom: float,
    difa_us_per_angstrom2: float,
    difb_us_angstrom: float,
    alpha_coefficient: float,
    beta0_per_us: float,
    beta1_angstrom4_per_us: float,
    betaq_angstrom2_per_us: float,
    sigma0_us2: float,
    sigma1_us2_per_angstrom2: float,
    sigma2_us2_per_angstrom4: float,
    sigmaq_us2_per_angstrom: float,
    x_us_per_angstrom: float,
    y_us_per_angstrom2: float,
    z_us: float,
) -> ReferenceTofProfileParameters:
    """Evaluate the documented TOF coefficient equations independently."""

    d = np.asarray(d_spacing_angstrom, dtype=np.float64)
    position = (
        zero_us + difc_us_per_angstrom * d + difa_us_per_angstrom2 * d**2 + difb_us_angstrom / d
    )
    alpha = alpha_coefficient / d
    beta = beta0_per_us + beta1_angstrom4_per_us / d**4 + betaq_angstrom2_per_us / d**2
    variance = (
        sigma0_us2
        + sigma1_us2_per_angstrom2 * d**2
        + sigma2_us2_per_angstrom4 * d**4
        + sigmaq_us2_per_angstrom * d
    )
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(variance)
    lorentzian = z_us + x_us_per_angstrom * d + y_us_per_angstrom2 * d**2
    return ReferenceTofProfileParameters(position, alpha, beta, variance, gaussian, lorentzian)


def accumulate(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    fwhms: ArrayLike,
    etas: ArrayLike,
    *,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loop over peaks in Python as a clear differential reference."""

    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    fwhms = np.asarray(fwhms, dtype=np.float64)
    etas = np.asarray(etas, dtype=np.float64)
    y = np.zeros_like(x)
    jacobian = np.zeros((positions.size, PARAMETER_COUNT, x.size), dtype=np.float64)

    for peak_index, (position, intensity, fwhm, eta) in enumerate(
        zip(positions, intensities, fwhms, etas, strict=True)
    ):
        active = np.abs(x - position) <= support_fwhm * fwhm
        delta = x[active] - position
        evaluated = profile(delta, float(fwhm), float(eta))
        y[active] += intensity * evaluated.value
        jacobian[peak_index, 0, active] = evaluated.value
        jacobian[peak_index, 1, active] = -intensity * evaluated.d_delta
        jacobian[peak_index, 2, active] = intensity * evaluated.d_fwhm
        jacobian[peak_index, 3, active] = intensity * evaluated.d_eta
    return y, jacobian


def accumulate_tch(
    x: ArrayLike,
    positions: ArrayLike,
    intensities: ArrayLike,
    gaussian_fwhms: ArrayLike,
    lorentzian_fwhms: ArrayLike,
    *,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loop over TCH peaks as a transparent differential reference."""

    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    gaussian_fwhms = np.asarray(gaussian_fwhms, dtype=np.float64)
    lorentzian_fwhms = np.asarray(lorentzian_fwhms, dtype=np.float64)
    y = np.zeros_like(x)
    jacobian = np.zeros((positions.size, PARAMETER_COUNT, x.size), dtype=np.float64)
    for peak_index, (position, intensity, gaussian, lorentzian) in enumerate(
        zip(
            positions,
            intensities,
            gaussian_fwhms,
            lorentzian_fwhms,
            strict=True,
        )
    ):
        shape = tch_shape_from_fwhm(float(gaussian), float(lorentzian))
        active = np.abs(x - position) <= support_fwhm * shape.total_fwhm
        evaluated = profile_tch(x[active] - position, float(gaussian), float(lorentzian))
        y[active] += intensity * evaluated.value
        jacobian[peak_index, 0, active] = evaluated.value
        jacobian[peak_index, 1, active] = -intensity * evaluated.d_delta
        jacobian[peak_index, 2, active] = intensity * evaluated.d_gaussian_fwhm
        jacobian[peak_index, 3, active] = intensity * evaluated.d_lorentzian_fwhm
    return y, jacobian


def accumulate_cw(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    *,
    u_deg2: float,
    v_deg2: float,
    w_deg2: float,
    x_deg: float,
    y_deg: float,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate CW reflections with independent local/global derivatives."""

    x_values = np.asarray(x, dtype=np.float64)
    positions = np.asarray(two_theta_deg, dtype=np.float64)
    intensities = np.asarray(integrated_intensities, dtype=np.float64)
    parameters = cw_profile_parameters(
        positions,
        u_deg2=u_deg2,
        v_deg2=v_deg2,
        w_deg2=w_deg2,
        x_deg=x_deg,
        y_deg=y_deg,
    )
    y_values = np.zeros_like(x_values)
    local = np.zeros((positions.size, 2, x_values.size), dtype=np.float64)
    global_jacobian = np.zeros((5, x_values.size), dtype=np.float64)
    for reflection, (position, intensity) in enumerate(zip(positions, intensities, strict=True)):
        active = np.abs(x_values - position) <= (
            support_fwhm * parameters.total_fwhm_deg[reflection]
        )
        evaluated = profile_tch(
            x_values[active] - position,
            float(parameters.gaussian_fwhm_deg[reflection]),
            float(parameters.lorentzian_fwhm_deg[reflection]),
        )
        y_values[active] += intensity * evaluated.value
        local[reflection, 0, active] = evaluated.value
        local[reflection, 1, active] = intensity * (
            -evaluated.d_delta
            + evaluated.d_gaussian_fwhm * parameters.d_component_fwhm_d_two_theta[reflection, 0]
            + evaluated.d_lorentzian_fwhm * parameters.d_component_fwhm_d_two_theta[reflection, 1]
        )
        for parameter in range(5):
            global_jacobian[parameter, active] += intensity * (
                evaluated.d_gaussian_fwhm
                * parameters.d_gaussian_fwhm_d_instrument[reflection, parameter]
                + evaluated.d_lorentzian_fwhm
                * parameters.d_lorentzian_fwhm_d_instrument[reflection, parameter]
            )
    return y_values, local, global_jacobian


def accumulate_cw_contributions(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    base_integrated_intensities: ArrayLike,
    *,
    gaussian_variance_deg2: ArrayLike,
    lorentzian_fwhm_deg: ArrayLike,
    intensity_multiplier: ArrayLike,
    d_gaussian_variance_d_position: ArrayLike,
    d_lorentzian_fwhm_d_position: ArrayLike,
    d_intensity_multiplier_d_position: ArrayLike,
    d_gaussian_variance_d_parameters: ArrayLike,
    d_lorentzian_fwhm_d_parameters: ArrayLike,
    d_intensity_multiplier_d_parameters: ArrayLike,
    u_deg2: float,
    v_deg2: float,
    w_deg2: float,
    x_deg: float,
    y_deg: float,
    support_fwhm: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Independently compose provider arrays with the CW profile equations."""

    x_values = np.asarray(x, dtype=np.float64)
    positions = np.asarray(two_theta_deg, dtype=np.float64)
    intensities = np.asarray(base_integrated_intensities, dtype=np.float64)
    q_add = np.asarray(gaussian_variance_deg2, dtype=np.float64)
    l_add = np.asarray(lorentzian_fwhm_deg, dtype=np.float64)
    multiplier = np.asarray(intensity_multiplier, dtype=np.float64)
    dq_position = np.asarray(d_gaussian_variance_d_position, dtype=np.float64)
    dl_position = np.asarray(d_lorentzian_fwhm_d_position, dtype=np.float64)
    dm_position = np.asarray(d_intensity_multiplier_d_position, dtype=np.float64)
    dq_parameters = np.asarray(d_gaussian_variance_d_parameters, dtype=np.float64)
    dl_parameters = np.asarray(d_lorentzian_fwhm_d_parameters, dtype=np.float64)
    dm_parameters = np.asarray(d_intensity_multiplier_d_parameters, dtype=np.float64)
    instrument = cw_profile_parameters(
        positions,
        u_deg2=u_deg2,
        v_deg2=v_deg2,
        w_deg2=w_deg2,
        x_deg=x_deg,
        y_deg=y_deg,
    )
    total_variance = instrument.gaussian_variance_deg2 + q_add
    gaussian = GAUSSIAN_FWHM_PER_SIGMA * np.sqrt(total_variance)
    lorentzian = instrument.lorentzian_fwhm_deg + l_add
    d_gaussian_d_variance = GAUSSIAN_FWHM_PER_SIGMA / (2.0 * np.sqrt(total_variance))
    instrument_scale = instrument.gaussian_fwhm_deg / gaussian
    parameter_count = dq_parameters.shape[0]
    y_values = np.zeros_like(x_values)
    local = np.zeros((positions.size, 2, x_values.size), dtype=np.float64)
    global_jacobian = np.zeros((5 + parameter_count, x_values.size), dtype=np.float64)
    for reflection, (position, base_intensity) in enumerate(
        zip(positions, intensities, strict=True)
    ):
        shape = tch_shape_from_fwhm(float(gaussian[reflection]), float(lorentzian[reflection]))
        active = np.abs(x_values - position) <= support_fwhm * shape.total_fwhm
        evaluated = profile_tch(
            x_values[active] - position,
            float(gaussian[reflection]),
            float(lorentzian[reflection]),
        )
        effective_intensity = base_intensity * multiplier[reflection]
        y_values[active] += effective_intensity * evaluated.value
        local[reflection, 0, active] = multiplier[reflection] * evaluated.value
        d_gaussian_position = (
            instrument.d_component_fwhm_d_two_theta[reflection, 0] * instrument_scale[reflection]
            + d_gaussian_d_variance[reflection] * dq_position[reflection]
        )
        d_lorentzian_position = (
            instrument.d_component_fwhm_d_two_theta[reflection, 1] + dl_position[reflection]
        )
        local[reflection, 1, active] = base_intensity * (
            dm_position[reflection] * evaluated.value
            + multiplier[reflection]
            * (
                -evaluated.d_delta
                + evaluated.d_gaussian_fwhm * d_gaussian_position
                + evaluated.d_lorentzian_fwhm * d_lorentzian_position
            )
        )
        for parameter in range(5):
            global_jacobian[parameter, active] += effective_intensity * (
                evaluated.d_gaussian_fwhm
                * instrument.d_gaussian_fwhm_d_instrument[reflection, parameter]
                * instrument_scale[reflection]
                + evaluated.d_lorentzian_fwhm
                * instrument.d_lorentzian_fwhm_d_instrument[reflection, parameter]
            )
        for parameter in range(parameter_count):
            global_jacobian[5 + parameter, active] += base_intensity * (
                dm_parameters[parameter, reflection] * evaluated.value
                + multiplier[reflection]
                * (
                    evaluated.d_gaussian_fwhm
                    * d_gaussian_d_variance[reflection]
                    * dq_parameters[parameter, reflection]
                    + evaluated.d_lorentzian_fwhm * dl_parameters[parameter, reflection]
                )
            )
    return y_values, local, global_jacobian


def accumulate_cw_fcj(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    *,
    u_deg2: float,
    v_deg2: float,
    w_deg2: float,
    x_deg: float,
    y_deg: float,
    sample_over_radius: float,
    detector_over_radius: float,
    support_fwhm: float = 20.0,
    quadrature_order: int = 48,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate FCJ-asymmetric CW reflections as a transparent reference."""

    x_values = np.asarray(x, dtype=np.float64)
    positions = np.asarray(two_theta_deg, dtype=np.float64)
    intensities = np.asarray(integrated_intensities, dtype=np.float64)
    parameters = cw_profile_parameters(
        positions,
        u_deg2=u_deg2,
        v_deg2=v_deg2,
        w_deg2=w_deg2,
        x_deg=x_deg,
        y_deg=y_deg,
    )
    y_values = np.zeros_like(x_values)
    local = np.zeros((positions.size, 2, x_values.size), dtype=np.float64)
    global_jacobian = np.zeros((7, x_values.size), dtype=np.float64)
    for reflection, (position, intensity) in enumerate(zip(positions, intensities, strict=True)):
        support_radius = support_fwhm * parameters.total_fwhm_deg[reflection]
        maximum_height = sample_over_radius + detector_over_radius
        apparent_limit = np.rad2deg(
            np.arccos(np.cos(np.deg2rad(position)) * np.sqrt(1.0 + maximum_height**2))
        )
        active = (x_values >= min(position, apparent_limit) - support_radius) & (
            x_values <= max(position, apparent_limit) + support_radius
        )
        evaluated = profile_fcj(
            x_values[active],
            float(position),
            float(parameters.gaussian_fwhm_deg[reflection]),
            float(parameters.lorentzian_fwhm_deg[reflection]),
            sample_over_radius,
            detector_over_radius,
            quadrature_order=quadrature_order,
            support_radius_deg=float(support_radius),
        )
        y_values[active] += intensity * evaluated.value
        local[reflection, 0, active] = evaluated.value
        local[reflection, 1, active] = intensity * (
            evaluated.d_position
            + evaluated.d_gaussian_fwhm * parameters.d_component_fwhm_d_two_theta[reflection, 0]
            + evaluated.d_lorentzian_fwhm * parameters.d_component_fwhm_d_two_theta[reflection, 1]
        )
        for parameter in range(5):
            global_jacobian[parameter, active] += intensity * (
                evaluated.d_gaussian_fwhm
                * parameters.d_gaussian_fwhm_d_instrument[reflection, parameter]
                + evaluated.d_lorentzian_fwhm
                * parameters.d_lorentzian_fwhm_d_instrument[reflection, parameter]
            )
        global_jacobian[5, active] += intensity * evaluated.d_sample_over_radius
        global_jacobian[6, active] += intensity * evaluated.d_detector_over_radius
    return y_values, local, global_jacobian


def wavelength_component_positions(
    base_two_theta_deg: ArrayLike,
    reference_wavelength_angstrom: float,
    wavelengths_angstrom: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply Bragg's law to discrete wavelengths and return its derivatives."""

    base = np.asarray(base_two_theta_deg, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_angstrom, dtype=np.float64)
    ratios = wavelengths / reference_wavelength_angstrom
    base_theta = np.deg2rad(base / 2.0)
    component_sines = np.sin(base_theta)[:, None] * ratios[None, :]
    if np.any(component_sines <= 0.0) or np.any(component_sines >= 1.0):
        raise ValueError("wavelength component lies outside the Bragg domain")
    component_theta = np.arcsin(component_sines)
    positions = np.rad2deg(2.0 * component_theta)
    positions[:, 0] = base
    d_position_d_base = ratios[None, :] * np.cos(base_theta)[:, None] / np.cos(component_theta)
    d_position_d_base[:, 0] = 1.0
    d_position_d_ratio = 360.0 / np.pi * np.sin(base_theta)[:, None] / np.cos(component_theta)
    return positions, d_position_d_base, d_position_d_ratio


def accumulate_cw_components(
    x: ArrayLike,
    two_theta_deg: ArrayLike,
    integrated_intensities: ArrayLike,
    *,
    reference_wavelength_angstrom: float,
    wavelengths_angstrom: ArrayLike,
    relative_component_intensities: ArrayLike,
    u_deg2: float,
    v_deg2: float,
    w_deg2: float,
    x_deg: float,
    y_deg: float,
    sample_over_radius: float | None = None,
    detector_over_radius: float | None = None,
    support_fwhm: float = 20.0,
    quadrature_order: int = 48,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate optional wavelength components as an independent reference."""

    x_values = np.asarray(x, dtype=np.float64)
    base_positions = np.asarray(two_theta_deg, dtype=np.float64)
    intensities = np.asarray(integrated_intensities, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_angstrom, dtype=np.float64)
    relative = np.asarray(relative_component_intensities, dtype=np.float64)
    component_positions, d_position_d_base, d_position_d_ratio = wavelength_component_positions(
        base_positions, reference_wavelength_angstrom, wavelengths
    )
    scaled_relative = relative / np.max(relative)
    weights = scaled_relative / np.sum(scaled_relative)
    include_fcj = sample_over_radius is not None or detector_over_radius is not None
    if include_fcj and (sample_over_radius is None or detector_over_radius is None):
        raise ValueError("both FCJ geometry ratios must be supplied together")
    sample = 0.0 if sample_over_radius is None else sample_over_radius
    detector = 0.0 if detector_over_radius is None else detector_over_radius
    secondary_count = wavelengths.size - 1
    global_count = 5 + (2 if include_fcj else 0) + 2 * secondary_count
    wavelength_start = 5 + (2 if include_fcj else 0)
    intensity_start = wavelength_start + secondary_count
    y_values = np.zeros_like(x_values)
    local = np.zeros((base_positions.size, 2, x_values.size), dtype=np.float64)
    global_jacobian = np.zeros((global_count, x_values.size), dtype=np.float64)

    for reflection, intensity in enumerate(intensities):
        component_values = np.zeros((wavelengths.size, x_values.size), dtype=np.float64)
        mixture_d_base = np.zeros_like(x_values)
        for component in range(wavelengths.size):
            position = component_positions[reflection, component]
            parameters = cw_profile_parameters(
                [position],
                u_deg2=u_deg2,
                v_deg2=v_deg2,
                w_deg2=w_deg2,
                x_deg=x_deg,
                y_deg=y_deg,
            )
            support_radius = support_fwhm * parameters.total_fwhm_deg[0]
            evaluated = profile_fcj(
                x_values,
                float(position),
                float(parameters.gaussian_fwhm_deg[0]),
                float(parameters.lorentzian_fwhm_deg[0]),
                sample,
                detector,
                quadrature_order=quadrature_order,
                support_radius_deg=float(support_radius),
            )
            component_values[component] = evaluated.value
            d_profile_d_position = (
                evaluated.d_position
                + evaluated.d_gaussian_fwhm * parameters.d_component_fwhm_d_two_theta[0, 0]
                + evaluated.d_lorentzian_fwhm * parameters.d_component_fwhm_d_two_theta[0, 1]
            )
            mixture_d_base += (
                weights[component] * d_profile_d_position * d_position_d_base[reflection, component]
            )
            for parameter in range(5):
                global_jacobian[parameter] += (
                    intensity
                    * weights[component]
                    * (
                        evaluated.d_gaussian_fwhm
                        * parameters.d_gaussian_fwhm_d_instrument[0, parameter]
                        + evaluated.d_lorentzian_fwhm
                        * parameters.d_lorentzian_fwhm_d_instrument[0, parameter]
                    )
                )
            if include_fcj:
                global_jacobian[5] += (
                    intensity * weights[component] * evaluated.d_sample_over_radius
                )
                global_jacobian[6] += (
                    intensity * weights[component] * evaluated.d_detector_over_radius
                )
            if component > 0:
                global_jacobian[wavelength_start + component - 1] += (
                    intensity
                    * weights[component]
                    * d_profile_d_position
                    * d_position_d_ratio[reflection, component]
                )
        mixture = np.sum(weights[:, None] * component_values, axis=0)
        y_values += intensity * mixture
        local[reflection, 0] = mixture
        local[reflection, 1] = intensity * mixture_d_base
        for secondary in range(secondary_count):
            global_jacobian[intensity_start + secondary] += (
                intensity * weights[0] * (component_values[secondary + 1] - mixture)
            )
    return y_values, local, global_jacobian
