"""Offline fundamental-parameters compression into PhaseSmith CW profiles.

The physical target is evaluated with the independent NumPy profile reference.
Only the candidate ``U/V/W/X/Y + SH/L`` profile uses the production Rust kernel.
This separation keeps the calibration target independent from the model being
fitted and keeps fundamental-parameters work outside ordinary refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import reference
from .fcj import accumulate_cw_fcj_components
from .instrument import ConstantWavelengthInstrument, FcjGeometry
from .radiation import WavelengthComponents

_GAUSSIAN_FWHM_PER_SIGMA = 2.354_820_045_030_949_3
_PARAMETER_NAMES = ("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg", "sh_over_l")
_TARGET_PROFILE_CHUNK_SIZE = 256


@dataclass(frozen=True, slots=True)
class FundamentalEmissionLine:
    """One source line with integrated intensity and intrinsic wavelength widths."""

    wavelength_angstrom: float
    relative_intensity: float
    gaussian_fwhm_angstrom: float
    lorentzian_fwhm_angstrom: float

    def __post_init__(self) -> None:
        values = (
            self.wavelength_angstrom,
            self.relative_intensity,
            self.gaussian_fwhm_angstrom,
            self.lorentzian_fwhm_angstrom,
        )
        if not np.isfinite(values).all():
            raise ValueError("emission-line values must be finite")
        if self.wavelength_angstrom <= 0.0 or self.relative_intensity <= 0.0:
            raise ValueError("emission wavelength and intensity must be positive")
        if self.gaussian_fwhm_angstrom < 0.0 or self.lorentzian_fwhm_angstrom < 0.0:
            raise ValueError("emission-line widths must be non-negative")
        if self.gaussian_fwhm_angstrom == 0.0 and self.lorentzian_fwhm_angstrom == 0.0:
            raise ValueError("an emission line must have a positive intrinsic width")


@dataclass(frozen=True, slots=True)
class SollerAxialGeometry:
    """Full axial lengths and optional triangular Soller angular filters.

    Lengths are full axial lengths in millimetres. Soller widths are full
    angular widths in degrees; ``None`` represents an open axial beam.
    """

    source_full_length_mm: float
    sample_full_length_mm: float
    receiving_slit_full_length_mm: float
    incident_soller_full_width_deg: float | None = None
    diffracted_soller_full_width_deg: float | None = None

    def __post_init__(self) -> None:
        lengths = (
            self.source_full_length_mm,
            self.sample_full_length_mm,
            self.receiving_slit_full_length_mm,
        )
        if not np.isfinite(lengths).all() or any(value < 0.0 for value in lengths):
            raise ValueError("full axial lengths must be finite and non-negative")
        if any(value == 0.0 for value in lengths) and not (
            self.source_full_length_mm == 0.0 and self.sample_full_length_mm == 0.0
        ):
            raise ValueError(
                "full axial lengths must all be positive, or source and sample "
                "must both be zero for the point-incident FCJ limit"
            )
        for name, value in (
            ("incident_soller_full_width_deg", self.incident_soller_full_width_deg),
            ("diffracted_soller_full_width_deg", self.diffracted_soller_full_width_deg),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite or None")


@dataclass(frozen=True, slots=True)
class BraggBrentanoFundamentalProfile:
    """Reviewed first-slice physical description of a laboratory CW instrument.

    Aperture widths are full equatorial widths. Axial values are half-lengths,
    matching :class:`~phasesmith.FcjGeometry` after division by the radius.
    """

    radius_mm: float
    source_width_mm: float
    receiving_slit_width_mm: float
    sample_half_length_mm: float
    detector_half_length_mm: float
    emission_lines: tuple[FundamentalEmissionLine, ...]
    soller_axial_geometry: SollerAxialGeometry | None = None

    def __post_init__(self) -> None:
        geometry = (
            self.radius_mm,
            self.source_width_mm,
            self.receiving_slit_width_mm,
            self.sample_half_length_mm,
            self.detector_half_length_mm,
        )
        if not np.isfinite(geometry).all():
            raise ValueError("fundamental-profile geometry must be finite")
        if self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive")
        if any(value < 0.0 for value in geometry[1:]):
            raise ValueError("fundamental-profile lengths must be non-negative")
        if not isinstance(self.emission_lines, tuple) or not self.emission_lines:
            raise ValueError("emission_lines must be a non-empty tuple")
        if not all(isinstance(line, FundamentalEmissionLine) for line in self.emission_lines):
            raise TypeError("emission_lines must contain FundamentalEmissionLine values")
        if self.soller_axial_geometry is not None:
            if not isinstance(self.soller_axial_geometry, SollerAxialGeometry):
                raise TypeError("soller_axial_geometry must be SollerAxialGeometry or None")
            sample_length = 2.0 * self.sample_half_length_mm
            detector_length = 2.0 * self.detector_half_length_mm
            if not np.isclose(
                sample_length,
                self.soller_axial_geometry.sample_full_length_mm,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "sample_half_length_mm must equal half the Soller sample full length"
                )
            if not np.isclose(
                detector_length,
                self.soller_axial_geometry.receiving_slit_full_length_mm,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "detector_half_length_mm must equal half the Soller receiver full length"
                )

    @property
    def components(self) -> WavelengthComponents:
        """Return the fixed discrete spectrum used by the compressed model."""

        return WavelengthComponents(
            [line.wavelength_angstrom for line in self.emission_lines],
            [line.relative_intensity for line in self.emission_lines],
        )

    @property
    def axial_geometry(self) -> FcjGeometry:
        """Return the physical target's FCJ half-height ratios."""

        return FcjGeometry(
            self.sample_half_length_mm / self.radius_mm,
            self.detector_half_length_mm / self.radius_mm,
        )


@dataclass(frozen=True, slots=True)
class FundamentalProfileCalibrationOptions:
    """Deterministic isolated-peak grid, optimizer, and acceptance controls."""

    peak_positions_deg: tuple[float, ...] = (
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
    )
    window_half_width_deg: float = 0.8
    step_deg: float = 0.002
    aperture_quadrature_order: int = 3
    axial_ray_quadrature_order: int = 255
    fcj_quadrature_order: int = 48
    support_fwhm: float = 40.0
    max_iterations: int = 40
    damping: float = 1.0e-8
    step_tolerance: float = 1.0e-8
    objective_tolerance: float = 1.0e-10
    maximum_relative_l2_error: float = 0.03
    maximum_peak_relative_l2_error: float = 0.05
    minimum_profile_correlation: float = 0.995

    def __post_init__(self) -> None:
        if not isinstance(self.peak_positions_deg, tuple):
            raise TypeError("peak_positions_deg must be a tuple")
        positions = np.asarray(self.peak_positions_deg, dtype=np.float64)
        if positions.ndim != 1 or positions.size < 6:
            raise ValueError("peak_positions_deg must contain at least six positions")
        if not np.isfinite(positions).all() or np.any((positions <= 0.0) | (positions >= 180.0)):
            raise ValueError("peak positions must be finite and lie within (0, 180)")
        if np.any(np.diff(positions) <= 2.0 * self.window_half_width_deg):
            raise ValueError("isolated-peak windows must be strictly separated")
        positive = (
            self.window_half_width_deg,
            self.step_deg,
            self.support_fwhm,
            self.step_tolerance,
            self.objective_tolerance,
            self.maximum_relative_l2_error,
            self.maximum_peak_relative_l2_error,
        )
        if not np.isfinite(positive).all() or any(value <= 0.0 for value in positive):
            raise ValueError("positive calibration controls must be finite and positive")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        integer_controls = (
            self.aperture_quadrature_order,
            self.axial_ray_quadrature_order,
            self.fcj_quadrature_order,
            self.max_iterations,
        )
        integers_are_valid = all(
            isinstance(value, int) and not isinstance(value, bool) for value in integer_controls
        )
        if not integers_are_valid:
            raise TypeError("quadrature orders and max_iterations must be integers")
        if (
            self.aperture_quadrature_order <= 0
            or self.axial_ray_quadrature_order <= 0
            or self.fcj_quadrature_order <= 0
        ):
            raise ValueError("quadrature orders must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not np.isfinite(self.minimum_profile_correlation) or not (
            -1.0 <= self.minimum_profile_correlation <= 1.0
        ):
            raise ValueError("minimum_profile_correlation must lie in [-1, 1]")


@dataclass(frozen=True, slots=True)
class FundamentalPeakDiagnostic:
    """Shape-compression diagnostics for one isolated synthetic peak."""

    position_deg: float
    scale: float
    relative_l2_error: float
    profile_correlation: float


@dataclass(frozen=True, slots=True)
class FundamentalPeakPattern:
    """Independent isolated-peak target generated from the physical inputs."""

    positions_deg: NDArray[np.float64]
    grid_deg: NDArray[np.float64]
    intensity: NDArray[np.float64]
    peak_slices: tuple[slice, ...]


@dataclass(frozen=True, slots=True)
class FundamentalProfileCalibrationResult:
    """Compressed production model and complete synthetic-fit diagnostics."""

    instrument: ConstantWavelengthInstrument
    components: WavelengthComponents
    axial_geometry: FcjGeometry
    sh_over_l: float
    parameter_names: tuple[str, ...]
    iterations: int
    converged: bool
    accepted: bool
    relative_l2_error: float
    maximum_peak_relative_l2_error: float
    minimum_profile_correlation: float
    peak_diagnostics: tuple[FundamentalPeakDiagnostic, ...]
    grid_deg: NDArray[np.float64]
    target_y: NDArray[np.float64]
    calculated_y: NDArray[np.float64]
    warnings: tuple[str, ...]


def simulate_fundamental_peaks(
    model: BraggBrentanoFundamentalProfile,
    options: FundamentalProfileCalibrationOptions | None = None,
) -> FundamentalPeakPattern:
    """Generate isolated physical peak targets with the independent NumPy model."""

    if not isinstance(model, BraggBrentanoFundamentalProfile):
        raise TypeError("model must be BraggBrentanoFundamentalProfile")
    selected = FundamentalProfileCalibrationOptions() if options is None else options
    if not isinstance(selected, FundamentalProfileCalibrationOptions):
        raise TypeError("options must be FundamentalProfileCalibrationOptions")
    positions = np.asarray(selected.peak_positions_deg, dtype=np.float64)
    grid, slices = _isolated_grid(positions, selected)
    intensity = _fundamental_target(model, positions, grid, slices, selected)
    return FundamentalPeakPattern(
        positions_deg=_readonly(positions),
        grid_deg=_readonly(grid),
        intensity=_readonly(intensity),
        peak_slices=slices,
    )


def calibrate_fundamental_profile(
    model: BraggBrentanoFundamentalProfile,
    options: FundamentalProfileCalibrationOptions | None = None,
) -> FundamentalProfileCalibrationResult:
    """Compress independently generated FPA-like peaks into the production model."""

    if not isinstance(model, BraggBrentanoFundamentalProfile):
        raise TypeError("model must be BraggBrentanoFundamentalProfile")
    selected = FundamentalProfileCalibrationOptions() if options is None else options
    if not isinstance(selected, FundamentalProfileCalibrationOptions):
        raise TypeError("options must be FundamentalProfileCalibrationOptions")

    target_pattern = simulate_fundamental_peaks(model, selected)
    positions = target_pattern.positions_deg
    grid = target_pattern.grid_deg
    slices = target_pattern.peak_slices
    target = target_pattern.intensity
    components = model.components
    parameters = _initial_parameters(model)
    lower = np.array([0.0, -0.2, 1.0e-12, 0.0, 0.0, 0.0], dtype=np.float64)
    upper = np.array([0.2, 0.2, 0.2, 1.0, 1.0, 0.2], dtype=np.float64)
    scales = np.array([1.0e-3, 1.0e-3, 1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2])
    parameters = np.clip(parameters, lower, upper)
    target_norms = np.array([np.linalg.norm(target[item]) for item in slices])
    converged = False
    iterations = 0
    calculated = np.zeros_like(target)
    peak_scales = np.ones(positions.size)

    for iteration in range(1, selected.max_iterations + 1):
        iterations = iteration
        evaluated, jacobian, peak_scales = _candidate(
            grid, positions, slices, target, components, parameters, selected
        )
        residual, weighted_jacobian = _weighted_system(
            target, evaluated, jacobian, slices, target_norms
        )
        objective = float(residual @ residual)
        columns = weighted_jacobian * scales[None, :]
        scaled_step = _bounded_gauss_newton_step(
            columns,
            residual,
            selected.damping,
            parameters,
            lower,
            upper,
            scales,
        )
        physical_step = scales * scaled_step
        relative_step = float(np.max(np.abs(scaled_step)))
        if relative_step <= selected.step_tolerance:
            converged = True
            calculated = evaluated
            break
        accepted_step = False
        factor = 1.0
        trial_objective = objective
        for _backtrack in range(14):
            trial = np.clip(parameters + factor * physical_step, lower, upper)
            try:
                trial_y, _trial_jacobian, trial_scales = _candidate(
                    grid, positions, slices, target, components, trial, selected
                )
            except ValueError:
                factor *= 0.5
                continue
            trial_residual, _ = _weighted_system(
                target, trial_y, _trial_jacobian, slices, target_norms
            )
            trial_objective = float(trial_residual @ trial_residual)
            if trial_objective < objective:
                parameters = trial
                calculated = trial_y
                peak_scales = trial_scales
                accepted_step = True
                break
            factor *= 0.5
        if not accepted_step:
            calculated = evaluated
            break
        relative_objective_change = (objective - trial_objective) / max(objective, 1.0e-30)
        if factor * relative_step <= selected.step_tolerance or (
            relative_objective_change <= selected.objective_tolerance
        ):
            converged = True
            break

    calculated, _jacobian, peak_scales = _candidate(
        grid, positions, slices, target, components, parameters, selected
    )
    diagnostics = _peak_diagnostics(positions, slices, target, calculated, peak_scales)
    relative_l2 = float(np.linalg.norm(calculated - target) / np.linalg.norm(target))
    maximum_peak_relative_l2 = max(item.relative_l2_error for item in diagnostics)
    minimum_correlation = min(item.profile_correlation for item in diagnostics)
    metrics_accepted = (
        relative_l2 <= selected.maximum_relative_l2_error
        and maximum_peak_relative_l2 <= selected.maximum_peak_relative_l2_error
        and minimum_correlation >= selected.minimum_profile_correlation
    )
    accepted = converged and metrics_accepted
    warnings = []
    if not converged:
        warnings.append("profile compression stopped before the convergence thresholds")
    if not metrics_accepted:
        warnings.append(
            "fundamental peaks are not represented within the requested empirical-profile limits"
        )
    instrument = _instrument(components.wavelengths_angstrom[0], parameters)
    return FundamentalProfileCalibrationResult(
        instrument=instrument,
        components=components,
        axial_geometry=FcjGeometry(parameters[5] / 2.0, parameters[5] / 2.0),
        sh_over_l=float(parameters[5]),
        parameter_names=_PARAMETER_NAMES,
        iterations=iterations,
        converged=converged,
        accepted=accepted,
        relative_l2_error=relative_l2,
        maximum_peak_relative_l2_error=maximum_peak_relative_l2,
        minimum_profile_correlation=minimum_correlation,
        peak_diagnostics=diagnostics,
        grid_deg=_readonly(grid),
        target_y=_readonly(target),
        calculated_y=_readonly(calculated),
        warnings=tuple(warnings),
    )


def _isolated_grid(
    positions: NDArray[np.float64], options: FundamentalProfileCalibrationOptions
) -> tuple[NDArray[np.float64], tuple[slice, ...]]:
    count = int(np.ceil(2.0 * options.window_half_width_deg / options.step_deg)) + 1
    local = np.linspace(-options.window_half_width_deg, options.window_half_width_deg, count)
    arrays = tuple(position + local for position in positions)
    slices = tuple(slice(index * count, (index + 1) * count) for index in range(positions.size))
    return np.concatenate(arrays), slices


def _fundamental_target(
    model: BraggBrentanoFundamentalProfile,
    positions: NDArray[np.float64],
    grid: NDArray[np.float64],
    slices: tuple[slice, ...],
    options: FundamentalProfileCalibrationOptions,
) -> NDArray[np.float64]:
    target = np.zeros_like(grid)
    reference_wavelength = model.emission_lines[0].wavelength_angstrom
    component_weights = model.components.normalized_intensities
    nodes, weights = np.polynomial.legendre.leggauss(options.aperture_quadrature_order)
    source_offsets = 0.5 * model.source_width_mm * nodes
    detector_offsets = 0.5 * model.receiving_slit_width_mm * nodes
    combined_offsets = np.rad2deg(
        (source_offsets[:, None] + detector_offsets[None, :]) / model.radius_mm
    ).ravel()
    combined_weights = (0.5 * weights[:, None] * 0.5 * weights[None, :]).ravel()
    combined_weights /= np.sum(combined_weights)
    geometry = model.axial_geometry
    for position, active in zip(positions, slices, strict=True):
        d_spacing = reference_wavelength / (2.0 * np.sin(np.deg2rad(position / 2.0)))
        local_grid = grid[active]
        for line, component_weight in zip(model.emission_lines, component_weights, strict=True):
            argument = line.wavelength_angstrom / (2.0 * d_spacing)
            if not 0.0 < argument < 1.0:
                raise ValueError("an emission component lies outside the Bragg domain")
            component_position = float(2.0 * np.rad2deg(np.arcsin(argument)))
            tangent = np.tan(np.deg2rad(component_position / 2.0))
            angular_per_wavelength = np.rad2deg(2.0 * tangent / line.wavelength_angstrom)
            gaussian = angular_per_wavelength * line.gaussian_fwhm_angstrom
            lorentzian = angular_per_wavelength * line.lorentzian_fwhm_angstrom
            if model.soller_axial_geometry is None:
                for offset, aperture_weight in zip(combined_offsets, combined_weights, strict=True):
                    evaluated = reference.profile_fcj(
                        local_grid,
                        component_position + float(offset),
                        gaussian,
                        lorentzian,
                        geometry.sample_over_radius,
                        geometry.detector_over_radius,
                        quadrature_order=options.fcj_quadrature_order,
                    )
                    target[active] += component_weight * aperture_weight * evaluated.value
                continue
            axial_centres, axial_weights = _soller_axial_rays(
                component_position,
                model.radius_mm,
                model.soller_axial_geometry,
                options.axial_ray_quadrature_order,
            )
            centres = (axial_centres[:, None] + combined_offsets[None, :]).ravel()
            ray_weights = (axial_weights[:, None] * combined_weights[None, :]).ravel()
            target[active] += component_weight * _weighted_tch_profiles(
                local_grid,
                centres,
                ray_weights,
                gaussian,
                lorentzian,
            )
    return target


def _soller_axial_rays(
    position_deg: float,
    radius_mm: float,
    geometry: SollerAxialGeometry,
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return apparent angles and normalized weights for the axial ray integral."""

    if (
        min(
            geometry.source_full_length_mm,
            geometry.sample_full_length_mm,
            geometry.receiving_slit_full_length_mm,
        )
        == 0.0
    ):
        return _coordinate_axial_rays(position_deg, radius_mm, geometry, quadrature_order)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    incident_limit = np.arctan(
        (geometry.source_full_length_mm + geometry.sample_full_length_mm) / (2.0 * radius_mm)
    )
    diffracted_limit = np.arctan(
        (geometry.sample_full_length_mm + geometry.receiving_slit_full_length_mm)
        / (2.0 * radius_mm)
    )
    if geometry.incident_soller_full_width_deg is not None:
        incident_limit = min(
            incident_limit,
            np.deg2rad(geometry.incident_soller_full_width_deg / 2.0),
        )
    if geometry.diffracted_soller_full_width_deg is not None:
        diffracted_limit = min(
            diffracted_limit,
            np.deg2rad(geometry.diffracted_soller_full_width_deg / 2.0),
        )
    incident = incident_limit * nodes[:, None]
    diffracted = diffracted_limit * nodes[None, :]
    lower = np.maximum.reduce(
        (
            np.full(
                (quadrature_order, quadrature_order),
                -geometry.sample_full_length_mm / 2.0,
            ),
            np.broadcast_to(
                -geometry.source_full_length_mm / 2.0 + radius_mm * np.tan(incident),
                (quadrature_order, quadrature_order),
            ),
            np.broadcast_to(
                -geometry.receiving_slit_full_length_mm / 2.0 - radius_mm * np.tan(diffracted),
                (quadrature_order, quadrature_order),
            ),
        )
    )
    upper = np.minimum.reduce(
        (
            np.broadcast_to(
                geometry.sample_full_length_mm / 2.0,
                (quadrature_order, quadrature_order),
            ),
            np.broadcast_to(
                geometry.source_full_length_mm / 2.0 + radius_mm * np.tan(incident),
                (quadrature_order, quadrature_order),
            ),
            np.broadcast_to(
                geometry.receiving_slit_full_length_mm / 2.0 - radius_mm * np.tan(diffracted),
                (quadrature_order, quadrature_order),
            ),
        )
    )
    overlap = np.maximum(upper - lower, 0.0)
    ray_weights = weights[:, None] * weights[None, :] * overlap
    ray_weights = ray_weights * _soller_transmission(
        incident, geometry.incident_soller_full_width_deg
    )
    ray_weights = ray_weights * _soller_transmission(
        diffracted, geometry.diffracted_soller_full_width_deg
    )
    apparent = _apparent_axial_angle(position_deg, incident, diffracted)
    ray_weights = ray_weights / (np.cos(incident) * np.cos(diffracted) * np.sin(apparent))
    return _normalize_axial_rays(apparent, ray_weights)


def _coordinate_axial_rays(
    position_deg: float,
    radius_mm: float,
    geometry: SollerAxialGeometry,
    quadrature_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if geometry.receiving_slit_full_length_mm == 0.0:
        return np.array([position_deg]), np.array([1.0])
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    diffracted = np.arctan(0.5 * geometry.receiving_slit_full_length_mm * nodes / radius_mm)
    incident = np.zeros_like(diffracted)
    ray_weights = weights.copy()
    ray_weights = ray_weights * _soller_transmission(
        diffracted, geometry.diffracted_soller_full_width_deg
    )
    apparent = _apparent_axial_angle(position_deg, incident, diffracted)
    ray_weights = ray_weights * np.cos(incident) * np.cos(diffracted) / np.sin(apparent)
    return _normalize_axial_rays(apparent, ray_weights)


def _apparent_axial_angle(
    position_deg: float,
    incident: NDArray[np.float64],
    diffracted: NDArray[np.float64],
) -> NDArray[np.float64]:
    position = np.deg2rad(position_deg)
    cosine = (np.cos(position) - np.sin(incident) * np.sin(diffracted)) / (
        np.cos(incident) * np.cos(diffracted)
    )
    domain_tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any((cosine < -1.0 - domain_tolerance) | (cosine > 1.0 + domain_tolerance)):
        raise ValueError("axial ray lies outside the apparent-angle domain")
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _normalize_axial_rays(
    apparent: NDArray[np.float64], ray_weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    apparent = np.rad2deg(apparent).ravel()
    ray_weights = ray_weights.ravel()
    selected = np.isfinite(ray_weights) & (ray_weights > 0.0)
    if not np.any(selected):
        raise ValueError("Soller filters reject every axial quadrature ray")
    apparent = apparent[selected]
    ray_weights = ray_weights[selected]
    ray_weights /= np.sum(ray_weights)
    return apparent, ray_weights


def _soller_transmission(
    angle_rad: NDArray[np.float64], full_width_deg: float | None
) -> NDArray[np.float64]:
    if full_width_deg is None:
        return np.ones_like(angle_rad)
    return np.maximum(0.0, 1.0 - np.abs(2.0 * np.rad2deg(angle_rad) / full_width_deg))


def _weighted_tch_profiles(
    grid: NDArray[np.float64],
    centres: NDArray[np.float64],
    weights: NDArray[np.float64],
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
) -> NDArray[np.float64]:
    result = np.zeros_like(grid)
    for begin in range(0, centres.size, _TARGET_PROFILE_CHUNK_SIZE):
        end = min(begin + _TARGET_PROFILE_CHUNK_SIZE, centres.size)
        delta = grid[None, :] - centres[begin:end, None]
        evaluated = reference.profile_tch(delta, gaussian_fwhm, lorentzian_fwhm)
        result += weights[begin:end] @ evaluated.value
    return result


def _initial_parameters(model: BraggBrentanoFundamentalProfile) -> NDArray[np.float64]:
    weights = model.components.normalized_intensities
    wavelengths = model.components.wavelengths_angstrom
    gaussian_sigmas = np.array(
        [line.gaussian_fwhm_angstrom / _GAUSSIAN_FWHM_PER_SIGMA for line in model.emission_lines]
    )
    lorentzian_widths = np.array([line.lorentzian_fwhm_angstrom for line in model.emission_lines])
    relative_gaussian_sigma = float(np.sqrt(np.sum(weights * (gaussian_sigmas / wavelengths) ** 2)))
    relative_lorentzian_fwhm = float(np.sum(weights * lorentzian_widths / wavelengths))
    u = np.rad2deg(2.0 * relative_gaussian_sigma) ** 2
    y = np.rad2deg(2.0 * relative_lorentzian_fwhm)
    source_angular_width = np.rad2deg(model.source_width_mm / model.radius_mm)
    detector_angular_width = np.rad2deg(model.receiving_slit_width_mm / model.radius_mm)
    w = (source_angular_width**2 + detector_angular_width**2) / 12.0
    w = max(w, 1.0e-8)
    sh_over_l = (model.sample_half_length_mm + model.detector_half_length_mm) / model.radius_mm
    return np.array([u, 0.0, w, 0.0, y, sh_over_l], dtype=np.float64)


def _instrument(
    wavelength_angstrom: float, parameters: NDArray[np.float64]
) -> ConstantWavelengthInstrument:
    return ConstantWavelengthInstrument(
        float(wavelength_angstrom),
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        float(parameters[3]),
        float(parameters[4]),
    )


def _candidate(
    grid: NDArray[np.float64],
    positions: NDArray[np.float64],
    slices: tuple[slice, ...],
    target: NDArray[np.float64],
    components: WavelengthComponents,
    parameters: NDArray[np.float64],
    options: FundamentalProfileCalibrationOptions,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    result = accumulate_cw_fcj_components(
        grid,
        positions,
        np.ones(positions.size),
        _instrument(float(components.wavelengths_angstrom[0]), parameters),
        components,
        FcjGeometry(float(parameters[5] / 2.0), float(parameters[5] / 2.0)),
        support_fwhm=options.support_fwhm,
        jacobian_layout="dense",
    )
    unit = result.y
    global_rows = result.derivatives.global_jacobian
    jacobian = np.column_stack(
        (
            global_rows[0],
            global_rows[1],
            global_rows[2],
            global_rows[3],
            global_rows[4],
            0.5 * (global_rows[5] + global_rows[6]),
        )
    )
    calculated = np.empty_like(unit)
    scaled_jacobian = np.empty_like(jacobian)
    peak_scales = np.empty(positions.size)
    for index, active in enumerate(slices):
        unit_peak = unit[active]
        target_peak = target[active]
        jacobian_peak = jacobian[active]
        denominator = float(unit_peak @ unit_peak)
        if denominator <= 0.0:
            raise ValueError("candidate profile has zero norm inside an isolated window")
        numerator = float(target_peak @ unit_peak)
        unconstrained_scale = numerator / denominator
        scale = max(unconstrained_scale, 0.0)
        peak_scales[index] = scale
        calculated[active] = scale * unit_peak
        if unconstrained_scale > 0.0:
            d_numerator = target_peak @ jacobian_peak
            d_denominator = 2.0 * (unit_peak @ jacobian_peak)
            d_scale = (d_numerator * denominator - numerator * d_denominator) / denominator**2
            scaled_jacobian[active] = scale * jacobian_peak + unit_peak[:, None] * d_scale[None, :]
        else:
            scaled_jacobian[active] = 0.0
    return calculated, scaled_jacobian, peak_scales


def _weighted_system(
    target: NDArray[np.float64],
    calculated: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    slices: tuple[slice, ...],
    target_norms: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    residual = target - calculated
    weighted_jacobian = jacobian.copy()
    for active, norm in zip(slices, target_norms, strict=True):
        residual[active] /= norm
        weighted_jacobian[active] /= norm
    return residual, weighted_jacobian


def _bounded_gauss_newton_step(
    columns: NDArray[np.float64],
    residual: NDArray[np.float64],
    damping: float,
    parameters: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    free = np.ones(parameters.size, dtype=np.bool_)
    scaled_step = np.zeros(parameters.size, dtype=np.float64)
    for _active_set_pass in range(parameters.size + 1):
        selected = columns[:, free]
        normal = selected.T @ selected + damping * np.eye(int(np.sum(free)))
        gradient = selected.T @ residual
        try:
            scaled_step[free] = np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            scaled_step[free] = np.linalg.lstsq(selected, residual, rcond=None)[0]
        physical_step = scales * scaled_step
        at_lower = np.isclose(parameters, lower, rtol=0.0, atol=1.0e-14)
        at_upper = np.isclose(parameters, upper, rtol=0.0, atol=1.0e-14)
        points_outside = (at_lower & (physical_step < 0.0)) | (at_upper & (physical_step > 0.0))
        newly_fixed = free & points_outside
        if not np.any(newly_fixed):
            return scaled_step
        free[newly_fixed] = False
        scaled_step[newly_fixed] = 0.0
    return scaled_step


def _peak_diagnostics(
    positions: NDArray[np.float64],
    slices: tuple[slice, ...],
    target: NDArray[np.float64],
    calculated: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> tuple[FundamentalPeakDiagnostic, ...]:
    diagnostics = []
    for position, active, scale in zip(positions, slices, scales, strict=True):
        target_peak = target[active]
        calculated_peak = calculated[active]
        relative_l2 = float(
            np.linalg.norm(calculated_peak - target_peak) / np.linalg.norm(target_peak)
        )
        correlation = float(np.corrcoef(target_peak, calculated_peak)[0, 1])
        diagnostics.append(
            FundamentalPeakDiagnostic(float(position), float(scale), relative_l2, correlation)
        )
    return tuple(diagnostics)


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result
