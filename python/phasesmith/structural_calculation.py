"""Scriptable structural-pattern calculation for one Rietveld phase."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import _vector
from .crystallography import calculate_structure_factor_values, p1_parameter_names
from .cw import CW_GLOBAL_PARAMETER_ORDER, CW_LOCAL_PARAMETER_ORDER, accumulate_cw_contributions
from .execution import ExecutionPolicy
from .extensions import (
    CompositePhysicsProvider,
    PhysicsContext,
    PhysicsContribution,
    evaluate_provider,
)
from .intensity_corrections import (
    BraggBrentanoPolarizedLp,
    BraggBrentanoUnpolarizedLp,
    ConstantWavelengthNeutronLorentz,
    NeutralIntegratedIntensityCorrection,
)
from .pattern import (
    PowderPattern,
    StructuralPatternCalculationResult,
    StructuralPatternJvpResult,
    StructuralPatternLinearizationResult,
    StructuralPatternVjpResult,
    StructuralReflectionResult,
)
from .phase import ReflectionGeometryBatch, RietveldPhase
from .radiation import (
    BraggBrentanoGeometry,
    ComponentRadiation,
    ConstantWavelengthExperiment,
    DebyeScherrerGeometry,
    MonochromaticRadiation,
    RadiationProbe,
)
from .results import AccumulationResult, _build_accumulation_result
from .sample import (
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
)
from .scattering import (
    NeutronNuclear,
    XrayFixedDispersion,
    XrayNonResonant,
    species_from_structure,
)


def _freeze(array: NDArray[np.generic]) -> None:
    array.flags.writeable = False


def _check_probe(phase: RietveldPhase, experiment: ConstantWavelengthExperiment) -> None:
    expected = "xray" if experiment.radiation.probe is RadiationProbe.X_RAY else "neutron"
    if phase.scattering.descriptor.probe != expected:
        raise ValueError(
            f"{phase.scattering.descriptor.probe} scattering is incompatible with "
            f"the {experiment.radiation.probe.value} experiment"
        )
    correction = phase.intensity_correction
    if isinstance(correction, ConstantWavelengthNeutronLorentz):
        if experiment.radiation.probe is not RadiationProbe.NEUTRON:
            raise ValueError("constant-wavelength neutron Lorentz correction requires neutrons")
        if correction.wavelength_angstrom != experiment.radiation.wavelength_angstrom:
            raise ValueError("correction and experiment wavelengths must match exactly")
    if isinstance(correction, (BraggBrentanoUnpolarizedLp, BraggBrentanoPolarizedLp)):
        if experiment.radiation.probe is not RadiationProbe.X_RAY:
            raise ValueError("Bragg-Brentano polarization correction requires X-ray radiation")
        if correction.wavelength_angstrom != experiment.radiation.wavelength_angstrom:
            raise ValueError("correction and experiment wavelengths must match exactly")


def _geometry(
    phase: RietveldPhase,
    experiment: ConstantWavelengthExperiment,
    base_intensity: ArrayLike,
) -> ReflectionGeometryBatch:
    spacing = phase.structure.cell.d_spacings(phase.reflections.hkl).d_spacing_angstrom
    sin_theta = experiment.radiation.wavelength_angstrom / (2.0 * spacing)
    if np.any((sin_theta <= 0.0) | (sin_theta >= 1.0)):
        raise ValueError(
            "all structural reflections must lie strictly within 0 < 2theta < 180 degrees"
        )
    base_two_theta = np.ascontiguousarray(2.0 * np.degrees(np.arcsin(sin_theta)))
    two_theta = base_two_theta + experiment.zero_shift_deg
    if isinstance(experiment.geometry, BraggBrentanoGeometry):
        theta = np.radians(0.5 * base_two_theta)
        two_theta -= np.degrees(
            2.0
            * experiment.geometry.sample_displacement_mm
            / experiment.geometry.goniometer_radius_mm
            * np.cos(theta)
        )
    elif isinstance(experiment.geometry, DebyeScherrerGeometry):
        position_radians = np.radians(base_two_theta)
        scale = 0.18 / (np.pi * experiment.geometry.goniometer_radius_mm)
        two_theta -= scale * (
            experiment.geometry.displace_x_micrometre * np.cos(position_radians)
            + experiment.geometry.displace_y_micrometre * np.sin(position_radians)
        )
    return ReflectionGeometryBatch(
        phase.reflections.hkl,
        spacing,
        two_theta,
        base_intensity,
    )


def _structural_parameter_names(phase: RietveldPhase) -> tuple[str, ...]:
    return p1_parameter_names(phase.structure.to_site_batch())


def _native_model_configuration(
    phase: RietveldPhase,
) -> tuple[str, str, float | None, float | None] | None:
    if type(phase.scattering) in (XrayNonResonant, XrayFixedDispersion):
        scattering_model = "xray_non_resonant"
    elif type(phase.scattering) is NeutronNuclear:
        scattering_model = "neutron_nuclear"
    else:
        return None
    if type(phase.intensity_correction) is NeutralIntegratedIntensityCorrection:
        return scattering_model, "neutral", None, None
    if type(phase.intensity_correction) is BraggBrentanoUnpolarizedLp:
        return (
            scattering_model,
            "bragg_brentano_unpolarized_lp",
            phase.intensity_correction.wavelength_angstrom,
            None,
        )
    if type(phase.intensity_correction) is BraggBrentanoPolarizedLp:
        return (
            scattering_model,
            "bragg_brentano_polarized_lp",
            phase.intensity_correction.wavelength_angstrom,
            phase.intensity_correction.polarization,
        )
    if type(phase.intensity_correction) is ConstantWavelengthNeutronLorentz:
        return (
            scattering_model,
            "constant_wavelength_neutron_lorentz",
            phase.intensity_correction.wavelength_angstrom,
            None,
        )
    return None


def _supports_fused_structural_physics(provider: object | None) -> bool:
    if provider is None or type(provider) in (
        IsotropicSizeBroadening,
        IsotropicMicrostrainBroadening,
        MarchDollasePreferredOrientation,
    ):
        return True
    return type(provider) is CompositePhysicsProvider and all(
        _supports_fused_structural_physics(child) for child in provider.providers
    )


def _physics_provider_is_thread_safe(provider: object | None) -> bool:
    if provider is None:
        return True
    if type(provider) is CompositePhysicsProvider:
        return all(_physics_provider_is_thread_safe(child) for child in provider.providers)
    descriptor = getattr(provider, "descriptor", None)
    return getattr(descriptor, "thread_safe", False) is True


def _fallback_is_thread_safe(phase: RietveldPhase) -> bool:
    scattering_descriptor = getattr(phase.scattering, "descriptor", None)
    return (
        getattr(scattering_descriptor, "thread_safe", False) is True
        and getattr(phase.intensity_correction, "thread_safe", False) is True
        and _physics_provider_is_thread_safe(phase.physics)
    )


def _native_phase(phase: RietveldPhase, execution: ExecutionPolicy) -> object | None:
    configuration = _native_model_configuration(phase)
    if configuration is None or not _supports_fused_structural_physics(phase.physics):
        return None
    scattering_model, correction_model, correction_wavelength, correction_polarization = (
        configuration
    )
    sites = phase.structure.to_site_batch()
    species = species_from_structure(phase.structure)
    species_keys = (
        [value.xray_key for value in species]
        if scattering_model == "xray_non_resonant"
        else [value.neutron_key for value in species]
    )
    offsets = (
        phase.scattering.corrections_for(species)
        if type(phase.scattering) is XrayFixedDispersion
        else np.empty(0, dtype=np.complex128)
    )
    return _core._StructuralPhase(
        phase.structure.space_group._native,
        np.ascontiguousarray(phase.reflections.hkl.reshape(-1)),
        phase.reflections.multiplicity,
        np.ascontiguousarray(sites.fractional_xyz.reshape(-1)),
        sites.occupancy,
        sites.u_iso_angstrom2,
        sites.anisotropic_mask,
        np.ascontiguousarray(sites.u_aniso_cif_angstrom2.reshape(-1)),
        species_keys,
        np.ascontiguousarray(offsets.real),
        np.ascontiguousarray(offsets.imag),
        *phase.structure.cell.as_tuple(),
        float(phase.scale),
        float(phase.coordinate_tolerance),
        execution._native,
        scattering_model,
        correction_model,
        correction_wavelength,
        correction_polarization,
    )


def _native_dynamic_arguments(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    contribution: PhysicsContribution,
    support_fwhm: float,
) -> tuple[object, ...]:
    instrument = experiment.instrument
    geometry = experiment.geometry
    axial = experiment.axial_geometry
    return (
        pattern.x,
        instrument.wavelength_angstrom,
        experiment.zero_shift_deg,
        (geometry.sample_displacement_mm if isinstance(geometry, BraggBrentanoGeometry) else None),
        (geometry.displace_x_micrometre if isinstance(geometry, DebyeScherrerGeometry) else None),
        (geometry.displace_y_micrometre if isinstance(geometry, DebyeScherrerGeometry) else None),
        None if geometry is None else geometry.goniometer_radius_mm,
        None if axial is None else axial.sample_over_radius,
        None if axial is None else axial.detector_over_radius,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        contribution.gaussian_variance_deg2,
        contribution.lorentzian_fwhm_deg,
        contribution.intensity_multiplier,
        contribution.d_gaussian_variance_d_position,
        contribution.d_lorentzian_fwhm_d_position,
        contribution.d_intensity_multiplier_d_position,
        contribution.d_gaussian_variance_d_parameters.reshape(-1),
        contribution.d_lorentzian_fwhm_d_parameters.reshape(-1),
        contribution.d_intensity_multiplier_d_parameters.reshape(-1),
        len(contribution.parameter_names),
        float(support_fwhm),
    )


def _native_spectrum_dynamic_arguments(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    support_fwhm: float,
) -> tuple[object, ...]:
    instrument = experiment.instrument
    geometry = experiment.geometry
    axial = experiment.axial_geometry
    return (
        pattern.x,
        experiment.zero_shift_deg,
        (geometry.sample_displacement_mm if isinstance(geometry, BraggBrentanoGeometry) else None),
        (geometry.displace_x_micrometre if isinstance(geometry, DebyeScherrerGeometry) else None),
        (geometry.displace_y_micrometre if isinstance(geometry, DebyeScherrerGeometry) else None),
        None if geometry is None else geometry.goniometer_radius_mm,
        None if axial is None else axial.sample_over_radius,
        None if axial is None else axial.detector_over_radius,
        instrument.wavelength_angstrom,
        instrument.u_deg2,
        instrument.v_deg2,
        instrument.w_deg2,
        instrument.x_deg,
        instrument.y_deg,
        float(support_fwhm),
    )


def _accumulation_from_native(
    arrays: tuple[object, ...],
    contribution: PhysicsContribution,
    experiment: ConstantWavelengthExperiment,
    jacobian_layout: Literal["support", "dense"],
    *,
    fixed_spectrum: bool = False,
) -> AccumulationResult:
    position_names = ("wavelength_angstrom", "zero_shift_deg")
    if experiment.geometry is not None:
        if isinstance(experiment.geometry, BraggBrentanoGeometry):
            position_names += ("sample_displacement_mm",)
        else:
            position_names += ("displace_x_micrometre", "displace_y_micrometre")
    axial_names: tuple[str, ...] = ()
    if experiment.axial_geometry is not None:
        axial_names = ("sample_over_radius", "detector_over_radius")
    global_names = (
        CW_GLOBAL_PARAMETER_ORDER + position_names + axial_names + contribution.parameter_names
    )
    if fixed_spectrum:
        global_names = tuple(name for name in global_names if name != "wavelength_angstrom")
    return _build_accumulation_result(
        *arrays,
        CW_LOCAL_PARAMETER_ORDER,
        global_names,
        jacobian_layout,
    )


def _reflection_result(
    phase: RietveldPhase,
    arrays: tuple[object, ...],
    component_count: int | None = None,
) -> StructuralReflectionResult:
    f_real, f_imag, f_squared, intensity, q_squared, s, d_spacing, two_theta = (
        np.asarray(value) for value in arrays
    )
    f = np.asarray(f_real + 1j * f_imag, dtype=np.complex128)
    for array in (f, f_squared, intensity, q_squared, s, d_spacing, two_theta):
        _freeze(array)
    reflection_count = phase.reflections.reflection_count
    if component_count is None:
        ids = phase.reflections.reflection_ids
        component_index = None
        base_index = None
    else:
        ids = tuple(
            f"{reflection_id}@component[{component}]"
            for component in range(component_count)
            for reflection_id in phase.reflections.reflection_ids
        )
        component_index = np.repeat(np.arange(component_count, dtype=np.int64), reflection_count)
        base_index = np.tile(np.arange(reflection_count, dtype=np.int64), component_count)
    return StructuralReflectionResult(
        ids,
        f,
        f_squared,
        intensity,
        q_squared,
        s,
        d_spacing,
        two_theta,
        component_index,
        base_index,
    )


def _native_spectrum(
    phase: RietveldPhase,
    experiment: ConstantWavelengthExperiment,
    components: tuple[PreparedStructuralPattern, ...],
    execution: ExecutionPolicy,
) -> object | None:
    radiation = experiment.radiation
    if not isinstance(radiation, ComponentRadiation):
        return None
    base = _native_phase(phase, execution)
    contributions = tuple(component._contribution for component in components)
    if base is None or any(value is None for value in contributions):
        return None
    selected = cast(tuple[PhysicsContribution, ...], contributions)
    parameter_names = selected[0].parameter_names
    if any(value.parameter_names != parameter_names for value in selected[1:]):
        return None

    def flattened(name: str) -> NDArray[np.float64]:
        return np.ascontiguousarray(
            np.concatenate([np.asarray(getattr(value, name)).reshape(-1) for value in selected])
        )

    return _core._StructuralSpectrum(
        base,
        radiation.components.wavelengths_angstrom,
        radiation.components.relative_intensities,
        flattened("gaussian_variance_deg2"),
        flattened("lorentzian_fwhm_deg"),
        flattened("intensity_multiplier"),
        flattened("d_gaussian_variance_d_position"),
        flattened("d_lorentzian_fwhm_d_position"),
        flattened("d_intensity_multiplier_d_position"),
        flattened("d_gaussian_variance_d_parameters"),
        flattened("d_lorentzian_fwhm_d_parameters"),
        flattened("d_intensity_multiplier_d_parameters"),
        len(parameter_names),
        execution._native,
    )


def _native_prepared_model(
    native: object | None,
    contribution: PhysicsContribution | None,
) -> object | None:
    if native is None or contribution is None:
        return None
    return _core._PreparedStructuralModel.monochromatic(
        native,
        contribution.gaussian_variance_deg2,
        contribution.lorentzian_fwhm_deg,
        contribution.intensity_multiplier,
        contribution.d_gaussian_variance_d_position,
        contribution.d_lorentzian_fwhm_d_position,
        contribution.d_intensity_multiplier_d_position,
        contribution.d_gaussian_variance_d_parameters.reshape(-1),
        contribution.d_lorentzian_fwhm_d_parameters.reshape(-1),
        contribution.d_intensity_multiplier_d_parameters.reshape(-1),
        len(contribution.parameter_names),
    )


def _calculation_result(
    phase: RietveldPhase,
    pattern: PowderPattern,
    accumulation: AccumulationResult,
    reflections: StructuralReflectionResult,
) -> StructuralPatternCalculationResult:
    profile_y = np.array(accumulation.y, copy=True)
    y = profile_y + pattern.background
    _freeze(profile_y)
    _freeze(y)
    return StructuralPatternCalculationResult(
        phase.phase_id,
        y,
        profile_y,
        pattern.background,
        accumulation,
        reflections,
    )


def _component_inputs(
    experiment: ConstantWavelengthExperiment,
    phase: RietveldPhase,
) -> tuple[tuple[ConstantWavelengthExperiment, RietveldPhase, float], ...]:
    radiation = experiment.radiation
    if not isinstance(radiation, ComponentRadiation):
        return ()
    if not isinstance(
        phase.intensity_correction,
        (
            NeutralIntegratedIntensityCorrection,
            BraggBrentanoUnpolarizedLp,
            BraggBrentanoPolarizedLp,
            ConstantWavelengthNeutronLorentz,
        ),
    ):
        raise NotImplementedError(
            "fixed structural wavelength components require a built-in neutral or "
            "Bragg-Brentano intensity correction"
        )
    weights = radiation.components.normalized_intensities
    prepared = []
    for wavelength, weight in zip(radiation.components.wavelengths_angstrom, weights, strict=True):
        component_instrument = replace(
            experiment.instrument,
            wavelength_angstrom=float(wavelength),
        )
        component_experiment = ConstantWavelengthExperiment(
            MonochromaticRadiation(radiation.probe, float(wavelength)),
            component_instrument,
            experiment.zero_shift_deg,
            experiment.geometry,
            experiment.axial_geometry,
        )
        correction = phase.intensity_correction
        if isinstance(correction, BraggBrentanoUnpolarizedLp):
            correction = BraggBrentanoUnpolarizedLp(float(wavelength))
        elif isinstance(correction, BraggBrentanoPolarizedLp):
            correction = BraggBrentanoPolarizedLp(
                float(wavelength),
                correction.polarization,
            )
        elif isinstance(correction, ConstantWavelengthNeutronLorentz):
            correction = ConstantWavelengthNeutronLorentz(float(wavelength))
        component_phase = replace(
            phase,
            scale=phase.scale * float(weight),
            intensity_correction=correction,
        )
        prepared.append((component_experiment, component_phase, float(weight)))
    return tuple(prepared)


def _combine_component_accumulations(
    results: tuple[StructuralPatternCalculationResult, ...],
    jacobian_layout: Literal["support", "dense"],
) -> AccumulationResult:
    if not results:
        raise ValueError("component calculation requires at least one result")
    sample_count = results[0].profile_y.size
    local_names = results[0].derivatives.local_parameter_names
    global_names = tuple(
        name
        for name in results[0].derivatives.global_parameter_names
        if name != "wavelength_angstrom"
    )
    starts = []
    offsets = [0]
    values = []
    global_jacobian = np.zeros((len(global_names), sample_count), dtype=np.float64)
    y = np.zeros(sample_count, dtype=np.float64)
    for result in results:
        if result.profile_y.size != sample_count:
            raise ValueError("component calculations must share the sample grid")
        if result.derivatives.local_parameter_names != local_names:
            raise ValueError("component local derivative names must match")
        selected_global_names = tuple(
            name
            for name in result.derivatives.global_parameter_names
            if name != "wavelength_angstrom"
        )
        if selected_global_names != global_names:
            raise ValueError("component global derivative names must match")
        local = result.derivatives.local
        starts.append(local.starts)
        values.append(local.values)
        cursor = offsets[-1]
        offsets.extend((local.offsets[1:] + cursor).tolist())
        selected_rows = [
            result.derivatives.global_parameter_names.index(name) for name in global_names
        ]
        global_jacobian += result.derivatives.global_jacobian[selected_rows]
        y += result.profile_y
    starts_array = np.ascontiguousarray(np.concatenate(starts), dtype=np.int64)
    offsets_array = np.ascontiguousarray(offsets, dtype=np.int64)
    values_array = np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float64)
    y = np.ascontiguousarray(y)
    for array in (starts_array, offsets_array, values_array, global_jacobian, y):
        array.flags.writeable = False
    return _build_accumulation_result(
        y,
        starts_array,
        offsets_array,
        values_array,
        global_jacobian,
        local_names,
        global_names,
        jacobian_layout,
    )


def _combine_component_reflections(
    phase: RietveldPhase,
    results: tuple[StructuralPatternCalculationResult, ...],
) -> StructuralReflectionResult:
    reflection_count = phase.reflections.reflection_count
    ids = tuple(
        f"{reflection_id}@component[{component}]"
        for component in range(len(results))
        for reflection_id in phase.reflections.reflection_ids
    )
    component_index = np.repeat(np.arange(len(results), dtype=np.int64), reflection_count)
    base_index = np.tile(np.arange(reflection_count, dtype=np.int64), len(results))
    return StructuralReflectionResult(
        ids,
        np.ascontiguousarray(np.concatenate([result.reflections.f for result in results])),
        np.ascontiguousarray(np.concatenate([result.reflections.f_squared for result in results])),
        np.ascontiguousarray(
            np.concatenate([result.reflections.integrated_intensity for result in results])
        ),
        np.ascontiguousarray(
            np.concatenate([result.reflections.q_squared_inverse_angstrom2 for result in results])
        ),
        np.ascontiguousarray(
            np.concatenate([result.reflections.s_inverse_angstrom for result in results])
        ),
        np.ascontiguousarray(
            np.concatenate([result.reflections.d_spacing_angstrom for result in results])
        ),
        np.ascontiguousarray(
            np.concatenate([result.reflections.two_theta_deg for result in results])
        ),
        component_index,
        base_index,
    )


def _combine_component_results(
    phase: RietveldPhase,
    pattern: PowderPattern,
    results: tuple[StructuralPatternCalculationResult, ...],
    jacobian_layout: Literal["support", "dense"],
) -> StructuralPatternCalculationResult:
    return _calculation_result(
        phase,
        pattern,
        _combine_component_accumulations(results, jacobian_layout),
        _combine_component_reflections(phase, results),
    )


def _fallback_calculate(
    phase: RietveldPhase,
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    support_fwhm: float,
    jacobian_layout: Literal["support", "dense"],
) -> StructuralPatternCalculationResult:
    structural = calculate_structure_factor_values(
        phase.structure,
        phase.reflections.hkl,
        phase.reflections.multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
        scale=phase.scale,
        coordinate_tolerance=phase.coordinate_tolerance,
    )
    geometry = _geometry(phase, experiment, structural.integrated_intensity)
    contribution = (
        PhysicsContribution.neutral(phase.reflections.reflection_count)
        if phase.physics is None
        else evaluate_provider(
            phase.physics,
            PhysicsContext(geometry, experiment.instrument, phase.structure.cell),
        )
    )
    accumulation = accumulate_cw_contributions(
        pattern.x,
        geometry.two_theta_deg,
        structural.integrated_intensity,
        experiment.instrument,
        contribution,
        geometry=experiment.axial_geometry,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    )
    reflections = StructuralReflectionResult(
        phase.reflections.reflection_ids,
        structural.f,
        structural.f_squared,
        structural.integrated_intensity,
        structural.q_squared_inverse_angstrom2,
        structural.s_inverse_angstrom,
        geometry.d_spacing_angstrom,
        geometry.two_theta_deg,
    )
    return _calculation_result(phase, pattern, accumulation, reflections)


@dataclass(frozen=True, slots=True, init=False)
class PreparedStructuralPattern:
    """Reusable one-phase structural pattern with a native built-in fast path.

    Built-in scattering and correction models with no Python physics provider
    use one native values/JVP/VJP call. Third-party providers use a fully
    vectorized fallback and are each evaluated exactly once per calculation.
    """

    pattern: PowderPattern
    experiment: ConstantWavelengthExperiment
    phase: RietveldPhase
    support_fwhm: float
    jacobian_layout: Literal["support", "dense"]
    execution: ExecutionPolicy
    _native: object | None
    _contribution: PhysicsContribution | None
    _components: tuple[PreparedStructuralPattern, ...]
    _component_weights: tuple[float, ...]
    _native_spectrum: object | None
    _native_model: object | None

    def __init__(
        self,
        pattern: PowderPattern,
        experiment: ConstantWavelengthExperiment,
        phase: RietveldPhase,
        *,
        support_fwhm: float = 20.0,
        jacobian_layout: Literal["support", "dense"] = "support",
        execution: ExecutionPolicy | None = None,
    ) -> None:
        """Validate immutable inputs and prepare native structural topology."""

        if not isinstance(pattern, PowderPattern):
            raise TypeError("pattern must be a PowderPattern")
        if not isinstance(experiment, ConstantWavelengthExperiment):
            raise TypeError("experiment must be a ConstantWavelengthExperiment")
        if not isinstance(phase, RietveldPhase):
            raise TypeError("phase must be a RietveldPhase")
        if not np.isfinite(support_fwhm) or support_fwhm <= 0.0:
            raise ValueError("support_fwhm must be positive and finite")
        if jacobian_layout not in ("support", "dense"):
            raise ValueError("jacobian_layout must be 'support' or 'dense'")
        selected_execution = ExecutionPolicy() if execution is None else execution
        if not isinstance(selected_execution, ExecutionPolicy):
            raise TypeError("execution must be an ExecutionPolicy")
        _check_probe(phase, experiment)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "experiment", experiment)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "support_fwhm", float(support_fwhm))
        object.__setattr__(self, "jacobian_layout", jacobian_layout)
        object.__setattr__(self, "execution", selected_execution)
        component_inputs = _component_inputs(experiment, phase)
        if component_inputs:
            object.__setattr__(self, "_native", None)
            object.__setattr__(self, "_contribution", None)
            components = tuple(
                PreparedStructuralPattern(
                    pattern,
                    component_experiment,
                    component_phase,
                    support_fwhm=support_fwhm,
                    jacobian_layout=jacobian_layout,
                    execution=selected_execution,
                )
                for component_experiment, component_phase, _weight in component_inputs
            )
            object.__setattr__(self, "_components", components)
            object.__setattr__(
                self,
                "_component_weights",
                tuple(weight for _experiment, _phase, weight in component_inputs),
            )
            native_spectrum = _native_spectrum(
                phase, experiment, components, selected_execution
            )
            object.__setattr__(
                self,
                "_native_spectrum",
                native_spectrum,
            )
            object.__setattr__(
                self,
                "_native_model",
                (
                    None
                    if native_spectrum is None
                    else _core._PreparedStructuralModel.fixed_spectrum(native_spectrum)
                ),
            )
            return
        native = _native_phase(phase, selected_execution)
        contribution = None
        if native is not None:
            geometry = _geometry(
                phase,
                experiment,
                np.ones(phase.reflections.reflection_count, dtype=np.float64),
            )
            contribution = (
                PhysicsContribution.neutral(phase.reflections.reflection_count)
                if phase.physics is None
                else evaluate_provider(
                    phase.physics,
                    PhysicsContext(geometry, experiment.instrument, phase.structure.cell),
                )
            )
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_contribution", contribution)
        object.__setattr__(self, "_components", ())
        object.__setattr__(self, "_component_weights", ())
        object.__setattr__(self, "_native_spectrum", None)
        object.__setattr__(self, "_native_model", _native_prepared_model(native, contribution))

    @property
    def uses_native_fused_path(self) -> bool:
        """Report whether values and structural derivatives stay in one native call."""

        if self._native_spectrum is not None:
            return True
        if self._components:
            return all(component.uses_native_fused_path for component in self._components)
        return self._native is not None

    @property
    def parallel_safe(self) -> bool:
        """Return whether independent calls may overlap on worker threads."""

        if self._native_spectrum is not None:
            return True
        if self._components:
            return all(component.parallel_safe for component in self._components)
        return self._native is not None or _fallback_is_thread_safe(self.phase)

    @property
    def leaf_count(self) -> int:
        """Return the number of independently schedulable component leaves."""

        return 1 if self._native_spectrum is not None else len(self._components) or 1

    def _calculation_tasks(self) -> tuple[PreparedStructuralPattern, ...]:
        if self._native_spectrum is not None:
            return (self,)
        return self._components if self._components else (self,)

    def _combine_calculations(
        self,
        results: tuple[StructuralPatternCalculationResult, ...],
    ) -> StructuralPatternCalculationResult:
        if self._native_spectrum is not None:
            if len(results) != 1:
                raise ValueError("one native spectrum result is required")
            return results[0]
        if not self._components:
            if len(results) != 1:
                raise ValueError("one monochromatic result is required")
            return results[0]
        if len(results) != len(self._components):
            raise ValueError("component result count does not match prepared components")
        return _combine_component_results(
            self.phase,
            self.pattern,
            results,
            self.jacobian_layout,
        )

    def _calculation_from_native_arrays(
        self,
        arrays: tuple[object, ...],
    ) -> StructuralPatternCalculationResult:
        if self._native_model is None:
            raise RuntimeError("native structural model was not prepared")
        contribution = (
            self._components[0]._contribution
            if self._native_spectrum is not None
            else self._contribution
        )
        if contribution is None:  # pragma: no cover - preparation invariant
            raise RuntimeError("native structural contribution was not prepared")
        accumulation_arrays, reflection_arrays = arrays
        return _calculation_result(
            self.phase,
            self.pattern,
            _accumulation_from_native(
                accumulation_arrays,
                contribution,
                self.experiment,
                self.jacobian_layout,
                fixed_spectrum=self._native_spectrum is not None,
            ),
            _reflection_result(
                self.phase,
                reflection_arrays,
                (len(self._components) if self._native_spectrum is not None else None),
            ),
        )

    def _linearization_from_native_arrays(
        self,
        arrays: tuple[object, ...],
    ) -> StructuralPatternLinearizationResult:
        result_arrays, jacobian = arrays
        _freeze(jacobian)
        return StructuralPatternLinearizationResult(
            self._calculation_from_native_arrays(result_arrays),
            _structural_parameter_names(self.phase),
            jacobian,
        )

    def _jvp_from_native_arrays(self, arrays: tuple[object, ...]) -> StructuralPatternJvpResult:
        result_arrays, d_y, d_intensity, d_position = arrays
        for array in (d_y, d_intensity, d_position):
            _freeze(array)
        return StructuralPatternJvpResult(
            self._calculation_from_native_arrays(result_arrays),
            _structural_parameter_names(self.phase),
            d_y,
            d_intensity,
            d_position,
        )

    def _vjp_from_native_arrays(self, arrays: tuple[object, ...]) -> StructuralPatternVjpResult:
        result_arrays, gradient = arrays
        _freeze(gradient)
        return StructuralPatternVjpResult(
            self._calculation_from_native_arrays(result_arrays),
            _structural_parameter_names(self.phase),
            gradient,
        )

    def calculate(self) -> StructuralPatternCalculationResult:
        """Calculate structural intensities, positions, and the powder profile."""

        if self._native_spectrum is not None:
            contribution = self._components[0]._contribution
            if contribution is None:  # pragma: no cover - preparation invariant
                raise RuntimeError("native spectrum contribution was not prepared")
            accumulation_arrays, reflection_arrays = self._native_spectrum.calculate(
                *_native_spectrum_dynamic_arguments(
                    self.pattern,
                    self.experiment,
                    self.support_fwhm,
                )
            )
            accumulation = _accumulation_from_native(
                accumulation_arrays,
                contribution,
                self.experiment,
                self.jacobian_layout,
                fixed_spectrum=True,
            )
            return _calculation_result(
                self.phase,
                self.pattern,
                accumulation,
                _reflection_result(
                    self.phase,
                    reflection_arrays,
                    len(self._components),
                ),
            )
        if self._components:
            return self._combine_calculations(
                tuple(component.calculate() for component in self._components)
            )
        if self._native is None:
            return _fallback_calculate(
                self.phase,
                self.pattern,
                self.experiment,
                self.support_fwhm,
                self.jacobian_layout,
            )
        contribution = self._contribution
        if contribution is None:  # pragma: no cover - native/contribution invariant
            raise RuntimeError("native structural contribution was not prepared")
        accumulation_arrays, reflection_arrays = self._native.calculate(
            *_native_dynamic_arguments(
                self.pattern,
                self.experiment,
                contribution,
                self.support_fwhm,
            )
        )
        accumulation = _accumulation_from_native(
            accumulation_arrays,
            contribution,
            self.experiment,
            self.jacobian_layout,
        )
        return _calculation_result(
            self.phase,
            self.pattern,
            accumulation,
            _reflection_result(self.phase, reflection_arrays),
        )

    def jvp(self, tangent: ArrayLike) -> StructuralPatternJvpResult:
        """Calculate one structural JVP without a dense pattern Jacobian."""

        if self._native_spectrum is not None:
            names = _structural_parameter_names(self.phase)
            direction = _vector(tangent, "tangent")
            if direction.shape != (len(names),):
                raise ValueError("tangent must match the structural parameter count")
            contribution = self._components[0]._contribution
            if contribution is None:  # pragma: no cover - preparation invariant
                raise RuntimeError("native spectrum contribution was not prepared")
            native_result, d_y, d_intensity, d_position = self._native_spectrum.jvp(
                direction,
                *_native_spectrum_dynamic_arguments(
                    self.pattern,
                    self.experiment,
                    self.support_fwhm,
                ),
            )
            accumulation_arrays, reflection_arrays = native_result
            result = _calculation_result(
                self.phase,
                self.pattern,
                _accumulation_from_native(
                    accumulation_arrays,
                    contribution,
                    self.experiment,
                    self.jacobian_layout,
                    fixed_spectrum=True,
                ),
                _reflection_result(self.phase, reflection_arrays, len(self._components)),
            )
            for array in (d_y, d_intensity, d_position):
                _freeze(array)
            return StructuralPatternJvpResult(result, names, d_y, d_intensity, d_position)
        if self._components:
            return self._combine_jvps(
                tuple(component.jvp(direction) for component, direction in self._jvp_tasks(tangent))
            )
        if self._native is None:
            raise NotImplementedError(
                "structural JVP is currently available for the built-in fused path only"
            )
        names = _structural_parameter_names(self.phase)
        direction = _vector(tangent, "tangent")
        if direction.shape != (len(names),):
            raise ValueError("tangent must match the structural parameter count")
        contribution = self._contribution
        if contribution is None:  # pragma: no cover - native/contribution invariant
            raise RuntimeError("native structural contribution was not prepared")
        native_result, d_y, d_intensity, d_position = self._native.jvp(
            direction,
            *_native_dynamic_arguments(
                self.pattern,
                self.experiment,
                contribution,
                self.support_fwhm,
            ),
        )
        accumulation_arrays, reflection_arrays = native_result
        accumulation = _accumulation_from_native(
            accumulation_arrays,
            contribution,
            self.experiment,
            self.jacobian_layout,
        )
        result = _calculation_result(
            self.phase,
            self.pattern,
            accumulation,
            _reflection_result(self.phase, reflection_arrays),
        )
        for array in (d_y, d_intensity, d_position):
            _freeze(array)
        return StructuralPatternJvpResult(result, names, d_y, d_intensity, d_position)

    def _jvp_tasks(
        self,
        tangent: ArrayLike,
    ) -> tuple[tuple[PreparedStructuralPattern, NDArray[np.float64]], ...]:
        names = _structural_parameter_names(self.phase)
        direction = _vector(tangent, "tangent")
        if direction.shape != (len(names),):
            raise ValueError("tangent must match the structural parameter count")
        if self._native_spectrum is not None or not self._components:
            return ((self, direction),)
        tasks = []
        for component, weight in zip(self._components, self._component_weights, strict=True):
            component_direction = np.array(direction, copy=True)
            component_direction[-1] *= weight
            tasks.append((component, component_direction))
        return tuple(tasks)

    def _combine_jvps(
        self,
        products: tuple[StructuralPatternJvpResult, ...],
    ) -> StructuralPatternJvpResult:
        if self._native_spectrum is not None:
            if len(products) != 1:
                raise ValueError("one native spectrum JVP is required")
            return products[0]
        if not self._components:
            if len(products) != 1:
                raise ValueError("one monochromatic JVP is required")
            return products[0]
        if len(products) != len(self._components):
            raise ValueError("component JVP count does not match prepared components")
        names = _structural_parameter_names(self.phase)
        return StructuralPatternJvpResult(
            self._combine_calculations(tuple(product.result for product in products)),
            names,
            np.ascontiguousarray(
                sum((item.d_y for item in products), np.zeros_like(self.pattern.x))
            ),
            np.ascontiguousarray(
                np.concatenate([item.d_integrated_intensity for item in products])
            ),
            np.ascontiguousarray(np.concatenate([item.d_two_theta_deg for item in products])),
        )

    def linearize(self) -> StructuralPatternLinearizationResult:
        """Calculate one reusable native structural pattern Jacobian."""

        names = _structural_parameter_names(self.phase)
        if self._native_spectrum is not None:
            contribution = self._components[0]._contribution
            if contribution is None:  # pragma: no cover - preparation invariant
                raise RuntimeError("native spectrum contribution was not prepared")
            native_result, jacobian = self._native_spectrum.linearize(
                *_native_spectrum_dynamic_arguments(
                    self.pattern,
                    self.experiment,
                    self.support_fwhm,
                )
            )
            accumulation_arrays, reflection_arrays = native_result
            result = _calculation_result(
                self.phase,
                self.pattern,
                _accumulation_from_native(
                    accumulation_arrays,
                    contribution,
                    self.experiment,
                    self.jacobian_layout,
                    fixed_spectrum=True,
                ),
                _reflection_result(self.phase, reflection_arrays, len(self._components)),
            )
            _freeze(jacobian)
            return StructuralPatternLinearizationResult(result, names, jacobian)
        if self._components:
            return self._combine_linearizations(
                tuple(component.linearize() for component in self._components)
            )
        if self._native is None:
            raise NotImplementedError(
                "dense structural linearization is available for the built-in fused path only"
            )
        contribution = self._contribution
        if contribution is None:  # pragma: no cover - native/contribution invariant
            raise RuntimeError("native structural contribution was not prepared")
        native_result, jacobian = self._native.linearize(
            *_native_dynamic_arguments(
                self.pattern,
                self.experiment,
                contribution,
                self.support_fwhm,
            )
        )
        accumulation_arrays, reflection_arrays = native_result
        result = _calculation_result(
            self.phase,
            self.pattern,
            _accumulation_from_native(
                accumulation_arrays,
                contribution,
                self.experiment,
                self.jacobian_layout,
            ),
            _reflection_result(self.phase, reflection_arrays),
        )
        _freeze(jacobian)
        return StructuralPatternLinearizationResult(result, names, jacobian)

    def _linearization_tasks(self) -> tuple[PreparedStructuralPattern, ...]:
        if self._native_spectrum is not None:
            return (self,)
        return self._components if self._components else (self,)

    def _combine_linearizations(
        self,
        products: tuple[StructuralPatternLinearizationResult, ...],
    ) -> StructuralPatternLinearizationResult:
        if self._native_spectrum is not None:
            if len(products) != 1:
                raise ValueError("one native spectrum linearization is required")
            return products[0]
        if not self._components:
            if len(products) != 1:
                raise ValueError("one monochromatic linearization is required")
            return products[0]
        if len(products) != len(self._components):
            raise ValueError("component linearization count does not match prepared components")
        names = _structural_parameter_names(self.phase)
        jacobian = np.zeros_like(products[0].jacobian)
        for product, weight in zip(products, self._component_weights, strict=True):
            if product.parameter_names != names:
                raise ValueError("component structural parameter names must match")
            jacobian[:-1] += product.jacobian[:-1]
            jacobian[-1] += weight * product.jacobian[-1]
        return StructuralPatternLinearizationResult(
            self._combine_calculations(tuple(product.result for product in products)),
            names,
            np.ascontiguousarray(jacobian),
        )

    def vjp(self, sample_weights: ArrayLike) -> StructuralPatternVjpResult:
        """Calculate one structural transpose product from pattern-sample weights."""

        if self._native_spectrum is not None:
            weights = _vector(sample_weights, "sample_weights")
            if weights.shape != self.pattern.x.shape:
                raise ValueError("sample_weights must match the pattern sample count")
            contribution = self._components[0]._contribution
            if contribution is None:  # pragma: no cover - preparation invariant
                raise RuntimeError("native spectrum contribution was not prepared")
            native_result, gradient = self._native_spectrum.vjp(
                weights,
                *_native_spectrum_dynamic_arguments(
                    self.pattern,
                    self.experiment,
                    self.support_fwhm,
                ),
            )
            accumulation_arrays, reflection_arrays = native_result
            result = _calculation_result(
                self.phase,
                self.pattern,
                _accumulation_from_native(
                    accumulation_arrays,
                    contribution,
                    self.experiment,
                    self.jacobian_layout,
                    fixed_spectrum=True,
                ),
                _reflection_result(self.phase, reflection_arrays, len(self._components)),
            )
            _freeze(gradient)
            return StructuralPatternVjpResult(
                result,
                _structural_parameter_names(self.phase),
                gradient,
            )
        if self._components:
            return self._combine_vjps(
                tuple(
                    component.vjp(weights) for component, weights in self._vjp_tasks(sample_weights)
                )
            )
        if self._native is None:
            raise NotImplementedError(
                "structural VJP is currently available for the built-in fused path only"
            )
        weights = _vector(sample_weights, "sample_weights")
        if weights.shape != self.pattern.x.shape:
            raise ValueError("sample_weights must match the pattern sample count")
        names = _structural_parameter_names(self.phase)
        contribution = self._contribution
        if contribution is None:  # pragma: no cover - native/contribution invariant
            raise RuntimeError("native structural contribution was not prepared")
        native_result, gradient = self._native.vjp(
            weights,
            *_native_dynamic_arguments(
                self.pattern,
                self.experiment,
                contribution,
                self.support_fwhm,
            ),
        )
        accumulation_arrays, reflection_arrays = native_result
        accumulation = _accumulation_from_native(
            accumulation_arrays,
            contribution,
            self.experiment,
            self.jacobian_layout,
        )
        result = _calculation_result(
            self.phase,
            self.pattern,
            accumulation,
            _reflection_result(self.phase, reflection_arrays),
        )
        _freeze(gradient)
        return StructuralPatternVjpResult(result, names, gradient)

    def _vjp_tasks(
        self,
        sample_weights: ArrayLike,
    ) -> tuple[tuple[PreparedStructuralPattern, NDArray[np.float64]], ...]:
        weights = _vector(sample_weights, "sample_weights")
        if weights.shape != self.pattern.x.shape:
            raise ValueError("sample_weights must match the pattern sample count")
        if self._native_spectrum is not None:
            return ((self, weights),)
        leaves = self._components if self._components else (self,)
        return tuple((component, weights) for component in leaves)

    def _combine_vjps(
        self,
        products: tuple[StructuralPatternVjpResult, ...],
    ) -> StructuralPatternVjpResult:
        if self._native_spectrum is not None:
            if len(products) != 1:
                raise ValueError("one native spectrum VJP is required")
            return products[0]
        if not self._components:
            if len(products) != 1:
                raise ValueError("one monochromatic VJP is required")
            return products[0]
        if len(products) != len(self._components):
            raise ValueError("component VJP count does not match prepared components")
        gradient = np.zeros_like(products[0].gradient)
        for product, component_weight in zip(products, self._component_weights, strict=True):
            gradient[:-1] += product.gradient[:-1]
            gradient[-1] += component_weight * product.gradient[-1]
        return StructuralPatternVjpResult(
            self._combine_calculations(tuple(product.result for product in products)),
            _structural_parameter_names(self.phase),
            np.ascontiguousarray(gradient),
        )


def calculate_structural_pattern(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phase: RietveldPhase,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
    execution: ExecutionPolicy | None = None,
) -> StructuralPatternCalculationResult:
    """Calculate one structural monochromatic CW phase through a scriptable API."""

    return PreparedStructuralPattern(
        pattern,
        experiment,
        phase,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
        execution=execution,
    ).calculate()
