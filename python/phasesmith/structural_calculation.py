"""Scriptable structural-pattern calculation for one Rietveld phase."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._api import _vector
from .crystallography import calculate_structure_factor_values, p1_parameter_names
from .cw import CW_GLOBAL_PARAMETER_ORDER, CW_LOCAL_PARAMETER_ORDER, accumulate_cw_contributions
from .extensions import (
    CompositePhysicsProvider,
    PhysicsContext,
    PhysicsContribution,
    evaluate_provider,
)
from .intensity_corrections import (
    BraggBrentanoPolarizedLp,
    BraggBrentanoUnpolarizedLp,
    NeutralIntegratedIntensityCorrection,
)
from .pattern import (
    PowderPattern,
    StructuralPatternCalculationResult,
    StructuralPatternJvpResult,
    StructuralPatternVjpResult,
    StructuralReflectionResult,
)
from .phase import ReflectionGeometryBatch, RietveldPhase
from .radiation import (
    ComponentRadiation,
    ConstantWavelengthExperiment,
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
    two_theta = np.ascontiguousarray(2.0 * np.degrees(np.arcsin(sin_theta)))
    two_theta += experiment.zero_shift_deg
    if experiment.geometry is not None:
        theta = np.radians(0.5 * (two_theta - experiment.zero_shift_deg))
        two_theta -= np.degrees(
            2.0
            * experiment.geometry.sample_displacement_mm
            / experiment.geometry.goniometer_radius_mm
            * np.cos(theta)
        )
    return ReflectionGeometryBatch(
        phase.reflections.hkl,
        spacing,
        two_theta,
        base_intensity,
    )


def _structural_parameter_names(phase: RietveldPhase) -> tuple[str, ...]:
    return p1_parameter_names(phase.structure.to_isotropic_site_batch())


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


def _native_phase(phase: RietveldPhase) -> object | None:
    configuration = _native_model_configuration(phase)
    if configuration is None or not _supports_fused_structural_physics(phase.physics):
        return None
    scattering_model, correction_model, correction_wavelength, correction_polarization = (
        configuration
    )
    sites = phase.structure.to_isotropic_site_batch()
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
        species_keys,
        np.ascontiguousarray(offsets.real),
        np.ascontiguousarray(offsets.imag),
        *phase.structure.cell.as_tuple(),
        float(phase.scale),
        float(phase.coordinate_tolerance),
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
    return (
        pattern.x,
        instrument.wavelength_angstrom,
        experiment.zero_shift_deg,
        None if geometry is None else geometry.sample_displacement_mm,
        None if geometry is None else geometry.goniometer_radius_mm,
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


def _accumulation_from_native(
    arrays: tuple[object, ...],
    contribution: PhysicsContribution,
    experiment: ConstantWavelengthExperiment,
    jacobian_layout: Literal["support", "dense"],
) -> AccumulationResult:
    position_names = ("wavelength_angstrom", "zero_shift_deg")
    if experiment.geometry is not None:
        position_names += ("sample_displacement_mm",)
    return _build_accumulation_result(
        *arrays,
        CW_LOCAL_PARAMETER_ORDER,
        CW_GLOBAL_PARAMETER_ORDER + position_names + contribution.parameter_names,
        jacobian_layout,
    )


def _reflection_result(
    phase: RietveldPhase,
    arrays: tuple[object, ...],
) -> StructuralReflectionResult:
    f_real, f_imag, f_squared, intensity, q_squared, s, d_spacing, two_theta = (
        np.asarray(value) for value in arrays
    )
    f = np.asarray(f_real + 1j * f_imag, dtype=np.complex128)
    for array in (f, f_squared, intensity, q_squared, s, d_spacing, two_theta):
        _freeze(array)
    return StructuralReflectionResult(
        phase.reflections.reflection_ids,
        f,
        f_squared,
        intensity,
        q_squared,
        s,
        d_spacing,
        two_theta,
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
        )
        correction = phase.intensity_correction
        if isinstance(correction, BraggBrentanoUnpolarizedLp):
            correction = BraggBrentanoUnpolarizedLp(float(wavelength))
        elif isinstance(correction, BraggBrentanoPolarizedLp):
            correction = BraggBrentanoPolarizedLp(
                float(wavelength),
                correction.polarization,
            )
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
    _native: object | None
    _contribution: PhysicsContribution | None
    _components: tuple[PreparedStructuralPattern, ...]
    _component_weights: tuple[float, ...]

    def __init__(
        self,
        pattern: PowderPattern,
        experiment: ConstantWavelengthExperiment,
        phase: RietveldPhase,
        *,
        support_fwhm: float = 20.0,
        jacobian_layout: Literal["support", "dense"] = "support",
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
        _check_probe(phase, experiment)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "experiment", experiment)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "support_fwhm", float(support_fwhm))
        object.__setattr__(self, "jacobian_layout", jacobian_layout)
        component_inputs = _component_inputs(experiment, phase)
        if component_inputs:
            object.__setattr__(self, "_native", None)
            object.__setattr__(self, "_contribution", None)
            object.__setattr__(
                self,
                "_components",
                tuple(
                    PreparedStructuralPattern(
                        pattern,
                        component_experiment,
                        component_phase,
                        support_fwhm=support_fwhm,
                        jacobian_layout=jacobian_layout,
                    )
                    for component_experiment, component_phase, _weight in component_inputs
                ),
            )
            object.__setattr__(
                self,
                "_component_weights",
                tuple(weight for _experiment, _phase, weight in component_inputs),
            )
            return
        native = _native_phase(phase)
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

    @property
    def uses_native_fused_path(self) -> bool:
        """Report whether values and structural derivatives stay in one native call."""

        if self._components:
            return all(component.uses_native_fused_path for component in self._components)
        return self._native is not None

    def calculate(self) -> StructuralPatternCalculationResult:
        """Calculate structural intensities, positions, and the powder profile."""

        if self._components:
            return _combine_component_results(
                self.phase,
                self.pattern,
                tuple(component.calculate() for component in self._components),
                self.jacobian_layout,
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

        if self._components:
            names = _structural_parameter_names(self.phase)
            direction = _vector(tangent, "tangent")
            if direction.shape != (len(names),):
                raise ValueError("tangent must match the structural parameter count")
            products = []
            for component, weight in zip(self._components, self._component_weights, strict=True):
                component_direction = np.array(direction, copy=True)
                component_direction[-1] *= weight
                products.append(component.jvp(component_direction))
            product_tuple = tuple(products)
            result = _combine_component_results(
                self.phase,
                self.pattern,
                tuple(product.result for product in product_tuple),
                self.jacobian_layout,
            )
            return StructuralPatternJvpResult(
                result,
                names,
                np.ascontiguousarray(
                    sum((item.d_y for item in product_tuple), np.zeros_like(self.pattern.x))
                ),
                np.ascontiguousarray(
                    np.concatenate([item.d_integrated_intensity for item in product_tuple])
                ),
                np.ascontiguousarray(
                    np.concatenate([item.d_two_theta_deg for item in product_tuple])
                ),
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

    def vjp(self, sample_weights: ArrayLike) -> StructuralPatternVjpResult:
        """Calculate one structural transpose product from pattern-sample weights."""

        if self._components:
            weights = _vector(sample_weights, "sample_weights")
            if weights.shape != self.pattern.x.shape:
                raise ValueError("sample_weights must match the pattern sample count")
            products = tuple(component.vjp(weights) for component in self._components)
            gradient = np.zeros_like(products[0].gradient)
            for product, component_weight in zip(products, self._component_weights, strict=True):
                gradient[:-1] += product.gradient[:-1]
                gradient[-1] += component_weight * product.gradient[-1]
            return StructuralPatternVjpResult(
                _combine_component_results(
                    self.phase,
                    self.pattern,
                    tuple(product.result for product in products),
                    self.jacobian_layout,
                ),
                _structural_parameter_names(self.phase),
                np.ascontiguousarray(gradient),
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


def calculate_structural_pattern(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phase: RietveldPhase,
    *,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> StructuralPatternCalculationResult:
    """Calculate one structural monochromatic CW phase through a scriptable API."""

    return PreparedStructuralPattern(
        pattern,
        experiment,
        phase,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    ).calculate()
