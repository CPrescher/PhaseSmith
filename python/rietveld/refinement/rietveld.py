"""Typed, script-first structural Rietveld refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..intensity_corrections import (
    IntegratedIntensityCorrectionProvider,
    NeutralIntegratedIntensityCorrection,
)
from ..pattern import PowderPattern, StructuralPatternCalculationResult
from ..phase import RietveldPhase, StructuralReflectionBatch
from ..radiation import ConstantWavelengthExperiment, RadiationProbe
from ..scattering import NeutronNuclear, ScatteringFactorProvider, XrayNonResonant
from ..structural_calculation import calculate_structural_pattern
from ..structure import AtomSite, CrystalStructure
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from .core import (
    Bounds,
    Constraint,
    ConstraintTransform,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
)
from .lattice import (
    CwStructuralReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
)

if TYPE_CHECKING:
    from ..extensions import ReflectionPhysicsProvider
    from ..io.cif import CifBackend, CifReadLimits


@dataclass(frozen=True, slots=True)
class RietveldParameterSelection:
    """Structural parameter families selected for one scripted refinement."""

    phase_scale: bool = True
    lattice: bool = True
    coordinates: bool = False
    occupancy: bool = False
    u_iso: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.phase_scale,
                self.lattice,
                self.coordinates,
                self.occupancy,
                self.u_iso,
            )
        ):
            raise TypeError("Rietveld parameter selections must be boolean")


@dataclass(frozen=True, slots=True)
class SiteCoordinateModel:
    """Symmetry-allowed tangent basis for one asymmetric atom site."""

    phase_id: str
    site_id: str
    parameter_names: tuple[str, ...]
    basis: NDArray[np.float64]
    special_position: bool

    def __post_init__(self) -> None:
        if self.basis.dtype != np.float64 or self.basis.shape != (
            3,
            len(self.parameter_names),
        ):
            raise ValueError("site coordinate basis must have shape (3, parameter_count)")
        if not np.isfinite(self.basis).all():
            raise ValueError("site coordinate basis must be finite")
        self.basis.flags.writeable = False


def _site_coordinate_model(
    phase_id: str,
    structure: CrystalStructure,
    site: AtomSite,
    tolerance: float,
) -> SiteCoordinateModel:
    coordinate = np.asarray(site.fractional_xyz, dtype=np.float64)
    stabilizer = []
    for operation in structure.space_group.operations:
        translation = np.asarray([float(value) for value in operation.translation])
        difference = operation.rotation @ coordinate + translation - coordinate
        if np.allclose(difference, np.rint(difference), rtol=0.0, atol=tolerance):
            stabilizer.append(operation.rotation - np.eye(3, dtype=np.int64))
    equations = np.vstack(stabilizer)
    _, singular_values, right = np.linalg.svd(equations.astype(np.float64), full_matrices=True)
    rank = int(np.count_nonzero(singular_values > 1.0e-12))
    basis = np.ascontiguousarray(right[rank:].T)
    for column in range(basis.shape[1]):
        nonzero = np.flatnonzero(np.abs(basis[:, column]) > 1.0e-12)
        if nonzero.size and basis[nonzero[0], column] < 0.0:
            basis[:, column] *= -1.0
    special = basis.shape[1] != 3
    if special:
        names = tuple(f"q{index}" for index in range(basis.shape[1]))
    else:
        basis = np.eye(3, dtype=np.float64)
        names = ("x", "y", "z")
    basis.flags.writeable = False
    return SiteCoordinateModel(phase_id, site.site_id, names, basis, special)


def phase_scale_key(phase_id: str) -> ParameterKey:
    return ParameterKey("phase", phase_id, "scale")


def lattice_parameter_key(phase_id: str, name: str) -> ParameterKey:
    return ParameterKey("lattice", phase_id, name)


def site_parameter_key(phase_id: str, site_id: str, name: str) -> ParameterKey:
    return ParameterKey("site", f"{phase_id}/{site_id}", name)


def build_parameter_set(
    phases: tuple[RietveldPhase, ...],
    lattice_domains: tuple[CwStructuralReflectionDomain | None, ...],
    selection: RietveldParameterSelection,
) -> ParameterSet:
    """Construct deterministic structural parameter records in physical units."""

    if len(phases) != len(lattice_domains):
        raise ValueError("lattice domains must align with Rietveld phases")
    specs = []
    for phase, domain in zip(phases, lattice_domains, strict=True):
        if selection.phase_scale:
            specs.append(
                ParameterSpec(
                    phase_scale_key(phase.phase_id),
                    phase.scale,
                    "relative",
                    Bounds(0.0, np.inf),
                    max(abs(phase.scale), 1.0),
                )
            )
        if selection.lattice:
            if domain is None:
                raise ValueError("lattice refinement requires a guarded structural domain")
            values = domain.parameterization.values_from_cell(phase.structure.cell)
            specs.extend(
                ParameterSpec(
                    lattice_parameter_key(phase.phase_id, name),
                    float(value),
                    "angstrom" if name.endswith("_angstrom") else "degree",
                    Bounds(float(lower), float(upper)),
                    max(abs(float(value)), 1.0),
                )
                for name, value, lower, upper in zip(
                    domain.parameterization.parameter_names,
                    values,
                    domain.bounds.lower,
                    domain.bounds.upper,
                    strict=True,
                )
            )
        for site in phase.structure.sites:
            if selection.coordinates:
                model = _site_coordinate_model(
                    phase.phase_id,
                    phase.structure,
                    site,
                    phase.coordinate_tolerance,
                )
                initial = (
                    site.fractional_xyz
                    if not model.special_position
                    else (0.0,) * len(model.parameter_names)
                )
                bounds = (
                    Bounds(-np.inf, np.inf) if not model.special_position else Bounds(-0.5, 0.5)
                )
                specs.extend(
                    ParameterSpec(
                        site_parameter_key(phase.phase_id, site.site_id, name),
                        float(value),
                        "fractional",
                        bounds,
                        1.0,
                    )
                    for name, value in zip(model.parameter_names, initial, strict=True)
                )
            if selection.occupancy:
                specs.append(
                    ParameterSpec(
                        site_parameter_key(phase.phase_id, site.site_id, "occupancy"),
                        site.occupancy,
                        "fraction",
                        Bounds(0.0, max(1.0, 2.0 * site.occupancy + 0.1)),
                        max(site.occupancy, 1.0),
                    )
                )
            if selection.u_iso:
                value = 0.0 if site.u_iso_angstrom2 is None else site.u_iso_angstrom2
                specs.append(
                    ParameterSpec(
                        site_parameter_key(phase.phase_id, site.site_id, "u_iso_angstrom2"),
                        value,
                        "angstrom^2",
                        Bounds(0.0, max(0.5, 2.0 * value + 0.05)),
                        max(value, 0.01),
                    )
                )
    return ParameterSet(specs)


@dataclass(frozen=True, slots=True)
class RietveldCalculationResult:
    """Combined structural pattern with one diagnostic calculation per phase."""

    y: NDArray[np.float64]
    profile_y: NDArray[np.float64]
    background: NDArray[np.float64]
    phase_calculations: tuple[StructuralPatternCalculationResult, ...]

    def __post_init__(self) -> None:
        count = self.y.size
        for name, array in (
            ("y", self.y),
            ("profile_y", self.profile_y),
            ("background", self.background),
        ):
            if array.dtype != np.float64 or array.shape != (count,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite float64 sample vector")
            array.flags.writeable = False
        if not self.phase_calculations:
            raise ValueError("at least one structural phase calculation is required")
        if any(item.y.shape != (count,) for item in self.phase_calculations):
            raise ValueError("phase calculations must share the combined sample grid")


def calculate(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[RietveldPhase, ...],
    *,
    support_fwhm: float = 20.0,
) -> RietveldCalculationResult:
    """Calculate and sum one or more structural phases without duplicating background."""

    selected = tuple(phases)
    if not selected:
        raise ValueError("at least one Rietveld phase is required")
    calculations = tuple(
        calculate_structural_pattern(pattern, experiment, phase, support_fwhm=support_fwhm)
        for phase in selected
    )
    profile = np.ascontiguousarray(
        sum((item.profile_y for item in calculations), np.zeros_like(pattern.x))
    )
    y = np.ascontiguousarray(profile + pattern.background)
    return RietveldCalculationResult(y, profile, pattern.background, calculations)


@dataclass(frozen=True, slots=True)
class RietveldInput:
    """Observed pattern, experiment, structural phases, and refinement contract."""

    pattern: PowderPattern
    experiment: ConstantWavelengthExperiment
    phases: tuple[RietveldPhase, ...]
    lattice_domains: tuple[CwStructuralReflectionDomain | None, ...]
    parameters: ParameterSet
    constraints: tuple[Constraint, ...] = ()
    selection: RietveldParameterSelection = RietveldParameterSelection()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "lattice_domains", tuple(self.lattice_domains))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not isinstance(self.pattern, PowderPattern) or self.pattern.observed_y is None:
            raise ValueError("Rietveld input requires an observed PowderPattern")
        if not isinstance(self.experiment, ConstantWavelengthExperiment):
            raise TypeError("experiment must be ConstantWavelengthExperiment")
        if not self.phases or any(not isinstance(phase, RietveldPhase) for phase in self.phases):
            raise TypeError("phases must be a non-empty tuple of RietveldPhase objects")
        if len(self.phases) != len(self.lattice_domains):
            raise ValueError("one optional lattice domain is required per phase")
        if len({phase.phase_id for phase in self.phases}) != len(self.phases):
            raise ValueError("Rietveld phase IDs must be unique")
        for phase, domain in zip(self.phases, self.lattice_domains, strict=True):
            if domain is not None:
                if domain.space_group != phase.structure.space_group:
                    raise ValueError("lattice domain and phase must use the same space group")
                if domain.wavelength_angstrom != self.experiment.radiation.wavelength_angstrom:
                    raise ValueError("lattice-domain and experiment wavelengths must match")
        if not isinstance(self.parameters, ParameterSet):
            raise TypeError("parameters must be a ParameterSet")
        ConstraintTransform(self.parameters, self.constraints)

    @classmethod
    def from_cif(
        cls,
        pattern: PowderPattern,
        experiment: ConstantWavelengthExperiment,
        path_or_text: str | Path,
        *,
        phase_id: str,
        selection: RietveldParameterSelection | None = None,
        block: str | None = None,
        strict: bool = True,
        name: str | None = None,
        scale: float = 1.0,
        scattering: ScatteringFactorProvider | None = None,
        intensity_correction: IntegratedIntensityCorrectionProvider | None = None,
        physics: ReflectionPhysicsProvider | None = None,
        lattice_relative_bound: float = 0.05,
        lattice_angle_bound_deg: float = 5.0,
        merge_friedel: bool = True,
        max_candidates: int = 50_000_000,
        coordinate_tolerance: float = 1.0e-10,
        limits: CifReadLimits | None = None,
        backend: CifBackend | None = None,
    ) -> RietveldInput:
        """Construct a runnable single-phase structural request from a CIF."""

        from ..io.cif import read_cif

        if not isinstance(pattern, PowderPattern) or pattern.observed_y is None:
            raise ValueError("CIF-backed Rietveld input requires an observed pattern")
        if not isinstance(experiment, ConstantWavelengthExperiment):
            raise TypeError("experiment must be ConstantWavelengthExperiment")
        selected_parameters = RietveldParameterSelection() if selection is None else selection
        if not isinstance(selected_parameters, RietveldParameterSelection):
            raise TypeError("selection must be RietveldParameterSelection")
        imported = read_cif(
            path_or_text, block=block, strict=strict, limits=limits, backend=backend
        )
        structure = imported.structure
        if not structure.sites:
            raise ValueError("Rietveld calculation requires at least one atom site")
        if selected_parameters.lattice:
            parameterization = LatticeParameterization(structure.space_group, structure.cell)
            lattice_bounds = LatticeParameterBounds.around(
                parameterization,
                relative_length=lattice_relative_bound,
                angle_delta_deg=lattice_angle_bound_deg,
            )
            domain = CwStructuralReflectionDomain(
                structure.space_group,
                parameterization,
                lattice_bounds,
                experiment.radiation.wavelength_angstrom,
                float(pattern.x[0]),
                float(pattern.x[-1]),
                merge_friedel,
                max_candidates,
            )
            reflections = domain.generate(structure.cell).reflections
            selected_domain: CwStructuralReflectionDomain | None = domain
        else:
            generated = PreparedReflectionGenerator(
                structure.space_group,
                merge_friedel=merge_friedel,
                max_candidates=max_candidates,
            ).generate(
                structure.cell,
                CwTwoThetaRange(
                    float(pattern.x[0]),
                    float(pattern.x[-1]),
                    experiment.radiation.wavelength_angstrom,
                ),
            )
            reflections = StructuralReflectionBatch.from_generated(generated)
            selected_domain = None
        selected_scattering = scattering
        if selected_scattering is None:
            selected_scattering = (
                XrayNonResonant()
                if experiment.radiation.probe is RadiationProbe.X_RAY
                else NeutronNuclear()
            )
        selected_correction = (
            NeutralIntegratedIntensityCorrection()
            if intensity_correction is None
            else intensity_correction
        )
        phase = RietveldPhase(
            phase_id,
            structure.name if name is None else name,
            structure,
            reflections,
            selected_scattering,
            selected_correction,
            scale,
            physics,
            coordinate_tolerance,
        )
        domains = (selected_domain,)
        parameters = build_parameter_set((phase,), domains, selected_parameters)
        return cls(pattern, experiment, (phase,), domains, parameters, (), selected_parameters)
