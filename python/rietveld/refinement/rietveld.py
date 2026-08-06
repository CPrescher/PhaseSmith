"""Typed, script-first structural Rietveld refinement orchestration."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..control import CancellationCallback
from ..crystallography import p1_parameter_names
from ..intensity_corrections import (
    IntegratedIntensityCorrectionProvider,
    NeutralIntegratedIntensityCorrection,
)
from ..pattern import PowderPattern, StructuralPatternCalculationResult
from ..phase import RietveldPhase, StructuralReflectionBatch
from ..radiation import ConstantWavelengthExperiment, RadiationProbe
from ..scattering import NeutronNuclear, ScatteringFactorProvider, XrayNonResonant
from ..structural_calculation import PreparedStructuralPattern, calculate_structural_pattern
from ..structure import AtomSite, CrystalStructure
from ..symmetry import CwTwoThetaRange, PreparedReflectionGenerator
from .core import (
    Bounds,
    Constraint,
    ConstraintTransform,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    ResidualOptions,
    TerminationReason,
    evaluate_residuals,
)
from .lattice import (
    CwStructuralReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
)
from .runtime import (
    CheckpointCallback,
    RefinementEventKind,
    RefinementLimits,
    RefinementLogger,
    RefinementRuntime,
    RefinementStopped,
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


def _domain_parameter_values(
    phases: tuple[RietveldPhase, ...],
    domains: tuple[CwStructuralReflectionDomain | None, ...],
    parameters: ParameterSet,
) -> dict[ParameterKey, float]:
    """Resolve typed parameter keys against the supplied structural state."""

    phase_by_id = {phase.phase_id: phase for phase in phases}
    domain_by_id = {phase.phase_id: domain for phase, domain in zip(phases, domains, strict=True)}
    site_by_owner = {
        f"{phase.phase_id}/{site.site_id}": (phase, site)
        for phase in phases
        for site in phase.structure.sites
    }
    values = {}
    for spec in parameters.specs:
        key = spec.key
        if key.module == "phase" and key.owner_id in phase_by_id and key.name == "scale":
            value = phase_by_id[key.owner_id].scale
        elif key.module == "lattice" and key.owner_id in phase_by_id:
            domain = domain_by_id[key.owner_id]
            if domain is None:
                raise ValueError(f"{key.label} requires a guarded lattice domain")
            try:
                index = domain.parameterization.parameter_names.index(key.name)
            except ValueError as error:
                raise ValueError(f"unknown lattice parameter {key.label}") from error
            value = float(
                domain.parameterization.values_from_cell(phase_by_id[key.owner_id].structure.cell)[
                    index
                ]
            )
        elif key.module == "site" and key.owner_id in site_by_owner:
            phase, site = site_by_owner[key.owner_id]
            model = _site_coordinate_model(
                phase.phase_id,
                phase.structure,
                site,
                phase.coordinate_tolerance,
            )
            if key.name in ("x", "y", "z") and not model.special_position:
                value = site.fractional_xyz[("x", "y", "z").index(key.name)]
            elif key.name in model.parameter_names and model.special_position:
                # Special-position q values are local accepted-state coordinates;
                # the phase itself is their physical anchor.
                value = spec.value
            elif key.name == "occupancy":
                value = site.occupancy
            elif key.name == "u_iso_angstrom2":
                value = 0.0 if site.u_iso_angstrom2 is None else site.u_iso_angstrom2
            else:
                raise ValueError(f"unknown site parameter {key.label}")
        else:
            raise ValueError(f"unsupported or unknown Rietveld parameter {key.label}")
        if not spec.bounds.contains(value):
            raise ValueError(f"domain value for {key.label} lies outside its bounds")
        values[key] = float(value)
    return values


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
        domain_values = _domain_parameter_values(self.phases, self.lattice_domains, self.parameters)
        for key, value in domain_values.items():
            if not np.isclose(
                value,
                self.parameters.spec(key).value,
                rtol=0.0,
                atol=2.0e-12,
            ):
                raise ValueError(f"parameter value for {key.label} does not match its domain")
        transform = ConstraintTransform(self.parameters, self.constraints)
        constrained_values = transform.unpack(transform.pack())
        for key, value in domain_values.items():
            if not np.isclose(value, constrained_values[key], rtol=0.0, atol=2.0e-12):
                raise ValueError(f"constraint for {key.label} is not satisfied initially")

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


@dataclass(frozen=True, slots=True)
class RietveldOptions:
    """Numerical controls for bounded matrix-free structural refinement."""

    limits: RefinementLimits = field(default_factory=lambda: RefinementLimits(max_iterations=50))
    min_iterations: int = 1
    objective_tolerance: float = 1.0e-10
    parameter_tolerance: float = 1.0e-7
    initial_damping: float = 1.0e-6
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    cg_tolerance: float = 1.0e-6
    max_cg_iterations: int = 30
    max_scaled_parameter_step: float = 0.25
    max_backtracks: int = 8
    use_uncertainty: bool = True
    support_fwhm: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.limits, RefinementLimits):
            raise TypeError("limits must be RefinementLimits")
        if not isinstance(self.min_iterations, int) or self.min_iterations <= 0:
            raise ValueError("min_iterations must be a positive integer")
        if self.min_iterations > self.limits.max_iterations:
            raise ValueError("min_iterations must not exceed the iteration limit")
        positive = (
            self.objective_tolerance,
            self.parameter_tolerance,
            self.initial_damping,
            self.damping_increase,
            self.damping_decrease,
            self.cg_tolerance,
            self.max_scaled_parameter_step,
            self.support_fwhm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Rietveld tolerances, damping, step, and support must be positive")
        if self.damping_increase <= 1.0 or self.damping_decrease >= 1.0:
            raise ValueError("damping must increase above one and decrease below one")
        if not isinstance(self.max_cg_iterations, int) or self.max_cg_iterations <= 0:
            raise ValueError("max_cg_iterations must be a positive integer")
        if not isinstance(self.max_backtracks, int) or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a non-negative integer")
        if not isinstance(self.use_uncertainty, bool):
            raise TypeError("use_uncertainty must be boolean")


@dataclass(frozen=True, slots=True)
class RietveldParameterChange:
    """One accepted physical parameter update."""

    key: ParameterKey
    before: float
    after: float
    scaled_change: float


@dataclass(frozen=True, slots=True)
class RietveldIterationRecord:
    """Diagnostics for one accepted matrix-free iteration."""

    iteration: int
    rp: float
    rwp: float
    chi_square: float
    reduced_chi_square: float
    objective: float
    objective_change: float
    scaled_step_norm: float
    damping: float
    cg_iterations: int
    backtracks: int
    parameter_changes: tuple[RietveldParameterChange, ...]
    topology_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_changes", tuple(self.parameter_changes))
        object.__setattr__(self, "topology_changes", tuple(self.topology_changes))
        if self.iteration <= 0 or self.cg_iterations < 0 or self.backtracks < 0:
            raise ValueError("iteration diagnostics contain invalid counters")


@dataclass(frozen=True, slots=True)
class RietveldCheckpoint:
    """Complete last-accepted state for deterministic continuation."""

    completed_iterations: int
    phases: tuple[RietveldPhase, ...]
    parameters: ParameterSet
    objective: float
    damping: float
    history: tuple[RietveldIterationRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "history", tuple(self.history))
        if self.completed_iterations != len(self.history) or self.completed_iterations < 0:
            raise ValueError("checkpoint iteration count must match its history")
        if not self.phases or any(not isinstance(phase, RietveldPhase) for phase in self.phases):
            raise TypeError("checkpoint phases must contain RietveldPhase values")
        if not isinstance(self.parameters, ParameterSet):
            raise TypeError("checkpoint parameters must be ParameterSet")
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise ValueError("checkpoint objective must be non-negative and finite")
        if not np.isfinite(self.damping) or self.damping <= 0.0:
            raise ValueError("checkpoint damping must be positive and finite")


@dataclass(frozen=True, slots=True)
class RietveldResult:
    """Last accepted structural state and auditable termination diagnostics."""

    calculation: RietveldCalculationResult
    phases: tuple[RietveldPhase, ...]
    parameters: ParameterSet
    metrics: ResidualEvaluation
    history: tuple[RietveldIterationRecord, ...]
    termination_reason: TerminationReason
    termination_message: str
    checkpoint: RietveldCheckpoint
    evaluations: int
    logger_error: Exception | None = None


def _weight_vector(pattern: PowderPattern, use_uncertainty: bool) -> NDArray[np.float64]:
    included = np.ones(pattern.x.size, dtype=np.bool_) if pattern.mask is None else pattern.mask
    weight = included.astype(np.float64)
    if use_uncertainty and pattern.uncertainty is not None:
        weight /= pattern.uncertainty
    return np.ascontiguousarray(weight)


def _phase_native_mapping(
    phase: RietveldPhase,
    domain: CwStructuralReflectionDomain | None,
    parameters: ParameterSet,
) -> NDArray[np.float64]:
    """Map physical parameter changes to the native structural tangent."""

    names = p1_parameter_names(phase.structure.to_isotropic_site_batch())
    native_row = {name: row for row, name in enumerate(names)}
    mapping = np.zeros((len(names), len(parameters.specs)), dtype=np.float64)
    site_by_id = {site.site_id: site for site in phase.structure.sites}
    coordinate_models = {
        site.site_id: _site_coordinate_model(
            phase.phase_id,
            phase.structure,
            site,
            phase.coordinate_tolerance,
        )
        for site in phase.structure.sites
    }
    lattice_jacobian = None
    if domain is not None:
        lattice_values = domain.parameterization.values_from_cell(phase.structure.cell)
        lattice_jacobian = domain.parameterization.cell_parameter_jacobian(lattice_values)
    for column, spec in enumerate(parameters.specs):
        key = spec.key
        if key.module == "phase" and key.owner_id == phase.phase_id and key.name == "scale":
            mapping[native_row["phase.scale"], column] = 1.0
        elif key.module == "lattice" and key.owner_id == phase.phase_id:
            if domain is None or lattice_jacobian is None:
                raise ValueError(f"{key.label} requires a guarded lattice domain")
            try:
                lattice_column = domain.parameterization.parameter_names.index(key.name)
            except ValueError as error:
                raise ValueError(f"unknown lattice parameter {key.label}") from error
            mapping[:6, column] = lattice_jacobian[:, lattice_column]
        elif key.module == "site":
            owner_phase, separator, site_id = key.owner_id.partition("/")
            if not separator or owner_phase != phase.phase_id:
                continue
            if site_id not in site_by_id:
                raise ValueError(f"unknown site parameter owner {key.owner_id!r}")
            model = coordinate_models[site_id]
            if key.name in model.parameter_names:
                coordinate_column = model.parameter_names.index(key.name)
                for component, component_name in enumerate(("x", "y", "z")):
                    mapping[native_row[f"site.{site_id}.{component_name}"], column] = model.basis[
                        component, coordinate_column
                    ]
            elif key.name == "occupancy":
                mapping[native_row[f"site.{site_id}.occupancy"], column] = 1.0
            elif key.name == "u_iso_angstrom2":
                mapping[native_row[f"site.{site_id}.u_iso"], column] = 1.0
            else:
                raise ValueError(f"unknown site parameter {key.label}")
    mapping.flags.writeable = False
    return mapping


@dataclass(slots=True)
class _RietveldLinearization:
    pattern: PowderPattern
    experiment: ConstantWavelengthExperiment
    phases: tuple[RietveldPhase, ...]
    physical_to_free: NDArray[np.float64]
    native_mappings: tuple[NDArray[np.float64], ...]
    prepared: tuple[PreparedStructuralPattern, ...]
    sample_weight: NDArray[np.float64]
    runtime: RefinementRuntime

    @classmethod
    def prepare(
        cls,
        input_data: RietveldInput,
        phases: tuple[RietveldPhase, ...],
        parameters: ParameterSet,
        options: RietveldOptions,
        runtime: RefinementRuntime,
    ) -> _RietveldLinearization:
        transform = ConstraintTransform(parameters, input_data.constraints)
        return cls(
            input_data.pattern,
            input_data.experiment,
            phases,
            transform.derivative_matrix(),
            tuple(
                _phase_native_mapping(phase, domain, parameters)
                for phase, domain in zip(phases, input_data.lattice_domains, strict=True)
            ),
            tuple(
                PreparedStructuralPattern(
                    input_data.pattern,
                    input_data.experiment,
                    phase,
                    support_fwhm=options.support_fwhm,
                )
                for phase in phases
            ),
            _weight_vector(input_data.pattern, options.use_uncertainty),
            runtime,
        )

    def calculate(self) -> RietveldCalculationResult:
        self.runtime.begin_evaluation()
        calculations = tuple(item.calculate() for item in self.prepared)
        profile = np.ascontiguousarray(
            sum((item.profile_y for item in calculations), np.zeros_like(self.pattern.x))
        )
        return RietveldCalculationResult(
            np.ascontiguousarray(profile + self.pattern.background),
            profile,
            self.pattern.background,
            calculations,
        )

    def jvp(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        self.runtime.begin_evaluation()
        if direction.shape != (self.physical_to_free.shape[1],):
            raise ValueError("free tangent has the wrong shape")
        physical = self.physical_to_free @ direction
        result = np.zeros_like(self.pattern.x)
        for prepared, mapping in zip(self.prepared, self.native_mappings, strict=True):
            result += prepared.jvp(mapping @ physical).d_y
        return np.ascontiguousarray(result * self.sample_weight)

    def vjp(self, weighted_samples: NDArray[np.float64]) -> NDArray[np.float64]:
        self.runtime.begin_evaluation()
        if weighted_samples.shape != self.pattern.x.shape:
            raise ValueError("weighted sample vector has the wrong shape")
        physical_gradient = np.zeros(self.physical_to_free.shape[0], dtype=np.float64)
        raw_samples = weighted_samples * self.sample_weight
        for prepared, mapping in zip(self.prepared, self.native_mappings, strict=True):
            physical_gradient += mapping.T @ prepared.vjp(raw_samples).gradient
        return np.ascontiguousarray(self.physical_to_free.T @ physical_gradient)


def _apply_parameter_values(
    phases: tuple[RietveldPhase, ...],
    domains: tuple[CwStructuralReflectionDomain | None, ...],
    current_parameters: ParameterSet,
    values: dict[ParameterKey, float],
) -> tuple[tuple[RietveldPhase, ...], tuple[str, ...]]:
    """Apply physical values and regenerate guarded topology at a trial state."""

    current_values = current_parameters.values()
    updated_phases = []
    topology_changes = []
    for phase, domain in zip(phases, domains, strict=True):
        structure = phase.structure
        if domain is not None:
            lattice_values = [
                values.get(
                    lattice_parameter_key(phase.phase_id, name),
                    float(value),
                )
                for name, value in zip(
                    domain.parameterization.parameter_names,
                    domain.parameterization.values_from_cell(structure.cell),
                    strict=True,
                )
            ]
            structure = replace(structure, cell=domain.parameterization.to_cell(lattice_values))
        sites = []
        for site in structure.sites:
            model = _site_coordinate_model(
                phase.phase_id,
                structure,
                site,
                phase.coordinate_tolerance,
            )
            coordinate = np.asarray(site.fractional_xyz, dtype=np.float64)
            if model.special_position:
                for index, name in enumerate(model.parameter_names):
                    key = site_parameter_key(phase.phase_id, site.site_id, name)
                    if key in values:
                        coordinate += model.basis[:, index] * (values[key] - current_values[key])
            else:
                coordinate = np.asarray(
                    [
                        values.get(
                            site_parameter_key(phase.phase_id, site.site_id, name),
                            float(coordinate[index]),
                        )
                        for index, name in enumerate(("x", "y", "z"))
                    ],
                    dtype=np.float64,
                )
            occupancy = values.get(
                site_parameter_key(phase.phase_id, site.site_id, "occupancy"),
                site.occupancy,
            )
            u_iso = values.get(
                site_parameter_key(phase.phase_id, site.site_id, "u_iso_angstrom2"),
                site.u_iso_angstrom2,
            )
            sites.append(
                replace(
                    site,
                    fractional_xyz=tuple(map(float, coordinate)),
                    occupancy=float(occupancy),
                    u_iso_angstrom2=None if u_iso is None else float(u_iso),
                )
            )
        structure = replace(structure, sites=tuple(sites))
        scale = values.get(phase_scale_key(phase.phase_id), phase.scale)
        reflections = phase.reflections
        if domain is not None:
            generated = domain.generate(structure.cell, phase.reflections)
            reflections = generated.reflections
            if generated.added_reflection_ids or generated.removed_reflection_ids:
                topology_changes.append(
                    f"phase {phase.phase_id}: +{len(generated.added_reflection_ids)} "
                    f"-{len(generated.removed_reflection_ids)} guarded families"
                )
        updated_phases.append(
            replace(
                phase,
                structure=structure,
                reflections=reflections,
                scale=float(scale),
            )
        )
    return tuple(updated_phases), tuple(topology_changes)


def _conjugate_gradient(
    operator: object,
    right_hand_side: NDArray[np.float64],
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int]:
    """Solve one positive-definite matrix-free system deterministically."""

    apply = operator
    if not callable(apply):
        raise TypeError("operator must be callable")
    solution = np.zeros_like(right_hand_side)
    residual = np.array(right_hand_side, copy=True)
    direction = np.array(residual, copy=True)
    squared = float(residual @ residual)
    target = tolerance * max(float(np.linalg.norm(right_hand_side)), 1.0)
    if np.sqrt(squared) <= target:
        return solution, 0
    for iteration in range(1, max_iterations + 1):
        product = apply(direction)
        denominator = float(direction @ product)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("matrix-free normal operator is not positive definite")
        alpha = squared / denominator
        solution += alpha * direction
        residual -= alpha * product
        next_squared = float(residual @ residual)
        if not np.isfinite(next_squared):
            raise FloatingPointError("matrix-free solve produced a non-finite residual")
        if np.sqrt(next_squared) <= target:
            return solution, iteration
        direction = residual + (next_squared / squared) * direction
        squared = next_squared
    return solution, max_iterations


def _normal_product(
    linearization: _RietveldLinearization,
    damping: float,
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    return linearization.vjp(linearization.jvp(direction)) + damping * direction


def _metrics(
    input_data: RietveldInput,
    calculation: RietveldCalculationResult,
    options: RietveldOptions,
    parameter_count: int,
) -> ResidualEvaluation:
    return evaluate_residuals(
        input_data.pattern,
        calculation.y,
        ResidualOptions(options.use_uncertainty, parameter_count),
    )


def _checkpoint(
    phases: tuple[RietveldPhase, ...],
    parameters: ParameterSet,
    objective: float,
    damping: float,
    history: list[RietveldIterationRecord],
) -> RietveldCheckpoint:
    return RietveldCheckpoint(len(history), phases, parameters, objective, damping, tuple(history))


def refine(
    input_data: RietveldInput,
    options: RietveldOptions | None = None,
    *,
    checkpoint: RietveldCheckpoint | None = None,
    cancellation: CancellationCallback | None = None,
    logger: RefinementLogger | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> RietveldResult:
    """Refine structural parameters with Rust JVP/VJP products and safe states."""

    if not isinstance(input_data, RietveldInput):
        raise TypeError("input_data must be RietveldInput")
    selected = RietveldOptions() if options is None else options
    if not isinstance(selected, RietveldOptions):
        raise TypeError("options must be RietveldOptions")
    runtime = RefinementRuntime(
        selected.limits,
        cancellation=cancellation,
        logger=logger,
        checkpoint=checkpoint_callback,
    )
    if checkpoint is None:
        phases = input_data.phases
        parameters = input_data.parameters
        history: list[RietveldIterationRecord] = []
        damping = selected.initial_damping
    else:
        if not isinstance(checkpoint, RietveldCheckpoint):
            raise TypeError("checkpoint must be RietveldCheckpoint")
        if tuple(phase.phase_id for phase in checkpoint.phases) != tuple(
            phase.phase_id for phase in input_data.phases
        ):
            raise ValueError("checkpoint phase identities do not match Rietveld input")
        if checkpoint.parameters.keys != input_data.parameters.keys:
            raise ValueError("checkpoint parameter identities do not match Rietveld input")
        phases = checkpoint.phases
        parameters = checkpoint.parameters
        history = list(checkpoint.history)
        damping = checkpoint.damping
        runtime.attempted_iteration = checkpoint.completed_iterations
        runtime.accepted_iterations = checkpoint.completed_iterations
    transform = ConstraintTransform(parameters, input_data.constraints)
    runtime.emit(
        RefinementEventKind.START,
        "rietveld",
        "structural refinement started",
        (("free_parameters", len(transform.free_keys)), ("phases", len(phases))),
    )
    linearization = _RietveldLinearization.prepare(
        input_data, phases, parameters, selected, runtime
    )
    termination = TerminationReason.MAX_ITERATIONS
    termination_message = "iteration limit reached"
    calculation: RietveldCalculationResult | None = None
    try:
        calculation = linearization.calculate()
        metrics = _metrics(input_data, calculation, selected, len(transform.free_keys))
        objective = 0.5 * metrics.chi_square
        if not np.any(linearization.sample_weight):
            termination = TerminationReason.NO_OBSERVATIONS
            termination_message = "no included observations"
        elif not transform.free_keys:
            termination = TerminationReason.CONVERGED
            termination_message = "no free structural parameters"
        else:
            first_iteration = len(history) + 1
            for iteration in range(first_iteration, selected.limits.max_iterations + 1):
                runtime.begin_iteration(iteration)
                weighted_residual = metrics.residual * linearization.sample_weight
                gradient = linearization.vjp(weighted_residual)

                step, cg_iterations = _conjugate_gradient(
                    partial(_normal_product, linearization, damping),
                    -gradient,
                    selected.cg_tolerance,
                    selected.max_cg_iterations,
                )
                step_norm = float(np.linalg.norm(step))
                if step_norm > selected.max_scaled_parameter_step:
                    step *= selected.max_scaled_parameter_step / step_norm
                    step_norm = selected.max_scaled_parameter_step
                if step_norm < selected.parameter_tolerance:
                    termination = TerminationReason.CONVERGED
                    termination_message = "scaled parameter step reached tolerance"
                    break
                packed = transform.pack()
                accepted = False
                for backtrack in range(selected.max_backtracks + 1):
                    factor = 0.5**backtrack
                    trial_values = transform.unpack(packed + factor * step, clip=True)
                    trial_phases, topology_changes = _apply_parameter_values(
                        phases,
                        input_data.lattice_domains,
                        parameters,
                        trial_values,
                    )
                    trial_parameters = parameters.replace_values(trial_values)
                    trial_linearization = _RietveldLinearization.prepare(
                        input_data,
                        trial_phases,
                        trial_parameters,
                        selected,
                        runtime,
                    )
                    trial_calculation = trial_linearization.calculate()
                    trial_metrics = _metrics(
                        input_data,
                        trial_calculation,
                        selected,
                        len(transform.free_keys),
                    )
                    trial_objective = 0.5 * trial_metrics.chi_square
                    runtime.emit(
                        RefinementEventKind.TRIAL,
                        "rietveld_step",
                        "structural trial evaluated",
                        (("objective", trial_objective), ("backtrack", backtrack)),
                    )
                    if trial_objective < objective:
                        before = parameters.values()
                        changes = tuple(
                            RietveldParameterChange(
                                key,
                                before[key],
                                trial_values[key],
                                (trial_values[key] - before[key]) / parameters.spec(key).scale,
                            )
                            for key in parameters.keys
                            if trial_values[key] != before[key]
                        )
                        change = objective - trial_objective
                        record = RietveldIterationRecord(
                            iteration,
                            trial_metrics.rp,
                            trial_metrics.rwp,
                            trial_metrics.chi_square,
                            trial_metrics.reduced_chi_square,
                            trial_objective,
                            change,
                            factor * step_norm,
                            damping,
                            cg_iterations,
                            backtrack,
                            changes,
                            topology_changes,
                        )
                        history.append(record)
                        phases = trial_phases
                        parameters = trial_parameters
                        calculation = trial_calculation
                        metrics = trial_metrics
                        objective = trial_objective
                        damping = max(damping * selected.damping_decrease, 1.0e-18)
                        transform = ConstraintTransform(parameters, input_data.constraints)
                        linearization = trial_linearization
                        state = _checkpoint(phases, parameters, objective, damping, history)
                        runtime.accept_step(state)
                        runtime.emit(
                            RefinementEventKind.STEP_ACCEPTED,
                            "rietveld_step",
                            "structural step accepted",
                            (("objective", objective), ("rwp", metrics.rwp)),
                        )
                        accepted = True
                        if (
                            iteration >= selected.min_iterations
                            and change <= selected.objective_tolerance * max(objective, 1.0)
                        ):
                            termination = TerminationReason.CONVERGED
                            termination_message = "objective change reached tolerance"
                        break
                    runtime.emit(
                        RefinementEventKind.STEP_REJECTED,
                        "rietveld_step",
                        "structural trial rejected",
                        (("objective", trial_objective), ("backtrack", backtrack)),
                    )
                    runtime.reject_step()
                if termination is TerminationReason.CONVERGED:
                    break
                if not accepted:
                    damping *= selected.damping_increase
                    termination = TerminationReason.STAGNATED
                    termination_message = "no improving bounded step was found"
                    break
                runtime.emit(
                    RefinementEventKind.ITERATION,
                    "rietveld",
                    "structural iteration completed",
                    (("objective", objective), ("rwp", metrics.rwp)),
                )
            else:
                termination = TerminationReason.MAX_ITERATIONS
                termination_message = "iteration limit reached"
    except RefinementStopped as stopped:
        termination = stopped.reason
        termination_message = str(stopped)
        if calculation is None:
            calculation = calculate(input_data.pattern, input_data.experiment, phases)
        metrics = _metrics(input_data, calculation, selected, len(transform.free_keys))
        objective = 0.5 * metrics.chi_square
    except Exception:
        runtime.emit(
            RefinementEventKind.FAILURE,
            "rietveld",
            "unexpected structural refinement failure",
        )
        emergency = _checkpoint(
            phases,
            parameters,
            0.5 * metrics.chi_square if calculation is not None else 0.0,
            damping,
            history,
        )
        if checkpoint_callback is not None:
            with suppress(Exception):
                checkpoint_callback(emergency)
        raise
    if calculation is None:  # pragma: no cover - every guarded path assigns it
        raise RuntimeError("structural refinement produced no calculation")
    final_checkpoint = _checkpoint(phases, parameters, objective, damping, history)
    runtime.emit(
        RefinementEventKind.TERMINATION,
        "rietveld",
        termination_message,
        (("objective", objective), ("rwp", metrics.rwp)),
    )
    return RietveldResult(
        calculation,
        phases,
        parameters,
        metrics,
        tuple(history),
        termination,
        termination_message,
        final_checkpoint,
        runtime.evaluations,
        runtime.logger_error,
    )
