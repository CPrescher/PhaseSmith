"""Common-model anatase/rutile workflow for the experimental XRED pattern."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..execution import ExecutionPolicy
from ..extensions import CompositePhysicsProvider
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..pattern import PowderPattern
from ..quantitative import QuantitativePhase, quantitative_phase_analysis
from ..radiation import ConstantWavelengthExperiment
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..refinement.runtime import RefinementLimits
from ..sample import IsotropicLorentzianMicrostrainBroadening, IsotropicSizeBroadening
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import _qarr_displacement_defaults, _qarr_initial_scales

_PHASE_FILES = {"anatase": "anatase.cif", "rutile": "rutile.cif"}
_TIO2_FORMULA_MASS = 79.866


@dataclass(frozen=True, slots=True)
class XredTio2Result:
    """Finite common-model result without a claimed certified composition."""

    sample_count: int
    reflection_count: int
    weight_fractions: dict[str, float]
    cells_angstrom: dict[str, tuple[float, float]]
    poisson_rwp: float
    unit_weight_rwp: float
    profile_correlation: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            *self.weight_fractions.values(),
            *(value for cell in self.cells_angstrom.values() for value in cell),
            self.poisson_rwp,
            self.unit_weight_rwp,
            self.profile_correlation,
            self.elapsed_seconds,
        )
        if set(self.weight_fractions) != set(_PHASE_FILES) or not all(map(math.isfinite, values)):
            raise ValueError("XRED TiO2 result is invalid or non-finite")
        if self.sample_count <= 0 or self.reflection_count <= 0:
            raise ValueError("XRED TiO2 result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _guarded_lattice_block(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    domains: tuple[Any, ...],
    background: ChebyshevBackground,
    execution: ExecutionPolicy,
) -> rietveld.RietveldResult:
    selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=True,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        background=False,
    )
    input_data = rietveld.RietveldInput(
        pattern,
        experiment,
        phases,
        domains,
        rietveld.build_parameter_set(
            phases,
            domains,
            selection,
            experiment=experiment,
            background=background,
        ),
        selection=selection,
        background=background,
    )
    return rietveld.refine(
        input_data,
        rietveld.RietveldOptions(
            limits=RefinementLimits(max_iterations=30, max_evaluations=2_000),
            min_iterations=2,
            max_scaled_parameter_step=0.03,
            support_fwhm=30.0,
            estimate_covariance=False,
            execution=execution,
        ),
    )


def run_xred_tio2_workflow(
    dataset_directory: str | Path,
    *,
    execution: ExecutionPolicy | None = None,
    cycles: int = 3,
) -> XredTio2Result:
    """Fit XRED TiO2 with a disclosed assumed Cu K-alpha common model."""

    started = perf_counter()
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    root = Path(dataset_directory)
    data = np.loadtxt(root / "data.csv", delimiter=",")
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 3:
        raise ValueError("XRED pattern must contain two numeric columns")
    x, observed = data.T
    if not np.all(np.isfinite(data)) or np.any(np.diff(x) <= 0.0) or np.any(observed < 0.0):
        raise ValueError("XRED pattern axis and counts are invalid")
    pattern = PowderPattern(
        np.ascontiguousarray(x),
        observed_y=np.ascontiguousarray(observed),
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=np.zeros_like(observed),
    )
    wavelength = 1.54051
    instrument = ConstantWavelengthInstrument(wavelength, 2e-4, -2e-4, 5e-4, 0.0, 0.0)
    experiment = ConstantWavelengthExperiment.x_ray(
        instrument, axial_geometry=FcjGeometry(0.001, 0.001)
    )
    guarded_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=True,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    phases = []
    domains = []
    for phase_id, filename in _PHASE_FILES.items():
        loaded = rietveld.RietveldInput.from_cif(
            pattern,
            experiment,
            root / filename,
            phase_id=phase_id,
            selection=guarded_selection,
            intensity_correction=BraggBrentanoPolarizedLp(wavelength, 0.7),
        )
        structure = _qarr_displacement_defaults(loaded.phases[0].structure)
        structure = replace(
            structure,
            sites=tuple(
                replace(site, type_symbol=site.element_symbol, charge=None)
                for site in structure.sites
            ),
        )
        phases.append(
            replace(
                loaded.phases[0],
                structure=structure,
                physics=CompositePhysicsProvider(
                    (
                        IsotropicSizeBroadening(250.0, shape_factor=1.0),
                        IsotropicLorentzianMicrostrainBroadening(1.0e-3),
                    )
                ),
            )
        )
        domains.append(loaded.lattice_domains[0])
    scales = _qarr_initial_scales(pattern, experiment, phases)
    selected_phases = tuple(
        replace(phase, scale=scale) for phase, scale in zip(phases, scales, strict=True)
    )
    selected_domains = tuple(domains)
    background = ChebyshevBackground("xred", (0.0,) * 8, (float(x[0]), float(x[-1])))
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    selected_phases, background = solve_linear_profile_block(
        pattern, experiment, selected_phases, background, selected_execution
    )
    instrument_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "zero_shift_deg"),
        background=False,
    )
    specimen_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=False,
    )
    for _cycle in range(cycles):
        lattice_stage = _guarded_lattice_block(
            pattern,
            experiment,
            selected_phases,
            selected_domains,
            background,
            selected_execution,
        )
        experiment = lattice_stage.experiment
        selected_domains = lattice_stage.checkpoint.lattice_domains
        selected_phases, background = solve_linear_profile_block(
            pattern, experiment, lattice_stage.phases, background, selected_execution
        )
        instrument_stage = refine_nonlinear_block(
            pattern,
            experiment,
            selected_phases,
            background,
            instrument_selection,
            selected_execution,
            iterations=25,
            step=0.12,
        )
        experiment = instrument_stage.experiment
        selected_phases, background = solve_linear_profile_block(
            pattern, experiment, instrument_stage.phases, background, selected_execution
        )
        specimen_stage = refine_nonlinear_block(
            pattern,
            experiment,
            selected_phases,
            background,
            specimen_selection,
            selected_execution,
            iterations=25,
            step=0.1,
        )
        experiment = specimen_stage.experiment
        selected_phases, background = solve_linear_profile_block(
            pattern, experiment, specimen_stage.phases, background, selected_execution
        )
    calculation = rietveld.calculate(
        pattern,
        experiment,
        selected_phases,
        background=background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    quantitative = tuple(
        QuantitativePhase(
            phase.phase_id,
            phase.scale,
            float(phase.structure.metadata["formula_units_per_cell"]),
            _TIO2_FORMULA_MASS,
            phase.structure.cell.geometry().volume_angstrom3,
        )
        for phase in selected_phases
    )
    fractions = {
        item.phase_id: item.weight_fraction for item in quantitative_phase_analysis(quantitative)
    }
    residual = calculation.y - observed
    weights = 1.0 / np.maximum(observed, 1.0)
    poisson = float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights * observed**2)))
    unit = float(np.linalg.norm(residual) / np.linalg.norm(observed))
    correlation = float(np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1])
    return XredTio2Result(
        sample_count=int(x.size),
        reflection_count=sum(phase.reflections.reflection_count for phase in selected_phases),
        weight_fractions=fractions,
        cells_angstrom={
            phase.phase_id: (phase.structure.cell.a_angstrom, phase.structure.cell.c_angstrom)
            for phase in selected_phases
        },
        poisson_rwp=poisson,
        unit_weight_rwp=unit,
        profile_correlation=correlation,
        elapsed_seconds=perf_counter() - started,
    )
