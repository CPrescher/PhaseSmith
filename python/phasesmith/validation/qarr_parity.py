"""Matched common-model PhaseSmith workflow for IUCr QARR 1g and 1h."""

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
from ..io.powder import read_powder_data
from ..pattern import PowderPattern
from ..quantitative import QuantitativePhase, quantitative_phase_analysis
from ..radiation import ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..sample import IsotropicLorentzianMicrostrainBroadening, IsotropicSizeBroadening
from ..scattering import XrayFixedDispersion
from ..structure import CrystalStructure
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import (
    _QARR_COORDINATE_TOLERANCE,
    _QARR_QPA_METADATA,
    QARR_1G_CUKA_FIXED_DISPERSION,
    _qarr_displacement_defaults,
    _qarr_initial_scales,
    _qarr_instrument_values,
)

_PATTERNS = {"1g": "cpd-1g.prn", "1h": "cpd-1h.prn"}
_TARGETS = {
    "1g": {"Al2O3": 0.3137, "ZnO": 0.3421, "CaF2": 0.3442},
    "1h": {"Al2O3": 0.3512, "ZnO": 0.3019, "CaF2": 0.3469},
}


def _trace_mean_isotropic_structure(structure: CrystalStructure) -> CrystalStructure:
    """Replace each CIF anisotropic tensor by its documented diagonal trace mean."""

    return replace(
        structure,
        sites=tuple(
            replace(
                site,
                u_iso_angstrom2=float(np.mean(site.anisotropic_displacement.u_cif_angstrom2[:3])),
                anisotropic_displacement=None,
            )
            if site.anisotropic_displacement is not None
            else site
            for site in structure.sites
        ),
    )


@dataclass(frozen=True, slots=True)
class QarrParityResult:
    """Finite result from one matched common-model QARR refinement."""

    sample: str
    sample_count: int
    reflection_count: int
    free_parameter_count: int
    weight_fractions: dict[str, float]
    maximum_weight_fraction_error: float
    poisson_rwp: float
    unit_weight_rwp: float
    profile_correlation: float
    termination_reasons: tuple[str, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            *self.weight_fractions.values(),
            self.maximum_weight_fraction_error,
            self.poisson_rwp,
            self.unit_weight_rwp,
            self.profile_correlation,
            self.elapsed_seconds,
        )
        if self.sample not in _PATTERNS or not all(math.isfinite(value) for value in values):
            raise ValueError("QARR parity result is invalid or non-finite")
        if self.sample_count <= 0 or self.reflection_count <= 0 or self.free_parameter_count <= 0:
            raise ValueError("QARR parity result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible record."""

        return asdict(self)


def run_qarr_gsasii_parity_workflow(
    dataset_directory: str | Path,
    sample: str,
    *,
    execution: ExecutionPolicy | None = None,
) -> QarrParityResult:
    """Refine QARR 1g or 1h with the model shared by the pinned GSAS-II worker."""

    if sample not in _PATTERNS:
        raise ValueError("QARR parity sample must be '1g' or '1h'")
    started = perf_counter()
    root = Path(dataset_directory)
    data = read_powder_data(root / _PATTERNS[sample], format="columns")
    observed = np.ascontiguousarray(data.observed_y)
    pattern = PowderPattern(
        data.x,
        observed_y=observed,
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=np.zeros_like(observed),
    )
    values = _qarr_instrument_values(root / "cuka.instprm")
    instrument = ConstantWavelengthInstrument(
        values["Lam1"],
        values["U"] * 1.0e-4,
        values["V"] * 1.0e-4,
        values["W"] * 1.0e-4,
        values["X"] * 1.0e-2,
        values["Y"] * 1.0e-2,
    )
    experiment = ConstantWavelengthExperiment.x_ray_components(
        instrument,
        WavelengthComponents.doublet(values["Lam1"], values["Lam2"], values["I(L2)/I(L1)"]),
        axial_geometry=FcjGeometry(values["SH/L"] / 2.0, values["SH/L"] / 2.0),
    )
    scattering = XrayFixedDispersion(QARR_1G_CUKA_FIXED_DISPERSION)
    correction = BraggBrentanoPolarizedLp(values["Lam1"], values["Polariz."])
    fixed_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    phases = []
    for phase_id in _TARGETS[sample]:
        loaded = rietveld.RietveldInput.from_cif(
            pattern,
            experiment,
            root / f"{phase_id}.cif",
            phase_id=phase_id,
            selection=fixed_selection,
            scattering=scattering,
            intensity_correction=correction,
            coordinate_tolerance=_QARR_COORDINATE_TOLERANCE,
        )
        structure = _trace_mean_isotropic_structure(
            _qarr_displacement_defaults(loaded.phases[0].structure)
        )
        physics = CompositePhysicsProvider(
            (
                IsotropicSizeBroadening(250.0, shape_factor=1.0),
                IsotropicLorentzianMicrostrainBroadening(1.0e-3),
            )
        )
        phases.append(replace(loaded.phases[0], structure=structure, physics=physics))
    initial_scales = _qarr_initial_scales(pattern, experiment, phases)
    phases = tuple(
        replace(phase, scale=scale) for phase, scale in zip(phases, initial_scales, strict=True)
    )
    background = ChebyshevBackground(
        "qarr_matched_background",
        tuple(0.0 for _ in range(10)),
        (float(data.x[0]), float(data.x[-1])),
    )
    selected_execution = ExecutionPolicy() if execution is None else execution
    phases, background = solve_linear_profile_block(
        pattern, experiment, phases, background, selected_execution
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
        u_iso=True,
        sample_physics=True,
        background=False,
    )
    stages = []
    for _cycle in range(3):
        instrument_stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            instrument_selection,
            selected_execution,
            iterations=24,
            step=0.12,
        )
        stages.append(instrument_stage)
        experiment = instrument_stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, instrument_stage.phases, background, selected_execution
        )
        specimen_stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            specimen_selection,
            selected_execution,
            iterations=30,
            step=0.10,
        )
        stages.append(specimen_stage)
        experiment = specimen_stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, specimen_stage.phases, background, selected_execution
        )
    calculation = rietveld.calculate(
        pattern,
        experiment,
        phases,
        background=background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    quantitative = tuple(
        QuantitativePhase(
            phase.phase_id,
            phase.scale,
            _QARR_QPA_METADATA[phase.phase_id][0],
            _QARR_QPA_METADATA[phase.phase_id][1],
            phase.structure.cell.geometry().volume_angstrom3,
        )
        for phase in phases
    )
    fractions = {
        item.phase_id: item.weight_fraction for item in quantitative_phase_analysis(quantitative)
    }
    maximum_error = max(abs(fractions[name] - _TARGETS[sample][name]) for name in fractions)
    residual = calculation.y - observed
    poisson_weight = 1.0 / np.maximum(observed, 1.0)
    poisson_rwp = float(
        np.sqrt(np.sum(poisson_weight * residual**2) / np.sum(poisson_weight * observed**2))
    )
    unit_rwp = float(np.sqrt((residual @ residual) / (observed @ observed)))
    correlation = float(np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1])
    free_parameter_count = (
        len(phases)
        + len(background.coefficients)
        + len(instrument_selection.instrument_parameters)
        + sum(len(phase.structure.sites) + 2 for phase in phases)
    )
    return QarrParityResult(
        sample=sample,
        sample_count=data.x.size,
        reflection_count=sum(phase.reflections.reflection_count for phase in phases),
        free_parameter_count=free_parameter_count,
        weight_fractions=fractions,
        maximum_weight_fraction_error=maximum_error,
        poisson_rwp=poisson_rwp,
        unit_weight_rwp=unit_rwp,
        profile_correlation=correlation,
        termination_reasons=tuple(stage.termination_reason.value for stage in stages),
        elapsed_seconds=perf_counter() - started,
    )
