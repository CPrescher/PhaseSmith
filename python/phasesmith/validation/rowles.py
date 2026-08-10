"""PhaseSmith workflow for the neutralized Rowles laboratory QPA patterns."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..background import SmoothBrucknerBackground
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
from ..refinement.runtime import RefinementLimits
from ..sample import IsotropicLorentzianMicrostrainBroadening, IsotropicSizeBroadening
from ..scattering import XrayFixedDispersion
from .real_data import (
    _QARR_COORDINATE_TOLERANCE,
    _QARR_QPA_METADATA,
    QARR_1G_CUKA_FIXED_DISPERSION,
    _qarr_displacement_defaults,
    _qarr_initial_scales,
)

_SCOPE = "curtin_rowles_qpa_topas_common_subset"


@dataclass(frozen=True, slots=True)
class RowlesQpaResult:
    """Finite scientific result from one common-subset PhaseSmith refinement."""

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
        if self.sample not in {"1a", "1e"} or not all(math.isfinite(value) for value in values):
            raise ValueError("Rowles QPA result is invalid or non-finite")
        if self.sample_count <= 0 or self.reflection_count <= 0 or self.free_parameter_count <= 0:
            raise ValueError("Rowles QPA result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible record."""

        return asdict(self)


def _load_manifest(root: Path, sample: str) -> dict[str, Any]:
    if sample not in {"1a", "1e"}:
        raise ValueError("Rowles sample must be '1a' or '1e'")
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != _SCOPE:
        raise ValueError("unsupported neutral Rowles experiment manifest")
    if sample not in manifest.get("patterns", {}):
        raise ValueError(f"neutral Rowles manifest does not contain sample {sample}")
    return manifest


def _solve_linear_profile_block(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    background: ChebyshevBackground,
    execution: ExecutionPolicy,
) -> tuple[tuple[rietveld.RietveldPhase, ...], ChebyshevBackground]:
    """Exactly solve non-negative phase scales and an unconstrained background block."""

    if pattern.observed_y is None or pattern.uncertainty is None:
        raise ValueError("Rowles linear block requires observed values and uncertainties")
    unit_profiles = np.column_stack(
        [
            rietveld.calculate(
                pattern,
                experiment,
                (replace(phase, scale=1.0),),
                support_fwhm=30.0,
                execution=execution,
            ).profile_y
            for phase in phases
        ]
    )
    background_basis = background.basis(pattern.x)
    target = pattern.observed_y - pattern.background
    inverse_uncertainty = 1.0 / pattern.uncertainty
    best_objective = np.inf
    best_solution: NDArray[np.float64] | None = None
    phase_count = len(phases)
    # With three phases, enumerating active scale sets gives an exact small
    # non-negative least-squares solution without adding a SciPy dependency.
    for mask in range(1, 1 << phase_count):
        active = [index for index in range(phase_count) if mask & (1 << index)]
        design = np.column_stack((unit_profiles[:, active], background_basis))
        weighted_design = design * inverse_uncertainty[:, None]
        solution = np.linalg.lstsq(
            weighted_design,
            target * inverse_uncertainty,
            rcond=None,
        )[0]
        if np.any(solution[: len(active)] < 0.0):
            continue
        residual = weighted_design @ solution - target * inverse_uncertainty
        objective = float(residual @ residual)
        if objective < best_objective:
            complete = np.zeros(phase_count + background_basis.shape[1], dtype=np.float64)
            complete[active] = solution[: len(active)]
            complete[phase_count:] = solution[len(active) :]
            best_solution = complete
            best_objective = objective
    if best_solution is None:
        raise ValueError("Rowles linear scale/background block has no feasible solution")
    updated_phases = tuple(
        replace(phase, scale=float(scale))
        for phase, scale in zip(phases, best_solution[:phase_count], strict=True)
    )
    updated_background = background.replace_coefficients(best_solution[phase_count:])
    return updated_phases, updated_background


def _refine_nonlinear_block(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[rietveld.RietveldPhase, ...],
    background: ChebyshevBackground,
    selection: rietveld.RietveldParameterSelection,
    execution: ExecutionPolicy,
    *,
    iterations: int,
    step: float,
) -> rietveld.RietveldResult:
    domains = (None,) * len(phases)
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
            limits=RefinementLimits(max_iterations=iterations, max_evaluations=2_000),
            min_iterations=2,
            max_scaled_parameter_step=step,
            support_fwhm=30.0,
            estimate_covariance=False,
            execution=execution,
        ),
    )


def run_rowles_qpa_workflow(
    bundle_directory: str | Path,
    sample: str,
    *,
    execution: ExecutionPolicy | None = None,
) -> RowlesQpaResult:
    """Refine one converted Rowles pattern with the documented common model."""

    started = perf_counter()
    root = Path(bundle_directory)
    manifest = _load_manifest(root, sample)
    model = manifest["common_model"]
    pattern_record = manifest["patterns"][sample]
    data = read_powder_data(root / pattern_record["file"], format="columns")
    low, high = map(float, model["range_two_theta_deg"])
    selected = (data.x >= low) & (data.x <= high)
    x = np.ascontiguousarray(data.x[selected])
    observed = np.ascontiguousarray(data.observed_y[selected])
    if x.size < 3 or np.any(observed < 0.0):
        raise ValueError("Rowles common range has insufficient or negative count data")
    fixed_background = SmoothBrucknerBackground(
        smooth_width=1.0,
        iterations=50,
        chebyshev_order=None,
    ).estimate(x, observed)
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=fixed_background,
    )
    radiation = model["radiation"]
    components = WavelengthComponents.doublet(
        radiation["lambda1_angstrom"],
        radiation["lambda2_angstrom"],
        radiation["lambda2_over_lambda1_intensity"],
    )
    reference_wavelength = float(components.wavelengths_angstrom[0])
    initializer = model["profile_initializer"]
    instrument = ConstantWavelengthInstrument(
        reference_wavelength,
        initializer["u_deg2"],
        initializer["v_deg2"],
        initializer["w_deg2"],
        initializer["x_deg"],
        initializer["y_deg"],
    )
    experiment = ConstantWavelengthExperiment.x_ray_components(
        instrument,
        components,
        axial_geometry=FcjGeometry(
            initializer["fcj_sample_over_radius"],
            initializer["fcj_detector_over_radius"],
        ),
    )
    correction = BraggBrentanoPolarizedLp(reference_wavelength, radiation["polarization_fraction"])
    scattering = XrayFixedDispersion(QARR_1G_CUKA_FIXED_DISPERSION)
    fixed_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    phases = []
    for phase_id in model["phases"]:
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
        structure = _qarr_displacement_defaults(loaded.phases[0].structure)
        physics = CompositePhysicsProvider(
            (
                IsotropicSizeBroadening(250.0, shape_factor=1.0),
                IsotropicLorentzianMicrostrainBroadening(1.0e-3),
            )
        )
        phases.append(
            replace(
                loaded.phases[0],
                structure=structure,
                physics=physics,
            )
        )
    scales = _qarr_initial_scales(pattern, experiment, phases)
    phases = [replace(phase, scale=scale) for phase, scale in zip(phases, scales, strict=True)]
    background = ChebyshevBackground(
        "rowles_residual",
        tuple(0.0 for _ in range(int(model["background"]["terms"]))),
        (float(x[0]), float(x[-1])),
    )
    selected_execution = ExecutionPolicy() if execution is None else execution
    phases, background = _solve_linear_profile_block(
        pattern, experiment, tuple(phases), background, selected_execution
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
    sample_selection = rietveld.RietveldParameterSelection(
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
        instrument_stage = _refine_nonlinear_block(
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
        phases, background = _solve_linear_profile_block(
            pattern, experiment, instrument_stage.phases, background, selected_execution
        )
        sample_stage = _refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            sample_selection,
            selected_execution,
            iterations=30,
            step=0.10,
        )
        stages.append(sample_stage)
        phases, background = _solve_linear_profile_block(
            pattern,
            sample_stage.experiment,
            sample_stage.phases,
            background,
            selected_execution,
        )
        experiment = sample_stage.experiment
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
    targets = {
        name: float(value) for name, value in pattern_record["weighed_weight_fractions"].items()
    }
    maximum_error = max(abs(fractions[name] - targets[name]) for name in targets)
    free_parameter_count = (
        len(phases)
        + len(background.coefficients)
        + len(instrument_selection.instrument_parameters)
        + sum(len(phase.structure.sites) + 2 for phase in phases)
    )
    residual = calculation.y - observed
    poisson_weight = 1.0 / np.maximum(observed, 1.0)
    poisson_rwp = float(
        np.sqrt(np.sum(poisson_weight * residual**2) / np.sum(poisson_weight * observed**2))
    )
    unit_rwp = float(np.sqrt((residual @ residual) / (observed @ observed)))
    correlation = float(np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1])
    return RowlesQpaResult(
        sample=sample,
        sample_count=x.size,
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
