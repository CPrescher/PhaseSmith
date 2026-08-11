"""Common-model PhaseSmith workflow for the Bath zeolite-L deposit."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..execution import ExecutionPolicy
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..pattern import PowderPattern
from ..radiation import ConstantWavelengthExperiment
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import _qarr_initial_scales

_SCOPE = "bath_ltl_lab_xray_conversion_fidelity"


@dataclass(frozen=True, slots=True)
class BathLtlResult:
    """Finite metrics from one primary-phase publication-CIF translation."""

    sample: str
    sample_count: int
    reflection_count: int
    released_poisson_rwp: float
    fixed_profile_poisson_rwp: float
    refined_profile_poisson_rwp: float
    unit_weight_rwp: float
    profile_correlation: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.released_poisson_rwp,
            self.fixed_profile_poisson_rwp,
            self.refined_profile_poisson_rwp,
            self.unit_weight_rwp,
            self.profile_correlation,
            self.elapsed_seconds,
        )
        if self.sample not in {"K", "Li", "Cs"} or not all(map(math.isfinite, values)):
            raise ValueError("Bath LTL result is invalid or non-finite")
        if self.sample_count <= 0 or self.reflection_count <= 0:
            raise ValueError("Bath LTL result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(calculation: Any, observed: np.ndarray) -> tuple[float, float, float]:
    residual = calculation.y - observed
    weights = 1.0 / np.maximum(observed, 1.0)
    poisson = float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights * observed**2)))
    unit = float(np.linalg.norm(residual) / np.linalg.norm(observed))
    correlation = float(np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1])
    return poisson, unit, correlation


def run_bath_ltl_workflow(
    bundle_directory: str | Path,
    sample: str,
    *,
    execution: ExecutionPolicy | None = None,
    profile_cycles: int = 2,
) -> BathLtlResult:
    """Refine the primary publication CIF against one released GSAS curve.

    The deposited background is fixed. A single residual constant and phase
    scale are solved linearly. The archived effective U/V/W/X/Y and axial
    profile initialize two bounded instrument-only refinement cycles.
    """

    started = perf_counter()
    if sample not in {"K", "Li", "Cs"}:
        raise ValueError("Bath LTL sample must be 'K', 'Li' or 'Cs'")
    if (
        not isinstance(profile_cycles, int)
        or isinstance(profile_cycles, bool)
        or profile_cycles < 0
    ):
        raise ValueError("profile_cycles must be a nonnegative integer")
    root = Path(bundle_directory)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != _SCOPE:
        raise ValueError("unsupported Bath LTL bundle manifest")
    record = manifest["samples"][sample]
    profile = np.loadtxt(root / record["released_profile"], delimiter=",", skiprows=1)
    if profile.ndim != 2 or profile.shape[1] != 4:
        raise ValueError("Bath released profile must have four columns")
    x, observed, _released_calculated, fixed_background = profile.T
    pattern = PowderPattern(
        np.ascontiguousarray(x),
        observed_y=np.ascontiguousarray(observed),
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=np.ascontiguousarray(fixed_background),
    )
    legacy = record["legacy_gsas"]
    parameters = legacy["profile"]
    wavelength = float(legacy["wavelength_angstrom"])
    instrument = ConstantWavelengthInstrument(
        wavelength,
        parameters["u_deg2"],
        parameters["v_deg2"],
        parameters["w_deg2"],
        parameters["x_deg"],
        parameters["y_deg"],
    )
    half_axial = float(parameters["sh_over_l"]) / 2.0
    experiment = ConstantWavelengthExperiment.x_ray(
        instrument,
        axial_geometry=FcjGeometry(half_axial, half_axial),
    )
    selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    loaded = rietveld.RietveldInput.from_cif(
        pattern,
        experiment,
        root / record["phase"],
        phase_id="LTL",
        selection=selection,
        intensity_correction=BraggBrentanoPolarizedLp(wavelength, 0.7),
    )
    phases = list(loaded.phases)
    scales = _qarr_initial_scales(pattern, experiment, phases)
    from dataclasses import replace

    phases = [replace(phases[0], scale=scales[0])]
    background = ChebyshevBackground("bath_residual", (0.0,), (float(x[0]), float(x[-1])))
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    phases, background = solve_linear_profile_block(
        pattern, experiment, tuple(phases), background, selected_execution
    )
    fixed_calculation = rietveld.calculate(
        pattern,
        experiment,
        phases,
        background=background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    fixed_poisson, _fixed_unit, _fixed_correlation = _metrics(fixed_calculation, observed)
    instrument_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=(
            "u_deg2",
            "v_deg2",
            "w_deg2",
            "x_deg",
            "y_deg",
            "zero_shift_deg",
        ),
        background=False,
    )
    for _cycle in range(profile_cycles):
        stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            instrument_selection,
            selected_execution,
            iterations=25,
            step=0.1,
        )
        experiment = stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, stage.phases, background, selected_execution
        )
    calculation = rietveld.calculate(
        pattern,
        experiment,
        phases,
        background=background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    poisson, unit, correlation = _metrics(calculation, observed)
    return BathLtlResult(
        sample=sample,
        sample_count=int(x.size),
        reflection_count=phases[0].reflections.reflection_count,
        released_poisson_rwp=float(legacy["recomputed_poisson_rwp"]),
        fixed_profile_poisson_rwp=fixed_poisson,
        refined_profile_poisson_rwp=poisson,
        unit_weight_rwp=unit,
        profile_correlation=correlation,
        elapsed_seconds=perf_counter() - started,
    )
