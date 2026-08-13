"""Common-model workflow for the IUCr citrate/Si internal-standard pattern."""

from __future__ import annotations

import json
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
from ..io.iucr_silicon_standard import IUCR_SILICON_PHASES
from ..pattern import PowderPattern
from ..quantitative import QuantitativePhase, quantitative_phase_analysis
from ..radiation import BraggBrentanoGeometry, ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..sample import IsotropicLorentzianMicrostrainBroadening, IsotropicSizeBroadening
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import _qarr_displacement_defaults, _qarr_initial_scales

_SCOPE = "iucr_dicesium_citrate_silicon_internal_standard"


@dataclass(frozen=True, slots=True)
class IucrSiliconStandardResult:
    """Finite common-model result compared with the deposited legacy refinement."""

    sample_count: int
    reflection_count: int
    free_parameter_count: int
    weight_fractions: dict[str, float]
    legacy_weight_fraction_errors: dict[str, float]
    maximum_legacy_weight_fraction_error: float
    poisson_rwp: float
    unit_weight_rwp: float
    profile_correlation: float
    legacy_curve_poisson_rwp: float
    legacy_curve_profile_correlation: float
    silicon_calibration_sample_count: int
    silicon_calibration_poisson_rwp: float
    calibrated_zero_shift_deg: float
    calibrated_sample_displacement_mm: float
    refined_instrument: dict[str, float]
    termination_reasons: tuple[str, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            *self.weight_fractions.values(),
            *self.legacy_weight_fraction_errors.values(),
            self.maximum_legacy_weight_fraction_error,
            self.poisson_rwp,
            self.unit_weight_rwp,
            self.profile_correlation,
            self.legacy_curve_poisson_rwp,
            self.legacy_curve_profile_correlation,
            self.silicon_calibration_poisson_rwp,
            self.calibrated_zero_shift_deg,
            self.calibrated_sample_displacement_mm,
            *self.refined_instrument.values(),
            self.elapsed_seconds,
        )
        expected = set(IUCR_SILICON_PHASES)
        if (
            set(self.weight_fractions) != expected
            or set(self.legacy_weight_fraction_errors) != expected
            or not all(math.isfinite(value) for value in values)
        ):
            raise ValueError("IUCr silicon-standard result is invalid or non-finite")
        if (
            self.sample_count <= 0
            or self.silicon_calibration_sample_count <= 0
            or self.reflection_count <= 0
            or self.free_parameter_count <= 0
        ):
            raise ValueError("IUCr silicon-standard result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _manifest(root: Path) -> dict[str, Any]:
    record = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("scope") != _SCOPE:
        raise ValueError("unsupported IUCr silicon-standard experiment manifest")
    return record


def run_iucr_silicon_standard_workflow(
    bundle_directory: str | Path,
    *,
    execution: ExecutionPolicy | None = None,
    cycles: int = 2,
) -> IucrSiliconStandardResult:
    """Refine the converted three-phase pattern with the disclosed common model."""

    started = perf_counter()
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    root = Path(bundle_directory)
    manifest = _manifest(root)
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    if data.shape != (2820, 4) or not np.all(np.isfinite(data)):
        raise ValueError("IUCr silicon-standard pattern has an unexpected shape or values")
    x, observed, legacy_calculated, legacy_background = data.T
    if np.any(np.diff(x) <= 0.0) or np.any(observed < 0.0):
        raise ValueError("IUCr silicon-standard axis or counts are invalid")
    pattern = PowderPattern(
        np.ascontiguousarray(x),
        observed_y=np.ascontiguousarray(observed),
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=np.ascontiguousarray(legacy_background),
    )
    instrument_record = manifest["instrument"]
    wavelengths = instrument_record["wavelengths_angstrom"]
    components = WavelengthComponents.doublet(
        float(wavelengths[0]),
        float(wavelengths[1]),
        float(instrument_record["k_alpha2_over_k_alpha1"]),
    )
    profile = instrument_record["initial_profile"]
    instrument = ConstantWavelengthInstrument(
        float(wavelengths[0]),
        float(profile["U"]) * 1.0e-4,
        float(profile["V"]) * 1.0e-4,
        float(profile["W"]) * 1.0e-4,
        float(profile["X"]) * 1.0e-2,
        float(profile["Y"]) * 1.0e-2,
    )
    half_axial = float(instrument_record["sh_over_l"]) / 2.0
    experiment = replace(
        ConstantWavelengthExperiment.x_ray_components(
            instrument,
            components,
            geometry=BraggBrentanoGeometry(141.5, 0.0),
            axial_geometry=FcjGeometry(half_axial, half_axial),
        ),
        zero_shift_deg=float(instrument_record["initial_zero_deg"]),
    )
    correction = BraggBrentanoPolarizedLp(
        float(wavelengths[0]), float(instrument_record["polarization_fraction"])
    )
    fixed_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    phases = []
    for phase_id in IUCR_SILICON_PHASES:
        loaded = rietveld.RietveldInput.from_cif(
            pattern,
            experiment,
            root / f"{phase_id}.cif",
            phase_id=phase_id,
            selection=fixed_selection,
            strict=False,
            intensity_correction=correction,
        )
        phases.append(
            replace(
                loaded.phases[0],
                structure=_qarr_displacement_defaults(loaded.phases[0].structure),
                physics=CompositePhysicsProvider(
                    (
                        IsotropicSizeBroadening(250.0, shape_factor=1.0),
                        IsotropicLorentzianMicrostrainBroadening(1.0e-3),
                    )
                ),
            )
        )
    calibration_windows = manifest["silicon_calibration"]["windows_two_theta_deg"]
    calibration_mask = np.zeros(x.shape, dtype=np.bool_)
    for low, high in calibration_windows:
        calibration_mask |= (x >= float(low)) & (x <= float(high))
    calibration_pattern = PowderPattern(
        np.ascontiguousarray(x[calibration_mask]),
        observed_y=np.ascontiguousarray(observed[calibration_mask]),
        uncertainty=np.sqrt(np.maximum(observed[calibration_mask], 1.0)),
        background=np.ascontiguousarray(legacy_background[calibration_mask]),
    )
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    silicon = next(phase for phase in phases if phase.phase_id == "silicon")
    silicon_scale = _qarr_initial_scales(calibration_pattern, experiment, (silicon,))[0]
    calibration_phase = replace(silicon, scale=silicon_scale)
    calibration_background = ChebyshevBackground(
        "silicon_calibration_residual",
        (0.0,),
        (float(calibration_pattern.x[0]), float(calibration_pattern.x[-1])),
    )
    calibration_phase_tuple, calibration_background = solve_linear_profile_block(
        calibration_pattern,
        experiment,
        (calibration_phase,),
        calibration_background,
        selected_execution,
    )
    position_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("sample_displacement_mm",),
        background=False,
    )
    for _calibration_cycle in range(3):
        position_stage = refine_nonlinear_block(
            calibration_pattern,
            experiment,
            calibration_phase_tuple,
            calibration_background,
            position_selection,
            selected_execution,
            iterations=30,
            step=0.5,
        )
        experiment = position_stage.experiment
        calibration_phase_tuple, calibration_background = solve_linear_profile_block(
            calibration_pattern,
            experiment,
            position_stage.phases,
            calibration_background,
            selected_execution,
        )
    calibrated_zero = experiment.zero_shift_deg
    if not isinstance(experiment.geometry, BraggBrentanoGeometry):
        raise RuntimeError("silicon calibration lost its Bragg-Brentano geometry")
    calibrated_displacement = experiment.geometry.sample_displacement_mm
    calibration_calculation = rietveld.calculate(
        calibration_pattern,
        experiment,
        calibration_phase_tuple,
        background=calibration_background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    calibration_residual = calibration_calculation.y - calibration_pattern.observed_y
    calibration_weight = 1.0 / np.maximum(calibration_pattern.observed_y, 1.0)
    calibration_rwp = float(
        np.sqrt(
            np.sum(calibration_weight * np.square(calibration_residual))
            / np.sum(calibration_weight * np.square(calibration_pattern.observed_y))
        )
    )
    scales = _qarr_initial_scales(pattern, experiment, phases)
    selected_phases = tuple(
        replace(phase, scale=scale) for phase, scale in zip(phases, scales, strict=True)
    )
    background = ChebyshevBackground(
        "iucr_legacy_background_residual", (0.0,), (float(x[0]), float(x[-1]))
    )
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
        instrument_parameters=(
            "u_deg2",
            "v_deg2",
            "w_deg2",
            "x_deg",
            "y_deg",
        ),
        background=False,
    )
    sample_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=False,
    )
    stages = []
    for _cycle in range(cycles):
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
        stages.append(instrument_stage)
        experiment = instrument_stage.experiment
        selected_phases, background = solve_linear_profile_block(
            pattern, experiment, instrument_stage.phases, background, selected_execution
        )
        sample_stage = refine_nonlinear_block(
            pattern,
            experiment,
            selected_phases,
            background,
            sample_selection,
            selected_execution,
            iterations=30,
            step=0.10,
        )
        stages.append(sample_stage)
        experiment = sample_stage.experiment
        selected_phases, background = solve_linear_profile_block(
            pattern, experiment, sample_stage.phases, background, selected_execution
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
            float(phase.structure.metadata["formula_mass_g_mol"]),
            phase.structure.cell.geometry().volume_angstrom3,
        )
        for phase in selected_phases
    )
    fractions = {
        item.phase_id: item.weight_fraction for item in quantitative_phase_analysis(quantitative)
    }
    targets = {
        key: float(value)
        for key, value in manifest["legacy_gsas_reference"]["weight_fractions"].items()
    }
    errors = {phase_id: fractions[phase_id] - targets[phase_id] for phase_id in targets}
    residual = calculation.y - observed
    weight = 1.0 / np.maximum(observed, 1.0)
    legacy_residual = legacy_calculated - observed
    return IucrSiliconStandardResult(
        sample_count=int(x.size),
        reflection_count=sum(phase.reflections.reflection_count for phase in selected_phases),
        free_parameter_count=len(selected_phases) + 1 + 6 + 2 * len(selected_phases),
        weight_fractions=fractions,
        legacy_weight_fraction_errors=errors,
        maximum_legacy_weight_fraction_error=max(abs(value) for value in errors.values()),
        poisson_rwp=float(
            np.sqrt(np.sum(weight * np.square(residual)) / np.sum(weight * np.square(observed)))
        ),
        unit_weight_rwp=float(np.linalg.norm(residual) / np.linalg.norm(observed)),
        profile_correlation=float(
            np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1]
        ),
        legacy_curve_poisson_rwp=float(
            np.sqrt(
                np.sum(weight * np.square(legacy_residual)) / np.sum(weight * np.square(observed))
            )
        ),
        legacy_curve_profile_correlation=float(
            np.corrcoef(observed - legacy_background, legacy_calculated - legacy_background)[0, 1]
        ),
        silicon_calibration_sample_count=int(np.count_nonzero(calibration_mask)),
        silicon_calibration_poisson_rwp=calibration_rwp,
        calibrated_zero_shift_deg=calibrated_zero,
        calibrated_sample_displacement_mm=calibrated_displacement,
        refined_instrument={
            "U": 1.0e4 * experiment.instrument.u_deg2,
            "V": 1.0e4 * experiment.instrument.v_deg2,
            "W": 1.0e4 * experiment.instrument.w_deg2,
            "X": 1.0e2 * experiment.instrument.x_deg,
            "Y": 1.0e2 * experiment.instrument.y_deg,
            "Zero": experiment.zero_shift_deg,
            "SH/L": 2.0 * half_axial,
        },
        termination_reasons=tuple(stage.termination_reason.value for stage in stages),
        elapsed_seconds=perf_counter() - started,
    )
