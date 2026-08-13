"""Common-model workflow for the IUCr tripotassium-citrate/Si holdout."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..execution import ExecutionPolicy
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..io.iucr_tripotassium_citrate_silicon import (
    IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES,
)
from ..pattern import PowderPattern
from ..quantitative import QuantitativePhase, quantitative_phase_analysis
from ..radiation import BraggBrentanoGeometry, ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block
from .real_data import _qarr_displacement_defaults, _qarr_initial_scales

_SCOPE = "iucr_anhydrous_tripotassium_citrate_silicon_holdout"


@dataclass(frozen=True, slots=True)
class IucrTripotassiumCitrateSiliconResult:
    """Finite common-model result with explicit source-model qualifications."""

    sample_count: int
    reflection_count: int
    free_parameter_count: int
    weight_fractions: dict[str, float]
    legacy_weight_fraction_errors: dict[str, float]
    maximum_legacy_weight_fraction_error: float
    poisson_rwp: float
    profile_correlation: float
    legacy_curve_poisson_rwp: float
    legacy_curve_profile_correlation: float
    silicon_calibration_sample_count: int
    silicon_calibration_poisson_rwp: float
    calibrated_sample_displacement_mm: float
    refined_instrument: dict[str, float]
    model_qualifications: tuple[str, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            *self.weight_fractions.values(),
            *self.legacy_weight_fraction_errors.values(),
            self.maximum_legacy_weight_fraction_error,
            self.poisson_rwp,
            self.profile_correlation,
            self.legacy_curve_poisson_rwp,
            self.legacy_curve_profile_correlation,
            self.silicon_calibration_poisson_rwp,
            self.calibrated_sample_displacement_mm,
            *self.refined_instrument.values(),
            self.elapsed_seconds,
        )
        expected = set(IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES)
        if (
            set(self.weight_fractions) != expected
            or set(self.legacy_weight_fraction_errors) != expected
            or not all(math.isfinite(value) for value in values)
            or not self.model_qualifications
        ):
            raise ValueError("IUCr tripotassium-citrate/Si result is invalid or non-finite")
        if (
            self.sample_count <= 0
            or self.silicon_calibration_sample_count <= 0
            or self.reflection_count <= 0
            or self.free_parameter_count <= 0
        ):
            raise ValueError("IUCr tripotassium-citrate/Si result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _manifest(root: Path) -> dict[str, Any]:
    record = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("scope") != _SCOPE:
        raise ValueError("unsupported IUCr tripotassium-citrate/Si experiment manifest")
    return record


def run_iucr_tripotassium_citrate_silicon_workflow(
    bundle_directory: str | Path,
    *,
    execution: ExecutionPolicy | None = None,
) -> IucrTripotassiumCitrateSiliconResult:
    """Refine scales/background after a disclosed fixed-profile Si calibration.

    The source reports a texture index of 1.001, so no preferred-orientation
    parameter is added. Deposited background, zero, structures, and all
    phase-width terms are fixed. The low-fraction Si phase independently
    diagnoses specimen displacement, but the diagnostic value is not applied
    to the two-phase linear solve unless a separate identifiability comparison
    establishes transferability.
    """

    started = perf_counter()
    root = Path(bundle_directory)
    manifest = _manifest(root)
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    expected_count = int(manifest["data_selection"]["refined_sample_count"])
    if data.shape != (expected_count, 4) or not np.all(np.isfinite(data)):
        raise ValueError("IUCr tripotassium-citrate/Si pattern has unexpected values")
    x, observed, legacy_calculated, legacy_background = data.T
    if np.any(np.diff(x) <= 0.0) or np.any(observed < 0.0):
        raise ValueError("IUCr tripotassium-citrate/Si axis or counts are invalid")
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
    profile = instrument_record["tripotassium_citrate_profile"]
    instrument = ConstantWavelengthInstrument(
        float(wavelengths[0]),
        float(profile["U"]) * 1.0e-4,
        float(profile["V"]) * 1.0e-4,
        float(profile["W"]) * 1.0e-4,
        float(profile["X"]) * 1.0e-2,
        float(profile["Y"]) * 1.0e-2,
    )
    experiment = replace(
        ConstantWavelengthExperiment.x_ray_components(
            instrument,
            components,
            geometry=BraggBrentanoGeometry(
                float(instrument_record["assumed_goniometer_radius_mm"]), 0.0
            ),
            axial_geometry=FcjGeometry(
                float(instrument_record["matched_sh_over_l"]) / 2.0,
                float(instrument_record["matched_sh_over_l"]) / 2.0,
            ),
        ),
        zero_shift_deg=float(instrument_record["initial_zero_deg"]),
    )
    silicon_profile = instrument_record["silicon_profile"]
    silicon_instrument = ConstantWavelengthInstrument(
        float(wavelengths[0]),
        float(silicon_profile["U"]) * 1.0e-4,
        float(silicon_profile["V"]) * 1.0e-4,
        float(silicon_profile["W"]) * 1.0e-4,
        float(silicon_profile["X"]) * 1.0e-2,
        float(silicon_profile["Y"]) * 1.0e-2,
    )
    calibration_experiment = replace(experiment, instrument=silicon_instrument)
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
    for phase_id in IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES:
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
            )
        )

    calibration_mask = np.zeros(x.shape, dtype=np.bool_)
    for low, high in manifest["silicon_standard"]["candidate_calibration_windows_two_theta_deg"]:
        calibration_mask |= (x >= float(low)) & (x <= float(high))
    calibration_pattern = PowderPattern(
        np.ascontiguousarray(x[calibration_mask]),
        observed_y=np.ascontiguousarray(observed[calibration_mask]),
        uncertainty=np.sqrt(np.maximum(observed[calibration_mask], 1.0)),
        background=np.ascontiguousarray(legacy_background[calibration_mask]),
    )
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    silicon = next(phase for phase in phases if phase.phase_id == "silicon")
    calibration_phases = (
        replace(
            silicon,
            scale=_qarr_initial_scales(calibration_pattern, calibration_experiment, (silicon,))[0],
        ),
    )
    calibration_background = ChebyshevBackground(
        "silicon_calibration_residual",
        (0.0,),
        (float(calibration_pattern.x[0]), float(calibration_pattern.x[-1])),
    )
    calibration_phases, calibration_background = solve_linear_profile_block(
        calibration_pattern,
        calibration_experiment,
        calibration_phases,
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
    for _ in range(3):
        position_stage = refine_nonlinear_block(
            calibration_pattern,
            calibration_experiment,
            calibration_phases,
            calibration_background,
            position_selection,
            selected_execution,
            iterations=30,
            step=0.5,
        )
        calibration_experiment = position_stage.experiment
        calibration_phases, calibration_background = solve_linear_profile_block(
            calibration_pattern,
            calibration_experiment,
            position_stage.phases,
            calibration_background,
            selected_execution,
        )
    if not isinstance(calibration_experiment.geometry, BraggBrentanoGeometry):
        raise RuntimeError("silicon calibration lost its Bragg-Brentano geometry")
    calibration_calculation = rietveld.calculate(
        calibration_pattern,
        calibration_experiment,
        calibration_phases,
        background=calibration_background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    calibration_weight = 1.0 / np.maximum(calibration_pattern.observed_y, 1.0)
    calibration_residual = calibration_calculation.y - calibration_pattern.observed_y
    calibration_rwp = float(
        np.sqrt(
            np.sum(calibration_weight * np.square(calibration_residual))
            / np.sum(calibration_weight * np.square(calibration_pattern.observed_y))
        )
    )
    selected_phases = tuple(
        replace(phase, scale=scale)
        for phase, scale in zip(
            phases, _qarr_initial_scales(pattern, experiment, phases), strict=True
        )
    )
    background = ChebyshevBackground(
        "iucr_legacy_background_residual", (0.0,), (float(x[0]), float(x[-1]))
    )
    selected_phases, background = solve_linear_profile_block(
        pattern, experiment, selected_phases, background, selected_execution
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
    return IucrTripotassiumCitrateSiliconResult(
        sample_count=int(x.size),
        reflection_count=sum(phase.reflections.reflection_count for phase in selected_phases),
        free_parameter_count=3,
        weight_fractions=fractions,
        legacy_weight_fraction_errors=errors,
        maximum_legacy_weight_fraction_error=max(abs(value) for value in errors.values()),
        poisson_rwp=float(
            np.sqrt(np.sum(weight * np.square(residual)) / np.sum(weight * np.square(observed)))
        ),
        profile_correlation=float(
            np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1]
        ),
        legacy_curve_poisson_rwp=float(manifest["legacy_gsas_reference"]["rwp"]),
        legacy_curve_profile_correlation=float(
            np.corrcoef(observed - legacy_background, legacy_calculated - legacy_background)[0, 1]
        ),
        silicon_calibration_sample_count=int(np.count_nonzero(calibration_mask)),
        silicon_calibration_poisson_rwp=calibration_rwp,
        calibrated_sample_displacement_mm=calibration_experiment.geometry.sample_displacement_mm,
        refined_instrument={
            "U": 1.0e4 * experiment.instrument.u_deg2,
            "V": 1.0e4 * experiment.instrument.v_deg2,
            "W": 1.0e4 * experiment.instrument.w_deg2,
            "X": 1.0e2 * experiment.instrument.x_deg,
            "Y": 1.0e2 * experiment.instrument.y_deg,
            "Zero": experiment.zero_shift_deg,
        },
        model_qualifications=(
            "The reported texture index is 1.001, so no preferred-orientation term is fitted.",
            "All phase size/strain terms are fixed because the deposited main phase uses "
            "Stephens anisotropic broadening without an identifiable isotropic subset.",
            "The deposited Suortti surface-roughness correction is omitted.",
            "The phase-specific silicon profile is used only for displacement calibration; "
            "the dominant citrate's deposited isotropic base profile is fixed for the fit.",
            manifest["translation_diagnostics"]["geometry_assumption"],
            manifest["translation_diagnostics"]["axial_geometry_translation"],
            "The 1.44 wt% silicon displacement calibration is diagnostic only; it is "
            "not applied unless a cross-implementation identifiability gate passes.",
        ),
        elapsed_seconds=perf_counter() - started,
    )
