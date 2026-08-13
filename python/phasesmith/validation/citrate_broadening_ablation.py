"""Controlled isotropic-broadening ablations for the citrate/Si holdouts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..execution import ExecutionPolicy
from ..extensions import CompositePhysicsProvider
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..pattern import PowderPattern
from ..quantitative import QuantitativePhase, quantitative_phase_analysis
from ..radiation import BraggBrentanoGeometry, ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..refinement.runtime import RefinementLimits
from ..sample import (
    IsotropicLorentzianMicrostrainBroadening,
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
)
from ._rietveld_parity import solve_linear_profile_block
from .real_data import _qarr_displacement_defaults, _qarr_initial_scales

CitrateCase = Literal["tripotassium", "trirubidium"]

_CASE_CONFIGURATION = {
    "tripotassium": {
        "scope": "iucr_anhydrous_tripotassium_citrate_silicon_holdout",
        "phases": ("tripotassium_citrate", "silicon"),
        "dominant_phase": "tripotassium_citrate",
        "profile_key": "tripotassium_citrate_profile",
        "radius_key": "assumed_goniometer_radius_mm",
    },
    "trirubidium": {
        "scope": "iucr_anhydrous_trirubidium_citrate_silicon_holdout",
        "phases": ("trirubidium_citrate", "silicon"),
        "dominant_phase": "trirubidium_citrate",
        "profile_key": "common_base_profile",
        "radius_key": "goniometer_radius_mm",
    },
}


@dataclass(frozen=True, slots=True)
class CitrateBroadeningModelResult:
    """Best repeat for one nested isotropic sample-broadening model."""

    model: str
    free_parameter_count: int
    jacobian_rank: int | None
    poisson_rwp: float
    delta_poisson_rwp: float
    profile_correlation: float
    weight_fractions: dict[str, float]
    sample_parameters: dict[str, float]
    repeat_rwp_spread: float
    maximum_sample_linear_correlation: float | None
    identifiable: bool
    qualifications: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.poisson_rwp,
            self.delta_poisson_rwp,
            self.profile_correlation,
            self.repeat_rwp_spread,
            *self.weight_fractions.values(),
            *self.sample_parameters.values(),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("citrate broadening result must be finite")
        if self.maximum_sample_linear_correlation is not None and not (
            0.0 <= self.maximum_sample_linear_correlation <= 1.0
        ):
            raise ValueError("sample/linear correlation must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class CitrateBroadeningAblationResult:
    """Nested-model evidence for or against an isotropic width explanation."""

    case: CitrateCase
    scope: str
    dominant_phase: str
    sample_count: int
    model_results: tuple[CitrateBroadeningModelResult, ...]
    conclusion: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _physics(name: str, values: tuple[float, ...]) -> object | None:
    providers: list[object] = []
    value_index = 0
    if "size" in name:
        providers.append(IsotropicSizeBroadening(values[value_index]))
        value_index += 1
    if "gaussian" in name:
        providers.append(IsotropicMicrostrainBroadening(values[value_index]))
        value_index += 1
    if "lorentzian" in name:
        providers.append(IsotropicLorentzianMicrostrainBroadening(values[value_index]))
    if not providers:
        return None
    if len(providers) == 1:
        return providers[0]
    return CompositePhysicsProvider(tuple(providers))


def _physics_values(provider: object | None) -> dict[str, float]:
    if provider is None:
        return {}
    if type(provider) is IsotropicSizeBroadening:
        return {"crystallite_size_nm": float(provider.crystallite_size_nm)}
    if type(provider) is IsotropicMicrostrainBroadening:
        return {"gaussian_rms_microstrain": float(provider.rms_microstrain)}
    if type(provider) is IsotropicLorentzianMicrostrainBroadening:
        return {"lorentzian_microstrain": float(provider.microstrain)}
    if type(provider) is CompositePhysicsProvider:
        values: dict[str, float] = {}
        for child in provider.providers:
            values.update(_physics_values(child))
        return values
    raise TypeError("unexpected physics provider in citrate broadening ablation")


def _model_starts() -> dict[str, tuple[tuple[float, ...], ...]]:
    return {
        "base": ((),),
        "size": ((20.0,), (100.0,), (500.0,)),
        "gaussian": ((1.0e-4,), (5.0e-4,), (2.0e-3,)),
        "lorentzian": ((1.0e-4,), (5.0e-4,), (2.0e-3,)),
        "size_gaussian": ((20.0, 2.0e-3), (100.0, 5.0e-4), (500.0, 1.0e-4)),
        "size_lorentzian": ((20.0, 2.0e-3), (100.0, 5.0e-4), (500.0, 1.0e-4)),
    }


def _prepare(
    root: Path,
    case: CitrateCase,
) -> tuple[
    dict[str, Any],
    PowderPattern,
    ConstantWavelengthExperiment,
    tuple[rietveld.RietveldPhase, ...],
    ChebyshevBackground,
]:
    configuration = _CASE_CONFIGURATION[case]
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != configuration["scope"]:
        raise ValueError(f"bundle is not the configured {case} citrate holdout")
    data = np.loadtxt(root / "pattern.csv", delimiter=",", skiprows=1)
    expected_count = int(manifest["data_selection"]["refined_sample_count"])
    if data.shape != (expected_count, 4) or not np.all(np.isfinite(data)):
        raise ValueError("citrate pattern has an unexpected shape or non-finite values")
    x, observed, _legacy_calculated, legacy_background = data.T
    if np.any(np.diff(x) <= 0.0) or np.any(observed < 0.0):
        raise ValueError("citrate pattern axis or counts are invalid")
    pattern = PowderPattern(
        np.ascontiguousarray(x),
        observed_y=np.ascontiguousarray(observed),
        uncertainty=np.sqrt(np.maximum(observed, 1.0)),
        background=np.ascontiguousarray(legacy_background),
    )
    instrument_record = manifest["instrument"]
    wavelengths = instrument_record["wavelengths_angstrom"]
    profile = instrument_record[configuration["profile_key"]]
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
            WavelengthComponents.doublet(
                float(wavelengths[0]),
                float(wavelengths[1]),
                float(instrument_record["k_alpha2_over_k_alpha1"]),
            ),
            geometry=BraggBrentanoGeometry(
                float(instrument_record[configuration["radius_key"]]), 0.0
            ),
            axial_geometry=FcjGeometry(
                float(instrument_record["matched_sh_over_l"]) / 2.0,
                float(instrument_record["matched_sh_over_l"]) / 2.0,
            ),
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
    for phase_id in configuration["phases"]:
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
    initial = tuple(
        replace(phase, scale=scale)
        for phase, scale in zip(
            phases,
            _qarr_initial_scales(pattern, experiment, tuple(phases)),
            strict=True,
        )
    )
    background = ChebyshevBackground(
        "iucr_legacy_background_residual", (0.0,), (float(x[0]), float(x[-1]))
    )
    return manifest, pattern, experiment, initial, background


def _quantitative(phases: tuple[rietveld.RietveldPhase, ...]) -> dict[str, float]:
    inputs = tuple(
        QuantitativePhase(
            phase.phase_id,
            phase.scale,
            float(phase.structure.metadata["formula_units_per_cell"]),
            float(phase.structure.metadata["formula_mass_g_mol"]),
            phase.structure.cell.geometry().volume_angstrom3,
        )
        for phase in phases
    )
    return {item.phase_id: item.weight_fraction for item in quantitative_phase_analysis(inputs)}


def _sample_linear_correlation(result: rietveld.RietveldResult) -> float | None:
    values = [
        abs(item.correlation)
        for item in result.unresolved_correlations
        if {item.left.module, item.right.module} & {"sample"}
        and ({item.left.module, item.right.module} & {"phase", "background"})
    ]
    return max(values, default=None)


def run_citrate_isotropic_broadening_ablation(
    bundle_directory: str | Path,
    case: CitrateCase,
    *,
    execution: ExecutionPolicy | None = None,
) -> CitrateBroadeningAblationResult:
    """Fit nested phase-local isotropic width models from three deterministic starts.

    Instrument, cell, structure, zero, displacement, axial divergence, and all
    silicon sample terms remain fixed. Only the dominant citrate's stated
    isotropic model, both phase scales, and one residual-background constant
    move. The joint final fit supplies rank and weighted-column correlations.
    """

    if case not in _CASE_CONFIGURATION:
        raise ValueError("case must be 'tripotassium' or 'trirubidium'")
    root = Path(bundle_directory)
    manifest, pattern, experiment, base_phases, base_background = _prepare(root, case)
    selected_execution = ExecutionPolicy(threads=1) if execution is None else execution
    dominant_phase = str(_CASE_CONFIGURATION[case]["dominant_phase"])
    joint_selection = rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=True,
    )
    raw_results: dict[str, list[rietveld.RietveldResult | tuple[Any, ...]]] = {}
    for model, starts in _model_starts().items():
        repeats: list[rietveld.RietveldResult | tuple[Any, ...]] = []
        for start in starts:
            phases = tuple(
                replace(phase, physics=_physics(model, start))
                if phase.phase_id == dominant_phase
                else phase
                for phase in base_phases
            )
            phases, background = solve_linear_profile_block(
                pattern, experiment, phases, base_background, selected_execution
            )
            if model == "base":
                calculation = rietveld.calculate(
                    pattern,
                    experiment,
                    phases,
                    background=background,
                    support_fwhm=30.0,
                    execution=selected_execution,
                )
                weight = 1.0 / np.maximum(pattern.observed_y, 1.0)
                residual = calculation.y - pattern.observed_y
                repeats.append(
                    (
                        float(
                            np.sqrt(
                                np.sum(weight * np.square(residual))
                                / np.sum(weight * np.square(pattern.observed_y))
                            )
                        ),
                        calculation,
                        phases,
                    )
                )
                continue
            input_data = rietveld.RietveldInput(
                pattern,
                experiment,
                phases,
                (None,) * len(phases),
                rietveld.build_parameter_set(
                    phases,
                    (None,) * len(phases),
                    joint_selection,
                    experiment=experiment,
                    background=background,
                ),
                selection=joint_selection,
                background=background,
            )
            repeats.append(
                rietveld.refine(
                    input_data,
                    rietveld.RietveldOptions(
                        limits=RefinementLimits(max_iterations=14, max_evaluations=1_000),
                        min_iterations=2,
                        max_scaled_parameter_step=0.30,
                        support_fwhm=30.0,
                        estimate_covariance=True,
                        unresolved_correlation=0.0,
                        execution=selected_execution,
                    ),
                )
            )
        raw_results[model] = repeats

    base_repeat = raw_results["base"][0]
    if not isinstance(base_repeat, tuple):
        raise RuntimeError("internal base-model result mismatch")
    base_rwp = float(base_repeat[0])
    model_results = []
    for model, repeats in raw_results.items():
        repeat_rwps = [
            float(item[0]) if isinstance(item, tuple) else float(item.metrics.rwp)
            for item in repeats
        ]
        best_index = int(np.argmin(repeat_rwps))
        best = repeats[best_index]
        if isinstance(best, tuple):
            rwp, calculation, phases = best
            rank = None
            free_count = 3
            correlation = None
        else:
            rwp = float(best.metrics.rwp)
            calculation = best.calculation
            phases = best.phases
            rank = best.jacobian_rank
            free_count = len(best.parameters.specs)
            correlation = _sample_linear_correlation(best)
        dominant = next(phase for phase in phases if phase.phase_id == dominant_phase)
        parameters = _physics_values(dominant.physics)
        boundary = any(
            ("microstrain" in name and value <= 1.0e-10)
            or (name == "crystallite_size_nm" and value >= 1.0e5)
            for name, value in parameters.items()
        )
        spread = max(repeat_rwps) - min(repeat_rwps)
        identifiable = model == "base" or (
            rank == free_count
            and spread <= 5.0e-5
            and not boundary
            and correlation is not None
            and correlation < 0.98
        )
        qualifications = []
        if rank is not None and rank != free_count:
            qualifications.append("weighted Jacobian is rank deficient")
        if spread > 5.0e-5:
            qualifications.append("three deterministic starts do not reach the same Rwp basin")
        if boundary:
            qualifications.append("at least one width term is inactive at the zero-width limit")
        if correlation is not None and correlation >= 0.98:
            qualifications.append(
                "a width column is at least 0.98 correlated with scale/background"
            )
        if not qualifications:
            qualifications.append(
                "passes rank, repeatability, boundary, and scale/background gates"
            )
        model_results.append(
            CitrateBroadeningModelResult(
                model=model,
                free_parameter_count=free_count,
                jacobian_rank=rank,
                poisson_rwp=float(rwp),
                delta_poisson_rwp=float(rwp - base_rwp),
                profile_correlation=float(
                    np.corrcoef(
                        pattern.observed_y - calculation.background,
                        calculation.profile_y,
                    )[0, 1]
                ),
                weight_fractions=_quantitative(phases),
                sample_parameters=parameters,
                repeat_rwp_spread=float(spread),
                maximum_sample_linear_correlation=correlation,
                identifiable=identifiable,
                qualifications=tuple(qualifications),
            )
        )
    accepted = [
        result
        for result in model_results
        if result.model != "base" and result.identifiable and result.delta_poisson_rwp < 0.0
    ]
    best_accepted = min(accepted, key=lambda result: result.poisson_rwp, default=None)
    if best_accepted is None:
        conclusion = (
            "No current isotropic size/strain model produces a repeatable identifiable "
            "improvement over the fixed base profile."
        )
    else:
        conclusion = (
            f"{best_accepted.model} is the best identifiable isotropic model, reducing "
            f"Poisson Rwp by {-best_accepted.delta_poisson_rwp:.6f}; the remaining gap to "
            "the deposited curve is not explained by isotropic broadening alone."
        )
    return CitrateBroadeningAblationResult(
        case=case,
        scope=str(manifest["scope"]),
        dominant_phase=dominant_phase,
        sample_count=int(pattern.x.size),
        model_results=tuple(model_results),
        conclusion=conclusion,
    )
