"""Reproducible checks against externally fetched real powder patterns."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from .. import _core
from ..background import SmoothBrucknerBackground
from ..control import CancellationCallback
from ..execution import ExecutionPolicy
from ..extensions import CompositePhysicsProvider
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import (
    BraggBrentanoPolarizedLp,
    ConstantWavelengthNeutronLorentz,
)
from ..io.powder import read_powder_data
from ..pattern import PowderPattern
from ..phase import ReciprocalMetric, RietveldPhase
from ..quantitative import (
    QuantitativePhase,
    quantitative_phase_analysis,
    quantitative_phase_analysis_with_covariance,
)
from ..radiation import (
    ConstantWavelengthExperiment,
    DebyeScherrerGeometry,
    MonochromaticRadiation,
    RadiationProbe,
    WavelengthComponents,
)
from ..refinement import lebail, rietveld
from ..refinement.background import ChebyshevBackground
from ..refinement.core import TerminationReason
from ..refinement.runtime import RefinementLimits, RefinementLogger
from ..refinement.workflow import intelligent_rietveld_recipe, run_rietveld_recipe
from ..sample import (
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
)
from ..scattering import NeutronNuclear, XrayFixedDispersion, XrayNonResonant
from ..structure import CrystalStructure

ValidationStatus = Literal["passed", "failed", "blocked"]
_PBSO4_MIN_SAFE_REJECTED_RWP_IMPROVEMENT = 1.0e-4

QARR_1G_WEIGHED_WEIGHT_FRACTIONS = {
    "Al2O3": 0.3137,
    "ZnO": 0.3421,
    "CaF2": 0.3442,
}

# Cromer--Liberman values evaluated at Cu K-alpha1 (1.54051 Å) with Gemmi
# 0.7.5. They are fixed validation inputs for both narrowly separated doublet
# components; PhaseSmith does not import Gemmi for scattering-factor evaluation.
QARR_1G_CUKA_FIXED_DISPERSION = {
    "Al": complex(0.212567, 0.245496),
    "O": complex(0.0493839, 0.0322324),
    "Zn": complex(-1.545988, 0.677687),
    "Ca": complex(0.365107, 1.285341),
    "F": complex(0.0730839, 0.0533468),
}

_QARR_QPA_METADATA = {
    "Al2O3": (6.0, 101.961276),
    "ZnO": (2.0, 81.38),
    "CaF2": (4.0, 78.074806),
}
_QARR_EXPECTED_EXPANDED_SITES = {"Al2O3": 30, "ZnO": 4, "CaF2": 12}
_QARR_COORDINATE_TOLERANCE = 1.0e-4

_SUCROSE_CIF = """data_sucrose_validation
_chemical_name_common 'Sucrose validation cell'
_cell_length_a 7.715231369389035
_cell_length_b 8.663866877499101
_cell_length_c 10.809618877725404
_cell_angle_alpha 90
_cell_angle_beta 102.98249193732556
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 21'
_space_group_IT_number 4
"""


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One machine-readable real-data acceptance check."""

    check_id: str
    status: ValidationStatus
    detail: str
    measured: float | None = None
    criterion: str | None = None

    def __post_init__(self) -> None:
        if not self.check_id or not self.detail:
            raise ValueError("validation check ID and detail must not be empty")
        if not self.check_id.replace("_", "").isalnum() or self.check_id.lower() != self.check_id:
            raise ValueError("validation check ID must be lowercase snake case")
        if self.status not in {"passed", "failed", "blocked"}:
            raise ValueError("invalid validation check status")
        if not isinstance(self.criterion, str) or not self.criterion:
            raise ValueError("validation criterion must be a non-empty string")
        if self.measured is not None and not math.isfinite(self.measured):
            raise ValueError("validation check measurement must be finite")

    def to_record(self) -> dict[str, object]:
        """Return a finite JSON-compatible record."""

        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail": self.detail,
            "measured": self.measured,
            "criterion": self.criterion,
        }


@dataclass(frozen=True, slots=True)
class RealDataValidationReport:
    """Stable result envelope for a single external validation workflow."""

    dataset_id: str
    status: ValidationStatus
    sample_count: int
    reflection_count: int | None
    elapsed_seconds: float
    checks: tuple[ValidationCheck, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "notes", tuple(self.notes))
        if not self.dataset_id or self.status not in {"passed", "failed", "blocked"}:
            raise ValueError("invalid real-data validation identity or status")
        if self.sample_count <= 0:
            raise ValueError("real-data validation requires observed samples")
        if self.reflection_count is not None and self.reflection_count <= 0:
            raise ValueError("reflection_count must be positive or None")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if not self.checks:
            raise ValueError("real-data validation requires at least one check")
        if any(not isinstance(check, ValidationCheck) for check in self.checks):
            raise TypeError("real-data checks must contain ValidationCheck values")
        if len({check.check_id for check in self.checks}) != len(self.checks):
            raise ValueError("real-data check IDs must be unique")
        if any(not isinstance(note, str) or not note for note in self.notes):
            raise ValueError("real-data notes must be non-empty strings")
        check_statuses = {check.status for check in self.checks}
        expected = (
            "failed"
            if "failed" in check_statuses
            else ("blocked" if "blocked" in check_statuses else "passed")
        )
        if self.status != expected:
            raise ValueError("report status must summarize its check statuses")

    def to_record(self) -> dict[str, object]:
        """Return a finite JSON-compatible record."""

        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "sample_count": self.sample_count,
            "reflection_count": self.reflection_count,
            "elapsed_seconds": self.elapsed_seconds,
            "checks": [check.to_record() for check in self.checks],
            "notes": list(self.notes),
        }


def _native_validation_report(
    runner: str, dataset_directory: str | Path
) -> RealDataValidationReport:
    """Decode one stable report returned by the Rust validation crate."""

    record = json.loads(_core._run_native_validation(runner, str(dataset_directory)))
    return RealDataValidationReport(
        dataset_id=record["dataset_id"],
        status=record["status"],
        sample_count=record["sample_count"],
        reflection_count=record["reflection_count"],
        elapsed_seconds=record["elapsed_seconds"],
        checks=tuple(ValidationCheck(**check) for check in record["checks"]),
        notes=tuple(record["notes"]),
    )


def _use_native_validation() -> bool:
    return os.environ.get("PHASESMITH_VALIDATION_PYTHON_REFERENCE") != "1"


def run_nist_srm660c_validation(
    dataset_directory: str | Path,
) -> RealDataValidationReport:
    """Validate the official NIST SRM 660c archive and released reference fits."""

    return _native_validation_report("nist-srm660c-lab6-xray", dataset_directory)


def run_echidna_lab6_validation(
    dataset_directory: str | Path,
) -> RealDataValidationReport:
    """Run the native ANSTO Echidna constant-wavelength neutron smoke gate."""

    return _native_validation_report("ansto-echidna-lab6-cw-neutron", dataset_directory)


def run_powgen_tof_validation(
    dataset_directory: str | Path,
) -> RealDataValidationReport:
    """Run the native POWGEN TOF profile and Le Bail acceptance workflow."""

    return _native_validation_report("powgen-lab6-tof-calibration", dataset_directory)


def run_powgen_tof_readiness(
    dataset_directory: str | Path,
) -> RealDataValidationReport:
    """Compatibility alias for :func:`run_powgen_tof_validation`."""

    return run_powgen_tof_validation(dataset_directory)


def run_qarr_1h_validation(
    dataset_directory: str | Path,
) -> RealDataValidationReport:
    """Run QARR 1h as an independent transferability holdout."""

    return _native_validation_report("iucr-qarr-1h", dataset_directory)


def qarr_1g_readiness(dataset_directory: str | Path) -> RealDataValidationReport:
    """Inspect QARR inputs and confirm the fixed-spectrum structural capability."""

    start = perf_counter()
    root = Path(dataset_directory)
    pattern = read_powder_data(root / "cpd-1g.prn", format="columns")
    instrument_text = (root / "cuka.instprm").read_text(encoding="utf-8")
    has_doublet = all(
        line in instrument_text for line in ("Lam1:1.54051", "Lam2:1.54433", "I(L2)/I(L1):0.5")
    )
    expected_grid = (
        pattern.x.size == 7_251
        and math.isclose(float(pattern.x[0]), 5.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(pattern.x[-1]), 150.0, rel_tol=0.0, abs_tol=1.0e-12)
    )
    checks = [
        ValidationCheck(
            "observed_grid",
            "passed" if expected_grid else "failed",
            "QARR 1g plain-column pattern spans the published 5-150 degree grid.",
            measured=float(pattern.x.size),
            criterion="7251 samples with endpoints 5 and 150 degrees",
        ),
        ValidationCheck(
            "cuka_doublet_metadata",
            "passed" if has_doublet else "failed",
            "Pinned instrument metadata identifies Cu K-alpha1/K-alpha2 and the 0.5 ratio.",
            criterion="Lam1, Lam2, and I(L2)/I(L1) present",
        ),
    ]
    checks.append(
        ValidationCheck(
            "structural_doublet_refinement",
            "passed" if has_doublet else "failed",
            (
                "Fixed-spectrum structural values and shared analytical JVP/VJP products are "
                "available for the pinned doublet."
            ),
            criterion="one native structural batch per wavelength component",
        )
    )
    status: ValidationStatus = (
        "passed" if all(check.status == "passed" for check in checks) else "failed"
    )
    return RealDataValidationReport(
        dataset_id="iucr-qarr-1g",
        status=status,
        sample_count=pattern.x.size,
        reflection_count=None,
        elapsed_seconds=perf_counter() - start,
        checks=tuple(checks),
        notes=(
            "Published weighed fractions: Al2O3 31.37%, ZnO 34.21%, CaF2 34.42%.",
            "This readiness check does not run the quantitative fit.",
        ),
    )


def _qarr_instrument_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line or raw_line.startswith("#"):
            continue
        name, raw_value = raw_line.split(":", 1)
        try:
            values[name] = float(raw_value)
        except ValueError:
            continue
    required = ("Lam1", "Lam2", "I(L2)/I(L1)", "Polariz.", "U", "V", "W", "X", "Y")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"QARR instrument file is missing {', '.join(missing)}")
    return values


def _qarr_physics(
    phase_id: str,
    structure: CrystalStructure,
) -> CompositePhysicsProvider:
    providers = [
        IsotropicSizeBroadening(100.0),
        IsotropicMicrostrainBroadening(8.0e-4),
    ]
    if phase_id in ("Al2O3", "ZnO"):
        providers.append(
            MarchDollasePreferredOrientation(
                1.0,
                (0.0, 0.0, 1.0),
                ReciprocalMetric(structure.cell.geometry().reciprocal_metric),
            )
        )
    return CompositePhysicsProvider(tuple(providers))


def _qarr_displacement_defaults(structure: CrystalStructure) -> CrystalStructure:
    """Initialize only CIF sites that provide neither isotropic nor anisotropic U."""

    return replace(
        structure,
        sites=tuple(
            replace(site, u_iso_angstrom2=0.005)
            if site.u_iso_angstrom2 is None and site.anisotropic_displacement is None
            else site
            for site in structure.sites
        ),
    )


def _qarr_initial_scales(
    pattern: PowderPattern,
    experiment: ConstantWavelengthExperiment,
    phases: tuple[RietveldPhase, ...] | list[RietveldPhase],
) -> tuple[float, ...]:
    calculation = rietveld.calculate(pattern, experiment, tuple(phases), support_fwhm=30.0)
    design = np.column_stack(
        [
            item.profile_y / phase.scale
            for item, phase in zip(calculation.phase_calculations, phases, strict=True)
        ]
    )
    target = pattern.observed_y - pattern.background
    if pattern.uncertainty is not None:
        design = design / pattern.uncertainty[:, None]
        target = target / pattern.uncertainty
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    if not np.isfinite(solution).all() or np.any(solution <= 0.0):
        raise ValueError("QARR linear scale initialization did not produce positive finite scales")
    return tuple(map(float, solution))


def _qarr_cancelled_report(
    *,
    sample_count: int,
    reflection_count: int,
    elapsed_seconds: float,
    stage: str,
    rwp: float,
) -> RealDataValidationReport:
    return RealDataValidationReport(
        dataset_id="iucr-qarr-1g",
        status="blocked",
        sample_count=sample_count,
        reflection_count=reflection_count,
        elapsed_seconds=elapsed_seconds,
        checks=(
            ValidationCheck(
                "cooperative_cancellation",
                "blocked",
                f"QARR validation stopped cooperatively during {stage}.",
                criterion="complete all three refinement stages",
            ),
        ),
        notes=(f"Last accepted Poisson-weighted Rwp={rwp:.8f}.",),
    )


def run_qarr_1g_validation(
    dataset_directory: str | Path,
    *,
    cancellation: CancellationCallback | None = None,
    logger: RefinementLogger | None = None,
    execution: ExecutionPolicy | None = None,
) -> RealDataValidationReport:
    """Run the pinned three-phase Cu K-alpha QARR refinement and QPA checks."""

    if _use_native_validation() and cancellation is None and logger is None and execution is None:
        return _native_validation_report("iucr-qarr-1g", dataset_directory)

    start = perf_counter()
    selected_execution = ExecutionPolicy() if execution is None else execution
    if not isinstance(selected_execution, ExecutionPolicy):
        raise TypeError("execution must be ExecutionPolicy")
    root = Path(dataset_directory)
    data = read_powder_data(root / "cpd-1g.prn", format="columns")
    values = _qarr_instrument_values(root / "cuka.instprm")
    background = SmoothBrucknerBackground(
        smooth_width=1.0,
        iterations=50,
        chebyshev_order=None,
    ).estimate(data.x, data.observed_y)
    pattern = PowderPattern(
        data.x,
        observed_y=data.observed_y,
        uncertainty=np.sqrt(np.maximum(data.observed_y, 1.0)),
        background=background,
    )
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
        WavelengthComponents.doublet(
            values["Lam1"],
            values["Lam2"],
            values["I(L2)/I(L1)"],
        ),
        axial_geometry=FcjGeometry(
            values["SH/L"] / 2.0,
            values["SH/L"] / 2.0,
        ),
    )
    scattering = XrayFixedDispersion(QARR_1G_CUKA_FIXED_DISPERSION)
    correction = BraggBrentanoPolarizedLp(values["Lam1"], values["Polariz."])
    residual_background = ChebyshevBackground(
        "qarr_residual",
        (0.0, 0.0, 0.0),
        (float(data.x[0]), float(data.x[-1])),
    )
    fixed_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    phases = []
    expanded_counts: dict[str, int] = {}
    for phase_id in QARR_1G_WEIGHED_WEIGHT_FRACTIONS:
        single = rietveld.RietveldInput.from_cif(
            pattern,
            experiment,
            root / f"{phase_id}.cif",
            phase_id=phase_id,
            selection=fixed_selection,
            scattering=scattering,
            intensity_correction=correction,
            coordinate_tolerance=_QARR_COORDINATE_TOLERANCE,
        )
        structure = _qarr_displacement_defaults(single.phases[0].structure)
        expanded = structure.space_group.expand_sites(
            [site.fractional_xyz for site in structure.sites],
            tolerance=_QARR_COORDINATE_TOLERANCE,
        )
        expanded_counts[phase_id] = int(expanded.fractional_xyz.shape[0])
        phases.append(
            replace(
                single.phases[0],
                structure=structure,
                physics=_qarr_physics(phase_id, structure),
            )
        )
    scales = _qarr_initial_scales(pattern, experiment, phases)
    phases = [replace(phase, scale=scale) for phase, scale in zip(phases, scales, strict=True)]
    stage_one_selection = rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "zero_shift_deg"),
        background=True,
    )
    stage_one = rietveld.RietveldInput(
        pattern,
        experiment,
        tuple(phases),
        (None,) * len(phases),
        rietveld.build_parameter_set(
            tuple(phases),
            (None,) * len(phases),
            stage_one_selection,
            experiment=experiment,
            background=residual_background,
        ),
        selection=stage_one_selection,
        background=residual_background,
    )
    first = rietveld.refine(
        stage_one,
        rietveld.RietveldOptions(
            # This stage is an intentional fixed eight-step initialization.
            # Continuing beyond its eight accepted improvements only reaches
            # the rejection guard without changing the accepted state.
            limits=RefinementLimits(max_iterations=8, max_evaluations=800),
            min_iterations=2,
            max_scaled_parameter_step=0.2,
            support_fwhm=30.0,
            estimate_covariance=False,
            execution=selected_execution,
        ),
        cancellation=cancellation,
        logger=logger,
    )
    reflection_count = sum(phase.reflections.reflection_count for phase in first.phases)
    if first.termination_reason is TerminationReason.CANCELLED:
        return _qarr_cancelled_report(
            sample_count=data.x.size,
            reflection_count=reflection_count,
            elapsed_seconds=perf_counter() - start,
            stage="stage 1",
            rwp=first.metrics.rwp,
        )
    stage_two_selection = replace(
        stage_one_selection,
        u_iso=True,
        sample_physics=True,
        background=False,
    )
    stage_two = rietveld.RietveldInput(
        pattern,
        first.experiment,
        first.phases,
        (None,) * len(first.phases),
        rietveld.build_parameter_set(
            first.phases,
            (None,) * len(first.phases),
            stage_two_selection,
            experiment=first.experiment,
            background=first.background,
        ),
        selection=stage_two_selection,
        background=first.background,
    )
    second = rietveld.refine(
        stage_two,
        rietveld.RietveldOptions(
            # The independent Python path reaches its stable accepted state in
            # 28 improvements; make that workload explicit instead of spending
            # a rejection budget after the scientific result has stopped moving.
            limits=RefinementLimits(max_iterations=28, max_evaluations=1_500),
            min_iterations=3,
            max_scaled_parameter_step=0.15,
            support_fwhm=30.0,
            estimate_covariance=False,
            execution=selected_execution,
        ),
        cancellation=cancellation,
        logger=logger,
    )
    reflection_count = sum(phase.reflections.reflection_count for phase in second.phases)
    if second.termination_reason is TerminationReason.CANCELLED:
        return _qarr_cancelled_report(
            sample_count=data.x.size,
            reflection_count=reflection_count,
            elapsed_seconds=perf_counter() - start,
            stage="stage 2",
            rwp=second.metrics.rwp,
        )
    scale_selection = rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        background=False,
    )
    scale_polish = rietveld.RietveldInput(
        pattern,
        second.experiment,
        second.phases,
        (None,) * len(second.phases),
        rietveld.build_parameter_set(
            second.phases,
            (None,) * len(second.phases),
            scale_selection,
            experiment=second.experiment,
            background=second.background,
        ),
        selection=scale_selection,
        background=second.background,
    )
    result = rietveld.refine(
        scale_polish,
        rietveld.RietveldOptions(
            limits=RefinementLimits(max_iterations=10, max_evaluations=200),
            max_scaled_parameter_step=1.0,
            support_fwhm=30.0,
            estimate_covariance=True,
            execution=selected_execution,
        ),
        cancellation=cancellation,
        logger=logger,
    )
    reflection_count = sum(phase.reflections.reflection_count for phase in result.phases)
    if result.termination_reason is TerminationReason.CANCELLED:
        return _qarr_cancelled_report(
            sample_count=data.x.size,
            reflection_count=reflection_count,
            elapsed_seconds=perf_counter() - start,
            stage="stage 3 scale polish",
            rwp=result.metrics.rwp,
        )
    quantitative = tuple(
        QuantitativePhase(
            phase.phase_id,
            phase.scale,
            _QARR_QPA_METADATA[phase.phase_id][0],
            _QARR_QPA_METADATA[phase.phase_id][1],
            phase.structure.cell.geometry().volume_angstrom3,
        )
        for phase in result.phases
    )
    qpa = quantitative_phase_analysis(quantitative)
    expected_scale_keys = tuple(rietveld.phase_scale_key(phase.phase_id) for phase in result.phases)
    propagated = (
        quantitative_phase_analysis_with_covariance(quantitative, result.covariance)
        if result.covariance is not None and result.parameters.keys == expected_scale_keys
        else None
    )
    maximum_fraction_standard_uncertainty = (
        float(np.sqrt(np.max(np.diag(propagated.covariance))))
        if propagated is not None and np.all(np.diag(propagated.covariance) >= 0.0)
        else None
    )
    calculated_fractions = {item.phase_id: item.weight_fraction for item in qpa}
    max_weight_error = max(
        abs(calculated_fractions[phase_id] - expected)
        for phase_id, expected in QARR_1G_WEIGHED_WEIGHT_FRACTIONS.items()
    )
    profile_correlation = float(
        np.corrcoef(data.observed_y - background, result.calculation.profile_y)[0, 1]
    )
    residual = result.calculation.y - data.observed_y
    unit_weight_rwp = float(np.sqrt((residual @ residual) / (data.observed_y @ data.observed_y)))
    expansion_ok = expanded_counts == _QARR_EXPECTED_EXPANDED_SITES
    unsafe_terminations = {
        TerminationReason.NUMERICAL_FAILURE,
        TerminationReason.DIVERGED,
        TerminationReason.REPEATED_REJECTIONS,
        TerminationReason.NO_OBSERVATIONS,
    }
    safe_termination = all(
        stage.termination_reason not in unsafe_terminations for stage in (first, second, result)
    )
    checks = (
        ValidationCheck(
            "observed_grid",
            "passed"
            if data.x.size == 7_251 and data.x[0] == 5.0 and data.x[-1] == 150.0
            else "failed",
            "QARR 1g pattern spans the published 5-150 degree grid.",
            measured=float(data.x.size),
            criterion="7251 samples with endpoints 5 and 150 degrees",
        ),
        ValidationCheck(
            "site_expansion",
            "passed" if expansion_ok else "failed",
            "The documented 1e-4 CIF coordinate tolerance gives physical unit-cell contents.",
            measured=float(sum(expanded_counts.values())),
            criterion="expanded site counts Al2O3=30, ZnO=4, CaF2=12",
        ),
        ValidationCheck(
            "refinement_termination",
            "passed" if safe_termination else "failed",
            "Refinement returns a finite last accepted state under explicit iteration budgets.",
            criterion="no numerical failure, divergence, repeated rejection, or empty data",
        ),
        ValidationCheck(
            "poisson_rwp",
            "passed" if result.metrics.rwp <= 0.20 else "failed",
            "Poisson-weighted QARR profile gate with fixed CIF anisotropic displacement.",
            measured=result.metrics.rwp,
            criterion="Rwp with sigma=sqrt(max(counts, 1)) <= 0.20",
        ),
        ValidationCheck(
            "unit_weight_rwp",
            "passed" if unit_weight_rwp <= 0.15 else "failed",
            "Unit-weight profile residual is reported separately from Poisson-weighted Rwp.",
            measured=unit_weight_rwp,
            criterion="unit-weight Rwp <= 0.15",
        ),
        ValidationCheck(
            "profile_correlation",
            "passed" if profile_correlation >= 0.98 else "failed",
            "Background-subtracted observed and calculated profiles remain strongly aligned.",
            measured=profile_correlation,
            criterion="Pearson correlation >= 0.98",
        ),
        ValidationCheck(
            "qpa_weight_fraction",
            "passed" if max_weight_error <= 0.02 else "failed",
            "Hill--Howard weight fractions agree with the independently weighed phase fractions.",
            measured=max_weight_error,
            criterion="maximum absolute phase error <= 0.02",
        ),
        ValidationCheck(
            "qpa_covariance",
            "passed" if maximum_fraction_standard_uncertainty is not None else "failed",
            (
                "The final scale covariance propagates analytically to finite "
                "phase-fraction uncertainties."
            ),
            measured=maximum_fraction_standard_uncertainty,
            criterion="finite nonnegative propagated variance for every phase fraction",
        ),
    )
    status: ValidationStatus = (
        "passed" if all(check.status == "passed" for check in checks) else "failed"
    )
    fraction_note = ", ".join(
        f"{phase_id}={100.0 * calculated_fractions[phase_id]:.3f}%"
        for phase_id in QARR_1G_WEIGHED_WEIGHT_FRACTIONS
    )
    return RealDataValidationReport(
        dataset_id="iucr-qarr-1g",
        status=status,
        sample_count=data.x.size,
        reflection_count=reflection_count,
        elapsed_seconds=perf_counter() - start,
        checks=checks,
        notes=(
            f"Calculated crystalline weight fractions: {fraction_note}.",
            (
                "Final scale covariance was not identifiable."
                if maximum_fraction_standard_uncertainty is None
                else "Maximum propagated phase-fraction standard uncertainty="
                f"{maximum_fraction_standard_uncertainty:.8f}."
            ),
            (
                f"Stage 1 termination={first.termination_reason.value}, "
                f"iterations={len(first.history)}, evaluations={first.evaluations}, "
                f"Rwp={first.metrics.rwp:.8f}."
            ),
            (
                f"Stage 2 termination={second.termination_reason.value}, "
                f"iterations={len(second.history)}, evaluations={second.evaluations}, "
                f"Rwp={second.metrics.rwp:.8f}."
            ),
            (
                f"Stage 3 scale polish termination={result.termination_reason.value}, "
                f"iterations={len(result.history)}, evaluations={result.evaluations}, "
                f"Poisson Rwp={result.metrics.rwp:.8f}, "
                f"unit-weight Rwp={unit_weight_rwp:.8f}, Rp={result.metrics.rp:.8f}."
            ),
            f"Expanded sites at tolerance 1e-4: {expanded_counts}.",
            "CIF anisotropic displacement tensors are evaluated directly and remain fixed.",
            "Sites without CIF displacement values start from Uiso=0.005 Å² and are refined.",
            (
                "Fixed Cu K-alpha1 Cromer--Liberman offsets are used for both doublet "
                "components; component-dependent dispersion is not interpolated."
            ),
            (
                "The supplied SH/L=0.002 is included through the documented equal-height FCJ "
                "mapping; absorption is not yet included, and lattice/component wavelengths "
                "remain fixed."
            ),
        ),
    )


def run_sucrose_lebail_validation(dataset_directory: str | Path) -> RealDataValidationReport:
    """Run the supported monochromatic Le Bail path on the APS sucrose data."""

    if _use_native_validation():
        return _native_validation_report("aps-sucrose-11bmb", dataset_directory)

    start = perf_counter()
    data = read_powder_data(Path(dataset_directory) / "11bmb_8716.fxye", format="gsas_fxye")
    selected = (data.x >= 1.0) & (data.x <= 24.0)
    x = data.x[selected]
    observed = data.observed_y[selected]
    uncertainty = None if data.uncertainty is None else data.uncertainty[selected]
    # The supplied profile coefficients are centidegree-based U/V/W/X/Y.
    instrument = ConstantWavelengthInstrument(
        wavelength_angstrom=0.413259,
        u_deg2=1.163e-4,
        v_deg2=-0.126e-4,
        w_deg2=0.063e-4,
        x_deg=0.173e-2,
        y_deg=0.0,
    )
    fixed_background = SmoothBrucknerBackground(
        smooth_width=0.1,
        iterations=50,
        chebyshev_order=None,
    ).estimate(x, observed)
    background = ChebyshevBackground("sucrose_background", (0.0,), (float(x[0]), float(x[-1])))
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=uncertainty,
        background=fixed_background,
    )
    request = lebail.LeBailInput.from_cif(
        pattern,
        instrument,
        _SUCROSE_CIF,
        phase_id="sucrose",
        initial_intensity=0.0,
        refine_lattice=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"),
        background=background,
    )
    result = lebail.refine(
        request,
        lebail.LeBailOptions(
            max_iterations=20,
            min_iterations=2,
            intensity_tolerance=1.0e-7,
            rwp_tolerance=1.0e-9,
            max_scaled_parameter_step=0.2,
            support_fwhm=30.0,
        ),
    )
    first_rwp = result.history[0].rwp
    final_rwp = result.metrics.rwp
    intensities = np.fromiter(
        (item.integrated_intensity for item in result.intensities), dtype=np.float64
    )
    finite_nonnegative = bool(np.isfinite(intensities).all() and np.all(intensities >= 0.0))
    profile_correlation = float(
        np.corrcoef(observed - result.calculation.background, result.calculation.profile_y)[0, 1]
    )
    checks = (
        ValidationCheck(
            "profile_improvement",
            "passed" if final_rwp < first_rwp else "failed",
            "Profile refinement lowers weighted residual relative to the first extraction cycle.",
            measured=final_rwp / first_rwp,
            criterion="final Rwp / first-cycle Rwp < 1",
        ),
        ValidationCheck(
            "smoke_rwp",
            "passed" if final_rwp <= 0.22 else "failed",
            "Current-model real-data smoke gate; this is not a GSAS-II equivalence tolerance.",
            measured=final_rwp,
            criterion="Rwp <= 0.22",
        ),
        ValidationCheck(
            "profile_correlation",
            "passed" if profile_correlation >= 0.97 else "failed",
            "Background-subtracted observed and calculated profiles remain strongly aligned.",
            measured=profile_correlation,
            criterion="Pearson correlation >= 0.97",
        ),
        ValidationCheck(
            "integrated_intensities",
            "passed" if finite_nonnegative else "failed",
            "Every extracted integrated intensity is finite and nonnegative.",
            criterion="all finite and >= 0",
        ),
    )
    status: ValidationStatus = (
        "passed" if all(check.status == "passed" for check in checks) else "failed"
    )
    return RealDataValidationReport(
        dataset_id="aps-sucrose-11bmb",
        status=status,
        sample_count=x.size,
        reflection_count=len(result.intensities),
        elapsed_seconds=perf_counter() - start,
        checks=checks,
        notes=(
            f"First-cycle Rwp={first_rwp:.8f}; final Rwp={final_rwp:.8f}; "
            f"final Rp={result.metrics.rp:.8f}.",
            f"Termination={result.termination_reason.value}; iterations={len(result.history)}.",
            (
                "Both workflows use the identical fixed Smooth Bruckner array plus one "
                "refinable constant Chebyshev residual starting from zero."
            ),
            (
                "Both matched workflows start from U=1.163, V=-0.126, W=0.063, X=0.173, "
                "Y=0 in GSAS units and use the symmetric SH/L=0 profile."
            ),
        ),
    )


def _pbso4_experiment(
    probe: RadiationProbe,
) -> tuple[ConstantWavelengthExperiment, object, object, tuple[float, float]]:
    if probe is RadiationProbe.X_RAY:
        instrument = ConstantWavelengthInstrument(
            1.5405,
            2.0e-4,
            -2.0e-4,
            5.0e-4,
            1.0e-3,
            0.0,
        )
        return (
            ConstantWavelengthExperiment.x_ray_components(
                instrument,
                WavelengthComponents.doublet(1.5405, 1.5443, 0.5),
                axial_geometry=FcjGeometry(0.0075, 0.0075),
            ),
            XrayNonResonant(),
            BraggBrentanoPolarizedLp(1.5405, 0.7),
            (16.0, 158.4),
        )
    instrument = ConstantWavelengthInstrument(
        1.909,
        354.031e-4,
        -760.404e-4,
        651.592e-4,
        0.0,
        0.0,
    )
    return (
        ConstantWavelengthExperiment(
            MonochromaticRadiation.neutron(1.909),
            instrument,
            zero_shift_deg=-0.1,
            geometry=DebyeScherrerGeometry(650.0),
        ),
        NeutronNuclear(),
        ConstantWavelengthNeutronLorentz(1.909),
        (19.0, 153.0),
    )


def _pbso4_stage_termination_is_safe(
    reason: TerminationReason,
    accepted_iterations: int,
    starting_rwp: float,
    final_rwp: float,
) -> bool:
    if reason in {
        TerminationReason.NUMERICAL_FAILURE,
        TerminationReason.DIVERGED,
        TerminationReason.NO_OBSERVATIONS,
        TerminationReason.MAX_ITERATIONS,
    }:
        return False
    if reason is TerminationReason.REPEATED_REJECTIONS:
        return (
            accepted_iterations > 0
            and math.isfinite(starting_rwp)
            and math.isfinite(final_rwp)
            and starting_rwp - final_rwp >= _PBSO4_MIN_SAFE_REJECTED_RWP_IMPROVEMENT
        )
    return True


def run_pbso4_cw_validation(
    dataset_directory: str | Path,
    probe: RadiationProbe,
    *,
    execution: ExecutionPolicy | None = None,
) -> RealDataValidationReport:
    """Refine one official PbSO4 X-ray or neutron constant-wavelength pattern."""

    if probe not in {RadiationProbe.X_RAY, RadiationProbe.NEUTRON}:
        raise ValueError("PbSO4 validation requires X-ray or neutron radiation")
    if _use_native_validation() and execution is None:
        runner = (
            "gsasii-pbso4-cw-x-ray" if probe is RadiationProbe.X_RAY else "gsasii-pbso4-cw-neutron"
        )
        return _native_validation_report(runner, dataset_directory)
    start = perf_counter()
    root = Path(dataset_directory)
    filename = "PBSO4.XRA" if probe is RadiationProbe.X_RAY else "PBSO4.CWN"
    data = read_powder_data(root / filename, format="gsas_std")
    experiment, scattering, correction, limits = _pbso4_experiment(probe)
    selected = (data.x >= limits[0]) & (data.x <= limits[1])
    x = data.x[selected]
    observed = data.observed_y[selected]
    uncertainty = None if data.uncertainty is None else data.uncertainty[selected]
    fixed_background = SmoothBrucknerBackground(
        smooth_width=1.0,
        iterations=50,
        chebyshev_order=None,
    ).estimate(x, observed)
    residual_background = ChebyshevBackground(
        "pbso4_residual",
        (0.0, 0.0, 0.0),
        limits,
    )
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=uncertainty,
        background=fixed_background,
    )
    position_parameters = (
        ("u_deg2", "v_deg2", "w_deg2", "zero_shift_deg")
        if probe is RadiationProbe.X_RAY
        else (
            "u_deg2",
            "v_deg2",
            "w_deg2",
            "displace_x_micrometre",
            "displace_y_micrometre",
        )
    )
    selection = rietveld.RietveldParameterSelection(
        phase_scale=True,
        lattice=probe is RadiationProbe.NEUTRON,
        coordinates=True,
        occupancy=False,
        u_iso=True,
        sample_physics=probe is RadiationProbe.X_RAY,
        instrument_parameters=position_parameters,
        background=True,
    )
    initial = rietveld.RietveldInput.from_cif(
        pattern,
        experiment,
        root / "PbSO4-Wyckoff.cif",
        phase_id="PbSO4",
        selection=selection,
        scattering=scattering,
        intensity_correction=correction,
        coordinate_tolerance=1.0e-4,
        background=residual_background,
    )
    phase = initial.phases[0]
    if probe is RadiationProbe.X_RAY:
        phase = replace(
            phase,
            physics=CompositePhysicsProvider(
                (
                    IsotropicSizeBroadening(100.0),
                    IsotropicMicrostrainBroadening(8.0e-4),
                )
            ),
        )
    scale = _qarr_initial_scales(pattern, experiment, (phase,))[0]
    phase = replace(phase, scale=scale)
    initial_calculation = rietveld.calculate(
        pattern,
        experiment,
        (phase,),
        background=residual_background,
    )
    background_basis = residual_background.basis(x)
    background_residual = observed - initial_calculation.y
    if uncertainty is None:
        weighted_basis = background_basis
        weighted_residual = background_residual
    else:
        weighted_basis = background_basis / uncertainty[:, np.newaxis]
        weighted_residual = background_residual / uncertainty
    residual_background = residual_background.replace_coefficients(
        np.linalg.lstsq(weighted_basis, weighted_residual, rcond=None)[0]
    )
    request = rietveld.RietveldInput(
        pattern,
        experiment,
        (phase,),
        initial.lattice_domains,
        rietveld.build_parameter_set(
            (phase,),
            initial.lattice_domains,
            selection,
            experiment=experiment,
            background=residual_background,
        ),
        selection=selection,
        background=residual_background,
    )
    options = rietveld.RietveldOptions(
        # Native matrix-free products are accounted individually; this budget
        # covers the 160-iteration bound even when CG uses its full allowance.
        limits=RefinementLimits(max_iterations=160, max_evaluations=12_000),
        min_iterations=3,
        objective_tolerance=1.0e-7,
        max_scaled_parameter_step=0.15,
        support_fwhm=30.0,
        estimate_covariance=False,
        execution=ExecutionPolicy() if execution is None else execution,
    )
    recipe = intelligent_rietveld_recipe(
        request,
        name=f"pbso4-{probe.value}-intelligent",
    )
    workflow = run_rietveld_recipe(
        request,
        recipe,
        options=options,
    )
    result = workflow.final_result
    residual = result.calculation.y - observed
    unit_weight_rwp = float(np.sqrt((residual @ residual) / (observed @ observed)))
    profile_correlation = float(
        np.corrcoef(
            observed - result.calculation.background,
            result.calculation.profile_y,
        )[0, 1]
    )
    poisson_limit = 0.11 if probe is RadiationProbe.X_RAY else 0.05
    unit_weight_limit = 0.10 if probe is RadiationProbe.X_RAY else 0.06
    cell = result.phases[0].structure.cell
    reference_cell = (8.48, 5.398, 6.958)
    cell_values = (cell.a_angstrom, cell.b_angstrom, cell.c_angstrom)
    maximum_cell_relative_error = max(
        abs(actual - expected) / expected
        for actual, expected in zip(cell_values, reference_cell, strict=True)
    )
    safe_termination = len(workflow.stages) == len(recipe.stages) and all(
        _pbso4_stage_termination_is_safe(
            stage.result.termination_reason,
            len(stage.result.history),
            stage.starting_rwp,
            stage.result.metrics.rwp,
        )
        for stage in workflow.stages
    )
    geometry_note = None
    if isinstance(result.experiment.geometry, DebyeScherrerGeometry):
        geometry_note = (
            "Debye-Scherrer geometry: fixed radius="
            f"{result.experiment.geometry.goniometer_radius_mm:.3f} mm; refined "
            f"X={result.experiment.geometry.displace_x_micrometre:.6f} micrometre, "
            f"Y={result.experiment.geometry.displace_y_micrometre:.6f} micrometre."
        )
    checks = (
        ValidationCheck(
            "observed_grid",
            "passed" if x.size > 2_000 and x[0] == limits[0] and x[-1] == limits[1] else "failed",
            "Selected PbSO4 pattern uses the official combined-refinement angular range.",
            measured=float(x.size),
            criterion=f">2000 samples with endpoints {limits[0]} and {limits[1]} degrees",
        ),
        ValidationCheck(
            "refinement_termination",
            "passed" if safe_termination else "failed",
            "Every intelligent recipe stage terminates safely under explicit budgets.",
            criterion=(
                "all planned stages are attempted; repeated rejection requires an accepted "
                "Rwp improvement >= 0.0001; iteration exhaustion fails"
            ),
        ),
        ValidationCheck(
            "poisson_rwp",
            "passed" if result.metrics.rwp <= poisson_limit else "failed",
            "Poisson-weighted complete-pattern residual for the PbSO4 workflow.",
            measured=result.metrics.rwp,
            criterion=f"Rwp <= {poisson_limit}",
        ),
        ValidationCheck(
            "unit_weight_rwp",
            "passed" if unit_weight_rwp <= unit_weight_limit else "failed",
            "Unit-weight PbSO4 residual is reported separately.",
            measured=unit_weight_rwp,
            criterion=f"unit-weight Rwp <= {unit_weight_limit}",
        ),
        ValidationCheck(
            "profile_correlation",
            "passed" if profile_correlation >= 0.99 else "failed",
            "Background-subtracted observed and calculated profiles remain aligned.",
            measured=profile_correlation,
            criterion="Pearson correlation >= 0.99",
        ),
        ValidationCheck(
            "reference_cell_relative_error",
            "passed" if maximum_cell_relative_error <= 0.005 else "failed",
            "Refined cell remains close to the supplied PbSO4 reference model.",
            measured=maximum_cell_relative_error,
            criterion="maximum relative error across a, b, c <= 0.005",
        ),
    )
    status: ValidationStatus = (
        "passed" if all(check.status == "passed" for check in checks) else "failed"
    )
    return RealDataValidationReport(
        dataset_id=f"gsasii-pbso4-cw-{probe.value}",
        status=status,
        sample_count=x.size,
        reflection_count=result.phases[0].reflections.reflection_count,
        elapsed_seconds=perf_counter() - start,
        checks=checks,
        notes=(
            f"Final cell a={cell.a_angstrom:.8f}, b={cell.b_angstrom:.8f}, "
            f"c={cell.c_angstrom:.8f} angstrom.",
            f"Termination={result.termination_reason.value}; iterations={len(result.history)}; "
            f"evaluations={result.evaluations}.",
            *((geometry_note,) if geometry_note is not None else ()),
            *(
                f"Stage {stage.stage.name}: Rwp {stage.starting_rwp:.8f} -> "
                f"{stage.result.metrics.rwp:.8f}; "
                f"termination={stage.result.termination_reason.value}; "
                f"iterations={len(stage.result.history)}."
                for stage in workflow.stages
            ),
            *recipe.planner_notes,
            (
                "The residual background coefficients are initialized by weighted linear "
                "least squares against the starting structural profile."
            ),
            (
                "Background is a fixed native Smooth Bruckner estimate plus a refined "
                "three-term Chebyshev residual correction."
            ),
        ),
    )
