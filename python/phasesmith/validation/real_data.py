"""Reproducible checks against externally fetched real powder patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from ..background import SmoothBrucknerBackground
from ..instrument import ConstantWavelengthInstrument
from ..io.powder import read_powder_data
from ..pattern import PowderPattern
from ..refinement import lebail

ValidationStatus = Literal["passed", "failed", "blocked"]

QARR_1G_WEIGHED_WEIGHT_FRACTIONS = {
    "Al2O3": 0.3137,
    "ZnO": 0.3421,
    "CaF2": 0.3442,
}

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
        if self.status not in {"passed", "failed", "blocked"}:
            raise ValueError("invalid validation check status")
        if self.criterion is not None and (
            not isinstance(self.criterion, str) or not self.criterion
        ):
            raise ValueError("validation criterion must be a non-empty string or None")
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


def qarr_1g_readiness(dataset_directory: str | Path) -> RealDataValidationReport:
    """Inspect the real QARR 1g inputs and expose the current capability gate."""

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
    if has_doublet:
        checks.append(
            ValidationCheck(
                "structural_doublet_refinement",
                "blocked",
                (
                    "The profile kernel supports wavelength components, but the current full "
                    "structure-factor Rietveld request accepts one monochromatic wavelength."
                ),
                criterion="one structural intensity calculation per wavelength component",
            )
        )
    status: ValidationStatus = (
        "failed" if any(check.status == "failed" for check in checks) else "blocked"
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
            "No fit or quantitative accuracy claim is made while the capability check is blocked.",
        ),
    )


def run_sucrose_lebail_validation(dataset_directory: str | Path) -> RealDataValidationReport:
    """Run the supported monochromatic Le Bail path on the APS sucrose data."""

    start = perf_counter()
    data = read_powder_data(Path(dataset_directory) / "11bmb_8716.fxye", format="gsas_fxye")
    selected = (data.x >= 1.0) & (data.x <= 24.0)
    x = data.x[selected]
    observed = data.observed_y[selected]
    uncertainty = None if data.uncertainty is None else data.uncertainty[selected]
    background = SmoothBrucknerBackground(
        smooth_width=0.1,
        iterations=50,
        chebyshev_order=None,
    ).estimate(x, observed)
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=uncertainty,
        background=background,
    )
    # The supplied profile coefficients are centidegree-based U/V/W/X/Y.
    instrument = ConstantWavelengthInstrument(
        wavelength_angstrom=0.413259,
        u_deg2=1.163e-4,
        v_deg2=-0.126e-4,
        w_deg2=0.063e-4,
        x_deg=0.173e-2,
        y_deg=0.0,
    )
    request = lebail.LeBailInput.from_cif(
        pattern,
        instrument,
        _SUCROSE_CIF,
        phase_id="sucrose",
        initial_intensity=0.0,
        refine_lattice=False,
        instrument_parameters=("u_deg2", "v_deg2", "w_deg2", "x_deg", "y_deg"),
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
        np.corrcoef(observed - background, result.calculation.profile_y)[0, 1]
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
            "passed" if profile_correlation >= 0.98 else "failed",
            "Background-subtracted observed and calculated profiles remain strongly aligned.",
            measured=profile_correlation,
            criterion="Pearson correlation >= 0.98",
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
                "The tutorial's lower final Rwp uses additional staged background, size, "
                "microstrain, lattice, and repeated extraction refinements."
            ),
        ),
    )
