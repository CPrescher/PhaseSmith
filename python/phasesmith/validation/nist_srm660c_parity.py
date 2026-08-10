"""Empirical common-model PhaseSmith checkpoint for NIST SRM 660c scans."""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..execution import ExecutionPolicy
from ..instrument import ConstantWavelengthInstrument, FcjGeometry
from ..intensity_corrections import BraggBrentanoPolarizedLp
from ..pattern import PowderPattern
from ..radiation import BraggBrentanoGeometry, ConstantWavelengthExperiment, WavelengthComponents
from ..refinement import rietveld
from ..refinement.background import ChebyshevBackground
from ..sample import IsotropicSizeBroadening
from ..scattering import XrayNonResonant
from ._rietveld_parity import refine_nonlinear_block, solve_linear_profile_block

_ARCHIVE = "srm_660c_cifs_20201029_081700.zip"
_MEMBER_TEMPLATE = "660_cert_cif_mosaic_consensus_{specimen}.cif"
_SPECIMEN_PATTERN = re.compile(r"(?:[1-9]00|1000)[ab]")
_PROFILE_HEADERS = (
    "_pd_proc_2theta_corrected",
    "_pd_proc_intensity_total",
    "_pd_proc_ls_weight",
)
_LAMBDA_1 = 1.5405929
_LAMBDA_2 = 1.5444274
_COMMON_PROFILE_SEED = {
    "u_deg2": 2.0e-4,
    "v_deg2": -2.0e-4,
    "w_deg2": 5.0e-4,
    "zero_shift_deg": 0.0,
    "size_nm": 1_000.0,
    "La_u_iso": 0.0045,
    "B_u_iso": 0.0035,
}


def _number_after_tag(text: str, tag: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(tag)}\s+([-+0-9.eE]+)\s*$", text)
    if match is None:
        raise ValueError(f"NIST pdCIF is missing {tag}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"NIST pdCIF has a non-finite {tag}")
    return value


def _profile_arrays(text: str) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    lines = text.splitlines()
    profiles = []
    for index, line in enumerate(lines):
        if line.strip() != _PROFILE_HEADERS[0]:
            continue
        if tuple(item.strip() for item in lines[index : index + 3]) != _PROFILE_HEADERS:
            raise ValueError("NIST pdCIF profile columns are not in the documented order")
        rows = []
        for row in lines[index + 3 :]:
            fields = row.split()
            if len(fields) != 3:
                break
            try:
                values = tuple(float(field) for field in fields)
            except ValueError:
                break
            rows.append(values)
        array = np.asarray(rows, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("NIST pdCIF contains an invalid profile loop")
        profiles.append(tuple(np.ascontiguousarray(array[:, column]) for column in range(3)))
    if len(profiles) != 2:
        raise ValueError("NIST pdCIF must contain measured and calculated profile loops")
    if not np.array_equal(profiles[0][0], profiles[1][0]):
        raise ValueError("NIST measured and calculated grids differ")
    if np.any(profiles[0][2] <= 0.0):
        raise ValueError("NIST measured least-squares weights must be positive")
    return tuple(profiles)


def read_nist_srm660c_specimen(
    dataset_directory: str | Path, specimen: str
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Read one bounded measured/reference pdCIF pair from the pinned archive."""

    if _SPECIMEN_PATTERN.fullmatch(specimen) is None:
        raise ValueError("NIST SRM 660c specimen must be 100a..1000a or 100b..1000b")
    member = _MEMBER_TEMPLATE.format(specimen=specimen)
    with zipfile.ZipFile(Path(dataset_directory) / _ARCHIVE) as archive:
        information = archive.getinfo(member)
        if information.file_size > 2 * 1024 * 1024:
            raise ValueError("NIST pdCIF exceeds the 2 MiB specimen limit")
        text = archive.read(information).decode("utf-8")
    measured, calculated = _profile_arrays(text)
    displacement = _number_after_tag(text, "_pd_spec_vertical_displacement_mm")
    return text, measured[0], measured[1], measured[2], calculated[1], displacement


@dataclass(frozen=True, slots=True)
class NistSrm660cParityResult:
    """Finite result from one empirical common-model SRM 660c refinement."""

    specimen: str
    sample_count: int
    reflection_count: int
    free_parameter_count: int
    poisson_rwp: float
    unit_weight_rwp: float
    profile_correlation: float
    nist_reference_rwp: float
    nist_reference_correlation: float
    termination_reasons: tuple[str, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.poisson_rwp,
            self.unit_weight_rwp,
            self.profile_correlation,
            self.nist_reference_rwp,
            self.nist_reference_correlation,
            self.elapsed_seconds,
        )
        if _SPECIMEN_PATTERN.fullmatch(self.specimen) is None or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("NIST SRM 660c parity result is invalid or non-finite")
        if self.sample_count <= 0 or self.reflection_count <= 0 or self.free_parameter_count <= 0:
            raise ValueError("NIST SRM 660c parity result counts must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def run_nist_srm660c_parity_workflow(
    dataset_directory: str | Path,
    specimen: str = "100a",
    *,
    execution: ExecutionPolicy | None = None,
) -> NistSrm660cParityResult:
    """Refine one NIST scan with the documented empirical common model."""

    started = perf_counter()
    text, x, observed, weight, nist_calculated, displacement = read_nist_srm660c_specimen(
        dataset_directory, specimen
    )
    pattern = PowderPattern(
        x,
        observed_y=observed,
        uncertainty=1.0 / np.sqrt(weight),
        background=np.zeros_like(observed),
    )
    instrument = ConstantWavelengthInstrument(
        _LAMBDA_1,
        _COMMON_PROFILE_SEED["u_deg2"],
        _COMMON_PROFILE_SEED["v_deg2"],
        _COMMON_PROFILE_SEED["w_deg2"],
        0.0,
        0.0,
    )
    experiment = replace(
        ConstantWavelengthExperiment.x_ray_components(
            instrument,
            WavelengthComponents.doublet(_LAMBDA_1, _LAMBDA_2, 0.5),
            geometry=BraggBrentanoGeometry(217.5, displacement),
            axial_geometry=FcjGeometry(0.01, 0.01),
        ),
        zero_shift_deg=_COMMON_PROFILE_SEED["zero_shift_deg"],
    )
    selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
    )
    # The released pdCIF writes the centrosymmetric LaB6 group as ``P m 3 m``.
    # Normalize that legacy spelling to its unambiguous modern Hermann--Mauguin form.
    normalized_cif = text.replace("'P m 3 m'", "'P m -3 m'")
    if normalized_cif == text:
        raise ValueError("NIST pdCIF does not contain the expected legacy LaB6 space group")
    loaded = rietveld.RietveldInput.from_cif(
        pattern,
        experiment,
        normalized_cif,
        phase_id="LaB6",
        block="cell_LaB6_660c",
        selection=selection,
        scattering=XrayNonResonant(),
        intensity_correction=BraggBrentanoPolarizedLp(_LAMBDA_1, 0.7),
        coordinate_tolerance=1.0e-6,
    )
    structure = replace(
        loaded.phases[0].structure,
        sites=tuple(
            replace(
                site,
                u_iso_angstrom2=_COMMON_PROFILE_SEED[f"{site.element_symbol}_u_iso"],
            )
            for site in loaded.phases[0].structure.sites
        ),
    )
    phase = replace(
        loaded.phases[0],
        structure=structure,
        physics=IsotropicSizeBroadening(_COMMON_PROFILE_SEED["size_nm"], shape_factor=1.0),
    )
    background = ChebyshevBackground(
        "nist_660c_empirical_background",
        tuple(0.0 for _ in range(12)),
        (float(x[0]), float(x[-1])),
    )
    selected_execution = ExecutionPolicy() if execution is None else execution
    phases, background = solve_linear_profile_block(
        pattern, experiment, (phase,), background, selected_execution
    )
    position_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=False,
        instrument_parameters=("zero_shift_deg",),
        background=False,
    )
    broadening_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=False,
        sample_physics=True,
        background=False,
    )
    displacement_selection = rietveld.RietveldParameterSelection(
        phase_scale=False,
        lattice=False,
        coordinates=False,
        occupancy=False,
        u_iso=True,
        sample_physics=False,
        background=False,
    )
    stages = []
    for _cycle in range(3):
        position_stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            position_selection,
            selected_execution,
            iterations=30,
            step=0.05,
        )
        stages.append(position_stage)
        experiment = position_stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, position_stage.phases, background, selected_execution
        )
        broadening_stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            broadening_selection,
            selected_execution,
            iterations=30,
            step=0.10,
        )
        stages.append(broadening_stage)
        experiment = broadening_stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, broadening_stage.phases, background, selected_execution
        )
        displacement_stage = refine_nonlinear_block(
            pattern,
            experiment,
            phases,
            background,
            displacement_selection,
            selected_execution,
            iterations=30,
            step=0.10,
        )
        stages.append(displacement_stage)
        experiment = displacement_stage.experiment
        phases, background = solve_linear_profile_block(
            pattern, experiment, displacement_stage.phases, background, selected_execution
        )
    calculation = rietveld.calculate(
        pattern,
        experiment,
        phases,
        background=background,
        support_fwhm=30.0,
        execution=selected_execution,
    )
    residual = calculation.y - observed
    reference_residual = nist_calculated - observed
    weighted_denominator = float(np.sum(weight * observed**2))
    poisson_rwp = float(np.sqrt(np.sum(weight * residual**2) / weighted_denominator))
    reference_rwp = float(np.sqrt(np.sum(weight * reference_residual**2) / weighted_denominator))
    return NistSrm660cParityResult(
        specimen=specimen,
        sample_count=x.size,
        reflection_count=phases[0].reflections.reflection_count,
        free_parameter_count=1 + 12 + 1 + 1 + len(phases[0].structure.sites),
        poisson_rwp=poisson_rwp,
        unit_weight_rwp=float(np.sqrt((residual @ residual) / (observed @ observed))),
        profile_correlation=float(
            np.corrcoef(observed - calculation.background, calculation.profile_y)[0, 1]
        ),
        nist_reference_rwp=reference_rwp,
        nist_reference_correlation=float(np.corrcoef(observed, nist_calculated)[0, 1]),
        termination_reasons=tuple(stage.termination_reason.value for stage in stages),
        elapsed_seconds=perf_counter() - started,
    )
