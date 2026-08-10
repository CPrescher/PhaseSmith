#!/usr/bin/env python3
"""Run one NIST SRM 660c scan through a pinned GSAS-II common-model fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
ARCHIVE = "srm_660c_cifs_20201029_081700.zip"
SPECIMEN_PATTERN = re.compile(r"(?:[1-9]00|1000)[ab]")
PROFILE_HEADERS = (
    "_pd_proc_2theta_corrected",
    "_pd_proc_intensity_total",
    "_pd_proc_ls_weight",
)
MATCHED_SH_OVER_L = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--specimen", default="100a")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--sh-over-l", type=float, default=MATCHED_SH_OVER_L)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def tagged_number(text: str, tag: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(tag)}\s+([-+0-9.eE]+)\s*$", text)
    if match is None:
        raise ValueError(f"NIST pdCIF is missing {tag}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"NIST pdCIF has a non-finite {tag}")
    return value


def profile_arrays(text: str) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    lines = text.splitlines()
    profiles = []
    for index, line in enumerate(lines):
        if line.strip() != PROFILE_HEADERS[0]:
            continue
        if tuple(item.strip() for item in lines[index : index + 3]) != PROFILE_HEADERS:
            raise ValueError("NIST pdCIF profile columns have changed")
        rows = []
        for row in lines[index + 3 :]:
            fields = row.split()
            if len(fields) != 3:
                break
            try:
                rows.append(tuple(float(field) for field in fields))
            except ValueError:
                break
        array = np.asarray(rows, dtype=np.float64)
        if array.shape != (5332, 3) or not np.isfinite(array).all():
            raise ValueError("NIST pdCIF profile loop is invalid")
        profiles.append(tuple(np.ascontiguousarray(array[:, column]) for column in range(3)))
    if len(profiles) != 2 or not np.array_equal(profiles[0][0], profiles[1][0]):
        raise ValueError("NIST pdCIF measured/reference profiles are invalid")
    return tuple(profiles)


def read_specimen(data: Path, specimen: str) -> tuple[str, Any, Any, Any, Any, float, float]:
    if SPECIMEN_PATTERN.fullmatch(specimen) is None:
        raise ValueError("invalid NIST SRM 660c specimen")
    member = f"660_cert_cif_mosaic_consensus_{specimen}.cif"
    with zipfile.ZipFile(data / ARCHIVE) as archive:
        info = archive.getinfo(member)
        if info.file_size > 2 * 1024 * 1024:
            raise ValueError("NIST pdCIF exceeds the specimen size limit")
        text = archive.read(info).decode("utf-8")
    measured, calculated = profile_arrays(text)
    return (
        text,
        measured[0],
        measured[1],
        measured[2],
        calculated[1],
        tagged_number(text, "_cell_length_a"),
        tagged_number(text, "_pd_spec_vertical_displacement_mm"),
    )


def gsas_shift_micrometre(displacement_mm: float) -> float:
    """Convert the NIST pdCIF specimen displacement to GSAS-II's shift unit."""

    if not math.isfinite(displacement_mm):
        raise ValueError("NIST specimen displacement must be finite")
    return 1_000.0 * displacement_mm


def instrument_text(sh_over_l: float = MATCHED_SH_OVER_L) -> str:
    if not math.isfinite(sh_over_l) or sh_over_l < 0.0:
        raise ValueError("SH/L must be finite and non-negative")
    return f"""#GSAS-II instrument parameter file
Type:PXC
Bank:1.0
Lam1:1.5405929
Lam2:1.5444274
Zero:0.0
Polariz.:0.7
Azimuth:0.0
I(L2)/I(L1):0.5
U:2.0
V:-2.0
W:5.0
X:0.0
Y:0.0
Z:0.0
SH/L:{sh_over_l:.12g}
Source:CuKa
"""


def structure_text(lattice: float) -> str:
    return f"""data_LaB6
_cell_length_a {lattice:.9f}
_cell_length_b {lattice:.9f}
_cell_length_c {lattice:.9f}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
La1 La 0 0 0 1 0.0045
B1 B 0.198 0.5 0.5 1 0.0035
"""


def configure_gsasii(root: Path, binary_directory: Path | None) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    sys.path.insert(0, str(root))
    from GSASII import GSASIIpath

    if binary_directory is not None:
        binary_directory = binary_directory.resolve()
        sys.path.insert(0, str(binary_directory))
        GSASIIpath.binaryPath = str(binary_directory)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIscriptable

    return GSASIIscriptable, (time.perf_counter_ns() - started) / 1.0e6


def refine(project: Any, histogram: Any, name: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    project.do_refinements([{}], outputnames=[None])
    rwp = histogram.get_wR()
    if rwp is None or not np.isfinite(rwp):
        raise RuntimeError(f"GSAS-II stage {name} has no finite Rwp")
    return {
        "name": name,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1.0e6,
        "rwp_percent": float(rwp),
    }


def run_workflow(scripting: Any, data: Path, arguments: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    text, x, observed, weight, reference, lattice, displacement = read_specimen(
        data, arguments.specimen
    )
    del text
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-nist660c-") as name:
        directory = Path(name)
        pattern_path = directory / "specimen.xye"
        instrument_path = directory / "instrument.instprm"
        structure_path = directory / "lab6.cif"
        np.savetxt(pattern_path, np.column_stack((x, observed, 1.0 / np.sqrt(weight))))
        instrument_path.write_text(instrument_text(arguments.sh_over_l), encoding="utf-8")
        structure_path.write_text(structure_text(lattice), encoding="utf-8")
        setup_started = time.perf_counter_ns()
        project = scripting.G2Project(newgpx=str(directory / "nist660c.gpx"))
        histogram = project.add_powder_histogram(
            str(pattern_path), str(instrument_path), fmthint="Topas"
        )
        histogram.set_refinements(
            {
                "Limits": [float(x[0]), float(x[-1])],
                "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 12},
            }
        )
        histogram.data["Sample Parameters"]["Scale"][1] = False
        # NIST publishes millimetres; GSAS-II's Bragg--Brentano Shift field is
        # expressed in micrometres.
        histogram.data["Sample Parameters"]["Shift"][0] = gsas_shift_micrometre(displacement)
        histogram.data["Sample Parameters"]["Gonio. radius"] = 217.5
        project.set_Controls("cycles", arguments.cycles)
        phase = project.add_phase(
            str(structure_path), phasename="LaB6", histograms=[histogram], fmthint="CIF"
        )
        phase.set_HAP_refinements({"Scale": True})
        setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6
        stages = [refine(project, histogram, "scale_background")]
        histogram.set_refinements({"Instrument Parameters": ["Zero"]})
        stages.append(refine(project, histogram, "instrument"))
        phase.set_HAP_refinements(
            {
                "Size": {"type": "isotropic", "value": 1.0, "refine": True},
            }
        )
        phase.data["Histograms"][histogram.name]["Mustrain"][1][0] = 0.0
        stages.append(refine(project, histogram, "sample_broadening"))
        phase.set_refinements({"Atoms": {"all": "U"}})
        stages.append(refine(project, histogram, "displacement"))
        final_started = time.perf_counter_ns()
        calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
        background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
        residual = calculated - observed
        denominator = float(np.sum(weight * observed**2))
        reference_residual = reference - observed
        covariance = project.data.get("Covariance", {}).get("data", {})
        instrument = histogram.data["Instrument Parameters"][0]
        hap = phase.data["Histograms"][histogram.name]
        result = {
            "specimen": arguments.specimen,
            "sample_count": int(x.size),
            "reflection_count": len(histogram.reflections()["LaB6"]["RefList"]),
            "free_parameter_count": len(covariance.get("varyList", [])),
            "sh_over_l": arguments.sh_over_l,
            "poisson_rwp": float(np.sqrt(np.sum(weight * residual**2) / denominator)),
            "unit_weight_rwp": float(np.sqrt((residual @ residual) / (observed @ observed))),
            "profile_correlation": float(
                np.corrcoef(observed - background, calculated - background)[0, 1]
            ),
            "nist_reference_rwp": float(
                np.sqrt(np.sum(weight * reference_residual**2) / denominator)
            ),
            "nist_reference_correlation": float(np.corrcoef(observed, reference)[0, 1]),
            "model_parameters": {
                "instrument": {key: float(instrument[key][1]) for key in ("U", "V", "W", "Zero")},
                "size_micrometre": float(hap["Size"][1][0]),
                "microstrain": float(hap["Mustrain"][1][0]) * 1.0e-6,
                "u_iso_angstrom2": {
                    atom.label: float(atom.data[atom.cia + 1]) for atom in phase.atoms()
                },
            },
        }
        scalar_values = (
            result["poisson_rwp"],
            result["unit_weight_rwp"],
            result["profile_correlation"],
            result["nist_reference_rwp"],
            result["nist_reference_correlation"],
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise RuntimeError("GSAS-II NIST SRM 660c result contains a non-finite value")
        final_ms = (time.perf_counter_ns() - final_started) / 1.0e6
        return {
            "result": result,
            "stage_rwp_percent": {stage["name"]: stage["rwp_percent"] for stage in stages},
            "timing_ms": {
                "setup": setup_ms,
                "stages": {stage["name"]: stage["elapsed_ms"] for stage in stages},
                "finalization": final_ms,
                "total_workflow": (time.perf_counter_ns() - total_started) / 1.0e6,
            },
        }


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    root = arguments.gsas_root.resolve()
    detected_revision = revision(root)
    if detected_revision != PINNED_REVISION:
        raise RuntimeError(f"GSAS-II revision mismatch: {detected_revision}")
    archive = arguments.data_directory.resolve() / ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(archive)
    scripting, import_ms = configure_gsasii(root, arguments.binary_dir)
    workflow = run_workflow(scripting, arguments.data_directory.resolve(), arguments)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": "nist_srm660c_empirical_common_model",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "recipe": {
            "cycles": arguments.cycles,
            "specimen": arguments.specimen,
            "sh_over_l": arguments.sh_over_l,
        },
        "input_sha256": {ARCHIVE: sha256(archive)},
        "import_ms": import_ms,
        **workflow,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
