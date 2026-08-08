#!/usr/bin/env python3
"""Run the official PbSO4 combined-refinement recipe with pinned GSAS-II.

This worker imports no PhaseSmith module. It performs one complete staged
X-ray/neutron refinement using GSAS-II's public scripting interface and writes
a finite, provenance-rich JSON report for the comparison driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "gsasii_pbso4_combined_cw_native_workflow"
REQUIRED_FILES = (
    "PBSO4.XRA",
    "PBSO4.CWN",
    "PbSO4-Wyckoff.cif",
    "INST_XRY.PRM",
    "inst_d1a.prm",
)
REFERENCE_CELL_ANGSTROM = {"a": 8.48, "b": 5.398, "c": 6.958}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=8)
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


def refine(project: Any, histograms: tuple[Any, Any], name: str) -> dict[str, Any]:
    started = time.perf_counter_ns()
    project.do_refinements([{}], outputnames=[None])
    rwp = {
        label: histogram.get_wR()
        for label, histogram in zip(("xray", "neutron"), histograms, strict=True)
    }
    if any(value is None or not np.isfinite(value) for value in rwp.values()):
        raise RuntimeError(f"GSAS-II stage {name!r} did not produce finite Rwp values")
    return {
        "name": name,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1.0e6,
        "rwp_percent": {label: float(value) for label, value in rwp.items()},
    }


def histogram_result(histogram: Any) -> dict[str, Any]:
    x = np.asarray(histogram.getdata("X"), dtype=np.float64)
    observed = np.asarray(histogram.getdata("Yobs"), dtype=np.float64)
    calculated = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
    background = np.asarray(histogram.getdata("Background"), dtype=np.float64)
    low, high = histogram.data["Limits"][1]
    selected = (x >= low) & (x <= high)
    residual = calculated[selected] - observed[selected]
    unit_rwp = float(np.sqrt(np.sum(residual**2) / np.sum(observed[selected] ** 2)))
    correlation = float(
        np.corrcoef(
            observed[selected] - background[selected],
            calculated[selected] - background[selected],
        )[0, 1]
    )
    rwp = histogram.get_wR()
    if rwp is None:
        raise RuntimeError("GSAS-II final histogram does not contain Rwp")
    result = {
        "sample_count": int(selected.sum()),
        "reflection_count": int(
            sum(len(value["RefList"]) for value in histogram.reflections().values())
        ),
        "poisson_rwp": float(rwp) / 100.0,
        "unit_weight_rwp": unit_rwp,
        "profile_correlation": correlation,
        "limits_deg": [float(low), float(high)],
    }
    if not all(
        np.isfinite(value)
        for value in (
            result["poisson_rwp"],
            result["unit_weight_rwp"],
            result["profile_correlation"],
        )
    ):
        raise RuntimeError("GSAS-II PbSO4 histogram result contains a non-finite value")
    return result


def run_workflow(scripting: Any, data: Path, cycles: int) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-pbso4-") as temporary:
        directory = Path(temporary)
        setup_started = time.perf_counter_ns()
        project = scripting.G2Project(newgpx=str(directory / "PbSO4.gpx"))
        xray = project.add_powder_histogram(str(data / "PBSO4.XRA"), str(data / "INST_XRY.PRM"))
        neutron = project.add_powder_histogram(str(data / "PBSO4.CWN"), str(data / "inst_d1a.prm"))
        phase = project.add_phase(
            str(data / "PbSO4-Wyckoff.cif"),
            phasename="PbSO4",
            histograms=[xray, neutron],
            fmthint="CIF",
        )
        project.set_Controls("cycles", cycles)
        histograms = (xray, neutron)
        setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

        xray.set_refinements({"Background": {"no. coeffs": 3, "refine": True}})
        neutron.set_refinements({"Background": {"no. coeffs": 3, "refine": True}})
        stages = [refine(project, histograms, "background")]

        phase.set_refinements({"Cell": True})
        stages.append(refine(project, histograms, "cell"))

        project.set_refinement({"set": {"HStrain": True}}, phase=phase, histogram=[xray])
        stages.append(refine(project, histograms, "xray_hstrain"))

        project.set_refinement(
            {
                "set": {
                    "Mustrain": {"type": "isotropic", "refine": True},
                    "Size": {"type": "isotropic", "refine": True},
                }
            },
            phase=phase,
            histogram=[xray],
        )
        stages.append(refine(project, histograms, "xray_size_microstrain"))

        xray.set_refinements({"Sample Parameters": ["Shift"]})
        neutron.set_refinements({"Sample Parameters": ["DisplaceX", "DisplaceY"]})
        neutron.data["Sample Parameters"]["Gonio. radius"] = 650.0
        phase.set_refinements({"Atoms": {"all": "XU"}})
        stages.append(refine(project, histograms, "sample_and_atoms"))

        xray.set_refinements({"Limits": [16.0, 158.4], "Instrument Parameters": ["U", "V", "W"]})
        neutron.set_refinements({"Limits": [19.0, 153.0], "Instrument Parameters": ["U", "V", "W"]})
        stages.append(refine(project, histograms, "limits_and_instrument"))

        final_started = time.perf_counter_ns()
        project.save()
        cell_data = phase.get_cell()
        cell = {
            "a": float(cell_data["length_a"]),
            "b": float(cell_data["length_b"]),
            "c": float(cell_data["length_c"]),
        }
        maximum_cell_relative_error = max(
            abs(cell[name] - expected) / expected
            for name, expected in REFERENCE_CELL_ANGSTROM.items()
        )
        covariance = project.data.get("Covariance", {}).get("data", {})
        neutron_sample = neutron.data["Sample Parameters"]
        final_ms = (time.perf_counter_ns() - final_started) / 1.0e6
        result = {
            "xray": histogram_result(xray),
            "neutron": histogram_result(neutron),
            "cell_angstrom": cell,
            "reference_cell_angstrom": REFERENCE_CELL_ANGSTROM,
            "maximum_reference_cell_relative_error": maximum_cell_relative_error,
            "neutron_debye_scherrer_geometry": {
                "goniometer_radius_mm": float(neutron_sample["Gonio. radius"]),
                "displace_x_micrometre": float(neutron_sample["DisplaceX"][0]),
                "displace_y_micrometre": float(neutron_sample["DisplaceY"][0]),
            },
            "free_parameter_count": len(covariance.get("varyList", [])),
        }
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
        raise RuntimeError(
            f"GSAS-II PbSO4 benchmark requires revision {PINNED_REVISION}, "
            f"detected {detected_revision}"
        )
    data = arguments.data_directory.resolve()
    missing = [name for name in REQUIRED_FILES if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(f"PbSO4 data directory is missing {', '.join(missing)}")
    scripting, import_ms = configure_gsasii(root, arguments.binary_dir)
    workflow = run_workflow(scripting, data, arguments.cycles)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": SCOPE,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "recipe": {
            "name": "official PbSO4 combined CW tutorial stages 4-9",
            "cycles_per_stage": arguments.cycles,
            "joint_xray_neutron_structure": True,
        },
        "input_sha256": {name: sha256(data / name) for name in REQUIRED_FILES},
        "import_ms": import_ms,
        **workflow,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
