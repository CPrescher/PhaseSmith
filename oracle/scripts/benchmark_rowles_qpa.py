#!/usr/bin/env python3
"""Run one converted Rowles QPA pattern with the pinned external GSAS-II oracle."""

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
SCOPE = "curtin_rowles_qpa_topas_common_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--bundle-directory", type=Path, required=True)
    parser.add_argument("--sample", choices=("1a", "1e"), required=True)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--instrument-profile",
        type=Path,
        help="optional fixed U/V/W/X/Y/SH/L JSON profile from an external calibration",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        help="optional NPZ with selected observed/calculated/background/reflection arrays",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def configure_gsasii(root: Path, binary_directory: Path | None) -> Any:
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

    return GSASIIscriptable


def refine(project: Any, histogram: Any, name: str) -> dict[str, float | str]:
    started = time.perf_counter_ns()
    project.do_refinements([{}], outputnames=[None])
    rwp = histogram.get_wR()
    if rwp is None or not np.isfinite(rwp):
        raise RuntimeError(f"GSAS-II stage {name!r} did not produce a finite Rwp")
    return {
        "name": name,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1.0e6,
        "rwp_percent": float(rwp),
    }


def plain_record(value: Any) -> Any:
    """Convert public scripting records into JSON-compatible plain values."""

    if isinstance(value, dict):
        return {str(key): plain_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_record(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_workflow(
    scripting: Any,
    bundle: Path,
    sample: str,
    cycles: int,
    arrays_output: Path | None,
    instrument_profile: dict[str, float] | None,
) -> dict[str, Any]:
    manifest = json.loads((bundle / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("scope") != SCOPE:
        raise ValueError("unsupported neutral Rowles experiment manifest")
    pattern_record = manifest["patterns"][sample]
    targets = pattern_record["weighed_weight_fractions"]
    model = manifest["common_model"]
    phase_names = tuple(model["phases"])
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-rowles-") as temporary:
        directory = Path(temporary)
        project = scripting.G2Project(newgpx=str(directory / "rowles.gpx"))
        histogram = project.add_powder_histogram(
            str(bundle / pattern_record["file"]),
            str(bundle / "common.instprm"),
            fmthint="Topas",
        )
        histogram.set_refinements(
            {
                "Limits": list(model["range_two_theta_deg"]),
                "Background": {
                    "type": "chebyschev-1",
                    "refine": True,
                    "no. coeffs": int(model["background"]["terms"]),
                },
            }
        )
        if instrument_profile is not None:
            for name, value in instrument_profile.items():
                histogram.InstrumentParameters[name][0] = value
                histogram.InstrumentParameters[name][1] = value
                histogram.InstrumentParameters[name][2] = False
        histogram.data["Sample Parameters"]["Scale"][1] = False
        project.set_Controls("cycles", cycles)
        phases = []
        for name in phase_names:
            phase = project.add_phase(
                str(bundle / f"{name}.cif"),
                phasename=name,
                histograms=[histogram],
                fmthint="CIF",
            )
            phase.set_HAP_refinements({"Scale": True})
            phases.append(phase)

        stages = [refine(project, histogram, "scale_background")]
        instrument_stage = ["Zero"] if instrument_profile is not None else ["U", "V", "W", "Zero"]
        histogram.set_refinements({"Instrument Parameters": instrument_stage})
        stages.append(refine(project, histogram, "instrument"))
        for phase in phases:
            phase.set_HAP_refinements(
                {
                    "Size": {"type": "isotropic", "value": 1.0, "refine": True},
                    "Mustrain": {"type": "isotropic", "refine": True},
                }
            )
        stages.append(refine(project, histogram, "sample_broadening"))
        for phase in phases:
            phase.set_refinements({"Atoms": {"all": "U"}})
        stages.append(refine(project, histogram, "displacement"))

        mass_fractions = histogram.ComputeMassFracs()
        fractions = {name: float(mass_fractions[name][0]) for name in phase_names}
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
        covariance = project.data.get("Covariance", {}).get("data", {})
        vary_list = covariance.get("varyList", [])
        instrument_parameters = {
            name: plain_record(value)
            for name, value in histogram.InstrumentParameters.items()
            if name
            in {
                "Type",
                "Lam1",
                "Lam2",
                "I(L2)/I(L1)",
                "Polariz.",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "SH/L",
                "Zero",
            }
        }
        phase_parameters = {}
        for phase in phases:
            hap = phase.getHAPvalues(histogram)
            phase_parameters[phase.name] = {
                "hap": {
                    name: plain_record(hap[name])
                    for name in ("Scale", "Size", "Mustrain", "Pref.Ori.")
                    if name in hap
                },
                "atoms": [
                    {
                        "label": atom.label,
                        "coordinates": list(map(float, atom.coordinates)),
                        "occupancy": float(atom.occupancy),
                        "uiso_angstrom2": float(atom.uiso),
                    }
                    for atom in phase.atoms()
                ],
            }
        if arrays_output is not None:
            arrays_output.parent.mkdir(parents=True, exist_ok=True)
            reflection_lists = histogram.reflections()
            arrays = {
                "x_deg": x[selected],
                "observed_y": observed[selected],
                "weight": np.asarray(histogram.getdata("Yweight"), dtype=np.float64)[selected],
                "calculated_y": calculated[selected],
                "background_y": background[selected],
            }
            arrays.update(
                {
                    f"reflection_list_{name}": np.asarray(
                        reflection_lists[name]["RefList"], dtype=np.float64
                    )
                    for name in phase_names
                }
            )
            np.savez_compressed(arrays_output, **arrays)
        rwp = histogram.get_wR()
        if rwp is None:
            raise RuntimeError("GSAS-II final result does not contain Rwp")
        result = {
            "sample": sample,
            "sample_count": int(selected.sum()),
            "reflection_count": int(
                sum(len(value["RefList"]) for value in histogram.reflections().values())
            ),
            "free_parameter_count": len(vary_list),
            "weight_fractions": fractions,
            "maximum_weight_fraction_error": max(
                abs(fractions[name] - float(targets[name])) for name in phase_names
            ),
            "poisson_rwp": float(rwp) / 100.0,
            "unit_weight_rwp": unit_rwp,
            "profile_correlation": correlation,
        }
        if not all(
            np.isfinite(value)
            for value in (
                *fractions.values(),
                result["maximum_weight_fraction_error"],
                result["poisson_rwp"],
                unit_rwp,
                correlation,
            )
        ):
            raise RuntimeError("GSAS-II Rowles result contains a non-finite value")
        return {
            "result": result,
            "final_parameters": {
                "instrument": instrument_parameters,
                "phases": phase_parameters,
                "vary_list": list(map(str, vary_list)),
            },
            "stage_rwp_percent": {stage["name"]: stage["rwp_percent"] for stage in stages},
            "stage_timing_ms": {stage["name"]: stage["elapsed_ms"] for stage in stages},
        }


def main() -> None:
    arguments = parse_args()
    if arguments.cycles <= 0:
        raise ValueError("cycles must be positive")
    root = arguments.gsas_root.resolve()
    detected_revision = revision(root)
    if detected_revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II Rowles benchmark requires revision {PINNED_REVISION}, "
            f"detected {detected_revision}"
        )
    bundle = arguments.bundle_directory.resolve()
    instrument_profile = None
    if arguments.instrument_profile is not None:
        instrument_profile = json.loads(arguments.instrument_profile.read_text(encoding="utf-8"))
        expected = {"U", "V", "W", "X", "Y", "SH/L"}
        if set(instrument_profile) != expected:
            raise ValueError("fixed instrument profile must contain exactly U/V/W/X/Y/SH/L")
        instrument_profile = {name: float(value) for name, value in instrument_profile.items()}
        if not all(np.isfinite(tuple(instrument_profile.values()))):
            raise ValueError("fixed instrument profile values must be finite")
        if instrument_profile["W"] <= 0.0 or instrument_profile["SH/L"] < 0.0:
            raise ValueError("fixed instrument profile requires W > 0 and SH/L >= 0")
    required = (
        "experiment.json",
        "common.instprm",
        f"{arguments.sample}.xy",
        "Al2O3.cif",
        "ZnO.cif",
        "CaF2.cif",
    )
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"neutral Rowles bundle is missing {', '.join(missing)}")
    scripting = configure_gsasii(root, arguments.binary_dir)
    workflow = run_workflow(
        scripting,
        bundle,
        arguments.sample,
        arguments.cycles,
        arguments.arrays_output,
        instrument_profile,
    )
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": SCOPE,
        "sample": arguments.sample,
        "recipe": {
            "cycles_per_stage": arguments.cycles,
            "common_subset": instrument_profile is None,
            "fixed_calibrated_instrument_profile": instrument_profile,
        },
        "input_sha256": {name: sha256(bundle / name) for name in required},
        "arrays_output_sha256": (
            sha256(arguments.arrays_output) if arguments.arrays_output is not None else None
        ),
        "instrument_profile_sha256": (
            sha256(arguments.instrument_profile)
            if arguments.instrument_profile is not None
            else None
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        **workflow,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
