#!/usr/bin/env python3
"""Run an IUCr QARR 1g/1h case with the pinned external GSAS-II oracle.

This worker imports no PhaseSmith module and performs exactly one workflow run.
The comparison driver launches it repeatedly with GSAS-II's own interpreter.
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
TARGETS_BY_SAMPLE = {
    "1g": {"Al2O3": 0.3137, "ZnO": 0.3421, "CaF2": 0.3442},
    "1h": {"Al2O3": 0.3512, "ZnO": 0.3019, "CaF2": 0.3469},
}
PATTERN_BY_SAMPLE = {"1g": "cpd-1g.prn", "1h": "cpd-1h.prn"}
COMMON_FILES = ("cuka.instprm", "Al2O3.cif", "ZnO.cif", "CaF2.cif")
FCJ_BELOW_MINIMUM = 1.0e-12
GSASII_FCJ_CALCULATION_FLOOR = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", choices=tuple(TARGETS_BY_SAMPLE), default="1g")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument(
        "--fcj",
        choices=("instrument", "below-minimum"),
        default="instrument",
    )
    parser.add_argument("--sample-broadening", choices=("refine", "fixed"), default="refine")
    parser.add_argument("--displacement", choices=("refine", "fixed"), default="refine")
    parser.add_argument(
        "--anisotropic",
        choices=("cif", "trace-mean-isotropic"),
        default="cif",
    )
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


def selected_instrument(source: Path, directory: Path, fcj: str) -> Path:
    if fcj == "instrument":
        return source
    lines = source.read_text(encoding="utf-8").splitlines()
    output = [
        f"SH/L:{FCJ_BELOW_MINIMUM:.12g}" if line.startswith("SH/L:") else line for line in lines
    ]
    if output == lines:
        raise RuntimeError("QARR instrument file does not contain SH/L")
    target = directory / "below-minimum-fcj.instprm"
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    return target


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


def apply_displacement_ablation(phase: Any, mode: str) -> None:
    """Apply a revision-gated trace-mean replacement unavailable in public scripting."""

    if mode == "cif":
        return
    for atom in phase.atoms():
        if atom.adp_flag != "A":
            continue
        anisotropic = np.asarray(atom.ADP, dtype=np.float64)
        atom.data[atom.cia] = "I"
        atom.data[atom.cia + 1] = float(np.mean(anisotropic[:3]))


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


def run_workflow(scripting: Any, data: Path, arguments: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter_ns()
    targets = TARGETS_BY_SAMPLE[arguments.sample]
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-qarr-") as temporary:
        directory = Path(temporary)
        setup_started = time.perf_counter_ns()
        project = scripting.G2Project(newgpx=str(directory / "qarr.gpx"))
        histogram = project.add_powder_histogram(
            str(data / PATTERN_BY_SAMPLE[arguments.sample]),
            str(selected_instrument(data / "cuka.instprm", directory, arguments.fcj)),
            fmthint="Topas",
        )
        histogram.set_refinements(
            {
                "Limits": [5.0, 150.0],
                "Background": {
                    "type": "chebyschev-1",
                    "refine": True,
                    "no. coeffs": 10,
                },
            }
        )
        # Overall sample scale is exactly degenerate with the three phase scales.
        histogram.data["Sample Parameters"]["Scale"][1] = False
        project.set_Controls("cycles", arguments.cycles)
        phases = []
        for name in targets:
            phase = project.add_phase(
                str(data / f"{name}.cif"),
                phasename=name,
                histograms=[histogram],
                fmthint="CIF",
            )
            apply_displacement_ablation(phase, arguments.anisotropic)
            phase.set_HAP_refinements({"Scale": True})
            phases.append(phase)
        setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

        stages = [refine(project, histogram, "scale_background")]
        histogram.set_refinements({"Instrument Parameters": ["U", "V", "W", "Zero"]})
        stages.append(refine(project, histogram, "instrument"))
        if arguments.sample_broadening == "refine":
            for phase in phases:
                phase.set_HAP_refinements(
                    {
                        "Size": {"type": "isotropic", "value": 1.0, "refine": True},
                        "Mustrain": {"type": "isotropic", "refine": True},
                    }
                )
            stages.append(refine(project, histogram, "sample_broadening"))
        if arguments.displacement == "refine":
            for phase in phases:
                phase.set_refinements({"Atoms": {"all": "U"}})
            stages.append(refine(project, histogram, "displacement"))

        final_started = time.perf_counter_ns()
        project.save()
        mass_fractions = histogram.ComputeMassFracs()
        fractions = {name: float(mass_fractions[name][0]) for name in targets}
        uncertainties = {name: float(mass_fractions[name][1]) for name in targets}
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
        covariance_matrix = np.asarray(covariance.get("covMatrix", []))
        final_ms = (time.perf_counter_ns() - final_started) / 1.0e6
        rwp = histogram.get_wR()
        if rwp is None:
            raise RuntimeError("GSAS-II final result does not contain Rwp")
        instrument = histogram.data["Instrument Parameters"][0]
        result = {
            "sample_count": int(selected.sum()),
            "reflection_count": int(
                sum(len(value["RefList"]) for value in histogram.reflections().values())
            ),
            "free_parameter_count": len(vary_list),
            "weight_fractions": fractions,
            "weight_fraction_su": uncertainties,
            "maximum_weight_fraction_error": max(
                abs(fractions[name] - target) for name, target in targets.items()
            ),
            "poisson_rwp": float(rwp) / 100.0,
            "unit_weight_rwp": unit_rwp,
            "profile_correlation": correlation,
            "covariance_shape": list(covariance_matrix.shape),
            "instrument_parameters": {
                name: {
                    "initial": float(instrument[name][0]),
                    "value": float(instrument[name][1]),
                    "refine": bool(instrument[name][2]),
                }
                for name in ("Lam1", "Lam2", "I(L2)/I(L1)", "U", "V", "W", "Zero", "SH/L")
            },
        }
        values = [
            *fractions.values(),
            *uncertainties.values(),
            result["maximum_weight_fraction_error"],
            result["poisson_rwp"],
            unit_rwp,
            correlation,
        ]
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError("GSAS-II QARR result contains a non-finite value")
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
            f"GSAS-II QARR benchmark requires revision {PINNED_REVISION}, "
            f"detected {detected_revision}"
        )
    data = arguments.data_directory.resolve()
    required_files = (PATTERN_BY_SAMPLE[arguments.sample], *COMMON_FILES)
    missing = [name for name in required_files if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(f"QARR data directory is missing {', '.join(missing)}")
    scripting, import_ms = configure_gsasii(root, arguments.binary_dir)
    workflow = run_workflow(scripting, data, arguments)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": detected_revision,
        "scope": f"iucr_qarr_{arguments.sample}_native_workflow",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "recipe": {
            "cycles": arguments.cycles,
            "sample": arguments.sample,
            "fcj": arguments.fcj,
            "sample_broadening": arguments.sample_broadening,
            "displacement": arguments.displacement,
            "anisotropic": arguments.anisotropic,
        },
        "input_sha256": {name: sha256(data / name) for name in required_files},
        "oracle_behavior": {
            "fcj_calculation_floor": GSASII_FCJ_CALCULATION_FLOOR,
            "note": "Values below the floor test the pinned clamp, not a zero-FCJ profile.",
        },
        "import_ms": import_ms,
        **workflow,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
