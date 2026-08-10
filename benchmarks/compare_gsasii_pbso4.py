#!/usr/bin/env python3
"""Compare PhaseSmith and pinned GSAS-II on real PbSO4 X-ray/neutron data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith.radiation import RadiationProbe
from phasesmith.validation import run_pbso4_cw_validation, verify_validation_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_pbso4.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "gsasii_pbso4_combined_cw_native_workflow"
PROBES = ("xray", "neutron")
REFERENCE_CELL_ANGSTROM = {"a": 8.48, "b": 5.398, "c": 6.958}
CELL_PATTERN = re.compile(r"Final cell a=([0-9.]+), b=([0-9.]+), c=([0-9.]+) angstrom\.")
GEOMETRY_PATTERN = re.compile(
    r"Debye-Scherrer geometry: fixed radius=([-0-9.]+) mm; refined "
    r"X=([-0-9.]+) micrometre, Y=([-0-9.]+) micrometre\."
)
CROSS_IMPLEMENTATION_LIMITS = {
    "xray_poisson_rwp_delta": 0.005,
    "xray_unit_weight_rwp_delta": 0.005,
    "xray_profile_correlation_delta": 0.001,
    "neutron_poisson_rwp_delta": 0.005,
    "neutron_unit_weight_rwp_delta": 0.006,
    "neutron_profile_correlation_delta": 0.002,
    "neutron_cell_relative_delta": 0.002,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=REPOSITORY_ROOT / "validation" / "data" / "gsasii-pbso4-cw",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--phasesmith-threads", type=int, default=1)
    parser.add_argument("--gsas-cycles", type=int, default=8)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)]


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "timings_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def check_measurement(report: Any, check_id: str) -> float:
    matches = [check.measured for check in report.checks if check.check_id == check_id]
    if len(matches) != 1 or matches[0] is None or not np.isfinite(matches[0]):
        raise RuntimeError(f"PhaseSmith PbSO4 report has no finite {check_id} measurement")
    return float(matches[0])


def phase_result(report: Any, probe: str) -> dict[str, Any]:
    if report.status not in {"passed", "failed"}:
        raise RuntimeError(
            f"PhaseSmith PbSO4 {probe} validation is not numerically comparable: {report.status!r}"
        )
    match = next(
        (CELL_PATTERN.fullmatch(note) for note in report.notes if note.startswith("Final cell")),
        None,
    )
    if match is None:
        raise RuntimeError("PhaseSmith PbSO4 report does not contain a structured final cell")
    cell = {name: float(value) for name, value in zip(("a", "b", "c"), match.groups(), strict=True)}
    stage_notes = [note for note in report.notes if note.startswith("Stage ")]
    termination = next(
        (note for note in report.notes if note.startswith("Termination=")),
        stage_notes[-1] if stage_notes else "not reported",
    )
    result = {
        "validation_status": report.status,
        "sample_count": report.sample_count,
        "reflection_count": report.reflection_count,
        "poisson_rwp": check_measurement(report, "poisson_rwp"),
        "unit_weight_rwp": check_measurement(report, "unit_weight_rwp"),
        "profile_correlation": check_measurement(report, "profile_correlation"),
        "cell_angstrom": cell,
        "cell_refined": probe == "neutron",
        "maximum_reference_cell_relative_error": check_measurement(
            report, "reference_cell_relative_error"
        ),
        "termination": termination,
        "intelligent_stage_notes": stage_notes,
    }
    if probe == "neutron":
        geometry_match = next(
            (
                GEOMETRY_PATTERN.fullmatch(note)
                for note in report.notes
                if note.startswith("Debye-Scherrer geometry:")
            ),
            None,
        )
        if geometry_match is None:
            raise RuntimeError("PhaseSmith PbSO4 report does not contain Debye-Scherrer geometry")
        radius, displace_x, displace_y = map(float, geometry_match.groups())
        result["debye_scherrer_geometry"] = {
            "goniometer_radius_mm": radius,
            "displace_x_micrometre": displace_x,
            "displace_y_micrometre": displace_y,
        }
    return result


def run_phasesmith(
    data_directory: Path,
    warmups: int,
    repetitions: int,
    execution: phasesmith.ExecutionPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_values = {
        "xray": RadiationProbe.X_RAY,
        "neutron": RadiationProbe.NEUTRON,
    }
    for _ in range(warmups):
        for probe in probe_values.values():
            run_pbso4_cw_validation(data_directory, probe, execution=execution)
    reports: dict[str, list[Any]] = {probe: [] for probe in PROBES}
    wall_ms: list[float] = []
    workflow_ms: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        for probe, value in probe_values.items():
            report = run_pbso4_cw_validation(data_directory, value, execution=execution)
            reports[probe].append(report)
            workflow_ms.append(1_000.0 * report.elapsed_seconds)
        wall_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    result = {probe: phase_result(reports[probe][-1], probe) for probe in PROBES}
    for probe in PROBES:
        repeated = [phase_result(report, probe) for report in reports[probe]]
        if any(item != repeated[0] for item in repeated[1:]):
            raise RuntimeError(f"PhaseSmith PbSO4 {probe} repetitions are not deterministic")
    return result, {
        "combined_driver_wall": timing_summary(wall_ms),
        "individual_reported_workflow": timing_summary(workflow_ms),
        "stage_timing_available": False,
    }


def run_gsas_once(
    arguments: argparse.Namespace, report_path: Path, environment: dict[str, str]
) -> tuple[dict[str, Any], float]:
    command = [
        str(arguments.gsas_python),
        str(WORKER),
        "--gsas-root",
        str(arguments.gsas_root),
        "--data-directory",
        str(arguments.data_directory),
        "--report",
        str(report_path),
        "--cycles",
        str(arguments.gsas_cycles),
    ]
    if arguments.binary_dir is not None:
        command.extend(["--binary-dir", str(arguments.binary_dir)])
    started = time.perf_counter_ns()
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    process_ms = (time.perf_counter_ns() - started) / 1.0e6
    if process.returncode != 0:
        raise RuntimeError(
            "GSAS-II PbSO4 worker failed\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    return json.loads(report_path.read_text(encoding="utf-8")), process_ms


def validate_gsas_reports(reports: list[dict[str, Any]]) -> None:
    stable_keys = ("recipe", "input_sha256", "result", "stage_rwp_percent")
    for report in reports:
        if (
            report.get("schema_version") != 1
            or report.get("implementation") != "GSAS-II"
            or report.get("revision") != PINNED_REVISION
            or report.get("scope") != SCOPE
        ):
            raise RuntimeError("GSAS-II PbSO4 worker returned invalid provenance")
    for report in reports[1:]:
        for key in stable_keys:
            if report[key] != reports[0][key]:
                raise RuntimeError(f"GSAS-II PbSO4 repetitions disagree for {key}")


def run_gsas(
    arguments: argparse.Namespace, temporary: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(temporary / "matplotlib"))
    for warmup in range(arguments.warmups):
        run_gsas_once(arguments, temporary / f"warmup-{warmup}.json", environment)
    reports: list[dict[str, Any]] = []
    process_ms: list[float] = []
    for repetition in range(arguments.repetitions):
        report, elapsed = run_gsas_once(
            arguments, temporary / f"result-{repetition}.json", environment
        )
        reports.append(report)
        process_ms.append(elapsed)
    validate_gsas_reports(reports)
    stage_names = tuple(reports[-1]["timing_ms"]["stages"])
    timing = {
        "import": timing_summary([report["import_ms"] for report in reports]),
        "setup": timing_summary([report["timing_ms"]["setup"] for report in reports]),
        "stages": {
            name: timing_summary([report["timing_ms"]["stages"][name] for report in reports])
            for name in stage_names
        },
        "finalization": timing_summary([report["timing_ms"]["finalization"] for report in reports]),
        "total_workflow": timing_summary(
            [report["timing_ms"]["total_workflow"] for report in reports]
        ),
        "external_process": timing_summary(process_ms),
    }
    return (
        reports[-1]["result"],
        timing,
        {
            "recipe": reports[-1]["recipe"],
            "input_sha256": reports[-1]["input_sha256"],
            "stage_rwp_percent": reports[-1]["stage_rwp_percent"],
            "python_version": reports[-1]["python_version"],
            "numpy_version": reports[-1]["numpy_version"],
            "platform": reports[-1]["platform"],
        },
    )


def compare_scientific_results(
    phasesmith_result: dict[str, Any], gsas_result: dict[str, Any]
) -> dict[str, Any]:
    for probe in PROBES:
        if phasesmith_result[probe]["sample_count"] != gsas_result[probe]["sample_count"]:
            raise RuntimeError(f"PhaseSmith and GSAS-II used different PbSO4 {probe} samples")
    measurements: dict[str, float] = {}
    for probe in PROBES:
        for metric in ("poisson_rwp", "unit_weight_rwp", "profile_correlation"):
            measurements[f"{probe}_{metric}_delta"] = abs(
                phasesmith_result[probe][metric] - gsas_result[probe][metric]
            )
    measurements["neutron_cell_relative_delta"] = max(
        abs(
            phasesmith_result["neutron"]["cell_angstrom"][name] - gsas_result["cell_angstrom"][name]
        )
        / gsas_result["cell_angstrom"][name]
        for name in REFERENCE_CELL_ANGSTROM
    )
    checks = {
        name: {
            "measured": measurements[name],
            "limit": limit,
            "passed": measurements[name] <= limit,
        }
        for name, limit in CROSS_IMPLEMENTATION_LIMITS.items()
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    if failed:
        raise RuntimeError("PbSO4 cross-implementation validation failed: " + ", ".join(failed))
    return {
        "status": "passed",
        "checks": checks,
        "reference_cell_angstrom": REFERENCE_CELL_ANGSTROM,
        "reflection_count_note": (
            "Reflection counts are reported but not gated because the implementations "
            "use different family/list conventions."
        ),
        "displacement_note": (
            "Debye-Scherrer X/Y values are reported in micrometres but not gated because "
            "PhaseSmith refines each probe independently while GSAS-II refines one structure "
            "jointly against both histograms."
        ),
    }


def main() -> None:
    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0 or arguments.gsas_cycles <= 0:
        raise ValueError("warmups must be non-negative and repetitions/cycles must be positive")
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    if arguments.phasesmith_threads < 0:
        raise ValueError("--phasesmith-threads must be non-negative")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    verify_validation_dataset("gsasii-pbso4-cw", arguments.data_directory)
    execution = phasesmith.ExecutionPolicy(
        threads=None if arguments.phasesmith_threads == 0 else arguments.phasesmith_threads
    )
    phase_result_record, phase_timing = run_phasesmith(
        arguments.data_directory, arguments.warmups, arguments.repetitions, execution
    )
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-pbso4-comparison-") as name:
        gsas_result, gsas_timing, gsas_metadata = run_gsas(arguments, Path(name))
    cross_validation = compare_scientific_results(phase_result_record, gsas_result)
    ratio = (
        gsas_timing["total_workflow"]["median_ms"]
        / phase_timing["combined_driver_wall"]["median_ms"]
    )
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "comparison_kind": "independent_native_vs_joint_native_workflows",
        "workload": {
            "dataset_id": "gsasii-pbso4-cw",
            "probes": list(PROBES),
            "reference_cell_angstrom": REFERENCE_CELL_ANGSTROM,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "phasesmith_threads": execution.threads,
        },
        "phasesmith": {
            "build_mode": phasesmith._core.BUILD_MODE,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "result": phase_result_record,
            "timing": phase_timing,
        },
        "gsasii": {
            "revision": PINNED_REVISION,
            **gsas_metadata,
            "result": gsas_result,
            "timing": gsas_timing,
        },
        "median_total_workflow_ratio_gsasii_over_phasesmith": ratio,
        "cross_implementation_validation": cross_validation,
        "limitations": [
            (
                "PhaseSmith refines the X-ray and neutron histograms independently; "
                "GSAS-II refines one joint structure."
            ),
            (
                "PhaseSmith refines a three-term Chebyshev correction above a fixed "
                "Smooth Bruckner baseline; GSAS-II refines three background coefficients "
                "from its native background model."
            ),
            (
                "The PhaseSmith X-ray doublet workflow keeps the supplied lattice "
                "fixed; its neutron lattice is refined."
            ),
            (
                "PhaseSmith uses a disclosed intelligent cumulative recipe; GSAS-II "
                "uses the official tutorial's caller-authored cumulative recipe."
            ),
            (
                "Both are complete native workflows, not matched parameterizations "
                "or same-kernel timings."
            ),
        ],
    }
    print(f"scope={SCOPE} repetitions={arguments.repetitions}")
    for probe in PROBES:
        phase = phase_result_record[probe]
        gsas = gsas_result[probe]
        print(
            f"probe={probe} phasesmith_rwp={100 * phase['poisson_rwp']:.3f}% "
            f"gsasii_rwp={100 * gsas['poisson_rwp']:.3f}% "
            f"phasesmith_corr={phase['profile_correlation']:.5f} "
            f"gsasii_corr={gsas['profile_correlation']:.5f}"
        )
    print(
        "cell_reference="
        + ",".join(f"{name}={value:.6f}" for name, value in REFERENCE_CELL_ANGSTROM.items())
    )
    print(
        "cell_phasesmith_neutron="
        + ",".join(
            f"{name}={value:.6f}"
            for name, value in phase_result_record["neutron"]["cell_angstrom"].items()
        )
    )
    print(
        "cell_gsasii_joint="
        + ",".join(f"{name}={value:.6f}" for name, value in gsas_result["cell_angstrom"].items())
    )
    print(
        f"phasesmith_median_ms={phase_timing['combined_driver_wall']['median_ms']:.3f} "
        f"gsasii_median_ms={gsas_timing['total_workflow']['median_ms']:.3f} "
        f"ratio_gsasii_over_phasesmith={ratio:.3f}x"
    )
    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
