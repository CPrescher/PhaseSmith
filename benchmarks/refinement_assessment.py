#!/usr/bin/env python3
"""Fixed real-data assessment; preserve failed gates and compare installed builds.

No oracle invocation, recipe tuning, solver changes or external downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "validation/refinement-assessment-v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def rowles_record(case, bundle):
    import numpy as np
    import phasesmith as ps
    from phasesmith.refinement import rietveld as rv
    from phasesmith.validation import run_rowles_qpa_workflow

    stages, final = [], []
    refine, calculate = rv.refine, rv.calculate

    def capture_refine(request, options, *args, **kwargs):
        result = refine(request, options, *args, **kwargs)
        stages.append(
            dict(
                termination=result.termination_reason.value,
                accepted_iterations=len(result.history),
                evaluations=result.evaluations,
                rwp=result.metrics.rwp,
                backend=result.backend,
                max_iterations=options.limits.max_iterations,
                max_evaluations=options.limits.max_evaluations,
            )
        )
        return result

    def capture_calculate(pattern, *args, **kwargs):
        result = calculate(pattern, *args, **kwargs)
        final[:] = [(pattern, result)]
        return result

    started = time.perf_counter()
    with (
        patch.object(rv, "refine", capture_refine),
        patch.object(rv, "calculate", capture_calculate),
    ):
        result = run_rowles_qpa_workflow(
            bundle, case["id"][-2:], execution=ps.ExecutionPolicy(threads=1)
        )
    elapsed = time.perf_counter() - started
    record = result.to_record()
    record.pop("elapsed_seconds")
    limits = case["limits"]
    checks = dict(
        poisson_rwp=record["poisson_rwp"] <= limits["poisson_rwp"],
        qpa=record["maximum_weight_fraction_error"] <= limits["maximum_weight_fraction_error"],
        correlation=record["profile_correlation"] >= limits["profile_correlation_min"],
        sample_count=record["sample_count"] in (14066, 14091),
        reflection_count=record["reflection_count"] == 109,
    )
    pattern, calculated = final[0]
    weighted_square = ((calculated.y - pattern.observed_y) / pattern.uncertainty) ** 2
    total = float(weighted_square.sum())
    intervals = []
    for indices in np.array_split(np.arange(len(pattern.x)), 10):
        intervals.append(
            dict(
                two_theta_range=[float(pattern.x[indices[0]]), float(pattern.x[indices[-1]])],
                chi_square_fraction=float(weighted_square[indices].sum()) / total if total else 0.0,
            )
        )
    return dict(
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        measurements=record,
        stages=stages,
        largest_residual_intervals=sorted(intervals, key=lambda v: -v["chi_square_fraction"])[:3],
        profile_sha256=hashlib.sha256(calculated.y.tobytes()).hexdigest(),
    ), elapsed


def worker():
    import phasesmith as ps
    from phasesmith.validation import (
        VALIDATION_CASES,
        run_validation_case,
        verify_validation_dataset,
    )

    if ps._core.BUILD_MODE != "release":
        raise RuntimeError("release extension required")
    protocol = json.loads(PROTOCOL.read_text())
    cases = {c["id"]: c for c in protocol["cases"]}
    registered = {c.case_id: c for c in VALIDATION_CASES}
    package = Path(ps.__file__).parent
    metadata = dict(
        package=str(package),
        native_sha256=digest(ps._core.__file__),
        sources={str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
    )
    print(encode(metadata), flush=True)
    with tempfile.TemporaryDirectory(prefix="phasesmith-assessment-") as temporary:
        verified, converted = set(), False
        for line in sys.stdin:
            case = cases[json.loads(line)["case"]]
            directory = ROOT / "validation/data" / case["dataset"]
            try:
                if case["dataset"] not in verified:
                    verify_validation_dataset(case["dataset"], directory)
                    verified.add(case["dataset"])
                if case["id"].startswith("rowles-"):
                    bundle = Path(temporary) / "rowles"
                    if not converted:
                        ps.io.convert_rowles_topas_bundle(directory, bundle)
                        converted = True
                    scientific, elapsed = rowles_record(case, bundle)
                else:
                    started = time.perf_counter()
                    scientific = run_validation_case(registered[case["id"]], directory).to_record()
                    elapsed = time.perf_counter() - started
                    scientific.pop("elapsed_seconds")
                    scientific["reported_terminations"] = re.findall(
                        r"[Tt]ermination=([a-z_]+)", " ".join(scientific["notes"])
                    )
                print(encode(dict(scientific=scientific, seconds=elapsed)), flush=True)
            except Exception as error:
                print(encode(dict(error=f"{type(error).__name__}: {error}")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if args.baseline_package is None or args.json_output is None:
        parser.error("--baseline-package and --json-output required")
    protocol = json.loads(PROTOCOL.read_text())
    output = dict(
        schema="phasesmith.refinement-assessment-results.v1",
        protocol=protocol,
        protocol_sha256=digest(PROTOCOL),
        driver_sha256=digest(__file__),
        environment=dict(python=platform.python_version(), platform=platform.platform()),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        working_tree_patch_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=ROOT)
        ).hexdigest(),
        datasets={
            str(p.relative_to(ROOT)): digest(p)
            for dataset in sorted({c["dataset"] for c in protocol["cases"]})
            for p in sorted((ROOT / "validation/data" / dataset).iterdir())
            if p.is_file()
        },
        builds={},
        results={},
        timing_note=(
            "Shared desktop; paired sequential fits, other workloads not controlled. "
            "Imports, verification and Rowles conversion excluded; complete fixed recipe included."
        ),
    )
    processes = {}
    try:
        for label, package in (("baseline", args.baseline_package.resolve()), ("candidate", None)):
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PHASESMITH_VALIDATION_PYTHON_REFERENCE", None)
            if package is not None:
                env["PYTHONPATH"] = str(package)
            env.update(
                {
                    k: "1"
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
                }
            )
            process = subprocess.Popen(
                [sys.executable, __file__, "--worker"],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes[label] = process
            output["builds"][label] = json.loads(process.stdout.readline())
        for case in protocol["cases"]:
            records = {
                label: dict(scientific=None, seconds=[], exact_repeatability=True)
                for label in processes
            }
            for repeat in range(3):
                order = list(processes) if repeat % 2 == 0 else list(reversed(processes))
                for label in order:
                    process = processes[label]
                    process.stdin.write(encode(dict(case=case["id"])) + "\n")
                    process.stdin.flush()
                    value = json.loads(process.stdout.readline())
                    record = records[label]
                    if "error" in value:
                        record.setdefault("errors", []).append(value["error"])
                        continue
                    if record["scientific"] is None:
                        record["scientific"] = value["scientific"]
                    elif record["scientific"] != value["scientific"]:
                        record["exact_repeatability"] = False
                        record.setdefault("differing_results", []).append(value["scientific"])
                    if repeat:
                        record["seconds"].append(value["seconds"])
            output["results"][case["id"]] = records
            args.json_output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            print(
                case["id"],
                {
                    k: (v["scientific"] or {}).get("status", v.get("errors"))
                    for k, v in records.items()
                },
                flush=True,
            )
    finally:
        for process in processes.values():
            process.stdin.close()
            process.wait(timeout=30)


if __name__ == "__main__":
    main()
