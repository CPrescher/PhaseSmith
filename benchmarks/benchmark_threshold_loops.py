"""Sequential before/after/DIALS timing on the frozen common Fe7C3 image bundle.

Each configuration has one full excluded warmup and three measured repetitions.
The baseline package must be an explicitly preserved package/native extension,
not a rebuild from a mutable checkout. No DIALS dependency enters PhaseSmith.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_dials_records(actual, frozen):
    """Compare all exported fields after canonical centroid/bounding-box sorting."""
    with np.load(actual) as a, np.load(frozen) as b:
        assert set(a.files) == set(b.files)

        def order(data):
            return np.lexsort(np.column_stack([data["xyz"], data["bbox"]]).T[::-1])

        ia, ib = order(a), order(b)
        for name in a.files:
            x, y = a[name][ia], b[name][ib]
            assert x.shape == y.shape and x.dtype == y.dtype, name
            assert x.tobytes() == y.tobytes(), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-site", type=Path, required=True)
    parser.add_argument("--dials-python", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "candidate", "dials"],
        default=["baseline", "candidate", "dials"],
    )
    args = parser.parse_args()
    if len(set(args.workers)) != len(args.workers) or any(w < 1 or w > 32 for w in args.workers):
        parser.error("workers must be distinct integers in 1..32")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_site.resolve()
    candidate = repo / "python"
    report = dict(
        platform=platform.platform(),
        native_python=sys.version,
        numpy=np.__version__,
        baseline_native_sha256=digest(baseline / "phasesmith/_core.abi3.so"),
        candidate_native_sha256=digest(candidate / "phasesmith/_core.abi3.so"),
        native_source_sha256=digest(repo / "crates/phasesmith-spots/src/lib.rs"),
        helpers_sha256={
            name: digest(repo / "benchmarks" / name)
            for name in (
                "rotation_speed_probe.py",
                "dials_position_probe.py",
                "benchmark_threshold_loops.py",
            )
        },
        timing="Sequential; 1 full warmup + 3 measured repeats; cached common images",
        workers=args.workers,
        methods=args.methods,
        rows=[],
    )
    cases = ["Fe7C3_P1_C3_D1s", "Fe7C3_P2_C2_D1s", "Fe7C3_P7_C2_D1s"]
    for case_name in cases:
        case = (args.bundle / case_name).resolve()
        row = dict(case=case_name, configurations={})
        # Alternate the workflow order between worker counts to reduce order bias.
        for index, workers in enumerate(args.workers):
            methods = args.methods if index % 2 == 0 else args.methods[::-1]
            for method in methods:
                label = f"{case_name}-{method}-w{workers}"
                output = (args.output / label).resolve()
                env = os.environ.copy()
                if method == "dials":
                    command = [
                        str(args.dials_python.resolve()),
                        str(repo / "benchmarks/dials_position_probe.py"),
                        "find",
                        str(case),
                        str(output),
                        "--workers",
                        str(workers),
                        "--repeats",
                        "3",
                        "--warmup",
                        "1",
                    ]
                    env.pop("PYTHONPATH", None)
                else:
                    env["PYTHONPATH"] = str(baseline if method == "baseline" else candidate)
                    command = [
                        sys.executable,
                        str(repo / "benchmarks/rotation_speed_probe.py"),
                        str(case),
                        str(output),
                        "--workers",
                        str(workers),
                        "--repeats",
                        "3",
                    ]
                with (args.output / f"{label}.log").open("w") as log:
                    subprocess.run(
                        command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                result = json.loads((output / "result.json").read_text())
                result.pop("geometry", None)
                if method == "dials":
                    check_dials_records(
                        output / "positions.npz", case / "dials-default/positions.npz"
                    )
                    result["exact_frozen_records"] = True
                else:
                    expected_hash = report[f"{method}_native_sha256"]
                    assert result["native_sha256"] == expected_hash
                    assert result["exact_baseline"]
                row["configurations"][f"{method}-w{workers}"] = result
                print(label, f"{result['seconds']:.4f} s", flush=True)
        report["rows"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
