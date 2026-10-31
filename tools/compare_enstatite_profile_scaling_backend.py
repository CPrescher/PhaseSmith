"""Compare saved v10 native profile/scaling output with preserved v9, after exit."""

import argparse
import json
from pathlib import Path

from compare_enstatite_geometry_backend import ROOT, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "validation/enstatite-raw-workflow-v10.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, json.loads(args.protocol.read_text()))
    report["schema"] = "phasesmith.enstatite-native-profile-scaling-comparison.v1"
    completion = report["completion"]
    report["gates"]["native_worker_python_import_guard"] = (
        set(completion.get("blocked_python_packages", [])) == {"scipy", "h5py", "hdf5plugin"}
        and completion.get("loaded_blocked_python_packages") == []
    )
    reduction = json.loads((args.candidate / "reduction.json").read_text())
    report["numerical_backends"] = {
        key: reduction["metadata"].get(key)
        for key in ("profile_fitting_backend", "empirical_scaling_backend")
    }
    report["gates"]["native_fitting_backends"] = all(
        isinstance(value, str) and value.startswith("phasesmith-spots native")
        for value in report["numerical_backends"].values()
    )
    report["passed"] = all(report["gates"].values())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "gates",
                    "reduction",
                    "amplitude_r1_difference",
                    "candidate_coordinate_difference",
                )
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
