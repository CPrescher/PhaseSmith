"""Exact per-stage execution comparison of already completed raw workflows."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline, candidate):
    arrays = {}
    for filename in [
        "frame-validity.npz",
        "positions.npz",
        "indexed-positions.npz",
        "edge-exclusions.npz",
        "raw-observations-and-bank.npz",
        "reduced-observations.npz",
    ]:
        with (
            np.load(baseline / filename, allow_pickle=False) as old,
            np.load(candidate / filename, allow_pickle=False) as new,
        ):
            keys = set(old.files) - {"metadata_json"}
            if not keys <= set(new.files):
                raise ValueError("missing numerical arrays")
            arrays[filename] = {
                key: old[key].shape == new[key].shape
                and old[key].dtype == new[key].dtype
                and old[key].tobytes() == new[key].tobytes()
                for key in sorted(keys)
            }

    def read(root, name):
        return json.loads((root / name).read_text())

    old, new = read(baseline, "candidate/result.json"), read(candidate, "candidate/result.json")
    report = read(candidate, "report.json")
    complete = read(candidate, "worker-complete.json")
    gates = dict(
        all_numerical_arrays_bitwise_identical=all(all(v.values()) for v in arrays.values()),
        full_candidate_records_identical=old["candidates"] == new["candidates"],
        phasing_histories_identical=old["phasing_histories"] == new["phasing_histories"],
        source_files_unchanged=report["source_files_unchanged"],
        structure_recovered=report["raw_to_structure_search_passed"],
        worker_isolation=not any(
            complete[k]
            for k in ["reference_model_read", "source_hkls_read", "archived_arrays_read"]
        ),
        no_optional_python_numerics=complete["loaded_blocked_python_packages"] == [],
    )
    reduction = read(candidate, "reduction.json")
    return dict(
        schema="phasesmith.enstatite-execution-equivalence.v1",
        baseline=str(baseline),
        candidate=str(candidate),
        passed=all(gates.values()),
        gates=gates,
        array_identity=arrays,
        r1=new["candidates"][0]["r1"],
        reference_assessment=report["reference_assessment"],
        execution={k: v for k, v in reduction["metadata"].items() if k.endswith("_execution")},
        worker=complete,
        timings=read(candidate, "timings-worker.json")
        if (candidate / "timings-worker.json").exists()
        else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = compare(a.baseline, a.candidate)
    with a.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ["passed", "gates", "r1", "execution"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
