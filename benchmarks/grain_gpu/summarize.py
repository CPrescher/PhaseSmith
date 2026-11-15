"""Condense frozen experiment receipts without rerunning searches."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifact_root", type=Path)
    args = p.parse_args()
    root = args.artifact_root
    result = dict(
        hardware="Apple M4 Pro: 14 CPU cores, 20 GPU cores, 48 GB",
        conclusion=(
            "No measured speed improvement from this GPU FFT/maxima prototype. "
            "Keep CPU production path."
        ),
        provenance=json.loads((root / "provenance.json").read_text()),
        single=[],
        ensemble=[],
        detector=[],
        full_receipts=str(root),
        caveats=[
            "Alternating paired measurements on a shared host, not idle-host confidence intervals.",
            "Same serial scheduling and worker budgets for each CPU/GPU pair; "
            "no multiprocessing GPU service.",
            "Measured-coordinate fixtures are exposed historical training populations, "
            "not raw example 4/5 detector reruns.",
            "GPU arithmetic changes raw FFT bits and some tied peak ordering; "
            "final labels and acceptance are separately audited.",
            "Host preparation in this prototype uses NumPy; "
            "a native staging implementation may reduce that cost.",
        ],
    )
    comparisons = []
    for mode, folder in [("single", "single-final"), ("ensemble", "ensemble")]:
        r = json.loads((root / folder / "results.json").read_text())
        for row in r["summary"]:
            result[mode].append(
                {k: v for k, v in row.items() if k != "comparisons"}
                | dict(
                    gpu_over_cpu=row["gpu_median"] / row["cpu_median"],
                    cpu_fft_fraction=row["cpu_fft_median"] / row["cpu_median"],
                )
            )
        comparisons.extend(r["warm_runs"][i]["comparison"] for i in range(len(r["warm_runs"])))
        assert all(x[k]["repeat_bitwise_equal"] for x in r["warm_runs"] for k in ["cpu", "gpu"])
    detector = json.loads((root / "detector/results.json").read_text())
    assert len(detector) == 3, "strict detector repetitions incomplete"
    for kind in ["cpu", "gpu"]:
        for native in [False, True]:
            paths = [
                root / "detector" / f"{i}-{kind}-native-{native}-fingerprint.json" for i in range(3)
            ]
            assert len({p.read_bytes() for p in paths}) == 1, "detector repeat drift"
    for index, native in enumerate([False, True]):
        rows = [x["comparison"][index] for x in detector]
        result["detector"].append(
            dict(
                native_refinement=native,
                cpu_seconds=float(np.median([x["cpu_seconds"] for x in rows])),
                gpu_seconds=float(np.median([x["gpu_seconds"] for x in rows])),
            )
        )
        comparisons.extend(x["comparison"] for x in rows)
    matches = [m for c in comparisons for m in c["accepted_matches"]]
    result["parity"] = dict(
        paired_complete_searches=len(comparisons),
        labels_exact=all(c["labels_exact"] for c in comparisons),
        acceptance_exact=all(c["acceptance_sequence_equal"] for c in comparisons),
        candidate_counts_equal=all(c["candidates_cpu"] == c["candidates_gpu"] for c in comparisons),
        accepted_models_checked=len(matches),
        fitted_basis_bitwise_equal=sum(m["basis_exact"] for m in matches),
        other_fitted_bases_exact_unimodular=sum(
            not m["basis_exact"]
            and m["basis_change_max_abs_error"] == 0
            and abs(m["basis_change_det"]) == 1
            for m in matches
        ),
        max_lattice_equivalence_error=max(m["basis_change_max_abs_error"] for m in matches),
        transformed_hkl_exact=all(m["transformed_hkl_exact"] for m in matches),
        max_residual_difference=max(m["residual_max_abs"] for m in matches),
        all_result_fields_bitwise_equal=all(
            c["all_result_fields_bitwise_equal"] for c in comparisons
        ),
    )
    assert result["parity"]["labels_exact"] and result["parity"]["acceptance_exact"]
    assert result["parity"]["max_lattice_equivalence_error"] == 0
    assert (
        result["parity"]["transformed_hkl_exact"]
        and result["parity"]["max_residual_difference"] == 0
    )
    result["kernels"] = json.loads((root / "kernels.json").read_text())["summary"]
    result["source_sha256"] = {
        str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    checks_path = root / "checks.json"
    if checks_path.exists():
        result["checks"] = json.loads(checks_path.read_text())
    for path, expected in result["provenance"]["production_files"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    (Path(__file__).parent / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["parity"], indent=2))


if __name__ == "__main__":
    main()
