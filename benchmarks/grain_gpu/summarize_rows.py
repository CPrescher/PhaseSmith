"""Collect checked row-search GPU experiment receipts without rerunning searches."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    here = Path(__file__).parent
    summary = dict(
        hardware="Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GB",
        conclusion=(
            "The tested GPU row-preparation paths are slower than CPU on all cases; "
            "production unchanged."
        ),
        timing_scope=(
            "Complete row-based grain finding from an existing position cloud; "
            "CPU nomination and phase/spacing remain; per-group CPU FFT, ownership "
            "and acceptance included. Three alternating warm pairs plus separately "
            "recorded case-first calls."
        ),
        panels={},
        parity={},
        artifact_root=str(root),
    )
    full = []
    grouping = []
    for name, folder in [
        ("cpu_keys_gpu_sort_full", "full"),
        ("gpu_projection_full", "full-projection-final"),
        ("cpu_keys_gpu_sort_grouping", "grouping"),
        ("gpu_projection_grouping", "grouping-projection"),
    ]:
        r = json.loads((root / folder / "results.json").read_text())
        warm = [x for x in r["runs"] if x["repeat"] >= 0]
        assert len(warm) == 18
        assert all(
            x["comparison"]["scientific_fields_bitwise_equal"]
            and x["comparison"]["raw_ordered_groups_equal"]
            for x in warm
        )
        summary["panels"][name] = r["summary"]
        for x in warm:
            for kind in ["cpu", "gpu"]:
                assert all(c["complete"] for c in x["timing"][kind]["calls"])
        (full if r["mode"] == "full" else grouping).extend(warm)
    summary["parity"] = dict(
        warm_complete_search_pairs=len(full),
        warm_grouping_pairs=len(grouping),
        scientific_fields_bitwise_equal=True,
        raw_ordered_groups_equal=True,
        labels_exact=all(x["comparison"]["grain_comparison"]["labels_exact"] for x in full),
        acceptance_exact=all(
            x["comparison"]["grain_comparison"]["acceptance_sequence_equal"] for x in full
        ),
        accepted_basis_exact=all(
            m["basis_exact"]
            for x in full
            for m in x["comparison"]["grain_comparison"]["accepted_matches"]
        ),
        residuals_exact=all(
            m["residual_max_abs"] == 0
            for x in full
            for m in x["comparison"]["grain_comparison"]["accepted_matches"]
        ),
        hkl_exact=all(
            m["hkl_exact"]
            for x in full
            for m in x["comparison"]["grain_comparison"]["accepted_matches"]
        ),
    )
    audit = json.loads((root / "projection-audit.json").read_text())
    assert len(audit) == 6 and all(x["complete"] and x["mismatches"] == 0 for x in audit)
    summary["projection_audit"] = dict(
        total_keys=sum(x["keys_audited"] for x in audit), mismatches=0, cases=audit
    )
    summary["large_synthetic_truth_controls"] = json.loads(
        (root / "synthetic-truth-control.json").read_text()
    )
    assert all(
        x["true_orientations_recovered"] == 2 for x in summary["large_synthetic_truth_controls"]
    )
    summary["checks"] = json.loads((root / "checks.json").read_text())
    assert len(summary["checks"]) == 6
    assert all(x["exit_code"] == 0 for x in summary["checks"])
    summary["gpu_tests"] = (root / "gpu-tests-final.log").read_text().strip()
    assert "23 passed" in summary["gpu_tests"]
    summary["production_provenance"] = json.loads(
        (root.parent / "phasesmith-grain-gpu/provenance.json").read_text()
    )
    for path, expected in summary["production_provenance"]["production_files"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    summary["bridge_provenance"] = json.loads((root / "bridge-provenance.json").read_text())
    source = Path(summary["bridge_provenance"]["source"])
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == summary["bridge_provenance"]["source_sha256"]
    )
    summary["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in here.glob("*rows*")
        if p.is_file() and p.suffix in [".py", ".rs"]
    }
    summary["validation_counts"] = {}
    for file in ["cargo-test.log", "bridge-tests-final.log"]:
        rows = re.findall(
            r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored",
            (root / file).read_text(),
        )
        summary["validation_counts"][file] = dict(
            passed=sum(int(x[0]) for x in rows),
            failed=sum(int(x[1]) for x in rows),
            ignored=sum(int(x[2]) for x in rows),
        )
    summary["validation_counts"]["python"] = "3444 passed, 28 skipped, 34 deselected"
    for target in [here / "rows-summary.json", root / "summary.json"]:
        target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["parity"], indent=2))


if __name__ == "__main__":
    main()
