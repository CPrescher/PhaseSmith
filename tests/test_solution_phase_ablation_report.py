"""Complete-panel and unchanged-policy guards for saved-prediction comparisons."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "phase_ablation", ROOT / "benchmarks/solution_phase_ablation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MANIFEST = ROOT / "validation/solution-corpus-v5.json"
CONTROL = ROOT / "benchmarks/structure-solution-phase-control-macos.json"
LEGACY = ROOT / "benchmarks/structure-solution-phase-legacy-score-control-macos.json"
UPDATED = ROOT / "benchmarks/structure-solution-phase-repaired-score-macos.json"


def test_pair_labels_are_distinct_from_prediction_case_pairs():
    report = module.compare(
        {"control": CONTROL, "legacy": LEGACY, "corrected": UPDATED},
        MANIFEST,
        [("legacy", "corrected")],
    )
    assert report["development_cases"] == 35
    assert not report["holdout_outcomes_opened"]
    assert report["comparisons"][0]["same_native_build"]
    assert report["comparisons"][0]["before"] == "legacy"


@pytest.mark.parametrize("corruption", ["partial", "holdout", "budget", "origin_flag"])
def test_comparison_rejects_incomplete_opened_or_changed_panels(tmp_path, corruption):
    report = json.loads(UPDATED.read_text())
    if corruption == "partial":
        report["runs"].pop()
    elif corruption == "holdout":
        report["holdout_opened"] = True
    elif corruption == "origin_flag":
        report["phase_origin_selection"] = True
    else:
        row = next(r for r in report["runs"] if r["method"] == "phasesmith")
        row["options"]["seed"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=r"complete|controls changed|feature mismatch"):
        module.compare({"control": CONTROL, "changed": path}, MANIFEST)
