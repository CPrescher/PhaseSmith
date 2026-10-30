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


@pytest.mark.parametrize(
    "corruption",
    [
        "partial",
        "holdout",
        "budget",
        "origin_flag",
        "chemical_flag",
        "iterative_flag",
        "missing_flag",
        "extension_flag",
        "score_only_flag",
        "workspace_flag",
        "work_slice_flag",
        "assignment_flag",
    ],
)
def test_comparison_rejects_incomplete_opened_or_changed_panels(tmp_path, corruption):
    report = json.loads(UPDATED.read_text())
    if corruption == "partial":
        report["runs"].pop()
    elif corruption == "holdout":
        report["holdout_opened"] = True
    elif corruption.endswith("_flag"):
        key = {
            "origin_flag": "phase_origin_selection",
            "chemical_flag": "chemical_assignment",
            "iterative_flag": "iterative_refinement",
            "missing_flag": "missing_reflections",
            "extension_flag": "resolution_extension",
            "score_only_flag": "score_only_evaluation",
            "workspace_flag": "reusable_fourier_workspaces",
            "work_slice_flag": "work_sliced_phasing",
            "assignment_flag": "isotropic_assignment_refinement",
        }[corruption]
        report[key] = 1.2 if corruption == "extension_flag" else True
    else:
        row = next(r for r in report["runs"] if r["method"] == "phasesmith")
        row["options"]["seed"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=r"complete|controls changed|feature mismatch"):
        module.compare({"control": CONTROL, "changed": path}, MANIFEST)


def test_paired_ratio_is_not_the_ratio_of_unpaired_medians():
    a = {"a": {"time": 1.0}, "b": {"time": 10.0}, "c": {"time": 11.0}}
    b = {"a": {"time": 0.5}, "b": {"time": 20.0}, "c": {"time": 11.0}}
    result = module.paired_timing(a, b, a, "time")
    assert result["cases"] == 3
    assert result["median_control_over_run"] == 1.0
    assert result["median_run_minus_control_seconds"] == 0.0
    assert result["faster_cases"] == result["slower_cases"] == 1
    assert module.paired_timing(a, b, a, "missing") is None
    assert module.paired_timing(a, b, [], "time") is None
    b["c"]["time"] = 0.0
    with pytest.raises(ValueError, match="positive finite"):
        module.paired_timing(a, b, a, "time")
