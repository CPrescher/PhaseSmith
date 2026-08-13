from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_runner() -> ModuleType:
    name = "run_benchmark_campaign_test"
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "tools/run_benchmark_campaign.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


RUNNER = load_runner()


def test_reviewed_manifest_is_complete_and_uses_safe_unique_drivers() -> None:
    manifest = RUNNER.load_manifest(REPOSITORY_ROOT / "validation/benchmark-campaign.json")

    assert manifest.schema_version == 1
    assert manifest.campaign_id == "gsasii-real-data"
    assert len(manifest.cases) == 14
    assert len({case.case_id for case in manifest.cases}) == len(manifest.cases)
    assert {case.outcome for case in manifest.cases} == {"pass", "diagnostic"}
    assert all((REPOSITORY_ROOT / case.driver).is_file() for case in manifest.cases)
    assert all(case.oracle_revision_paths for case in manifest.cases)


def test_manifest_contract_accepts_every_existing_reviewed_campaign_result() -> None:
    manifest = RUNNER.load_manifest(REPOSITORY_ROOT / "validation/benchmark-campaign.json")
    cases = {case.case_id: case for case in manifest.cases}
    result_files = {
        "rowles-qpa": "2026-08-11-campaign-rowles-gsasii.json",
        "rowles-fpa-diagnostic": "2026-08-11-rowles-gsasii-fpa.json",
        "qarr-1g": "2026-08-11-campaign-qarr-1g.json",
        "qarr-1h": "2026-08-11-campaign-qarr-1h.json",
        "nist-srm660c": "2026-08-11-campaign-nist-srm660c.json",
        "aps-sucrose": "2026-08-11-campaign-sucrose.json",
        "ansto-echidna": "2026-08-11-campaign-echidna.json",
        "pbso4-combined": "2026-08-11-campaign-pbso4.json",
        "powgen-lab6": "2026-08-11-campaign-powgen.json",
        "bath-ltl": "2026-08-11-campaign-bath-ltl.json",
        "xred-tio2": "2026-08-11-campaign-xred-tio2.json",
        "iucr-citrate-silicon": "2026-08-11-campaign-iucr-si-standard.json",
    }
    revision = RUNNER.pinned_revision(REPOSITORY_ROOT / "oracle/PINNED_GSASII.json")

    for case_id, filename in result_files.items():
        report = json.loads((REPOSITORY_ROOT / "validation/results" / filename).read_text())
        RUNNER.verify_case_result(cases[case_id], report, revision)


def test_manifest_rejects_duplicate_cases_and_unsafe_driver_paths(tmp_path: Path) -> None:
    record = {
        "schema_version": 1,
        "campaign_id": "campaign",
        "cases": [
            {
                "case_id": "same",
                "title": "First",
                "dataset_id": "dataset",
                "driver": "../outside.py",
                "arguments": [],
                "outcome": "pass",
                "oracle_revision_paths": ["revision"],
                "assertions": [],
            },
            {
                "case_id": "same",
                "title": "Second",
                "dataset_id": "dataset",
                "driver": "benchmarks/driver.py",
                "arguments": [],
                "outcome": "pass",
                "oracle_revision_paths": ["revision"],
                "assertions": [],
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RUNNER.CampaignError, match="safe relative Python file"):
        RUNNER.load_manifest(path)

    record["cases"][0]["driver"] = "benchmarks/driver.py"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RUNNER.CampaignError, match="duplicate campaign case"):
        RUNNER.load_manifest(path)


def test_json_paths_and_case_assertions_fail_closed() -> None:
    case = RUNNER.CampaignCase(
        case_id="case",
        title="Case",
        dataset_id="dataset",
        driver="benchmarks/compare_gsasii.py",
        arguments=(),
        outcome="pass",
        oracle_revision_paths=("oracle.revision",),
        assertions=(RUNNER.Assertion("checks.0.status", "passed"),),
    )
    revision = "a" * 40
    report = {"oracle": {"revision": revision}, "checks": [{"status": "passed"}]}

    RUNNER.verify_case_result(case, report, revision)
    assert RUNNER.json_path(report, "checks.0.status") == "passed"
    with pytest.raises(RUNNER.CampaignError, match="expected 'passed'"):
        RUNNER.verify_case_result(
            case, {"oracle": {"revision": revision}, "checks": [{"status": "failed"}]}, revision
        )
    with pytest.raises(RUNNER.CampaignError, match="indexes past"):
        RUNNER.json_path(report, "checks.1.status")


def test_driver_and_oracle_interpreters_remain_separate(tmp_path: Path) -> None:
    case = RUNNER.load_manifest(REPOSITORY_ROOT / "validation/benchmark-campaign.json").cases[0]
    driver_python = tmp_path / "phase-python"
    gsas_python = tmp_path / "gsas-python"
    command = RUNNER._command(
        case,
        driver_python=driver_python,
        gsas_python=gsas_python,
        gsas_root=tmp_path / "GSAS-II",
        binary_directory=tmp_path / "bin",
        dataset_directory=tmp_path / "data",
        result_path=tmp_path / "result.json",
    )

    assert command[0] == str(driver_python)
    assert command[command.index("--gsas-python") + 1] == str(gsas_python)
    assert command[command.index("--json-output") + 1] == str(tmp_path / "result.json")
    assert command[command.index("--binary-dir") + 1] == str(tmp_path / "bin")


def test_checkout_preflight_requires_the_exact_pin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="b" * 40 + "\n", stderr=""),
    )

    with pytest.raises(RUNNER.CampaignError, match="revision mismatch"):
        RUNNER.verify_gsasii_checkout(tmp_path, "a" * 40)


def test_list_mode_needs_no_oracle_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tools/run_benchmark_campaign.py"), "--list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "rowles-qpa" in completed.stdout
    assert "iucr-citrate-silicon" in completed.stdout


def test_existing_campaign_output_is_never_replaced_implicitly(tmp_path: Path) -> None:
    output = tmp_path / "reviewed.json"
    output.write_text("reviewed\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools/run_benchmark_campaign.py"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "refusing to replace" in completed.stderr
    assert output.read_text(encoding="utf-8") == "reviewed\n"
