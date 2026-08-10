"""Live real-data comparisons against the exact pinned GSAS-II checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from phasesmith.validation import (
    run_nist_srm660c_validation,
    validation_dataset,
    verify_validation_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "validation" / "data"


def required_path(variable: str, *, directory: bool) -> Path:
    raw = os.environ.get(variable)
    if raw is None:
        pytest.fail(f"{variable} is required by the external-oracle job", pytrace=False)
    path = Path(raw).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        pytest.fail(f"{variable} does not name a valid path: {path}", pytrace=False)
    return path


def run_driver(
    tmp_path: Path,
    name: str,
    arguments: list[str],
    *,
    expected_cross_status: str = "passed",
) -> dict[str, object]:
    gsas_python = required_path("GSASII_PYTHON", directory=False)
    gsas_root = required_path("PHASESMITH_GSASII_ROOT", directory=True)
    binary_dir = required_path("PHASESMITH_GSASII_BINARY_DIR", directory=True)
    output = tmp_path / f"{name}.json"
    command = [
        sys.executable,
        *arguments,
        "--gsas-python",
        str(gsas_python),
        "--gsas-root",
        str(gsas_root),
        "--binary-dir",
        str(binary_dir),
        "--warmups",
        "0",
        "--repetitions",
        "1",
        "--json-output",
        str(output),
    ]
    process = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if process.returncode != 0:
        pytest.fail(
            f"{name} oracle comparison failed\n"
            f"stdout:\n{process.stdout[-5000:]}\n"
            f"stderr:\n{process.stderr[-5000:]}",
            pytrace=False,
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["gsasii"]["revision"] == "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
    assert report["cross_implementation_validation"]["status"] == expected_cross_status
    return report


@pytest.mark.external_oracle
@pytest.mark.parametrize(
    "dataset_id",
    ["aps-sucrose-11bmb", "ansto-echidna-lab6-cw-neutron"],
)
def test_real_lebail_workflows_are_on_par_with_pinned_gsasii(
    tmp_path: Path, dataset_id: str
) -> None:
    verify_validation_dataset(dataset_id, DATA_ROOT / dataset_id)
    run_driver(
        tmp_path,
        dataset_id,
        [
            "benchmarks/compare_gsasii_real_lebail.py",
            "--case",
            dataset_id,
            "--data-directory",
            str(DATA_ROOT / dataset_id),
        ],
        expected_cross_status="passed",
    )


@pytest.mark.external_oracle
@pytest.mark.parametrize("sample", ["1g", "1h"])
def test_qarr_workflows_are_compared_with_pinned_gsasii(tmp_path: Path, sample: str) -> None:
    dataset_id = f"iucr-qarr-{sample}"
    verify_validation_dataset(dataset_id, DATA_ROOT / dataset_id)
    report = run_driver(
        tmp_path,
        dataset_id,
        [
            "benchmarks/compare_gsasii_qarr.py",
            "--sample",
            sample,
            "--data-directory",
            str(DATA_ROOT / dataset_id),
            "--phasesmith-threads",
            "1",
        ],
        expected_cross_status="passed" if sample == "1g" else "failed",
    )
    expected = "passed" if sample == "1g" else "failed"
    assert report["workload"]["expected_phasesmith_status"] == expected
    assert report["phasesmith"]["result"]["validation_status"] == expected


@pytest.mark.external_oracle
def test_pbso4_joint_workflow_is_on_par_with_pinned_gsasii(tmp_path: Path) -> None:
    dataset_id = "gsasii-pbso4-cw"
    verify_validation_dataset(dataset_id, DATA_ROOT / dataset_id)
    run_driver(
        tmp_path,
        dataset_id,
        [
            "benchmarks/compare_gsasii_pbso4.py",
            "--data-directory",
            str(DATA_ROOT / dataset_id),
            "--phasesmith-threads",
            "1",
        ],
    )


@pytest.mark.external_oracle
def test_powgen_tof_kernel_and_pattern_are_on_par_with_pinned_gsasii(
    tmp_path: Path,
) -> None:
    dataset_id = "powgen-lab6-tof-calibration"
    verify_validation_dataset(dataset_id, DATA_ROOT / dataset_id)
    output = tmp_path / "powgen-tof.json"
    command = [
        sys.executable,
        "benchmarks/compare_gsasii_powgen_tof.py",
        "--gsas-python",
        str(required_path("GSASII_PYTHON", directory=False)),
        "--gsas-root",
        str(required_path("PHASESMITH_GSASII_ROOT", directory=True)),
        "--binary-dir",
        str(required_path("PHASESMITH_GSASII_BINARY_DIR", directory=True)),
        "--data-directory",
        str(DATA_ROOT / dataset_id),
        "--json-output",
        str(output),
    ]
    process = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if process.returncode != 0:
        pytest.fail(
            "POWGEN TOF oracle comparison failed\n"
            f"stdout:\n{process.stdout[-5000:]}\n"
            f"stderr:\n{process.stderr[-5000:]}",
            pytrace=False,
        )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["oracle_revision"] == "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
    assert all(check["passed"] for check in report["checks"].values())


@pytest.mark.external_oracle
def test_nist_certified_release_remains_the_primary_external_oracle() -> None:
    dataset_id = "nist-srm660c-lab6-xray"
    dataset = validation_dataset(dataset_id)
    assert dataset.purpose == "oracle_integrity"
    report = run_nist_srm660c_validation(DATA_ROOT / dataset_id)
    assert report.status == "passed"
    assert all(check.status == "passed" for check in report.checks)
