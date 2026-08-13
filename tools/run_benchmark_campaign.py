#!/usr/bin/env python3
"""Run the reviewed PhaseSmith/GSAS-II comparison campaign from one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "validation" / "benchmark-campaign.json"
DEFAULT_PIN = REPOSITORY_ROOT / "oracle" / "PINNED_GSASII.json"
ALLOWED_OUTCOMES = frozenset({"pass", "diagnostic", "expected_failure"})
MANAGED_DRIVER_ARGUMENTS = frozenset(
    {"--gsas-python", "--gsas-root", "--binary-dir", "--data-directory", "--json-output"}
)


class CampaignError(RuntimeError):
    """The campaign configuration or one of its reviewed cases is invalid."""


@dataclass(frozen=True, slots=True)
class Assertion:
    """One exact JSON result assertion declared by the manifest."""

    path: str
    equals: object


@dataclass(frozen=True, slots=True)
class CampaignCase:
    """One comparison driver and its reviewed orchestration contract."""

    case_id: str
    title: str
    dataset_id: str
    driver: str
    arguments: tuple[str, ...]
    outcome: str
    oracle_revision_paths: tuple[str, ...]
    assertions: tuple[Assertion, ...]


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Validated campaign manifest."""

    schema_version: int
    campaign_id: str
    cases: tuple[CampaignCase, ...]


def _stable_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.replace("-", "").isalnum():
        raise CampaignError(f"{field} must be a non-empty alphanumeric/hyphen identifier")
    return value


def _safe_relative_file(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise CampaignError(f"{field} must be a safe relative Python file")
    return value


def _nonempty_strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CampaignError(f"{field} must be an array of non-empty strings")
    return tuple(value)


def load_manifest(path: Path) -> CampaignManifest:
    """Load and strictly validate one campaign manifest."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign manifest {path}: {error}") from error
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise CampaignError("campaign manifest schema_version must be 1")
    campaign_id = _stable_id(record.get("campaign_id"), "campaign_id")
    raw_cases = record.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CampaignError("campaign manifest must contain at least one case")
    cases: list[CampaignCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise CampaignError(f"cases[{index}] must be an object")
        case_id = _stable_id(raw.get("case_id"), f"cases[{index}].case_id")
        if case_id in seen:
            raise CampaignError(f"duplicate campaign case {case_id!r}")
        seen.add(case_id)
        title = raw.get("title")
        if not isinstance(title, str) or not title.strip():
            raise CampaignError(f"cases[{index}].title must not be empty")
        outcome = raw.get("outcome")
        if outcome not in ALLOWED_OUTCOMES:
            raise CampaignError(f"cases[{index}].outcome must be one of {sorted(ALLOWED_OUTCOMES)}")
        assertions_record = raw.get("assertions", [])
        if not isinstance(assertions_record, list):
            raise CampaignError(f"cases[{index}].assertions must be an array")
        assertions: list[Assertion] = []
        for assertion_index, assertion in enumerate(assertions_record):
            if not isinstance(assertion, dict) or set(assertion) != {"path", "equals"}:
                raise CampaignError(
                    f"cases[{index}].assertions[{assertion_index}] requires path and equals"
                )
            assertion_path = assertion["path"]
            if not isinstance(assertion_path, str) or not assertion_path:
                raise CampaignError("assertion paths must not be empty")
            assertions.append(Assertion(assertion_path, assertion["equals"]))
        driver_arguments = _nonempty_strings(raw.get("arguments", []), "arguments")
        forbidden = MANAGED_DRIVER_ARGUMENTS.intersection(driver_arguments)
        if forbidden:
            raise CampaignError(
                f"cases[{index}].arguments may not override managed options: "
                f"{', '.join(sorted(forbidden))}"
            )
        cases.append(
            CampaignCase(
                case_id=case_id,
                title=title,
                dataset_id=_stable_id(raw.get("dataset_id"), f"cases[{index}].dataset_id"),
                driver=_safe_relative_file(raw.get("driver"), f"cases[{index}].driver"),
                arguments=driver_arguments,
                outcome=outcome,
                oracle_revision_paths=_nonempty_strings(
                    raw.get("oracle_revision_paths"), "oracle_revision_paths"
                ),
                assertions=tuple(assertions),
            )
        )
    return CampaignManifest(1, campaign_id, tuple(cases))


def json_path(record: object, path: str) -> object:
    """Resolve a dot-separated object/array path without evaluating expressions."""

    current = record
    for component in path.split("."):
        if not component:
            raise CampaignError(f"invalid empty component in JSON path {path!r}")
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdecimal():
            index = int(component)
            if index >= len(current):
                raise CampaignError(f"JSON path {path!r} indexes past the result array")
            current = current[index]
        else:
            raise CampaignError(f"JSON path {path!r} is absent from the driver result")
    return current


def pinned_revision(pin_path: Path) -> str:
    """Return the canonical 40-character GSAS-II revision."""

    try:
        record = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read GSAS-II pin {pin_path}: {error}") from error
    revision = record.get("revision") if isinstance(record, dict) else None
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise CampaignError("GSAS-II pin must contain a lowercase 40-character revision")
    return revision


def verify_gsasii_checkout(root: Path, expected_revision: str) -> None:
    """Fail before scientific work when the oracle checkout is not the exact pin."""

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git could not inspect the checkout"
        raise CampaignError(f"cannot inspect GSAS-II checkout {root}: {detail}")
    if actual != expected_revision:
        raise CampaignError(
            f"GSAS-II revision mismatch: expected {expected_revision}, detected {actual}"
        )


def verify_case_result(case: CampaignCase, report: object, expected_revision: str) -> None:
    """Validate oracle provenance and reviewed result assertions for one case."""

    for path in case.oracle_revision_paths:
        actual = json_path(report, path)
        if actual != expected_revision:
            raise CampaignError(
                f"{case.case_id}: {path} has oracle revision {actual!r}, "
                f"expected {expected_revision}"
            )
    for assertion in case.assertions:
        actual = json_path(report, assertion.path)
        if actual != assertion.equals:
            raise CampaignError(
                f"{case.case_id}: {assertion.path} is {actual!r}, expected {assertion.equals!r}"
            )


def canonical_sha256(record: object) -> str:
    """Hash a finite canonical JSON result for provenance, not tolerance comparison."""

    encoded = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def repository_revision() -> str | None:
    """Return the source revision when the campaign runs from a Git checkout."""

    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and len(revision) == 40 else None


def driver_interpreter() -> Path:
    """Return the active launcher without resolving a virtualenv symlink."""

    return Path(sys.executable).absolute()


def _verify_dataset(dataset_id: str, directory: Path, fetch: bool) -> None:
    from phasesmith.validation import fetch_validation_dataset, verify_validation_dataset

    try:
        if fetch:
            fetch_validation_dataset(dataset_id, directory)
        else:
            verify_validation_dataset(dataset_id, directory)
    except Exception as error:
        raise CampaignError(f"dataset {dataset_id!r} failed verification: {error}") from error


def _command(
    case: CampaignCase,
    *,
    driver_python: Path,
    gsas_python: Path,
    gsas_root: Path,
    binary_directory: Path | None,
    dataset_directory: Path,
    result_path: Path,
) -> list[str]:
    driver = (REPOSITORY_ROOT / case.driver).resolve()
    try:
        driver.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise CampaignError(f"driver escapes the repository: {case.driver}") from error
    if not driver.is_file():
        raise CampaignError(f"campaign driver does not exist: {case.driver}")
    command = [
        str(driver_python),
        str(driver),
        "--gsas-python",
        str(gsas_python),
        "--gsas-root",
        str(gsas_root),
        "--data-directory",
        str(dataset_directory),
        *case.arguments,
        "--json-output",
        str(result_path),
    ]
    if binary_directory is not None:
        command.extend(("--binary-dir", str(binary_directory)))
    return command


def run_case(
    case: CampaignCase,
    *,
    driver_python: Path,
    gsas_python: Path,
    gsas_root: Path,
    binary_directory: Path | None,
    data_root: Path,
    temporary_root: Path,
    expected_revision: str,
) -> dict[str, object]:
    """Run one driver and return its aggregate campaign entry."""

    result_path = temporary_root / f"{case.case_id}.json"
    command = _command(
        case,
        driver_python=driver_python,
        gsas_python=gsas_python,
        gsas_root=gsas_root,
        binary_directory=binary_directory,
        dataset_directory=data_root / case.dataset_id,
        result_path=result_path,
    )
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise CampaignError(
            f"{case.case_id}: driver exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise CampaignError(f"{case.case_id}: driver did not create its JSON result")
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        json.dumps(report, allow_nan=False)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CampaignError(f"{case.case_id}: invalid finite JSON result: {error}") from error
    verify_case_result(case, report, expected_revision)
    return {
        "case_id": case.case_id,
        "title": case.title,
        "dataset_id": case.dataset_id,
        "outcome": case.outcome,
        "status": "completed",
        "elapsed_seconds": elapsed_seconds,
        "result_sha256": canonical_sha256(report),
        "result": report,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--data-directory", type=Path, default=Path("validation/data"))
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("PHASESMITH_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("PHASESMITH_GSASII_BINARY_DIR")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--fetch", action="store_true", help="fetch missing pinned datasets")
    parser.add_argument("--overwrite", action="store_true", help="replace --output explicitly")
    parser.add_argument("--list", action="store_true", help="list manifest cases and exit")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run later cases after a failed comparison and report every failure",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    manifest = load_manifest(arguments.manifest.resolve())
    if arguments.list:
        for case in manifest.cases:
            print(f"{case.case_id:28s} {case.outcome:16s} {case.title}")
        return 0
    if arguments.output is None:
        raise CampaignError("--output is required unless --list is used")
    if arguments.output.exists() and not arguments.overwrite:
        raise CampaignError(
            f"refusing to replace {arguments.output}; pass --overwrite or choose a new path"
        )
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise CampaignError(
            "set --gsas-python and --gsas-root or their documented environment variables"
        )
    gsas_python = arguments.gsas_python.resolve()
    # Preserve a virtual-environment launcher instead of resolving its symlink
    # to the base interpreter, which would discard that environment's packages.
    driver_python = driver_interpreter()
    gsas_root = arguments.gsas_root.resolve()
    binary_directory = arguments.binary_dir.resolve() if arguments.binary_dir else None
    if not gsas_python.is_file():
        raise CampaignError(f"GSAS-II Python does not exist: {gsas_python}")
    if binary_directory is not None and not binary_directory.is_dir():
        raise CampaignError(f"GSAS-II binary directory does not exist: {binary_directory}")
    expected_revision = pinned_revision(arguments.pin.resolve())
    verify_gsasii_checkout(gsas_root, expected_revision)
    selected_ids = set(arguments.case_ids or ())
    available_ids = {case.case_id for case in manifest.cases}
    unknown = selected_ids - available_ids
    if unknown:
        raise CampaignError(f"unknown campaign cases: {', '.join(sorted(unknown))}")
    selected = tuple(
        case for case in manifest.cases if not selected_ids or case.case_id in selected_ids
    )
    data_root = arguments.data_directory.resolve()
    prepared: set[str] = set()
    for case in selected:
        if case.dataset_id not in prepared:
            _verify_dataset(case.dataset_id, data_root / case.dataset_id, arguments.fetch)
            prepared.add(case.dataset_id)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="phasesmith-benchmark-campaign-") as name:
        temporary_root = Path(name)
        for case in selected:
            print(f"RUN  {case.case_id}: {case.title}", flush=True)
            try:
                result = run_case(
                    case,
                    driver_python=driver_python,
                    gsas_python=gsas_python,
                    gsas_root=gsas_root,
                    binary_directory=binary_directory,
                    data_root=data_root,
                    temporary_root=temporary_root,
                    expected_revision=expected_revision,
                )
            except CampaignError as error:
                print(f"FAIL {case.case_id}: {error}", file=sys.stderr, flush=True)
                failures.append({"case_id": case.case_id, "error": str(error)})
                if not arguments.continue_on_error:
                    break
            else:
                results.append(result)
                print(f"PASS {case.case_id} ({result['elapsed_seconds']:.3f} s)", flush=True)

    record = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "status": "passed" if not failures and len(results) == len(selected) else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "phasesmith_revision": repository_revision(),
        "oracle": {"revision": expected_revision},
        "selected_case_ids": [case.case_id for case in selected],
        "completed_case_count": len(results),
        "cases": results,
        "failures": failures,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"campaign={manifest.campaign_id} status={record['status']} "
        f"completed={len(results)}/{len(selected)} output={arguments.output}"
    )
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        raise SystemExit(f"error: {error}") from error
