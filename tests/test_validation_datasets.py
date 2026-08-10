from hashlib import sha256
from pathlib import Path

import pytest
from phasesmith.validation import VALIDATION_DATASETS, fetch_validation_dataset
from phasesmith.validation import datasets as datasets_module


def test_registry_has_unique_stable_ids_and_pinned_files() -> None:
    assert {item.dataset_id for item in VALIDATION_DATASETS} == {
        "ansto-echidna-lab6-cw-neutron",
        "aps-sucrose-11bmb",
        "gsasii-pbso4-cw",
        "iucr-qarr-1g",
        "iucr-qarr-1h",
        "nist-srm660c-lab6-xray",
        "powgen-lab6-tof-calibration",
    }
    for dataset in VALIDATION_DATASETS:
        assert dataset.source_url.startswith("https://")
        assert dataset.purpose in {"acceptance", "oracle_integrity", "holdout", "capability"}
        assert dataset.expected_status in {"passed", "failed", "blocked"}
        for external_file in dataset.files:
            assert len(external_file.sha256) == 64
            assert external_file.size_bytes > 0
            assert all(url.startswith("https://") for url in external_file.urls)


def test_fetch_is_explicit_atomic_and_reuses_verified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned pattern\n"
    descriptor = datasets_module.ExternalValidationFile(
        name="pattern.xy",
        sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        urls=("https://example.invalid/pattern.xy",),
    )
    dataset = datasets_module.ValidationDataset(
        dataset_id="test-data",
        title="Test",
        source_url="https://example.invalid/source",
        citation="Test citation",
        license_note="Test-only bytes",
        files=(descriptor,),
    )
    monkeypatch.setitem(datasets_module._DATASETS_BY_ID, "test-data", dataset)
    calls = 0

    def fake_fetch(external_file: object, target: Path) -> None:
        nonlocal calls
        calls += 1
        target.write_bytes(payload)

    monkeypatch.setattr(datasets_module, "_fetch_file", fake_fetch)

    first = fetch_validation_dataset("test-data", tmp_path)
    second = fetch_validation_dataset("test-data", tmp_path)

    assert first == second == (tmp_path / "pattern.xy",)
    assert calls == 1


def test_existing_corrupt_file_fails_without_silent_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = next(iter(VALIDATION_DATASETS)).files[0]
    dataset = datasets_module.ValidationDataset(
        dataset_id="corrupt-data",
        title="Test",
        source_url="https://example.invalid/source",
        citation="Test citation",
        license_note="Test-only bytes",
        files=(descriptor,),
    )
    monkeypatch.setitem(datasets_module._DATASETS_BY_ID, "corrupt-data", dataset)
    (tmp_path / descriptor.name).write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="size mismatch"):
        fetch_validation_dataset("corrupt-data", tmp_path)


def test_unknown_dataset_lists_available_ids(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="aps-sucrose-11bmb"):
        fetch_validation_dataset("not-registered", tmp_path)


def test_reviewed_nonpassing_outcomes_are_explicit() -> None:
    by_id = {dataset.dataset_id: dataset for dataset in VALIDATION_DATASETS}
    assert (by_id["iucr-qarr-1h"].purpose, by_id["iucr-qarr-1h"].expected_status) == (
        "holdout",
        "failed",
    )
    assert (
        by_id["powgen-lab6-tof-calibration"].purpose,
        by_id["powgen-lab6-tof-calibration"].expected_status,
    ) == ("acceptance", "passed")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("purpose", "benchmark", "purpose"),
        ("expected_status", "unknown", "status"),
    ],
)
def test_dataset_policy_rejects_unknown_vocabulary(field: str, value: str, message: str) -> None:
    descriptor = VALIDATION_DATASETS[0].files[0]
    arguments = {
        "dataset_id": "policy-fixture",
        "title": "Policy fixture",
        "source_url": "https://example.invalid/source",
        "citation": "Test citation",
        "license_note": "Test-only bytes",
        "files": (descriptor,),
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        datasets_module.ValidationDataset(**arguments)  # type: ignore[arg-type]
