from hashlib import sha256
from pathlib import Path

import pytest
from phasesmith.validation import VALIDATION_DATASETS, fetch_validation_dataset
from phasesmith.validation import datasets as datasets_module


def test_registry_has_unique_stable_ids_and_pinned_files() -> None:
    assert {item.dataset_id for item in VALIDATION_DATASETS} == {
        "ansto-echidna-lab6-cw-neutron",
        "aps-sucrose-11bmb",
        "bath-ltl-lab-xray",
        "curtin-rowles-qpa-topas",
        "gsasii-pbso4-cw",
        "iucr-ceria-size-strain-round-robin",
        "iucr-qarr-1g",
        "iucr-qarr-1h",
        "iucr-dicesium-citrate-si-standard",
        "iucr-sodium-dihydrogen-citrate-si-standard",
        "iucr-tripotassium-citrate-si-standard",
        "iucr-trirubidium-citrate-si-standard",
        "iucr-trirubidium-citrate-monohydrate-si-standard",
        "lanl-nickel-tof",
        "nist-srm660c-lab6-xray",
        "opxrd-robustness-v1",
        "powgen-lab6-tof-calibration",
        "xred-tio2-anatase-rutile",
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


class _DownloadResponse:
    def __init__(self, payload: bytes) -> None:
        self._blocks = iter((payload, b""))

    def __enter__(self) -> "_DownloadResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://example.invalid/pattern.xy"

    def read(self, _size: int) -> bytes:
        return next(self._blocks)


def _download_descriptor(payload: bytes) -> datasets_module.ExternalValidationFile:
    return datasets_module.ExternalValidationFile(
        name="pattern.xy",
        sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        urls=("https://example.invalid/pattern.xy",),
    )


def test_fetch_retries_a_transient_timeout_then_verifies_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned pattern\n"
    descriptor = _download_descriptor(payload)
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(_request: object, *, timeout: int) -> _DownloadResponse:
        nonlocal calls
        calls += 1
        assert timeout == 120
        if calls == 1:
            raise TimeoutError("transient read timeout")
        return _DownloadResponse(payload)

    monkeypatch.setattr(datasets_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets_module.time, "sleep", sleeps.append)

    target = tmp_path / descriptor.name
    datasets_module._fetch_file(descriptor, target)

    assert calls == 2
    assert sleeps == [1.0]
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))


def test_fetch_exhausts_bounded_retries_and_cleans_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned pattern\n"
    descriptor = _download_descriptor(payload)
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(_request: object, *, timeout: int) -> _DownloadResponse:
        nonlocal calls
        calls += 1
        assert timeout == 120
        raise TimeoutError("persistent read timeout")

    monkeypatch.setattr(datasets_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets_module.time, "sleep", sleeps.append)

    target = tmp_path / descriptor.name
    with pytest.raises(RuntimeError, match=r"attempt 3/3.*persistent read timeout"):
        datasets_module._fetch_file(descriptor, target)

    assert calls == 3
    assert sleeps == [1.0, 2.0]
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_fetch_retries_but_never_accepts_a_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned pattern\n"
    corrupt = b"corrupt bytes!\n"
    assert len(corrupt) == len(payload)
    descriptor = _download_descriptor(payload)
    calls = 0

    def fake_urlopen(_request: object, *, timeout: int) -> _DownloadResponse:
        nonlocal calls
        calls += 1
        assert timeout == 120
        return _DownloadResponse(corrupt)

    monkeypatch.setattr(datasets_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(datasets_module.time, "sleep", lambda _delay: None)

    target = tmp_path / descriptor.name
    with pytest.raises(RuntimeError, match=r"attempt 3/3.*SHA-256 mismatch"):
        datasets_module._fetch_file(descriptor, target)

    assert calls == 3
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


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
    assert (
        by_id["bath-ltl-lab-xray"].purpose,
        by_id["bath-ltl-lab-xray"].expected_status,
    ) == ("capability", "failed")
    assert (
        by_id["iucr-sodium-dihydrogen-citrate-si-standard"].purpose,
        by_id["iucr-sodium-dihydrogen-citrate-si-standard"].expected_status,
    ) == ("holdout", "passed")
    assert (
        by_id["iucr-ceria-size-strain-round-robin"].purpose,
        by_id["iucr-ceria-size-strain-round-robin"].expected_status,
    ) == ("holdout", "passed")
    assert (
        by_id["iucr-tripotassium-citrate-si-standard"].purpose,
        by_id["iucr-tripotassium-citrate-si-standard"].expected_status,
    ) == ("holdout", "failed")
    assert (
        by_id["iucr-trirubidium-citrate-si-standard"].purpose,
        by_id["iucr-trirubidium-citrate-si-standard"].expected_status,
    ) == ("holdout", "failed")
    assert (
        by_id["iucr-trirubidium-citrate-monohydrate-si-standard"].purpose,
        by_id["iucr-trirubidium-citrate-monohydrate-si-standard"].expected_status,
    ) == ("holdout", "blocked")
    assert (
        by_id["opxrd-robustness-v1"].purpose,
        by_id["opxrd-robustness-v1"].expected_status,
    ) == ("capability", "passed")


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
