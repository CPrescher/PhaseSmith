"""External dataset integrity and exclusion of appended structure instructions."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_integrity_rejects_missing_truncated_and_changed_bytes(tmp_path):
    verifier = load_tool("tools/verify_single_crystal_data.py")
    path = tmp_path / "sample.dat"
    original = b"signed measured data\n"
    record = {
        "path": path.name,
        "size_bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    with pytest.raises(FileNotFoundError):
        verifier.verify_file(tmp_path, record)
    path.write_bytes(original)
    assert verifier.verify_file(tmp_path, record)["sha256"] == record["sha256"]
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="byte-size"):
        verifier.verify_file(tmp_path, record)
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="SHA-256"):
        verifier.verify_file(tmp_path, record)


def test_verification_never_claims_workflow_success_and_rejects_bad_manifests(tmp_path):
    verifier = load_tool("tools/verify_single_crystal_data.py")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"counts")
    record = {"path": path.name, "size_bytes": 6, "sha256": hashlib.sha256(b"counts").hexdigest()}
    manifest = {
        "schema": "phasesmith.local-single-crystal-data.v1",
        "dataset_id": "test",
        "files": [record],
        "total_size_bytes": 6,
    }
    assert verifier.verify_dataset(manifest, tmp_path)["scientific_workflow"] == "not_executed"
    with pytest.raises(ValueError, match="nonempty and unique"):
        verifier.verify_dataset(dict(manifest, files=[record, record]), tmp_path)
    with pytest.raises(ValueError, match="total byte size"):
        verifier.verify_dataset(dict(manifest, total_size_bytes=7), tmp_path)
    with pytest.raises(ValueError, match="must be relative"):
        verifier.verify_file(tmp_path, dict(record, path="../raw.bin"))


def test_source_hkl_conversion_excludes_model_trailer_and_preserves_signed_rows(tmp_path):
    probe = load_tool("benchmarks/enstatite_structure_search.py")
    path = tmp_path / "source.hkl"
    reflection_text = "   1   2   3   -2.00    1.00\n   0   0   0    0.00    0.00\n"
    path.write_text(reflection_text + "\nUNIT 48 16 16\nO1 1 0.1 0.2 0.3 11 0.02\n")
    data, converted, trailer = probe.source_reflections(path)
    assert data.hkl.tolist() == [[1, 2, 3]]
    assert data.intensity.tolist() == [-2.0]
    assert converted == reflection_text
    assert "UNIT" in trailer and "O1" in trailer
    path.write_text(reflection_text.replace("0.00    0.00", "1.00    0.00"))
    with pytest.raises(ValueError, match="nonzero source HKL terminator"):
        probe.source_reflections(path)
