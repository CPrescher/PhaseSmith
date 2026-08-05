"""Versioned reader and validator for external oracle fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ._pinned_probe import PINNED_REVISION

FORMAT_VERSION: Final = 1


class FixtureValidationError(ValueError):
    """Raised when fixture provenance or array content is invalid."""


@dataclass(frozen=True, slots=True)
class OracleFixture:
    """Validated manifest and immutable mapping of copied NumPy arrays."""

    path: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, NDArray[np.generic]]

    @property
    def cases(self) -> tuple[Mapping[str, Any], ...]:
        """Return fixture cases in their recorded order."""

        return tuple(self.manifest["cases"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FixtureValidationError(f"{name} must be a JSON object")
    return value


def _expect_keys(mapping: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = keys.difference(mapping)
    if missing:
        raise FixtureValidationError(f"{name} is missing required keys: {sorted(missing)!r}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _expect_keys(
        manifest,
        {
            "format_version",
            "fixture_id",
            "kind",
            "provenance",
            "source",
            "input",
            "archive",
            "arrays",
            "cases",
        },
        "manifest",
    )
    if manifest["format_version"] != FORMAT_VERSION:
        raise FixtureValidationError(
            f"unsupported fixture format {manifest['format_version']!r}; expected {FORMAT_VERSION}"
        )
    provenance = _expect_mapping(manifest["provenance"], "provenance")
    _expect_keys(
        provenance,
        {
            "gsasii_revision",
            "gsasii_tag",
            "python_version",
            "numpy_version",
            "platform",
            "generated_at_utc",
            "generator_sha256",
        },
        "provenance",
    )
    if provenance["gsasii_revision"] != PINNED_REVISION:
        raise FixtureValidationError(
            f"fixture revision {provenance['gsasii_revision']!r} does not match {PINNED_REVISION}"
        )
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise FixtureValidationError("cases must be a non-empty JSON array")


def load_fixture(path: str | Path) -> OracleFixture:
    """Load an oracle fixture after validating provenance, hashes, and arrays.

    Args:
        path: Fixture directory or its `manifest.json` path.

    Raises:
        FixtureValidationError: If metadata, pin, archive, or an array is invalid.
    """

    path = Path(path).resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    fixture_path = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureValidationError(
            f"cannot read fixture manifest {manifest_path}: {error}"
        ) from error
    manifest = _expect_mapping(manifest, "manifest")
    _validate_manifest(manifest)

    archive_metadata = _expect_mapping(manifest["archive"], "archive")
    _expect_keys(archive_metadata, {"file", "sha256"}, "archive")
    archive_name = archive_metadata["file"]
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
        raise FixtureValidationError("archive file must be a filename within the fixture directory")
    archive_path = fixture_path / archive_name
    actual_archive_hash = _sha256_file(archive_path)
    if actual_archive_hash != archive_metadata["sha256"]:
        raise FixtureValidationError(
            f"archive SHA-256 mismatch: {actual_archive_hash} != {archive_metadata['sha256']}"
        )

    descriptors = _expect_mapping(manifest["arrays"], "arrays")
    arrays: dict[str, NDArray[np.generic]] = {}
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(descriptors):
                raise FixtureValidationError(
                    "archive member names do not exactly match manifest array descriptors"
                )
            for name, raw_descriptor in descriptors.items():
                descriptor = _expect_mapping(raw_descriptor, f"arrays.{name}")
                _expect_keys(descriptor, {"dtype", "shape", "sha256", "finite"}, f"arrays.{name}")
                array = np.ascontiguousarray(archive[name])
                if str(array.dtype) != descriptor["dtype"]:
                    raise FixtureValidationError(f"array {name!r} dtype does not match manifest")
                if list(array.shape) != descriptor["shape"]:
                    raise FixtureValidationError(f"array {name!r} shape does not match manifest")
                digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
                if digest != descriptor["sha256"]:
                    raise FixtureValidationError(f"array {name!r} SHA-256 does not match manifest")
                if descriptor["finite"] and not np.isfinite(array).all():
                    raise FixtureValidationError(f"array {name!r} contains a non-finite value")
                array.flags.writeable = False
                arrays[name] = array
    except (OSError, ValueError) as error:
        if isinstance(error, FixtureValidationError):
            raise
        raise FixtureValidationError(
            f"cannot read fixture archive {archive_path}: {error}"
        ) from error

    for case_index, raw_case in enumerate(manifest["cases"]):
        case = _expect_mapping(raw_case, f"cases[{case_index}]")
        _expect_keys(case, {"id", "case_kind", "arrays", "parameters"}, f"cases[{case_index}]")
        references = _expect_mapping(case["arrays"], f"cases[{case_index}].arrays")
        missing_arrays = set(references.values()).difference(arrays)
        if missing_arrays:
            raise FixtureValidationError(
                f"case {case['id']!r} references missing arrays: {sorted(missing_arrays)!r}"
            )
        for table in case.get("reflection_tables", []):
            table = _expect_mapping(table, f"cases[{case_index}].reflection_tables")
            _expect_keys(table, {"array", "columns"}, "reflection table")
            if table["array"] not in arrays:
                raise FixtureValidationError(
                    f"case {case['id']!r} references missing reflection array {table['array']!r}"
                )
            reflection_array = arrays[table["array"]]
            if reflection_array.ndim != 2 or reflection_array.shape[1] != len(table["columns"]):
                raise FixtureValidationError(
                    f"reflection array {table['array']!r} does not match recorded columns"
                )

    return OracleFixture(
        path=fixture_path,
        manifest=_freeze_json(manifest),
        arrays=MappingProxyType(arrays),
    )
