"""Narrow conversion of the checksum-pinned Bath zeolite-L archives.

The deposit contains Rigaku SmartLab ASCII scans, legacy GSAS projects and
multi-block publication CIFs.  This module reads only the reviewed members
listed below.  It does not provide a general GSAS project or ZIP converter.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

BATH_LTL_SAMPLES = ("K", "Li", "Cs")

_PATTERN_MEMBERS = {
    "K": "XRD_Patterns/K-LTL_XRD_data.asc",
    "Li": "XRD_Patterns/Li_LTL_XRD_data.asc",
    "Cs": "XRD_Patterns/Cs_LTL_XRD_data.asc",
}
_CIF_MEMBERS = {
    "K": "CIF files- final refined structures all zeolites/K_LTL_Parent.CIF",
    "Li": "CIF files- final refined structures all zeolites/Li_LTL_Exchanged.CIF",
    "Cs": "CIF files- final refined structures all zeolites/Cs_LTL_Exchanged.CIF",
}
_CIF_PHASE_BLOCKS = {
    "K": "KLTL_O6_2A_PUB_phase_1",
    "Li": "LILTL_LI_K_phase_1",
    "Cs": "CSLTL_CHI80_PART_phase_1",
}
_PROFILE_MEMBERS = {
    "K": "GSAS refinement files/K-LTL refinement/K-LTL.csv",
    "Li": "GSAS refinement files/Li-LTL refinement/LILTL_LI_K_1.csv",
    "Cs": "GSAS refinement files/Cs-LTL refinement/CSLTL_CHI80_PART_2.csv",
}
_EXPERIMENT_MEMBERS = {
    "K": "GSAS refinement files/K-LTL refinement/KLTL_O6_2A.EXP",
    "Li": "GSAS refinement files/Li-LTL refinement/LILTL_LI_K.EXP",
    "Cs": "GSAS refinement files/Cs-LTL refinement/CSLTL_CHI80_PART.EXP",
}


@dataclass(frozen=True, slots=True)
class RigakuAscPattern:
    """Validated one-dimensional scan and selected source metadata."""

    two_theta_deg: np.ndarray
    counts: np.ndarray
    metadata: dict[str, str]


def _read_zip_text(archive: zipfile.ZipFile, member: str, *, limit: int) -> str:
    try:
        info = archive.getinfo(member)
    except KeyError as error:
        raise ValueError(f"Bath archive is missing reviewed member {member!r}") from error
    if info.file_size <= 0 or info.file_size > limit:
        raise ValueError(f"Bath archive member {member!r} has an invalid expanded size")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise ValueError(f"Bath archive member {member!r} is truncated")
    return payload.decode("utf-8-sig")


def read_rigaku_asc_text(text: str) -> RigakuAscPattern:
    """Parse the reviewed packed-count Rigaku ASC representation."""

    lines = text.splitlines()
    metadata: dict[str, str] = {}
    count_line: int | None = None
    end_line: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == "*END":
            end_line = index
        if not line.startswith("*") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        metadata[key] = value.strip()
        if key == "*COUNT":
            count_line = index
    required = ("*START", "*STOP", "*STEP", "*COUNT", "*XUNIT", "*YUNIT")
    if any(key not in metadata for key in required) or count_line is None or end_line is None:
        raise ValueError("Rigaku ASC scan metadata is incomplete")
    if end_line <= count_line:
        raise ValueError("Rigaku ASC count payload is missing")
    try:
        start = float(metadata["*START"])
        stop = float(metadata["*STOP"])
        step = float(metadata["*STEP"])
        count = int(metadata["*COUNT"])
        values = [
            float(token)
            for line in lines[count_line + 1 : end_line]
            for token in line.split(",")
            if token.strip()
        ]
    except ValueError as error:
        raise ValueError("Rigaku ASC scan contains invalid numeric fields") from error
    if metadata["*XUNIT"] != "deg." or metadata["*YUNIT"] != "counts":
        raise ValueError("Rigaku ASC scan must use degrees and counts")
    if count < 3 or step <= 0.0 or len(values) != count:
        raise ValueError("Rigaku ASC count or step is inconsistent")
    x = start + step * np.arange(count, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Rigaku ASC counts must be finite and nonnegative")
    if not math.isclose(float(x[-1]), stop, rel_tol=0.0, abs_tol=5e-7):
        raise ValueError("Rigaku ASC start, stop, step and count are inconsistent")
    return RigakuAscPattern(np.ascontiguousarray(x), np.ascontiguousarray(y), metadata)


def _released_profile(text: str) -> np.ndarray:
    rows: list[tuple[float, float, float, float]] = []
    for row in csv.reader(StringIO(text)):
        if len(row) < 5 or not row[1].strip():
            continue
        try:
            values = tuple(float(value) for value in row[1:5])
        except ValueError:
            continue
        rows.append(values)  # type: ignore[arg-type]
    profile = np.asarray(rows, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[0] < 3 or profile.shape[1] != 4:
        raise ValueError("legacy GSAS profile CSV does not contain four numeric columns")
    if not np.all(np.isfinite(profile)) or np.any(np.diff(profile[:, 0]) <= 0.0):
        raise ValueError("legacy GSAS profile CSV is invalid")
    return np.ascontiguousarray(profile)


def _one_cif_block(text: str, block: str) -> str:
    marker = f"data_{block}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"publication CIF does not contain {block!r}")
    next_block = re.search(r"(?m)^data_", text[start + len(marker) :])
    stop = len(text) if next_block is None else start + len(marker) + next_block.start()
    selected = text[start:stop].rstrip() + "\n"
    # GSAS2CIF truncated O-2 to O-. Preserve the deposited intent in standard
    # charge-after-count notation accepted by the PhaseSmith CIF boundary.
    selected = re.sub(r"(?m)^O-\s*$", "O2-", selected)
    selected = re.sub(r"(?m)^(\s+)O-(\s)", r"\1O2-\2", selected)
    return selected


def _legacy_experiment(text: str) -> dict[str, Any]:
    wavelength = re.search(r"(?m)^HST\s+1 ICONS\s+([0-9.Ee+-]+)", text)
    rpowd = re.search(r"(?m)^HST\s+1 RPOWD\s+([0-9.Ee+-]+)", text)
    first = re.search(
        r"(?m)^HAP1 1PRCF 1\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+"
        r"([0-9.Ee+-]+)\s+([0-9.Ee+-]+)",
        text,
    )
    second = re.search(
        r"(?m)^HAP1 1PRCF 2\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+"
        r"([0-9.Ee+-]+)",
        text,
    )
    if None in (wavelength, rpowd, first, second):
        raise ValueError("legacy GSAS experiment lacks the reviewed CW profile records")
    assert wavelength is not None and rpowd is not None and first is not None and second is not None
    u, v, w, x = map(float, first.groups())
    y, _z, sh_percent = map(float, second.groups())
    return {
        "wavelength_angstrom": float(wavelength.group(1)),
        "reported_poisson_rwp": float(rpowd.group(1)),
        "profile": {
            "u_deg2": u / 10_000.0,
            "v_deg2": v / 10_000.0,
            "w_deg2": w / 10_000.0,
            "x_deg": x / 100.0,
            "y_deg": y / 100.0,
            "sh_over_l": sh_percent / 100.0,
        },
    }


def convert_bath_ltl_bundle(source: str | Path, destination: str | Path) -> Path:
    """Convert the four pinned Bath files into a small neutral benchmark bundle."""

    source_root = Path(source)
    destination_root = Path(destination)
    if source_root.resolve() == destination_root.resolve():
        raise ValueError("source and destination directories must differ")
    required = {
        "patterns.zip": source_root / "patterns.zip",
        "cifs.zip": source_root / "cifs.zip",
        "gsas.zip": source_root / "gsas.zip",
        "README.txt": source_root / "README.txt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Bath source files: {', '.join(missing)}")
    destination_root.mkdir(parents=True, exist_ok=False)
    samples: dict[str, Any] = {}
    with (
        zipfile.ZipFile(required["patterns.zip"]) as patterns,
        zipfile.ZipFile(required["cifs.zip"]) as cifs,
        zipfile.ZipFile(required["gsas.zip"]) as projects,
    ):
        for sample in BATH_LTL_SAMPLES:
            raw_text = _read_zip_text(patterns, _PATTERN_MEMBERS[sample], limit=2_000_000)
            raw = read_rigaku_asc_text(raw_text)
            cif_text = _read_zip_text(cifs, _CIF_MEMBERS[sample], limit=3_000_000)
            phase_text = _one_cif_block(cif_text, _CIF_PHASE_BLOCKS[sample])
            profile = _released_profile(
                _read_zip_text(projects, _PROFILE_MEMBERS[sample], limit=3_000_000)
            )
            legacy = _legacy_experiment(
                _read_zip_text(projects, _EXPERIMENT_MEMBERS[sample], limit=3_000_000)
            )
            np.savetxt(
                destination_root / f"{sample}-raw.xy",
                np.column_stack((raw.two_theta_deg, raw.counts)),
                fmt="%.10g",
            )
            np.savetxt(
                destination_root / f"{sample}-released.csv",
                profile,
                delimiter=",",
                header="two_theta_deg,observed,calculated,background",
                comments="",
                fmt="%.10g",
            )
            (destination_root / f"{sample}-phase1.cif").write_text(phase_text, encoding="utf-8")
            residual = profile[:, 2] - profile[:, 1]
            weights = 1.0 / np.maximum(profile[:, 1], 1.0)
            recomputed = float(
                np.sqrt(np.sum(weights * residual**2) / np.sum(weights * profile[:, 1] ** 2))
            )
            samples[sample] = {
                "raw_pattern": f"{sample}-raw.xy",
                "released_profile": f"{sample}-released.csv",
                "phase": f"{sample}-phase1.cif",
                "raw_scan": {
                    "sample_count": int(raw.two_theta_deg.size),
                    "range_two_theta_deg": [
                        float(raw.two_theta_deg[0]),
                        float(raw.two_theta_deg[-1]),
                    ],
                    "step_deg": float(raw.two_theta_deg[1] - raw.two_theta_deg[0]),
                    "monochromator": raw.metadata.get("*I_MONOCHRO"),
                    "radiation": raw.metadata.get("*XRAY_CHAR"),
                    "wavelength1_angstrom": float(raw.metadata.get("*WAVE_LENGTH1", "nan")),
                    "wavelength2_angstrom": float(raw.metadata.get("*WAVE_LENGTH2", "nan")),
                },
                "legacy_gsas": {**legacy, "recomputed_poisson_rwp": recomputed},
            }
    readme = required["README.txt"].read_text(encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "scope": "bath_ltl_lab_xray_conversion_fidelity",
        "source_dataset_id": "bath-ltl-lab-xray",
        "samples": samples,
        "metadata_discrepancy": {
            "readme_claim": "K-LTL doublet; Li-LTL and Cs-LTL monochromated single wavelength",
            "raw_header_evidence": {
                sample: {
                    "monochromator": record["raw_scan"]["monochromator"],
                    "radiation": record["raw_scan"]["radiation"],
                }
                for sample, record in samples.items()
            },
            "policy": (
                "Prefer instrument-generated ASC headers and archived GSAS records over "
                "README prose."
            ),
            "readme_contains_expected_description": "monochromator" in readme.lower(),
        },
        "translation_notes": [
            "Released GSAS profile CSVs are preserved as observed/calculated/background "
            "diagnostics.",
            "Only the primary zeolite phase block is exported; minor secondary phases are "
            "excluded.",
            "GSAS2CIF O- type symbols are normalized to O2- and recorded as a source-export "
            "repair.",
            "This bundle cannot reconstruct every legacy GSAS project semantic from "
            "publication CIFs.",
        ],
    }
    manifest_path = destination_root / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
