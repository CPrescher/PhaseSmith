#!/usr/bin/env python3
"""Generate pinned Rust scattering tables from reviewed public-domain sources."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

XRAY_COMMIT = "663d2171bd301dc51dbe048cae459934e60347c2"
XRAY_SHA256 = "047208c2e0e48808cc8a01eaf1e79cd133ba448c583d922ef162000771339089"
NEUTRON_COMMIT = "182ef63a9ec118ef725aae5bb81860f4ba0fb573"
NEUTRON_SHA256 = "47c5f841100c91fb1d063be9503d0a1bc35a80ae1e9d0791f387ad05e5589ed9"
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


@dataclass(frozen=True)
class XrayRow:
    key: str
    atomic_number: int
    a: tuple[float, float, float, float, float]
    b: tuple[float, float, float, float, float]
    c: float


@dataclass(frozen=True)
class NeutronRow:
    key: str
    atomic_number: int
    isotope: int | None
    b_c_fm: float
    uncertainty_fm: float | None
    energy_dependent: bool
    derived_alias_of: str | None = None


def _revision(checkout: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_source(checkout: Path, expected_commit: str, relative: str, expected_sha: str) -> Path:
    actual_commit = _revision(checkout)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"unexpected revision for {checkout}: {actual_commit}; expected {expected_commit}"
        )
    source = checkout / relative
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"unexpected SHA-256 for {source}: {actual_sha}; expected {expected_sha}"
        )
    return source


def _xray_rows(source: Path) -> list[XrayRow]:
    lines = source.read_text(encoding="ascii").splitlines()
    rows: list[XrayRow] = []
    for index, line in enumerate(lines):
        if not line.startswith("#S "):
            continue
        fields = line[3:].split()
        if len(fields) != 2:
            raise ValueError(f"malformed X-ray species header: {line!r}")
        atomic_number, key = int(fields[0]), fields[1]
        numeric = None
        for candidate in lines[index + 1 :]:
            if candidate.startswith("#S "):
                break
            if candidate and not candidate.startswith("#"):
                numeric = [float(value) for value in candidate.split()]
                break
        if numeric is None or len(numeric) != 11:
            raise ValueError(f"missing X-ray coefficients for {key}")
        rows.append(
            XrayRow(
                key,
                atomic_number,
                tuple(numeric[:5]),
                tuple(numeric[6:]),
                numeric[5],
            )
        )
    if len(rows) != 211 or len({row.key for row in rows}) != len(rows):
        raise ValueError("expected 211 unique Waasmaier--Kirfel species")
    return sorted(rows, key=lambda row: row.key)


def _number_with_uncertainty(raw: str) -> tuple[float | None, float | None]:
    text = raw.replace("<", "").replace("*", "").strip()
    if not text:
        return None, None
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\(([^)]+)\))?", text)
    if match is None:
        raise ValueError(f"unsupported neutron number {raw!r}")
    value_text, uncertainty_text = match.groups()
    value = float(value_text)
    if uncertainty_text is None:
        return value, None
    if "." in uncertainty_text or "." not in value_text:
        return value, float(uncertainty_text)
    decimals = len(value_text.partition(".")[2])
    return value, int(uncertainty_text) / 10**decimals


def _embedded_neutron_table(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    marker = 'nsftable = """\\\n'
    start = text.index(marker) + len(marker)
    end = text.index('\n"""', start)
    return [line.removesuffix("\\") for line in text[start:end].splitlines()]


def _neutron_rows(source: Path) -> list[NeutronRow]:
    parsed: list[NeutronRow] = []
    first_isotope: dict[str, NeutronRow] = {}
    natural_symbols: set[str] = set()
    for line in _embedded_neutron_table(source):
        columns = line.split(",")
        if len(columns) != 11:
            raise ValueError(f"malformed neutron row: {line!r}")
        identity = columns[0].split("-")
        atomic_number = int(identity[0])
        if atomic_number == 0:
            continue
        symbol = identity[1]
        isotope = int(identity[2]) if len(identity) == 3 else None
        value, uncertainty = _number_with_uncertainty(columns[3])
        if value is None:
            continue
        key = symbol if isotope is None else f"{symbol}-{isotope}"
        row = NeutronRow(
            key,
            atomic_number,
            isotope,
            value,
            uncertainty,
            columns[6] == "E",
        )
        parsed.append(row)
        if isotope is None:
            natural_symbols.add(symbol)
        else:
            first_isotope.setdefault(symbol, row)
    aliases = [
        NeutronRow(
            symbol,
            source_row.atomic_number,
            None,
            source_row.b_c_fm,
            source_row.uncertainty_fm,
            source_row.energy_dependent,
            source_row.key,
        )
        for symbol, source_row in first_isotope.items()
        if symbol not in natural_symbols
    ]
    rows = sorted([*parsed, *aliases], key=lambda row: row.key)
    if len(rows) != 367 or len({row.key for row in rows}) != len(rows):
        raise ValueError(f"expected 367 unique usable neutron identities, found {len(rows)}")
    return rows


def _fnv_update(value: int, data: bytes) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _xray_digest(rows: list[XrayRow]) -> int:
    value = FNV_OFFSET
    for row in rows:
        value = _fnv_update(value, row.key.encode() + b"\0")
        value = _fnv_update(value, bytes([row.atomic_number]))
        for number in (*row.a, *row.b, row.c):
            value = _fnv_update(value, struct.pack("<d", number))
    return value


def _neutron_digest(rows: list[NeutronRow]) -> int:
    value = FNV_OFFSET
    for row in rows:
        value = _fnv_update(value, row.key.encode() + b"\0")
        value = _fnv_update(value, bytes([row.atomic_number]))
        value = _fnv_update(value, struct.pack("<H", row.isotope or 0))
        value = _fnv_update(value, struct.pack("<d", row.b_c_fm))
        value = _fnv_update(value, bytes([row.uncertainty_fm is not None]))
        if row.uncertainty_fm is not None:
            value = _fnv_update(value, struct.pack("<d", row.uncertainty_fm))
        value = _fnv_update(value, bytes([row.energy_dependent]))
        value = _fnv_update(value, (row.derived_alias_of or "").encode() + b"\0")
    return value


def _float(value: float) -> str:
    result = repr(value)
    return result if any(character in result for character in ".eE") else f"{result}.0"


def _option_float(value: float | None) -> str:
    return "None" if value is None else f"Some({_float(value)})"


def _option_u16(value: int | None) -> str:
    return "None" if value is None else f"Some({value})"


def _option_text(value: str | None) -> str:
    return "None" if value is None else f'Some("{value}")'


def _render(xray: list[XrayRow], neutron: list[NeutronRow]) -> str:
    lines = [
        "//! Generated scattering data; do not edit by hand.",
        "//! See `docs/scattering-models.md` and `tools/generate_scattering_tables.py`.",
        "#![allow(clippy::approx_constant, clippy::unreadable_literal)]",
        "",
        "use super::{NeutronTableRow, XrayTableRow};",
        "",
        f'pub(super) const XRAY_SOURCE_COMMIT: &str = "{XRAY_COMMIT}";',
        "pub(super) const XRAY_SOURCE_SHA256: &str =",
        f'    "{XRAY_SHA256}";',
        f"pub(super) const XRAY_TABLE_FNV64: u64 = 0x{_xray_digest(xray):016x};",
        f'pub(super) const NEUTRON_SOURCE_COMMIT: &str = "{NEUTRON_COMMIT}";',
        "pub(super) const NEUTRON_SOURCE_SHA256: &str =",
        f'    "{NEUTRON_SHA256}";',
        f"pub(super) const NEUTRON_TABLE_FNV64: u64 = 0x{_neutron_digest(neutron):016x};",
        "",
        "pub(super) static XRAY_ROWS: &[XrayTableRow] = &[",
    ]
    for row in xray:
        a = ", ".join(_float(value) for value in row.a)
        b = ", ".join(_float(value) for value in row.b)
        lines.extend(
            [
                "    XrayTableRow {",
                f'        key: "{row.key}",',
                f"        atomic_number: {row.atomic_number},",
                f"        a: [{a}],",
                f"        b: [{b}],",
                f"        c: {_float(row.c)},",
                "    },",
            ]
        )
    lines.append("];")
    lines.append("")
    lines.append("pub(super) static NEUTRON_ROWS: &[NeutronTableRow] = &[")
    for row in neutron:
        lines.extend(
            [
                "    NeutronTableRow {",
                f'        key: "{row.key}",',
                f"        atomic_number: {row.atomic_number},",
                f"        isotope: {_option_u16(row.isotope)},",
                f"        b_c_fm: {_float(row.b_c_fm)},",
                f"        uncertainty_fm: {_option_float(row.uncertainty_fm)},",
                f"        energy_dependent: {str(row.energy_dependent).lower()},",
                f"        derived_alias_of: {_option_text(row.derived_alias_of)},",
                "    },",
            ]
        )
    lines.extend(["];"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xraydb", type=Path, required=True)
    parser.add_argument("--periodictable", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crates/rietveld-crystallography/src/scattering_data.rs"),
    )
    arguments = parser.parse_args()
    xray_source = _verify_source(
        arguments.xraydb,
        XRAY_COMMIT,
        "data_sources/waasmaeir_kirfel.dat",
        XRAY_SHA256,
    )
    neutron_source = _verify_source(
        arguments.periodictable,
        NEUTRON_COMMIT,
        "periodictable/nsf.py",
        NEUTRON_SHA256,
    )
    xray = _xray_rows(xray_source)
    neutron = _neutron_rows(neutron_source)
    arguments.output.write_text(_render(xray, neutron), encoding="utf-8")
    print(f"wrote {len(xray)} X-ray and {len(neutron)} neutron rows to {arguments.output}")


if __name__ == "__main__":
    main()
