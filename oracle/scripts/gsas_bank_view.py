"""Prepare deterministic one-bank views of legacy multi-bank GSAS powder data."""

from __future__ import annotations

from pathlib import Path


def write_gsas_bank_view(source: Path, bank: int, destination: Path) -> None:
    """Write one numbered ``BANK`` block while preserving the file preamble."""

    lines = source.read_text(encoding="latin-1").splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("BANK")]
    if not starts:
        raise RuntimeError(f"no BANK records in {source}")
    blocks: dict[int, list[str]] = {}
    for position, start in enumerate(starts):
        fields = lines[start].split()
        if len(fields) < 2:
            raise RuntimeError(f"malformed BANK record in {source}: {lines[start]!r}")
        number = int(fields[1])
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        if number in blocks:
            raise RuntimeError(f"duplicate BANK {number} in {source}")
        blocks[number] = lines[start:end]
    if bank not in blocks:
        raise RuntimeError(f"BANK {bank} is absent from {source}")
    destination.write_text("".join(lines[: starts[0]] + blocks[bank]), encoding="latin-1")
