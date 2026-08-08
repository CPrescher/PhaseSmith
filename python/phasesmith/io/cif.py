"""Backend-neutral, size-limited CIF import through the native Rust core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .. import _core
from ..structure import CrystalStructure, StructureDiagnostic, structure_from_record


@dataclass(frozen=True, slots=True)
class CifReadLimits:
    """Explicit resource limits applied before and after optional parsing."""

    max_bytes: int = 16 * 1024 * 1024
    max_blocks: int = 100
    max_loop_rows: int = 1_000_000
    max_atom_sites: int = 100_000

    def __post_init__(self) -> None:
        """Require positive limits."""

        if min(self.max_bytes, self.max_blocks, self.max_loop_rows, self.max_atom_sites) <= 0:
            raise ValueError("all CIF read limits must be positive")


@dataclass(frozen=True, slots=True)
class CifReadResult:
    """One parser-independent structure plus visible import diagnostics."""

    structure: CrystalStructure
    diagnostics: tuple[StructureDiagnostic, ...]
    selected_block: str
    available_blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze and validate result records."""

        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "available_blocks", tuple(self.available_blocks))
        if not isinstance(self.structure, CrystalStructure):
            raise TypeError("CIF result structure must be a CrystalStructure")
        if any(not isinstance(item, StructureDiagnostic) for item in self.diagnostics):
            raise TypeError("CIF diagnostics must contain StructureDiagnostic values")
        if not self.selected_block or self.selected_block not in self.available_blocks:
            raise ValueError("selected CIF block must be listed in available_blocks")


@runtime_checkable
class CifBackend(Protocol):
    """Replaceable vector-free CIF parser adapter contract."""

    name: str
    version: str

    def parse_text(
        self,
        text: str,
        *,
        source_name: str | None,
        block: str | None,
        strict: bool,
        limits: CifReadLimits,
    ) -> CifReadResult:
        """Parse text and return only public typed models and diagnostics."""


class NativeCifBackend:
    """Default pure-Rust CIF and space-group adapter."""

    name = _core.NATIVE_CIF_BACKEND
    version = _core.NATIVE_CIF_BACKEND_VERSION

    def parse_text(
        self,
        text: str,
        *,
        source_name: str | None,
        block: str | None,
        strict: bool,
        limits: CifReadLimits,
    ) -> CifReadResult:
        """Parse bounded text and reconstruct the stable Python structure model."""

        record, selected_block, available_blocks = _core._parse_cif_text(
            text,
            source_name,
            block,
            strict,
            limits.max_bytes,
            limits.max_blocks,
            limits.max_loop_rows,
            limits.max_atom_sites,
        )
        structure = structure_from_record(record)
        return CifReadResult(
            structure,
            structure.diagnostics,
            selected_block,
            tuple(available_blocks),
        )


def read_cif(
    path_or_text: str | Path,
    *,
    block: str | None = None,
    strict: bool = True,
    limits: CifReadLimits | None = None,
    backend: CifBackend | None = None,
) -> CifReadResult:
    """Read CIF text or a local path through the native or an injected backend.

    A ``Path`` is always treated as a path. A string containing a newline or
    beginning with ``data_`` is treated as CIF text; other strings are treated
    as local paths when they exist and as text otherwise.
    """

    selected_limits = limits or CifReadLimits()
    if not isinstance(selected_limits, CifReadLimits):
        raise TypeError("limits must be CifReadLimits")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool")
    if block is not None and not isinstance(block, str):
        raise TypeError("block must be a string or None")
    text, source_name = _read_source(path_or_text, selected_limits.max_bytes)
    selected_backend = backend or _load_default_backend()
    if not isinstance(selected_backend, CifBackend):
        raise TypeError("backend must implement the CifBackend protocol")
    return selected_backend.parse_text(
        text,
        source_name=source_name,
        block=block,
        strict=strict,
        limits=selected_limits,
    )


def _read_source(path_or_text: str | Path, max_bytes: int) -> tuple[str, str | None]:
    if isinstance(path_or_text, Path):
        return _read_path(path_or_text, max_bytes)
    if not isinstance(path_or_text, str):
        raise TypeError("path_or_text must be a string or pathlib.Path")
    stripped = path_or_text.lstrip()
    if "\n" in path_or_text or "\r" in path_or_text or stripped.lower().startswith("data_"):
        encoded = path_or_text.encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError("CIF text exceeds max_bytes")
        return path_or_text, None
    candidate = Path(path_or_text)
    try:
        if candidate.exists():
            return _read_path(candidate, max_bytes)
    except OSError:
        pass
    encoded = path_or_text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("CIF text exceeds max_bytes")
    return path_or_text, None


def _read_path(path: Path, max_bytes: int) -> tuple[str, str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"CIF file exceeds max_bytes: {size} > {max_bytes}")
    return path.read_text(encoding="utf-8"), str(path)


def _load_default_backend() -> CifBackend:
    return NativeCifBackend()
