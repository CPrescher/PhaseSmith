"""Optional file-format adapters that return parser-independent models."""

from .cif import CifBackend, CifReadLimits, CifReadResult, read_cif

__all__ = ["CifBackend", "CifReadLimits", "CifReadResult", "read_cif"]
