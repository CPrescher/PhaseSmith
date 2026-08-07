"""Optional file-format adapters that return parser-independent models."""

from .cif import CifBackend, CifReadLimits, CifReadResult, read_cif
from .space_groups import SpaceGroupInfo, space_group_by_number, space_group_by_symbol

__all__ = [
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "SpaceGroupInfo",
    "read_cif",
    "space_group_by_number",
    "space_group_by_symbol",
]
