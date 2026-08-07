"""Optional file-format adapters that return parser-independent models."""

from .cif import CifBackend, CifReadLimits, CifReadResult, read_cif
from .powder import PowderData, PowderFormat, PowderReadLimits, read_powder_data
from .space_groups import SpaceGroupInfo, space_group_by_number, space_group_by_symbol

__all__ = [
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "PowderData",
    "PowderFormat",
    "PowderReadLimits",
    "SpaceGroupInfo",
    "read_cif",
    "read_powder_data",
    "space_group_by_number",
    "space_group_by_symbol",
]
