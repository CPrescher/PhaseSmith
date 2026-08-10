"""File-format adapters that return parser-independent models."""

from .cif import CifBackend, CifReadLimits, CifReadResult, NativeCifBackend, read_cif
from .powder import PowderData, PowderFormat, PowderReadLimits, read_powder_data
from .space_groups import SpaceGroupInfo, space_group_by_number, space_group_by_symbol
from .topas import (
    ROWLES_SAMPLES,
    ROWLES_WEIGHED_WEIGHT_FRACTIONS,
    convert_rowles_topas_bundle,
)

__all__ = [
    "ROWLES_SAMPLES",
    "ROWLES_WEIGHED_WEIGHT_FRACTIONS",
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "NativeCifBackend",
    "PowderData",
    "PowderFormat",
    "PowderReadLimits",
    "SpaceGroupInfo",
    "convert_rowles_topas_bundle",
    "read_cif",
    "read_powder_data",
    "space_group_by_number",
    "space_group_by_symbol",
]
