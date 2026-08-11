"""File-format adapters that return parser-independent models."""

from .bath_ltl import (
    BATH_LTL_SAMPLES,
    RigakuAscPattern,
    convert_bath_ltl_bundle,
    read_rigaku_asc_text,
)
from .cif import CifBackend, CifReadLimits, CifReadResult, NativeCifBackend, read_cif
from .powder import PowderData, PowderFormat, PowderReadLimits, read_powder_data
from .space_groups import SpaceGroupInfo, space_group_by_number, space_group_by_symbol
from .topas import (
    ROWLES_SAMPLES,
    ROWLES_WEIGHED_WEIGHT_FRACTIONS,
    convert_rowles_topas_bundle,
)
from .xred import convert_xred_tio2_bundle

__all__ = [
    "BATH_LTL_SAMPLES",
    "ROWLES_SAMPLES",
    "ROWLES_WEIGHED_WEIGHT_FRACTIONS",
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "NativeCifBackend",
    "PowderData",
    "PowderFormat",
    "PowderReadLimits",
    "RigakuAscPattern",
    "SpaceGroupInfo",
    "convert_bath_ltl_bundle",
    "convert_rowles_topas_bundle",
    "convert_xred_tio2_bundle",
    "read_cif",
    "read_powder_data",
    "read_rigaku_asc_text",
    "space_group_by_number",
    "space_group_by_symbol",
]
