"""General file-format adapters that return parser-independent models.

Dataset-specific conversion helpers remain available from their named
submodules; they are not part of the general :mod:`phasesmith.io` namespace.
"""

from __future__ import annotations

from . import cif, powder, space_groups, tof_instrument
from .cif import CifBackend, CifReadLimits, CifReadResult, NativeCifBackend, read_cif
from .powder import (
    PowderData,
    PowderFormat,
    PowderReadLimits,
    TofPowderData,
    TofPowderFormat,
    read_powder_data,
    read_tof_powder_data,
)
from .space_groups import (
    SpaceGroupInfo,
    space_group_by_number,
    space_group_by_symbol,
    space_group_from_hall_symbol,
)
from .tof_instrument import (
    GsasTofInstrumentData,
    GsasTofInstrumentReadLimits,
    read_gsas_tof_instrument,
)

__all__ = [
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "GsasTofInstrumentData",
    "GsasTofInstrumentReadLimits",
    "NativeCifBackend",
    "PowderData",
    "PowderFormat",
    "PowderReadLimits",
    "SpaceGroupInfo",
    "TofPowderData",
    "TofPowderFormat",
    "cif",
    "powder",
    "read_cif",
    "read_gsas_tof_instrument",
    "read_powder_data",
    "read_tof_powder_data",
    "space_group_by_number",
    "space_group_by_symbol",
    "space_group_from_hall_symbol",
    "space_groups",
    "tof_instrument",
]
