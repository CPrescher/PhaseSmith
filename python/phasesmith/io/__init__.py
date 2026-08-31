"""General file-format adapters that return parser-independent models.

Dataset-specific conversion helpers remain available from their named
submodules; they are not part of the general :mod:`phasesmith.io` namespace.
"""

from __future__ import annotations

import importlib as _importlib
import warnings as _warnings

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

_DEPRECATED_EXPORTS = {
    "BATH_LTL_SAMPLES": ("phasesmith.io.bath_ltl", "BATH_LTL_SAMPLES"),
    "IUCR_SILICON_PHASES": ("phasesmith.io.iucr_silicon_standard", "IUCR_SILICON_PHASES"),
    "IUCR_SODIUM_CITRATE_SILICON_PHASES": (
        "phasesmith.io.iucr_sodium_citrate_silicon",
        "IUCR_SODIUM_CITRATE_SILICON_PHASES",
    ),
    "IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES": (
        "phasesmith.io.iucr_tripotassium_citrate_silicon",
        "IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES",
    ),
    "IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES": (
        "phasesmith.io.iucr_trirubidium_citrate_silicon",
        "IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES",
    ),
    "ROWLES_SAMPLES": ("phasesmith.io.topas", "ROWLES_SAMPLES"),
    "ROWLES_WEIGHED_WEIGHT_FRACTIONS": (
        "phasesmith.io.topas",
        "ROWLES_WEIGHED_WEIGHT_FRACTIONS",
    ),
    "RigakuAscPattern": ("phasesmith.io.bath_ltl", "RigakuAscPattern"),
    "convert_bath_ltl_bundle": ("phasesmith.io.bath_ltl", "convert_bath_ltl_bundle"),
    "convert_iucr_silicon_standard_bundle": (
        "phasesmith.io.iucr_silicon_standard",
        "convert_iucr_silicon_standard_bundle",
    ),
    "convert_iucr_sodium_citrate_silicon_bundle": (
        "phasesmith.io.iucr_sodium_citrate_silicon",
        "convert_iucr_sodium_citrate_silicon_bundle",
    ),
    "convert_iucr_tripotassium_citrate_silicon_bundle": (
        "phasesmith.io.iucr_tripotassium_citrate_silicon",
        "convert_iucr_tripotassium_citrate_silicon_bundle",
    ),
    "convert_iucr_trirubidium_citrate_silicon_bundle": (
        "phasesmith.io.iucr_trirubidium_citrate_silicon",
        "convert_iucr_trirubidium_citrate_silicon_bundle",
    ),
    "convert_rowles_topas_bundle": ("phasesmith.io.topas", "convert_rowles_topas_bundle"),
    "convert_xred_tio2_bundle": ("phasesmith.io.xred", "convert_xred_tio2_bundle"),
    "read_rigaku_asc_text": ("phasesmith.io.bath_ltl", "read_rigaku_asc_text"),
}


def __getattr__(name: str) -> object:
    """Resolve one-release-cycle compatibility aliases for moved adapters."""

    target = _DEPRECATED_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    _warnings.warn(
        f"phasesmith.io.{name} moved to {module_name}.{attribute_name}; "
        "the aggregate alias is deprecated and will be removed in 1.0",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(_importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value
