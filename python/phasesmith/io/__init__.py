"""File-format adapters that return parser-independent models."""

from .bath_ltl import (
    BATH_LTL_SAMPLES,
    RigakuAscPattern,
    convert_bath_ltl_bundle,
    read_rigaku_asc_text,
)
from .cif import CifBackend, CifReadLimits, CifReadResult, NativeCifBackend, read_cif
from .iucr_silicon_standard import (
    IUCR_SILICON_PHASES,
    convert_iucr_silicon_standard_bundle,
)
from .iucr_sodium_citrate_silicon import (
    IUCR_SODIUM_CITRATE_SILICON_PHASES,
    convert_iucr_sodium_citrate_silicon_bundle,
)
from .iucr_tripotassium_citrate_silicon import (
    IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES,
    convert_iucr_tripotassium_citrate_silicon_bundle,
)
from .powder import (
    PowderData,
    PowderFormat,
    PowderReadLimits,
    TofPowderData,
    TofPowderFormat,
    read_powder_data,
    read_tof_powder_data,
)
from .space_groups import SpaceGroupInfo, space_group_by_number, space_group_by_symbol
from .tof_instrument import (
    GsasTofInstrumentData,
    GsasTofInstrumentReadLimits,
    read_gsas_tof_instrument,
)
from .topas import (
    ROWLES_SAMPLES,
    ROWLES_WEIGHED_WEIGHT_FRACTIONS,
    convert_rowles_topas_bundle,
)
from .xred import convert_xred_tio2_bundle

__all__ = [
    "BATH_LTL_SAMPLES",
    "IUCR_SILICON_PHASES",
    "IUCR_SODIUM_CITRATE_SILICON_PHASES",
    "IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES",
    "ROWLES_SAMPLES",
    "ROWLES_WEIGHED_WEIGHT_FRACTIONS",
    "CifBackend",
    "CifReadLimits",
    "CifReadResult",
    "GsasTofInstrumentData",
    "GsasTofInstrumentReadLimits",
    "NativeCifBackend",
    "PowderData",
    "PowderFormat",
    "PowderReadLimits",
    "RigakuAscPattern",
    "SpaceGroupInfo",
    "TofPowderData",
    "TofPowderFormat",
    "convert_bath_ltl_bundle",
    "convert_iucr_silicon_standard_bundle",
    "convert_iucr_sodium_citrate_silicon_bundle",
    "convert_iucr_tripotassium_citrate_silicon_bundle",
    "convert_rowles_topas_bundle",
    "convert_xred_tio2_bundle",
    "read_cif",
    "read_gsas_tof_instrument",
    "read_powder_data",
    "read_rigaku_asc_text",
    "read_tof_powder_data",
    "space_group_by_number",
    "space_group_by_symbol",
]
