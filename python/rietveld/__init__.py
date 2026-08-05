"""Fast, validated powder-diffraction profile calculations."""

from ._api import (
    PARAMETER_ORDER,
    TCH_PARAMETER_ORDER,
    ProfileResult,
    TchFwhmShape,
    TchProfileResult,
    TchSigmaProfileResult,
    TchSigmaShape,
    accumulate,
    accumulate_tch,
    profile,
    profile_tch,
    profile_tch_from_gaussian_sigma,
    tch_shape_from_fwhm,
    tch_shape_from_gaussian_sigma,
)
from .cw import (
    CW_GLOBAL_PARAMETER_ORDER,
    CW_LOCAL_PARAMETER_ORDER,
    CwProfileParameters,
    accumulate_cw,
    cw_profile_parameters,
)
from .fcj import (
    CW_FCJ_GLOBAL_PARAMETER_ORDER,
    CW_FCJ_LOCAL_PARAMETER_ORDER,
    FCJ_PARAMETER_ORDER,
    FcjProfileResult,
    accumulate_cw_fcj,
    profile_fcj,
)
from .instrument import ConstantWavelengthInstrument, FcjGeometry
from .results import AccumulationResult, PatternDerivatives, SupportJacobian

__all__ = [
    "CW_FCJ_GLOBAL_PARAMETER_ORDER",
    "CW_FCJ_LOCAL_PARAMETER_ORDER",
    "CW_GLOBAL_PARAMETER_ORDER",
    "CW_LOCAL_PARAMETER_ORDER",
    "FCJ_PARAMETER_ORDER",
    "PARAMETER_ORDER",
    "TCH_PARAMETER_ORDER",
    "AccumulationResult",
    "ConstantWavelengthInstrument",
    "CwProfileParameters",
    "FcjGeometry",
    "FcjProfileResult",
    "PatternDerivatives",
    "ProfileResult",
    "SupportJacobian",
    "TchFwhmShape",
    "TchProfileResult",
    "TchSigmaProfileResult",
    "TchSigmaShape",
    "accumulate",
    "accumulate_cw",
    "accumulate_cw_fcj",
    "accumulate_tch",
    "cw_profile_parameters",
    "profile",
    "profile_fcj",
    "profile_tch",
    "profile_tch_from_gaussian_sigma",
    "tch_shape_from_fwhm",
    "tch_shape_from_gaussian_sigma",
]
