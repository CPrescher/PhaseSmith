"""Fast, validated powder-diffraction profile calculations."""

from ._api import PARAMETER_ORDER, AccumulationResult, ProfileResult, accumulate, profile

__all__ = [
    "PARAMETER_ORDER",
    "AccumulationResult",
    "ProfileResult",
    "accumulate",
    "profile",
]
