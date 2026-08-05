"""Fast, validated powder-diffraction profile calculations."""

from ._api import (
    PARAMETER_ORDER,
    AccumulationResult,
    PatternDerivatives,
    ProfileResult,
    SupportJacobian,
    accumulate,
    profile,
)

__all__ = [
    "PARAMETER_ORDER",
    "AccumulationResult",
    "PatternDerivatives",
    "ProfileResult",
    "SupportJacobian",
    "accumulate",
    "profile",
]
