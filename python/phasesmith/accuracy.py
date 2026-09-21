"""Explicit numerical accuracy controls for structural CW calculations."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfileAccuracy:
    """Opt-in profile approximations; defaults retain established arithmetic.

    ``fast_fcj`` uses lower-order quadrature for small axial spans, retaining
    asymmetry and its derivatives. ``tail_area_tolerance`` bounds continuous
    discarded node-profile area and overrides ``support_fwhm`` when supplied.
    Neither setting bounds the error of fitted parameters or final Rwp.
    """

    fast_fcj: bool = False
    tail_area_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fast_fcj, bool):
            raise TypeError("fast_fcj must be boolean")
        value = self.tail_area_tolerance
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("tail_area_tolerance must be a real number or None")
            if not math.isfinite(value) or not 1e-8 <= value <= 0.1:
                raise ValueError("tail_area_tolerance must be finite and in [1e-8, 0.1]")

    def _apply(self, native: object) -> None:
        native.set_profile_accuracy(self.fast_fcj, self.tail_area_tolerance)
