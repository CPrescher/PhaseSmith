"""Pinned-oracle unit translations into public physical models."""

from __future__ import annotations

import numpy as np

from ..instrument import ConstantWavelengthInstrument
from ..sample import StephensOrthorhombicBroadening


def cw_instrument_from_gsas_centidegrees(
    *,
    wavelength_angstrom: float,
    u: float,
    v: float,
    w: float,
    x: float,
    y: float,
) -> ConstantWavelengthInstrument:
    """Convert GSAS-style U/V/W/X/Y coefficients to public degree units."""

    return ConstantWavelengthInstrument(
        wavelength_angstrom=float(wavelength_angstrom),
        u_deg2=float(u) / 10_000.0,
        v_deg2=float(v) / 10_000.0,
        w_deg2=float(w) / 10_000.0,
        x_deg=float(x) / 100.0,
        y_deg=float(y) / 100.0,
    )


def orthorhombic_stephens_from_gsasii(
    coefficients: tuple[float, float, float, float, float, float],
    lorentzian_fraction: float,
) -> StephensOrthorhombicBroadening:
    """Translate pinned GSAS-II generalized microstrain into physical units.

    GSAS-II stores the orthorhombic basis as ``h^4, k^4, l^4,
    3h^2k^2, 3h^2l^2, 3k^2l^2`` and evaluates its intermediate FWHM in
    centidegrees. PhaseSmith instead stores the variance of ``1/d^2`` in
    ångström⁻⁴ with unscaled mixed monomials. This adapter is deliberately in
    the optional oracle namespace; the production model has no GSAS-II
    convention dependency.
    """

    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("GSAS-II orthorhombic Stephens coefficients must be six finite values")
    scale = 1.0e-12 / (8.0 * np.log(2.0))
    physical = values * scale * np.asarray((1.0, 1.0, 1.0, 3.0, 3.0, 3.0))
    return StephensOrthorhombicBroadening(tuple(map(float, physical)), lorentzian_fraction)
