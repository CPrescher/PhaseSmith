"""Explicit integrated-reflection intensity correction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core


def _q_squared(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(
            "q_squared_inverse_angstrom2 must be finite, positive, and one-dimensional"
        )
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, init=False)
class IntegratedIntensityCorrection:
    """Immutable correction values and analytical ``dC/d(q²)``."""

    values: NDArray[np.float64]
    d_values_d_q_squared: NDArray[np.float64]
    model_id: str

    def __init__(
        self,
        values: ArrayLike,
        d_values_d_q_squared: ArrayLike,
        model_id: str,
    ) -> None:
        """Copy and validate one correction row per reflection."""

        correction = np.array(values, dtype=np.float64, copy=True, order="C")
        derivative = np.array(d_values_d_q_squared, dtype=np.float64, copy=True, order="C")
        if (
            correction.ndim != 1
            or derivative.shape != correction.shape
            or not np.isfinite(correction).all()
            or not np.isfinite(derivative).all()
            or np.any(correction < 0.0)
        ):
            raise ValueError("correction values and derivatives must be finite matching vectors")
        if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
            raise ValueError("model_id must be a non-empty trimmed string")
        correction.flags.writeable = False
        derivative.flags.writeable = False
        object.__setattr__(self, "values", correction)
        object.__setattr__(self, "d_values_d_q_squared", derivative)
        object.__setattr__(self, "model_id", model_id)


@runtime_checkable
class IntegratedIntensityCorrectionProvider(Protocol):
    """Batch-only correction interface; providers are passed explicitly."""

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike) -> IntegratedIntensityCorrection:
        """Evaluate one complete reflection batch."""


@dataclass(frozen=True, slots=True)
class NeutralIntegratedIntensityCorrection:
    """Raw multiplicity-weighted structural intensity with ``C_h = 1``."""

    thread_safe: ClassVar[bool] = True

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike) -> IntegratedIntensityCorrection:
        """Return exact neutral values and zero derivatives."""

        q_squared = _q_squared(q_squared_inverse_angstrom2)
        values, derivatives = _core.integrated_intensity_correction(q_squared, "neutral", None)
        return IntegratedIntensityCorrection(values, derivatives, "neutral")


@dataclass(frozen=True, slots=True)
class ConstantWavelengthNeutronLorentz:
    """Monochromatic neutron powder Lorentz factor ``1/(sinθ sin2θ)``."""

    wavelength_angstrom: float
    thread_safe: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive and finite")

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike) -> IntegratedIntensityCorrection:
        """Return Lorentz values and analytical reciprocal-metric derivatives."""

        q_squared = _q_squared(q_squared_inverse_angstrom2)
        values, derivatives = _core.integrated_intensity_correction(
            q_squared,
            "constant_wavelength_neutron_lorentz",
            self.wavelength_angstrom,
        )
        return IntegratedIntensityCorrection(
            values,
            derivatives,
            "constant_wavelength_neutron_lorentz",
        )


@dataclass(frozen=True, slots=True)
class BraggBrentanoUnpolarizedLp:
    """Monochromatic unpolarized symmetric Bragg--Brentano integrated LP."""

    wavelength_angstrom: float
    thread_safe: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Require a positive finite monochromatic wavelength."""

        if not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive and finite")

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike) -> IntegratedIntensityCorrection:
        """Return integrated LP values and analytical metric derivatives."""

        q_squared = _q_squared(q_squared_inverse_angstrom2)
        values, derivatives = _core.integrated_intensity_correction(
            q_squared,
            "bragg_brentano_unpolarized_lp",
            self.wavelength_angstrom,
        )
        return IntegratedIntensityCorrection(
            values,
            derivatives,
            "bragg_brentano_unpolarized_lp",
        )


@dataclass(frozen=True, slots=True)
class BraggBrentanoPolarizedLp:
    """Polarized symmetric Bragg--Brentano integrated Lorentz--polarization.

    ``polarization`` is the fraction in the constant term of
    ``[P + (1-P) cos²(2θ)] / [sin²(θ) cos(θ)]``. Therefore ``P=0.5`` is
    exactly the unpolarized model.
    """

    wavelength_angstrom: float
    polarization: float
    thread_safe: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Require a physical wavelength and polarization fraction."""

        if not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive and finite")
        if not np.isfinite(self.polarization) or not 0.0 <= self.polarization <= 1.0:
            raise ValueError("polarization must be finite and within [0, 1]")

    def evaluate(self, q_squared_inverse_angstrom2: ArrayLike) -> IntegratedIntensityCorrection:
        """Return polarized LP values and analytical metric derivatives."""

        q_squared = _q_squared(q_squared_inverse_angstrom2)
        values, derivatives = _core.integrated_intensity_correction(
            q_squared,
            "bragg_brentano_polarized_lp",
            self.wavelength_angstrom,
            self.polarization,
        )
        return IntegratedIntensityCorrection(
            values,
            derivatives,
            "bragg_brentano_polarized_lp",
        )


def evaluate_intensity_correction(
    provider: IntegratedIntensityCorrectionProvider,
    q_squared_inverse_angstrom2: ArrayLike,
) -> IntegratedIntensityCorrection:
    """Validate one complete custom or built-in correction-provider call."""

    q_squared = _q_squared(q_squared_inverse_angstrom2)
    result = provider.evaluate(q_squared)
    if not isinstance(result, IntegratedIntensityCorrection):
        raise TypeError("correction provider must return IntegratedIntensityCorrection")
    if result.values.shape != q_squared.shape:
        raise ValueError("correction result shape must match q_squared")
    return result
