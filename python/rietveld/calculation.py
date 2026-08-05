"""Script-friendly stateless powder-profile calculations."""

from __future__ import annotations

from typing import Literal

from numpy.typing import ArrayLike

from .cw import accumulate_cw, accumulate_cw_contributions
from .extensions import PhysicsContext, ReflectionPhysicsProvider, evaluate_provider
from .instrument import ConstantWavelengthInstrument
from .phase import ReflectionGeometryBatch
from .results import AccumulationResult


def calculate_cw_pattern(
    x: ArrayLike,
    reflections: ReflectionGeometryBatch,
    instrument: ConstantWavelengthInstrument,
    *,
    physics: ReflectionPhysicsProvider | None = None,
    support_fwhm: float = 20.0,
    jacobian_layout: Literal["support", "dense"] = "support",
) -> AccumulationResult:
    """Calculate a monochromatic CW profile with an optional batch provider.

    The provider is called exactly once. Its arrays are then consumed by one
    native fused accumulation call; no Python callback occurs per reflection or
    per profile sample.
    """

    if physics is None:
        return accumulate_cw(
            x,
            reflections.two_theta_deg,
            reflections.base_integrated_intensity,
            instrument,
            support_fwhm=support_fwhm,
            jacobian_layout=jacobian_layout,
        )
    contribution = evaluate_provider(physics, PhysicsContext(reflections, instrument))
    return accumulate_cw_contributions(
        x,
        reflections.two_theta_deg,
        reflections.base_integrated_intensity,
        instrument,
        contribution,
        support_fwhm=support_fwhm,
        jacobian_layout=jacobian_layout,
    )
