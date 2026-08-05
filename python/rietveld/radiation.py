"""Typed wavelength-component models for constant-wavelength radiation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .instrument import ConstantWavelengthInstrument


class RadiationProbe(StrEnum):
    """Physical probe used by a radiation model."""

    X_RAY = "x-ray"
    NEUTRON = "neutron"


@dataclass(frozen=True, slots=True)
class MonochromaticRadiation:
    """One explicitly typed constant wavelength with no spectral components."""

    probe: RadiationProbe
    wavelength_angstrom: float

    def __post_init__(self) -> None:
        """Validate the explicit probe and positive wavelength."""

        if not isinstance(self.probe, RadiationProbe):
            raise TypeError("probe must be a RadiationProbe")
        if not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive and finite")

    @classmethod
    def x_ray(cls, wavelength_angstrom: float) -> MonochromaticRadiation:
        """Construct monochromatic X-ray radiation."""

        return cls(RadiationProbe.X_RAY, wavelength_angstrom)

    @classmethod
    def neutron(cls, wavelength_angstrom: float) -> MonochromaticRadiation:
        """Construct monochromatic neutron radiation."""

        return cls(RadiationProbe.NEUTRON, wavelength_angstrom)


@dataclass(frozen=True, slots=True)
class ConstantWavelengthExperiment:
    """Typed pairing of a monochromatic probe and CW profile instrument."""

    radiation: MonochromaticRadiation
    instrument: ConstantWavelengthInstrument

    def __post_init__(self) -> None:
        """Reject ambiguous or inconsistent wavelength ownership."""

        if not isinstance(self.radiation, MonochromaticRadiation):
            raise TypeError("radiation must be MonochromaticRadiation")
        if not isinstance(self.instrument, ConstantWavelengthInstrument):
            raise TypeError("instrument must be ConstantWavelengthInstrument")
        if self.radiation.wavelength_angstrom != self.instrument.wavelength_angstrom:
            raise ValueError("radiation and instrument wavelengths must match exactly")

    @classmethod
    def x_ray(cls, instrument: ConstantWavelengthInstrument) -> ConstantWavelengthExperiment:
        """Construct an explicitly monochromatic X-ray experiment."""

        return cls(MonochromaticRadiation.x_ray(instrument.wavelength_angstrom), instrument)

    @classmethod
    def neutron(cls, instrument: ConstantWavelengthInstrument) -> ConstantWavelengthExperiment:
        """Construct an explicitly monochromatic neutron experiment."""

        return cls(MonochromaticRadiation.neutron(instrument.wavelength_angstrom), instrument)


@dataclass(frozen=True, slots=True, init=False)
class WavelengthComponents:
    """Discrete wavelengths and relative integrated intensities.

    Component zero is the reference wavelength and must have positive relative
    intensity. Intensities are normalized to unit sum for calculation, so their
    common scale is irrelevant. Secondary refinement parameters are wavelength
    ratios and intensity ratios relative to component zero.

    A one-component model is exactly monochromatic. Multiple components are
    optional and can represent K-alpha1/K-alpha2 or another discrete spectrum.
    """

    wavelengths_angstrom: NDArray[np.float64]
    relative_intensities: NDArray[np.float64]

    def __init__(
        self,
        wavelengths_angstrom: ArrayLike,
        relative_intensities: ArrayLike,
    ) -> None:
        """Copy and validate component arrays into immutable contiguous storage."""

        wavelengths = np.array(wavelengths_angstrom, dtype=np.float64, copy=True, order="C")
        intensities = np.array(relative_intensities, dtype=np.float64, copy=True, order="C")
        if wavelengths.ndim != 1 or intensities.ndim != 1:
            raise ValueError("wavelength component arrays must be one-dimensional")
        if wavelengths.size != intensities.size:
            raise ValueError("wavelength and relative-intensity arrays must have equal length")
        if wavelengths.size == 0:
            raise ValueError("at least one wavelength component is required")
        if not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0.0):
            raise ValueError("component wavelengths must be positive and finite")
        if not np.isfinite(intensities).all() or np.any(intensities < 0.0):
            raise ValueError("component relative intensities must be non-negative and finite")
        if intensities[0] <= 0.0:
            raise ValueError("reference component relative intensity must be positive")
        wavelengths.flags.writeable = False
        intensities.flags.writeable = False
        object.__setattr__(self, "wavelengths_angstrom", wavelengths)
        object.__setattr__(self, "relative_intensities", intensities)

    @classmethod
    def monochromatic(cls, wavelength_angstrom: float) -> WavelengthComponents:
        """Create an explicit one-component monochromatic model."""

        return cls([wavelength_angstrom], [1.0])

    @classmethod
    def doublet(
        cls,
        reference_wavelength_angstrom: float,
        secondary_wavelength_angstrom: float,
        secondary_to_reference_intensity: float,
    ) -> WavelengthComponents:
        """Create a two-component spectrum such as K-alpha1/K-alpha2."""

        return cls(
            [reference_wavelength_angstrom, secondary_wavelength_angstrom],
            [1.0, secondary_to_reference_intensity],
        )

    @property
    def component_count(self) -> int:
        """Return the number of discrete wavelengths."""

        return int(self.wavelengths_angstrom.size)

    @property
    def normalized_intensities(self) -> NDArray[np.float64]:
        """Return newly allocated unit-sum component weights."""

        scale = float(np.max(self.relative_intensities))
        scaled = self.relative_intensities / scale
        return scaled / np.sum(scaled)

    @property
    def wavelength_ratios(self) -> NDArray[np.float64]:
        """Return wavelengths divided by component-zero wavelength."""

        return self.wavelengths_angstrom / self.wavelengths_angstrom[0]

    @property
    def intensity_ratios(self) -> NDArray[np.float64]:
        """Return relative intensities divided by component-zero intensity."""

        return self.relative_intensities / self.relative_intensities[0]


def component_global_parameter_names(
    components: WavelengthComponents,
    *,
    include_fcj: bool,
) -> tuple[str, ...]:
    """Return stable shared-Jacobian row names for a component calculation."""

    names = ["u", "v", "w", "x", "y"]
    if include_fcj:
        names.extend(("sample_over_radius", "detector_over_radius"))
    names.extend(
        f"wavelength_ratio[{component}]" for component in range(1, components.component_count)
    )
    names.extend(
        f"intensity_ratio[{component}]" for component in range(1, components.component_count)
    )
    return tuple(names)
