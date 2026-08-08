"""Typed built-in and third-party atomic scattering-factor providers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from .structure import AtomSite, CrystalStructure

SCATTERING_PROVIDER_API_VERSION = 1
_PROVIDER_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_ELEMENT = re.compile(r"^[A-Z][a-z]?$")


def _s_vector(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError("s_inverse_angstrom must be one-dimensional")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("s_inverse_angstrom must be finite and non-negative")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class ScatteringSpecies:
    """Probe-independent chemical identity plus an optional exact X-ray state."""

    element_symbol: str
    isotope: int | None = None
    charge: int | None = None
    xray_table_key: str | None = None

    def __post_init__(self) -> None:
        """Validate stable identity fields without guessing unavailable physics."""

        if not isinstance(self.element_symbol, str) or not _ELEMENT.fullmatch(self.element_symbol):
            raise ValueError("element_symbol must use canonical chemical-symbol capitalization")
        if self.isotope is not None and (
            not isinstance(self.isotope, int) or isinstance(self.isotope, bool) or self.isotope <= 0
        ):
            raise ValueError("isotope must be a positive integer or None")
        if self.charge is not None and (
            not isinstance(self.charge, int) or isinstance(self.charge, bool) or self.charge == 0
        ):
            raise ValueError("charge must be a non-zero integer or None")
        if self.xray_table_key is not None and (
            not isinstance(self.xray_table_key, str)
            or not self.xray_table_key
            or self.xray_table_key != self.xray_table_key.strip()
            or any(character.isspace() for character in self.xray_table_key)
        ):
            raise ValueError("xray_table_key must be a non-empty token or None")

    @property
    def xray_key(self) -> str:
        """Return the exact Waasmaier--Kirfel state requested by this identity."""

        if self.xray_table_key is not None:
            return self.xray_table_key
        if self.charge is None:
            return self.element_symbol
        sign = "+" if self.charge > 0 else "-"
        return f"{self.element_symbol}{abs(self.charge)}{sign}"

    @property
    def neutron_key(self) -> str:
        """Return the exact natural/isotope identity requested by this species."""

        if self.isotope is None:
            return self.element_symbol
        return f"{self.element_symbol}-{self.isotope}"

    @classmethod
    def from_atom_site(cls, site: AtomSite) -> ScatteringSpecies:
        """Preserve one CIF site's element/isotope/charge and specialist X-ray state."""

        if not isinstance(site, AtomSite):
            raise TypeError("site must be an AtomSite")
        ordinary = site.element_symbol
        if site.charge is not None:
            ordinary += f"{abs(site.charge)}{'+' if site.charge > 0 else '-'}"
        raw_without_isotope = re.sub(r"^\d+", "", site.type_symbol)
        special_key = None
        if site.type_symbol not in ("D", "T") and raw_without_isotope != ordinary:
            special_key = site.type_symbol
        return cls(site.element_symbol, site.isotope, site.charge, special_key)


def species_from_structure(structure: CrystalStructure) -> tuple[ScatteringSpecies, ...]:
    """Return one typed scattering identity per independent structure site."""

    if not isinstance(structure, CrystalStructure):
        raise TypeError("structure must be a CrystalStructure")
    return tuple(ScatteringSpecies.from_atom_site(site) for site in structure.sites)


@dataclass(frozen=True, slots=True)
class ScatteringProviderDescriptor:
    """Stable identity, probe, unit, and compatibility for one provider."""

    provider_id: str
    provider_version: str
    probe: Literal["xray", "neutron"]
    amplitude_unit: Literal["electrons", "fm"]
    api_version: int = SCATTERING_PROVIDER_API_VERSION
    thread_safe: bool = False

    def __post_init__(self) -> None:
        """Validate persistence-safe metadata."""

        if not isinstance(self.provider_id, str) or not _PROVIDER_ID.fullmatch(self.provider_id):
            raise ValueError("provider_id must be a lowercase dotted identifier")
        if (
            not isinstance(self.provider_version, str)
            or not self.provider_version
            or any(character.isspace() for character in self.provider_version)
        ):
            raise ValueError("provider_version must be a non-empty token")
        if self.probe not in ("xray", "neutron"):
            raise ValueError("probe must be xray or neutron")
        expected_unit = "electrons" if self.probe == "xray" else "fm"
        if self.amplitude_unit != expected_unit:
            raise ValueError(f"{self.probe} provider amplitude_unit must be {expected_unit}")
        if not isinstance(self.api_version, int) or self.api_version <= 0:
            raise ValueError("api_version must be a positive integer")
        if not isinstance(self.thread_safe, bool):
            raise TypeError("thread_safe must be a boolean")


@dataclass(frozen=True, slots=True)
class ScatteringContext:
    """Immutable species and scattering-vector inputs for one provider call."""

    species: tuple[ScatteringSpecies, ...]
    s_inverse_angstrom: NDArray[np.float64]

    def __init__(
        self,
        species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies],
        s_inverse_angstrom: ArrayLike,
    ) -> None:
        """Copy, validate, and freeze one complete reflection/species context."""

        identities = tuple(species)
        if any(not isinstance(value, ScatteringSpecies) for value in identities):
            raise TypeError("species must contain ScatteringSpecies values")
        object.__setattr__(self, "species", identities)
        object.__setattr__(self, "s_inverse_angstrom", _s_vector(s_inverse_angstrom))

    @property
    def reflection_count(self) -> int:
        """Return the number of scattering-vector rows."""

        return int(self.s_inverse_angstrom.size)


@dataclass(frozen=True, slots=True, init=False)
class ScatteringFactorBatch:
    """Immutable complex amplitudes and analytical `df/ds` batch."""

    amplitudes: NDArray[np.complex128]
    d_amplitudes_d_s: NDArray[np.complex128]
    descriptor: ScatteringProviderDescriptor

    def __init__(
        self,
        amplitudes: ArrayLike,
        d_amplitudes_d_s: ArrayLike,
        descriptor: ScatteringProviderDescriptor,
    ) -> None:
        """Copy and validate equal reflection-major complex matrices."""

        if not isinstance(descriptor, ScatteringProviderDescriptor):
            raise TypeError("descriptor must be a ScatteringProviderDescriptor")
        values = np.array(amplitudes, dtype=np.complex128, copy=True, order="C")
        derivatives = np.array(d_amplitudes_d_s, dtype=np.complex128, copy=True, order="C")
        if values.ndim != 2 or derivatives.shape != values.shape:
            raise ValueError("amplitudes and d_amplitudes_d_s must have the same 2-D shape")
        if not (
            np.isfinite(values.real).all()
            and np.isfinite(values.imag).all()
            and np.isfinite(derivatives.real).all()
            and np.isfinite(derivatives.imag).all()
        ):
            raise ValueError("scattering amplitudes and derivatives must be finite")
        values.flags.writeable = False
        derivatives.flags.writeable = False
        object.__setattr__(self, "amplitudes", values)
        object.__setattr__(self, "d_amplitudes_d_s", derivatives)
        object.__setattr__(self, "descriptor", descriptor)

    @property
    def reflection_count(self) -> int:
        """Return the number of reflection rows."""

        return int(self.amplitudes.shape[0])

    @property
    def site_count(self) -> int:
        """Return the number of site/species columns."""

        return int(self.amplitudes.shape[1])


@runtime_checkable
class ScatteringFactorProvider(Protocol):
    """Explicit batch interface for built-in or research scattering models."""

    @property
    def descriptor(self) -> ScatteringProviderDescriptor:
        """Return stable provider identity and compatibility metadata."""

    def evaluate(self, context: ScatteringContext) -> ScatteringFactorBatch:
        """Evaluate the complete reflection/species batch in one call."""


def evaluate_scattering_provider(
    provider: ScatteringFactorProvider,
    context: ScatteringContext,
) -> ScatteringFactorBatch:
    """Validate provider compatibility, output type, shape, and identity."""

    if not isinstance(context, ScatteringContext):
        raise TypeError("context must be a ScatteringContext")
    descriptor = provider.descriptor
    if not isinstance(descriptor, ScatteringProviderDescriptor):
        raise TypeError("provider.descriptor must be a ScatteringProviderDescriptor")
    if descriptor.api_version != SCATTERING_PROVIDER_API_VERSION:
        raise ValueError(
            f"provider API {descriptor.api_version} is incompatible with "
            f"API {SCATTERING_PROVIDER_API_VERSION}"
        )
    result = provider.evaluate(context)
    if not isinstance(result, ScatteringFactorBatch):
        raise TypeError("provider.evaluate() must return ScatteringFactorBatch")
    if result.descriptor != descriptor:
        raise ValueError("provider result descriptor does not match the provider")
    if result.amplitudes.shape != (context.reflection_count, len(context.species)):
        raise ValueError("provider result shape does not match its context")
    return result


XRAY_NON_RESONANT_DESCRIPTOR = ScatteringProviderDescriptor(
    "phasesmith.xray.waasmaier_kirfel",
    "1995+xraydb-663d2171",
    "xray",
    "electrons",
    thread_safe=True,
)
NEUTRON_NUCLEAR_DESCRIPTOR = ScatteringProviderDescriptor(
    "phasesmith.neutron.bound_coherent",
    "periodictable-182ef63a",
    "neutron",
    "fm",
    thread_safe=True,
)
XRAY_FIXED_DISPERSION_DESCRIPTOR = ScatteringProviderDescriptor(
    "phasesmith.xray.fixed_dispersion",
    "1",
    "xray",
    "electrons",
    thread_safe=True,
)


class PreparedXrayNonResonant:
    """Exact species cache for repeated non-resonant X-ray evaluation."""

    __slots__ = ("_native", "_species")
    descriptor = XRAY_NON_RESONANT_DESCRIPTOR

    def __init__(self, species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies]) -> None:
        """Resolve exact state keys once in the native table."""

        identities = tuple(species)
        if any(not isinstance(value, ScatteringSpecies) for value in identities):
            raise TypeError("species must contain ScatteringSpecies values")
        self._species = identities
        self._native = _core._PreparedXrayScattering([value.xray_key for value in identities])

    @property
    def species(self) -> tuple[ScatteringSpecies, ...]:
        """Return the immutable species order represented by native columns."""

        return self._species

    @property
    def unique_species_count(self) -> int:
        """Return the number of distinct native table rows evaluated."""

        return int(self._native.unique_species_count)

    def evaluate(self, s_inverse_angstrom: ArrayLike) -> ScatteringFactorBatch:
        """Evaluate all reflections with the cached exact species rows."""

        s = _s_vector(s_inverse_angstrom)
        real, imag, d_real, d_imag = self._native.evaluate(s)
        return ScatteringFactorBatch(real + 1j * imag, d_real + 1j * d_imag, self.descriptor)


class PreparedNeutronNuclear:
    """Exact species cache for repeated constant coherent neutron evaluation."""

    __slots__ = ("_native", "_species")
    descriptor = NEUTRON_NUCLEAR_DESCRIPTOR

    def __init__(self, species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies]) -> None:
        """Resolve natural/isotope keys and reject energy-dependent rows."""

        identities = tuple(species)
        if any(not isinstance(value, ScatteringSpecies) for value in identities):
            raise TypeError("species must contain ScatteringSpecies values")
        self._species = identities
        self._native = _core._PreparedNeutronScattering([value.neutron_key for value in identities])

    @property
    def species(self) -> tuple[ScatteringSpecies, ...]:
        """Return the immutable species order represented by native columns."""

        return self._species

    @property
    def unique_species_count(self) -> int:
        """Return the number of distinct native table rows copied."""

        return int(self._native.unique_species_count)

    def evaluate(self, s_inverse_angstrom: ArrayLike) -> ScatteringFactorBatch:
        """Evaluate all reflections with the cached exact species rows."""

        s = _s_vector(s_inverse_angstrom)
        real, imag, d_real, d_imag = self._native.evaluate(s)
        return ScatteringFactorBatch(real + 1j * imag, d_real + 1j * d_imag, self.descriptor)


@dataclass(frozen=True, slots=True)
class XrayNonResonant:
    """Built-in non-resonant Waasmaier--Kirfel provider."""

    descriptor = XRAY_NON_RESONANT_DESCRIPTOR

    def prepare(
        self, species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies]
    ) -> PreparedXrayNonResonant:
        """Resolve exact species once for repeated batch evaluation."""

        return PreparedXrayNonResonant(tuple(species))

    def evaluate(self, context: ScatteringContext) -> ScatteringFactorBatch:
        """Evaluate one complete context through a temporary prepared cache."""

        return self.prepare(context.species).evaluate(context.s_inverse_angstrom)


@dataclass(frozen=True, slots=True, init=False)
class XrayFixedDispersion:
    """Non-resonant X-ray factors plus fixed element-wise ``f' + i f''``.

    Offsets are wavelength-specific input data; this model does not interpolate an
    absorption-edge database or hide an energy convention. The constant offsets do
    not change the analytical derivative with respect to scattering vector ``s``.
    """

    corrections: tuple[tuple[str, complex], ...]
    descriptor = XRAY_FIXED_DISPERSION_DESCRIPTOR

    def __init__(self, corrections: dict[str, complex]) -> None:
        """Copy and validate one finite complex correction per element."""

        if not isinstance(corrections, dict) or not corrections:
            raise ValueError("corrections must be a non-empty element-to-complex mapping")
        normalized: list[tuple[str, complex]] = []
        for element, value in corrections.items():
            if not isinstance(element, str) or not _ELEMENT.fullmatch(element):
                raise ValueError("dispersion correction keys must be canonical element symbols")
            correction = complex(value)
            if not np.isfinite(correction.real) or not np.isfinite(correction.imag):
                raise ValueError("dispersion corrections must be finite")
            normalized.append((element, correction))
        object.__setattr__(self, "corrections", tuple(sorted(normalized)))

    def corrections_for(
        self, species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies]
    ) -> NDArray[np.complex128]:
        """Return one immutable fixed correction for each requested species."""

        identities = tuple(species)
        if any(not isinstance(value, ScatteringSpecies) for value in identities):
            raise TypeError("species must contain ScatteringSpecies values")
        by_element = dict(self.corrections)
        missing = sorted({value.element_symbol for value in identities} - by_element.keys())
        if missing:
            raise ValueError(f"missing fixed dispersion corrections for {', '.join(missing)}")
        result = np.ascontiguousarray(
            [by_element[value.element_symbol] for value in identities], dtype=np.complex128
        )
        result.flags.writeable = False
        return result

    def evaluate(self, context: ScatteringContext) -> ScatteringFactorBatch:
        """Evaluate non-resonant factors and add the fixed complex offsets."""

        if not isinstance(context, ScatteringContext):
            raise TypeError("context must be a ScatteringContext")
        baseline = XrayNonResonant().evaluate(context)
        amplitudes = baseline.amplitudes + self.corrections_for(context.species)[None, :]
        return ScatteringFactorBatch(
            amplitudes,
            baseline.d_amplitudes_d_s,
            self.descriptor,
        )


@dataclass(frozen=True, slots=True)
class NeutronNuclear:
    """Built-in constant real bound coherent nuclear provider."""

    descriptor = NEUTRON_NUCLEAR_DESCRIPTOR

    def prepare(
        self, species: tuple[ScatteringSpecies, ...] | list[ScatteringSpecies]
    ) -> PreparedNeutronNuclear:
        """Resolve exact natural/isotope species once for repeated evaluation."""

        return PreparedNeutronNuclear(tuple(species))

    def evaluate(self, context: ScatteringContext) -> ScatteringFactorBatch:
        """Evaluate one complete context through a temporary prepared cache."""

        return self.prepare(context.species).evaluate(context.s_inverse_angstrom)


@dataclass(frozen=True, slots=True)
class ScatteringTableProvenance:
    """Native generated-table source and integrity metadata."""

    name: str
    upstream_commit: str
    source_sha256: str
    table_fnv64: int
    row_count: int


@dataclass(frozen=True, slots=True)
class XraySpeciesMetadata:
    """Metadata for one exact Waasmaier--Kirfel state key."""

    key: str
    atomic_number: int


@dataclass(frozen=True, slots=True)
class NeutronSpeciesMetadata:
    """Metadata for one exact bound coherent neutron identity."""

    key: str
    atomic_number: int
    isotope: int | None
    b_c_fm: float
    uncertainty_fm: float | None
    energy_dependent: bool
    derived_alias_of: str | None


def xray_species_metadata(key: str) -> XraySpeciesMetadata | None:
    """Inspect one exact native X-ray state without evaluating a batch."""

    record = _core.xray_scattering_species_metadata(key)
    return None if record is None else XraySpeciesMetadata(*record)


def neutron_species_metadata(key: str) -> NeutronSpeciesMetadata | None:
    """Inspect one exact native neutron identity without evaluating a batch."""

    record = _core.neutron_scattering_species_metadata(key)
    return None if record is None else NeutronSpeciesMetadata(*record)


_XRAY_PROVENANCE, _NEUTRON_PROVENANCE = _core.scattering_table_provenance()
XRAY_TABLE_PROVENANCE = ScatteringTableProvenance(*_XRAY_PROVENANCE)
NEUTRON_TABLE_PROVENANCE = ScatteringTableProvenance(*_NEUTRON_PROVENANCE)
