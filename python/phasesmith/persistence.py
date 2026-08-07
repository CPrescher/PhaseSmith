"""Versioned JSON+NPZ persistence for public calculation and Le Bail models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .calculation import CalculationOptions
from .extensions import CompositePhysicsProvider, ReflectionPhysicsProvider
from .instrument import ConstantWavelengthInstrument, FcjGeometry, TofInstrument
from .intensity_corrections import (
    BraggBrentanoPolarizedLp,
    BraggBrentanoUnpolarizedLp,
    NeutralIntegratedIntensityCorrection,
)
from .pattern import (
    PatternCalculationResult,
    PhasePatternComponent,
    PowderPattern,
)
from .phase import (
    Phase,
    ReciprocalMetric,
    ReflectionBatch,
    RietveldPhase,
    StructuralReflectionBatch,
)
from .radiation import (
    BraggBrentanoGeometry,
    ComponentRadiation,
    ConstantWavelengthExperiment,
    MonochromaticRadiation,
    RadiationProbe,
    WavelengthComponents,
)
from .refinement.background import (
    AmorphousBackground,
    AmorphousPeak,
    ChebyshevBackground,
    CompositeBackground,
    DifferentiableBackground,
    PointBackground,
    PolynomialBackground,
)
from .refinement.core import (
    AffineConstraint,
    Bounds,
    Constraint,
    FixedConstraint,
    LinearConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    TerminationReason,
)
from .refinement.lattice import (
    CwLatticeReflectionDomain,
    CwStructuralReflectionDomain,
    LatticeParameterBounds,
    LatticeParameterization,
)
from .refinement.lebail import (
    CoincidentReflectionGroup,
    IterationRecord,
    LeBailCheckpoint,
    LeBailInput,
    LeBailOptions,
    LeBailPhase,
    LeBailResult,
    ParameterChange,
    ReflectionIntensity,
)
from .refinement.rietveld import (
    RietveldCheckpoint,
    RietveldInput,
    RietveldIterationRecord,
    RietveldOptions,
    RietveldParameterChange,
    RietveldParameterSelection,
)
from .refinement.runtime import RefinementLimits
from .results import AccumulationResult, PatternDerivatives, SupportJacobian
from .sample import (
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
)
from .scattering import NeutronNuclear, XrayFixedDispersion, XrayNonResonant
from .structure import CrystalStructure, structure_from_record, structure_to_record

FORMAT_VERSION: Final = 7
MANIFEST_NAME: Final = "manifest.json"
ARCHIVE_NAME: Final = "arrays.npz"
Instrument = ConstantWavelengthInstrument | TofInstrument


class PersistenceError(ValueError):
    """Raised for incompatible, incomplete, or corrupt persisted data."""


@runtime_checkable
class PhysicsProviderCodec(Protocol):
    """Explicit serializer for one third-party physics provider family."""

    provider_id: str

    def encode(self, provider: ReflectionPhysicsProvider) -> dict[str, Any] | None:
        """Return plain configuration when this codec owns ``provider``."""

    def decode(
        self, configuration: dict[str, Any], provider_version: str
    ) -> ReflectionPhysicsProvider:
        """Reconstruct or migrate a provider from versioned plain configuration."""


@dataclass(frozen=True, slots=True)
class PersistenceBundle:
    """Top-level models that can be saved together without pickle."""

    pattern: PowderPattern | None = None
    instrument: Instrument | None = None
    experiment: ConstantWavelengthExperiment | None = None
    fcj_geometry: FcjGeometry | None = None
    wavelength_components: WavelengthComponents | None = None
    phases: tuple[Phase, ...] = ()
    rietveld_phases: tuple[RietveldPhase, ...] = ()
    rietveld_domains: tuple[CwStructuralReflectionDomain | None, ...] = ()
    rietveld_selection: RietveldParameterSelection | None = None
    rietveld_options: RietveldOptions | None = None
    rietveld_checkpoint: RietveldCheckpoint | None = None
    rietveld_background: DifferentiableBackground | None = None
    calculation_options: CalculationOptions | None = None
    calculation_result: PatternCalculationResult | None = None
    parameters: ParameterSet | None = None
    lebail_options: LeBailOptions | None = None
    lebail_checkpoint: LeBailCheckpoint | None = None
    lebail_result: LeBailResult | None = None
    constraints: tuple[Constraint, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Freeze sequences and require JSON-compatible user metadata."""

        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "rietveld_phases", tuple(self.rietveld_phases))
        domains = tuple(self.rietveld_domains)
        if not domains and self.rietveld_phases:
            domains = (None,) * len(self.rietveld_phases)
        object.__setattr__(self, "rietveld_domains", domains)
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if any(not isinstance(phase, Phase) for phase in self.phases):
            raise TypeError("phases must contain only Phase objects")
        if any(not isinstance(phase, RietveldPhase) for phase in self.rietveld_phases):
            raise TypeError("rietveld_phases must contain only RietveldPhase objects")
        if len(self.rietveld_domains) != len(self.rietveld_phases):
            raise ValueError("rietveld domains must align with persisted structural phases")
        if any(
            domain is not None and not isinstance(domain, CwStructuralReflectionDomain)
            for domain in self.rietveld_domains
        ):
            raise TypeError("rietveld_domains must contain guarded structural domains or None")
        if self.rietveld_selection is not None and not isinstance(
            self.rietveld_selection, RietveldParameterSelection
        ):
            raise TypeError("rietveld_selection must be RietveldParameterSelection")
        if self.rietveld_options is not None and not isinstance(
            self.rietveld_options, RietveldOptions
        ):
            raise TypeError("rietveld_options must be RietveldOptions")
        if self.rietveld_checkpoint is not None and not isinstance(
            self.rietveld_checkpoint, RietveldCheckpoint
        ):
            raise TypeError("rietveld_checkpoint must be RietveldCheckpoint")
        if self.rietveld_background is not None and not isinstance(
            self.rietveld_background, DifferentiableBackground
        ):
            raise TypeError("rietveld_background must implement DifferentiableBackground")
        if self.calculation_result is not None and not isinstance(
            self.calculation_result, PatternCalculationResult
        ):
            raise TypeError("calculation_result must be PatternCalculationResult")
        if self.parameters is not None and not isinstance(self.parameters, ParameterSet):
            raise TypeError("parameters must be ParameterSet")
        if self.lebail_options is not None and not isinstance(self.lebail_options, LeBailOptions):
            raise TypeError("lebail_options must be LeBailOptions")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON-compatible plain data") from error
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))
        if (
            self.experiment is not None
            and self.instrument is not None
            and self.experiment.instrument != self.instrument
        ):
            raise ValueError("experiment and top-level instrument must agree")

    def to_lebail_input(self) -> LeBailInput:
        """Reconstruct a complete Le Bail input from the top-level models."""

        if self.pattern is None:
            raise ValueError("a persisted pattern is required for Le Bail input")
        if not isinstance(self.instrument, ConstantWavelengthInstrument):
            raise ValueError("a persisted constant-wavelength instrument is required")
        if not self.phases:
            raise ValueError("persisted phases are required for Le Bail input")
        return LeBailInput(
            self.pattern,
            self.instrument,
            self.phases,
            self.parameters,
            self.constraints,
        )

    def to_rietveld_input(self) -> RietveldInput:
        """Reconstruct a complete structural input for calculation or resume."""

        if self.pattern is None or self.experiment is None:
            raise ValueError("persisted pattern and CW experiment are required")
        if not self.rietveld_phases or self.parameters is None:
            raise ValueError("persisted structural phases and parameters are required")
        if len(self.rietveld_domains) != len(self.rietveld_phases):
            raise ValueError("persisted structural domains must align with phases")
        return RietveldInput(
            self.pattern,
            self.experiment,
            self.rietveld_phases,
            self.rietveld_domains,
            self.parameters,
            self.constraints,
            (
                RietveldParameterSelection()
                if self.rietveld_selection is None
                else self.rietveld_selection
            ),
            self.rietveld_background,
        )


class _ArrayWriter:
    def __init__(self) -> None:
        self.arrays: dict[str, NDArray[np.generic]] = {}

    def add(self, name: str, value: NDArray[np.generic]) -> str:
        key = name
        suffix = 1
        while key in self.arrays:
            suffix += 1
            key = f"{name}_{suffix}"
        self.arrays[key] = np.ascontiguousarray(value)
        return key


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_descriptor(value: NDArray[np.generic]) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _bounds_record(bounds: Bounds) -> dict[str, float | None]:
    return {
        "lower": bounds.lower if np.isfinite(bounds.lower) else None,
        "upper": bounds.upper if np.isfinite(bounds.upper) else None,
    }


def _bounds_from_record(record: dict[str, Any]) -> Bounds:
    return Bounds(
        -np.inf if record["lower"] is None else float(record["lower"]),
        np.inf if record["upper"] is None else float(record["upper"]),
    )


def _key_record(key: ParameterKey) -> dict[str, str]:
    return {"module": key.module, "owner_id": key.owner_id, "name": key.name}


def _key_from_record(record: dict[str, Any]) -> ParameterKey:
    return ParameterKey(record["module"], record["owner_id"], record["name"])


def _ratio_record(value: float) -> float | None:
    """Represent an undefined positive-infinite ratio as JSON null."""

    if np.isfinite(value):
        return float(value)
    if np.isposinf(value):
        return None
    raise ValueError("persisted ratio metrics must be finite or positive infinity")


def _ratio_from_record(value: float | None) -> float:
    return np.inf if value is None else float(value)


def _parameters_record(parameters: ParameterSet | None) -> dict[str, Any] | None:
    if parameters is None:
        return None
    return {
        "specs": [
            {
                "key": _key_record(spec.key),
                "value": spec.value,
                "unit": spec.unit,
                "bounds": _bounds_record(spec.bounds),
                "scale": spec.scale,
                "refine": spec.refine,
            }
            for spec in parameters.specs
        ]
    }


def _parameters_from_record(record: dict[str, Any] | None) -> ParameterSet | None:
    if record is None:
        return None
    return ParameterSet(
        [
            ParameterSpec(
                _key_from_record(item["key"]),
                float(item["value"]),
                item["unit"],
                _bounds_from_record(item["bounds"]),
                float(item["scale"]),
                bool(item["refine"]),
            )
            for item in record["specs"]
        ]
    )


def _constraint_record(constraint: Constraint) -> dict[str, Any]:
    if isinstance(constraint, FixedConstraint):
        return {
            "type": "fixed",
            "target": _key_record(constraint.target),
            "value": constraint.value,
        }
    if isinstance(constraint, AffineConstraint):
        return {
            "type": "affine",
            "target": _key_record(constraint.target),
            "source": _key_record(constraint.source),
            "multiplier": constraint.multiplier,
            "offset": constraint.offset,
        }
    return {
        "type": "linear",
        "target": _key_record(constraint.target),
        "terms": [
            {"source": _key_record(source), "coefficient": coefficient}
            for source, coefficient in constraint.terms
        ],
        "offset": constraint.offset,
    }


def _constraint_from_record(record: dict[str, Any]) -> Constraint:
    if record["type"] == "fixed":
        return FixedConstraint(_key_from_record(record["target"]), float(record["value"]))
    if record["type"] == "affine":
        return AffineConstraint(
            _key_from_record(record["target"]),
            _key_from_record(record["source"]),
            float(record["multiplier"]),
            float(record["offset"]),
        )
    if record["type"] == "linear":
        return LinearConstraint(
            _key_from_record(record["target"]),
            tuple(
                (_key_from_record(item["source"]), float(item["coefficient"]))
                for item in record["terms"]
            ),
            float(record["offset"]),
        )
    raise PersistenceError(f"unknown constraint type {record['type']!r}")


def _instrument_record(instrument: Instrument | None) -> dict[str, Any] | None:
    if instrument is None:
        return None
    if isinstance(instrument, ConstantWavelengthInstrument):
        return {
            "type": "constant_wavelength",
            "values": {
                name: getattr(instrument, name)
                for name in (
                    "wavelength_angstrom",
                    "u_deg2",
                    "v_deg2",
                    "w_deg2",
                    "x_deg",
                    "y_deg",
                )
            },
        }
    if isinstance(instrument, TofInstrument):
        names = tuple(TofInstrument.__dataclass_fields__)
        return {"type": "tof", "values": {name: getattr(instrument, name) for name in names}}
    raise TypeError(f"unsupported instrument type {type(instrument).__name__}")


def _instrument_from_record(record: dict[str, Any] | None) -> Instrument | None:
    if record is None:
        return None
    if record["type"] == "constant_wavelength":
        return ConstantWavelengthInstrument(**record["values"])
    if record["type"] == "tof":
        return TofInstrument(**record["values"])
    raise PersistenceError(f"unknown instrument type {record['type']!r}")


def _experiment_record(
    experiment: ConstantWavelengthExperiment | None,
) -> dict[str, Any] | None:
    if experiment is None:
        return None
    if isinstance(experiment.radiation, ComponentRadiation):
        radiation = {
            "type": "components",
            "probe": experiment.radiation.probe.value,
            "wavelengths_angstrom": (experiment.radiation.components.wavelengths_angstrom.tolist()),
            "relative_intensities": (experiment.radiation.components.relative_intensities.tolist()),
        }
    else:
        radiation = {
            "type": "monochromatic",
            "probe": experiment.radiation.probe.value,
            "wavelength_angstrom": experiment.radiation.wavelength_angstrom,
        }
    return {
        "radiation": radiation,
        "instrument": _instrument_record(experiment.instrument),
        "zero_shift_deg": experiment.zero_shift_deg,
        "geometry": (
            None
            if experiment.geometry is None
            else {
                "type": "bragg_brentano",
                "goniometer_radius_mm": experiment.geometry.goniometer_radius_mm,
                "sample_displacement_mm": experiment.geometry.sample_displacement_mm,
            }
        ),
    }


def _experiment_from_record(
    record: dict[str, Any] | None,
) -> ConstantWavelengthExperiment | None:
    if record is None:
        return None
    instrument = _instrument_from_record(record["instrument"])
    if not isinstance(instrument, ConstantWavelengthInstrument):
        raise PersistenceError("constant-wavelength experiment instrument is invalid")
    geometry_record = record.get("geometry")
    if geometry_record is not None and geometry_record.get("type") != "bragg_brentano":
        raise PersistenceError("unknown constant-wavelength experiment geometry")
    radiation_record = record.get("radiation")
    if radiation_record is None:
        # Formats 1--5 stored only monochromatic radiation as flat fields.
        radiation: MonochromaticRadiation | ComponentRadiation = MonochromaticRadiation(
            RadiationProbe(record["probe"]), float(record["wavelength_angstrom"])
        )
    elif radiation_record.get("type") == "monochromatic":
        radiation = MonochromaticRadiation(
            RadiationProbe(radiation_record["probe"]),
            float(radiation_record["wavelength_angstrom"]),
        )
    elif radiation_record.get("type") == "components":
        radiation = ComponentRadiation(
            RadiationProbe(radiation_record["probe"]),
            WavelengthComponents(
                radiation_record["wavelengths_angstrom"],
                radiation_record["relative_intensities"],
            ),
        )
    else:
        raise PersistenceError("unknown constant-wavelength radiation type")
    return ConstantWavelengthExperiment(
        radiation,
        instrument,
        float(record.get("zero_shift_deg", 0.0)),
        (
            None
            if geometry_record is None
            else BraggBrentanoGeometry(
                float(geometry_record["goniometer_radius_mm"]),
                float(geometry_record["sample_displacement_mm"]),
            )
        ),
    )


def _fcj_record(geometry: FcjGeometry | None) -> dict[str, float] | None:
    if geometry is None:
        return None
    return {
        "sample_over_radius": geometry.sample_over_radius,
        "detector_over_radius": geometry.detector_over_radius,
    }


def _components_record(
    components: WavelengthComponents | None, arrays: _ArrayWriter
) -> dict[str, str] | None:
    if components is None:
        return None
    return {
        "wavelengths_angstrom": arrays.add(
            "wavelength_components_wavelengths", components.wavelengths_angstrom
        ),
        "relative_intensities": arrays.add(
            "wavelength_components_intensities", components.relative_intensities
        ),
    }


def _components_from_record(
    record: dict[str, Any] | None, arrays: dict[str, NDArray[np.generic]]
) -> WavelengthComponents | None:
    if record is None:
        return None
    return WavelengthComponents(
        arrays[record["wavelengths_angstrom"]], arrays[record["relative_intensities"]]
    )


def _provider_record(
    provider: ReflectionPhysicsProvider | None,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> dict[str, Any] | None:
    if provider is None:
        return None
    if isinstance(provider, IsotropicSizeBroadening):
        return {
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
            "configuration": {
                "crystallite_size_nm": (
                    provider.crystallite_size_nm
                    if np.isfinite(provider.crystallite_size_nm)
                    else None
                ),
                "shape_factor": provider.shape_factor,
            },
        }
    if isinstance(provider, IsotropicMicrostrainBroadening):
        return {
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
            "configuration": {"rms_microstrain": provider.rms_microstrain},
        }
    if isinstance(provider, MarchDollasePreferredOrientation):
        return {
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
            "configuration": {
                "march_ratio": provider.march_ratio,
                "preferred_axis_hkl": list(provider.preferred_axis_hkl),
                "reciprocal_metric": provider.reciprocal_metric.matrix.tolist(),
            },
        }
    if isinstance(provider, CompositePhysicsProvider):
        return {
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
            "configuration": {
                "providers": [_provider_record(child, codecs) for child in provider.providers]
            },
        }
    for codec in codecs:
        configuration = codec.encode(provider)
        if configuration is not None:
            return {
                "provider_id": codec.provider_id,
                "provider_version": provider.descriptor.provider_version,
                "configuration": configuration,
            }
    raise TypeError(
        f"provider {provider.descriptor.provider_id!r} requires an explicit PhysicsProviderCodec"
    )


def _provider_from_record(
    record: dict[str, Any] | None,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> ReflectionPhysicsProvider | None:
    if record is None:
        return None
    provider_id = record["provider_id"]
    provider_version = record["provider_version"]
    config = record["configuration"]
    if provider_id == "phasesmith.isotropic-size":
        if provider_version != "1":
            raise PersistenceError("unsupported phasesmith.isotropic-size provider version")
        return IsotropicSizeBroadening(
            np.inf if config["crystallite_size_nm"] is None else config["crystallite_size_nm"],
            config["shape_factor"],
        )
    if provider_id == "phasesmith.isotropic-microstrain":
        if provider_version != "1":
            raise PersistenceError("unsupported phasesmith.isotropic-microstrain provider version")
        return IsotropicMicrostrainBroadening(**config)
    if provider_id == "phasesmith.march-dollase":
        if provider_version != "1":
            raise PersistenceError("unsupported phasesmith.march-dollase provider version")
        return MarchDollasePreferredOrientation(
            config["march_ratio"],
            tuple(config["preferred_axis_hkl"]),
            ReciprocalMetric(config["reciprocal_metric"]),
        )
    if provider_id == "phasesmith.composite":
        if provider_version != "1":
            raise PersistenceError("unsupported phasesmith.composite provider version")
        return CompositePhysicsProvider(
            tuple(_provider_from_record(child, codecs) for child in config["providers"])
        )
    for codec in codecs:
        if codec.provider_id == provider_id:
            return codec.decode(config, provider_version)
    raise PersistenceError(f"no PhysicsProviderCodec is available for {provider_id!r}")


def _phase_record(
    phase: Phase, arrays: _ArrayWriter, prefix: str, codecs: tuple[PhysicsProviderCodec, ...]
) -> dict[str, Any]:
    reflections = phase.reflections
    record = {
        "phase_id": phase.phase_id,
        "name": phase.name,
        "scale": phase.scale,
        "physics": _provider_record(phase.physics, codecs),
        "reflections": {
            "reflection_ids": list(reflections.reflection_ids),
            "hkl": arrays.add(f"{prefix}_hkl", reflections.hkl),
            "d_spacing_angstrom": arrays.add(
                f"{prefix}_d_spacing_angstrom", reflections.d_spacing_angstrom
            ),
            "two_theta_deg": arrays.add(f"{prefix}_two_theta_deg", reflections.two_theta_deg),
            "integrated_intensity": arrays.add(
                f"{prefix}_integrated_intensity", reflections.integrated_intensity
            ),
        },
    }
    if isinstance(phase, LeBailPhase):
        record.update(
            {
                "phase_type": "lebail",
                "structure": structure_to_record(phase.structure),
                "reflection_domain": _reflection_domain_record(phase.reflection_domain),
                "reflections_generated": phase.reflections_generated,
            }
        )
    return record


def _reflection_domain_record(
    domain: CwLatticeReflectionDomain | None,
) -> dict[str, Any] | None:
    if domain is None:
        return None
    return {
        "lower": domain.bounds.lower.tolist(),
        "upper": domain.bounds.upper.tolist(),
        "wavelength_angstrom": domain.wavelength_angstrom,
        "visible_two_theta_min_deg": domain.visible_two_theta_min_deg,
        "visible_two_theta_max_deg": domain.visible_two_theta_max_deg,
        "initial_intensity": domain.initial_intensity,
        "merge_friedel": domain.merge_friedel,
        "max_candidates": domain.max_candidates,
        "guard_scale": domain.guard_scale,
    }


def _reflection_domain_from_record(
    record: dict[str, Any] | None,
    structure: CrystalStructure,
) -> CwLatticeReflectionDomain | None:
    if record is None:
        return None
    parameterization = LatticeParameterization(structure.space_group, structure.cell)
    bounds = LatticeParameterBounds(parameterization, record["lower"], record["upper"])
    return CwLatticeReflectionDomain(
        structure.space_group,
        parameterization,
        bounds,
        float(record["wavelength_angstrom"]),
        float(record["visible_two_theta_min_deg"]),
        float(record["visible_two_theta_max_deg"]),
        float(record["initial_intensity"]),
        record["merge_friedel"],
        record["max_candidates"],
        float(record["guard_scale"]),
    )


def _phase_from_record(
    record: dict[str, Any],
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> Phase:
    reflections = record["reflections"]
    arguments = (
        record["phase_id"],
        record["name"],
        ReflectionBatch(
            reflections["reflection_ids"],
            arrays[reflections["hkl"]],
            arrays[reflections["d_spacing_angstrom"]],
            arrays[reflections["two_theta_deg"]],
            arrays[reflections["integrated_intensity"]],
        ),
        float(record["scale"]),
        _provider_from_record(record["physics"], codecs),
    )
    if record.get("phase_type") != "lebail":
        return Phase(*arguments)
    structure = structure_from_record(record["structure"])
    return LeBailPhase(
        *arguments,
        structure,
        _reflection_domain_from_record(record["reflection_domain"], structure),
        record["reflections_generated"],
    )


def _scattering_record(provider: object) -> dict[str, Any]:
    if type(provider) is XrayNonResonant:
        return {
            "model": "xray_non_resonant",
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
        }
    if type(provider) is NeutronNuclear:
        return {
            "model": "neutron_nuclear",
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
        }
    if type(provider) is XrayFixedDispersion:
        return {
            "model": "xray_fixed_dispersion",
            "provider_id": provider.descriptor.provider_id,
            "provider_version": provider.descriptor.provider_version,
            "corrections": [
                {"element": element, "real": value.real, "imag": value.imag}
                for element, value in provider.corrections
            ],
        }
    raise TypeError("structural scattering persistence currently supports built-in models only")


def _scattering_from_record(
    record: dict[str, Any],
) -> XrayNonResonant | XrayFixedDispersion | NeutronNuclear:
    model = record["model"]
    if model == "xray_non_resonant":
        provider: XrayNonResonant | XrayFixedDispersion | NeutronNuclear = XrayNonResonant()
    elif model == "xray_fixed_dispersion":
        provider = XrayFixedDispersion(
            {
                str(value["element"]): complex(float(value["real"]), float(value["imag"]))
                for value in record["corrections"]
            }
        )
    elif model == "neutron_nuclear":
        provider = NeutronNuclear()
    else:
        raise PersistenceError(f"unsupported structural scattering model {model!r}")
    if (
        record["provider_id"] != provider.descriptor.provider_id
        or record["provider_version"] != provider.descriptor.provider_version
    ):
        raise PersistenceError("structural scattering provider version is incompatible")
    return provider


def _intensity_correction_record(provider: object) -> dict[str, Any]:
    if type(provider) is NeutralIntegratedIntensityCorrection:
        return {"model": "neutral"}
    if type(provider) is BraggBrentanoUnpolarizedLp:
        return {
            "model": "bragg_brentano_unpolarized_lp",
            "wavelength_angstrom": provider.wavelength_angstrom,
        }
    if type(provider) is BraggBrentanoPolarizedLp:
        return {
            "model": "bragg_brentano_polarized_lp",
            "wavelength_angstrom": provider.wavelength_angstrom,
            "polarization": provider.polarization,
        }
    raise TypeError(
        "structural intensity-correction persistence currently supports built-in models only"
    )


def _intensity_correction_from_record(
    record: dict[str, Any],
) -> NeutralIntegratedIntensityCorrection | BraggBrentanoUnpolarizedLp | BraggBrentanoPolarizedLp:
    model = record["model"]
    if model == "neutral":
        return NeutralIntegratedIntensityCorrection()
    if model == "bragg_brentano_unpolarized_lp":
        return BraggBrentanoUnpolarizedLp(float(record["wavelength_angstrom"]))
    if model == "bragg_brentano_polarized_lp":
        return BraggBrentanoPolarizedLp(
            float(record["wavelength_angstrom"]),
            float(record["polarization"]),
        )
    raise PersistenceError(f"unsupported structural intensity correction {model!r}")


def _rietveld_phase_record(
    phase: RietveldPhase,
    arrays: _ArrayWriter,
    prefix: str,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> dict[str, Any]:
    return {
        "phase_id": phase.phase_id,
        "name": phase.name,
        "structure": structure_to_record(phase.structure),
        "reflections": {
            "reflection_ids": list(phase.reflections.reflection_ids),
            "hkl": arrays.add(f"{prefix}_hkl", phase.reflections.hkl),
            "multiplicity": arrays.add(f"{prefix}_multiplicity", phase.reflections.multiplicity),
        },
        "scattering": _scattering_record(phase.scattering),
        "intensity_correction": _intensity_correction_record(phase.intensity_correction),
        "scale": phase.scale,
        "physics": _provider_record(phase.physics, codecs),
        "coordinate_tolerance": phase.coordinate_tolerance,
    }


def _rietveld_phase_from_record(
    record: dict[str, Any],
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> RietveldPhase:
    reflections = record["reflections"]
    return RietveldPhase(
        record["phase_id"],
        record["name"],
        structure_from_record(record["structure"]),
        StructuralReflectionBatch(
            reflections["reflection_ids"],
            arrays[reflections["hkl"]],
            arrays[reflections["multiplicity"]],
        ),
        _scattering_from_record(record["scattering"]),
        _intensity_correction_from_record(record["intensity_correction"]),
        float(record["scale"]),
        _provider_from_record(record["physics"], codecs),
        float(record["coordinate_tolerance"]),
    )


def _structural_domain_record(
    domain: CwStructuralReflectionDomain | None,
) -> dict[str, Any] | None:
    if domain is None:
        return None
    return {
        "lower": domain.bounds.lower.tolist(),
        "upper": domain.bounds.upper.tolist(),
        "wavelength_angstrom": domain.wavelength_angstrom,
        "visible_two_theta_min_deg": domain.visible_two_theta_min_deg,
        "visible_two_theta_max_deg": domain.visible_two_theta_max_deg,
        "merge_friedel": domain.merge_friedel,
        "max_candidates": domain.max_candidates,
        "guard_scale": domain.guard_scale,
    }


def _structural_domain_from_record(
    record: dict[str, Any] | None,
    phase: RietveldPhase,
) -> CwStructuralReflectionDomain | None:
    if record is None:
        return None
    parameterization = LatticeParameterization(phase.structure.space_group, phase.structure.cell)
    return CwStructuralReflectionDomain(
        phase.structure.space_group,
        parameterization,
        LatticeParameterBounds(parameterization, record["lower"], record["upper"]),
        float(record["wavelength_angstrom"]),
        float(record["visible_two_theta_min_deg"]),
        float(record["visible_two_theta_max_deg"]),
        bool(record["merge_friedel"]),
        int(record["max_candidates"]),
        float(record["guard_scale"]),
    )


def _rietveld_selection_record(
    selection: RietveldParameterSelection | None,
) -> dict[str, bool] | None:
    if selection is None:
        return None
    return {
        name: getattr(selection, name) for name in RietveldParameterSelection.__dataclass_fields__
    }


def _background_record(
    background: DifferentiableBackground | None,
) -> dict[str, Any] | None:
    if background is None:
        return None
    if type(background) is PolynomialBackground:
        return {
            "type": "polynomial",
            "background_id": background.background_id,
            "coefficients": list(background.coefficients),
        }
    if type(background) is ChebyshevBackground:
        return {
            "type": "chebyshev",
            "background_id": background.background_id,
            "coefficients": list(background.coefficients),
            "domain_deg": list(background.domain_deg),
        }
    if type(background) is PointBackground:
        return {
            "type": "point",
            "background_id": background.background_id,
            "knot_x": list(background.knot_x),
            "values": list(background.values),
        }
    if type(background) is AmorphousBackground:
        return {
            "type": "amorphous",
            "background_id": background.background_id,
            "peaks": [
                {
                    "area": peak.area,
                    "center_deg": peak.center_deg,
                    "fwhm_deg": peak.fwhm_deg,
                }
                for peak in background.peaks
            ],
        }
    if type(background) is CompositeBackground:
        return {
            "type": "composite",
            "background_id": background.background_id,
            "components": [_background_record(item) for item in background.components],
        }
    raise TypeError(f"unsupported differentiable background {type(background).__name__}")


def _background_from_record(
    record: dict[str, Any] | None,
) -> DifferentiableBackground | None:
    if record is None:
        return None
    model = record.get("type", "polynomial")
    if model == "polynomial":
        return PolynomialBackground(record["background_id"], tuple(record["coefficients"]))
    if model == "chebyshev":
        return ChebyshevBackground(
            record["background_id"], tuple(record["coefficients"]), tuple(record["domain_deg"])
        )
    if model == "point":
        return PointBackground(
            record["background_id"], tuple(record["knot_x"]), tuple(record["values"])
        )
    if model == "amorphous":
        return AmorphousBackground(
            record["background_id"],
            tuple(AmorphousPeak(**peak) for peak in record["peaks"]),
        )
    if model == "composite":
        components = tuple(_background_from_record(item) for item in record["components"])
        if any(item is None for item in components):
            raise PersistenceError("composite background components cannot be null")
        return CompositeBackground(record["background_id"], components)
    raise PersistenceError(f"unknown differentiable background type {model!r}")


def _rietveld_options_record(options: RietveldOptions | None) -> dict[str, Any] | None:
    if options is None:
        return None
    return {
        **{
            name: getattr(options, name)
            for name in RietveldOptions.__dataclass_fields__
            if name != "limits"
        },
        "limits": {
            name: getattr(options.limits, name) for name in RefinementLimits.__dataclass_fields__
        },
    }


def _rietveld_options_from_record(record: dict[str, Any] | None) -> RietveldOptions | None:
    if record is None:
        return None
    values = dict(record)
    values["limits"] = RefinementLimits(**values["limits"])
    return RietveldOptions(**values)


def _rietveld_iteration_record(item: RietveldIterationRecord) -> dict[str, Any]:
    return {
        "iteration": item.iteration,
        "rp": _ratio_record(item.rp),
        "rwp": _ratio_record(item.rwp),
        "chi_square": item.chi_square,
        "reduced_chi_square": _ratio_record(item.reduced_chi_square),
        "objective": item.objective,
        "objective_change": item.objective_change,
        "scaled_step_norm": item.scaled_step_norm,
        "damping": item.damping,
        "cg_iterations": item.cg_iterations,
        "backtracks": item.backtracks,
        "parameter_changes": [
            {
                "key": _key_record(change.key),
                "before": change.before,
                "after": change.after,
                "scaled_change": change.scaled_change,
            }
            for change in item.parameter_changes
        ],
        "topology_changes": list(item.topology_changes),
    }


def _rietveld_iteration_from_record(item: dict[str, Any]) -> RietveldIterationRecord:
    return RietveldIterationRecord(
        int(item["iteration"]),
        _ratio_from_record(item["rp"]),
        _ratio_from_record(item["rwp"]),
        float(item["chi_square"]),
        _ratio_from_record(item["reduced_chi_square"]),
        float(item["objective"]),
        float(item["objective_change"]),
        float(item["scaled_step_norm"]),
        float(item["damping"]),
        int(item["cg_iterations"]),
        int(item["backtracks"]),
        tuple(
            RietveldParameterChange(
                _key_from_record(change["key"]),
                float(change["before"]),
                float(change["after"]),
                float(change["scaled_change"]),
            )
            for change in item["parameter_changes"]
        ),
        tuple(item["topology_changes"]),
    )


def _rietveld_checkpoint_record(
    checkpoint: RietveldCheckpoint | None,
    arrays: _ArrayWriter,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "completed_iterations": checkpoint.completed_iterations,
        "phases": [
            _rietveld_phase_record(phase, arrays, f"rietveld_checkpoint_{index}", codecs)
            for index, phase in enumerate(checkpoint.phases)
        ],
        "parameters": _parameters_record(checkpoint.parameters),
        "objective": checkpoint.objective,
        "damping": checkpoint.damping,
        "history": [_rietveld_iteration_record(item) for item in checkpoint.history],
        "experiment": _experiment_record(checkpoint.experiment),
        "background": _background_record(checkpoint.background),
    }


def _rietveld_checkpoint_from_record(
    record: dict[str, Any] | None,
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> RietveldCheckpoint | None:
    if record is None:
        return None
    parameters = _parameters_from_record(record["parameters"])
    if parameters is None:
        raise PersistenceError("Rietveld checkpoint parameters are missing")
    return RietveldCheckpoint(
        int(record["completed_iterations"]),
        tuple(_rietveld_phase_from_record(phase, arrays, codecs) for phase in record["phases"]),
        parameters,
        float(record["objective"]),
        float(record["damping"]),
        tuple(_rietveld_iteration_from_record(item) for item in record["history"]),
        _experiment_from_record(record.get("experiment")),
        _background_from_record(record.get("background")),
    )


def _pattern_record(pattern: PowderPattern | None, arrays: _ArrayWriter) -> dict[str, Any] | None:
    if pattern is None:
        return None
    return {
        name: None if value is None else arrays.add(f"pattern_{name}", value)
        for name, value in (
            ("x", pattern.x),
            ("observed_y", pattern.observed_y),
            ("uncertainty", pattern.uncertainty),
            ("mask", pattern.mask),
            ("background", pattern.background),
        )
    }


def _pattern_from_record(
    record: dict[str, Any] | None, arrays: dict[str, NDArray[np.generic]]
) -> PowderPattern | None:
    if record is None:
        return None

    def optional(name: str) -> NDArray[np.generic] | None:
        return None if record[name] is None else arrays[record[name]]

    return PowderPattern(
        arrays[record["x"]],
        observed_y=optional("observed_y"),
        uncertainty=optional("uncertainty"),
        mask=optional("mask"),
        background=arrays[record["background"]],
    )


def _iteration_record(item: IterationRecord) -> dict[str, Any]:
    return {
        "iteration": item.iteration,
        "rp": _ratio_record(item.rp),
        "rwp": _ratio_record(item.rwp),
        "chi_square": item.chi_square,
        "reduced_chi_square": _ratio_record(item.reduced_chi_square),
        "maximum_relative_intensity_change": item.maximum_relative_intensity_change,
        "scaled_profile_step_norm": item.scaled_profile_step_norm,
        "parameter_changes": [
            {
                "key": _key_record(change.key),
                "before": change.before,
                "after": change.after,
                "scaled_change": change.scaled_change,
            }
            for change in item.parameter_changes
        ],
        "warnings": list(item.warnings),
    }


def _iteration_from_record(item: dict[str, Any]) -> IterationRecord:
    return IterationRecord(
        iteration=int(item["iteration"]),
        rp=_ratio_from_record(item["rp"]),
        rwp=_ratio_from_record(item["rwp"]),
        chi_square=float(item["chi_square"]),
        reduced_chi_square=_ratio_from_record(item["reduced_chi_square"]),
        maximum_relative_intensity_change=float(item["maximum_relative_intensity_change"]),
        scaled_profile_step_norm=float(item["scaled_profile_step_norm"]),
        parameter_changes=tuple(
            ParameterChange(
                _key_from_record(change["key"]),
                float(change["before"]),
                float(change["after"]),
                float(change["scaled_change"]),
            )
            for change in item["parameter_changes"]
        ),
        warnings=tuple(item["warnings"]),
    )


def _checkpoint_record(
    checkpoint: LeBailCheckpoint | None,
    arrays: _ArrayWriter,
    prefix: str,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {
        "completed_iterations": checkpoint.completed_iterations,
        "instrument": _instrument_record(checkpoint.instrument),
        "phases": [
            _phase_record(phase, arrays, f"{prefix}_phase_{index}", codecs)
            for index, phase in enumerate(checkpoint.phases)
        ],
        "intensities": arrays.add(f"{prefix}_intensities", checkpoint.intensities),
        "parameters": _parameters_record(checkpoint.parameters),
        "previous_rwp": _ratio_record(checkpoint.previous_rwp),
        "history": [_iteration_record(item) for item in checkpoint.history],
    }


def _checkpoint_from_record(
    record: dict[str, Any] | None,
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> LeBailCheckpoint | None:
    if record is None:
        return None
    instrument = _instrument_from_record(record["instrument"])
    if not isinstance(instrument, ConstantWavelengthInstrument):
        raise PersistenceError("Le Bail checkpoints require a constant-wavelength instrument")
    return LeBailCheckpoint(
        int(record["completed_iterations"]),
        instrument,
        tuple(_phase_from_record(phase, arrays, codecs) for phase in record["phases"]),
        np.asarray(arrays[record["intensities"]], dtype=np.float64),
        _parameters_from_record(record["parameters"]),
        _ratio_from_record(record["previous_rwp"]),
        tuple(_iteration_from_record(item) for item in record["history"]),
    )


def _calculation_record(
    calculation: PatternCalculationResult, arrays: _ArrayWriter, prefix: str
) -> dict[str, Any]:
    derivatives = calculation.derivatives
    local = derivatives.local
    accumulation = calculation.accumulation
    dense = accumulation.jacobian if accumulation.jacobian_layout == "dense" else None
    return {
        "y": arrays.add(f"{prefix}_y", calculation.y),
        "profile_y": arrays.add(f"{prefix}_profile_y", calculation.profile_y),
        "background": arrays.add(f"{prefix}_background", calculation.background),
        "accumulation_y": arrays.add(f"{prefix}_accumulation_y", accumulation.y),
        "local_starts": arrays.add(f"{prefix}_local_starts", local.starts),
        "local_offsets": arrays.add(f"{prefix}_local_offsets", local.offsets),
        "local_values": arrays.add(f"{prefix}_local_values", local.values),
        "global_jacobian": arrays.add(f"{prefix}_global_jacobian", derivatives.global_jacobian),
        "local_parameter_names": list(derivatives.local_parameter_names),
        "global_parameter_names": list(derivatives.global_parameter_names),
        "jacobian_layout": accumulation.jacobian_layout,
        "dense_jacobian": (
            None if dense is None else arrays.add(f"{prefix}_dense_jacobian", np.asarray(dense))
        ),
        "reflection_keys": [list(key) for key in calculation.reflection_keys],
        "phase_offsets": arrays.add(f"{prefix}_phase_offsets", calculation.phase_offsets),
        "phase_components": [
            {"phase_id": item.phase_id, "y": arrays.add(f"{prefix}_component", item.y)}
            for item in calculation.phase_components
        ],
    }


def _calculation_from_record(
    record: dict[str, Any], arrays: dict[str, NDArray[np.generic]]
) -> PatternCalculationResult:
    local = SupportJacobian(
        np.asarray(arrays[record["local_starts"]], dtype=np.int64),
        np.asarray(arrays[record["local_offsets"]], dtype=np.int64),
        np.asarray(arrays[record["local_values"]], dtype=np.float64),
    )
    derivatives = PatternDerivatives(
        local,
        np.asarray(arrays[record["global_jacobian"]], dtype=np.float64),
        tuple(record["local_parameter_names"]),
        tuple(record["global_parameter_names"]),
    )
    accumulation = AccumulationResult(
        np.asarray(arrays[record["accumulation_y"]], dtype=np.float64),
        derivatives,
        record["jacobian_layout"],
        (
            None
            if record["dense_jacobian"] is None
            else np.asarray(arrays[record["dense_jacobian"]], dtype=np.float64)
        ),
    )
    return PatternCalculationResult(
        np.asarray(arrays[record["y"]], dtype=np.float64),
        np.asarray(arrays[record["profile_y"]], dtype=np.float64),
        np.asarray(arrays[record["background"]], dtype=np.float64),
        accumulation,
        tuple(tuple(key) for key in record["reflection_keys"]),
        np.asarray(arrays[record["phase_offsets"]], dtype=np.int64),
        tuple(
            PhasePatternComponent(item["phase_id"], np.asarray(arrays[item["y"]], dtype=np.float64))
            for item in record["phase_components"]
        ),
    )


def _residual_record(
    metrics: ResidualEvaluation, arrays: _ArrayWriter, prefix: str
) -> dict[str, Any]:
    return {
        "included": arrays.add(f"{prefix}_included", metrics.included),
        "residual": arrays.add(f"{prefix}_residual", metrics.residual),
        "weighted_residual": arrays.add(f"{prefix}_weighted_residual", metrics.weighted_residual),
        "rp": _ratio_record(metrics.rp),
        "rwp": _ratio_record(metrics.rwp),
        "chi_square": metrics.chi_square,
        "reduced_chi_square": _ratio_record(metrics.reduced_chi_square),
    }


def _residual_from_record(
    record: dict[str, Any], arrays: dict[str, NDArray[np.generic]]
) -> ResidualEvaluation:
    return ResidualEvaluation(
        np.asarray(arrays[record["included"]], dtype=np.bool_),
        np.asarray(arrays[record["residual"]], dtype=np.float64),
        np.asarray(arrays[record["weighted_residual"]], dtype=np.float64),
        _ratio_from_record(record["rp"]),
        _ratio_from_record(record["rwp"]),
        float(record["chi_square"]),
        _ratio_from_record(record["reduced_chi_square"]),
    )


def _result_record(
    result: LeBailResult | None,
    arrays: _ArrayWriter,
    codecs: tuple[PhysicsProviderCodec, ...],
) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "calculation": _calculation_record(result.calculation, arrays, "result_calculation"),
        "instrument": _instrument_record(result.instrument),
        "phases": [
            _phase_record(phase, arrays, f"result_phase_{index}", codecs)
            for index, phase in enumerate(result.phases)
        ],
        "intensities": [
            {
                "phase_id": item.phase_id,
                "reflection_id": item.reflection_id,
                "integrated_intensity": item.integrated_intensity,
            }
            for item in result.intensities
        ],
        "metrics": _residual_record(result.metrics, arrays, "result_metrics"),
        "history": [_iteration_record(item) for item in result.history],
        "termination_reason": result.termination_reason.value,
        "rank_deficient_groups": [
            {
                "reflection_keys": [list(key) for key in group.reflection_keys],
                "rank": group.rank,
            }
            for group in result.rank_deficient_groups
        ],
        "parameters": _parameters_record(result.parameters),
        "covariance": (
            None
            if result.covariance is None
            else arrays.add("result_covariance", result.covariance)
        ),
        "checkpoint": _checkpoint_record(result.checkpoint, arrays, "result_checkpoint", codecs),
    }


def _result_from_record(
    record: dict[str, Any] | None,
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> LeBailResult | None:
    if record is None:
        return None
    instrument = _instrument_from_record(record["instrument"])
    checkpoint = _checkpoint_from_record(record["checkpoint"], arrays, codecs)
    if not isinstance(instrument, ConstantWavelengthInstrument) or checkpoint is None:
        raise PersistenceError("Le Bail result instrument or checkpoint is invalid")
    return LeBailResult(
        _calculation_from_record(record["calculation"], arrays),
        instrument,
        tuple(_phase_from_record(phase, arrays, codecs) for phase in record["phases"]),
        tuple(
            ReflectionIntensity(
                item["phase_id"],
                item["reflection_id"],
                float(item["integrated_intensity"]),
            )
            for item in record["intensities"]
        ),
        _residual_from_record(record["metrics"], arrays),
        tuple(_iteration_from_record(item) for item in record["history"]),
        TerminationReason(record["termination_reason"]),
        tuple(
            CoincidentReflectionGroup(
                tuple(tuple(key) for key in group["reflection_keys"]), int(group["rank"])
            )
            for group in record["rank_deficient_groups"]
        ),
        _parameters_from_record(record["parameters"]),
        (
            None
            if record["covariance"] is None
            else np.asarray(arrays[record["covariance"]], dtype=np.float64)
        ),
        checkpoint,
    )


def save_bundle(
    path: str | Path,
    bundle: PersistenceBundle,
    *,
    overwrite: bool = False,
    provider_codecs: tuple[PhysicsProviderCodec, ...] = (),
) -> Path:
    """Save a bundle as finite JSON metadata plus an `allow_pickle=False` NPZ."""

    if not isinstance(bundle, PersistenceBundle):
        raise TypeError("bundle must be PersistenceBundle")
    destination = Path(path).resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"persistence path exists and is not a directory: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"persistence directory already exists: {destination}")
    codecs = tuple(provider_codecs)
    writer = _ArrayWriter()
    record = {
        "pattern": _pattern_record(bundle.pattern, writer),
        "instrument": _instrument_record(bundle.instrument),
        "experiment": _experiment_record(bundle.experiment),
        "fcj_geometry": _fcj_record(bundle.fcj_geometry),
        "wavelength_components": _components_record(bundle.wavelength_components, writer),
        "phases": [
            _phase_record(phase, writer, f"phase_{index}", codecs)
            for index, phase in enumerate(bundle.phases)
        ],
        "rietveld_phases": [
            _rietveld_phase_record(phase, writer, f"rietveld_phase_{index}", codecs)
            for index, phase in enumerate(bundle.rietveld_phases)
        ],
        "rietveld_domains": [
            _structural_domain_record(domain) for domain in bundle.rietveld_domains
        ],
        "rietveld_selection": _rietveld_selection_record(bundle.rietveld_selection),
        "rietveld_options": _rietveld_options_record(bundle.rietveld_options),
        "rietveld_checkpoint": _rietveld_checkpoint_record(
            bundle.rietveld_checkpoint, writer, codecs
        ),
        "rietveld_background": _background_record(bundle.rietveld_background),
        "calculation_options": (
            None
            if bundle.calculation_options is None
            else {
                "support_fwhm": bundle.calculation_options.support_fwhm,
                "jacobian_layout": bundle.calculation_options.jacobian_layout,
                "return_phase_components": bundle.calculation_options.return_phase_components,
            }
        ),
        "calculation_result": (
            None
            if bundle.calculation_result is None
            else _calculation_record(bundle.calculation_result, writer, "calculation")
        ),
        "parameters": _parameters_record(bundle.parameters),
        "lebail_options": (
            None
            if bundle.lebail_options is None
            else {
                name: getattr(bundle.lebail_options, name)
                for name in LeBailOptions.__dataclass_fields__
            }
        ),
        "lebail_checkpoint": _checkpoint_record(
            bundle.lebail_checkpoint, writer, "checkpoint", codecs
        ),
        "lebail_result": _result_record(bundle.lebail_result, writer, codecs),
        "constraints": [_constraint_record(item) for item in bundle.constraints],
        "metadata": bundle.metadata,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        archive_path = temporary / ARCHIVE_NAME
        np.savez_compressed(archive_path, **writer.arrays)
        manifest = {
            "format_version": FORMAT_VERSION,
            "archive": {"file": ARCHIVE_NAME, "sha256": _sha256_file(archive_path)},
            "arrays": {name: _array_descriptor(value) for name, value in writer.arrays.items()},
            "bundle": record,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination.mkdir(parents=True, exist_ok=True)
        os.replace(temporary / ARCHIVE_NAME, destination / ARCHIVE_NAME)
        os.replace(temporary / MANIFEST_NAME, destination / MANIFEST_NAME)
        temporary.rmdir()
    except Exception:
        for name in (ARCHIVE_NAME, MANIFEST_NAME):
            candidate = temporary / name
            if candidate.exists():
                candidate.unlink()
        temporary.rmdir()
        raise
    return destination


def load_bundle(
    path: str | Path,
    *,
    provider_codecs: tuple[PhysicsProviderCodec, ...] = (),
) -> PersistenceBundle:
    """Validate and load a supported versioned bundle without pickle."""

    source = Path(path).resolve()
    try:
        manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersistenceError(f"cannot read persistence manifest: {error}") from error
    version = manifest.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in (1, 2, 3, 4, 5, 6, FORMAT_VERSION)
    ):
        raise PersistenceError(f"unsupported persistence format {manifest.get('format_version')!r}")
    if manifest.get("archive", {}).get("file") != ARCHIVE_NAME:
        raise PersistenceError("persistence archive filename is invalid")
    if not isinstance(manifest.get("arrays"), dict) or not isinstance(manifest.get("bundle"), dict):
        raise PersistenceError("persistence arrays and bundle must be JSON objects")
    archive_path = source / ARCHIVE_NAME
    if _sha256_file(archive_path) != manifest["archive"]["sha256"]:
        raise PersistenceError("persistence archive SHA-256 mismatch")
    arrays: dict[str, NDArray[np.generic]] = {}
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(manifest["arrays"]):
                raise PersistenceError("archive members do not match the manifest")
            for name, descriptor in manifest["arrays"].items():
                value = np.ascontiguousarray(archive[name])
                if (
                    str(value.dtype) != descriptor["dtype"]
                    or list(value.shape) != descriptor["shape"]
                ):
                    raise PersistenceError(f"array {name!r} dtype or shape mismatch")
                if hashlib.sha256(value.tobytes(order="C")).hexdigest() != descriptor["sha256"]:
                    raise PersistenceError(f"array {name!r} SHA-256 mismatch")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise PersistenceError(f"array {name!r} contains a non-finite value")
                value.flags.writeable = False
                arrays[name] = value
    except (OSError, ValueError) as error:
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError(f"cannot read persistence archive: {error}") from error
    record = manifest["bundle"]
    codecs = tuple(provider_codecs)
    options = record["calculation_options"]
    calculation_result = record["calculation_result"]
    structural_phases = tuple(
        _rietveld_phase_from_record(phase, arrays, codecs)
        for phase in record.get("rietveld_phases", ())
    )
    domain_records = record.get("rietveld_domains")
    structural_domains = (
        (None,) * len(structural_phases)
        if domain_records is None
        else tuple(
            _structural_domain_from_record(domain, phase)
            for domain, phase in zip(domain_records, structural_phases, strict=True)
        )
    )
    return PersistenceBundle(
        pattern=_pattern_from_record(record["pattern"], arrays),
        instrument=_instrument_from_record(record["instrument"]),
        experiment=_experiment_from_record(record["experiment"]),
        fcj_geometry=(
            None if record["fcj_geometry"] is None else FcjGeometry(**record["fcj_geometry"])
        ),
        wavelength_components=_components_from_record(record["wavelength_components"], arrays),
        phases=tuple(_phase_from_record(phase, arrays, codecs) for phase in record["phases"]),
        rietveld_phases=structural_phases,
        rietveld_domains=structural_domains,
        rietveld_selection=(
            None
            if record.get("rietveld_selection") is None
            else RietveldParameterSelection(**record["rietveld_selection"])
        ),
        rietveld_options=_rietveld_options_from_record(record.get("rietveld_options")),
        rietveld_checkpoint=_rietveld_checkpoint_from_record(
            record.get("rietveld_checkpoint"), arrays, codecs
        ),
        rietveld_background=_background_from_record(record.get("rietveld_background")),
        calculation_options=None if options is None else CalculationOptions(**options),
        calculation_result=(
            None
            if calculation_result is None
            else _calculation_from_record(calculation_result, arrays)
        ),
        parameters=_parameters_from_record(record["parameters"]),
        lebail_options=(
            None if record["lebail_options"] is None else LeBailOptions(**record["lebail_options"])
        ),
        lebail_checkpoint=_checkpoint_from_record(record["lebail_checkpoint"], arrays, codecs),
        lebail_result=_result_from_record(record["lebail_result"], arrays, codecs),
        constraints=tuple(_constraint_from_record(item) for item in record["constraints"]),
        metadata=record["metadata"],
    )
