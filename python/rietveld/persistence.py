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
from .pattern import (
    PatternCalculationResult,
    PhasePatternComponent,
    PowderPattern,
)
from .phase import Phase, ReciprocalMetric, ReflectionBatch
from .radiation import (
    ConstantWavelengthExperiment,
    MonochromaticRadiation,
    RadiationProbe,
    WavelengthComponents,
)
from .refinement.core import (
    AffineConstraint,
    Bounds,
    Constraint,
    FixedConstraint,
    ParameterKey,
    ParameterSet,
    ParameterSpec,
    ResidualEvaluation,
    TerminationReason,
)
from .refinement.lebail import (
    CoincidentReflectionGroup,
    IterationRecord,
    LeBailCheckpoint,
    LeBailInput,
    LeBailOptions,
    LeBailResult,
    ParameterChange,
    ReflectionIntensity,
)
from .results import AccumulationResult, PatternDerivatives, SupportJacobian
from .sample import (
    IsotropicMicrostrainBroadening,
    IsotropicSizeBroadening,
    MarchDollasePreferredOrientation,
)

FORMAT_VERSION: Final = 1
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
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if any(not isinstance(phase, Phase) for phase in self.phases):
            raise TypeError("phases must contain only Phase objects")
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
    return {
        "type": "affine",
        "target": _key_record(constraint.target),
        "source": _key_record(constraint.source),
        "multiplier": constraint.multiplier,
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
    return {
        "probe": experiment.radiation.probe.value,
        "wavelength_angstrom": experiment.radiation.wavelength_angstrom,
        "instrument": _instrument_record(experiment.instrument),
    }


def _experiment_from_record(
    record: dict[str, Any] | None,
) -> ConstantWavelengthExperiment | None:
    if record is None:
        return None
    instrument = _instrument_from_record(record["instrument"])
    if not isinstance(instrument, ConstantWavelengthInstrument):
        raise PersistenceError("constant-wavelength experiment instrument is invalid")
    return ConstantWavelengthExperiment(
        MonochromaticRadiation(
            RadiationProbe(record["probe"]), float(record["wavelength_angstrom"])
        ),
        instrument,
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
    if provider_id == "rietveld.isotropic-size":
        if provider_version != "1":
            raise PersistenceError("unsupported rietveld.isotropic-size provider version")
        return IsotropicSizeBroadening(
            np.inf if config["crystallite_size_nm"] is None else config["crystallite_size_nm"],
            config["shape_factor"],
        )
    if provider_id == "rietveld.isotropic-microstrain":
        if provider_version != "1":
            raise PersistenceError("unsupported rietveld.isotropic-microstrain provider version")
        return IsotropicMicrostrainBroadening(**config)
    if provider_id == "rietveld.march-dollase":
        if provider_version != "1":
            raise PersistenceError("unsupported rietveld.march-dollase provider version")
        return MarchDollasePreferredOrientation(
            config["march_ratio"],
            tuple(config["preferred_axis_hkl"]),
            ReciprocalMetric(config["reciprocal_metric"]),
        )
    if provider_id == "rietveld.composite":
        if provider_version != "1":
            raise PersistenceError("unsupported rietveld.composite provider version")
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
    return {
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


def _phase_from_record(
    record: dict[str, Any],
    arrays: dict[str, NDArray[np.generic]],
    codecs: tuple[PhysicsProviderCodec, ...],
) -> Phase:
    reflections = record["reflections"]
    return Phase(
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
    """Validate and load a version-1 persistence bundle without pickle."""

    source = Path(path).resolve()
    try:
        manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PersistenceError(f"cannot read persistence manifest: {error}") from error
    if manifest.get("format_version") != FORMAT_VERSION:
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
    return PersistenceBundle(
        pattern=_pattern_from_record(record["pattern"], arrays),
        instrument=_instrument_from_record(record["instrument"]),
        experiment=_experiment_from_record(record["experiment"]),
        fcj_geometry=(
            None if record["fcj_geometry"] is None else FcjGeometry(**record["fcj_geometry"])
        ),
        wavelength_components=_components_from_record(record["wavelength_components"], arrays),
        phases=tuple(_phase_from_record(phase, arrays, codecs) for phase in record["phases"]),
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
