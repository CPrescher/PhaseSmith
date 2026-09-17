"""Explicit empirical conventions for otherwise ambiguous Gaussian widths."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .refinement.rietveld import RietveldInput

METADATA_KEY = "phasesmith.empirical_gaussian"
_STRAIN_NAME = "isotropic_microstrain.rms"
_VARIANCE_COEFFICIENT = (2.0 * 180.0 / np.pi) ** 2


def _strain(phase):
    from .extensions import CompositePhysicsProvider
    from .sample import IsotropicMicrostrainBroadening

    providers = (
        phase.physics.providers
        if type(phase.physics) is CompositePhysicsProvider
        else (phase.physics,)
    )
    matches = [p for p in providers if type(p) is IsotropicMicrostrainBroadening]
    if len(matches) != 1:
        raise ValueError(
            "empirical Gaussian widths require one isotropic strain provider per phase"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class EmpiricalGaussianConvention:
    """Anchor one phase's RMS strain to define an empirical width decomposition.

    The reference is an explicit convention, not a measured strain or calibrated
    instrument resolution. A positive reference avoids initializing equal-width
    phases at the zero derivative of an RMS-strain parameter. Other phases must
    remain representable with non-negative strain variance.
    """

    reference_phase_id: str
    reference_rms_microstrain: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_phase_id, str)
            or not self.reference_phase_id
            or self.reference_phase_id != self.reference_phase_id.strip()
        ):
            raise ValueError("reference_phase_id must be a non-empty trimmed string")
        value = self.reference_rms_microstrain
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.floating)):
            raise TypeError("reference_rms_microstrain must be a real scalar")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0 or not 0.0 < value * value < np.inf:
            raise ValueError("reference_rms_microstrain must have positive finite squared value")
        object.__setattr__(self, "reference_rms_microstrain", value)

    def to_record(self) -> dict[str, Any]:
        """Return the persisted convention and its scientific interpretation."""
        return {
            "schema": "phasesmith.empirical-gaussian.v1",
            "reference_phase_id": self.reference_phase_id,
            "reference_rms_microstrain": self.reference_rms_microstrain,
            "interpretation": "empirical; instrument Gaussian widths and sample strains "
            "are convention-dependent, not independently measured",
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EmpiricalGaussianConvention:
        """Read only the documented, versioned convention record."""
        if not isinstance(record, dict) or set(record) != {
            "schema",
            "reference_phase_id",
            "reference_rms_microstrain",
            "interpretation",
        }:
            raise ValueError("invalid empirical Gaussian convention record")
        convention = cls(record["reference_phase_id"], record["reference_rms_microstrain"])
        if record != convention.to_record():
            raise ValueError("unknown empirical Gaussian convention schema or interpretation")
        return convention

    def validate_phases(self, phases) -> None:
        """Check that the explicitly anchored reference is still present and fixed."""
        matches = [p for p in phases if p.phase_id == self.reference_phase_id]
        if len(matches) != 1:
            raise ValueError("empirical Gaussian reference phase is missing or ambiguous")
        for phase in phases:
            _strain(phase)
        if _strain(matches[0]).rms_microstrain != self.reference_rms_microstrain:
            raise ValueError("empirical Gaussian reference strain differs from its convention")

    def apply(self, input_data: RietveldInput) -> RietveldInput:
        """Preserve initial total widths and attach a persistent reference constraint.

        Existing unrelated parameter settings and constraints are retained.
        Constraints or custom scales/bounds involving transferred U/strain
        variables are rejected rather than silently reinterpreted. The returned
        input is preflighted through the normal Rust-backed calculation.
        """
        from .extensions import CompositePhysicsProvider
        from .refinement import rietveld as rv
        from .refinement.core import ParameterSet
        from .refinement.workflow import _constraint_keys

        if not isinstance(input_data, rv.RietveldInput):
            raise TypeError("input_data must be RietveldInput")
        if input_data.empirical_gaussian is not None:
            if input_data.empirical_gaussian != self:
                raise ValueError(
                    "an empirical convention is already attached; cannot silently reanchor"
                )
            return input_data
        strains = {p.phase_id: _strain(p) for p in input_data.phases}
        if self.reference_phase_id not in strains:
            raise ValueError("empirical Gaussian reference phase is missing")
        old_reference = strains[self.reference_phase_id].rms_microstrain
        reference = self.reference_rms_microstrain
        shift = _VARIANCE_COEFFICIENT * (old_reference - reference) * (old_reference + reference)
        if not np.isfinite(shift):
            raise ValueError("empirical Gaussian variance transfer is not finite")
        phases = []
        for phase in input_data.phases:
            provider = strains[phase.phase_id]
            old = provider.rms_microstrain
            variance = (old - old_reference) * (old + old_reference) + reference**2
            if not np.isfinite(variance) or variance < 0.0:
                raise ValueError(f"phase {phase.phase_id} cannot use this empirical reference")
            value = (
                reference if phase.phase_id == self.reference_phase_id else float(np.sqrt(variance))
            )
            updated = replace(provider, rms_microstrain=value)
            physics = (
                replace(
                    phase.physics,
                    providers=tuple(
                        updated if p is provider else p for p in phase.physics.providers
                    ),
                )
                if type(phase.physics) is CompositePhysicsProvider
                else updated
            )
            phases.append(replace(phase, physics=physics))
        phases = tuple(phases)
        instrument = replace(
            input_data.experiment.instrument,
            u_deg2=input_data.experiment.instrument.u_deg2 + shift,
        )
        experiment = replace(input_data.experiment, instrument=instrument)
        affected = {rv.instrument_parameter_key("u_deg2")} | {
            rv.sample_parameter_key(p.phase_id, _STRAIN_NAME) for p in input_data.phases
        }
        if any(affected.intersection(_constraint_keys(c)) for c in input_data.constraints):
            raise ValueError(
                "existing constraints involve transferred instrument U or sample strain"
            )
        before = rv.build_parameter_set(
            input_data.phases,
            input_data.lattice_domains,
            input_data.selection,
            experiment=input_data.experiment,
            background=input_data.background,
        )
        after = rv.build_parameter_set(
            phases,
            input_data.lattice_domains,
            input_data.selection,
            experiment=experiment,
            background=input_data.background,
        )
        specs = []
        for spec in input_data.parameters.specs:
            if spec.key in affected:
                if spec != before.spec(spec.key):
                    raise ValueError(
                        "custom U/strain parameter settings require explicit reparameterization"
                    )
                specs.append(after.spec(spec.key))
            else:
                specs.append(spec)
        result = replace(
            input_data,
            phases=phases,
            experiment=experiment,
            parameters=ParameterSet(specs),
            empirical_gaussian=self,
        )
        rv.calculate(result.pattern, result.experiment, result.phases, background=result.background)
        return result


def _from_metadata(metadata: dict | None) -> EmpiricalGaussianConvention | None:
    if not metadata or METADATA_KEY not in metadata:
        return None
    return EmpiricalGaussianConvention.from_record(json.loads(metadata[METADATA_KEY]))


def _metadata(convention: EmpiricalGaussianConvention | None) -> dict[str, str]:
    return (
        {}
        if convention is None
        else {METADATA_KEY: json.dumps(convention.to_record(), allow_nan=False)}
    )
