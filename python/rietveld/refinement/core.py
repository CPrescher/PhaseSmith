"""Shared typed parameters, constraints, residuals, and Jacobian products."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pattern import PowderPattern
from ..results import PatternDerivatives


@dataclass(frozen=True, slots=True, order=True)
class ParameterKey:
    """Stable structured identity for one refinable scalar."""

    module: str
    owner_id: str
    name: str

    def __post_init__(self) -> None:
        """Reject empty or delimiter-ambiguous key segments."""

        for field_name, value in (
            ("module", self.module),
            ("owner_id", self.owner_id),
            ("name", self.name),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"parameter {field_name} must be a non-empty trimmed string")
            if any(character in value for character in "[]\n\r\t"):
                raise ValueError(f"parameter {field_name} contains a reserved character")

    @property
    def label(self) -> str:
        """Return a deterministic human-readable label."""

        return f"{self.module}[{self.owner_id}].{self.name}"


@dataclass(frozen=True, slots=True)
class Bounds:
    """Closed scalar lower and upper bounds."""

    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self) -> None:
        """Validate ordered non-NaN limits."""

        if np.isnan(self.lower) or np.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("bounds must be ordered and must not contain NaN")

    def contains(self, value: float) -> bool:
        """Return whether a scalar lies in the closed interval."""

        return bool(self.lower <= value <= self.upper)

    def clip(self, value: float) -> float:
        """Project a scalar onto the closed interval."""

        return float(np.clip(value, self.lower, self.upper))


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Value, units, scale, bounds, and selection for one scalar."""

    key: ParameterKey
    value: float
    unit: str
    bounds: Bounds = Bounds()
    scale: float = 1.0
    refine: bool = True

    def __post_init__(self) -> None:
        """Validate finite value/scale and bound membership."""

        if not isinstance(self.key, ParameterKey):
            raise TypeError("key must be a ParameterKey")
        if not np.isfinite(self.value):
            raise ValueError("parameter value must be finite")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("parameter unit must be a non-empty string")
        if not isinstance(self.bounds, Bounds):
            raise TypeError("bounds must be Bounds")
        if not self.bounds.contains(self.value):
            raise ValueError("parameter value lies outside its bounds")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("parameter scale must be positive and finite")


@dataclass(frozen=True, slots=True, init=False)
class ParameterSet:
    """Deterministically ordered immutable scalar specifications."""

    specs: tuple[ParameterSpec, ...]

    def __init__(self, specs: tuple[ParameterSpec, ...] | list[ParameterSpec]) -> None:
        """Validate unique keys while preserving explicit input order."""

        values = tuple(specs)
        if any(not isinstance(spec, ParameterSpec) for spec in values):
            raise TypeError("parameter sets contain only ParameterSpec objects")
        keys = tuple(spec.key for spec in values)
        if len(set(keys)) != len(keys):
            raise ValueError("parameter keys must be unique")
        object.__setattr__(self, "specs", values)

    @property
    def keys(self) -> tuple[ParameterKey, ...]:
        """Return all keys in deterministic packing order."""

        return tuple(spec.key for spec in self.specs)

    def spec(self, key: ParameterKey) -> ParameterSpec:
        """Return one specification by stable key."""

        for spec in self.specs:
            if spec.key == key:
                return spec
        raise KeyError(key)

    def values(self) -> dict[ParameterKey, float]:
        """Return a plain key-to-value copy."""

        return {spec.key: spec.value for spec in self.specs}

    def replace_values(self, values: dict[ParameterKey, float]) -> ParameterSet:
        """Return a new set with selected bounded values replaced."""

        unknown = set(values).difference(self.keys)
        if unknown:
            raise KeyError(f"unknown parameter keys: {sorted(key.label for key in unknown)!r}")
        return ParameterSet(
            [
                ParameterSpec(
                    key=spec.key,
                    value=float(values.get(spec.key, spec.value)),
                    unit=spec.unit,
                    bounds=spec.bounds,
                    scale=spec.scale,
                    refine=spec.refine,
                )
                for spec in self.specs
            ]
        )


@dataclass(frozen=True, slots=True)
class FixedConstraint:
    """Set one parameter to a fixed value during vector expansion."""

    target: ParameterKey
    value: float

    def __post_init__(self) -> None:
        """Reject non-finite fixed values."""

        if not np.isfinite(self.value):
            raise ValueError("fixed constraint value must be finite")


@dataclass(frozen=True, slots=True)
class AffineConstraint:
    """Define ``target = multiplier * source + offset``."""

    target: ParameterKey
    source: ParameterKey
    multiplier: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Validate distinct keys and finite affine coefficients."""

        if self.target == self.source:
            raise ValueError("affine target and source must differ")
        if not np.isfinite(self.multiplier) or not np.isfinite(self.offset):
            raise ValueError("affine coefficients must be finite")


Constraint = FixedConstraint | AffineConstraint


@dataclass(frozen=True, slots=True, init=False)
class ConstraintTransform:
    """Pack independent values and expand typed fixed/affine constraints."""

    parameters: ParameterSet
    constraints: tuple[Constraint, ...]
    free_keys: tuple[ParameterKey, ...]

    def __init__(
        self,
        parameters: ParameterSet,
        constraints: tuple[Constraint, ...] | list[Constraint] = (),
    ) -> None:
        """Validate target ownership, ordering, and acyclic dependencies."""

        if not isinstance(parameters, ParameterSet):
            raise TypeError("parameters must be a ParameterSet")
        selected = tuple(constraints)
        if any(not isinstance(item, (FixedConstraint, AffineConstraint)) for item in selected):
            raise TypeError("constraints must be fixed or affine constraints")
        known = set(parameters.keys)
        targets = tuple(item.target for item in selected)
        if any(target not in known for target in targets):
            raise ValueError("every constraint target must belong to the parameter set")
        if len(set(targets)) != len(targets):
            raise ValueError("each parameter can be constrained only once")
        resolved = known.difference(targets)
        for item in selected:
            if isinstance(item, AffineConstraint):
                if item.source not in known:
                    raise ValueError("every affine source must belong to the parameter set")
                if item.source not in resolved:
                    raise ValueError("affine constraints must be ordered without cycles")
            resolved.add(item.target)
        free_keys = tuple(
            spec.key for spec in parameters.specs if spec.refine and spec.key not in targets
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "constraints", selected)
        object.__setattr__(self, "free_keys", free_keys)

    def pack(self, values: dict[ParameterKey, float] | None = None) -> NDArray[np.float64]:
        """Pack scaled independent values in stable order."""

        source = self.parameters.values() if values is None else values
        packed = np.asarray(
            [source[key] / self.parameters.spec(key).scale for key in self.free_keys],
            dtype=np.float64,
        )
        if not np.isfinite(packed).all():
            raise ValueError("packed parameter values must be finite")
        return packed

    def unpack(self, vector: ArrayLike, *, clip: bool = False) -> dict[ParameterKey, float]:
        """Expand a scaled free vector into all bounded physical values."""

        packed = np.asarray(vector, dtype=np.float64)
        if packed.shape != (len(self.free_keys),) or not np.isfinite(packed).all():
            raise ValueError("free parameter vector has an invalid shape or value")
        values = self.parameters.values()
        for key, scaled in zip(self.free_keys, packed, strict=True):
            spec = self.parameters.spec(key)
            value = float(scaled * spec.scale)
            values[key] = spec.bounds.clip(value) if clip else value
        for constraint in self.constraints:
            if isinstance(constraint, FixedConstraint):
                values[constraint.target] = constraint.value
            else:
                values[constraint.target] = (
                    constraint.multiplier * values[constraint.source] + constraint.offset
                )
        for key, value in values.items():
            if not self.parameters.spec(key).bounds.contains(value):
                raise ValueError(f"expanded value for {key.label} lies outside its bounds")
        return values


@dataclass(frozen=True, slots=True)
class ResidualOptions:
    """Mask, uncertainty, and degrees-of-freedom controls."""

    use_uncertainty: bool = True
    parameter_count: int = 0

    def __post_init__(self) -> None:
        """Validate the non-negative fitted-parameter count."""

        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")


@dataclass(frozen=True, slots=True)
class ResidualEvaluation:
    """Selected residual arrays and standard powder residual metrics."""

    included: NDArray[np.bool_]
    residual: NDArray[np.float64]
    weighted_residual: NDArray[np.float64]
    rp: float
    rwp: float
    chi_square: float
    reduced_chi_square: float


def evaluate_residuals(
    pattern: PowderPattern,
    calculated_y: ArrayLike,
    options: ResidualOptions | None = None,
) -> ResidualEvaluation:
    """Evaluate ``calculated - observed`` and standard weighted metrics.

    A pattern mask uses ``True`` for included samples. Unmasked samples are all
    included. Uncertainty is interpreted as one standard deviation.
    """

    selected_options = ResidualOptions() if options is None else options
    if pattern.observed_y is None:
        raise ValueError("observed_y is required for residual evaluation")
    calculated = np.asarray(calculated_y, dtype=np.float64)
    if calculated.shape != pattern.x.shape or not np.isfinite(calculated).all():
        raise ValueError("calculated_y must be a finite vector matching the pattern")
    included = (
        np.ones(pattern.x.size, dtype=np.bool_)
        if pattern.mask is None
        else np.array(pattern.mask, copy=True)
    )
    residual = calculated - pattern.observed_y
    if selected_options.use_uncertainty and pattern.uncertainty is not None:
        weighted = residual / pattern.uncertainty
        weights = 1.0 / np.square(pattern.uncertainty)
    else:
        weighted = residual.copy()
        weights = np.ones(pattern.x.size, dtype=np.float64)
    selected_residual = residual[included]
    selected_weighted = weighted[included]
    selected_observed = pattern.observed_y[included]
    selected_weights = weights[included]
    chi_square = float(selected_weighted @ selected_weighted)
    denominator_rp = float(np.sum(np.abs(selected_observed)))
    denominator_rwp = float(np.sum(selected_weights * np.square(selected_observed)))
    rp = float(np.sum(np.abs(selected_residual)) / denominator_rp) if denominator_rp else np.inf
    rwp = float(np.sqrt(chi_square / denominator_rwp)) if denominator_rwp else np.inf
    degrees = int(np.count_nonzero(included)) - selected_options.parameter_count
    reduced = chi_square / degrees if degrees > 0 else np.inf
    for array in (included, residual, weighted):
        array.flags.writeable = False
    return ResidualEvaluation(
        included=included,
        residual=residual,
        weighted_residual=weighted,
        rp=rp,
        rwp=rwp,
        chi_square=chi_square,
        reduced_chi_square=float(reduced),
    )


def jacobian_vector_product(
    derivatives: PatternDerivatives,
    local_vector: ArrayLike,
    global_vector: ArrayLike,
) -> NDArray[np.float64]:
    """Apply the hybrid local/global Jacobian without dense materialization."""

    local = np.asarray(local_vector, dtype=np.float64)
    shared = np.asarray(global_vector, dtype=np.float64)
    expected_local = (
        derivatives.local.peak_count,
        derivatives.local.parameter_count,
    )
    if local.shape != expected_local or not np.isfinite(local).all():
        raise ValueError(f"local_vector must have shape {expected_local}")
    expected_global = (derivatives.global_jacobian.shape[0],)
    if shared.shape != expected_global or not np.isfinite(shared).all():
        raise ValueError(f"global_vector must have shape {expected_global}")
    result = shared @ derivatives.global_jacobian
    result = np.asarray(result, dtype=np.float64)
    for peak, start_value in enumerate(derivatives.local.starts):
        begin = int(derivatives.local.offsets[peak])
        end = int(derivatives.local.offsets[peak + 1])
        start = int(start_value)
        result[start : start + end - begin] += (
            derivatives.local.values[begin:end] @ local[peak]
        )
    return result


def transpose_jacobian_vector_product(
    derivatives: PatternDerivatives,
    sample_vector: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the transpose hybrid Jacobian and return local/global blocks."""

    samples = np.asarray(sample_vector, dtype=np.float64)
    sample_count = derivatives.global_jacobian.shape[1]
    if samples.shape != (sample_count,) or not np.isfinite(samples).all():
        raise ValueError(f"sample_vector must have shape ({sample_count},)")
    local_result = np.zeros(
        (derivatives.local.peak_count, derivatives.local.parameter_count),
        dtype=np.float64,
    )
    for peak, start_value in enumerate(derivatives.local.starts):
        begin = int(derivatives.local.offsets[peak])
        end = int(derivatives.local.offsets[peak + 1])
        start = int(start_value)
        local_result[peak] = (
            derivatives.local.values[begin:end].T
            @ samples[start : start + end - begin]
        )
    global_result = derivatives.global_jacobian @ samples
    return local_result, global_result


@runtime_checkable
class LeastSquaresOptimizer(Protocol):
    """Minimal adapter protocol for optional maintained optimizers."""

    def solve(
        self,
        residual: NDArray[np.float64],
        jacobian: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one bounded parameter step for a linearized objective."""


class TerminationReason(StrEnum):
    """Stable refinement termination labels."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    NO_OBSERVATIONS = "no_observations"
    NUMERICAL_FAILURE = "numerical_failure"
    CANCELLED = "cancelled"
