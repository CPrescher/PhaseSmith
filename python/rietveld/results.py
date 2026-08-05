"""Typed, NumPy-native calculation result containers.

These containers are deliberately independent of instrument and refinement
models.  Calculation, Le Bail, and integration layers can therefore exchange
the same stable plain-array results without importing the native extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SupportJacobian:
    """Per-reflection derivatives stored only at active grid samples.

    ``starts[p]`` is the first active grid index for reflection ``p``.
    ``values[offsets[p]:offsets[p + 1]]`` contains sample-major derivative
    rows in the order declared by :class:`PatternDerivatives`.
    """

    starts: NDArray[np.int64]
    offsets: NDArray[np.int64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate sparse-array structure at construction."""

        if self.starts.dtype != np.int64 or self.starts.ndim != 1:
            raise ValueError("starts must be a one-dimensional int64 array")
        if self.offsets.dtype != np.int64 or self.offsets.ndim != 1:
            raise ValueError("offsets must be a one-dimensional int64 array")
        if self.values.dtype != np.float64 or self.values.ndim != 2:
            raise ValueError("values must be a two-dimensional float64 array")
        if self.offsets.size != self.starts.size + 1 or self.offsets[0] != 0:
            raise ValueError("offsets must have peak_count + 1 entries and begin at zero")
        if np.any(self.starts < 0) or np.any(np.diff(self.offsets) < 0):
            raise ValueError("support indices must be non-negative and nondecreasing")
        if self.values.shape[0] != int(self.offsets[-1]) or self.values.shape[1] == 0:
            raise ValueError("values shape does not match offsets and local parameter count")

    @property
    def peak_count(self) -> int:
        """Return the number of reflection support blocks."""

        return int(self.starts.size)

    @property
    def active_sample_count(self) -> int:
        """Return the number of stored reflection/sample derivative rows."""

        return int(self.offsets[-1]) if self.offsets.size else 0

    @property
    def parameter_count(self) -> int:
        """Return derivatives stored per active sample."""

        return int(self.values.shape[1])

    @property
    def nbytes(self) -> int:
        """Return bytes owned by all sparse index and value arrays."""

        return self.starts.nbytes + self.offsets.nbytes + self.values.nbytes

    def to_dense(self, sample_count: int) -> NDArray[np.float64]:
        """Materialize a ``(reflection, parameter, sample)`` array."""

        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        dense = np.zeros((self.peak_count, self.parameter_count, sample_count), dtype=np.float64)
        for peak_index, start_value in enumerate(self.starts):
            begin = int(self.offsets[peak_index])
            end = int(self.offsets[peak_index + 1])
            start = int(start_value)
            stop = start + end - begin
            if start < 0 or stop > sample_count:
                raise ValueError("support block lies outside the requested sample grid")
            dense[peak_index, :, start:stop] = self.values[begin:end].T
        return dense


@dataclass(frozen=True, slots=True)
class PatternDerivatives:
    """Sparse local and dense shared derivatives for a calculated pattern."""

    local: SupportJacobian
    global_jacobian: NDArray[np.float64]
    local_parameter_names: tuple[str, ...]
    global_parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate derivative names and dense global rows."""

        if len(self.local_parameter_names) != self.local.parameter_count:
            raise ValueError("local parameter names do not match stored derivative columns")
        if self.global_jacobian.dtype != np.float64 or self.global_jacobian.ndim != 2:
            raise ValueError("global_jacobian must be a two-dimensional float64 array")
        if self.global_jacobian.shape[0] != len(self.global_parameter_names):
            raise ValueError("global Jacobian rows must match global parameter names")


@dataclass(frozen=True, slots=True)
class AccumulationResult:
    """Calculated pattern and analytical derivatives."""

    y: NDArray[np.float64]
    derivatives: PatternDerivatives
    jacobian_layout: Literal["support", "dense"]
    _dense_jacobian: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate the selected representation and sample dimension."""

        if self.y.dtype != np.float64 or self.y.ndim != 1:
            raise ValueError("y must be a one-dimensional float64 array")
        if self.derivatives.global_jacobian.shape[1] != self.y.size:
            raise ValueError("global Jacobian sample count must match y")
        expected_shape = (
            self.derivatives.local.peak_count,
            self.derivatives.local.parameter_count,
            self.y.size,
        )
        if self.jacobian_layout == "support":
            if self._dense_jacobian is not None:
                raise ValueError("support layout must not contain a dense Jacobian")
        elif self.jacobian_layout == "dense":
            if self._dense_jacobian is None or self._dense_jacobian.shape != expected_shape:
                raise ValueError("dense Jacobian has an inconsistent shape")
        else:
            raise ValueError("jacobian_layout must be 'support' or 'dense'")

    @property
    def jacobian(self) -> SupportJacobian | NDArray[np.float64]:
        """Return the requested local Jacobian representation."""

        if self.jacobian_layout == "support":
            return self.derivatives.local
        if self._dense_jacobian is None:  # pragma: no cover - construction invariant
            raise RuntimeError("dense Jacobian was not materialized")
        return self._dense_jacobian


def _build_accumulation_result(
    y: NDArray[np.float64],
    starts: NDArray[np.int64],
    offsets: NDArray[np.int64],
    local_values: NDArray[np.float64],
    global_values: NDArray[np.float64],
    local_parameter_names: tuple[str, ...],
    global_parameter_names: tuple[str, ...],
    jacobian_layout: Literal["support", "dense"],
) -> AccumulationResult:
    """Build a validated public result from native arrays."""

    local = SupportJacobian(starts=starts, offsets=offsets, values=local_values)
    derivatives = PatternDerivatives(
        local=local,
        global_jacobian=global_values,
        local_parameter_names=local_parameter_names,
        global_parameter_names=global_parameter_names,
    )
    dense = local.to_dense(y.size) if jacobian_layout == "dense" else None
    return AccumulationResult(
        y=y,
        derivatives=derivatives,
        jacobian_layout=jacobian_layout,
        _dense_jacobian=dense,
    )
