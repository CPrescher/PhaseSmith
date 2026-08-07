"""Quantitative phase analysis from compatible refined phase scales."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_vector(
    values: ArrayLike, name: str, *, allow_zero: bool = False
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if allow_zero:
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative")
    elif np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def weight_fractions_from_scale(
    scale: ArrayLike,
    formula_units_per_cell: ArrayLike,
    formula_mass_g_mol: ArrayLike,
    cell_volume_angstrom3: ArrayLike,
) -> NDArray[np.float64]:
    r"""Convert compatible Rietveld scales to crystalline weight fractions.

    This implements the Hill--Howard relation

    ``W_p = S_p (Z M V)_p / sum_i S_i (Z M V)_i``.

    All scales must use the same structure-factor and intensity normalization.
    The result describes only the supplied crystalline phases; it does not
    measure unidentified or amorphous material without an appropriate standard.
    """

    scales = _positive_vector(scale, "scale", allow_zero=True)
    z = _positive_vector(formula_units_per_cell, "formula_units_per_cell")
    mass = _positive_vector(formula_mass_g_mol, "formula_mass_g_mol")
    volume = _positive_vector(cell_volume_angstrom3, "cell_volume_angstrom3")
    if not (scales.shape == z.shape == mass.shape == volume.shape):
        raise ValueError("all quantitative-phase arrays must have the same shape")
    with np.errstate(over="ignore", invalid="ignore"):
        contributions = scales * z * mass * volume
    if not np.isfinite(contributions).all():
        raise ValueError("quantitative-phase contributions overflowed")
    total = float(np.sum(contributions))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("at least one phase scale must be positive")
    fractions = np.ascontiguousarray(contributions / total)
    fractions.flags.writeable = False
    return fractions


@dataclass(frozen=True, slots=True)
class QuantitativePhase:
    """Metadata required to interpret one compatible refined phase scale."""

    phase_id: str
    scale: float
    formula_units_per_cell: float
    formula_mass_g_mol: float
    cell_volume_angstrom3: float

    def __post_init__(self) -> None:
        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("phase_id must be a non-empty string")
        values = (
            self.scale,
            self.formula_units_per_cell,
            self.formula_mass_g_mol,
            self.cell_volume_angstrom3,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("quantitative-phase numeric fields must be real scalars")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("quantitative-phase numeric fields must be finite")
        if self.scale < 0.0:
            raise ValueError("phase scale must be nonnegative")
        if (
            min(
                self.formula_units_per_cell,
                self.formula_mass_g_mol,
                self.cell_volume_angstrom3,
            )
            <= 0.0
        ):
            raise ValueError("Z, formula mass, and cell volume must be positive")


@dataclass(frozen=True, slots=True)
class PhaseWeightFraction:
    """One labeled normalized result from crystalline quantitative analysis."""

    phase_id: str
    weight_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("phase_id must be a non-empty string")
        if not math.isfinite(self.weight_fraction) or not 0.0 <= self.weight_fraction <= 1.0:
            raise ValueError("weight_fraction must be finite and between zero and one")


def quantitative_phase_analysis(
    phases: Iterable[QuantitativePhase],
) -> tuple[PhaseWeightFraction, ...]:
    """Return labeled weight fractions while preserving caller phase order."""

    records = tuple(phases)
    if not records:
        raise ValueError("quantitative phase analysis requires at least one phase")
    if any(not isinstance(phase, QuantitativePhase) for phase in records):
        raise TypeError("phases must contain QuantitativePhase values")
    phase_ids = tuple(phase.phase_id for phase in records)
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError("quantitative phase IDs must be unique")
    fractions = weight_fractions_from_scale(
        [phase.scale for phase in records],
        [phase.formula_units_per_cell for phase in records],
        [phase.formula_mass_g_mol for phase in records],
        [phase.cell_volume_angstrom3 for phase in records],
    )
    return tuple(
        PhaseWeightFraction(phase_id, float(fraction))
        for phase_id, fraction in zip(phase_ids, fractions, strict=True)
    )
