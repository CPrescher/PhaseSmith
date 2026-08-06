"""Optional Gemmi-backed lookup of exact space-group operation sets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from ..symmetry import SpaceGroup, SymmetryOperation


@dataclass(frozen=True, slots=True)
class SpaceGroupInfo:
    """Human identifiers plus the exact engine-owned operation set."""

    number: int
    hm_symbol: str
    hall_symbol: str
    setting: str
    space_group: SpaceGroup


def _gemmi() -> Any:
    try:
        import gemmi
    except ImportError as error:
        raise ImportError(
            "space-group lookup requires the optional 'rietveld-engine[cif]' dependency"
        ) from error
    return gemmi


def _convert(group: Any) -> SpaceGroupInfo:
    operations = []
    for operation in group.operations():
        denominator = int(operation.DEN)
        rotation = np.asarray(operation.rot, dtype=np.int64) // denominator
        translation = tuple(Fraction(int(value), denominator) for value in operation.tran)
        operations.append(SymmetryOperation(rotation, translation))
    setting = str(group.qualifier).strip("\x00")
    return SpaceGroupInfo(
        int(group.number),
        str(group.hm),
        str(group.hall),
        setting,
        SpaceGroup(operations),
    )


def space_group_by_number(number: int) -> SpaceGroupInfo:
    """Resolve an International Tables number in Gemmi's reference setting."""

    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 230:
        raise ValueError("space-group number must be an integer in [1, 230]")
    return _convert(_gemmi().SpaceGroup(number))


def space_group_by_symbol(symbol: str) -> SpaceGroupInfo:
    """Resolve a Hermann--Mauguin, extended, or Hall-compatible symbol."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("space-group symbol must be a non-empty string")
    group = _gemmi().find_spacegroup_by_name(symbol.strip())
    if group is None:
        raise ValueError(f"unknown or ambiguous space-group symbol {symbol!r}")
    return _convert(group)
