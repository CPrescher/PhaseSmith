"""Pure-Rust lookup of exact conventional space-group operation sets."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .. import _core
from ..symmetry import SpaceGroup, SymmetryOperation


@dataclass(frozen=True, slots=True)
class SpaceGroupInfo:
    """Human identifiers plus the exact engine-owned operation set."""

    number: int
    hm_symbol: str
    hall_symbol: str
    setting: str
    space_group: SpaceGroup


def _convert(record: dict[str, Any]) -> SpaceGroupInfo:
    operations = []
    for operation in record["operations"]:
        rotation = np.asarray(operation["rotation"], dtype=np.int64)
        translation = tuple(
            Fraction(int(value[0]), int(value[1])) for value in operation["translation"]
        )
        operations.append(SymmetryOperation(rotation, translation))
    return SpaceGroupInfo(
        int(record["number"]),
        str(record["hm_symbol"]),
        str(record["hall_symbol"]),
        str(record["setting"]),
        SpaceGroup(operations),
    )


def space_group_by_number(number: int) -> SpaceGroupInfo:
    """Resolve an International Tables number in the native reference setting."""

    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 230:
        raise ValueError("space-group number must be an integer in [1, 230]")
    return _convert(_core._space_group_by_number(number))


def space_group_by_symbol(symbol: str) -> SpaceGroupInfo:
    """Resolve a Hermann--Mauguin, extended, or Hall-compatible symbol."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("space-group symbol must be a non-empty string")
    return _convert(_core._space_group_by_symbol(symbol.strip()))
