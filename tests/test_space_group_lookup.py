from __future__ import annotations

import numpy as np
import pytest
from phasesmith.io.space_groups import (
    space_group_by_number,
    space_group_by_symbol,
    space_group_from_hall_symbol,
)


def test_space_group_number_lookup_returns_exact_engine_operations_and_labels() -> None:
    info = space_group_by_number(225)
    assert info.number == 225
    assert info.hm_symbol == "F m -3 m"
    assert info.hall_symbol
    assert info.space_group.crystal_system == "cubic"
    absent = info.space_group.systematic_absences([[1, 0, 0], [1, 1, 1], [2, 0, 0]])
    np.testing.assert_array_equal(absent, [True, False, False])


def test_space_group_symbol_lookup_preserves_requested_setting_operations() -> None:
    info = space_group_by_symbol("P 21/c")
    assert info.number == 14
    assert info.setting == "b1"
    assert len(info.space_group.operations) == 4
    assert info.space_group.crystal_system == "monoclinic"


def test_general_hall_parser_preserves_exact_noncanonical_operations() -> None:
    canonical = space_group_from_hall_symbol("A 2 -2ab")
    redundant = space_group_from_hall_symbol("A 2 -2ac")
    assert redundant == canonical
    assert redundant == space_group_by_number(41).space_group

    shifted = space_group_from_hall_symbol("P 2y (3 0 0)")
    assert shifted != space_group_from_hall_symbol("P 2y")

    with pytest.raises(ValueError, match="F m 3 m"):
        space_group_from_hall_symbol("F m 3 m")


@pytest.mark.parametrize("value", [0, 231, True, 1.5])
def test_space_group_number_lookup_rejects_invalid_numbers(value: object) -> None:
    with pytest.raises(ValueError, match=r"\[1, 230\]"):
        space_group_by_number(value)  # type: ignore[arg-type]


def test_space_group_symbol_lookup_rejects_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        space_group_by_symbol("not a space group")


@pytest.mark.parametrize(
    ("number", "hm", "transformed"),
    [
        (2, "A -1", "-P 1 (-x,-1/2*y+1/2*z,1/2*y+1/2*z)"),
        (2, "B -1", "-P 1 (-1/2*x+1/2*z,-y,1/2*x+1/2*z)"),
        (2, "C -1", "-P 1 (1/2*x+1/2*y,1/2*x-1/2*y,-z)"),
        (2, "I -1", "-P 1 (-1/2*x+1/2*y+1/2*z,1/2*x-1/2*y+1/2*z,1/2*x+1/2*y-1/2*z)"),
        (2, "F -1", "-P 1 (1/2*y+1/2*z,1/2*x+1/2*z,1/2*x+1/2*y)"),
        (4, "P 1 21 1 (a,b,c-1/4)", "P 2yb (x,y,z+1/4)"),
        (9, "F 1 d 1", "C -2yc (x+1/2*z,y,1/2*z)"),
        (12, "C 1 2/m 1 (a+c-1/4,b+1/4,c)", "-C 2y (x+1/4,y-1/4,-x+z-1/4)"),
        (139, "I 4/m m m (a+b,-a+b,c)", "-I 4 2 (1/2*x+1/2*y,-1/2*x+1/2*y,z)"),
    ],
)
def test_deposited_basis_transforms_match_independent_hall_generators(
    number: int, hm: str, transformed: str
) -> None:
    from fractions import Fraction

    gemmi = pytest.importorskip("gemmi")
    info = space_group_by_symbol(hm)
    assert info.number == number
    assert info.space_group == space_group_from_hall_symbol(transformed)
    expected = {
        (
            tuple(value // op.DEN for row in op.rot for value in row),
            tuple(Fraction(value, op.DEN) % 1 for value in op.tran),
        )
        for op in gemmi.symops_from_hall(transformed)
    }
    assert {
        (tuple(op.rotation.flat), op.translation) for op in info.space_group.operations
    } == expected
