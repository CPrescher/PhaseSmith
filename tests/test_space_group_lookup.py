from __future__ import annotations

import numpy as np
import phasesmith
import pytest


def test_space_group_number_lookup_returns_exact_engine_operations_and_labels() -> None:
    info = phasesmith.space_group_by_number(225)
    assert info.number == 225
    assert info.hm_symbol == "F m -3 m"
    assert info.hall_symbol
    assert info.space_group.crystal_system == "cubic"
    absent = info.space_group.systematic_absences([[1, 0, 0], [1, 1, 1], [2, 0, 0]])
    np.testing.assert_array_equal(absent, [True, False, False])


def test_space_group_symbol_lookup_preserves_requested_setting_operations() -> None:
    info = phasesmith.space_group_by_symbol("P 21/c")
    assert info.number == 14
    assert info.setting == "b1"
    assert len(info.space_group.operations) == 4
    assert info.space_group.crystal_system == "monoclinic"


def test_general_hall_parser_preserves_exact_noncanonical_operations() -> None:
    canonical = phasesmith.space_group_from_hall_symbol("A 2 -2ab")
    redundant = phasesmith.space_group_from_hall_symbol("A 2 -2ac")
    assert redundant == canonical
    assert redundant == phasesmith.space_group_by_number(41).space_group

    shifted = phasesmith.space_group_from_hall_symbol("P 2y (3 0 0)")
    assert shifted != phasesmith.space_group_from_hall_symbol("P 2y")

    with pytest.raises(ValueError, match="F m 3 m"):
        phasesmith.space_group_from_hall_symbol("F m 3 m")


@pytest.mark.parametrize("value", [0, 231, True, 1.5])
def test_space_group_number_lookup_rejects_invalid_numbers(value: object) -> None:
    with pytest.raises(ValueError, match=r"\[1, 230\]"):
        phasesmith.space_group_by_number(value)  # type: ignore[arg-type]


def test_space_group_symbol_lookup_rejects_unknown_symbol() -> None:
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        phasesmith.space_group_by_symbol("not a space group")
