"""Comparator input must not disclose screw/glide translations or atom counts."""

import json
from types import SimpleNamespace

import gemmi
import pytest

from benchmarks.unknown_group_corpus import (
    grade,
    symmetry_instructions,
    symmetry_prior_from_instructions,
)


@pytest.mark.parametrize("number", [1, 4, 14, 61, 129, 139, 152, 167, 194, 221, 225])
def test_comparator_preserves_only_laue_rotations_and_centring(number):
    group = gemmi.find_spacegroup_by_number(number)
    text = symmetry_instructions(
        {
            "space_group": group.xhm(),
            "cell": [8, 8, 8, 90, 90, 90],
            "wavelength": 0.71,
            "elements": ["Fe", "O"],
        }
    )
    operations = [gemmi.Op("x,y,z")] + [
        gemmi.Op(line[5:]) for line in text.splitlines() if line.startswith("SYMM ")
    ]
    inversion = gemmi.Op("-x,-y,-z")

    def laue(ops):
        return {str(value.rot) for op in ops for value in (op, inversion.combine(op))}

    assert laue(operations) == laue(group.operations().sym_ops)
    assert all(op.tran == [0, 0, 0] for op in operations)
    lattice = {"P": 1, "I": 2, "R": 3, "F": 4, "A": 5, "B": 6, "C": 7}[
        group.operations().find_centering()
    ]
    assert f"LATT {lattice}\n" in text
    assert "UNIT 1 1\n" in text
    assert "SFAC Fe O\n" in text
    assert group.xhm() not in text.splitlines()[0]
    prior = symmetry_prior_from_instructions(text)
    expected = {
        tuple(tuple(sign * v // 24 for v in row) for row in op.rot)
        for op in group.operations().sym_ops
        for sign in (1, -1)
    }
    assert {tuple(map(tuple, r)) for r in prior["laue_rotations"]} == expected
    from fractions import Fraction

    assert {tuple(Fraction(n, d) for n, d in row) for row in prior["centring_translations"]} == {
        tuple(Fraction(v, 24) for v in shift) for shift in group.operations().cen_ops
    }


def test_grading_refuses_to_open_references_before_all_predictions_are_saved(tmp_path):
    pytest.importorskip("scipy.optimize")
    with pytest.raises(ValueError, match="save every prediction"):
        grade(SimpleNamespace(output=tmp_path))


def test_no_model_is_retained_as_an_outcome_with_its_effort_trace(tmp_path):
    pytest.importorskip("scipy.optimize")
    source = tmp_path / "case"
    model = source / "fast-1/model"
    model.mkdir(parents=True)
    (tmp_path / "predictions-complete.json").write_text("{}")
    (tmp_path / "protocol.json").write_text(
        json.dumps({"cases": [{"id": "case", "files": {}}], "repeats": {"fast": 1}})
    )
    (source / "source-inputs.json").write_text(
        json.dumps({"space_group": "P 1", "cell": [5, 5, 5, 90, 90, 90]})
    )
    (source / "reference.cif").write_text(
        "data_test\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_name_H-M_alt 'P 1'\nloop_\n_atom_site_label\n"
        "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\nC1 C 0 0 0\n"
    )
    (source / "fast-1/process.json").write_text(json.dumps({"returncode": 1}))
    effort = {"accepted": False, "passes": [{"mode": "fast", "reason": "no_model"}]}
    (model / "space-group-search.json").write_text(
        json.dumps({"effort": effort, "selected": None, "termination": "candidate_budget"})
    )
    grade(SimpleNamespace(output=tmp_path))
    row = json.loads((tmp_path / "graded.json").read_text())["results"][0]
    assert row["model_status"] == "no_model"
    assert row["effort"] == effort
    assert row["returncode"] == 1
    assert "geometry" not in row
    assert "grading_error" not in row
