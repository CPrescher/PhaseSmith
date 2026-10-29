"""Corpus leakage controls, measured-intensity translation and independent scoring."""

import importlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
import pytest
from phasesmith.io.hkl import HklData, parse_hkl

PACKAGE = Path(__file__).resolve().parents[1] / "benchmarks/solution_corpus"
spec = importlib.util.spec_from_file_location(
    "solution_corpus", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
data_module = importlib.import_module("solution_corpus.data")
run_module = importlib.import_module("solution_corpus.run")
assign_splits, common_hkl, family_components, field8, reflection_data, shelxt_name = (
    getattr(data_module, name)
    for name in [
        "assign_splits",
        "common_hkl",
        "family_components",
        "field8",
        "reflection_data",
        "shelxt_name",
    ]
)
INPUT_KEYS, ROOT, instructions, launch = (
    getattr(run_module, name) for name in ["INPUT_KEYS", "ROOT", "instructions", "launch"]
)


def match_geometry(*args, **kwargs):
    # Keep optional SciPy out of test collection and ordinary package imports.
    pytest.importorskip("scipy.optimize")
    return importlib.import_module("solution_corpus.audit").match_geometry(*args, **kwargs)


def shelxt_candidate(*args, **kwargs):
    pytest.importorskip("scipy.optimize")
    return importlib.import_module("solution_corpus.audit").shelxt_candidate(*args, **kwargs)


def test_common_hkl_preserves_signed_values_and_the_same_rows_for_both_programs():
    data = HklData(
        np.array([[1, 2, 3], [-1, -2, -3], [2, 1, 0], [3, 0, 1]]),
        [-1234567.0, 0.0000123456, 1234.56789, 0.0],
        [0.1234567, 0.000000123456, 0.1, 2.0],
    )
    text, audit = common_hkl(data)
    read = parse_hkl(text)
    np.testing.assert_array_equal(read.hkl, data.hkl)
    assert read.intensity[0] < 0 and read.intensity[-1] == 0
    np.testing.assert_allclose(read.intensity, data.intensity, rtol=5e-4, atol=0)
    np.testing.assert_allclose(read.sigma, data.sigma, rtol=5e-4, atol=0)
    assert audit["intensity_maximum_relative_change"] <= 5e-4
    assert all("." in field8(v).split("E")[0] for v in [0.0, 1.0, -1e6, 2e-8])
    assert all(len(field8(v)) == 8 for v in [1e-30, -1e30, 0.1234567])


def test_measured_f_squared_only_never_substitutes_calculated_data(tmp_path):
    path = tmp_path / "observations.cif"
    text = """data_test
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_F_squared_meas
_refln_F_squared_sigma
_refln_F_squared_calc
1 0 0 -3 2 99
0 1 0 7 1 88
"""
    path.write_text(text)
    data, provenance = reflection_data(path)
    np.testing.assert_array_equal(data.intensity, [-3.0, 7.0])
    assert provenance["negative_intensities"] == 1
    path.write_text(text.replace("_refln_F_squared_meas", "_refln_not_observed"))
    with pytest.raises(ValueError, match="measured"):
        reflection_data(path)
    path.write_text(text.replace("-3 2 99", "-3 0 99"))
    with pytest.raises(ValueError, match="sigma"):
        reflection_data(path)


def test_connected_publication_and_formula_families_cannot_cross_splits():
    cases = [
        {"id": str(i), "curation": {"doi": doi, "formula_group": formula}}
        for i, (doi, formula) in enumerate(
            [("doi-a", "AB"), ("doi-a", "AC"), ("doi-b", "AC"), ("doi-c", "XY")]
        )
    ]
    components = family_components(cases)
    assert components["0"] == components["1"] == components["2"]
    assign_splits(cases)
    assert len({c["split"] for c in cases[:3]}) == 1
    first = json.dumps(cases, sort_keys=True)
    assign_splits(cases)
    assert json.dumps(cases, sort_keys=True) == first


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("P 21/c", "P2(1)_c"),
        ("P 21 21 21", "P2(1)2(1)2(1)"),
        ("P 31 2 1", "P3(1)21"),
        ("I4/mcm", "I4_mcm"),
    ],
)
def test_fixed_space_group_spelling(symbol, expected):
    assert shelxt_name(gemmi.find_spacegroup_by_name(symbol)) == expected


def test_external_instructions_use_dummy_unit_and_input_types_only():
    packet = {
        "id": "case",
        "cell": [8, 9, 10, 90, 90, 90],
        "space_group": "Pnma",
        "space_group_number": 62,
        "shelxt_space_group": "Pnma",
        "wavelength": 0.71073,
        "elements": ["Bi", "Se", "I"],
    }
    assert set(packet) == INPUT_KEYS
    text = instructions(packet)
    assert "UNIT 1 1 1\n" in text and "SFAC Bi Se I\n" in text
    assert not any("atom" in key or "formula" in key for key in packet)
    assert Path(__file__).resolve().parents[1] == ROOT


def test_special_position_occupancy_and_output_parser(tmp_path):
    path = tmp_path / "sample_a.res"
    path.write_text("""TITL sample_a.res in I4/mcm
REM SHELXT solution in I4/mcm: R1 0.021, Rweak 0.002
CELL 0.71073 6.0999 6.0999 11.0509 90 90 90
SFAC BR F K
UNIT 4 16 4
BR01 1 .5 0 .5 10.125 .01 30
K002 3 .5 .5 .75 10.125 .01 20
F003 2 .655 .155 .621 10.5 .01 10
HKLF 4
END
Q1 1 .1 .2 .3 11 .01 10
""")
    model = shelxt_candidate(path)
    assert Counter(model["species"]) == {"Br": 4, "F": 16, "K": 4}
    path.write_text(path.read_text().replace("10.125 .01 30", "10.0625 .01 30"))
    with pytest.raises(ValueError, match="occupancy"):
        shelxt_candidate(path)


def test_geometry_matches_elements_counts_hand_origin_and_periodicity():
    cell = [8.0, 9.0, 10.0, 85.0, 100.0, 95.0]
    ref = np.array([[0.1, 0.2, 0.3], [0.4, 0.6, 0.8], [0.7, 0.1, 0.5]])
    species = ["Si", "O", "O"]
    model = {
        "cell": cell,
        "species": ["O", "Si", "O"],
        "fractional_xyz": ((-ref[[2, 0, 1]] + np.array([0.2, 0.3, 0.4])) % 1).tolist(),
    }
    audit = match_geometry(model, species, ref, cell)
    assert audit["correct"] and audit["maximum_error_angstrom"] < 1e-8
    model["species"][0] = "Si"
    assert not match_geometry(model, species, ref, cell)["counts_match"]
    model["species"][0] = "O"
    model["fractional_xyz"][0][0] += 0.2
    assert not match_geometry(model, species, ref, cell)["correct"]


def test_timeout_returns_failure_instead_of_hanging(tmp_path):
    value = launch([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, 0.1)
    assert value["hard_timeout"] and value["returncode"] != 0
    assert value["wall_seconds_including_startup"] < 4


def test_holdout_requires_explicit_open_and_does_not_start_jobs(tmp_path):
    script = ROOT / "benchmarks/solution_corpus_benchmark.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--split",
            "holdout",
            "--output",
            str(tmp_path / "forbidden"),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2 and b"holdout stays sealed" in result.stderr
    assert not (tmp_path / "forbidden").exists()


def test_resource_bound_uses_only_cell_symmetry_and_observed_indices():
    function = run_module.resource_atom_cap
    data = HklData(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    cap, expanded = function(data, [8.0, 9.0, 10.0, 90.0, 90.0, 90.0], "P1")
    assert expanded == 6 and cap == 90
    changed = HklData(data.hkl, [-4.0, 19.0, 99.0], [5.0, 1.0, 2.0])
    assert function(changed, [8.0, 9.0, 10.0, 90.0, 90.0, 90.0], "P1") == (cap, expanded)


def test_latt_and_reduced_symmetry_instructions_reconstruct_all_reference_groups():
    centering = {
        1: [(0, 0, 0)],
        2: [(0, 0, 0), (0.5, 0.5, 0.5)],
        3: [(0, 0, 0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],
        4: [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
        5: [(0, 0, 0), (0, 0.5, 0.5)],
        6: [(0, 0, 0), (0.5, 0, 0.5)],
        7: [(0, 0, 0), (0.5, 0.5, 0)],
    }
    for group in gemmi.spacegroup_table():
        if not group.is_reference_setting():
            continue
        packet = {
            "space_group": group.xhm(),
            "cell": [8, 8, 8, 90, 90, 90],
            "wavelength": 0.71073,
            "elements": ["Si", "O"],
        }
        text = instructions(packet)
        lattice = int(
            next(line.split()[1] for line in text.splitlines() if line.startswith("LATT "))
        )
        ops = [gemmi.Op("x,y,z")] + [
            gemmi.Op(line[5:]) for line in text.splitlines() if line.startswith("SYMM ")
        ]
        if lattice > 0:
            ops += [gemmi.Op("-x,-y,-z").combine(op) for op in ops.copy()]
        expected = {op.wrap().triplet() for op in group.operations()}
        actual = set()
        for op in ops:
            for shift in centering[abs(lattice)]:
                translated = gemmi.Op(op.triplet())
                translated.tran = [round(op.tran[i] + 24 * shift[i]) for i in range(3)]
                actual.add(translated.wrap().triplet())
        assert actual == expected, group.xhm()


@pytest.mark.parametrize("symbol,number", [("P2(1)", 4), ("Pmn2(1)", 31), ("P2(1)/c", 14)])
def test_external_screw_axis_names_are_normalized_for_gemmi(tmp_path, symbol, number):
    path = tmp_path / "sample_a.res"
    path.write_text(f"""TITL sample_a.res
REM SHELXT solution in {symbol}: R1 0.05, Rweak 0.01
CELL 0.71073 8 9 10 90 90 90
SFAC C
C1 1 .137 .259 .381 11 .01 10
HKLF 4
END
""")
    assert shelxt_candidate(path)["space_group_number"] == number


def test_native_reference_settings_match_supplied_gemmi_operations():
    from fractions import Fraction

    from phasesmith.io.space_groups import space_group_by_number

    for number in range(1, 231):
        group = next(
            g for g in gemmi.spacegroup_table() if g.number == number and g.is_reference_setting()
        )
        native = space_group_by_number(number).space_group
        actual = {
            (tuple(op.rotation.ravel()), tuple(x % 1 for x in op.translation))
            for op in native.operations
        }
        expected = {
            (
                tuple((np.array(op.rot, dtype=np.int64) // 24).ravel()),
                tuple(Fraction(v, 24) % 1 for v in op.tran),
            )
            for op in group.operations()
        }
        assert actual == expected, number


def test_external_origin_choice_is_recovered_from_output_operations(tmp_path):
    group = next(
        g for g in gemmi.spacegroup_table() if g.number == 129 and g.is_reference_setting()
    )
    packet = {
        "space_group": group.xhm(),
        "cell": [4.0, 4.0, 9.0, 90.0, 90.0, 90.0],
        "wavelength": 0.71073,
        "elements": ["Si"],
    }
    symm = "\n".join(
        line for line in instructions(packet).splitlines() if line.startswith(("LATT ", "SYMM "))
    )
    path = tmp_path / "sample_a.res"
    path.write_text(f"""TITL sample_a.res
REM SHELXT solution in P4/nmm: R1 0.05, Rweak 0.01
CELL 0.71073 4 4 9 90 90 90
{symm}
SFAC Si
SI01 1 .137 .259 .381 11 .01 10
HKLF 4
END
""")
    model = shelxt_candidate(path)
    assert model["space_group"].endswith(":2")
    expected = []
    for op in group.operations():
        p = np.mod(op.apply_to_xyz([0.137, 0.259, 0.381]), 1)
        if not any(np.linalg.norm((p - q + 0.5) % 1 - 0.5) < 1e-7 for q in expected):
            expected.append(p)
    assert len(model["species"]) == len(expected)


def test_reference_special_position_rounding_keeps_one_crystallographic_site(tmp_path):
    path = tmp_path / "reference.cif"
    path.write_text("""data_reference
_cell_length_a 10
_cell_length_b 10
_cell_length_c 9
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_space_group_name_H-M_alt 'R -3 c'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Sc1 Sc .3333 .6667 .6667 1
""")
    group = gemmi.find_spacegroup_by_name("R -3 c:H")
    block = data_module.atom_block(path)
    assert len(data_module.expanded_sites(block, group)[0]) == 6
    assert len(data_module.expanded_sites(block, group, tolerance=1e-7)[0]) == 36


def test_external_printed_thirds_do_not_create_partial_occupancy(tmp_path):
    group = gemmi.find_spacegroup_by_name("R -3 c:H")
    packet = {
        "space_group": group.xhm(),
        "cell": [10.0, 10.0, 9.0, 90.0, 90.0, 120.0],
        "wavelength": 0.71073,
        "elements": ["Sc"],
    }
    symm = "\n".join(
        line for line in instructions(packet).splitlines() if line.startswith(("LATT ", "SYMM "))
    )
    path = tmp_path / "sample_a.res"
    path.write_text(f"""TITL sample_a.res
REM SHELXT solution in R-3c: R1 0.05, Rweak 0.01
CELL 0.71073 10 10 9 90 90 120
{symm}
SFAC SC
SC1 1 .33333 .66667 .66667 10.16667 .01 10
HKLF 4
END
""")
    assert len(shelxt_candidate(path)["species"]) == 6
