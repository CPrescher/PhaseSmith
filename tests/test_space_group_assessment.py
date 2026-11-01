"""Independent equations and optional native-executable symmetry cross-comparison."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from phasesmith.single_crystal.reference.space_group_search import assess, prior_eligible


def _settings():
    identity = dict(rotation=np.eye(3, dtype=int).tolist(), translation=[[0, 1]] * 3)
    twofold = dict(rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[[0, 1]] * 3)
    screw = dict(twofold, translation=[[0, 1], [1, 2], [0, 1]])
    return [dict(operations=[identity, twofold]), dict(operations=[identity, screw])]


def _rows():
    return np.array(
        [
            [h, k, ell, -0.5 if h == ell == 0 and k % 2 else 100 + h * h + k * k + ell * ell, 2]
            for h in range(-2, 3)
            for k in range(-3, 4)
            for ell in range(-2, 3)
            if (h, k, ell) != (0, 0, 0)
        ],
        dtype=float,
    )


def test_reference_screw_evidence_retains_negative_measurements_and_missing_ambiguity():
    rows = _rows()
    result = assess(rows, [7, 8, 9, 90, 103, 90], _settings())
    assert result["settings"][1]["absence_log_evidence"] > 0
    assert result["settings"][1]["strong_violations"] == 0
    measured = rows[~((rows[:, 0] == 0) & (rows[:, 2] == 0) & (rows[:, 1] % 2 != 0))]
    result = assess(measured, [7, 8, 9, 90, 103, 90], _settings())
    assert result["settings"][1]["absent_families"] == 0
    assert result["settings"][1]["absence_log_evidence"] == 0


@pytest.mark.parametrize("column,value", [(0, 1.5), (3, np.nan), (4, 0)])
def test_reference_rejects_invalid_observations(column, value):
    rows = _rows()
    rows[0, column] = value
    with pytest.raises(ValueError):
        assess(rows, [7, 8, 9, 90, 103, 90], _settings())


@pytest.mark.parametrize(
    "cell",
    [
        [7.1, 8.3, 9.7, 90, 90, 90],
        [7.1, 8.3, 9.7, 90, 103.7, 90],
        [7.1, 7.1, 9.7, 90, 90, 120],
        [5.3, 5.3, 5.3, 90, 90, 90],
    ],
)
def test_native_catalogue_and_scores_against_independent_randomized_reference(tmp_path, cell):
    executable = os.environ.get("PHASESMITH_NATIVE_WORKFLOW")
    if not executable:
        pytest.skip("set PHASESMITH_NATIVE_WORKFLOW for native cross-comparison")
    executable = Path(executable).resolve()
    if os.name == "nt" and not executable.exists():
        executable = executable.with_suffix(".exe")
    rng = np.random.default_rng(840271)
    rows = np.array(
        [
            [h, k, ell, rng.normal(5, 8), rng.uniform(0.7, 2.5)]
            for h in range(-3, 4)
            for k in range(-3, 4)
            for ell in range(-3, 4)
            if (h, k, ell) != (0, 0, 0) and rng.random() > 0.25
        ]
    )
    packet, output = tmp_path / "input.json", tmp_path / "native.json"
    packet.write_text(json.dumps(dict(cell=cell, observations=rows.tolist())))
    subprocess.run([str(executable), "--assess-symmetry", str(packet), str(output)], check=True)
    native = json.loads(output.read_text())
    reference = assess(rows, cell, native["settings"])
    assert native["measured_rows"] == reference["measured_rows"]
    assert native["friedel_families"] == reference["friedel_families"]
    for actual, expected in zip(native["laue"], reference["laue"], strict=True):
        assert actual["status"] == expected["status"]
        assert actual["constraints"] == expected["constraints"]
        assert actual["families"] == expected["families"]
        for key in ["reduced_chi_square", "r_int"]:
            if expected[key] is None:
                assert actual[key] is None
            else:
                assert actual[key] == pytest.approx(expected[key], rel=1e-11, abs=1e-10)
    for actual, expected in zip(native["settings"], reference["settings"], strict=True):
        assert actual["laue"] == expected["laue"]
        assert actual["absent_families"] == expected["absent_families"]
        assert actual["strong_violations"] == expected["strong_violations"]
        assert actual["absence_log_evidence"] == pytest.approx(
            expected["absence_log_evidence"], rel=1e-11, abs=1e-8
        )


@pytest.mark.parametrize("number", [2, 4, 56, 139, 167, 194, 221, 225])
def test_supplied_prior_eligibility_matches_exact_independent_reference(tmp_path, number):
    import gemmi

    from benchmarks.unknown_group_corpus import (
        symmetry_instructions,
        symmetry_prior_from_instructions,
    )

    executable = os.environ.get("PHASESMITH_NATIVE_WORKFLOW")
    if not executable:
        pytest.skip("set PHASESMITH_NATIVE_WORKFLOW for native cross-comparison")
    group = gemmi.find_spacegroup_by_number(number)
    system = group.crystal_system_str()
    cell = {
        "triclinic": [7, 8, 9, 83, 101, 96],
        "monoclinic": [7, 8, 9, 90, 103, 90],
        "orthorhombic": [7, 8, 9, 90, 90, 90],
        "tetragonal": [7, 7, 9, 90, 90, 90],
        "trigonal": [7, 7, 9, 90, 90, 120],
        "hexagonal": [7, 7, 9, 90, 90, 120],
        "cubic": [7, 7, 7, 90, 90, 90],
    }[system]
    prior = symmetry_prior_from_instructions(
        symmetry_instructions(
            {"space_group": group.xhm(), "cell": cell, "wavelength": 0.71, "elements": ["C", "O"]}
        )
    )
    # No measured non-Friedel equivalents: the prior must not fabricate support.
    rows = [[h, 0, 0, 100 - h, 2] for h in range(1, 21)]
    packet, output = tmp_path / "prior.json", tmp_path / "assessed.json"
    packet.write_text(json.dumps(dict(cell=cell, observations=rows, symmetry_prior=prior)))
    subprocess.run([executable, "--assess-symmetry", str(packet), str(output)], check=True)
    actual = json.loads(output.read_text())
    assert actual["prior_eligible"] == prior_eligible(actual["settings"], prior)
    assert any(actual["prior_eligible"])
    processing = actual["laue"][actual["processing_laue"]]
    assert processing["rotations"] == json.loads(json.dumps(prior["laue_rotations"]))
    assert processing["constraints"] == 0
    assert processing["status"] in {"insufficient_data", "friedel_only"}
    assert actual["measured_rows"] == 20
