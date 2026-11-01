"""Application boundary for the complete native file coordinator."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from phasesmith import _core
from phasesmith.single_crystal import workflow
from phasesmith.single_crystal.solution import SolutionCancellation
from phasesmith.single_crystal.workflow import WorkflowConfig, available, run_workflow


def config(tmp_path):
    return WorkflowConfig(
        str(tmp_path / "raw.h5"),
        str(tmp_path / "geometry.poni"),
        str(tmp_path / "new"),
        "/entry/eiger/data",
        (8.0, 5.0, 18.0, 90.0, 90.0, 90.0),
        ("Mg", "Si", "O"),
        2,
        143,
        -36.0,
        0.5,
        0.5,
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        35.0,
        True,
    )


def test_ordinary_build_exposes_the_native_coordinator():
    assert available()
    assert hasattr(_core, "_single_crystal_workflow_protocol")


def test_recipe_roundtrip_and_domain_validation(tmp_path):
    c = config(tmp_path)
    assert WorkflowConfig.from_record(json.loads(json.dumps(c.to_record()))) == c
    for field, value in [
        ("last", 2),
        ("first", True),
        ("omega_step", 0.0),
        ("rotation_axis", (2.0, 0.0, 0.0)),
        ("cell", (8, 5, 18, 90, 100, 90)),
        ("elements", ("Mg", "Mg")),
        ("dac_half_opening", 90.0),
    ]:
        with pytest.raises((ValueError, TypeError)):
            replace(c, **{field: value})
    with pytest.raises(TypeError):
        WorkflowConfig.from_record({**c.to_record(), "atomic_counts": [16, 16, 48]})


def test_packet_withholds_exact_group_and_atomic_reference_and_preserves_acquisition(
    tmp_path, monkeypatch
):
    class Metadata:
        shape = (144, 2162, 2068)

        def __init__(self, *args):
            assert Path(args[0]).is_absolute()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(workflow, "Hdf5Frames", Metadata)
    monkeypatch.chdir(tmp_path)
    p = workflow._packet(replace(config(tmp_path), raw_path="raw.h5"))
    assert set(p["priors"]) == {"cell_angstrom_degrees", "elements", "provenance"}
    a = p["protocol"]["acquisition"]
    assert (a["first_frame"], a["last_frame_inclusive"], a["vertical_flip"]) == (2, 143, True)
    assert (
        a["omega_at_original_frame_zero_degrees"],
        a["step_degrees"],
        a["exposure_width_degrees"],
    ) == (-36.0, 0.5, 0.5)
    assert p["protocol"]["space_group_search"]["effort"]["mode"] == "auto"
    assert p["raw_array"]["shape"] == (144, 2162, 2068)


def test_precancel_is_effective_before_any_input_or_output_io(tmp_path, monkeypatch):
    token = SolutionCancellation()
    token.cancel()
    monkeypatch.setattr(workflow, "_packet", lambda _: pytest.fail("input I/O after cancellation"))
    with pytest.raises(InterruptedError):
        run_workflow(config(tmp_path), cancellation=token)
    with pytest.raises(InterruptedError):
        _core._run_single_crystal_workflow("{}", tmp_path / "native", token._native)
    assert not (tmp_path / "native").exists()


def test_frozen_application_protocol_matches_validated_v40():
    root = Path(__file__).resolve().parents[1]
    assert json.loads(_core._single_crystal_workflow_protocol()) == json.loads(
        (root / "validation/enstatite-raw-workflow-v40.json").read_text()
    )
