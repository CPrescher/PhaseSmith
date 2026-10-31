"""Published native example assets must preserve their frozen input contracts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "crates/phasesmith-rs/examples/data"


def test_native_example_controls_preserve_the_reviewed_v1_recipe():
    bundled = json.loads((ASSETS / "native-workflow-controls-v1.json").read_text())
    frozen = json.loads((ROOT / "validation/enstatite-raw-workflow-v25.json").read_text())
    assert bundled == {key: frozen[key] for key in bundled}


def test_native_source_record_contains_only_registered_input_priors_and_identity():
    bundled = json.loads((ASSETS / "mgsio3-source.json").read_text())
    source = json.loads((ROOT / "validation/enstatite-mgsio3-local-v1.json").read_text())
    assert set(bundled) == {"raw_array", "priors", "files"}
    assert bundled["priors"] == source["search_inputs"]
    assert bundled["raw_array"] == source["raw_array"]
    assert bundled["files"] == [
        row for row in source["files"] if row["role"] in {"raw_frames", "initial_geometry"}
    ]


def test_native_null_rotations_are_the_exact_predeclared_pre_frame_fixture():
    assert (ASSETS / "grain-null-rotations-v1.json").read_bytes() == (
        ROOT / "validation/grain-null-rotations-v1.json"
    ).read_bytes()
