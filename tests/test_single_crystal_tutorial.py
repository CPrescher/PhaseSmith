"""Keep the documented NumPy-only example executable through ordinary builds."""

import json
import runpy
from pathlib import Path

import numpy as np


def test_first_search_example_preserves_inputs_and_writes_a_candidate(tmp_path):
    example = Path(__file__).parents[1] / "examples/single_crystal/first_search.py"
    tutorial = runpy.run_path(str(example))
    output = tmp_path / "first-search"
    report = tutorial["run"](output)
    assert report["spots"] == 1 and report["input_reflections"] == 196
    assert report["target_met"] and report["candidates"] == 1
    # Ideal one-atom data, rounded only by text I/O; not a measured recovery gate.
    assert report["best_r1"] < 1e-6
    assert report["truth_coordinates_supplied_to_search"] is False
    saved = json.loads((output / "search/result.json").read_text())
    assert saved["source_sha256"]
    assert (output / "search/candidate-p1.cif").is_file()
    with np.load(output / "search/density-and-observations.npz", allow_pickle=False) as arrays:
        assert arrays["input_hkl"].shape == (196, 3)
        np.testing.assert_array_equal(arrays["input_sigma"], np.ones(196))
