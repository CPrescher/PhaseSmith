"""Pinned-table initialization and external Pawley comparison contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from phasesmith.oracle import load_fixture
from phasesmith.oracle._pinned_probe import PINNED_REVISION, initialize_pawley
from phasesmith.validation.pawley_oracle import compare_fixture

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "oracle/fixtures/pawley_optimizer_v1"


def records():
    phase = SimpleNamespace(
        data={
            "General": {"doPawley": False, "Cell": [False, 3.0, 3.0, 3.0, 90.0, 90.0, 90.0, 27.0]},
            "Pawley ref": [],
        }
    )
    histogram = SimpleNamespace(
        data={"data": [None, [np.array([20.0, 30.0]), np.zeros(2), np.zeros(2)]]}
    )
    refs = np.zeros((1, 15))
    refs[0, :5] = [1, 0, 0, 2, 3.0]
    refs[0, 9] = 8.0
    return phase, histogram, refs


def test_private_initializer_is_revision_gated_and_checks_schema_before_writing(monkeypatch):
    monkeypatch.setattr("phasesmith.oracle._pinned_probe.detected_revision", lambda _: "wrong")
    phase, histogram, refs = records()
    with pytest.raises(RuntimeError, match="requires"):
        initialize_pawley(phase, histogram, None, refs, [1.0, 2.0])
    assert not phase.data["General"]["doPawley"]
    monkeypatch.setattr(
        "phasesmith.oracle._pinned_probe.detected_revision", lambda _: PINNED_REVISION
    )
    before = copy.deepcopy(phase.data)
    with pytest.raises(ValueError, match="schema"):
        initialize_pawley(phase, histogram, None, refs[:, :14], [1.0, 2.0])
    with pytest.raises(TypeError, match="real"):
        initialize_pawley(phase, histogram, None, refs.astype(complex), [1.0, 2.0])
    assert phase.data == before
    initialize_pawley(
        phase, histogram, None, refs, [1.0, 2.0], starting_cell=[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
    )
    assert phase.data["Pawley ref"] == [[1, 0, 0, 2, 3.0, True, 4.0, 0.0]]
    assert phase.data["General"]["Cell"][7] == pytest.approx(64.0)
    np.testing.assert_array_equal(histogram.data["data"][1][1], [1.0, 2.0])
    np.testing.assert_array_equal(histogram.data["data"][1][2], [1.0, 1.0])


def test_pinned_optimizer_fixture_and_profile_model_agreement():
    fixture = load_fixture(FIXTURE)
    assert fixture.manifest["provenance"]["gsasii_revision"] == PINNED_REVISION
    assert fixture.manifest["provenance"]["binary_sha256"]
    assert len(fixture.cases) == 2
    for case in fixture.cases:
        assert case["parameters"]["method"] == "Pawley"
        assert case["parameters"]["signed_f_squared"]
    report = compare_fixture(FIXTURE)
    assert report["passed"]
    for case in report["results"]:
        assert case["termination"] == "converged"
        assert case["isolated_reflections"] == 42
        assert len(case["overlap_groups"]) == 5
        # The stricter equivalence diagnostic is kept separate and is not relabelled.
        assert case["strict_fixed_profile_equivalence"] == (
            case["fixed_profile_relative_l2"] < 1e-5
        )


def test_fixed_profile_diagnosis_separates_axial_kernel_and_cutoffs():
    from phasesmith.validation.pawley_profile_diagnostic import investigate

    report = investigate(FIXTURE, ROOT / "oracle/diagnostics/pawley-profile-controls-20260917")
    assert report["passed"]
    metrics = report["metrics"]
    assert metrics["original_full_pattern_relative_l2"] == pytest.approx(
        3.691309812920e-4, rel=1e-7
    )
    assert (
        metrics["axial_kernel_component_relative_l2"] > 30 * metrics["cutoff_component_relative_l2"]
    )
    assert metrics["histogram_reconstruction_relative_l2"] < 1e-8
    assert metrics["diagnostic_translation_area_and_cutoff_relative_l2"] < 1e-6
    assert metrics["symmetric_kernel_relative_l2"] < 3e-6
    assert metrics["oracle_basis_area_max_relative_error"] < 2e-7
    assert max(q["native_vs_order128"] for q in report["independent_quadrature"]) < 2e-11
    # A 100-fold refinement of observation spacing leaves the isolated mismatch unchanged.
    errors = [g["relative_l2"] for g in report["grid_checks"]]
    assert max(errors) - min(errors) < 3e-8


def test_fcj_diagnostic_probe_checks_revision_before_private_calls(monkeypatch):
    from phasesmith.oracle import _pinned_probe as probe

    monkeypatch.setattr(probe, "detected_revision", lambda module: "wrong")
    with pytest.raises(RuntimeError, match="requires"):
        probe.probe_fcj_profile_and_support(
            SimpleNamespace(),
            [20.0],
            position_deg=20.0,
            sigma2_centideg2=1.0,
            gamma_centideg=0.1,
            axial_sum=0.002,
        )
