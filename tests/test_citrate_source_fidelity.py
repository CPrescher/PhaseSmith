from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from phasesmith.io import convert_iucr_trirubidium_citrate_silicon_bundle
from phasesmith.validation import run_citrate_source_reflection_fidelity


def test_source_reflection_fidelity_clears_low_angle_structure_translation(
    tmp_path: Path,
) -> None:
    source = os.environ.get("PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip("set PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA for the external test")
    bundle = tmp_path / "bundle"
    convert_iucr_trirubidium_citrate_silicon_bundle(source, bundle)
    result = run_citrate_source_reflection_fidelity(bundle)

    assert result.source_row_count == 1197
    assert result.unique_reflection_count == 600
    assert result.duplicate_wavelength_f_squared_max_abs_delta == 0.0
    assert all(value is True for key, value in result.review.items() if key != "conclusion")
    rubidium = result.phases["trirubidium_citrate"]
    assert rubidium["reflection_count"] == 591
    assert rubidium["low_angle_reflection_count"] == 20
    assert rubidium["d_spacing_max_abs_error_angstrom"] < 2.0e-5
    source_low = rubidium["source_cromer_mann"]["low_angle"]
    assert source_low["median_abs_relative_error"] < 0.002
    assert source_low["p95_abs_relative_error"] < 0.006
    assert source_low["source_weighted_l1_relative_error"] < 0.0003
    production_low = rubidium["production_waasmaier_kirfel"]["low_angle"]
    assert production_low["source_weighted_l1_relative_error"] < 0.0031
    assert source_low["p95_abs_relative_error"] < production_low["p95_abs_relative_error"]


def test_source_reflection_fidelity_rejects_incompatible_bundle(tmp_path: Path) -> None:
    (tmp_path / "experiment.json").write_text(
        '{"scope":"unrelated","source_reflections":{}}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reviewed rubidium bundle"):
        run_citrate_source_reflection_fidelity(tmp_path)


def test_reviewed_source_reflection_report_records_small_weighted_error() -> None:
    report_path = (
        Path(__file__).resolve().parents[1]
        / "validation/results/2026-08-13-citrate-rubidium-source-reflection-fidelity.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source_row_count"] == 1197
    assert report["unique_reflection_count"] == 600
    assert all(value is True for key, value in report["review"].items() if key != "conclusion")
    rubidium = report["phases"]["trirubidium_citrate"]
    assert rubidium["source_cromer_mann"]["low_angle"][
        "source_weighted_l1_relative_error"
    ] == pytest.approx(0.00028926706155009174)
    assert rubidium["production_waasmaier_kirfel"]["low_angle"][
        "source_weighted_l1_relative_error"
    ] == pytest.approx(0.003006209939580019)
