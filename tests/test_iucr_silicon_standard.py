import json
import os
from pathlib import Path

import pytest
from phasesmith.io import IUCR_SILICON_PHASES, convert_iucr_silicon_standard_bundle
from phasesmith.validation import run_iucr_silicon_standard_workflow


def test_real_iucr_silicon_standard_conversion_and_workflow_when_configured(
    tmp_path: Path,
) -> None:
    source = os.environ.get("PHASESMITH_IUCR_SILICON_DATA")
    if not source:
        pytest.skip("set PHASESMITH_IUCR_SILICON_DATA for the external IUCr test")
    bundle = tmp_path / "bundle"
    manifest_path = convert_iucr_silicon_standard_bundle(source, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["data_selection"]["refined_sample_count"] == 2820
    assert manifest["legacy_gsas_reference"]["rwp"] == pytest.approx(0.0622569592)
    assert manifest["legacy_gsas_reference"]["weight_fractions"]["silicon"] == 0.1302
    assert manifest["silicon_calibration"]["role"] == "absolute specimen-displacement standard"
    assert "111" in manifest["silicon_calibration"]["excluded_silicon_reflections"]
    assert {path.stem for path in bundle.glob("*.cif")} == set(IUCR_SILICON_PHASES)

    result = run_iucr_silicon_standard_workflow(bundle, cycles=1)
    assert result.sample_count == 2820
    assert set(result.weight_fractions) == set(IUCR_SILICON_PHASES)
    assert sum(result.weight_fractions.values()) == pytest.approx(1.0)
    assert result.poisson_rwp < 0.16
    assert abs(result.weight_fractions["silicon"] - 0.1302) < 0.04
    assert result.refined_instrument["Zero"] == pytest.approx(-0.0448)
    assert result.refined_instrument["Zero"] == result.calibrated_zero_shift_deg
    assert -0.3 < result.calibrated_sample_displacement_mm < -0.05
