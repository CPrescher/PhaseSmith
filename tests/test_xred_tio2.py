import json
import os
from pathlib import Path

import pytest
from phasesmith.io import convert_xred_tio2_bundle
from phasesmith.validation import run_xred_tio2_workflow


def test_real_xred_conversion_and_workflow_when_configured(tmp_path: Path) -> None:
    source_text = os.environ.get("PHASESMITH_XRED_TIO2_DATA")
    if not source_text:
        pytest.skip("set PHASESMITH_XRED_TIO2_DATA for the external XRED test")
    bundle = tmp_path / "bundle"
    manifest_path = convert_xred_tio2_bundle(source_text, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["setting_translation"]["matched_group"] == "I 41/a m d :1"
    result = run_xred_tio2_workflow(bundle, cycles=0)
    assert result.sample_count == 2501
    assert set(result.weight_fractions) == {"anatase", "rutile"}
    assert sum(result.weight_fractions.values()) == pytest.approx(1.0)
