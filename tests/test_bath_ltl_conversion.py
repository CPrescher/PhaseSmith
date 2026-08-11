import json
from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import convert_bath_ltl_bundle, read_rigaku_asc_text
from phasesmith.validation import run_bath_ltl_workflow


def asc_text(*, count: int = 4, stop: float = 1.3) -> str:
    return f"""*TYPE = Raw
*I_MONOCHRO = Ge(220)x2, 0.000000
*XRAY_CHAR = K-ALPHA1
*WAVE_LENGTH1 = 1.54059
*WAVE_LENGTH2 = 1.54441
*XUNIT = deg.
*YUNIT = counts
*START = 1.0
*STOP = {stop}
*STEP = 0.1
*COUNT = {count}
1, 2, 3, 4
*END
"""


def test_rigaku_asc_parser_validates_packed_count_axis() -> None:
    pattern = read_rigaku_asc_text(asc_text())

    np.testing.assert_allclose(pattern.two_theta_deg, [1.0, 1.1, 1.2, 1.3])
    np.testing.assert_array_equal(pattern.counts, [1.0, 2.0, 3.0, 4.0])
    assert pattern.metadata["*XRAY_CHAR"] == "K-ALPHA1"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (asc_text(count=5), "count or step"),
        (asc_text(stop=1.4), "inconsistent"),
        (asc_text().replace("1, 2, 3, 4", "1, -2, 3, 4"), "nonnegative"),
    ],
)
def test_rigaku_asc_parser_rejects_inconsistent_scans(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_rigaku_asc_text(text)


def test_real_bath_bundle_conversion_and_workflow_when_configured(tmp_path: Path) -> None:
    source_text = __import__("os").environ.get("PHASESMITH_BATH_LTL_DATA")
    if not source_text:
        pytest.skip("set PHASESMITH_BATH_LTL_DATA for the external Bath conversion test")
    bundle = tmp_path / "bundle"
    manifest_path = convert_bath_ltl_bundle(source_text, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["scope"] == "bath_ltl_lab_xray_conversion_fidelity"
    assert manifest["samples"]["K"]["raw_scan"]["radiation"] == "K-ALPHA1"
    result = run_bath_ltl_workflow(bundle, "K", profile_cycles=0)
    assert result.sample_count == 2698
    assert result.released_poisson_rwp == pytest.approx(0.07791200480489903)
