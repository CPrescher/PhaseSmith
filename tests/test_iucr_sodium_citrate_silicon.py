import json
import os
from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import (
    IUCR_SODIUM_CITRATE_SILICON_PHASES,
    convert_iucr_sodium_citrate_silicon_bundle,
)


def test_real_iucr_sodium_citrate_silicon_conversion_when_configured(
    tmp_path: Path,
) -> None:
    source = os.environ.get("PHASESMITH_IUCR_SODIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip("set PHASESMITH_IUCR_SODIUM_CITRATE_SILICON_DATA for the external IUCr test")
    bundle = tmp_path / "bundle"
    manifest_path = convert_iucr_sodium_citrate_silicon_bundle(source, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pattern = np.loadtxt(bundle / "pattern.csv", delimiter=",", skiprows=1)

    assert pattern.shape == (4452, 4)
    assert manifest["data_selection"]["source_index_interval_inclusive"] == [248, 4699]
    assert manifest["legacy_gsas_reference"]["rwp"] == pytest.approx(0.08432524796)
    assert manifest["legacy_gsas_reference"]["rp"] == pytest.approx(0.06315980737)
    assert manifest["legacy_gsas_reference"]["weight_fractions"] == {
        "silicon": 0.1874,
        "sodium_dihydrogen_citrate": 0.8126,
    }
    assert manifest["silicon_standard"]["fixed_lattice_a_angstrom"] == 5.43105
    assert (
        "main-phase generalized spherical-harmonic preferred orientation"
        in manifest["translation_diagnostics"]["source_only_terms"]
    )
    assert {path.stem for path in bundle.glob("*.cif")} == set(IUCR_SODIUM_CITRATE_SILICON_PHASES)
    silicon = (bundle / "silicon.cif").read_text(encoding="utf-8")
    assert "normalization of IUCr RAMM012A_phase_2" in silicon
    assert "Si1 Si 0.125 0.125 0.125 1.0 0.030352" in silicon


def test_iucr_sodium_citrate_converter_rejects_wrong_archive(tmp_path: Path) -> None:
    source = tmp_path / "wrong.cif"
    source.write_text("data_unrelated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed RAMM012A archive"):
        convert_iucr_sodium_citrate_silicon_bundle(source, tmp_path / "bundle")
