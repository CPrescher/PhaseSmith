import json
import os
from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import (
    IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES,
    convert_iucr_trirubidium_citrate_silicon_bundle,
)
from phasesmith.validation import run_iucr_trirubidium_citrate_silicon_workflow


def test_real_iucr_trirubidium_citrate_silicon_conversion_when_configured(
    tmp_path: Path,
) -> None:
    source = os.environ.get("PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip("set PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA for the external test")
    bundle = tmp_path / "bundle"
    manifest_path = convert_iucr_trirubidium_citrate_silicon_bundle(source, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pattern = np.loadtxt(bundle / "pattern.csv", delimiter=",", skiprows=1)

    assert pattern.shape == (4106, 4)
    assert manifest["data_selection"]["source_index_interval_inclusive"] == [594, 4699]
    assert manifest["data_selection"]["exclusion_reason"].startswith("beam spillover")
    assert manifest["legacy_gsas_reference"]["rwp"] == pytest.approx(0.02458337665)
    assert manifest["legacy_gsas_reference"]["rp"] == pytest.approx(0.01950459520)
    assert manifest["legacy_gsas_reference"]["weight_fractions"] == {
        "silicon": 0.0215,
        "trirubidium_citrate": 0.9785,
    }
    assert manifest["instrument"]["goniometer_radius_mm"] == pytest.approx(141.5)
    assert manifest["instrument"]["radius_is_source_deposited"] is True
    assert manifest["instrument"]["matched_sh_over_l"] == pytest.approx(0.0194)
    assert manifest["instrument"]["common_base_profile"] == {
        "U": 0.0,
        "V": 0.0,
        "W": 5.109,
        "X": 3.634,
        "Y": 0.0,
    }
    assert manifest["instrument"]["source_translated_sample_displacement_mm"] == pytest.approx(
        -0.10805049346761132
    )
    assert manifest["instrument"]["legacy_manual_physical_sample_shift_mm"] == pytest.approx(
        0.10805049346761132
    )
    assert manifest["instrument"]["translated_gsasii_transparency_field_cm"] == pytest.approx(
        0.6421066318087139
    )
    assert {path.stem for path in bundle.glob("*.cif")} == set(
        IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES
    )
    silicon = (bundle / "silicon.cif").read_text(encoding="utf-8")
    assert "normalization of IUCr RAMM077C_phase_2" in silicon
    assert "Si1 Si 0.125 0.125 0.125 1.0 0.01" in silicon


def test_iucr_trirubidium_converter_rejects_wrong_archive(tmp_path: Path) -> None:
    source = tmp_path / "wrong.cif"
    source.write_text("data_unrelated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed RAMM077C archive"):
        convert_iucr_trirubidium_citrate_silicon_bundle(source, tmp_path / "bundle")


def test_real_iucr_trirubidium_common_model_when_configured(tmp_path: Path) -> None:
    source = os.environ.get("PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip("set PHASESMITH_IUCR_TRIRUBIDIUM_CITRATE_SILICON_DATA for the external test")
    bundle = tmp_path / "bundle"
    convert_iucr_trirubidium_citrate_silicon_bundle(source, bundle)

    result = run_iucr_trirubidium_citrate_silicon_workflow(bundle)

    assert result.sample_count == 4106
    assert result.free_parameter_count == 3
    assert set(result.weight_fractions) == set(IUCR_TRIRUBIDIUM_CITRATE_SILICON_PHASES)
    assert sum(result.weight_fractions.values()) == pytest.approx(1.0)
    assert 0.0 < result.profile_correlation < 1.0
    assert 0.0 < result.poisson_rwp < 1.0
    assert result.refined_instrument["Zero"] == pytest.approx(0.0)
    assert result.refined_instrument["W"] == pytest.approx(5.109)
    assert result.refined_instrument["X"] == pytest.approx(3.634)
    assert result.refined_instrument["Y"] == pytest.approx(0.0)
    assert result.legacy_curve_poisson_rwp == pytest.approx(0.02458326145)
    assert len(result.model_qualifications) == 8
