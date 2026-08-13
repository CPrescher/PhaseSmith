import json
import os
from pathlib import Path

import numpy as np
import pytest
from phasesmith.io import (
    IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES,
    convert_iucr_tripotassium_citrate_silicon_bundle,
)
from phasesmith.validation import run_iucr_tripotassium_citrate_silicon_workflow


def test_real_iucr_tripotassium_citrate_silicon_conversion_when_configured(
    tmp_path: Path,
) -> None:
    source = os.environ.get("PHASESMITH_IUCR_TRIPOTASSIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip(
            "set PHASESMITH_IUCR_TRIPOTASSIUM_CITRATE_SILICON_DATA for the external IUCr test"
        )
    bundle = tmp_path / "bundle"
    manifest_path = convert_iucr_tripotassium_citrate_silicon_bundle(source, bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pattern = np.loadtxt(bundle / "pattern.csv", delimiter=",", skiprows=1)

    assert pattern.shape == (2696, 4)
    assert manifest["data_selection"]["source_index_interval_inclusive"] == [520, 3215]
    assert manifest["legacy_gsas_reference"]["rwp"] == pytest.approx(0.04852875265)
    assert manifest["legacy_gsas_reference"]["rp"] == pytest.approx(0.03808682863)
    assert manifest["legacy_gsas_reference"]["weight_fractions"] == {
        "silicon": 0.0144,
        "tripotassium_citrate": 0.9856,
    }
    assert manifest["silicon_standard"]["fixed_lattice_a_angstrom"] == 5.43105
    assert manifest["silicon_standard"]["source_does_not_identify_nist_srm"] is True
    assert manifest["instrument"]["matched_sh_over_l"] == pytest.approx(0.0368)
    assert manifest["instrument"]["initial_zero_deg"] == pytest.approx(0.09073)
    assert manifest["instrument"]["assumed_goniometer_radius_mm"] == pytest.approx(141.5)
    assert manifest["instrument"]["silicon_profile"] == {
        "U": 1.949,
        "V": 0.0,
        "W": 1.735,
        "X": 2.372,
        "Y": 11.329,
    }
    assert manifest["instrument"]["tripotassium_citrate_profile"] == {
        "U": 2.58,
        "V": 0.0,
        "W": 1.999,
        "X": 0.0,
        "Y": 2.708,
    }
    assert (
        "second-order generalized spherical-harmonic texture (reported index 1.001)"
        in manifest["translation_diagnostics"]["source_only_terms"]
    )
    assert {path.stem for path in bundle.glob("*.cif")} == set(
        IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES
    )
    silicon = (bundle / "silicon.cif").read_text(encoding="utf-8")
    assert "normalization of IUCr KADU1578_phase_2" in silicon
    assert "Si1 Si 0.125 0.125 0.125 1.0 0.01" in silicon


def test_iucr_tripotassium_citrate_converter_rejects_wrong_archive(tmp_path: Path) -> None:
    source = tmp_path / "wrong.cif"
    source.write_text("data_unrelated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed KADU1578 archive"):
        convert_iucr_tripotassium_citrate_silicon_bundle(source, tmp_path / "bundle")


def test_real_iucr_tripotassium_common_model_when_configured(tmp_path: Path) -> None:
    source = os.environ.get("PHASESMITH_IUCR_TRIPOTASSIUM_CITRATE_SILICON_DATA")
    if not source:
        pytest.skip(
            "set PHASESMITH_IUCR_TRIPOTASSIUM_CITRATE_SILICON_DATA for the external IUCr test"
        )
    bundle = tmp_path / "bundle"
    convert_iucr_tripotassium_citrate_silicon_bundle(source, bundle)

    result = run_iucr_tripotassium_citrate_silicon_workflow(bundle)

    assert result.sample_count == 2696
    assert result.free_parameter_count == 3
    assert set(result.weight_fractions) == set(IUCR_TRIPOTASSIUM_CITRATE_SILICON_PHASES)
    assert sum(result.weight_fractions.values()) == pytest.approx(1.0)
    assert 0.60 < result.profile_correlation < 0.70
    assert 0.20 < result.poisson_rwp < 0.25
    assert result.refined_instrument["Zero"] == pytest.approx(0.09073)
    assert result.refined_instrument["U"] == pytest.approx(2.58)
    assert result.refined_instrument["Y"] == pytest.approx(2.708)
    assert len(result.model_qualifications) == 7
