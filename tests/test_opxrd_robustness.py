from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from phasesmith.validation.opxrd import (
    OpxrdCaseResult,
    OpxrdRejectedCaseResult,
    run_opxrd_robustness_campaign,
    run_opxrd_structural_common_model,
    write_opxrd_common_model_bundle,
)


def _encoded_pattern(x: np.ndarray, y: np.ndarray, *, structural: bool) -> bytes:
    atoms = [json.dumps({"x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0, "symbol": "Si0+"})]
    phase = {
        "lengths": [5.43, 5.43, 5.43] if structural else ["nan"] * 3,
        "angles": [90.0, 90.0, 90.0] if structural else ["nan"] * 3,
        "base": json.dumps(atoms if structural else []),
        "phase_fraction": 1.0,
        "chemical_composition": "Si" if structural else None,
        "spacegroup": 1 if structural else None,
    }
    label = {
        "phases": [json.dumps(phase)],
        "xray_info": json.dumps({"primary_wavelength": 1.5444, "secondary_wavelength": 1.5406}),
    }
    outer = {
        "two_theta_values": x.tolist(),
        "intensities": y.tolist(),
        "label": json.dumps(label),
        "metadata": json.dumps({"fixture": True}),
    }
    return json.dumps(outer, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    x = np.linspace(10.0, 80.0, 201)
    positive = 20.0 + 300.0 * np.exp(-0.5 * ((x - 28.4) / 0.18) ** 2)
    irregular_x = x.copy()
    irregular_x[101:] += 0.02
    negative = positive - 25.0
    nonmonotonic_x = x.copy()
    nonmonotonic_x[120:] -= 20.0
    payloads = {
        "source/positive.json": _encoded_pattern(x, positive, structural=True),
        "source/negative.json": _encoded_pattern(irregular_x, negative, structural=False),
        "source/nonmonotonic.json": _encoded_pattern(nonmonotonic_x, positive, structural=False),
    }
    archive = tmp_path / "opxrd.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, payload in payloads.items():
            target.writestr(name, payload)
    cases = []
    definitions = (
        ("positive", "positive.json", "full_structure", "structural_common_model"),
        ("negative", "negative.json", "unlabeled", "negative_intensity"),
        ("nonmonotonic", "nonmonotonic.json", "partial_structure", "metadata_boundary"),
    )
    for case_id, filename, metadata_class, role in definitions:
        member = f"source/{filename}"
        cases.append(
            {
                "case_id": case_id,
                "expected_findings": ["synthetic test fixture"],
                "member_path": member,
                "member_sha256": hashlib.sha256(payloads[member]).hexdigest(),
                "metadata_class": metadata_class,
                "role": role,
                "source_group": "source",
            }
        )
    manifest = {
        "archive": {
            "filename": "opxrd.zip",
            "member_count": 3,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "size_bytes": archive.stat().st_size,
            "zenodo_record": 1,
        },
        "cases": cases,
        "dataset_id": "opxrd-test",
        "procedure": {
            "background": {
                "chebyshev_order": None,
                "iterations": 5,
                "smooth_width_deg": 0.7,
            },
            "independent_baseline": {"bin_count": 16, "quantile": 0.2},
            "peak_detection": {
                "minimum_separation_deg": 0.2,
                "noise_multiplier": 4.0,
                "relative_height_floor": 0.01,
            },
            "structural_common_model": {
                "axial_sample_over_radius": 0.001,
                "axial_detector_over_radius": 0.001,
                "background_terms": 4,
                "doublet_secondary_to_primary_intensity": 0.5,
                "initial_profile_deg2": [0.0002, -0.0002, 0.0005],
                "large_gaussian_fwhm_warning_deg": 0.5,
                "large_zero_shift_warning_deg": 0.2,
                "polarization_fraction": 0.5,
                "support_fwhm": 30.0,
                "zero_alignment_max_deg": 1.0,
                "zero_alignment_min_deg": -1.0,
                "zero_alignment_step_deg": 0.1,
            },
        },
        "schema_version": 1,
        "selection_policy": {"fixture": "unit test"},
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(manifest), encoding="utf-8")
    return archive, selection


def test_campaign_records_metrics_and_input_boundaries(tmp_path: Path) -> None:
    archive, selection = _fixture(tmp_path)

    result = run_opxrd_robustness_campaign(archive, selection)

    assert all(result.checks.values())
    assert result.selected_case_count == 3
    positive, negative, nonmonotonic = result.cases
    assert isinstance(positive, OpxrdCaseResult)
    assert positive.phasesmith_background_status == "evaluated"
    assert positive.background_deterministic
    assert positive.quantile_only_metrics.poisson_rwp is not None
    assert positive.wavelength_order_corrected
    assert isinstance(negative, OpxrdCaseResult)
    assert negative.phasesmith_background_status == "rejected_nonuniform_grid"
    assert negative.quantile_only_metrics.poisson_rwp is None
    assert isinstance(nonmonotonic, OpxrdRejectedCaseResult)
    assert nonmonotonic.nonincreasing_step_count == 1


def test_campaign_rejects_archive_drift(tmp_path: Path) -> None:
    archive, selection = _fixture(tmp_path)
    with archive.open("ab") as stream:
        stream.write(b"drift")

    with pytest.raises(ValueError, match="size mismatch"):
        run_opxrd_robustness_campaign(archive, selection)


def test_structural_bundle_is_sorted_disclosed_and_finite(tmp_path: Path) -> None:
    archive, selection = _fixture(tmp_path)
    bundle = tmp_path / "bundle"

    model_path = write_opxrd_common_model_bundle(archive, selection, "positive", bundle)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    result = run_opxrd_structural_common_model(bundle, cycles=0)

    assert model["wavelengths_angstrom"] == [1.5406, 1.5444]
    assert model["wavelength_order_corrected"] is True
    assert "represented in P1" in model["assumptions"][0]
    assert result.sample_count == 201
    assert result.reflection_count > 0
    assert result.position_alignment.candidate_count == 21
    assert result.position_alignment.selected_poisson_rwp <= (
        result.position_alignment.initial_poisson_rwp
    )
    assert np.isfinite(result.residual_metrics.relative_l2)
