from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from phasesmith.validation import (
    run_ceria_profile_transferability,
    verify_validation_dataset,
)
from phasesmith.validation.ceria_transferability import _prepare_pattern

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "validation" / "data" / "iucr-ceria-size-strain-round-robin"


def _write_range(path: Path, start: float, step: float, count: int, scale: float) -> None:
    lines = ["synthetic range", "------------------------"]
    for index in range(count):
        if index == count // 2:
            lines.extend(("repeated textual header", "------------------------"))
        x_value = start + step * index
        observed = scale + 20.0 * np.exp(-0.5 * ((x_value - (start + 0.5)) / 0.04) ** 2)
        lines.append(f"{x_value:.8f} {observed:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_range_parser_ignores_text_and_defines_duplicate_boundary(tmp_path: Path) -> None:
    _write_range(tmp_path / "first.xy", 0.0, 0.01, 101, 10.0)
    _write_range(tmp_path / "second.xy", 1.0, 0.01, 101, 30.0)

    prepared = _prepare_pattern(
        tmp_path,
        ("first.xy", "second.xy"),
        smooth_width_deg=0.20,
    )

    assert prepared.range_lengths == (101, 100)
    assert prepared.pattern.x.size == 201
    assert np.count_nonzero(prepared.pattern.x == 1.0) == 1
    assert prepared.pattern.observed_y[100] < 20.0  # closing value from the earlier range


@pytest.mark.real_data
def test_ceria_external_transferability_gate() -> None:
    try:
        verify_validation_dataset("iucr-ceria-size-strain-round-robin", DATASET)
    except (FileNotFoundError, ValueError):
        pytest.skip("checksum-pinned IUCr ceria round-robin files are unavailable")

    result = run_ceria_profile_transferability(DATASET)
    record = result.to_record()

    assert result.status == "passed"
    assert all(
        value for key, value in result.checks.items() if key != "specialized_term_inputs_complete"
    )
    assert result.checks["specialized_term_inputs_complete"] is False
    assert result.checks["calibration_profile_identifiable"] is True
    assert all(
        start.weighted_rank == start.free_parameter_count == 1
        and start.maximum_absolute_correlation == 0.0
        for start in result.profile_starts
    )
    assert result.promotion_decision == "no_specialized_term_promoted"
    assert result.blocked_specialized_terms == (
        "lpsd_defocusing",
        "tube_tail",
        "continuum",
        "coupled_dispersion",
    )
    assert result.holdout_poisson_rwp < 0.06
    assert result.holdout_profile_correlation > 0.985
    assert result.weighted_sse_improvement_fraction > 0.90
    encoded = json.dumps(record, allow_nan=False)
    assert json.loads(encoded)["dataset_id"] == "iucr-ceria-size-strain-round-robin"
