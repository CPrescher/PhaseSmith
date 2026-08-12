from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from phasesmith.io.powder import read_powder_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEBAIL = load_script("compare_gsasii_real_lebail", "benchmarks/compare_gsasii_real_lebail.py")
QARR = load_script("compare_gsasii_qarr_contract", "benchmarks/compare_gsasii_qarr.py")
PBSO4 = load_script("compare_gsasii_pbso4_contract", "benchmarks/compare_gsasii_pbso4.py")
NICKEL_TOF = load_script(
    "compare_gsasii_nickel_tof_contract",
    "benchmarks/compare_gsasii_nickel_tof_multibank.py",
)


def test_real_lebail_parity_contract_gates_counts_rwp_and_correlation() -> None:
    assert LEBAIL.CASES["aps-sucrose-11bmb"]["maximum_rwp_delta"] == 0.005
    assert LEBAIL.CASES["aps-sucrose-11bmb"]["maximum_correlation_delta"] == 0.02
    phase = {
        "sample_count": 2_111,
        "reflection_count": 13,
        "poisson_rwp": 0.40,
        "profile_correlation": 0.91,
    }
    gsas = {
        "sample_count": 2_111,
        "reflection_count": 13,
        "poisson_rwp": 0.30,
        "profile_correlation": 0.96,
    }
    comparison = LEBAIL.compare_scientific_results("ansto-echidna-lab6-cw-neutron", phase, gsas)
    assert comparison["status"] == "passed"
    assert all(check["passed"] for check in comparison["checks"].values())

    with pytest.raises(RuntimeError, match="sample counts"):
        LEBAIL.compare_scientific_results(
            "ansto-echidna-lab6-cw-neutron", phase, gsas | {"sample_count": 2_110}
        )
    drift = LEBAIL.compare_scientific_results(
        "aps-sucrose-11bmb",
        phase,
        gsas | {"poisson_rwp": 0.01},
    )
    assert drift["status"] == "failed"
    assert "poisson_rwp_delta" in drift["failed_checks"]


def test_sucrose_oracle_background_uses_the_full_identical_grid(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    x_centideg = np.arange(0.0, 3_050.0, 50.0)
    observed = 100.0 + 0.02 * x_centideg + 5.0 * np.sin(x_centideg / 300.0)
    rows = [
        "Synthetic GSAS FXYE contract fixture",
        f"BANK 1 {x_centideg.size} {x_centideg.size} CONS 0 50 0 0 FXYE",
        *(
            f"{x_value:.1f} {y_value:.17g} 1.0"
            for x_value, y_value in zip(x_centideg, observed, strict=True)
        ),
    ]
    (data / "11bmb_8716.fxye").write_text("\n".join(rows) + "\n", encoding="utf-8")
    path = LEBAIL.prepare_fixed_background("aps-sucrose-11bmb", data, tmp_path)

    assert path is not None
    prepared = np.loadtxt(path)
    pattern = read_powder_data(data / "11bmb_8716.fxye", format="gsas_fxye")
    selected = (pattern.x >= 1.0) & (pattern.x <= 24.0)
    np.testing.assert_array_equal(prepared[:, 0], pattern.x)
    np.testing.assert_array_equal(prepared[~selected, 1], 0.0)
    assert np.all(np.isfinite(prepared[selected, 1]))
    assert np.all(prepared[selected, 1] >= 0.0)


def test_qarr_parity_contract_accepts_reviewed_metrics_and_rejects_drift() -> None:
    phase = {
        "sample_count": 7_251,
        "reflection_count": 110,
        "weight_fractions": {"Al2O3": 0.3211, "ZnO": 0.3365, "CaF2": 0.3424},
        "poisson_rwp": 0.1823,
        "unit_weight_rwp": 0.1342,
        "profile_correlation": 0.9903,
    }
    gsas = {
        "sample_count": 7_251,
        "reflection_count": 110,
        "weight_fractions": {"Al2O3": 0.3247, "ZnO": 0.3330, "CaF2": 0.3423},
        "poisson_rwp": 0.1839,
        "unit_weight_rwp": 0.1374,
        "profile_correlation": 0.9896,
    }
    comparison = QARR.compare_scientific_results(phase, gsas)
    assert all(check["passed"] for check in comparison["checks"].values())

    drifted = gsas | {"weight_fractions": {"Al2O3": 0.40, "ZnO": 0.30, "CaF2": 0.30}}
    drift = QARR.compare_scientific_results(phase, drifted)
    assert drift["status"] == "failed"
    assert "maximum_phase_fraction_delta" in drift["failed_checks"]


def test_pbso4_parity_contract_keeps_probe_specific_limits() -> None:
    phase = {
        "xray": {
            "sample_count": 5_697,
            "poisson_rwp": 0.103,
            "unit_weight_rwp": 0.087,
            "profile_correlation": 0.996,
            "cell_angstrom": {"a": 8.48, "b": 5.398, "c": 6.958},
        },
        "neutron": {
            "sample_count": 2_681,
            "poisson_rwp": 0.042,
            "unit_weight_rwp": 0.045,
            "profile_correlation": 0.997,
            "cell_angstrom": {"a": 8.470, "b": 5.392, "c": 6.951},
        },
    }
    gsas = {
        "xray": {
            "sample_count": 5_697,
            "poisson_rwp": 0.101,
            "unit_weight_rwp": 0.085,
            "profile_correlation": 0.9965,
        },
        "neutron": {
            "sample_count": 2_681,
            "poisson_rwp": 0.040,
            "unit_weight_rwp": 0.043,
            "profile_correlation": 0.996,
        },
        "cell_angstrom": {"a": 8.474, "b": 5.394, "c": 6.954},
    }
    comparison = PBSO4.compare_scientific_results(phase, gsas)
    assert comparison["status"] == "passed"


def test_pbso4_comparison_preserves_failed_native_status_for_oracle_diagnosis() -> None:
    checks = [
        SimpleNamespace(check_id="poisson_rwp", measured=0.103),
        SimpleNamespace(check_id="unit_weight_rwp", measured=0.087),
        SimpleNamespace(check_id="profile_correlation", measured=0.996),
        SimpleNamespace(check_id="reference_cell_relative_error", measured=0.0),
    ]
    report = SimpleNamespace(
        status="failed",
        sample_count=5_697,
        reflection_count=383,
        checks=checks,
        notes=[
            "Final cell a=8.48000000, b=5.39800000, c=6.95800000 angstrom.",
            "Stage final_polish: termination=repeated_rejections.",
        ],
    )

    result = PBSO4.phase_result(report, "xray")

    assert result["validation_status"] == "failed"
    assert result["termination"].startswith("Stage final_polish:")


def test_oracle_workers_never_import_the_normal_phasesmith_package() -> None:
    workers = (
        "oracle/scripts/benchmark_iucr_silicon_standard.py",
        "oracle/scripts/benchmark_real_lebail.py",
        "oracle/scripts/benchmark_qarr.py",
        "oracle/scripts/benchmark_pbso4.py",
        "oracle/scripts/benchmark_powgen_tof.py",
        "oracle/scripts/benchmark_nickel_tof_multibank.py",
    )
    for worker in workers:
        source = (REPOSITORY_ROOT / worker).read_text(encoding="utf-8")
        assert "import phasesmith" not in source
        assert "from phasesmith" not in source


def test_nickel_multibank_oracle_contract_keeps_like_for_like_gates_separate() -> None:
    assert NICKEL_TOF.BANKS == (2, 3, 4)
    assert NICKEL_TOF.LIMITS["reconstructed_pattern_relative_l2"] == 0.015
    assert NICKEL_TOF.LIMITS["reconstructed_pattern_minimum_correlation"] == 0.9999
    assert "native_oracle_joint_rwp_delta" not in NICKEL_TOF.LIMITS
    assert NICKEL_TOF.LIMITS["native_oracle_cell_delta_angstrom"] == 5.0e-4
