from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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


def test_real_lebail_parity_contract_gates_counts_rwp_and_correlation() -> None:
    assert LEBAIL.CASES["aps-sucrose-11bmb"]["maximum_rwp_delta"] == 0.005
    assert LEBAIL.CASES["aps-sucrose-11bmb"]["maximum_correlation_delta"] == 0.02
    phase = {
        "sample_count": 2_111,
        "reflection_count": 13,
        "poisson_rwp": 0.400,
        "profile_correlation": 0.91,
    }
    gsas = {
        "sample_count": 2_111,
        "reflection_count": 13,
        "poisson_rwp": 0.398,
        "profile_correlation": 0.92,
    }
    comparison = LEBAIL.compare_scientific_results("aps-sucrose-11bmb", phase, gsas)
    assert comparison["status"] == "passed"
    assert all(check["passed"] for check in comparison["checks"].values())

    with pytest.raises(RuntimeError, match="sample counts"):
        LEBAIL.compare_scientific_results(
            "aps-sucrose-11bmb", phase, gsas | {"sample_count": 2_110}
        )
    drift = LEBAIL.compare_scientific_results(
        "aps-sucrose-11bmb",
        phase,
        gsas | {"poisson_rwp": 0.01},
    )
    assert drift["status"] == "failed"
    assert "poisson_rwp_delta" in drift["failed_checks"]


def test_sucrose_oracle_background_uses_the_full_identical_grid(tmp_path: Path) -> None:
    data = REPOSITORY_ROOT / "validation/data/aps-sucrose-11bmb"
    path = LEBAIL.prepare_fixed_background("aps-sucrose-11bmb", data, tmp_path)

    assert path is not None
    prepared = np.loadtxt(path)
    pattern = read_powder_data(data / "11bmb_8716.fxye", format="gsas_fxye")
    selected = (pattern.x >= 1.0) & (pattern.x <= 24.0)
    np.testing.assert_array_equal(prepared[:, 0], pattern.x)
    np.testing.assert_array_equal(prepared[~selected, 1], 0.0)
    assert np.all(np.isfinite(prepared[selected, 1]))
    assert np.all(prepared[selected, 1] >= 0.0)


def test_oracle_workers_never_import_the_normal_phasesmith_package() -> None:
    workers = ("oracle/scripts/benchmark_real_lebail.py",)
    for worker in workers:
        source = (REPOSITORY_ROOT / worker).read_text(encoding="utf-8")
        assert "import phasesmith" not in source
        assert "from phasesmith" not in source
