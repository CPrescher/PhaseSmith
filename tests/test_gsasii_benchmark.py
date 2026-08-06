from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import rietveld

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_comparison_workload_and_validation_contract() -> None:
    benchmark = load_script("benchmarks/compare_gsasii.py", "compare_gsasii_test")
    x, positions, intensities, instrument = benchmark.benchmark_inputs(8, 501)
    actual = rietveld.accumulate_cw(x, positions, intensities, instrument)
    local = actual.derivatives.local
    oracle = {
        "y": actual.y.copy(),
        "starts": local.starts.copy(),
        "offsets": local.offsets.copy(),
        "local": local.values.copy(),
        "global_jacobian": actual.derivatives.global_jacobian.copy(),
    }

    assert benchmark.validate_outputs(actual, oracle) == {
        "profile": 0.0,
        "intensity_derivatives": 0.0,
        "position_derivatives": 0.0,
        "global_derivatives": 0.0,
    }
    oracle["y"][np.argmax(np.abs(oracle["y"]))] *= 1.001
    with pytest.raises(RuntimeError, match="numerical validation failed"):
        benchmark.validate_outputs(actual, oracle)
    oracle["y"][0] = np.nan
    with pytest.raises(RuntimeError, match="must be finite"):
        benchmark.validate_outputs(actual, oracle)


def test_external_worker_width_and_support_equations_match_public_model() -> None:
    worker = load_script("oracle/scripts/benchmark_cw_profile.py", "benchmark_cw_profile_test")
    instrument = rietveld.ConstantWavelengthInstrument(
        wavelength_angstrom=1.5406,
        u_deg2=2.0e-4,
        v_deg2=-1.0e-4,
        w_deg2=1.2e-4,
        x_deg=1.5e-3,
        y_deg=3.0e-3,
    )
    instrument_array = np.array(
        [
            instrument.u_deg2,
            instrument.v_deg2,
            instrument.w_deg2,
            instrument.x_deg,
            instrument.y_deg,
        ]
    )
    positions = np.array([15.0, 70.0, 145.0])
    expected = rietveld.cw_profile_parameters(positions, instrument)

    for index, position in enumerate(positions):
        terms = worker.width_terms(position, instrument_array)
        assert terms[0] == pytest.approx(expected.gaussian_variance_deg2[index], rel=3e-15)
        assert terms[1] == pytest.approx(expected.gaussian_fwhm_deg[index], rel=3e-15)
        assert terms[2] == pytest.approx(expected.lorentzian_fwhm_deg[index], rel=3e-15)
        np.testing.assert_allclose(
            terms[3], expected.d_gaussian_fwhm_d_instrument[index], rtol=3e-15
        )
        np.testing.assert_allclose(
            terms[4], expected.d_lorentzian_fwhm_d_instrument[index], rtol=3e-15
        )
        np.testing.assert_allclose(
            terms[5:7], expected.d_component_fwhm_d_two_theta[index], rtol=3e-15
        )
        assert worker.tch_fwhm(terms[1], terms[2]) == pytest.approx(
            expected.total_fwhm_deg[index], rel=3e-15
        )


def test_external_worker_unit_and_derivative_chain_matches_fused_kernel() -> None:
    benchmark = load_script("benchmarks/compare_gsasii.py", "compare_gsasii_chain_test")
    worker = load_script("oracle/scripts/benchmark_cw_profile.py", "gsasii_chain_test")
    gaussian_fwhm_per_sigma = np.sqrt(8.0 * np.log(2.0))

    class FakeGsasProfile:
        @staticmethod
        def getdPsVoigt(
            position: float, sigma2_centideg2: float, gamma_centideg: float, x: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            gaussian_sigma = np.sqrt(sigma2_centideg2) / 100.0
            gaussian_fwhm = gaussian_fwhm_per_sigma * gaussian_sigma
            profile = rietveld.profile_tch(x - position, gaussian_fwhm, gamma_centideg / 100.0)
            d_sigma2_d_gaussian = 20_000.0 * gaussian_fwhm / gaussian_fwhm_per_sigma**2
            return (
                profile.value / 100.0,
                profile.d_delta / 100.0,
                profile.d_gaussian_fwhm / (100.0 * d_sigma2_d_gaussian),
                profile.d_lorentzian_fwhm / 10_000.0,
            )

    x, positions, intensities, instrument = benchmark.benchmark_inputs(12, 1_001)
    expected = rietveld.accumulate_cw(x, positions, intensities, instrument)
    model = np.array(
        [
            instrument.u_deg2,
            instrument.v_deg2,
            instrument.w_deg2,
            instrument.x_deg,
            instrument.y_deg,
        ]
    )
    actual = worker.accumulate(FakeGsasProfile(), x, positions, intensities, model, 20.0)
    oracle = {
        "y": actual[0],
        "starts": actual[1],
        "offsets": actual[2],
        "local": actual[3],
        "global_jacobian": actual[4],
    }
    errors = benchmark.validate_outputs(expected, oracle)
    assert max(errors.values()) < 3.0e-15


@pytest.mark.parametrize(
    "script",
    ["benchmarks/compare_gsasii.py", "oracle/scripts/benchmark_cw_profile.py"],
)
def test_benchmark_help_does_not_require_gsasii(script: str) -> None:
    subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_external_worker_does_not_import_rietveld() -> None:
    source = (REPOSITORY_ROOT / "oracle/scripts/benchmark_cw_profile.py").read_text()
    assert "import rietveld" not in source
