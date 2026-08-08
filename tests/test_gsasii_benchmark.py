from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import phasesmith
import pytest

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
    actual = phasesmith.accumulate_cw(x, positions, intensities, instrument)
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
    instrument = phasesmith.ConstantWavelengthInstrument(
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
    expected = phasesmith.cw_profile_parameters(positions, instrument)

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
            profile = phasesmith.profile_tch(x - position, gaussian_fwhm, gamma_centideg / 100.0)
            d_sigma2_d_gaussian = 20_000.0 * gaussian_fwhm / gaussian_fwhm_per_sigma**2
            return (
                profile.value / 100.0,
                profile.d_delta / 100.0,
                profile.d_gaussian_fwhm / (100.0 * d_sigma2_d_gaussian),
                profile.d_lorentzian_fwhm / 10_000.0,
            )

    x, positions, intensities, instrument = benchmark.benchmark_inputs(12, 1_001)
    expected = phasesmith.accumulate_cw(x, positions, intensities, instrument)
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


def test_structural_comparison_validation_contract() -> None:
    benchmark = load_script(
        "benchmarks/compare_gsasii_structural.py", "compare_gsasii_structural_test"
    )
    hkl = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int64)
    multiplicity = np.array([2, 4, 2], dtype=np.int64)
    phase = benchmark.benchmark_phase(hkl, multiplicity, 4)
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406, 2.0e-4, -1.0e-4, 1.2e-4, 1.5e-3, 3.0e-3
    )
    structural = phasesmith.calculate_structure_factor_values(
        phase.structure,
        hkl,
        multiplicity,
        phase.scattering,
        correction=phase.intensity_correction,
    )
    pattern = phasesmith.PreparedStructuralPattern(
        phasesmith.PowderPattern(np.linspace(5.0, 125.0, 2_001)),
        phasesmith.ConstantWavelengthExperiment.neutron(instrument),
        phase,
    ).calculate()
    local = pattern.accumulation.derivatives.local
    oracle = {
        "f_fm": structural.f.copy(),
        "f_squared_fm2": structural.f_squared.copy(),
        "integrated_intensity": structural.integrated_intensity.copy(),
        "d_spacing_angstrom": pattern.reflections.d_spacing_angstrom.copy(),
        "position_deg": pattern.reflections.two_theta_deg.copy(),
        "y": pattern.profile_y.copy(),
        "starts": local.starts.copy(),
        "offsets": local.offsets.copy(),
        "local": local.values.copy(),
        "global_jacobian": pattern.accumulation.derivatives.global_jacobian[:5].copy(),
    }

    assert max(benchmark.validate_outputs(structural, pattern, oracle).values()) == 0.0
    oracle["f_squared_fm2"][np.argmax(np.abs(oracle["f_squared_fm2"]))] *= 1.001
    with pytest.raises(RuntimeError, match="structural numerical validation failed"):
        benchmark.validate_outputs(structural, pattern, oracle)


def test_structural_worker_converts_instrument_units_explicitly() -> None:
    worker = load_script(
        "oracle/scripts/benchmark_structural_pattern.py", "benchmark_structural_pattern_test"
    )
    text = worker.instrument_text(1.5406, np.array([2.0e-4, -1.0e-4, 1.2e-4, 1.5e-3, 3.0e-3]))
    assert "Type:PNC" in text
    assert "U:2" in text
    assert "V:-1" in text
    assert "W:1.2" in text
    assert "X:0.15" in text
    assert "Y:0.3" in text
    assert worker.GSAS_F_SQUARED_TO_FM_SQUARED == 100.0


def test_qarr_worker_fcj_ablation_preserves_source(tmp_path: Path) -> None:
    worker = load_script("oracle/scripts/benchmark_qarr.py", "benchmark_qarr_worker_test")
    source = tmp_path / "source.instprm"
    source.write_text("Type:PXC\nSH/L:0.002\n", encoding="utf-8")

    assert worker.selected_instrument(source, tmp_path, "instrument") == source
    zero = worker.selected_instrument(source, tmp_path, "below-minimum")
    assert zero != source
    assert zero.read_text(encoding="utf-8") == "Type:PXC\nSH/L:1e-12\n"
    assert source.read_text(encoding="utf-8") == "Type:PXC\nSH/L:0.002\n"


def test_qarr_comparison_parses_results_and_rejects_oracle_drift() -> None:
    benchmark = load_script("benchmarks/compare_gsasii_qarr.py", "compare_gsasii_qarr_test")
    report = SimpleNamespace(
        notes=("Calculated crystalline weight fractions: Al2O3=31.1%, ZnO=34.2%, CaF2=34.7%.",)
    )
    assert benchmark.phase_fractions(report) == {
        "Al2O3": pytest.approx(0.311),
        "ZnO": pytest.approx(0.342),
        "CaF2": pytest.approx(0.347),
    }
    base = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": benchmark.PINNED_REVISION,
        "scope": benchmark.SCOPE,
        "recipe": {"cycles": 8},
        "input_sha256": {"pattern": "abc"},
        "oracle_behavior": {"fcj_calculation_floor": 0.002},
        "result": {"poisson_rwp": 0.18},
        "stage_rwp_percent": {"instrument": 20.0},
    }
    benchmark.validate_gsas_reports([base, dict(base)])
    changed = json.loads(json.dumps(base))
    changed["result"]["poisson_rwp"] = 0.19
    with pytest.raises(RuntimeError, match="repetitions disagree"):
        benchmark.validate_gsas_reports([base, changed])


def test_qarr_comparison_numerically_gates_cross_implementation_results() -> None:
    benchmark = load_script("benchmarks/compare_gsasii_qarr.py", "qarr_cross_gate_test")
    phasesmith_result = {
        "sample_count": 7_251,
        "reflection_count": 110,
        "weight_fractions": {"Al2O3": 0.308, "ZnO": 0.342, "CaF2": 0.350},
        "poisson_rwp": 0.198,
        "unit_weight_rwp": 0.132,
        "profile_correlation": 0.991,
    }
    gsas_result = {
        "sample_count": 7_251,
        "reflection_count": 110,
        "weight_fractions": {"Al2O3": 0.315, "ZnO": 0.337, "CaF2": 0.349},
        "poisson_rwp": 0.184,
        "unit_weight_rwp": 0.137,
        "profile_correlation": 0.990,
    }

    validation = benchmark.compare_scientific_results(phasesmith_result, gsas_result)

    assert validation["checks"]["maximum_phase_fraction_delta"]["passed"] is True
    assert validation["phase_fraction_deltas"]["Al2O3"] == pytest.approx(0.007)
    changed = json.loads(json.dumps(gsas_result))
    changed["weight_fractions"]["Al2O3"] = 0.40
    with pytest.raises(RuntimeError, match="cross-implementation validation failed"):
        benchmark.compare_scientific_results(phasesmith_result, changed)


def test_pbso4_comparison_gates_profiles_and_refined_cell() -> None:
    benchmark = load_script("benchmarks/compare_gsasii_pbso4.py", "pbso4_cross_gate_test")
    phasesmith_result = {
        "xray": {
            "sample_count": 5_697,
            "poisson_rwp": 0.10346,
            "unit_weight_rwp": 0.08732,
            "profile_correlation": 0.99555,
        },
        "neutron": {
            "sample_count": 2_681,
            "poisson_rwp": 0.04761,
            "unit_weight_rwp": 0.05059,
            "profile_correlation": 0.99580,
            "cell_angstrom": {"a": 8.46474, "b": 5.38802, "c": 6.94670},
        },
    }
    gsas_result = {
        "xray": {
            "sample_count": 5_697,
            "poisson_rwp": 0.106,
            "unit_weight_rwp": 0.088,
            "profile_correlation": 0.996,
        },
        "neutron": {
            "sample_count": 2_681,
            "poisson_rwp": 0.045,
            "unit_weight_rwp": 0.045,
            "profile_correlation": 0.997,
        },
        "cell_angstrom": {"a": 8.4740, "b": 5.3939, "c": 6.9544},
    }

    validation = benchmark.compare_scientific_results(phasesmith_result, gsas_result)

    assert validation["status"] == "passed"
    assert validation["checks"]["neutron_cell_relative_delta"]["passed"] is True
    changed = json.loads(json.dumps(phasesmith_result))
    changed["neutron"]["cell_angstrom"]["a"] = 8.40
    with pytest.raises(RuntimeError, match="cross-implementation validation failed"):
        benchmark.compare_scientific_results(changed, gsas_result)


def test_practical_workflow_benchmark_covers_xray_and_neutron() -> None:
    benchmark = load_script("benchmarks/practical_workflow.py", "practical_workflow_test")
    for probe in (phasesmith.RadiationProbe.X_RAY, phasesmith.RadiationProbe.NEUTRON):
        pattern, experiment, phases, background = benchmark.benchmark_case(probe, 1_001)
        result = benchmark.structural_refinement.calculate(
            pattern, experiment, phases, background=background
        )
        assert result.y.shape == (1_001,)
        assert np.isfinite(result.y).all()
        assert phases[0].physics.providers[2].descriptor.provider_id == "phasesmith.march-dollase"


@pytest.mark.parametrize(
    "script",
    [
        "benchmarks/compare_gsasii.py",
        "benchmarks/compare_gsasii_pbso4.py",
        "benchmarks/compare_gsasii_qarr.py",
        "benchmarks/compare_gsasii_structural.py",
        "benchmarks/practical_workflow.py",
        "benchmarks/real_data.py",
        "oracle/scripts/benchmark_cw_profile.py",
        "oracle/scripts/benchmark_pbso4.py",
        "oracle/scripts/benchmark_qarr.py",
        "oracle/scripts/benchmark_structural_pattern.py",
    ],
)
def test_benchmark_help_does_not_require_gsasii(script: str) -> None:
    subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_external_worker_does_not_import_rietveld() -> None:
    for relative_path in (
        "oracle/scripts/benchmark_cw_profile.py",
        "oracle/scripts/benchmark_pbso4.py",
        "oracle/scripts/benchmark_qarr.py",
        "oracle/scripts/benchmark_structural_pattern.py",
    ):
        source = (REPOSITORY_ROOT / relative_path).read_text()
        assert "import phasesmith" not in source
