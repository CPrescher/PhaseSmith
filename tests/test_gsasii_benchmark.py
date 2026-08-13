from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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


def test_qarr_comparison_rejects_oracle_drift() -> None:
    benchmark = load_script("benchmarks/compare_gsasii_qarr.py", "compare_gsasii_qarr_test")
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
        "weight_fractions": {"Al2O3": 0.3211, "ZnO": 0.3365, "CaF2": 0.3424},
        "poisson_rwp": 0.1823,
        "unit_weight_rwp": 0.1342,
        "profile_correlation": 0.9903,
    }
    gsas_result = {
        "sample_count": 7_251,
        "reflection_count": 110,
        "weight_fractions": {"Al2O3": 0.3247, "ZnO": 0.3330, "CaF2": 0.3423},
        "poisson_rwp": 0.1839,
        "unit_weight_rwp": 0.1374,
        "profile_correlation": 0.9896,
    }

    validation = benchmark.compare_scientific_results(phasesmith_result, gsas_result)

    assert validation["checks"]["maximum_phase_fraction_delta"]["passed"] is True
    assert validation["phase_fraction_deltas"]["Al2O3"] == pytest.approx(0.0036)
    changed = json.loads(json.dumps(gsas_result))
    changed["weight_fractions"]["Al2O3"] = 0.40
    drift = benchmark.compare_scientific_results(phasesmith_result, changed)
    assert drift["status"] == "failed"
    assert "maximum_phase_fraction_delta" in drift["failed_checks"]


def test_nist_srm660c_fcj_cases_have_distinct_acceptance_contracts() -> None:
    benchmark = load_script(
        "benchmarks/compare_gsasii_nist_srm660c.py", "compare_gsasii_nist_srm660c_test"
    )
    phasesmith_result = {
        "specimen": "100a",
        "sample_count": 5_332,
        "reflection_count": 30,
        "free_parameter_count": 17,
        "sh_over_l": 0.02,
        "poisson_rwp": 0.1891,
        "unit_weight_rwp": 0.2476,
        "profile_correlation": 0.9693,
        "nist_reference_rwp": 0.060546,
        "nist_reference_correlation": 0.999479,
    }
    gsas_result = {
        **phasesmith_result,
        "poisson_rwp": 0.1398,
        "unit_weight_rwp": 0.1452,
        "profile_correlation": 0.9898,
    }

    comparison = benchmark.compare_scientific_results(phasesmith_result, gsas_result)

    assert benchmark.CASES["matched-small-fcj"]["expected_cross_status"] == "passed"
    assert benchmark.CASES["large-fcj-stress"]["expected_cross_status"] == "failed"
    assert list(benchmark.CASES["large-fcj-stress"]["expected_failed_checks"]) == [
        "poisson_rwp_delta",
        "unit_weight_rwp_delta",
        "profile_correlation_delta",
    ]
    assert comparison["status"] == "failed"
    assert comparison["failed_checks"] == [
        "poisson_rwp_delta",
        "unit_weight_rwp_delta",
        "profile_correlation_delta",
    ]
    matched = benchmark.compare_scientific_results(phasesmith_result, phasesmith_result)
    assert matched["status"] == "passed"


def test_nist_worker_converts_pdcif_displacement_to_gsasii_shift_units() -> None:
    worker = load_script(
        "oracle/scripts/benchmark_nist_srm660c.py", "benchmark_nist_srm660c_worker_test"
    )

    assert worker.gsas_shift_micrometre(-0.07877) == pytest.approx(-78.77)
    assert "SH/L:0.002" in worker.instrument_text()
    assert "SH/L:0.02" in worker.instrument_text(0.02)
    with pytest.raises(ValueError, match="finite"):
        worker.gsas_shift_micrometre(float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        worker.instrument_text(-0.001)


def test_rowles_comparison_gates_converted_common_subset() -> None:
    benchmark = load_script("benchmarks/compare_gsasii_rowles_qpa.py", "compare_gsasii_rowles_test")
    phasesmith_result = {
        "sample": "1e",
        "weight_fractions": {"Al2O3": 0.5728, "ZnO": 0.1401, "CaF2": 0.2871},
        "poisson_rwp": 0.0826,
        "unit_weight_rwp": 0.0897,
        "profile_correlation": 0.9952,
    }
    gsas_result = {
        "sample": "1e",
        "weight_fractions": {"Al2O3": 0.5704, "ZnO": 0.1408, "CaF2": 0.2888},
        "poisson_rwp": 0.0820,
        "unit_weight_rwp": 0.0911,
        "profile_correlation": 0.9950,
    }

    comparison = benchmark.compare_scientific_results(phasesmith_result, gsas_result)

    assert comparison["status"] == "passed"
    assert comparison["phase_fraction_deltas"]["Al2O3"] == pytest.approx(0.0024)
    changed = json.loads(json.dumps(gsas_result))
    changed["weight_fractions"]["Al2O3"] = 0.45
    assert benchmark.compare_scientific_results(phasesmith_result, changed)["status"] == "failed"


def test_powgen_worker_and_comparison_share_the_real_instrument_convention() -> None:
    worker = load_script(
        "oracle/scripts/benchmark_powgen_tof.py", "benchmark_powgen_tof_worker_test"
    )
    benchmark = load_script(
        "benchmarks/compare_gsasii_powgen_tof.py", "compare_gsasii_powgen_tof_test"
    )
    values = {
        "Zero": 4.41,
        "difC": 22_581.63,
        "difA": 0.0,
        "difB": 0.0,
        "alpha": 0.25746,
        "beta-0": 0.091563,
        "beta-1": 0.017334,
        "sig-0": 0.0,
        "sig-1": 10.0,
        "sig-2": 203.581,
        "X": 0.0,
        "Y": 0.0,
        "Z": 0.0,
    }
    instrument = benchmark.phase_instrument(values)
    actual = phasesmith.tof_profile_parameters([0.4, 1.0, 4.0], instrument)

    assert worker.PINNED_REVISION == benchmark.PINNED_REVISION
    np.testing.assert_allclose(actual.position_us, [9_037.062, 22_586.04, 90_330.93], rtol=2e-16)
    np.testing.assert_allclose(actual.alpha_per_us, [0.64365, 0.25746, 0.064365], rtol=2e-16)
    np.testing.assert_allclose(
        actual.beta_per_us,
        [0.091563 + 0.017334 / 0.4**4, 0.108897, 0.091563 + 0.017334 / 4.0**4],
        rtol=4e-16,
    )
    assert benchmark.LIMITS["reconstructed_pattern_minimum_correlation"] == 0.99999
    assert benchmark.LIMITS["native_workflow_rwp_delta"] == 0.03
    assert benchmark.LIMITS["native_workflow_profile_correlation_delta"] == 0.02


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


def test_sodium_citrate_silicon_comparison_gates_orientation_subset() -> None:
    benchmark = load_script(
        "benchmarks/compare_gsasii_iucr_sodium_citrate_silicon.py",
        "sodium_citrate_silicon_gate_test",
    )
    native = {
        "sample_count": 4452,
        "poisson_rwp": 0.1818,
        "profile_correlation": 0.9559,
        "weight_fractions": {
            "sodium_dihydrogen_citrate": 0.7783,
            "silicon": 0.2217,
        },
        "refined_march_ratio": 0.6385,
        "silicon_calibration_poisson_rwp": 0.1217,
        "calibrated_sample_displacement_mm": -0.1398,
    }
    oracle = {
        "sample_count": 4452,
        "poisson_rwp": 0.1895,
        "profile_correlation": 0.9500,
        "weight_fractions": {
            "sodium_dihydrogen_citrate": 0.7835,
            "silicon": 0.2165,
        },
        "refined_hap": {"sodium_dihydrogen_citrate": {"Pref.Ori.": ["MD", 0.6324]}},
        "silicon_calibration": {
            "poisson_rwp": 0.1203,
            "calibrated_sample_displacement_mm": -0.2010,
        },
    }

    result = benchmark.comparison_checks(native, oracle)

    assert result["status"] == "passed"
    assert all(check["passed"] for check in result["checks"].values())
    changed = json.loads(json.dumps(oracle))
    changed["refined_hap"]["sodium_dihydrogen_citrate"]["Pref.Ori."][1] = 0.60
    assert benchmark.comparison_checks(native, changed)["status"] == "failed"


def test_tripotassium_citrate_comparison_gates_expected_transfer_failure() -> None:
    benchmark = load_script(
        "benchmarks/compare_gsasii_iucr_tripotassium_citrate_silicon.py",
        "tripotassium_citrate_silicon_gate_test",
    )
    native = {
        "sample_count": 2696,
        "poisson_rwp": 0.23965,
        "profile_correlation": 0.61846,
        "weight_fractions": {"tripotassium_citrate": 0.95550, "silicon": 0.04450},
        "silicon_calibration_poisson_rwp": 0.19993,
        "calibrated_sample_displacement_mm": -0.02492,
    }
    oracle = {
        "sample_count": 2696,
        "poisson_rwp": 0.23933,
        "profile_correlation": 0.62079,
        "weight_fractions": {"tripotassium_citrate": 0.95657, "silicon": 0.04343},
        "silicon_calibration": {
            "poisson_rwp": 0.20083,
            "calibrated_sample_displacement_mm": 0.00180,
        },
    }

    result = benchmark.comparison_checks(native, oracle)

    assert result["status"] == "qualified_pass"
    assert result["checks"]["silicon_anchor_not_transferable"]["passed"] is True
    changed = json.loads(json.dumps(oracle))
    changed["profile_correlation"] = 0.65
    assert benchmark.comparison_checks(native, changed)["status"] == "failed"


@pytest.mark.parametrize(
    "script",
    [
        "benchmarks/compare_gsasii.py",
        "benchmarks/compare_gsasii_iucr_silicon_standard.py",
        "benchmarks/compare_gsasii_iucr_sodium_citrate_silicon.py",
        "benchmarks/compare_gsasii_iucr_tripotassium_citrate_silicon.py",
        "benchmarks/compare_gsasii_nist_srm660c.py",
        "benchmarks/compare_gsasii_pbso4.py",
        "benchmarks/compare_gsasii_powgen_tof.py",
        "benchmarks/compare_gsasii_qarr.py",
        "benchmarks/compare_gsasii_rowles_fpa.py",
        "benchmarks/compare_gsasii_rowles_qpa.py",
        "benchmarks/compare_gsasii_real_lebail.py",
        "benchmarks/compare_gsasii_structural.py",
        "benchmarks/practical_workflow.py",
        "benchmarks/real_data.py",
        "oracle/scripts/benchmark_cw_profile.py",
        "oracle/scripts/benchmark_iucr_silicon_standard.py",
        "oracle/scripts/benchmark_iucr_sodium_citrate_silicon.py",
        "oracle/scripts/benchmark_nist_srm660c.py",
        "oracle/scripts/benchmark_pbso4.py",
        "oracle/scripts/benchmark_powgen_tof.py",
        "oracle/scripts/benchmark_qarr.py",
        "oracle/scripts/benchmark_rowles_qpa.py",
        "oracle/scripts/benchmark_real_lebail.py",
        "oracle/scripts/benchmark_structural_pattern.py",
        "oracle/scripts/calibrate_rowles_fpa.py",
        "tools/fetch_validation_data.py",
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
        "oracle/scripts/benchmark_iucr_silicon_standard.py",
        "oracle/scripts/benchmark_iucr_sodium_citrate_silicon.py",
        "oracle/scripts/benchmark_nist_srm660c.py",
        "oracle/scripts/benchmark_pbso4.py",
        "oracle/scripts/benchmark_powgen_tof.py",
        "oracle/scripts/benchmark_qarr.py",
        "oracle/scripts/benchmark_rowles_qpa.py",
        "oracle/scripts/benchmark_real_lebail.py",
        "oracle/scripts/benchmark_structural_pattern.py",
        "oracle/scripts/calibrate_rowles_fpa.py",
    ):
        source = (REPOSITORY_ROOT / relative_path).read_text()
        assert "import phasesmith" not in source
