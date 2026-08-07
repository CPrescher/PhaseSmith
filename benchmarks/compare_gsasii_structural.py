#!/usr/bin/env python3
"""Compare structural-intensity and CW-pattern performance with pinned GSAS-II."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import rietveld

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPOSITORY_ROOT / "oracle" / "scripts" / "benchmark_structural_pattern.py"
PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "neutron_structure_factors_and_controlled_symmetric_cw_pattern"
SPECIES = ("C", "O", "Si", "Fe")
STRUCTURAL_TOLERANCE = 1.25e-4
PROFILE_TOLERANCE = 1.5e-4
PROFILE_KERNEL_TOLERANCE = 6.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-python", type=Path, default=os.environ.get("GSASII_PYTHON"))
    parser.add_argument("--gsas-root", type=Path, default=os.environ.get("RIETVELD_GSASII_ROOT"))
    parser.add_argument(
        "--binary-dir", type=Path, default=os.environ.get("RIETVELD_GSASII_BINARY_DIR")
    )
    parser.add_argument("--reflections", type=int, default=256)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--samples", type=int, default=20_001)
    parser.add_argument("--support-fwhm", type=float, default=20.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def benchmark_sites(site_count: int) -> tuple[rietveld.AtomSite, ...]:
    return tuple(
        rietveld.AtomSite(
            f"site-{site}",
            f"{SPECIES[site % len(SPECIES)]}{site}",
            SPECIES[site % len(SPECIES)],
            SPECIES[site % len(SPECIES)],
            (
                (0.137 * site) % 1.0,
                (0.271 * site + 0.11) % 1.0,
                (0.419 * site + 0.23) % 1.0,
            ),
            1.0,
            0.005 + 0.0002 * site,
        )
        for site in range(site_count)
    )


def benchmark_phase(
    hkl: np.ndarray, multiplicity: np.ndarray, site_count: int
) -> rietveld.RietveldPhase:
    structure = rietveld.CrystalStructure(
        "benchmark",
        "Benchmark structure",
        rietveld.UnitCell(15.0, 15.0, 15.0, 90.0, 90.0, 90.0),
        rietveld.SpaceGroup.p1(),
        benchmark_sites(site_count),
    )
    return rietveld.RietveldPhase(
        "benchmark",
        "Benchmark phase",
        structure,
        rietveld.StructuralReflectionBatch(
            tuple(f"{h},{k},{ell}" for h, k, ell in hkl), hkl, multiplicity
        ),
        rietveld.NeutronNuclear(),
        rietveld.NeutralIntegratedIntensityCorrection(),
        1.0,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1)]


def timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "timings_ms": values,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def measure(operation: Any, warmups: int, repetitions: int) -> tuple[Any, list[float]]:
    for _ in range(warmups):
        operation()
    result = None
    timings_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = operation()
        timings_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    return result, timings_ms


def normalized_maximum_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"benchmark output shape mismatch: engine {actual.shape}, GSAS-II {expected.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError("benchmark outputs must be finite")
    if actual.size == 0:
        return 0.0
    scale = float(np.max(np.abs(expected)))
    difference = float(np.max(np.abs(actual - expected)))
    return difference if scale == 0.0 else difference / scale


def validate_outputs(
    structural: rietveld.StructureFactorValuesResult,
    pattern: rietveld.StructuralPatternCalculationResult,
    oracle: dict[str, np.ndarray],
) -> dict[str, float]:
    np.testing.assert_array_equal(pattern.accumulation.derivatives.local.starts, oracle["starts"])
    np.testing.assert_array_equal(pattern.accumulation.derivatives.local.offsets, oracle["offsets"])
    comparisons = {
        "structure_factor": (structural.f, oracle["f_fm"]),
        "f_squared": (structural.f_squared, oracle["f_squared_fm2"]),
        "integrated_intensity": (
            structural.integrated_intensity,
            oracle["integrated_intensity"],
        ),
        "d_spacing": (pattern.reflections.d_spacing_angstrom, oracle["d_spacing_angstrom"]),
        "position": (pattern.reflections.two_theta_deg, oracle["position_deg"]),
        "profile": (pattern.profile_y, oracle["y"]),
        "intensity_derivatives": (
            pattern.accumulation.derivatives.local.values[:, 0],
            oracle["local"][:, 0],
        ),
        "position_derivatives": (
            pattern.accumulation.derivatives.local.values[:, 1],
            oracle["local"][:, 1],
        ),
        "global_derivatives": (
            pattern.accumulation.derivatives.global_jacobian[:5],
            oracle["global_jacobian"],
        ),
    }
    errors = {
        name: normalized_maximum_error(engine, gsas) for name, (engine, gsas) in comparisons.items()
    }
    tolerances = {
        "structure_factor": STRUCTURAL_TOLERANCE,
        "f_squared": STRUCTURAL_TOLERANCE,
        "integrated_intensity": STRUCTURAL_TOLERANCE,
        "d_spacing": 2.0e-14,
        "position": 2.0e-14,
        "profile": PROFILE_TOLERANCE,
        "intensity_derivatives": PROFILE_KERNEL_TOLERANCE,
        "position_derivatives": PROFILE_TOLERANCE,
        "global_derivatives": PROFILE_TOLERANCE,
    }
    failing = {
        name: {"error": errors[name], "tolerance": tolerance}
        for name, tolerance in tolerances.items()
        if errors[name] >= tolerance
    }
    if failing:
        raise RuntimeError(f"GSAS-II structural numerical validation failed: {failing}")
    return errors


def validate_report(report: dict[str, Any], repetitions: int) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("implementation") != "GSAS-II"
        or report.get("revision") != PINNED_REVISION
        or report.get("scope") != SCOPE
    ):
        raise RuntimeError("external structural benchmark reported invalid provenance")
    for key in (
        "structure_factor_values",
        "structure_to_profile_values_and_cw_derivatives",
    ):
        values = report.get(key, {}).get("timings_ms")
        if (
            not isinstance(values, list)
            or len(values) != repetitions
            or not all(np.isfinite(value) and value >= 0.0 for value in values)
        ):
            raise RuntimeError(f"external structural benchmark returned invalid {key} timings")


def main() -> None:
    arguments = parse_args()
    if (
        arguments.reflections <= 0
        or arguments.sites <= 0
        or arguments.samples < 2
        or arguments.support_fwhm <= 0.0
        or arguments.warmups < 0
        or arguments.repetitions <= 0
    ):
        raise ValueError(
            "positive reflections/sites/support/repetitions, at least two samples, "
            "and non-negative warmups required"
        )
    if arguments.gsas_python is None or arguments.gsas_root is None:
        raise RuntimeError(
            "set --gsas-python and --gsas-root (or GSASII_PYTHON and RIETVELD_GSASII_ROOT)"
        )
    if arguments.require_release and rietveld._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {rietveld._core.BUILD_MODE!r}")

    x = np.linspace(5.0, 125.0, arguments.samples, dtype=np.float64)
    wavelength = 1.5406
    instrument = rietveld.ConstantWavelengthInstrument(
        wavelength,
        2.0e-4,
        -1.0e-4,
        1.2e-4,
        1.5e-3,
        3.0e-3,
    )
    instrument_array = np.array(
        [
            instrument.u_deg2,
            instrument.v_deg2,
            instrument.w_deg2,
            instrument.x_deg,
            instrument.y_deg,
        ],
        dtype=np.float64,
    )
    with tempfile.TemporaryDirectory(prefix="rietveld-gsasii-structural-comparison-") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        report_path = directory / "report.json"
        np.savez(
            input_path,
            x=x,
            instrument=instrument_array,
            wavelength_angstrom=np.array([wavelength]),
            support_fwhm=np.array([arguments.support_fwhm]),
            site_count=np.array([arguments.sites], dtype=np.int64),
            reflection_count=np.array([arguments.reflections], dtype=np.int64),
        )
        command = [
            str(arguments.gsas_python),
            str(WORKER),
            "--gsas-root",
            str(arguments.gsas_root),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--warmups",
            str(arguments.warmups),
            "--repetitions",
            str(arguments.repetitions),
        ]
        if arguments.binary_dir is not None:
            command.extend(["--binary-dir", str(arguments.binary_dir)])
        subprocess.run(command, check=True)
        gsas_report = json.loads(report_path.read_text())
        with np.load(output_path, allow_pickle=False) as archive:
            oracle = {name: archive[name].copy() for name in archive.files}
    validate_report(gsas_report, arguments.repetitions)

    phase = benchmark_phase(oracle["hkl"], oracle["multiplicity"], arguments.sites)
    experiment = rietveld.ConstantWavelengthExperiment.neutron(instrument)
    powder_pattern = rietveld.PowderPattern(x)
    prepared = rietveld.PreparedStructuralPattern(
        powder_pattern,
        experiment,
        phase,
        support_fwhm=arguments.support_fwhm,
    )
    if not prepared.uses_native_fused_path:
        raise RuntimeError("structural benchmark did not select the fused native path")

    def structure_operation() -> rietveld.StructureFactorValuesResult:
        return rietveld.calculate_structure_factor_values(
            phase.structure,
            phase.reflections.hkl,
            phase.reflections.multiplicity,
            phase.scattering,
            correction=phase.intensity_correction,
            scale=phase.scale,
        )

    structural, structural_timings = measure(
        structure_operation, arguments.warmups, arguments.repetitions
    )
    pattern_result, pattern_timings = measure(
        prepared.calculate, arguments.warmups, arguments.repetitions
    )
    errors = validate_outputs(structural, pattern_result, oracle)
    engine_structural = timing_summary(structural_timings)
    engine_pattern = timing_summary(pattern_timings)
    ratios = {
        "structure_factor_values": (
            gsas_report["structure_factor_values"]["median_ms"] / engine_structural["median_ms"]
        ),
        "structure_to_profile_values_and_cw_derivatives": (
            gsas_report["structure_to_profile_values_and_cw_derivatives"]["median_ms"]
            / engine_pattern["median_ms"]
        ),
    }
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "workload": {
            "probe": "monochromatic_neutron_cw",
            "space_group": "P 1",
            "sites": arguments.sites,
            "reflections": arguments.reflections,
            "samples": arguments.samples,
            "support_fwhm": arguments.support_fwhm,
            "active_peak_samples": (
                pattern_result.accumulation.derivatives.local.active_sample_count
            ),
            "outputs": [
                "complex_structure_factors",
                "f_squared",
                "integrated_intensity",
                "reflection_geometry",
                "profile",
                "local_intensity_position_derivatives",
                "uvwxy_derivatives",
            ],
        },
        "unit_convention": {
            "rietveld_engine_scattering_length": "fm",
            "gsasii_neutron_scattering_length": "1e-12_cm",
            "gsasii_f_squared_to_fm_squared": 100.0,
        },
        "numerical_normalized_maximum_errors": errors,
        "rietveld_engine": {
            "build_mode": rietveld._core.BUILD_MODE,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "structure_factor_values": engine_structural,
            "structure_to_profile_values_and_cw_derivatives": engine_pattern,
        },
        "gsasii": gsas_report,
        "median_speed_ratios_gsasii_over_rietveld": ratios,
    }
    print(
        f"scope={SCOPE} sites={arguments.sites} reflections={arguments.reflections} "
        f"samples={arguments.samples}"
    )
    for key, label in (
        ("structure_factor_values", "structure_factor_values"),
        (
            "structure_to_profile_values_and_cw_derivatives",
            "structure_to_profile_values_and_cw_derivatives",
        ),
    ):
        engine = report["rietveld_engine"][key]
        gsas = gsas_report[key]
        print(
            f"case={label} rietveld_median_ms={engine['median_ms']:.3f} "
            f"gsasii_median_ms={gsas['median_ms']:.3f} "
            f"ratio_gsasii_over_rietveld={ratios[key]:.3f}x"
        )
    print(
        "normalized_maximum_errors="
        + ",".join(f"{name}:{value:.3e}" for name, value in errors.items())
    )
    if arguments.json_output is not None:
        arguments.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
