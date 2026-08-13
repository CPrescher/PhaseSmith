"""Benchmark fused and separately orchestrated structural CW calculations."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import phasesmith


def measure(operation: Callable[[], Any], warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        operation()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    return timings


def report(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
    print(
        f"case={name} min_ms={min(timings):.3f} "
        f"median_ms={statistics.median(timings):.3f} p95_ms={p95:.3f}"
    )


def benchmark_hkl(reflection_count: int) -> np.ndarray:
    candidates = np.array(
        [
            (h, k, ell)
            for h in range(20)
            for k in range(20)
            for ell in range(20)
            if (h, k, ell) != (0, 0, 0) and h * h + k * k + ell * ell < 340
        ],
        dtype=np.int64,
    )
    if candidates.shape[0] < reflection_count:
        raise ValueError("requested reflection count exceeds the benchmark search set")
    return np.ascontiguousarray(candidates[:reflection_count])


def benchmark_phase(reflection_count: int, site_count: int) -> phasesmith.RietveldPhase:
    sites = tuple(
        phasesmith.AtomSite(
            f"site-{site}",
            f"{('C', 'O', 'Si', 'Fe')[site % 4]}{site}",
            ("C", "O", "Si", "Fe")[site % 4],
            ("C", "O", "Si", "Fe")[site % 4],
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
    structure = phasesmith.CrystalStructure(
        "benchmark",
        "Benchmark structure",
        phasesmith.UnitCell(15.0, 15.0, 15.0, 90.0, 90.0, 90.0),
        phasesmith.SpaceGroup.p1(),
        sites,
    )
    hkl = benchmark_hkl(reflection_count)
    return phasesmith.RietveldPhase(
        "benchmark",
        "Benchmark phase",
        structure,
        phasesmith.StructuralReflectionBatch(
            tuple(f"{h},{k},{ell}" for h, k, ell in hkl),
            hkl,
            np.ones(reflection_count, dtype=np.int64),
        ),
        phasesmith.XrayNonResonant(),
        phasesmith.NeutralIntegratedIntensityCorrection(),
        1.3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflections", type=int, default=256)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--require-release", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.reflections <= 0
        or arguments.sites <= 0
        or arguments.warmups < 0
        or arguments.repetitions <= 0
    ):
        raise ValueError("counts and repetitions must be positive; warmups non-negative")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")

    phase = benchmark_phase(arguments.reflections, arguments.sites)
    instrument = phasesmith.ConstantWavelengthInstrument(
        1.5406,
        2.0e-4,
        -1.0e-4,
        1.2e-4,
        1.5e-3,
        3.0e-3,
    )
    experiment = phasesmith.ConstantWavelengthExperiment.x_ray(instrument)
    pattern = phasesmith.PowderPattern(np.linspace(5.0, 125.0, 20_001))
    prepared = phasesmith.PreparedStructuralPattern(pattern, experiment, phase)
    if not prepared.uses_native_fused_path:
        raise RuntimeError("benchmark phase did not select the fused native path")

    def separate() -> phasesmith.AccumulationResult:
        structural = phasesmith.calculate_structure_factors(
            phase.structure,
            phase.reflections.hkl,
            phase.reflections.multiplicity,
            phase.scattering,
            correction=phase.intensity_correction,
            scale=phase.scale,
        )
        spacing = phase.structure.cell.d_spacings(phase.reflections.hkl).d_spacing_angstrom
        positions = 2.0 * np.degrees(np.arcsin(instrument.wavelength_angstrom / (2.0 * spacing)))
        return phasesmith.accumulate_cw(
            pattern.x,
            positions,
            structural.integrated_intensity,
            instrument,
        )

    fused_result = prepared.calculate()
    separate_result = separate()
    np.testing.assert_allclose(fused_result.profile_y, separate_result.y, rtol=5e-12, atol=1e-6)
    fused = measure(prepared.calculate, arguments.warmups, arguments.repetitions)
    stephens_phase = replace(
        phase,
        physics=phasesmith.StephensOrthorhombicBroadening(
            (2.0e-12, 3.0e-12, 1.0e-12, 8.0e-13, 6.0e-13, 7.0e-13), 0.35
        ),
    )
    prepared_stephens = phasesmith.PreparedStructuralPattern(pattern, experiment, stephens_phase)
    if not prepared_stephens.uses_native_fused_path:
        raise RuntimeError("Stephens benchmark did not select the fused native path")
    stephens = measure(prepared_stephens.calculate, arguments.warmups, arguments.repetitions)
    separated = measure(separate, arguments.warmups, arguments.repetitions)
    print(
        f"python={platform.python_version()} platform={platform.platform()} "
        f"build_mode={phasesmith._core.BUILD_MODE} reflections={arguments.reflections} "
        f"sites={arguments.sites} samples={pattern.x.size}"
    )
    report("fused_values_and_products_ready", fused)
    report("fused_with_orthorhombic_stephens", stephens)
    report("separate_public_dense_structural_derivatives", separated)


if __name__ == "__main__":
    main()
