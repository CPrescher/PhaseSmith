#!/usr/bin/env python3
"""Compare public PhaseSmith and XRD-Rust powder-stick calculations."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import phasesmith
from phasesmith.intensity_corrections import BraggBrentanoUnpolarizedLp

WAVELENGTH_ANGSTROM = 0.71073
TWO_THETA_RANGE_DEG = (2.0, 60.0)
SPECIES = ("C", "O", "Si", "Fe")
POSITION_TOLERANCE_DEG = 1.0e-5
MINIMUM_INTENSITY_CORRELATION = 0.999


@dataclass(frozen=True)
class Case:
    name: str
    atom_count: int
    cell: tuple[float, float, float, float, float, float]
    repetitions: int


CASES = (
    Case("small", 8, (10.0, 11.0, 12.0, 88.0, 91.0, 94.0), 9),
    Case("medium", 64, (18.0, 19.0, 20.0, 88.0, 91.0, 94.0), 7),
    Case("large", 256, (26.0, 27.0, 28.0, 88.0, 91.0, 94.0), 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *(case.name for case in CASES)), default="all")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--repetitions",
        type=int,
        help="override the case-specific repetition count",
    )
    parser.add_argument("--xrd-rust-threads", type=int, default=8)
    parser.add_argument("--phasesmith-threads", type=int, default=2)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def fractional_coordinates(atom_count: int) -> list[tuple[float, float, float]]:
    return [
        (
            (0.137 * index + 0.113) % 1.0,
            (0.271 * index + 0.217) % 1.0,
            (0.419 * index + 0.319) % 1.0,
        )
        for index in range(atom_count)
    ]


def make_structures(case: Case, structure_type: Any, lattice_type: Any) -> tuple[Any, Any]:
    a, b, c, alpha, beta, gamma = case.cell
    coordinates = fractional_coordinates(case.atom_count)
    species = [SPECIES[index % len(SPECIES)] for index in range(case.atom_count)]
    xrd_rust_structure = structure_type(
        lattice_type.from_parameters(a, b, c, alpha, beta, gamma),
        species,
        coordinates,
    )
    phasesmith_structure = phasesmith.CrystalStructure(
        case.name,
        case.name,
        phasesmith.UnitCell(a, b, c, alpha, beta, gamma),
        phasesmith.SpaceGroup.p1(),
        tuple(
            phasesmith.AtomSite(
                f"site-{index}",
                f"{species[index]}{index}",
                species[index],
                species[index],
                coordinates[index],
                1.0,
                0.0,
            )
            for index in range(case.atom_count)
        ),
    )
    return xrd_rust_structure, phasesmith_structure


def timing_summary(timings_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(timings_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min_ms": min(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": ordered[p95_index],
        "timings_ms": timings_ms,
    }


def measure_interleaved(
    operations: dict[str, Callable[[], Any]],
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float | list[float]]]]:
    names = tuple(operations)
    results: dict[str, Any] = {}
    for _ in range(warmups):
        for name in names:
            results[name] = operations[name]()

    timings = {name: [] for name in names}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for repetition in range(repetitions):
            offset = repetition % len(names)
            for name in (*names[offset:], *names[:offset]):
                started = time.perf_counter_ns()
                results[name] = operations[name]()
                timings[name].append((time.perf_counter_ns() - started) / 1.0e6)
    finally:
        if gc_enabled:
            gc.enable()
    return results, {name: timing_summary(values) for name, values in timings.items()}


def position_and_intensity_check(
    xrd_pattern: Any,
    phasesmith_positions: np.ndarray,
    phasesmith_intensities: np.ndarray,
) -> dict[str, float | int]:
    xrd_positions = np.asarray(xrd_pattern.x, dtype=np.float64)
    xrd_intensities = np.asarray(xrd_pattern.y, dtype=np.float64)
    phase_order = np.argsort(phasesmith_positions)
    phasesmith_positions = phasesmith_positions[phase_order]
    phasesmith_intensities = phasesmith_intensities[phase_order]
    phasesmith_intensities = 100.0 * phasesmith_intensities / np.max(phasesmith_intensities)
    xrd_intensities = 100.0 * xrd_intensities / np.max(xrd_intensities)

    insertion = np.searchsorted(phasesmith_positions, xrd_positions)
    position_errors: list[float] = []
    xrd_matched: list[float] = []
    phasesmith_matched: list[float] = []
    for source_index, candidate in enumerate(insertion):
        options = []
        if candidate < phasesmith_positions.size:
            options.append(candidate)
        if candidate > 0:
            options.append(candidate - 1)
        if not options:
            continue
        selected = min(
            options,
            key=lambda index: abs(phasesmith_positions[index] - xrd_positions[source_index]),
        )
        error = abs(phasesmith_positions[selected] - xrd_positions[source_index])
        if error <= POSITION_TOLERANCE_DEG:
            position_errors.append(error)
            xrd_matched.append(xrd_intensities[source_index])
            phasesmith_matched.append(phasesmith_intensities[selected])

    correlation = float("nan")
    if len(xrd_matched) > 1:
        correlation = float(np.corrcoef(xrd_matched, phasesmith_matched)[0, 1])
    result: dict[str, float | int] = {
        "xrd_rust_peak_count": int(xrd_positions.size),
        "phasesmith_family_count": int(phasesmith_positions.size),
        "matched_xrd_rust_peaks": len(position_errors),
        "max_position_error_deg": max(position_errors, default=float("nan")),
        "normalized_intensity_correlation": correlation,
    }
    if len(position_errors) != xrd_positions.size:
        raise RuntimeError(f"not every XRD-Rust peak matched a PhaseSmith reflection: {result}")
    if not np.isfinite(correlation) or correlation < MINIMUM_INTENSITY_CORRELATION:
        raise RuntimeError(f"normalized intensities did not agree: {result}")
    return result


def run_case(
    case: Case,
    arguments: argparse.Namespace,
    xrd_calculator_type: Any,
    structure_type: Any,
    lattice_type: Any,
) -> dict[str, Any]:
    xrd_rust_structure, phasesmith_structure = make_structures(case, structure_type, lattice_type)
    generator = phasesmith.PreparedReflectionGenerator(phasesmith_structure.space_group)
    reflection_range = phasesmith.CwTwoThetaRange(
        TWO_THETA_RANGE_DEG[0], TWO_THETA_RANGE_DEG[1], WAVELENGTH_ANGSTROM
    )
    scattering = phasesmith.XrayNonResonant()
    correction = BraggBrentanoUnpolarizedLp(WAVELENGTH_ANGSTROM)
    phasesmith_serial_policy = phasesmith.ExecutionPolicy(threads=1)
    phasesmith_parallel_policy = phasesmith.ExecutionPolicy(threads=arguments.phasesmith_threads)
    xrd_rust_serial = xrd_calculator_type(
        wavelength=WAVELENGTH_ANGSTROM,
        symprec=0,
        parallel=False,
        use_simd=True,
    )
    xrd_rust_parallel = xrd_calculator_type(
        wavelength=WAVELENGTH_ANGSTROM,
        symprec=0,
        parallel=True,
        num_threads=arguments.xrd_rust_threads,
        use_simd=True,
    )

    def run_phasesmith(execution: phasesmith.ExecutionPolicy) -> tuple[np.ndarray, np.ndarray]:
        generated = generator.generate(phasesmith_structure.cell, reflection_range)
        values = phasesmith.calculate_structure_factor_values(
            phasesmith_structure,
            generated.hkl,
            generated.multiplicity,
            scattering,
            correction=correction,
            execution=execution,
        )
        positions = 2.0 * np.degrees(
            np.arcsin(WAVELENGTH_ANGSTROM / (2.0 * generated.d_spacing_angstrom))
        )
        return positions, values.integrated_intensity

    def run_xrd_rust(calculator: Any) -> Any:
        return calculator.get_pattern(
            xrd_rust_structure,
            scaled=False,
            two_theta_range=TWO_THETA_RANGE_DEG,
        )

    operations = {
        "phasesmith_serial": lambda: run_phasesmith(phasesmith_serial_policy),
        f"phasesmith_{arguments.phasesmith_threads}_thread": lambda: run_phasesmith(
            phasesmith_parallel_policy
        ),
        "xrd_rust_serial_simd": lambda: run_xrd_rust(xrd_rust_serial),
        f"xrd_rust_{arguments.xrd_rust_threads}_thread_simd": lambda: run_xrd_rust(
            xrd_rust_parallel
        ),
    }
    repetitions = case.repetitions if arguments.repetitions is None else arguments.repetitions
    results, timings = measure_interleaved(operations, arguments.warmups, repetitions)
    phase_serial = results["phasesmith_serial"]
    phase_parallel = results[f"phasesmith_{arguments.phasesmith_threads}_thread"]
    np.testing.assert_array_equal(phase_parallel[0], phase_serial[0])
    np.testing.assert_array_equal(phase_parallel[1], phase_serial[1])
    numerical = position_and_intensity_check(
        results["xrd_rust_serial_simd"], phase_serial[0], phase_serial[1]
    )

    phase_serial_median = float(timings["phasesmith_serial"]["median_ms"])
    phase_parallel_median = float(
        timings[f"phasesmith_{arguments.phasesmith_threads}_thread"]["median_ms"]
    )
    xrd_serial_median = float(timings["xrd_rust_serial_simd"]["median_ms"])
    xrd_parallel_median = float(
        timings[f"xrd_rust_{arguments.xrd_rust_threads}_thread_simd"]["median_ms"]
    )
    return {
        "name": case.name,
        "atoms": case.atom_count,
        "cell": case.cell,
        "repetitions": repetitions,
        "numerical": numerical,
        "timing": timings,
        "ratios": {
            "phasesmith_parallel_speedup": phase_serial_median / phase_parallel_median,
            "xrd_rust_parallel_speedup": xrd_serial_median / xrd_parallel_median,
            "xrd_rust_serial_over_phasesmith_serial": xrd_serial_median / phase_serial_median,
            "xrd_rust_parallel_over_phasesmith_parallel": xrd_parallel_median
            / phase_parallel_median,
        },
    }


def main() -> None:
    arguments = parse_args()
    if (
        arguments.warmups < 0
        or (arguments.repetitions is not None and arguments.repetitions <= 0)
        or arguments.xrd_rust_threads <= 0
        or arguments.phasesmith_threads <= 0
    ):
        raise ValueError("thread/repetition counts must be positive and warmups non-negative")
    if arguments.require_release and phasesmith._core.BUILD_MODE != "release":
        raise RuntimeError(f"release extension required, imported {phasesmith._core.BUILD_MODE!r}")
    try:
        from pymatgen.core import Lattice, Structure
        from xrd_rust_calculator import XRDCalculatorRust
    except ImportError as error:
        raise RuntimeError(
            "install the optional benchmark dependencies xrd-rust==0.3.5 and pymatgen"
        ) from error

    selected_cases = (
        CASES
        if arguments.case == "all"
        else tuple(case for case in CASES if case.name == arguments.case)
    )
    report = {
        "schema_version": 1,
        "scope": (
            "Public APIs from already-constructed in-memory structures through reciprocal "
            "reflection generation, X-ray structure factors, Bragg-Brentano LP correction, "
            "and stick output; excludes CIF parsing, sampled profiles, and derivatives."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "phasesmith": importlib.metadata.version("phasesmith"),
            "xrd-rust": importlib.metadata.version("xrd-rust"),
            "pymatgen": importlib.metadata.version("pymatgen"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "phasesmith_build_mode": phasesmith._core.BUILD_MODE,
        "wavelength_angstrom": WAVELENGTH_ANGSTROM,
        "two_theta_range_deg": TWO_THETA_RANGE_DEG,
        "phasesmith_threads": arguments.phasesmith_threads,
        "xrd_rust_threads": arguments.xrd_rust_threads,
        "warmups": arguments.warmups,
        "cases": [
            run_case(case, arguments, XRDCalculatorRust, Structure, Lattice)
            for case in selected_cases
        ],
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if arguments.json_output is not None:
        arguments.json_output.write_text(f"{rendered}\n", encoding="utf-8")
        print(f"wrote {arguments.json_output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
