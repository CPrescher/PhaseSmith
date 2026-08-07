#!/usr/bin/env python3
"""Benchmark pinned GSAS-II structure factors and controlled CW composition.

This external-only worker imports no PhaseSmith module. Public scripting
constructs the crystal and reflection list once. A revision-gated internal
adapter then times prepared numerical calls without project I/O.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_cw_profile import accumulate

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
SCOPE = "neutron_structure_factors_and_controlled_symmetric_cw_pattern"
SPECIES = ("C", "O", "Si", "Fe")
GSAS_F_SQUARED_TO_FM_SQUARED = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def import_gsasii(gsas_root: Path, binary_dir: Path | None) -> tuple[Any, ...]:
    revision = git_revision(gsas_root)
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"GSAS-II benchmark requires revision {PINNED_REVISION}, detected {revision}"
        )
    sys.path.insert(0, str(gsas_root))
    from GSASII import GSASIIpath

    if binary_dir is not None:
        sys.path.insert(0, str(binary_dir))
        GSASIIpath.binaryPath = str(binary_dir)
        GSASIIpath.BinaryPathLoaded = True
        GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIlattice, GSASIIpwd, GSASIIscriptable, GSASIIstrIO, GSASIIstrMath

    if not hasattr(GSASIIpwd, "pyd"):
        raise RuntimeError("GSAS-II pypowder binary is unavailable")
    return GSASIIscriptable, GSASIIstrIO, GSASIIstrMath, GSASIIlattice, GSASIIpwd


def instrument_text(wavelength: float, instrument: np.ndarray) -> str:
    u_deg2, v_deg2, w_deg2, x_deg, y_deg = instrument
    return "".join(
        [
            "# GSAS-II neutron CW instrument for a structural benchmark\n",
            "Type:PNC;Bank:1\n",
            f"Lam:{wavelength:.12g};Zero:0;Polariz.:0;Azimuth:0\n",
            f"U:{10_000.0 * u_deg2:.12g};V:{10_000.0 * v_deg2:.12g};",
            f"W:{10_000.0 * w_deg2:.12g};X:{100.0 * x_deg:.12g};",
            f"Y:{100.0 * y_deg:.12g};Z:0;SH/L:0\n",
        ]
    )


def configure_neutral_sample(phase: Any) -> None:
    """Disable sample broadening and preferred orientation for this workload."""

    for key, configure in (
        ("Size", lambda current: ["isotropic", [1.0e12, current[1][1], 1.0], *current[2:]]),
        ("Mustrain", lambda current: ["isotropic", [0.0, current[1][1], 0.0], *current[2:]]),
        ("Pref.Ori.", lambda current: ["MD", 1.0, False, current[3], *current[4:]]),
    ):
        entries = phase.getHAPentryList(0, key)
        if len(entries) != 1:
            raise RuntimeError(f"expected one {key} HAP entry, found {len(entries)}")
        path = entries[0][0]
        phase.setHAPentryValue(path, configure(phase.getHAPentryValue(path)))


def prepare_structure_factor_call(
    modules: tuple[Any, ...],
    wavelength: float,
    instrument: np.ndarray,
    site_count: int,
    reflection_count: int,
    x: np.ndarray,
) -> tuple[Any, dict[str, Any], tuple[Any, ...], Any]:
    scripting, structure_io, structure_math, lattice, profile_module = modules
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-structural-benchmark-") as temporary:
        np.random.seed(20_260_806)
        directory = Path(temporary)
        instrument_path = directory / "neutron.instprm"
        instrument_path.write_text(instrument_text(wavelength, instrument), encoding="utf-8")
        project = scripting.G2Project(newgpx=str(directory / "benchmark.gpx"))
        phase = project.add_phase(
            phasename="benchmark",
            spacegroup="P 1",
            cell=[15.0, 15.0, 15.0, 90.0, 90.0, 90.0],
        )
        for site in range(site_count):
            element = SPECIES[site % len(SPECIES)]
            phase.add_atom(
                (0.137 * site) % 1.0,
                (0.271 * site + 0.11) % 1.0,
                (0.419 * site + 0.23) % 1.0,
                element=element,
                lbl=f"{element}{site}",
                occ=1.0,
                uiso=0.005 + 0.0002 * site,
            )
        project.add_simulated_powder_histogram(
            "benchmark",
            str(instrument_path),
            float(x[0]),
            float(x[-1]),
            Npoints=int(x.size),
            scale=1.0,
            phases=[phase],
        )
        configure_neutral_sample(phase)
        project.do_refinements([{}], outputnames=[None])

        project_path = project.filename
        controls = structure_io.GetControls(project_path)
        calculation_controls = dict(controls)
        histograms, phases = structure_io.GetUsedHistogramsAndPhases(project_path)
        rigid_bodies = structure_io.GetRigidBodies(project_path)
        rigid_body_ids = rigid_bodies.get("RBIds", {"Vector": [], "Residue": []})
        _, rigid_body_parameters = structure_io.GetRigidBodyModels(rigid_bodies, Print=False)
        phase_data = structure_io.GetPhaseData(phases, {}, rigid_body_ids, Print=False)
        (
            atom_count,
            atom_indices,
            _,
            phase_parameters,
            _,
            xray_tables,
            electron_tables,
            orbital_tables,
            neutron_tables,
            magnetic_tables,
            maximum_modulation_wave,
        ) = phase_data
        calculation_controls.update(
            atomIndx=atom_indices,
            Natoms=atom_count,
            FFtables=xray_tables,
            EFtables=electron_tables,
            ORBtables=orbital_tables,
            BLtables=neutron_tables,
            MFtables=magnetic_tables,
            maxSSwave=maximum_modulation_wave,
        )
        _, histogram_phase_parameters, controls = structure_io.GetHistogramPhaseData(
            phases, histograms, Controls=calculation_controls, Print=False
        )
        calculation_controls.update(controls)
        _, histogram_parameters, _, controls = structure_io.GetHistogramData(
            histograms, Print=False
        )
        calculation_controls.update(controls)
        parameters: dict[str, Any] = {}
        parameters.update(rigid_body_parameters)
        parameters.update(phase_parameters)
        parameters.update(histogram_phase_parameters)
        parameters.update(histogram_parameters)
        structure_io.GetFprime(calculation_controls, histograms)

        histogram_name = next(name for name in histograms if name.startswith("PWDR"))
        histogram = histograms[histogram_name]
        phase_record = phases["benchmark"]
        histogram_prefix = f":{histogram['hId']}:"
        phase_prefix = f"{phase_record['pId']}::"
        reciprocal_parameters = [parameters[f"{phase_prefix}A{index}"] for index in range(6)]
        reciprocal_metric, _ = lattice.A2Gmat(reciprocal_parameters)
        source_reflections = histogram["Reflection Lists"]["benchmark"]["RefList"]
        if len(source_reflections) < reflection_count:
            raise RuntimeError(
                f"GSAS-II generated {len(source_reflections)} reflections, "
                f"but {reflection_count} were requested"
            )
        reflection_dictionary = {
            "RefList": np.array(source_reflections[:reflection_count], copy=True),
            "FF": {},
        }
        call = (
            reflection_dictionary,
            reciprocal_metric,
            histogram_prefix,
            phase_prefix,
            phase_record["General"]["SGData"],
            calculation_controls,
            parameters,
        )
    return structure_math, reflection_dictionary, call, profile_module


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


def main() -> None:
    arguments = parse_args()
    if arguments.warmups < 0 or arguments.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    modules = import_gsasii(arguments.gsas_root.resolve(), arguments.binary_dir)
    with np.load(arguments.input, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive["x"], dtype=np.float64)
        instrument = np.ascontiguousarray(archive["instrument"], dtype=np.float64)
        wavelength = float(archive["wavelength_angstrom"][0])
        support_fwhm = float(archive["support_fwhm"][0])
        site_count = int(archive["site_count"][0])
        reflection_count = int(archive["reflection_count"][0])
    structure_math, reflection_dictionary, call, profile_module = prepare_structure_factor_call(
        modules, wavelength, instrument, site_count, reflection_count, x
    )

    def structure_operation() -> np.ndarray:
        structure_math.StructureFactor2(*call)
        return reflection_dictionary["RefList"][:, 9]

    _, structure_timings = measure(structure_operation, arguments.warmups, arguments.repetitions)
    reflection_list = reflection_dictionary["RefList"]
    positions = np.ascontiguousarray(reflection_list[:, 5], dtype=np.float64)
    multiplicity = np.ascontiguousarray(reflection_list[:, 3], dtype=np.int64)

    def pattern_operation() -> tuple[np.ndarray, ...]:
        structure_math.StructureFactor2(*call)
        integrated = (
            GSAS_F_SQUARED_TO_FM_SQUARED * multiplicity * reflection_dictionary["RefList"][:, 9]
        )
        return accumulate(profile_module, x, positions, integrated, instrument, support_fwhm)

    pattern_result, pattern_timings = measure(
        pattern_operation, arguments.warmups, arguments.repetitions
    )
    assert pattern_result is not None
    structure_operation()
    reflection_list = reflection_dictionary["RefList"]
    f_squared = np.ascontiguousarray(
        GSAS_F_SQUARED_TO_FM_SQUARED * reflection_list[:, 9], dtype=np.float64
    )
    f_fm = np.ascontiguousarray(
        np.sqrt(f_squared) * np.exp(1j * np.deg2rad(reflection_list[:, 10])),
        dtype=np.complex128,
    )
    integrated_intensity = np.ascontiguousarray(multiplicity * f_squared, dtype=np.float64)
    np.savez(
        arguments.output,
        hkl=np.ascontiguousarray(reflection_list[:, :3], dtype=np.int64),
        multiplicity=multiplicity,
        d_spacing_angstrom=np.ascontiguousarray(reflection_list[:, 4], dtype=np.float64),
        position_deg=positions,
        f_fm=f_fm,
        f_squared_fm2=f_squared,
        integrated_intensity=integrated_intensity,
        y=pattern_result[0],
        starts=pattern_result[1],
        offsets=pattern_result[2],
        local=pattern_result[3],
        global_jacobian=pattern_result[4],
    )
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "revision": PINNED_REVISION,
        "scope": SCOPE,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "warmups": arguments.warmups,
        "repetitions": arguments.repetitions,
        "structure_factor_values": timing_summary(structure_timings),
        "structure_to_profile_values_and_cw_derivatives": timing_summary(pattern_timings),
    }
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
