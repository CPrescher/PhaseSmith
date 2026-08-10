#!/usr/bin/env python3
"""Run the real POWGEN LaB6 TOF case with pinned external GSAS-II.

This worker imports no PhaseSmith module. It exports only plain NumPy arrays
and a JSON report for the comparison driver. Public scripting APIs perform the
data import and extraction; the version-gated ``newLeBail`` call matches the
existing pinned Le Bail oracle adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PINNED_REVISION = "c0bc79b259cdf0065480b5fbd57674ddf12c4a23"
PROFILE_D_SPACINGS = (0.4, 1.0, 4.0)
REFLECTION_COLUMNS = (
    "h",
    "k",
    "l",
    "multiplicity",
    "d_spacing_angstrom",
    "position_us",
    "sigma2_us2",
    "gamma_us",
    "f_obs2",
    "f_calc2",
    "phase_deg",
    "intensity_correction",
    "alpha_per_us",
    "beta_per_us",
    "wavelength_angstrom",
    "preferred_orientation",
    "transmission",
    "extinction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=8)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_gsasii(root: Path, binary_dir: Path) -> tuple[Any, Any, Any]:
    sys.path[:0] = [str(root), str(binary_dir)]
    from GSASII import GSASIIpath

    GSASIIpath.binaryPath = str(binary_dir)
    GSASIIpath.BinaryPathLoaded = True
    GSASIIpath.BinaryPathFailed = False
    GSASIIpath.LoadConfig()
    GSASIIpath.AddConfigValue({"Multiprocessing_cores": 0})
    from GSASII import GSASIIpwd, GSASIIscriptable, GSASIIstrMain

    return GSASIIpath, GSASIIpwd, (GSASIIscriptable, GSASIIstrMain)


def neutralize_sample_broadening(phase: Any) -> None:
    """Make the GSAS-II calculation match PhaseSmith's instrument-only scope."""

    for key, configure in (
        (
            "Size",
            lambda current: ["isotropic", [1.0e12, current[1][1], 1.0], *current[2:]],
        ),
        (
            "Mustrain",
            lambda current: ["isotropic", [0.0, current[1][1], 0.0], *current[2:]],
        ),
        (
            "Pref.Ori.",
            lambda current: ["MD", 1.0, False, current[3], *current[4:]],
        ),
    ):
        matches = phase.getHAPentryList(0, key)
        if len(matches) != 1:
            raise RuntimeError(f"expected one {key} HAP entry")
        path = matches[0][0]
        phase.setHAPentryValue(path, configure(phase.getHAPentryValue(path)))


def instrument_values(histogram: Any) -> dict[str, float]:
    instrument = histogram.data["Instrument Parameters"][0]
    names = (
        "Zero",
        "difC",
        "difA",
        "difB",
        "alpha",
        "beta-0",
        "beta-1",
        "sig-0",
        "sig-1",
        "sig-2",
        "X",
        "Y",
        "Z",
    )
    return {name: float(instrument[name][1]) for name in names}


def profile_probes(profile_module: Any, instrument: dict[str, float]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for d_spacing in PROFILE_D_SPACINGS:
        d = d_spacing
        position = (
            instrument["Zero"]
            + instrument["difC"] * d
            + instrument["difA"] * d**2
            + instrument["difB"] / d
        )
        alpha = instrument["alpha"] / d
        beta = instrument["beta-0"] + instrument["beta-1"] / d**4
        sigma2 = instrument["sig-0"] + instrument["sig-1"] * d**2 + instrument["sig-2"] * d**4
        scale = max(np.sqrt(sigma2), 1.0 / alpha, 1.0 / beta)
        x = np.linspace(position - 30.0 * scale, position + 30.0 * scale, 12_001)
        y, reported_integral = profile_module.getEpsVoigt(position, alpha, beta, sigma2, 0.0, x)
        key = str(d_spacing).replace(".", "_")
        arrays[f"profile_{key}_x"] = np.ascontiguousarray(x, dtype=np.float64)
        arrays[f"profile_{key}_y"] = np.ascontiguousarray(y, dtype=np.float64)
        arrays[f"profile_{key}_integral"] = np.asarray([reported_integral], dtype=np.float64)
    return arrays


def run_workflow(
    scripting: Any,
    structure_main: Any,
    profile_module: Any,
    data: Path,
    cycles: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    with tempfile.TemporaryDirectory(prefix="phasesmith-gsasii-powgen-") as temporary:
        project = scripting.G2Project(newgpx=str(Path(temporary) / "powgen.gpx"))
        histogram = project.add_powder_histogram(
            str(data / "PG3_17541.gsa"),
            str(data / "PGHR_60-2015A.prm"),
            fmthint="GSAS",
            databank=1,
            instbank=2,
        )
        phase = project.add_phase(
            phasename="LaB6",
            spacegroup="P m -3 m",
            cell=[4.156826, 4.156826, 4.156826, 90.0, 90.0, 90.0],
            histograms=[histogram],
        )
        phase.add_atom(0.0, 0.0, 0.0, element="La", lbl="La", occ=1.0, uiso=0.01)
        phase.add_atom(0.2, 0.5, 0.5, element="B", lbl="B", occ=1.0, uiso=0.01)
        neutralize_sample_broadening(phase)
        x = np.asarray(histogram.getdata("X"), dtype=np.float64)
        histogram.set_refinements(
            {
                "Limits": [float(x[0]), float(x[-1])],
                "Background": {
                    "type": "chebyschev-1",
                    "refine": True,
                    "no. coeffs": 16,
                },
            }
        )
        histogram.data["Sample Parameters"]["Scale"][1] = False
        phase.set_HAP_refinements({"Scale": False})
        project.set_Controls("cycles", 1)
        project.do_refinements([{}], outputnames=[None])
        phase.set_refinements({"LeBail": True})
        project.index_ids()
        structure_main.Refine(project.filename, newLeBail=True)
        project.reload()

        convergence = []
        for cycle in range(cycles):
            current = project.histogram(0)
            convergence.append(
                [
                    cycle,
                    float(current.residuals["R"]),
                    float(current.residuals["wR"]),
                    float(current.residuals["Rb"]),
                    float(current.residuals["wRb"]),
                ]
            )
            if cycle + 1 < cycles:
                project.refine(makeBack=False)

        histogram = project.histogram(0)
        reflections = np.ascontiguousarray(
            histogram.reflections()["LaB6"]["RefList"], dtype=np.float64
        )
        if reflections.shape[1] != len(REFLECTION_COLUMNS):
            raise RuntimeError(f"unexpected reflection-list shape {reflections.shape}")
        instrument = instrument_values(histogram)
        arrays = {
            "x_us": np.ascontiguousarray(histogram.getdata("X"), dtype=np.float64),
            "observed_y": np.ascontiguousarray(histogram.getdata("Yobs"), dtype=np.float64),
            "calculated_y": np.ascontiguousarray(histogram.getdata("Ycalc"), dtype=np.float64),
            "background_y": np.ascontiguousarray(histogram.getdata("Background"), dtype=np.float64),
            "weight": np.ascontiguousarray(histogram.getdata("Yweight"), dtype=np.float64),
            "reflection_list": reflections,
            "convergence": np.ascontiguousarray(convergence, dtype=np.float64),
        }
        arrays.update(profile_probes(profile_module, instrument))

        observed = arrays["observed_y"]
        calculated = arrays["calculated_y"]
        background = arrays["background_y"]
        residual = calculated - observed
        result = {
            "sample_count": int(observed.size),
            "reflection_count": int(reflections.shape[0]),
            "poisson_rwp": float(histogram.get_wR()) / 100.0,
            "unit_weight_rwp": float(np.sqrt(np.sum(residual**2) / np.sum(observed**2))),
            "profile_correlation": float(
                np.corrcoef(observed - background, calculated - background)[0, 1]
            ),
            "instrument": instrument,
            "convergence_columns": (
                "cycle",
                "R_percent",
                "wR_percent",
                "Rb_percent",
                "wRb_percent",
            ),
        }
        return result, arrays


def main() -> None:
    arguments = parse_args()
    root = arguments.gsas_root.resolve()
    revision = git_revision(root)
    if revision != PINNED_REVISION:
        raise RuntimeError("GSAS-II checkout does not match the recorded pin")
    gsasii_path, profile_module, modules = configure_gsasii(root, arguments.binary_dir.resolve())
    scripting, structure_main = modules
    result, arrays = run_workflow(
        scripting,
        structure_main,
        profile_module,
        arguments.data_directory.resolve(),
        arguments.cycles,
    )
    arguments.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.archive, **arrays)
    report = {
        "schema_version": 1,
        "implementation": "GSAS-II",
        "scope": "powgen_lab6_tof_lebail",
        "revision": revision,
        "tag": int(gsasii_path.GetVersionNumber()),
        "input_sha256": {
            name: sha256(arguments.data_directory / name)
            for name in ("PG3_17541.gsa", "PGHR_60-2015A.prm")
        },
        "oracle_behavior": {
            "coordinate_convention": "GSAS-II calculation bin centers",
            "sample_broadening": "effectively neutral fixed HAP values",
            "background": "16-term refined chebyschev-1",
            "private_probe": ["GSASIIstrMain.Refine(newLeBail=True)", "GSASIIpwd.getEpsVoigt"],
            "reflection_columns": REFLECTION_COLUMNS,
        },
        "archive": {"file": arguments.archive.name, "sha256": sha256(arguments.archive)},
        "result": result,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
