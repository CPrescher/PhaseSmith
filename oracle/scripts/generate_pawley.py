#!/usr/bin/env python3
"""Explicit pinned GSAS-II Pawley optimizer fixture; never a runtime dependency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import generate_lebail as common
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CELL = [3.7, 4.2, 5.1, 87.0, 95.0, 102.0]
START = [3.7005, 4.1995, 5.1005, 87.001, 94.999, 102.001]


def descriptor(name, array):
    record = common.array_descriptor(name, array)
    if name.endswith("x_deg"):
        record["unit"] = "degree_2theta"
    elif name.endswith("weight"):
        record["unit"] = "inverse_intensity_squared"
    elif "reflections" in name:
        record["unit"] = "mixed_by_column"
    elif name.endswith("cell"):
        record["unit"] = "angstrom_lengths_degree_angles"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=24001)
    args = parser.parse_args()
    pin = json.loads((ROOT / "oracle/PINNED_GSASII.json").read_text())
    if common.git_revision(args.gsas_root) != pin["revision"]:
        raise RuntimeError("GSAS-II revision does not match pin")
    if args.output.exists():
        raise FileExistsError("choose a new output directory for explicit regeneration")
    if not 3001 <= args.points <= 25000:
        raise ValueError("points must be between 3001 and the pinned public API limit of 25000")
    adapter_path = ROOT / "python/phasesmith/oracle/_pinned_probe.py"
    spec = importlib.util.spec_from_file_location("pinned_probe", adapter_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    gsas_path, gsas, _ = common.import_gsasii(args.gsas_root, args.binary_dir)
    np.random.seed(20_260_917)
    arrays, cases = {}, []
    for mode in ("fixed_cell", "refined_cell"):
        with tempfile.TemporaryDirectory(prefix="phasesmith-pawley-oracle-") as temp:
            work = Path(temp)
            instrument = work / "pawley.instprm"
            instrument.write_text(common.instrument_text())
            project = gsas.G2Project(newgpx=str(work / "pawley.gpx"))
            phase = project.add_phase(phasename="pawley", spacegroup="P 1", cell=CELL)
            phase.add_atom(0, 0, 0, element="Si", lbl="Si1", occ=1.0, uiso=0.01)
            histogram = project.add_simulated_powder_histogram(
                "pawley",
                str(instrument),
                20.0,
                65.0,
                Npoints=args.points,
                scale=100.0,
                phases=[phase],
            )
            common.configure_neutral_sample(phase)
            histogram.clear_refinements({"Sample Parameters": ["Scale"]})
            project.do_refinements([{}], outputnames=[None])
            phase, histogram = project.phase(0), project.histogram(0)
            refs = np.asarray(histogram.reflections()["pawley"]["RefList"], dtype=np.float64)
            observed = np.asarray(histogram.getdata("Ycalc"), dtype=np.float64)
            probe.initialize_pawley(
                phase,
                histogram,
                gsas,
                refs,
                observed,
                starting_cell=START if mode == "refined_cell" else None,
            )
            if mode == "refined_cell":
                phase.set_refinements({"Cell": True})
            project.set_Controls("cycles", 30)
            project.refine(makeBack=False)
            histogram, phase = project.histogram(0), project.phase(0)
            cell = phase.get_cell()
            current = {
                "x_deg": histogram.getdata("X"),
                "observed_y": observed,
                "ycalc": histogram.getdata("Ycalc"),
                "background": histogram.getdata("Background"),
                "weight": histogram.getdata("Yweight"),
                "initial_reflections": refs,
                "reflections": histogram.reflections()["pawley"]["RefList"],
                "cell": [
                    cell[k]
                    for k in (
                        "length_a",
                        "length_b",
                        "length_c",
                        "angle_alpha",
                        "angle_beta",
                        "angle_gamma",
                    )
                ],
            }
            for key, value in current.items():
                arrays[f"{mode}_{key}"] = np.ascontiguousarray(value, dtype=np.float64)
            cases.append(
                {
                    "id": mode,
                    "case_kind": "histogram",
                    "arrays": {k: f"{mode}_{k}" for k in current},
                    "parameters": {
                        "method": "Pawley",
                        "instrument": common.INSTRUMENT,
                        "range_deg": [20.0, 65.0],
                        "formal_sh_over_l": 0.0,
                        "pinned_minimum_sh_over_l": 0.002,
                        "reflection_columns": common.REFLECTION_COLUMNS,
                        "cell": CELL,
                        "starting_cell": START if mode == "refined_cell" else CELL,
                        "initial_f_squared_fraction": 0.5,
                        "signed_f_squared": True,
                        "gsasii_residuals": {
                            k: float(v)
                            for k, v in histogram.residuals.items()
                            if isinstance(v, (int, float))
                        },
                    },
                }
            )
    args.output.mkdir(parents=True)
    archive = args.output / "data.npz"
    np.savez_compressed(archive, **arrays)
    inputs = {
        "instrument": common.INSTRUMENT,
        "cell": CELL,
        "starting_cell": START,
        "points": args.points,
    }
    manifest = {
        "format_version": 1,
        "fixture_id": "gsasii_pawley_optimizer_v1",
        "kind": "powder_histogram",
        "provenance": {
            "gsasii_repository": pin["repository"],
            "gsasii_revision": pin["revision"],
            "gsasii_tag": int(gsas_path.GetVersionNumber()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generator_sha256": common.sha256_file(Path(__file__)),
            "adapter_sha256": common.sha256_file(adapter_path),
            "helper_sha256": common.sha256_file(Path(common.__file__)),
            "binary_sha256": {
                p.name: common.sha256_file(p) for p in sorted(args.binary_dir.glob("*.so"))
            },
        },
        "source": {
            "api": ["G2Project.refine", "G2PwdrData.getdata/reflections", "G2Phase.get_cell"],
            "private_probe": True,
            "adapter_version": 1,
            "native_units": {
                "position": "degree",
                "sigma2": "centidegree^2",
                "gamma": "centidegree",
                "F_squared": "GSAS-II phase convention",
                "area": (
                    "0.01 * F_obs_squared * intensity_correction; multiplicity already included"
                ),
            },
        },
        "input": {
            "kind": "parameter_cases",
            "sha256": common.sha256_bytes(json.dumps(inputs, sort_keys=True).encode()),
        },
        "archive": {"file": "data.npz", "sha256": common.sha256_file(archive)},
        "arrays": {name: descriptor(name, array) for name, array in arrays.items()},
        "cases": cases,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
