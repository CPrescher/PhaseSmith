#!/usr/bin/env python3
"""Build unmodified external FCJ sources to distinguish precision and quadrature.

No GSAS-II implementation is copied into this repository. This explicit,
optional diagnostic requires gfortran and writes into a new output directory.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(library, scalar, row, grid):
    """Call the external PSVFCJO ABI using its documented centidegree units."""
    routine = library.psvfcjo_
    routine.restype = None
    routine.argtypes = [ct.POINTER(scalar)] * 12
    values = []
    for observation in grid:
        delta = scalar(scalar(observation - row[5]).value * 100).value
        centre = scalar(scalar(row[5]).value * 100).value
        half_height = scalar(scalar(0.002).value / 2).value
        inputs = list(map(scalar, [delta, centre, row[6], row[7], half_height, half_height]))
        outputs = [scalar() for _ in range(6)]
        routine(*[ct.byref(value) for value in inputs + outputs])
        values.append(100 * outputs[0].value)
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((ROOT / "oracle/PINNED_GSASII.json").read_text())["revision"]
    revision = subprocess.check_output(
        ["git", "-C", str(args.gsas_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != pin:
        raise ValueError("source revision differs from pin")
    directory = args.gsas_root / "sources/powsubs"
    names = ["psvfcjo", "psvoigt", "gauleg", "sind", "cosd", "tand", "acosd", "lorentz"]
    sources = [directory / (name + ".for") for name in names]
    for source in sources:
        relative = source.relative_to(args.gsas_root)
        original = subprocess.check_output(
            ["git", "-C", str(args.gsas_root), "show", f"HEAD:{relative}"]
        )
        if source.read_bytes() != original:
            raise ValueError(f"external source modified: {relative}")
    args.output.mkdir(parents=True, exist_ok=False)
    builds = []
    for name, flags, scalar in (
        ("single", [], ct.c_float),
        ("double", ["-fdefault-real-8", "-freal-4-real-8"], ct.c_double),
    ):
        destination = (args.output / f"{name}.dylib").resolve()
        command = [
            args.compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-std=legacy",
            "-ffixed-line-length-none",
            *flags,
            "-I",
            str(directory),
            *map(str, sources),
            "-o",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        builds.append((name, ct.CDLL(str(destination)), scalar, command, sha(destination)))
    audit_path = ROOT / "validation/results/fcj-angular-20260918-corrected.json"
    audit = json.loads(audit_path.read_text())
    with np.load(ROOT / "oracle/fixtures/pawley_optimizer_v1/data.npz", allow_pickle=False) as data:
        rows = data["fixed_cell_reflections"]
    # Published FCJ endpoint geometry, independently of the external implementation.
    spans = rows[:, 5] - np.rad2deg(
        np.arccos(np.cos(np.deg2rad(rows[:, 5])) * np.sqrt(1 + 0.002**2))
    )
    results = []
    for case in audit["results"]:
        index = case["reflection"]
        reference, pinned = np.asarray(case["angular_reference"]), np.asarray(case["oracle"])
        comparisons = {}
        for name, library, scalar, _command, _hash in builds:
            values = evaluate(library, scalar, rows[index], case["x_deg"])
            comparisons[name] = dict(
                values=values.tolist(),
                vs_integral=float(np.linalg.norm(values - reference) / np.linalg.norm(reference)),
                vs_pinned=float(np.linalg.norm(values - pinned) / np.linalg.norm(pinned)),
            )
        # Mathematical consequence of retaining only the positive root of P2:
        # every normalized quadrature weight cancels, leaving one shifted R.
        # Use the external intrinsic profile to isolate quadrature from TCH
        # coefficient conventions. This is a control, not a replacement FCJ.
        centre = rows[index, 5] - spans[index] * (1 - 1 / np.sqrt(3))
        intrinsic = builds[1][1].psvoigt_
        intrinsic.restype = None
        intrinsic.argtypes = [ct.POINTER(ct.c_double)] * 7
        predicted = []
        for observation in case["x_deg"]:
            arguments = list(
                map(
                    ct.c_double,
                    [(observation - centre) * 100, rows[index, 6], rows[index, 7], 0, 0, 0, 0],
                )
            )
            intrinsic(*[ct.byref(value) for value in arguments])
            predicted.append(100 * arguments[3].value)
        double_values = np.asarray(comparisons["double"]["values"])
        one_node_error = float(
            np.linalg.norm(predicted - double_values) / np.linalg.norm(double_values)
        )
        results.append(
            dict(
                reflection=index,
                position_deg=float(rows[index, 5]),
                axial_span_deg=float(spans[index]),
                one_node_shift_deg=float(centre - rows[index, 5]),
                double_vs_single_node_prediction=one_node_error,
                comparisons=comparisons,
            )
        )
    report = dict(
        gsasii_revision=revision,
        runner_sha256=sha(Path(__file__)),
        angular_audit_sha256=sha(audit_path),
        compiler=subprocess.check_output([args.compiler, "--version"], text=True),
        sources_sha256={str(p.relative_to(args.gsas_root)): sha(p) for p in sources},
        builds={
            name: dict(command=command, binary_sha256=digest)
            for name, _lib, _type, command, digest in builds
        },
        # Source inspection: this indicator is truncated to an integer to select
        # the table. Its maximum <1 proves the first table is used for all rows.
        selection_indicator_range=[
            float(min(300 * spans / rows[:, 7])),
            float(max(300 * spans / rows[:, 7])),
        ],
        results=results,
        scope="Unmodified pinned source rebuilds; precision promotion does not increase quadrature",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
