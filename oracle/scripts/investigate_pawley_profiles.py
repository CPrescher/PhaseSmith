#!/usr/bin/env python3
"""Explicit black-box FCJ controls; leaves the pinned Pawley golden fixture untouched."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import generate_fcj_profile as common
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsas-root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((ROOT / "oracle/PINNED_GSASII.json").read_text())
    if common.git_revision(args.gsas_root) != pin["revision"]:
        raise ValueError("oracle revision differs from pin")
    source = ROOT / "oracle/fixtures/pawley_optimizer_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if sha(source / "data.npz") != manifest["archive"]["sha256"]:
        raise ValueError("original fixture checksum mismatch")
    spec = importlib.util.spec_from_file_location(
        "probe", ROOT / "python/phasesmith/oracle/_pinned_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    _, pwd = common.import_gsasii(args.gsas_root, args.binary_dir)
    fixture = np.load(source / "data.npz", allow_pickle=False)
    x = fixture["fixed_cell_x_deg"]
    refs = fixture["fixed_cell_reflections"]
    arrays = {}
    for axial, name in ((0.0, "symmetric"), (0.002, "axial")):
        profiles, supports = [], []
        for row in refs:
            values, limits = probe.probe_fcj_profile_and_support(
                pwd,
                x,
                position_deg=float(row[5]),
                sigma2_centideg2=float(row[6]),
                gamma_centideg=float(row[7]),
                axial_sum=axial,
            )
            profiles.append(values)
            supports.append(limits)
        arrays[name] = np.asarray(profiles)
        arrays[name + "_support"] = np.asarray(supports)
    row = refs[0]
    for count in (401, 4001, 40001):
        grid = np.linspace(row[5] - 0.2, row[5] + 0.2, count)
        arrays[f"grid_{count}"] = grid
        arrays[f"profile_{count}"], _ = probe.probe_fcj_profile_and_support(
            pwd,
            grid,
            position_deg=float(row[5]),
            sigma2_centideg2=float(row[6]),
            gamma_centideg=float(row[7]),
            axial_sum=0.002,
        )
    sweep = []
    grid = np.linspace(row[5] - 0.5, row[5] + 0.5, 4001)
    arrays["sweep_grid"] = grid
    for axial in (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        for width in (0.1, 1.0, 10.0):
            key = f"sweep_{len(sweep)}"
            arrays[key], _ = probe.probe_fcj_profile_and_support(
                pwd,
                grid,
                position_deg=float(row[5]),
                sigma2_centideg2=float(row[6]) * width**2,
                gamma_centideg=float(row[7]) * width,
                axial_sum=axial,
            )
            sweep.append(dict(key=key, axial_sum=axial, width_scale=width))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "profiles.npz", **arrays)
    record = dict(
        gsasii_revision=pin["revision"],
        original_manifest_sha256=sha(source / "manifest.json"),
        archive_sha256=sha(args.output / "profiles.npz"),
        generator_sha256=sha(Path(__file__)),
        adapter_sha256=sha(ROOT / "python/phasesmith/oracle/_pinned_probe.py"),
        binaries_sha256={p.name: sha(p) for p in sorted(args.binary_dir.glob("*.so"))},
        sweep=sweep,
        axial_sum=0.002,
        scope="Black-box fixed-profile controls, not a replacement golden fixture",
    )
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
