"""Benchmark a Rust-only MgSiO3 worker; verify/grade independently after it exits.

The executable also runs directly without Python. This optional observer keeps
source hashing and reference grading comparable with enstatite_raw_workflow.py.
No detector counts, imported reduction or atomic model cross into this parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_single_crystal_data import verify_dataset  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--executable", required=True, type=Path)
    p.add_argument(
        "--protocol", type=Path, default=ROOT / "validation/enstatite-raw-workflow-v28.json"
    )
    args = p.parse_args()
    started = time.perf_counter()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        p.error("large output must be outside the repository")
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads((ROOT / protocol["dataset_manifest"]).read_text())
    rotations = protocol["native_coordinator"]["null_rotations"]
    rotation_path = ROOT / rotations["path"]
    if digest(rotation_path) != rotations["sha256"]:
        raise ValueError("predeclared null rotations changed")
    checked = time.perf_counter()
    verify_dataset(manifest, args.data_root)
    source_seconds = time.perf_counter() - checked
    out.mkdir(parents=True, exist_ok=False)
    packet = dict(
        raw_path=str((args.data_root / "eiger_0000.h5").resolve()),
        poni_path=str((args.data_root / "CeO2_20240620_1849_dioptas.poni").resolve()),
        raw_array=manifest["raw_array"],
        priors=manifest["search_inputs"],
        rotations=[],  # Rust's native space-group adapter owns exact operations.
        protocol=protocol,
        output=str(out / "native"),
        through="search",
    )
    save(out / "input.json", packet)
    save(
        out / "provenance.json",
        dict(
            protocol_sha256=digest(args.protocol),
            executable_sha256=digest(args.executable),
            observer_sha256=digest(Path(__file__)),
            git_commit=subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            source_hashes={r["path"]: r["sha256"] for r in manifest["files"]},
        ),
    )
    tick = time.perf_counter()
    with (out / "worker.log").open("x") as stream:
        result = subprocess.run(
            [
                str(args.executable.resolve()),
                str(out / "input.json"),
                str(rotation_path),
                str(out / "native"),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
            env=dict(
                os.environ,
                PATH="/usr/bin:/bin",
                HDF5_PLUGIN_PATH="",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
            ),
        )
    worker_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    verify_dataset(manifest, args.data_root)
    source_seconds += time.perf_counter() - tick
    report = dict(
        schema="phasesmith.enstatite-native-workflow.v1",
        execution_returncode=result.returncode,
        source_files_unchanged=True,
        raw_to_structure_search_passed=False,
        reference_assessment="not_executed",
        standalone_structure_refinement="not_implemented",
        native_worker_seconds=worker_seconds,
        source_verification_seconds=source_seconds,
    )
    if result.returncode == 0:
        import gemmi
        from solution_corpus.audit import match_geometry
        from solution_corpus.data import atom_block, expanded_sites

        tick = time.perf_counter()
        saved = json.loads((out / "native/candidate/result.json").read_text())
        if saved["candidates"]:
            model = dict(saved["candidates"][0], cell=saved["cell"])
            labels, xyz = expanded_sites(
                atom_block(args.data_root / "dom1_v9.cif"),
                gemmi.find_spacegroup_by_name(manifest["search_inputs"]["space_group_symbol"]),
            )
            assessment = match_geometry(model, labels, xyz, saved["cell"])
            assessment["reference_read_after_worker_stopped"] = True
            report["reference_assessment"] = assessment
            report["raw_to_structure_search_passed"] = assessment["correct"]
        report["reference_assessment_seconds"] = time.perf_counter() - tick
    report["observed_wall_seconds"] = time.perf_counter() - started
    save(out / "report.json", report)
    if result.returncode == 0:
        save(out / "native/report.json", report)
        save(out / "native/provenance.json", json.loads((out / "provenance.json").read_text()))
    print(json.dumps(report, indent=2))
    return 0 if result.returncode == 0 and report["raw_to_structure_search_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
