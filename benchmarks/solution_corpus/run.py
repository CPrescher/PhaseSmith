"""Isolated bounded solves; references are opened only after saving all jobs."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .data import sha

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/solution_corpus_benchmark.py"
SHELXT_SHA = "4d712cb17f8952e1f59f476af7c61ad4ab5261e594c75441c7f81dcefe7b5560"
INPUT_KEYS = {
    "id",
    "cell",
    "space_group",
    "space_group_number",
    "shelxt_space_group",
    "wavelength",
    "elements",
}


def worker(args):
    from phasesmith.crystallography import UnitCell
    from phasesmith.experimental.solution import (
        SolutionOptions,
        solve_structure_elements,
        write_solution,
    )
    from phasesmith.io.hkl import read_hkl
    from phasesmith.io.space_groups import space_group_by_number

    packet = json.loads(args.input.read_text())
    if set(packet) != INPUT_KEYS:
        raise ValueError("worker accepts only the whitelisted input fields")
    options = SolutionOptions(
        phasing="dual_space",
        trials=4,
        cycles=800,
        seed=0,
        max_seconds=max(0.1, args.seconds - 2),
        r1_target=0.06,
        completion=False,
    )
    observations = read_hkl(args.hkl)
    group = space_group_by_number(packet["space_group_number"]).space_group
    maximum_atoms = args.maximum_atoms
    start = time.perf_counter()
    result = solve_structure_elements(
        observations,
        cell=UnitCell(*packet["cell"]),
        elements=packet["elements"],
        space_group=group,
        options=options,
        maximum_atoms=maximum_atoms,
        symmetry_recycling=True,
        tangent_refinement=True,
        adaptive_counts=not args.legacy_counts,
        wilson_normalization=args.wilson_normalization,
        phase_quality_selection=args.phase_quality_selection,
        phase_origin_selection=args.phase_origin_selection,
        chemical_assignment=args.chemical_assignment,
        iterative_refinement=args.iterative_refinement,
        missing_reflections=args.missing_reflections,
        adaptive_checkpoints=args.adaptive_checkpoints,
        score_only_evaluation=args.score_only_evaluation,
        work_sliced_phasing=args.work_sliced_phasing,
        isotropic_assignment_refinement=args.isotropic_assignment_refinement,
        filtered_peak_search=args.filtered_peak_search,
        batched_fft=args.batched_fft,
        real_fft=args.real_fft,
        cached_normal_refinement=args.cached_normal_refinement,
        reusable_fourier_workspaces=args.reusable_fourier_workspaces,
        independent_origin_sampling=args.independent_origin_sampling,
        resolution_extension=args.resolution_extension,
        phase_first_selection=args.phase_first_selection,
        calculated_shell_normalization=not args.legacy_quality_normalization,
    )
    elapsed = time.perf_counter() - start
    write_solution(result, args.output)
    model = None
    if result.best is not None:
        model = {
            "species": list(result.best.species),
            "fractional_xyz": result.best.fractional_xyz.tolist(),
            "cell": list(result.cell.as_tuple()),
            "reported_r1": result.best.r1,
            "space_group": packet["space_group"],
            "space_group_number": packet["space_group_number"],
        }
    (args.output / "worker.json").write_text(
        json.dumps(
            {
                "model": model,
                "kernel_seconds": elapsed,
                "termination": result.termination,
                "target_met": result.target_met,
                "options": asdict(options),
                "effort": asdict(result.shared_search),
                "input_only_resource_atom_cap": maximum_atoms,
            },
            indent=2,
        )
        + "\n"
    )


def resource_atom_cap(observations, cell, symbol):
    import gemmi
    import numpy as np
    from phasesmith.crystallography import UnitCell
    from phasesmith.io.space_groups import space_group_by_number

    group = space_group_by_number(gemmi.find_spacegroup_by_name(symbol).number).space_group
    hkl = observations.hkl[~group.systematic_absences(observations.hkl)]
    rotations = {
        tuple(v for row in op.rot for v in row)
        for op in gemmi.find_spacegroup_by_name(symbol).operations()
    }
    images = [hkl @ np.array(r, dtype=np.int64).reshape(3, 3) // 24 for r in sorted(rotations)]
    full = np.concatenate([*images, *[-h for h in images]])
    measured_rows = len(np.unique(full, axis=0))
    if not measured_rows:
        raise ValueError("no allowed measured symmetry family")
    memory_cap = (2_000_000 // measured_rows - 7) // 5
    volume_cap = int(np.ceil(UnitCell(*cell).geometry().volume_angstrom3 / 8))
    cap = min(128, memory_cap, volume_cap)
    if cap < 1:
        raise ValueError("no positive atom cap fits the existing dense-memory guard")
    return cap, measured_rows


def instructions(packet):
    import gemmi

    group = gemmi.find_spacegroup_by_name(packet["space_group"])
    operations = group.operations()
    lattice = {"P": 1, "I": 2, "R": 3, "F": 4, "A": 5, "B": 6, "C": 7}[operations.find_centering()]
    inversion = gemmi.Op("-x,-y,-z")
    origin_centric = inversion.triplet() in {op.wrap().triplet() for op in operations.sym_ops}
    if not origin_centric:
        lattice = -lattice
    symm, seen = [], set()
    for op in operations.sym_ops:
        key = op.wrap().triplet()
        if key in seen:
            continue
        seen.add(key)
        if origin_centric:
            seen.add(inversion.combine(op).wrap().triplet())
        if key != "x,y,z":
            symm.append("SYMM " + key)
    return "\n".join(
        [
            "TITL Blind element-only corpus comparison",
            "CELL " + " ".join(str(v) for v in [packet["wavelength"], *packet["cell"]]),
            f"LATT {lattice}",
            *symm,
            "SFAC " + " ".join(packet["elements"]),
            "UNIT " + " ".join("1" for _ in packet["elements"]),
            "HKLF 4",
            "END",
            "",
        ]
    )


def launch(command, directory, seconds):
    started = time.perf_counter()
    environment = {
        **os.environ,
        **{
            name: "1"
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
    }
    with (directory / "stdout.txt").open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        expired = False
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            expired = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    return {
        "wall_seconds_including_startup": time.perf_counter() - started,
        "returncode": code,
        "hard_timeout": expired,
        "command": command,
    }


def run(args):
    for _ in collect_jobs(args):
        pass


def collect_jobs(args):
    """Yield saved serial jobs and a reference-audit barrier for balanced scheduling."""
    from phasesmith import _core
    from phasesmith.io.hkl import read_hkl

    from .audit import audit_case, common_score, shelxt_candidate

    manifest = json.loads(args.manifest.read_text())
    lock_path = args.manifest.with_suffix(f".benchmark{args.lock_revision}.lock.json")
    lock = json.loads(lock_path.read_text())
    if lock["manifest_sha256"] != sha(args.manifest):
        raise ValueError("manifest differs from sealed corpus")
    for name, expected in lock["verified_implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("benchmark implementation changed after sealing: " + name)
    cases = [c for c in manifest["cases"] if c["split"] == args.split]
    if not cases:
        raise ValueError("selected split is empty")
    if sha(args.shelxt) != SHELXT_SHA:
        raise ValueError("external executable differs from pinned user-provided SHELXT 2018/2")
    previous = None
    if args.resume:
        previous = json.loads((args.output / "before-reference-audit.json").read_text())
        if (
            previous["manifest_sha256"] != sha(args.manifest)
            or previous["seconds_per_job"] != args.seconds
        ):
            raise ValueError("resume inputs/budget differ from collection")
        if previous["split"] != args.split:
            raise ValueError("resume split differs")
        if previous.get("adaptive_counts", True) != (not args.legacy_counts):
            raise ValueError("resume count policy differs")
        if (
            previous.get("wilson_normalization", False) != args.wilson_normalization
            or previous.get("phase_quality_selection", False) != args.phase_quality_selection
            or previous.get("phase_first_selection", False) != args.phase_first_selection
            or previous.get("phase_origin_selection", False) != args.phase_origin_selection
            or previous.get("chemical_assignment", False) != args.chemical_assignment
            or previous.get("iterative_refinement", False) != args.iterative_refinement
            or previous.get("missing_reflections", False) != args.missing_reflections
            or previous.get("adaptive_checkpoints", False) != args.adaptive_checkpoints
            or previous.get("score_only_evaluation", False) != args.score_only_evaluation
            or previous.get("work_sliced_phasing", False) != args.work_sliced_phasing
            or previous.get("filtered_peak_search", False) != args.filtered_peak_search
            or previous.get("batched_fft", False) != args.batched_fft
            or previous.get("real_fft", False) != args.real_fft
            or previous.get("cached_normal_refinement", False) != args.cached_normal_refinement
            or previous.get("isotropic_assignment_refinement", False)
            != args.isotropic_assignment_refinement
            or previous.get("reusable_fourier_workspaces", False)
            != args.reusable_fourier_workspaces
            or previous.get("independent_origin_sampling", False)
            != args.independent_origin_sampling
            or previous.get("resolution_extension", 1.0) != args.resolution_extension
            or previous.get("calculated_shell_normalization", False)
            == args.legacy_quality_normalization
        ):
            raise ValueError("resume normalization/phase ranking differs")
        if previous["native_sha256"] != sha(Path(_core.__file__)) or previous[
            "shelxt_sha256"
        ] != sha(args.shelxt):
            raise ValueError("resume numerical executable differs")
        for name, expected in previous["implementation_sha256"].items():
            if (
                name.startswith(
                    ("crates/phasesmith-solution/src/", "python/phasesmith/experimental/")
                )
                and sha(ROOT / name) != expected
            ):
                raise ValueError("resume numerical source differs: " + name)
        for name, expected in previous["implementation_sha256"].items():
            candidates = [args.output / previous.get("source_snapshot", "source-snapshot") / name]
            if not candidates[0].exists() or sha(candidates[0]) != expected:
                raise ValueError("missing exact earlier collection source snapshot: " + name)
    else:
        args.output.mkdir(exist_ok=False, parents=True)
    args.output = args.output.resolve()
    sources = (
        sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs"))
        + sorted((ROOT / "benchmarks/solution_corpus").glob("*.py"))
        + [SCRIPT, ROOT / "python/phasesmith/experimental/solution.py"]
    )
    if getattr(args, "collection_schedule", "serial_panel") != "serial_panel":
        sources.append(ROOT / "benchmarks/solution_execution_interleaved.py")
        scheduler = getattr(args, "collection_scheduler_source", None)
        if scheduler is not None:
            sources.append(ROOT / scheduler)
    header = {
        "schema": "phasesmith.solution-corpus-comparison.v1",
        "collection_schedule": getattr(args, "collection_schedule", "serial_panel"),
        "split": args.split,
        "holdout_opened": args.split == "holdout",
        "manifest_sha256": sha(args.manifest),
        "corpus_lock_sha256": sha(lock_path),
        "platform": platform.platform(),
        "python": sys.version,
        "seconds_per_job": args.seconds,
        "adaptive_counts": not args.legacy_counts,
        "wilson_normalization": args.wilson_normalization,
        "phase_quality_selection": args.phase_quality_selection,
        "phase_origin_selection": args.phase_origin_selection,
        "chemical_assignment": args.chemical_assignment,
        "iterative_refinement": args.iterative_refinement,
        "missing_reflections": args.missing_reflections,
        "adaptive_checkpoints": args.adaptive_checkpoints,
        "score_only_evaluation": args.score_only_evaluation,
        "work_sliced_phasing": args.work_sliced_phasing,
        "isotropic_assignment_refinement": args.isotropic_assignment_refinement,
        "filtered_peak_search": args.filtered_peak_search,
        "batched_fft": args.batched_fft,
        "real_fft": args.real_fft,
        "cached_normal_refinement": args.cached_normal_refinement,
        "reusable_fourier_workspaces": args.reusable_fourier_workspaces,
        "independent_origin_sampling": args.independent_origin_sampling,
        "resolution_extension": args.resolution_extension,
        "phase_first_selection": args.phase_first_selection,
        "calculated_shell_normalization": not args.legacy_quality_normalization,
        "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "native_sha256": sha(Path(_core.__file__)),
        "shelxt_sha256": sha(args.shelxt),
        "architecture_qualification": (
            "PhaseSmith arm64 native; supplied SHELXT x86_64 uses host compatibility runtime"
        ),
        "resource_policy": (
            "min(128,ceil(V/8),floor((floor(2000000/M)-7)/5)); "
            "M from observed symmetry/Friedel families only"
        ),
        "input_contract": (
            "same quantized measured HKLF4, element types, cell, wavelength "
            "and fixed reference-setting space group; no counts or coordinates"
        ),
        "thread_policy": "serial jobs; all numerical thread environment limits=1; SHELXT -t1",
        "score_policy": {
            "counts": "exact non-H full cell",
            "maximum_error_angstrom": 0.05,
            "rms_error_angstrom": 0.025,
            "strict_secondary_maximum_angstrom": 0.02,
            "reference_cell_setting": (
                "same supplied reference setting; changed output cells explicitly unassessed"
            ),
        },
        "timing_qualification": (
            "final-model recovery at stop; not time of first correct candidate; "
            "programs use different internal refinement models"
        ),
    }
    stage = 1 if previous is None else len(previous.get("collection_stages", [None])) + 1
    snapshot = args.output / f"source-snapshot-stage{stage}"
    for name, expected in header["implementation_sha256"].items():
        source = ROOT / name
        if sha(source) != expected:
            raise ValueError("implementation changed while taking source snapshot")
        target = snapshot / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source, target)
    header["source_snapshot"] = snapshot.name
    header["collection_stages"] = (
        []
        if previous is None
        else previous.get(
            "collection_stages",
            [
                {
                    "stage": 1,
                    "implementation_sha256": previous["implementation_sha256"],
                    "source_snapshot": previous.get("source_snapshot", "source-snapshot"),
                }
            ],
        )
    ) + [
        {
            "stage": stage,
            "implementation_sha256": header["implementation_sha256"],
            "source_snapshot": snapshot.name,
        }
    ]
    if previous is not None:
        shutil.copyfile(
            args.output / "protocol.json", args.output / f"protocol-before-stage{stage}.json"
        )
        shutil.copyfile(
            args.output / "before-reference-audit.json",
            args.output / f"predictions-before-stage{stage}.json",
        )
    (args.output / "protocol.json").write_text(json.dumps(header, indent=2) + "\n")
    records = [] if previous is None else previous["runs"]
    for record in records:
        record.setdefault("collection_stage", 1)
    completed = {(r["id"], r["method"]) for r in records}
    for index, case in enumerate(cases):
        for key in ["inputs", "common_hkl"]:
            entry = case["files"][key]
            if sha(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("input checksum mismatch")
        packet = json.loads((ROOT / case["files"]["inputs"]["path"]).read_text())
        if set(packet) != INPUT_KEYS:
            raise ValueError("input packet contains non-input fields")
        if all((case["id"], method) in completed for method in ["phasesmith", "shelxt"]):
            continue
        prep_started = time.perf_counter()
        atom_cap, expanded_rows = resource_atom_cap(
            read_hkl(ROOT / case["files"]["common_hkl"]["path"]),
            packet["cell"],
            packet["space_group"],
        )
        prep_seconds = time.perf_counter() - prep_started
        for method in ["phasesmith", "shelxt"] if index % 2 == 0 else ["shelxt", "phasesmith"]:
            if (case["id"], method) in completed:
                continue
            work = args.output / case["id"] / method
            work.mkdir(parents=True)
            (work / "inputs.json").write_text(json.dumps(packet, indent=2) + "\n")
            (work / "sample.hkl").write_bytes(
                (ROOT / case["files"]["common_hkl"]["path"]).read_bytes()
            )
            if method == "phasesmith":
                command = [
                    sys.executable,
                    str(SCRIPT),
                    "worker",
                    "--input",
                    str(work / "inputs.json"),
                    "--hkl",
                    str(work / "sample.hkl"),
                    "--output",
                    str(work / "solution"),
                    "--seconds",
                    str(args.seconds),
                    "--maximum-atoms",
                    str(atom_cap),
                    *(["--legacy-counts"] if args.legacy_counts else []),
                    *(["--wilson-normalization"] if args.wilson_normalization else []),
                    *(["--phase-origin-selection"] if args.phase_origin_selection else []),
                    *(["--chemical-assignment"] if args.chemical_assignment else []),
                    *(["--iterative-refinement"] if args.iterative_refinement else []),
                    *(["--missing-reflections"] if args.missing_reflections else []),
                    *(["--adaptive-checkpoints"] if args.adaptive_checkpoints else []),
                    *(["--score-only-evaluation"] if args.score_only_evaluation else []),
                    *(["--work-sliced-phasing"] if args.work_sliced_phasing else []),
                    *(["--filtered-peak-search"] if args.filtered_peak_search else []),
                    *(["--batched-fft"] if args.batched_fft else []),
                    *(["--real-fft"] if args.real_fft else []),
                    *(["--cached-normal-refinement"] if args.cached_normal_refinement else []),
                    *(
                        ["--isotropic-assignment-refinement"]
                        if args.isotropic_assignment_refinement
                        else []
                    ),
                    *(
                        ["--reusable-fourier-workspaces"]
                        if args.reusable_fourier_workspaces
                        else []
                    ),
                    *(
                        ["--independent-origin-sampling"]
                        if args.independent_origin_sampling
                        else []
                    ),
                    "--resolution-extension",
                    str(args.resolution_extension),
                    *(["--phase-quality-selection"] if args.phase_quality_selection else []),
                    *(
                        ["--phase-first-selection"]
                        if args.phase_first_selection
                        else ["--quality-checkpoints"]
                    ),
                    *(
                        ["--legacy-quality-normalization"]
                        if args.legacy_quality_normalization
                        else []
                    ),
                ]
            else:
                (work / "sample.ins").write_text(instructions(packet))
                command = [
                    str(args.shelxt.resolve()),
                    "sample",
                    "-t1",
                    "-s" + packet["shelxt_space_group"],
                ]
            record = {
                "id": case["id"],
                "collection_stage": stage,
                "method": method,
                "directory": str(work.relative_to(args.output)),
                "input_only_resource_atom_cap": atom_cap,
                "expanded_measured_rows_for_resource_cap": expanded_rows,
                "adapter_preparation_seconds_outside_timing": prep_seconds,
                "host_load_average_before_job": list(os.getloadavg()),
                **launch(command, work, args.seconds),
            }
            try:
                if method == "phasesmith":
                    native = json.loads((work / "solution/worker.json").read_text())
                    record.update(native)
                else:
                    model_path = work / "sample_a.res"
                    record["model"] = shelxt_candidate(model_path)
                    log = (work / "stdout.txt").read_text(errors="replace")
                    if "1 threads running in parallel" not in log:
                        raise ValueError("one-thread execution not confirmed")
                    record["outputs_sha256"] = {
                        p.name: sha(p) for p in work.iterdir() if p.is_file()
                    }
                record["status"] = "model_returned" if record.get("model") else "no_model"
            except Exception as error:
                record.update(
                    status="no_model", model=None, diagnostic=f"{type(error).__name__}: {error}"
                )
            if record["status"] == "no_model":
                log = (work / "stdout.txt").read_text(errors="replace")
                errors = [
                    line.strip() for line in log.splitlines() if "Error:" in line or "**" in line
                ]
                record["solver_diagnostic"] = errors[-3:]
                if not record["hard_timeout"] and record["returncode"] != 0:
                    record["status"] = "solver_input_or_runtime_error"
            records.append(record)
            (args.output / "before-reference-audit.json").write_text(
                json.dumps({**header, "runs": records}, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in [
                            "id",
                            "method",
                            "status",
                            "wall_seconds_including_startup",
                            "hard_timeout",
                        ]
                    }
                ),
                flush=True,
            )
            yield {"id": record["id"], "method": record["method"]}
    yield {"reference_audit_barrier": True}
    # Parse already chosen external _a files again for resumed collections. No
    # candidate selection or search is repeated, and no reference has been opened.
    for record in records:
        if record["method"] == "shelxt":
            model_path = args.output / record["directory"] / "sample_a.res"
            if model_path.exists():
                try:
                    record["model"] = shelxt_candidate(model_path)
                    record["collection_output_status"] = record["status"]
                    record["status"] = "model_returned"
                except Exception as error:
                    record["output_adapter_error"] = f"{type(error).__name__}: {error}"
    (args.output / "before-reference-audit.json").write_text(
        json.dumps({**header, "runs": records}, indent=2) + "\n"
    )
    # Both programs' complete predictions have been saved before opening any atom reference.
    for case in cases:
        entry = case["files"]["reference"]
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("reference checksum mismatch")
        observations = read_hkl(ROOT / case["files"]["common_hkl"]["path"])
        for record in [r for r in records if r["id"] == case["id"]]:
            record["correct"] = False
            if record.get("model") is None:
                continue
            try:
                record["reference_audit"] = audit_case(case, record["model"], ROOT)
                record["correct"] = record["reference_audit"]["correct"]
                record["common_fixed_uiso_score"] = common_score(
                    record["model"], observations, case["curation"]["cell"]
                )
            except Exception as error:
                record["audit_error"] = f"{type(error).__name__}: {error}"
    summary = {}
    import numpy as np

    for method in ["phasesmith", "shelxt"]:
        rows = [r for r in records if r["method"] == method]
        times = [r["wall_seconds_including_startup"] for r in rows]
        summary[method] = {
            "cases": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "hard_timeouts": sum(r["hard_timeout"] for r in rows),
            "median_wall_seconds_all_outcomes": float(np.median(times)),
            "p95_wall_seconds_all_outcomes": float(np.percentile(times, 95)),
            "status_counts": dict(Counter(r["status"] for r in rows)),
        }
    report = {
        **header,
        "summary": summary,
        "runs": records,
        "before_reference_audit_sha256": sha(args.output / "before-reference-audit.json"),
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
