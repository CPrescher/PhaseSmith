"""Explicit opt-in experimental HKL recovery and optional pinned SHELXT comparison.

Reference coordinates are read only after the native solve has completed.
Use --download-data explicitly to retrieve checksum-pinned public ZIP members.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from phasesmith.crystallography import UnitCell
from phasesmith.experimental.solution import SolutionOptions, solve_structure, write_solution
from phasesmith.io.hkl import read_hkl
from phasesmith.io.space_groups import space_group_by_symbol

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/structure-solution-thiourea.json"
SHELXT_SHA256 = "4d712cb17f8952e1f59f476af7c61ad4ab5261e594c75441c7f81dcefe7b5560"


class _Ranges(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else (self.position + offset if whence == 1 else self.size + offset)
        )
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else size
        if not 0 <= size <= 2 * 1024 * 1024:
            raise ValueError("ZIP member/range exceeds explicit 2 MiB bound")
        if not size:
            return b""
        end = min(self.position + size, self.size) - 1
        request = urllib.request.Request(
            self.url + f"?range_start={self.position}",
            headers={
                "Range": f"bytes={self.position}-{end}",
                "User-Agent": "PhaseSmith validation",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            expected_range = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("server did not honor exact bounded ZIP byte range")
            data = response.read(size + 1)
        if len(data) != end - self.position + 1:
            raise ValueError("truncated or oversized byte range")
        self.position += len(data)
        return data


def _verify(path, record):
    data = path.read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"source byte/hash mismatch: {path}")


def _download(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        _Ranges(source["archive_url"], source["archive_bytes_from_repository"])
    ) as archive:
        for record in source["members"].values():
            path = directory / record["filename"]
            if path.exists():
                _verify(path, record)
                continue
            if archive.getinfo(record["archive_member"]).file_size != record["bytes"]:
                raise ValueError("ZIP member size differs from pinned source size")
            data = archive.read(record["archive_member"])
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("downloaded ZIP member failed pinned checks")
            with path.open("xb") as output:
                output.write(data)


def _reference_error(candidate, cif_path, cell, symbol):
    # This optional post-solve diagnostic uses Gemmi/SciPy only in the benchmark.
    import gemmi
    from scipy.optimize import linear_sum_assignment

    structure = gemmi.make_small_structure_from_block(gemmi.cif.read_file(str(cif_path))[-1])
    reference, elements = [], []
    for site in structure.sites:
        if site.element.name == "H":
            continue
        unique = []
        for op in gemmi.find_spacegroup_by_name(symbol).operations():
            point = np.mod(op.apply_to_xyz([site.fract.x, site.fract.y, site.fract.z]), 1)
            if not any(np.linalg.norm((point - other + 0.5) % 1 - 0.5) < 1e-7 for other in unique):
                unique.append(point)
        reference.extend(unique)
        elements.extend([site.element.name] * len(unique))
    reference, elements = np.array(reference), np.array(elements)
    import itertools

    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    best = (float("inf"), float("inf"))
    for sign in (-1, 1):
        for anchor in reference[elements == candidate.species[0]]:
            model = (
                sign * candidate.fractional_xyz + anchor - sign * candidate.fractional_xyz[0]
            ) % 1
            errors = []
            for element in set(elements):
                delta = (
                    reference[elements == element, None, :]
                    - model[None, np.array(candidate.species) == element, :]
                    + 0.5
                ) % 1 - 0.5
                distance = np.linalg.norm(
                    (delta[:, :, None] + shifts) @ cell.geometry().direct_basis.T, axis=-1
                ).min(axis=2)
                left, right = linear_sum_assignment(distance**2)
                errors.extend(distance[left, right])
            summary = (float(np.max(errors)), float(np.sqrt(np.mean(np.square(errors)))))
            best = min(best, summary)
    return {
        "max_element_matched_error_angstrom": best[0],
        "rms_error_angstrom": best[1],
        "reference_used_after_solve_only": True,
        "non_h_atom_count": len(elements),
    }


def _shelxt(executable, source, hkl_path, output):
    import gemmi

    executable = executable.resolve()
    if hashlib.sha256(executable.read_bytes()).hexdigest() != SHELXT_SHA256:
        raise ValueError("SHELXT executable does not match the pinned local 2018/2 binary")
    output.mkdir()
    settings = source["solver_inputs"]
    cell = " ".join(str(v) for v in settings["cell_angstrom_degrees"])
    operations = gemmi.find_spacegroup_by_name(settings["known_space_group"]).operations()
    instructions = [
        "TITL External HKL comparison; no coordinates",
        f"CELL 0.71073 {cell}",
        "LATT -1",
    ]
    instructions += ["SYMM " + op.triplet() for op in operations if op.triplet() != "x,y,z"]
    instructions += ["SFAC C N S", "UNIT 4 8 4", "HKLF 4", "END"]
    (output / "sample.ins").write_text("\n".join(instructions) + "\n")
    shutil.copyfile(hkl_path, output / "sample.hkl")
    start = time.perf_counter()
    with (output / "solution.txt").open("wb") as log:
        run = subprocess.run(
            [str(executable), "sample"],
            cwd=output,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=45,
            check=True,
        )
    return {
        "binary_sha256": SHELXT_SHA256,
        "version": "2018/2",
        "returncode": run.returncode,
        "wall_seconds_including_startup": time.perf_counter() - start,
        "instructions": instructions,
        "qualification": "Default SHELXT model/space-group search differs; "
        "timings and R1 are not equal-objective comparisons.",
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory", type=Path, default=ROOT / "validation/data/structure-solution/thiourea"
    )
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shelxt-executable", type=Path)
    parser.add_argument(
        "--refinement",
        choices=("none", "isotropic", "anisotropic"),
        default="none",
        help="Historical P1 baseline is the default; explicitly select the follow-up model.",
    )
    parser.add_argument(
        "--phasing", choices=("charge_flip", "dual_space", "portfolio"), default="charge_flip"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    source = json.loads(SOURCE.read_text())
    if args.download_data:
        _download(source, args.data_directory)
    for record in source["members"].values():
        _verify(args.data_directory / record["filename"], record)
    settings = source["solver_inputs"]
    cell = UnitCell(*settings["cell_angstrom_degrees"])
    hkl_path = args.data_directory / source["members"]["hkl"]["filename"]
    options = SolutionOptions(
        trials=8,
        cycles=1200,
        max_seconds=60,
        r1_target=0.06,
        refinement=args.refinement,
        phasing=args.phasing,
    )
    start = time.perf_counter()
    result = solve_structure(
        read_hkl(hkl_path),
        cell=cell,
        composition=settings["composition_full_cell_non_hydrogen"],
        space_group=space_group_by_symbol(settings["known_space_group"]).space_group,
        options=options,
    )
    elapsed = time.perf_counter() - start
    write_solution(result, args.output)
    report = {
        "source": source,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "wall_seconds": elapsed,
        "native_termination": result.termination,
        "amplitude_target_met": result.target_met,
        "options": result.to_record()["options"],
        "trial_count": len(result.histories),
        "phasing_trials": result.to_record()["phasing_trials"],
        "best_candidate": None if result.best is None else result.best.to_record(),
    }
    from phasesmith import _core

    implementation = [
        "Cargo.lock",
        "crates/phasesmith-solution/Cargo.toml",
        "crates/phasesmith-py/src/solution.rs",
        "crates/phasesmith-py/src/solution_symmetry.rs",
        "crates/phasesmith-py/src/solution_dual.rs",
        "python/phasesmith/experimental/solution_dual_reference.py",
        "python/phasesmith/experimental/solution_symmetry.py",
        "python/phasesmith/experimental/solution_reference.py",
        "python/phasesmith/experimental/solution.py",
        "python/phasesmith/io/hkl.py",
        "crates/phasesmith-crystallography/src/p1.rs",
        "crates/phasesmith-crystallography/src/scattering.rs",
        "crates/phasesmith-crystallography/src/scattering_data.rs",
        "crates/phasesmith-crystallography/src/cell.rs",
        "crates/phasesmith-crystallography/src/symmetry.rs",
    ]
    implementation += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "crates/phasesmith-solution/src").glob("*.rs"))
    ]
    report["implementation_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in implementation
    }
    report["native_extension_sha256"] = hashlib.sha256(
        Path(_core.__file__).read_bytes()
    ).hexdigest()
    report["native_features"] = ["experimental-solution"]
    if hasattr(_core, "ExperimentalLatticeEngine"):
        report["native_features"].append("experimental-spots")
    if result.best is not None:
        report["post_solve_reference_audit"] = _reference_error(
            result.best,
            args.data_directory / source["members"]["reference_cif"]["filename"],
            cell,
            settings["known_space_group"],
        )
    if args.shelxt_executable:
        report["external_shelxt"] = _shelxt(
            args.shelxt_executable, source, hkl_path, args.output / "shelxt"
        )
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
