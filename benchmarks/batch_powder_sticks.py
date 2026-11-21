"""Time complete one-shot API calls on three nested panels of real COD structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import phasesmith as ps
from phasesmith.io.cif import read_cif

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/downloads/2026-11-21-xrd-rust-47a238b.zip"
SCAN = ps.CwTwoThetaRange(2, 60, 0.71073)
SIZES = (1, 2, 4, 8, 16, 32, 64, 256)
SEED = 20261121


def run(cifs: Path, dest: Path, repetitions: int, check_default: bool) -> None:
    if ps._core.BUILD_MODE != "release":
        raise RuntimeError("benchmark requires a release build")
    with ZipFile(ARCHIVE) as archive:
        rows = csv.DictReader(io.StringIO(archive.read("per-entry.csv").decode()))
        ids = sorted(row["COD"] for row in rows if row["outcome"] == "qualified")
        checksums = {
            r["id"]: r["sha256"] for r in json.loads(archive.read("inorganic-inputs.json"))
        }
    rng = random.Random(SEED)
    rng.shuffle(ids)
    panels, timings = [], []
    if check_default:
        ids = ids[3 * max(SIZES) :]
    for panel in range(3):
        selected = ids[panel * max(SIZES) : (panel + 1) * max(SIZES)]
        panels.append(selected)
        models = []
        for key in selected:
            path = cifs / f"{key}.cif"
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksums[key]:
                raise ValueError(f"checksum mismatch for {key}")
            original = read_cif(path, strict=True).structure
            # Repeat the previous comparison's neutral-atom/zero-displacement convention.
            sites = tuple(
                replace(
                    site,
                    type_symbol=site.element_symbol,
                    charge=None,
                    u_iso_angstrom2=0.0,
                    anisotropic_displacement=None,
                )
                for site in original.sites
            )
            models.append(replace(original, sites=sites))
        reference = ps.calculate_powder_sticks_batch(models, SCAN, workers=1)
        for size in SIZES:
            inputs = models[:size]
            configs = [("threads", 1)]
            if check_default:
                configs.append(("threads", None))
            else:
                configs += [
                    (kind, n) for n in (2, 4, 8) if n <= size for kind in ("threads", "processes")
                ]
            samples = {f"{kind}-{n}": [] for kind, n in configs}
            for _ in range(repetitions):
                rng.shuffle(configs)
                for kind, n in configs:
                    start = time.perf_counter_ns()
                    patterns = ps.calculate_powder_sticks_batch(
                        inputs, SCAN, workers=n, backend=kind
                    )
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    for result, expected in zip(patterns, reference[:size], strict=True):
                        assert result.structure_id == expected.structure_id
                        for actual, wanted in (
                            (result.two_theta_deg, expected.two_theta_deg),
                            (result.integrated_intensity, expected.integrated_intensity),
                        ):
                            assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                            assert actual.tobytes() == wanted.tobytes()
                    samples[f"{kind}-{n}"].append(elapsed)
            timings.append(dict(panel=panel, size=size, samples_ms=samples))
            print(
                panel + 1,
                size,
                {k: round(statistics.median(v), 2) for k, v in samples.items()},
                flush=True,
            )
    receipt = dict(
        version=ps.__version__,
        build=ps._core.BUILD_MODE,
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=platform.processor(),
        cpu_count=os.cpu_count(),
        numpy=np.__version__,
        blas_threads={
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        native_sha256=hashlib.sha256(Path(ps._core.__file__).read_bytes()).hexdigest(),
        input_archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        seed=SEED,
        check_default=check_default,
        module_sha256=hashlib.sha256(Path(ps.powder_sticks.__file__).read_bytes()).hexdigest(),
        panels=panels,
        timings=timings,
        scope=(
            "Each call includes worker setup, job submission, structure/output transfer where "
            "applicable, and shutdown. CIF parsing and correctness checks are outside the timer."
        ),
    )
    dest.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cifs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--check-default",
        action="store_true",
        help="time serial and automatic settings on three disjoint holdout panels",
    )
    args = parser.parse_args()
    run(args.cifs, args.output, args.repetitions, args.check_default)
