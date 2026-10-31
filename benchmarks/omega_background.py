"""Realistic synthetic omega arPLS scaling; no file I/O or GUI dependency.

python benchmarks/omega_background.py --pixels 262144 --frames 76 --workers 1,4,8
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.single_crystal.omega_background import arpls_masked_rows, arpls_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=262144)
    parser.add_argument("--frames", type=int, default=76)
    parser.add_argument("--workers", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solver", choices=("cholesky", "ldlt"), default="cholesky")
    parser.add_argument(
        "--extra-peaks", action="store_true", help="Add three narrow peaks to every sixteenth trace"
    )
    parser.add_argument(
        "--baseline-native", help="Preserved native binary for alternating matched comparisons"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.pixels, args.repeats) < 1 or args.frames < 20:
        parser.error("Positive pixels/repeats and at least 20 frames required")
    rng = np.random.default_rng(20261024)
    x = np.arange(args.frames)
    rows = (
        100
        + 0.1 * x
        + 5 * np.sin(x / args.frames * np.pi)
        + rng.normal(0, 3, (args.pixels, args.frames))
    )
    centers = rng.uniform(0.15 * args.frames, 0.85 * args.frames, args.pixels)
    amplitudes = rng.choice([0.0, 500.0], args.pixels, p=[0.9, 0.1])
    rows += amplitudes[:, None] * np.exp(-0.5 * ((x - centers[:, None]) / 1.1) ** 2)
    if args.extra_peaks:
        for fraction, amplitude in ((0.2, 300.0), (0.5, 500.0), (0.8, 1000.0)):
            rows[::16] += amplitude * np.exp(-0.5 * ((x - fraction * args.frames) / 0.9) ** 2)
    group = (np.arange(len(rows)) // 16) % 20
    mask = (x < (group * args.frames // 100)[:, None]) | (
        x >= (args.frames - (19 - group) * args.frames // 100)[:, None]
    )
    report = dict(
        platform=platform.platform(),
        seed=20261024,
        shape=list(rows.shape),
        solver=args.solver,
        extra_peaks=args.extra_peaks,
        runs=[],
        comparisons=[],
    )
    baseline = None
    if args.baseline_native:
        path = Path(args.baseline_native).absolute()
        spec = importlib.util.spec_from_file_location("omega_baseline._core", path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        report["baseline_native"] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        path = Path(_core.__file__)
        report["current_native"] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        report["comparison_scope"] = (
            "Identical direct native array calls; input snapshots and result allocation included; "
            "hashing outside timing"
        )
    for workers in [int(v) for v in args.workers.split(",")]:
        arpls_rows(rows[:256], workers=workers, solver=args.solver)
        engines = (
            {"current": _core.ExperimentalArPLSEngine(workers, args.solver)} if baseline else {}
        )
        if baseline:
            engines["before"] = baseline.ExperimentalArPLSEngine(workers)
        for masked in (False, True):
            runs = {label: [] for label in engines} if baseline else {"current": []}
            for _repeat in range(args.repeats):
                labels = list(runs)
                if _repeat % 2:
                    labels.reverse()
                fingerprints = {}
                outputs = {}
                for label in labels:
                    results = []
                    start = time.perf_counter()
                    for offset in range(0, len(rows), 4096):
                        block = rows[offset : offset + 4096]
                        exclusions = mask[offset : offset + 4096] if masked else None
                        result = (
                            engines[label].baselines(
                                block, exclusions, 1e4, 0.001, 30, 12 if masked else 3
                            )
                            if baseline
                            else arpls_masked_rows(
                                block, exclusions, workers=workers, solver=args.solver
                            )
                            if masked
                            else arpls_rows(block, workers=workers, solver=args.solver)
                        )
                        results.append(result)
                    elapsed = time.perf_counter() - start
                    digest = hashlib.sha256()
                    for result in results:
                        for array in result:
                            digest.update(array.tobytes())
                    fingerprints[label] = digest.hexdigest()
                    if baseline:
                        outputs[label] = results
                    runs[label].append(
                        dict(
                            seconds=elapsed,
                            checksum=sum(float(np.nansum(r[0])) for r in results),
                            steps=sum(int(r[1].sum()) for r in results),
                            capped=sum(int((~r[2]).sum()) for r in results),
                            output_sha256=fingerprints[label],
                        )
                    )
                    del results, result
                if baseline:
                    squared, samples, largest = 0.0, 0, 0.0
                    for before, current in zip(outputs["before"], outputs["current"], strict=True):
                        if args.solver == "cholesky":
                            np.testing.assert_array_equal(current[0], before[0])
                        else:
                            np.testing.assert_allclose(
                                current[0], before[0], rtol=1e-9, atol=1e-7, equal_nan=True
                            )
                        np.testing.assert_array_equal(current[1], before[1])
                        np.testing.assert_array_equal(current[2], before[2])
                        difference = (current[0] - before[0])[np.isfinite(before[0])]
                        samples += difference.size
                        squared += float(np.dot(difference, difference))
                        largest = max(largest, float(np.abs(difference).max(initial=0)))
                    report["comparisons"].append(
                        dict(
                            workers=workers,
                            masked=masked,
                            repeat=_repeat,
                            max_baseline_difference=largest,
                            rms_baseline_difference=(squared / samples) ** 0.5,
                            samples=samples,
                            exact_diagnostics=True,
                        )
                    )
            for label, values in runs.items():
                record = dict(
                    workers=workers,
                    masked=masked,
                    label=label,
                    runs=values,
                    median_seconds=statistics.median(r["seconds"] for r in values),
                )
                report["runs"].append(record)
                print(json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
